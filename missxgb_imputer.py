"""

A Python module for the MissXGBImputer class, which performs missing data imputation
using XGBoost. Can utilize either CPU or GPU. 

Author: Benjamin Jargow / ChatGPT
Created: 13.08.2024
"""

import time
from xgboost import XGBClassifier, XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
import numpy as np
import pandas as pd
from tqdm import tqdm
try:
    import cupy as cp
except:
    print('Please install cupy to enable GPU support')

class MissXGBImputer:
    """
    MissXGBImputer is a class for imputing missing data in a DataFrame using XGBoost.
    The imputer can operate on both numerical and categorical data, and supports
    GPU acceleration for faster computation.

    Parameters:
    -----------
    max_iter : int, optional (default=10)
        Maximum number of imputation iterations.

    random_state : int or None, optional (default=None)
        Seed for random number generator.

    use_gpu : bool, optional (default=False)
        If True, use GPU for computations. Requires a compatible GPU and CuPy.

    convergence_criterium : float, optional (default=1e-5)
        The threshold for detecting convergence based on the mean difference
        between consecutive iterations.

    Attributes:
    -----------
    column_trans : ColumnTransformer
        A transformer for preprocessing numerical and categorical columns.

    enc : LabelEncoder
        Encoder for transforming categorical labels into integers.

    col_index_mapping : dict
        Mapping of original columns to indices that do not correspond to the original column.
    """

    def __init__(self, max_iter=10, random_state=None, use_gpu=False, convergence_criterium=1e-5):
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.convergence_criterion = convergence_criterium
        self.col_index_mapping = dict()

        # Initialize the ColumnTransformer and LabelEncoder
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.column_trans = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), selector(dtype_exclude=["object"])),
                ('cat', one_hot_encoder, selector(dtype_include=["object"]))
            ],
            remainder='drop',
            sparse_threshold=0
        )
        self.enc = LabelEncoder()

    def _initialize(self, X):
        """
        Initializes missing values with column means or most frequent values.

        Parameters:
        -----------
        X : pd.DataFrame
            The input DataFrame with missing values.

        Returns:
        --------
        pd.DataFrame
            A DataFrame with missing values initialized.
        """
        X_filled = X.copy()
        for col in X.columns:
            if X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    X_filled[col] = X_filled[col].fillna(X[col].mean())
                else:
                    X_filled[col] = X_filled[col].fillna(X[col].mode()[0])
        return X_filled

    def _create_index_mapping(self, X):
        """
        Creates a dictionary mapping original columns to indices that do not correspond to the original column.

        Parameters:
        -----------
        X : pd.DataFrame
            The input DataFrame.
        """
        feature_names = self.column_trans.get_feature_names_out()

        for col in X.columns:
            # Store the indices that do NOT correspond to the current column
            include_mask = np.array([col not in name for name in feature_names])
            self.col_index_mapping[col] = include_mask
            
    def _impute_column(self, X_filled, transformed_X, missing_mask, col):
        """
        Impute a single column in the DataFrame using the transformed dataset.

        Parameters:
        -----------
        X_filled : pd.DataFrame
            The DataFrame with initialized missing values.

        transformed_X : np.ndarray or cp.ndarray
            The transformed dataset after applying ColumnTransformer.

        missing_mask : pd.DataFrame
            A boolean mask indicating missing values in the original dataset.

        col : str
            The column to be imputed.
        """
        missing_idx = missing_mask[col]
        if missing_idx.any():
            y_train = X_filled.loc[~missing_idx, col]

            # Use the precomputed include_indices for the current column
            include_indices = self.col_index_mapping[col]

            X_train = transformed_X[~missing_idx][:, include_indices]
            X_missing = transformed_X[missing_idx][:, include_indices]

            # Ensure X_train and X_missing are NumPy arrays or CuPy arrays
            if self.use_gpu:
                X_train = cp.array(X_train)
                X_missing = cp.array(X_missing)
            else:
                X_train = np.array(X_train)
                X_missing = np.array(X_missing)

            # Fit the XGBoost model
            if pd.api.types.is_numeric_dtype(X_filled[col]):
                y_train = y_train.to_numpy()
                y_train = cp.array(y_train) if self.use_gpu else y_train
                rf = XGBRegressor(tree_method='hist', device='cuda') if self.use_gpu else XGBRegressor(tree_method='hist')
            else:
                y_train = self.enc.fit_transform(y_train)
                y_train = cp.array(y_train) if self.use_gpu else y_train
                rf = XGBClassifier(tree_method='hist', device='cuda') if self.use_gpu else XGBClassifier(tree_method='hist')

            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_missing)
            y_pred = self.enc.inverse_transform(cp.asnumpy(y_pred)) if not pd.api.types.is_numeric_dtype(X_filled[col]) else y_pred
            X_filled.loc[missing_idx, col] = y_pred

    def fit_transform(self, X):
        """
        Fits the MissXGB imputer on the data and returns the imputed dataset.

        Parameters:
        -----------
        X : pd.DataFrame
            The input DataFrame with missing values.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with missing values imputed.
        """
        X_filled = self._initialize(X)
        missing_mask = X.isnull()
        previous_filled = X_filled.copy()

        previous_diff = np.inf

        # Apply the ColumnTransformer at the beginning and create the index mapping
        transformed_X = self.column_trans.fit_transform(X_filled)
        self._create_index_mapping(X)  # Create the mapping of original columns to transformed indices

        for iteration in range(self.max_iter):
            print('Iteration: ', iteration + 1)
            iteration_start_time = time.time()

            # Impute each column
            for col in tqdm(X.columns):
                self._impute_column(X_filled, transformed_X, missing_mask, col)

            print(f"Iteration {iteration + 1} took {time.time() - iteration_start_time:.2f} seconds")

            # Re-transform the dataset at the end of the iteration
            transformed_X = self.column_trans.transform(X_filled)
            previous_filled_transformed = self.column_trans.transform(previous_filled)
            current_diff = np.mean(np.abs(transformed_X - previous_filled_transformed))
            print(f"Mean difference for iteration {iteration + 1}: {current_diff:.6f}")

            if np.abs(previous_diff - current_diff) < self.convergence_criterion:
                print("Convergence detected.")
                break

            previous_diff = current_diff
            previous_filled = X_filled.copy()

        return X_filled
