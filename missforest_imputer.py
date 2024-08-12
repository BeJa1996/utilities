from xgboost import XGBClassifier, XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# Define the ColumnTransformer and LabelEncoder
column_trans = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), lambda df: df.select_dtypes(include=[np.number]).columns),
        ('cat', OneHotEncoder(), lambda df: df.select_dtypes(include=['object', 'category']).columns)
    ],
    remainder='drop'
)
enc = LabelEncoder()

class MissForestImputer:
    """
    MissForestImputer is an implementation of the MissForest algorithm for 
    handling missing data by iteratively imputing missing values using 
    Random Forests (here, XGBoost models).

    Parameters:
    ----------
    max_iter : int, optional (default=10)
        The maximum number of iterations for the imputation process.

    random_state : int or None, optional (default=None)
        Controls the randomness of the imputer.
    
    n_jobs : int, optional (default=-1)
        The number of parallel jobs to run for model fitting. -1 means using all processors.

    use_gpu : bool, optional (default=False)
        Whether to use GPU for XGBoost model training. Requires a compatible GPU and the correct version of XGBoost.
    """

    def __init__(self, max_iter=10, random_state=None, n_jobs=-1, use_gpu=False):
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu

    def _initialize(self, X):
        """
        Initializes missing values with column means for numerical columns 
        or most frequent values for categorical columns.

        Parameters:
        ----------
        X : pd.DataFrame
            Input data with missing values.

        Returns:
        -------
        pd.DataFrame
            DataFrame with initial missing value imputations.
        """
        X_filled = X.copy()
        for col in X.columns:
            if X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    X_filled[col] = X_filled[col].fillna(X[col].mean())
                else:
                    X_filled[col] = X_filled[col].fillna(X[col].mode()[0])
        return X_filled

    def _impute_column(self, X_filled, missing_mask, col):
        """
        Impute a single column in the DataFrame.

        Parameters:
        ----------
        X_filled : pd.DataFrame
            The DataFrame with imputed values so far.

        missing_mask : pd.DataFrame
            Boolean DataFrame indicating the missing values.

        col : str
            The column to be imputed.
        """
        missing_idx = missing_mask[col]
        if missing_idx.any():
            # Define features and target
            y_train = X_filled.loc[~missing_idx, col].copy()

            # Ensure no NaN values in y_train
            if y_train.isnull().any():
                raise ValueError(f"NaN values found in y_train for column: {col}")

            # Drop the column for model training
            X_temp = X_filled.drop(columns=[col])
            %time X_between = column_trans.fit_transform(X_temp)
            X_train = X_between[~missing_idx]
            X_missing = X_between[missing_idx]

            # Fit the Random Forest
            if pd.api.types.is_numeric_dtype(X_filled[col]):
                rf = XGBRegressor(tree_method='hist', device='cuda') if self.use_gpu else XGBRegressor(tree_method='hist')
            else:
                y_train = enc.fit_transform(y_train)
                rf = XGBClassifier(tree_method='hist', device='cuda') if self.use_gpu else XGBClassifier(tree_method='hist')

            %time rf.fit(X_train, y_train)
            y_pred = rf.predict(X_missing)
            y_pred = enc.inverse_transform(y_pred) if not pd.api.types.is_numeric_dtype(X_filled[col]) else y_pred
            X_filled.loc[missing_idx, col] = y_pred

    def fit_transform(self, X, convergence_criterium=1e-5):
        """
        Fits the MissForest imputer on the data and returns the imputed dataset.

        Parameters:
        ----------
        X : pd.DataFrame
            Input data with missing values.

        convergence_criterium : float, optional (default=1e-5)
            The threshold for convergence, stopping the iterations if the 
            mean difference between successive iterations is below this value.

        Returns:
        -------
        pd.DataFrame
            The dataset with missing values imputed.
        """
        # Initialize missing values
        X_filled = self._initialize(X)
        missing_mask = X.isnull()
        previous_filled = X_filled.copy()
        
        previous_diff = np.inf
        
        for iteration in range(self.max_iter):
            print('Iteration: ', iteration + 1)
            
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._impute_column)(X_filled, missing_mask, col) for col in tqdm(X.columns)
            )

            # Calculate differences
            X_filled_transformed = column_trans.fit_transform(X_filled)
            previous_filled_transformed = column_trans.transform(previous_filled)
            current_diff = np.mean(np.abs(X_filled_transformed - previous_filled_transformed))
            print(f"Mean difference for iteration {iteration + 1}: {current_diff:.6f}")

            # Check if the difference is not decreasing
            if np.abs(previous_diff - current_diff) < convergence_criterium:
                print("Convergence detected.")
                break

            previous_diff = current_diff
            previous_filled = X_filled.copy()

        return X_filled
