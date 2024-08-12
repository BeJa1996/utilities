"""
missforest_imputer.py

This module implements the MissForest algorithm for handling missing data
by iteratively imputing missing values using XGBoost models. The module is 
designed to work with datasets containing both numerical and categorical data.

Classes:
--------
MissForestImputer: 
    A class for imputing missing data using an iterative approach with 
    XGBoost models.

Functions:
----------
None

Dependencies:
-------------
- xgboost
- scikit-learn
- numpy
- pandas
- tqdm

Author:
-------
Benjamin Jargow (and ChatGPT)

Date:
-----
August 12, 2024

Version:
--------
1.0.0
"""


from xgboost import XGBClassifier, XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define the ColumnTransformer and LabelEncoder
column_trans = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), lambda df: df.select_dtypes(include=[np.number]).columns),
        ('cat', OneHotEncoder(), lambda df: df.select_dtypes(include=['object', 'category']).columns)
    ],
    remainder='drop',
    sparse_threshold=0
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

    Methods:
    -------
    fit_transform(X, convergence_criterium=1e-5):
        Fits the MissForest imputer on the data and returns the imputed dataset.
    """

    def __init__(self, max_iter=10, random_state=None):
        """
        Initializes the MissForestImputer with the specified maximum number 
        of iterations and random state.
        """
        self.max_iter = max_iter
        self.random_state = random_state

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
