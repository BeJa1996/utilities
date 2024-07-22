"""
Utility module with different helper functions.

Author: Benjamin Jargow
Last Edited: 22.07.2024
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def boxplot_num(dataset, features=None):
    """
    Provides a boxplot for each numerical feature in the dataset.
    
    Parameters:
    dataset (pd.DataFrame): The DataFrame containing the data.
    features (list): List of numerical feature names to plot. If None, selects all numerical features.
    """
    if features is None:
        features = dataset.select_dtypes(include=["int64", "float64"]).columns

    # Plot boxplot for each numerical feature
    plt.figure(figsize=(10, 24))
    for idx, feature in enumerate(features, 1):
        plt.subplot(len(features), 2, idx)
        sns.boxplot(dataset[feature])
        plt.title(f"{feature}")

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

def scatter_num(dataset, features=None):
    """
    Provides a scatter plot for each numerical feature in the dataset, showing counts.
    
    Parameters:
    dataset (pd.DataFrame): The DataFrame containing the data.
    features (list): List of numerical feature names to plot. If None, selects all numerical features.
    """
    if features is None:
        features = dataset.select_dtypes(include=["int64", "float64"]).columns

    # Plot scatter plot for each numerical feature
    plt.figure(figsize=(10, 24))
    for idx, feature in enumerate(features, 1):
        plt.subplot(len(features), 2, idx)
        t = dataset.groupby(feature, as_index=False)[feature].count()
        sns.scatterplot(y=t[feature], x=t.index)
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.title(f"{feature}")

def hist_numerical(dataset, features=None):
    """
    Provides a histogram and KDE plot for each numerical feature in the dataset.
    
    Parameters:
    dataset (pd.DataFrame): The DataFrame containing the data.
    features (list): List of numerical feature names to plot. If None, selects all numerical features.
    """
    if features is None:
        features = dataset.select_dtypes(include=["int64", "float64"]).columns

    # Plot distribution of each numerical feature
    plt.figure(figsize=(14, 18))
    for idx, feature in enumerate(features, 1):
        plt.subplot(len(features), 2, idx)
        sns.histplot(dataset[feature], kde=True)
        plt.title(f"{feature} | Skewness: {round(dataset[feature].skew(), 2)}")

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

def trends_numerical(dataset, x=None, features=None):
    """
    Provides a trend plot for each numerical feature in the dataset.
    
    Parameters:
    dataset (pd.DataFrame): The DataFrame containing the data.
    x (str): The feature to use as the x-axis. If None, uses the DataFrame index.
    features (list): List of numerical feature names to plot. If None, selects all numerical features.
    """
    if features is None:
        features = dataset.select_dtypes(include=["int64", "float64"]).columns
    x = dataset.index if x is None else dataset[x]
    
    # Plot trend of each numerical feature
    plt.figure(figsize=(14, 20))
    for idx, feature in enumerate(features, 1):
        plt.subplot(len(features), 2, idx)
        sns.lineplot(x=x, y=dataset[feature])
        plt.title(f"{feature}")

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

def countplot_all(dataset, features=None):
    """
    Provides a count plot for all categorical features in the dataset.
    
    Parameters:
    dataset (pd.DataFrame): The DataFrame containing the data.
    features (list): List of categorical feature names to plot. If None, selects all categorical features.
    """
    if features is None:
        features = dataset.select_dtypes(include=["category", "object", "bool"]).columns

    # Plot count plot for each categorical feature
    plt.figure(figsize=(14, 18))
    for idx, feature in enumerate(features, 1):
        plt.subplot(len(features), 2, idx)
        p = sns.countplot(data=dataset, x=feature)
        p.bar_label(p.containers[0])
        plt.title(f"{feature}")

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

def pair_plot(dataset, features=None):
    """
    Creates pair plots for the numerical features in the dataset with correlation annotations.
    
    Parameters:
    dataset (pd.DataFrame): The DataFrame containing the data.
    features (list): List of numerical feature names to plot. If None, selects all numerical features.
    """
    if features is None:
        features = dataset.select_dtypes(include=["int64", "float64"]).columns
    
    dataset = dataset[features]
        
    def corrdot(*args, **kwargs): # adapted from https://stackoverflow.com/a/50690729
        corr_r = args[0].corr(args[1], 'pearson')
        corr_text = f"{corr_r:2.2f}".replace("0.", ".")
        ax = plt.gca()
        ax.set_axis_off()
        marker_size = abs(corr_r) * 8000
        ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
                   vmin=-1, vmax=1, transform=ax.transAxes)
        font_size = abs(corr_r) * 15 + 15
        ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                    ha='center', va='center', fontsize=font_size)
    
    g = sns.PairGrid(dataset, diag_sharey=False, aspect=1.4)
    g.map_lower(sns.regplot, lowess=True, 
                    line_kws=dict(color="black"), marker=".", ci=None)
    g.map_diag(sns.histplot, kde=True, alpha=1)
    g.map_upper(corrdot)

def boxplot_cat(dataset, y, features=None):
    """
    Provides a boxplot and violin plot for each categorical feature in the dataset against a specified y variable.
    
    Parameters:
    dataset (pd.DataFrame): The DataFrame containing the data.
    y (str): The dependent variable.
    features (list): List of categorical feature names to plot. If None, selects all categorical features.
    """
    if features is None:
        features = dataset.select_dtypes(include=["category", "object", "bool"]).columns

    # Plot boxplot and violin plot for each categorical feature
    plt.figure(figsize=(14, 18))
    for idx, feature in enumerate(features, 1):
        plt.subplot(len(features), 2, idx)
        ax = sns.violinplot(data=dataset, x=feature, y=y, inner=None, color='black', zorder=100)
        for violin in ax.collections:
            violin.set_alpha(.1)
        sns.boxplot(data=dataset, x=feature, y=y, ax=ax)
        plt.title(f"{feature}")

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

def rel_plots(dataset, y, x=None):
    """
    Creates correlation plots for every numerical variable with a set dependent variable.
    Also provides a smooth average to check for linearity.
    
    Parameters:
    dataset (pd.DataFrame): The DataFrame containing the data.
    y (str): The dependent variable.
    x (list): List of numerical feature names to plot against y. If None, selects all numerical features except y.
    """
    if x is None:
        features = dataset.select_dtypes(include=["int64", "float64"]).columns
        features = features.drop(y).to_list()
    else:
        features = x

    # Plot regression and scatter plot for each numerical feature
    plt.figure(figsize=(14, 20))
    for idx, feature in enumerate(features, 1):
        plt.subplot(len(features), 2, idx)
        sns.regplot(data=dataset, x=feature, y=y,
                    lowess=True, x_jitter=.15, y_jitter=.15, 
                    line_kws=dict(color="r"), marker=".")
        plt.title(f"{feature}")

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

def optimal_config_count(data, feature_columns, comparison_column, outcome, optimum='max'):
    """
    Count the number of configurations where each level of the comparison column 
    has the optimal (maximum or minimum) value in the outcome column.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    feature_columns (list): The list of feature columns to consider for configurations.
    comparison_column (str): The column to compare within each configuration.
    outcome (str): The column containing the outcome values.
    optimum (str): 'max' to consider the maximum value as optimal, 'min' to consider the minimum value as optimal.

    Returns:
    dict: A dictionary with the count of optimal outcomes for each level of the comparison column.
    """
    
    # Group by feature columns and comparison column, and calculate the mean outcome
    grouped = data.groupby(feature_columns + [comparison_column])[outcome].mean().reset_index()

    # Initialize a dictionary to count optimal outcomes for each level of the comparison column
    optimal_counts = {level: 0 for level in data[comparison_column].unique()}
    
    # Group by feature columns and find the optimal outcome for each configuration
    for _, group in grouped.groupby(feature_columns):
        if optimum == 'max':
            # Find the row with the maximum outcome
            optimal_row = group.loc[group[outcome].idxmax()]
        elif optimum == 'min':
            # Find the row with the minimum outcome
            optimal_row = group.loc[group[outcome].idxmin()]
        else:
            raise ValueError("Parameter 'optimum' must be either 'max' or 'min'.")
        
        optimal_level = optimal_row[comparison_column]
        optimal_counts[optimal_level] += 1
    
    return optimal_counts
