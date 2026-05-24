"""
Utility functions for the Sentiment Analysis and Engagement Prediction project.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from config import LOG_LEVEL, RESULTS_FIGURES

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set matplotlib and seaborn styles
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def get_logger(name):
    """Get a logger instance."""
    return logging.getLogger(name)


def standardize_column_names(df):
    """
    Standardize column names to lowercase with underscores.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    return df


def check_missing_values(df, return_df=False):
    """
    Check and report missing values in the dataset.
    
    Args:
        df: pandas DataFrame
        return_df: If True, return a DataFrame with missing value info
        
    Returns:
        DataFrame with missing value statistics (if return_df=True)
    """
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_percent.values,
        'Data_Type': df.dtypes.values
    }).sort_values('Missing_Count', ascending=False)
    
    missing_data = missing_data[missing_data['Missing_Count'] > 0]
    
    if len(missing_data) > 0:
        logger.info(f"Found {len(missing_data)} columns with missing values:")
        logger.info(missing_data.to_string())
    else:
        logger.info("No missing values found.")
    
    if return_df:
        return missing_data
    
    return missing


def check_duplicates(df):
    """
    Check for duplicate rows.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Number of duplicate rows
    """
    n_duplicates = df.duplicated().sum()
    logger.info(f"Found {n_duplicates} duplicate rows.")
    return n_duplicates


def detect_impossible_values(df, column, min_val=None, max_val=None):
    """
    Detect impossible values in a column.
    
    Args:
        df: pandas DataFrame
        column: Column name
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value
        
    Returns:
        DataFrame with impossible values
    """
    if column not in df.columns:
        return pd.DataFrame()
    
    mask = pd.Series([False] * len(df))
    
    if min_val is not None:
        mask = mask | (df[column] < min_val)
    
    if max_val is not None:
        mask = mask | (df[column] > max_val)
    
    impossible_rows = df[mask]
    
    if len(impossible_rows) > 0:
        logger.warning(f"Found {len(impossible_rows)} rows with impossible values in '{column}'")
    
    return impossible_rows


def save_figure(fig, filename, dpi=300, bbox_inches='tight'):
    """
    Save a matplotlib figure.
    
    Args:
        fig: matplotlib figure object
        filename: Name of the file (without path)
        dpi: Resolution
        bbox_inches: Bounding box
    """
    filepath = RESULTS_FIGURES / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f"Saved figure: {filepath}")
    plt.close(fig)


def save_table(df, filename):
    """
    Save a table to CSV.
    
    Args:
        df: pandas DataFrame
        filename: Name of the file (without path)
    """
    from config import RESULTS_TABLES
    filepath = RESULTS_TABLES / filename
    df.to_csv(filepath, index=False)
    logger.info(f"Saved table: {filepath}")


def plot_distribution(data, title, xlabel, ylabel='Frequency', bins=30, log_scale=False):
    """
    Plot distribution of a variable.
    
    Args:
        data: Series or array
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        bins: Number of bins
        log_scale: If True, use log scale for y-axis
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Remove NaN values
    data_clean = data.dropna() if isinstance(data, pd.Series) else data[~np.isnan(data)]
    
    ax.hist(data_clean, bins=bins, edgecolor='black', alpha=0.7)
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_correlation_heatmap(df_numeric, figsize=(12, 10)):
    """
    Plot correlation heatmap.
    
    Args:
        df_numeric: DataFrame with numeric columns
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    corr_matrix = df_numeric.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    
    ax.set_title('Correlation Matrix of Numeric Features', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    return fig, ax


def calculate_summary_stats(df):
    """
    Calculate summary statistics for numeric columns.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame with summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    stats = df[numeric_cols].describe().T
    stats['skewness'] = df[numeric_cols].skew()
    stats['kurtosis'] = df[numeric_cols].kurtosis()
    
    return stats


def group_and_summarize(df, groupby_col, numeric_cols, agg_funcs=['mean', 'median', 'std', 'count']):
    """
    Group by a column and summarize numeric columns.
    
    Args:
        df: pandas DataFrame
        groupby_col: Column to group by
        numeric_cols: List of numeric columns to aggregate
        agg_funcs: Aggregation functions
        
    Returns:
        Aggregated DataFrame
    """
    agg_dict = {col: agg_funcs for col in numeric_cols}
    result = df.groupby(groupby_col).agg(agg_dict)
    return result


def print_dataset_overview(df):
    """
    Print a comprehensive overview of the dataset.
    
    Args:
        df: pandas DataFrame
    """
    logger.info("=" * 80)
    logger.info("DATASET OVERVIEW")
    logger.info("=" * 80)
    logger.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info("\nColumn Information:")
    logger.info(df.info())
    logger.info("\nFirst few rows:")
    logger.info(df.head())
    logger.info("\nData types:")
    logger.info(df.dtypes)
    logger.info("=" * 80)


def create_threshold_binary_labels(series, threshold, label_name):
    """
    Create binary labels based on a threshold.
    
    Args:
        series: pandas Series
        threshold: Threshold value
        label_name: Name of the label
        
    Returns:
        Binary Series (1 if >= threshold, 0 otherwise)
    """
    return (series >= threshold).astype(int)


def create_percentile_based_labels(series, percentile, label_name):
    """
    Create binary labels based on percentile.
    
    Args:
        series: pandas Series
        percentile: Percentile value (0-100)
        label_name: Name of the label
        
    Returns:
        Binary Series (1 if >= percentile, 0 otherwise)
    """
    threshold = series.quantile(percentile / 100)
    return (series >= threshold).astype(int)
