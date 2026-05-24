"""
Data loading and initial exploration module.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from config import (
    DATASET_RAW, DATASET_CLEANED, CATEGORICAL_FEATURES,
    NUMERICAL_MISSING_STRATEGY, CATEGORICAL_MISSING_STRATEGY,
    TEXT_MISSING_VALUE, MIN_AGE, MAX_AGE, MIN_LIKES, MIN_COMMENTS,
    MIN_SHARES, MIN_IMPRESSIONS, LEAKAGE_FEATURES
)
from utils import (
    standardize_column_names, check_missing_values, check_duplicates,
    detect_impossible_values, print_dataset_overview, get_logger
)

logger = get_logger(__name__)


def load_raw_dataset(filepath=DATASET_RAW):
    """
    Load the raw dataset from Excel file.
    
    Args:
        filepath: Path to the Excel file
        
    Returns:
        pandas DataFrame
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    logger.info(f"Loading dataset from: {filepath}")
    
    # Load Excel file
    df = pd.read_excel(filepath)
    
    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    
    return df


def clean_dataset(df):
    """
    Clean the dataset: standardize names, handle missing values, remove duplicates, etc.
    
    Args:
        df: Raw pandas DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning process...")
    
    # 1. Standardize column names
    logger.info("Step 1: Standardizing column names...")
    df = standardize_column_names(df)
    
    # 2. Check initial state
    logger.info(f"Initial dataset shape: {df.shape}")
    check_missing_values(df)
    n_dups_before = check_duplicates(df)
    
    # 3. Remove duplicates
    logger.info("Step 2: Removing duplicate rows...")
    df = df.drop_duplicates()
    n_dups_after = check_duplicates(df)
    logger.info(f"Removed {n_dups_before - n_dups_after} duplicate rows")
    
    # 4. Convert datetime columns
    logger.info("Step 3: Converting datetime columns...")
    datetime_columns = ['post_timestamp', 'date']
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            logger.info(f"Converted '{col}' to datetime")
    
    # 5. Detect and remove impossible values
    logger.info("Step 4: Detecting and handling impossible values...")
    
    impossible_masks = []
    
    if 'audience_age' in df.columns:
        impossible = detect_impossible_values(df, 'audience_age', MIN_AGE, MAX_AGE)
        if len(impossible) > 0:
            df = df[~df.index.isin(impossible.index)]
            logger.info(f"Removed {len(impossible)} rows with invalid age values")
    
    for col in ['likes', 'comments', 'shares', 'impressions']:
        if col in df.columns:
            impossible = detect_impossible_values(df, col, 0, None)
            if len(impossible) > 0:
                df = df[~df.index.isin(impossible.index)]
                logger.info(f"Removed {len(impossible)} rows with negative {col} values")
    
    # 6. Handle missing values
    logger.info("Step 5: Handling missing values...")
    
    df = handle_missing_values(df)
    
    # 7. Create date/time features if needed
    logger.info("Step 6: Creating time-based features...")
    df = create_datetime_features(df)
    
    # 8. Final report
    logger.info("Step 7: Final data quality report...")
    logger.info(f"Final dataset shape: {df.shape}")
    check_missing_values(df)
    check_duplicates(df)
    
    logger.info("Data cleaning completed successfully!")
    
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    missing_info = check_missing_values(df, return_df=True)
    
    if len(missing_info) == 0:
        logger.info("No missing values to handle.")
        return df
    
    # Separate by data type
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle numeric columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if NUMERICAL_MISSING_STRATEGY == "median":
                fill_value = df[col].median()
                logger.info(f"Filling '{col}' with median: {fill_value:.2f}")
            else:
                fill_value = df[col].mean()
                logger.info(f"Filling '{col}' with mean: {fill_value:.2f}")
            
            df[col] = df[col].fillna(fill_value)
    
    # Handle categorical columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            if CATEGORICAL_MISSING_STRATEGY == "most_frequent":
                fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                logger.info(f"Filling '{col}' with mode: {fill_value}")
            else:
                fill_value = "Unknown"
                logger.info(f"Filling '{col}' with 'Unknown'")
            
            df[col] = df[col].fillna(fill_value)
    
    # Handle text columns
    text_cols = [col for col in df.columns if col not in numeric_cols and col not in categorical_cols]
    for col in text_cols:
        if df[col].isnull().sum() > 0:
            logger.info(f"Filling '{col}' with empty string")
            df[col] = df[col].fillna(TEXT_MISSING_VALUE)
    
    return df


def create_datetime_features(df):
    """
    Create additional datetime features from timestamp columns.
    
    Args:
        df: pandas DataFrame with datetime columns
        
    Returns:
        DataFrame with new datetime features
    """
    if 'post_timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['post_timestamp']):
        df['publish_hour'] = df['post_timestamp'].dt.hour
        df['publish_day'] = df['post_timestamp'].dt.day
        df['publish_month'] = df['post_timestamp'].dt.month
        df['publish_weekday'] = df['post_timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['is_weekend'] = df['publish_weekday'].isin([5, 6]).astype(int)
        
        logger.info("Created datetime features: publish_hour, publish_day, publish_month, publish_weekday, is_weekend")
    
    return df


def save_cleaned_dataset(df, filepath=DATASET_CLEANED):
    """
    Save the cleaned dataset to CSV.
    
    Args:
        df: Cleaned DataFrame
        filepath: Path to save the file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Cleaned dataset saved to: {filepath}")


def get_dataset_overview(df):
    """
    Get a comprehensive overview of the dataset.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Dictionary with overview statistics
    """
    overview = {
        'n_rows': df.shape[0],
        'n_columns': df.shape[1],
        'n_numeric': len(df.select_dtypes(include=[np.number]).columns),
        'n_categorical': len(df.select_dtypes(include=['object', 'category']).columns),
        'n_datetime': len(df.select_dtypes(include=['datetime64']).columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return overview


def verify_no_leakage(df):
    """
    Verify that leakage features are not in the dataset (after cleaning).
    
    Args:
        df: pandas DataFrame
        
    Returns:
        List of leakage features found in the dataset
    """
    leakage_found = [col for col in LEAKAGE_FEATURES if col in df.columns]
    
    if leakage_found:
        logger.warning(f"Found potential leakage features: {leakage_found}")
        logger.warning("These features should be removed before training models.")
    else:
        logger.info("No obvious leakage features detected.")
    
    return leakage_found


if __name__ == "__main__":
    # Load and clean the dataset
    df_raw = load_raw_dataset()
    print_dataset_overview(df_raw)
    
    df_cleaned = clean_dataset(df_raw)
    print_dataset_overview(df_cleaned)
    
    overview = get_dataset_overview(df_cleaned)
    logger.info(f"Dataset Overview: {overview}")
    
    verify_no_leakage(df_cleaned)
    
    save_cleaned_dataset(df_cleaned)
