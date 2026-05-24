"""
Configuration file for the Sentiment Analysis and Engagement Prediction project.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_REG = PROJECT_ROOT / "models" / "regression"
MODELS_CLF = PROJECT_ROOT / "models" / "classification"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FIGURES = RESULTS_DIR / "figures"
RESULTS_TABLES = RESULTS_DIR / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [DATA_PROCESSED, MODELS_REG, MODELS_CLF, RESULTS_FIGURES, RESULTS_TABLES, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================

DATASET_RAW = DATA_RAW / "social_media_engagement_data.xlsx"
DATASET_CLEANED = DATA_PROCESSED / "cleaned_dataset.csv"
DATASET_TRAIN = DATA_PROCESSED / "train_dataset.csv"
DATASET_VAL = DATA_PROCESSED / "val_dataset.csv"
DATASET_TEST = DATA_PROCESSED / "test_dataset.csv"
DATASET_FEATURES = DATA_PROCESSED / "features_dataset.csv"

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

# Missing value handling
NUMERICAL_MISSING_STRATEGY = "median"  # "median" or "mean"
CATEGORICAL_MISSING_STRATEGY = "most_frequent"  # "most_frequent" or "unknown"
TEXT_MISSING_VALUE = ""  # empty string for missing text

# Outlier handling - impossible values
MIN_AGE = 0
MAX_AGE = 100
MIN_LIKES = 0
MIN_COMMENTS = 0
MIN_SHARES = 0
MIN_IMPRESSIONS = 0

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

# TF-IDF parameters
TFIDF_MAX_FEATURES = 500
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

# Sentiment analysis
SENTIMENT_MODEL = "vader"  # "vader", "textblob", or "transformer"

# Text features
EXTRACT_TEXT_FEATURES = True
EXTRACT_TFIDF_FEATURES = True
EXTRACT_SENTIMENT_FEATURES = True

# ============================================================================
# TARGET VARIABLES AND THRESHOLDS
# ============================================================================

# Regression targets
REGRESSION_TARGETS = [
    "likes",
    "comments",
    "shares",
    "impressions",
    "reach",
    "engagement_rate"
]

# Classification targets
CLASSIFICATION_TARGETS = {
    "sentiment": None,  # Already in dataset if present
    "viral": None,  # To be created
    "high_engagement": None  # To be created
}

# Thresholds for binary classification targets
VIRAL_THRESHOLD = 0.75  # 75th percentile
HIGH_ENGAGEMENT_THRESHOLD = 0.5  # 50th percentile or median

# ============================================================================
# FEATURES TO EXCLUDE (Data Leakage Prevention)
# ============================================================================

# These must NOT be used as input features
LEAKAGE_FEATURES = [
    "likes",
    "comments",
    "shares",
    "impressions",
    "reach",
    "engagement_rate",
    "post_id"  # Identifier only
]

# These should be carefully considered
FEATURES_TO_EXPERIMENT = [
    "campaign_id",
    "influencer_id"
]

# ============================================================================
# FEATURE SETS FOR EXPERIMENTS
# ============================================================================

# Experiment A: Without Campaign ID and Influencer ID
EXPERIMENT_A_EXCLUDE = ["campaign_id", "influencer_id"]

# Experiment B: With Campaign ID and Influencer ID
EXPERIMENT_B_EXCLUDE = []

# ============================================================================
# TRAIN/TEST SPLIT PARAMETERS
# ============================================================================

TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

RANDOM_STATE = 42

# Use time-based split (if temporal column exists)
USE_TIME_BASED_SPLIT = False
TIME_COLUMN = "post_timestamp"

# ============================================================================
# CATEGORICAL FEATURES
# ============================================================================

CATEGORICAL_FEATURES = [
    "platform",
    "post_type",
    "weekday_type",
    "time_periods",
    "age_group",
    "audience_gender",
    "audience_location",
    "audience_content",
    "audience_interests"
]

# High-cardinality features
HIGH_CARDINALITY_FEATURES = [
    "campaign_id",
    "influencer_id"
]

# ============================================================================
# MODEL TRAINING PARAMETERS
# ============================================================================

# Random state for reproducibility
SEED = 42

# Cross-validation
CV_FOLDS = 5
USE_STRATIFIED_KF = True  # For classification

# Regression models
REGRESSION_MODELS = [
    "linear_regression",
    "ridge_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "svr"
]

# Classification models
CLASSIFICATION_MODELS = [
    "logistic_regression",
    "random_forest",
    "svm",
    "xgboost",
    "lightgbm",
    "catboost"
]

# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================

USE_HYPERPARAMETER_TUNING = True
OPTUNA_N_TRIALS = 50  # Number of trials for Optuna
OPTUNA_TIMEOUT = 3600  # Timeout in seconds per model

# ============================================================================
# EVALUATION METRICS
# ============================================================================

REGRESSION_METRICS = ["mae", "rmse", "r2", "mape", "median_absolute_error"]
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1_macro", "f1_weighted", "roc_auc"]

# ============================================================================
# INTERPRETABILITY PARAMETERS
# ============================================================================

USE_SHAP = True
USE_PERMUTATION_IMPORTANCE = True
TOP_N_FEATURES = 20  # Top N features to display

# ============================================================================
# LOGGING AND VERBOSITY
# ============================================================================

VERBOSE = True
LOG_LEVEL = "INFO"
