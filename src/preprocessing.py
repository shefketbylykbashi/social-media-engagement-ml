"""
Advanced feature engineering and preprocessing module.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import logging
import emoji
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import string

from config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_MAX_DF,
    SENTIMENT_MODEL, EXTRACT_TEXT_FEATURES, EXTRACT_TFIDF_FEATURES,
    EXTRACT_SENTIMENT_FEATURES, LEAKAGE_FEATURES, HIGH_ENGAGEMENT_THRESHOLD,
    VIRAL_THRESHOLD, REGRESSION_TARGETS
)
from utils import get_logger, create_percentile_based_labels

logger = get_logger(__name__)

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Download NLTK resources if needed
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def create_text_features(df):
    """
    Create text-based features from post content.
    
    Args:
        df: pandas DataFrame with 'post_content' column
        
    Returns:
        DataFrame with added text features
    """
    if 'post_content' not in df.columns:
        logger.warning("'post_content' column not found. Skipping text features.")
        return df
    
    logger.info("Creating text features...")
    
    df = df.copy()
    
    # Length features
    df['text_length_chars'] = df['post_content'].str.len()
    df['text_length_words'] = df['post_content'].str.split().str.len()
    
    # Sentence count
    df['sentence_count'] = df['post_content'].apply(
        lambda x: len(sent_tokenize(str(x))) if pd.notna(x) else 0
    )
    
    # Average word length
    df['average_word_length'] = df['post_content'].apply(
        lambda x: np.mean([len(word) for word in str(x).split()]) if pd.notna(x) and len(str(x)) > 0 else 0
    )
    
    # Question and exclamation marks
    df['has_question'] = df['post_content'].str.contains('\\?', na=False).astype(int)
    df['question_count'] = df['post_content'].str.count('\\?')
    
    df['has_exclamation'] = df['post_content'].str.contains('!', na=False).astype(int)
    df['exclamation_count'] = df['post_content'].str.count('!')
    
    # Hashtags and mentions
    df['has_hashtag'] = df['post_content'].str.contains('#', na=False).astype(int)
    df['hashtag_count'] = df['post_content'].str.count('#')
    
    df['has_mention'] = df['post_content'].str.contains('@', na=False).astype(int)
    df['mention_count'] = df['post_content'].str.count('@')
    
    # URLs
    df['has_url'] = df['post_content'].apply(
        lambda x: int(bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(x)))) if pd.notna(x) else 0
    )
    
    # Emojis
    df['has_emoji'] = df['post_content'].apply(
        lambda x: int(bool(emoji.emoji_list(str(x)))) if pd.notna(x) else 0
    )
    
    # Numbers
    df['has_number'] = df['post_content'].str.contains('\\d', na=False).astype(int)
    
    # Case ratio
    df['uppercase_ratio'] = df['post_content'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
    )
    
    # Punctuation count
    # Punctuation count
    df['punctuation_count'] = df['post_content'].apply(
        lambda x: sum(1 for c in str(x) if c in string.punctuation) if pd.notna(x) else 0
    )
    
    logger.info("Text features created successfully.")
    
    return df


def create_sentiment_features(df, model=SENTIMENT_MODEL):
    """
    Create sentiment features from post content text.
    
    Args:
        df: pandas DataFrame with 'post_content' column
        model: Sentiment model to use ('vader' or 'textblob')
        
    Returns:
        DataFrame with sentiment features
    """
    if 'post_content' not in df.columns:
        logger.warning("'post_content' column not found. Skipping sentiment features.")
        return df
    
    logger.info(f"Creating sentiment features using {model} model...")
    
    df = df.copy()
    
    if model == 'vader':
        # VADER Sentiment Analysis
        def get_vader_sentiment(text):
            if pd.isna(text):
                return 0, 'Neutral'
            scores = vader_analyzer.polarity_scores(str(text))
            sentiment_score = scores['compound']  # -1 to 1
            
            if sentiment_score > 0.05:
                label = 'Positive'
            elif sentiment_score < -0.05:
                label = 'Negative'
            else:
                label = 'Neutral'
            
            return sentiment_score, label
        
        df[['pre_post_sentiment_score', 'pre_post_sentiment_label']] = df['post_content'].apply(
            lambda x: pd.Series(get_vader_sentiment(x))
        )
        
    elif model == 'textblob':
        # TextBlob Sentiment Analysis
        def get_textblob_sentiment(text):
            if pd.isna(text):
                return 0, 'Neutral'
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity  # -1 to 1
            
            if polarity > 0.1:
                label = 'Positive'
            elif polarity < -0.1:
                label = 'Negative'
            else:
                label = 'Neutral'
            
            return polarity, label
        
        df[['pre_post_sentiment_score', 'pre_post_sentiment_label']] = df['post_content'].apply(
            lambda x: pd.Series(get_textblob_sentiment(x))
        )
    
    else:
        logger.warning(f"Unknown sentiment model: {model}. Using VADER as default.")
        return create_sentiment_features(df, model='vader')
    
    logger.info("Sentiment features created successfully.")
    
    return df


def create_tfidf_features(df, max_features=TFIDF_MAX_FEATURES, 
                         ngram_range=TFIDF_NGRAM_RANGE,
                         min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF):
    """
    Create TF-IDF features from post content.
    
    Args:
        df: pandas DataFrame with 'post_content' column
        max_features: Maximum number of features
        ngram_range: Ngram range for TF-IDF
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        
    Returns:
        DataFrame with TF-IDF features
    """
    if 'post_content' not in df.columns:
        logger.warning("'post_content' column not found. Skipping TF-IDF features.")
        return df, None
    
    logger.info(f"Creating TF-IDF features (max_features={max_features})...")
    
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words='english'
    )
    
    # Fit and transform
    tfidf_features = tfidf.fit_transform(df['post_content'].fillna(''))
    
    # Convert to dense array and DataFrame
    tfidf_array = tfidf_features.toarray()
    feature_names = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_array, columns=[f'tfidf_{i}_{name}' for i, name in enumerate(feature_names)])
    
    logger.info(f"Created {len(feature_names)} TF-IDF features.")
    
    return tfidf_df, tfidf


def encode_categorical_features(df, categorical_cols):
    """
    One-hot encode categorical features.
    
    Args:
        df: pandas DataFrame
        categorical_cols: List of categorical column names
        
    Returns:
        DataFrame with encoded features
    """
    logger.info(f"One-hot encoding {len(categorical_cols)} categorical features...")
    
    df = df.copy()
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    logger.info(f"Categorical encoding completed. New shape: {df_encoded.shape}")
    
    return df_encoded


def create_target_variables(df, engagement_col='engagement_rate'):
    """
    Create target variables for classification tasks.
    
    Args:
        df: pandas DataFrame
        engagement_col: Column to use for binary classification targets
        
    Returns:
        DataFrame with target variables
    """
    logger.info("Creating target variables...")
    
    df = df.copy()
    
    # Create 'viral' target based on engagement rate
    if engagement_col in df.columns:
        df['viral'] = create_percentile_based_labels(
            df[engagement_col], VIRAL_THRESHOLD * 100, 'viral'
        )
        logger.info(f"Created 'viral' target (threshold: {VIRAL_THRESHOLD}th percentile)")
        
        # Create 'high_engagement' target
        df['high_engagement'] = create_percentile_based_labels(
            df[engagement_col], HIGH_ENGAGEMENT_THRESHOLD * 100, 'high_engagement'
        )
        logger.info(f"Created 'high_engagement' target (threshold: {HIGH_ENGAGEMENT_THRESHOLD}th percentile)")
    
    return df


def remove_leakage_features(df, exclude_list=None):
    """
    Remove features that could cause data leakage.
    
    Args:
        df: pandas DataFrame
        exclude_list: List of additional features to exclude
        
    Returns:
        DataFrame without leakage features
    """
    logger.info("Removing leakage features...")
    
    df = df.copy()
    
    features_to_drop = list(LEAKAGE_FEATURES)
    if exclude_list:
        features_to_drop.extend(exclude_list)
    
    # Remove duplicates from the list
    features_to_drop = list(set(features_to_drop))
    
    # Only drop if they exist in the DataFrame
    features_to_drop = [col for col in features_to_drop if col in df.columns]
    
    if features_to_drop:
        logger.info(f"Removing features: {features_to_drop}")
        df = df.drop(columns=features_to_drop)
    
    logger.info(f"Leakage features removed. New shape: {df.shape}")
    
    return df


def preprocess_dataset(df, experiment_exclude=None):
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Raw pandas DataFrame
        experiment_exclude: List of additional features to exclude
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info("=" * 80)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    
    df = df.copy()
    
    # Step 1: Create target variables
    df = create_target_variables(df)
    
    # Step 2: Create text features
    if EXTRACT_TEXT_FEATURES:
        df = create_text_features(df)
    
    # Step 3: Create sentiment features
    if EXTRACT_SENTIMENT_FEATURES:
        df = create_sentiment_features(df, model=SENTIMENT_MODEL)
    
    # Step 4: Create TF-IDF features
    if EXTRACT_TFIDF_FEATURES:
        tfidf_df, tfidf_vectorizer = create_tfidf_features(df)
        df = pd.concat([df, tfidf_df], axis=1)
    
    # Step 5: Remove the original post_content (it's been processed)
    if 'post_content' in df.columns:
        df = df.drop(columns=['post_content'])
    
    # Step 6: Remove leakage features
    df = remove_leakage_features(df, exclude_list=experiment_exclude)
    
    # Step 7: Remove datetime columns (models can't handle them)
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        logger.info(f"Removing {len(datetime_cols)} datetime columns: {datetime_cols}")
        df = df.drop(columns=datetime_cols)
    
    # Step 8: Encode categorical features (memory-efficient approach)
    # Find all categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Remove target columns from encoding (they should be numeric/binary already)
    target_cols = ['viral', 'high_engagement', 'sentiment']
    categorical_cols = [col for col in categorical_cols if col not in target_cols]
    
    if categorical_cols:
        logger.info(f"Encoding {len(categorical_cols)} categorical features using LabelEncoder: {categorical_cols}")
        # Use LabelEncoder for memory efficiency instead of one-hot encoding
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
        logger.info(f"Categorical encoding completed. New shape: {df.shape}")
    
    logger.info("=" * 80)
    logger.info("PREPROCESSING PIPELINE COMPLETED")
    logger.info(f"Final shape: {df.shape}")
    logger.info("=" * 80)
    
    return df


if __name__ == "__main__":
    # This would be called from main.py or notebooks
    pass
