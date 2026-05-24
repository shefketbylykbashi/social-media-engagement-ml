"""
Prediction system for new unpublished social media posts.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime

from config import MODELS_REG, MODELS_CLF, DATASET_PROCESSED, REGRESSION_TARGETS
from preprocessing import (
    create_text_features, create_sentiment_features, create_tfidf_features,
    encode_categorical_features, remove_leakage_features
)
from utils import get_logger

logger = get_logger(__name__)


class PostPerformancePredictor:
    """
    Prediction system for social media post performance before publication.
    """
    
    def __init__(self):
        """Initialize the predictor by loading trained models."""
        self.regression_models = {}
        self.classification_models = {}
        self.tfidf_vectorizer = None
        self.categorical_features = [
            'platform', 'post_type', 'weekday_type', 'time_periods',
            'age_group', 'audience_gender', 'audience_location',
            'audience_content', 'audience_interests'
        ]
        self.feature_columns = None
        
        self._load_models()
    
    def _load_models(self):
        """Load trained regression and classification models."""
        logger.info("Loading trained models...")
        
        # Load regression models
        for target in REGRESSION_TARGETS:
            model_path = MODELS_REG / f"{target}_XGBoost.pkl"  # Use best model
            if model_path.exists():
                self.regression_models[target] = joblib.load(model_path)
                logger.info(f"Loaded {target} regression model")
        
        # Load classification models
        for target in ['viral', 'high_engagement']:
            model_path = MODELS_CLF / f"{target}_XGBoost.pkl"
            if model_path.exists():
                self.classification_models[target] = joblib.load(model_path)
                logger.info(f"Loaded {target} classification model")
        
        logger.info("Models loaded successfully")
    
    def preprocess_new_post(self, post_data):
        """
        Preprocess a new post for prediction.
        
        Args:
            post_data: Dictionary with post information
            
        Returns:
            DataFrame with preprocessed post
        """
        logger.info("Preprocessing new post...")
        
        # Convert to DataFrame
        df = pd.DataFrame([post_data])
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Create text features
        df = create_text_features(df)
        
        # Create sentiment features
        df = create_sentiment_features(df)
        
        # Create TF-IDF features (if original data available)
        if 'post_content' in df.columns:
            tfidf_df, _ = create_tfidf_features(df)
            df = pd.concat([df, tfidf_df], axis=1)
            df = df.drop(columns=['post_content'])
        
        # Encode categorical features
        # Note: This should match the training data encoding
        df = encode_categorical_features(df, self.categorical_features)
        
        # Remove unnecessary columns
        df = remove_leakage_features(df)
        
        logger.info("Post preprocessing completed")
        
        return df
    
    def predict(self, post_data):
        """
        Predict performance metrics for a new post.
        
        Args:
            post_data: Dictionary with post information
                Example:
                {
                    'platform': 'Instagram',
                    'post_type': 'Image',
                    'post_content': 'Your post text here',
                    'post_timestamp': '2026-05-24 18:30:00',
                    'weekday_type': 'Weekend',
                    'time_periods': 'Evening',
                    'audience_age': 25,
                    'age_group': 'Young Adults',
                    'audience_gender': 'Female',
                    'audience_location': 'Kosovo',
                    'audience_content': 'Fashion',
                    'audience_interests': 'Shopping'
                }
        
        Returns:
            Dictionary with predictions
        """
        logger.info("Making predictions for new post...")
        
        # Preprocess
        df_processed = self.preprocess_new_post(post_data)
        
        predictions = {
            'input': post_data,
            'regression_predictions': {},
            'classification_predictions': {},
            'summary': {}
        }
        
        # Regression predictions
        for target, model in self.regression_models.items():
            try:
                pred_value = model.predict(df_processed)[0]
                predictions['regression_predictions'][target] = float(pred_value)
                logger.info(f"Predicted {target}: {pred_value:.2f}")
            except Exception as e:
                logger.warning(f"Error predicting {target}: {str(e)}")
                predictions['regression_predictions'][target] = None
        
        # Classification predictions
        for target, model in self.classification_models.items():
            try:
                pred_class = model.predict(df_processed)[0]
                pred_proba = model.predict_proba(df_processed)[0]
                predictions['classification_predictions'][target] = {
                    'class': int(pred_class),
                    'probability': float(pred_proba[1]) if len(pred_proba) > 1 else float(pred_proba[0])
                }
                logger.info(f"Predicted {target}: {pred_class} (confidence: {pred_proba[1]:.2%})")
            except Exception as e:
                logger.warning(f"Error predicting {target}: {str(e)}")
                predictions['classification_predictions'][target] = None
        
        # Generate summary
        predictions['summary'] = self._generate_summary(predictions)
        
        return predictions
    
    def _generate_summary(self, predictions):
        """
        Generate a human-readable summary of predictions.
        
        Args:
            predictions: Dictionary with predictions
            
        Returns:
            Summary string
        """
        summary = []
        
        # Engagement rate prediction
        engagement_rate = predictions['regression_predictions'].get('engagement_rate')
        if engagement_rate is not None:
            if engagement_rate > 0.05:
                engagement_level = "HIGH"
            elif engagement_rate > 0.02:
                engagement_level = "MEDIUM"
            else:
                engagement_level = "LOW"
            summary.append(f"Expected engagement rate: {engagement_rate:.4f} ({engagement_level})")
        
        # Viral prediction
        viral_pred = predictions['classification_predictions'].get('viral')
        if viral_pred is not None:
            viral_prob = viral_pred['probability']
            summary.append(f"Viral probability: {viral_prob:.2%}")
        
        # Likes prediction
        likes_pred = predictions['regression_predictions'].get('likes')
        if likes_pred is not None:
            summary.append(f"Expected likes: {int(likes_pred)}")
        
        # Sentiment (if available from input)
        input_sentiment = predictions['input'].get('pre_post_sentiment_label')
        if input_sentiment:
            summary.append(f"Post content sentiment: {input_sentiment}")
        
        # Platform impact
        platform = predictions['input'].get('platform')
        if platform:
            summary.append(f"Platform: {platform}")
        
        # Recommendation
        summary.append("\n--- RECOMMENDATION ---")
        if engagement_level == "HIGH" and viral_prob and viral_prob > 0.5:
            summary.append("This post is expected to perform very well. Good time to publish!")
        elif engagement_level == "MEDIUM":
            summary.append("This post is expected to achieve moderate engagement.")
        else:
            summary.append("This post might benefit from revision before publishing.")
        
        return "\n".join(summary)
    
    def predict_batch(self, posts_list):
        """
        Predict performance for multiple posts.
        
        Args:
            posts_list: List of post dictionaries
            
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Making predictions for {len(posts_list)} posts...")
        
        predictions_list = []
        for i, post_data in enumerate(posts_list):
            logger.info(f"Predicting post {i+1}/{len(posts_list)}")
            pred = self.predict(post_data)
            predictions_list.append(pred)
        
        return predictions_list


def example_usage():
    """
    Example usage of the PostPerformancePredictor.
    """
    predictor = PostPerformancePredictor()
    
    # Example new post
    new_post = {
        "platform": "Instagram",
        "post_type": "Image",
        "post_content": "New summer collection is finally here! Check it out now. 🌞 #Fashion #Summer",
        "post_timestamp": "2026-05-24 18:30:00",
        "weekday_type": "Weekend",
        "time_periods": "Evening",
        "audience_age": 25,
        "age_group": "Young Adults",
        "audience_gender": "Female",
        "audience_location": "Kosovo",
        "audience_content": "Fashion",
        "audience_interests": "Shopping"
    }
    
    # Make predictions
    predictions = predictor.predict(new_post)
    
    # Print results
    print("\n" + "=" * 80)
    print("POST PERFORMANCE PREDICTION")
    print("=" * 80)
    print(predictions['summary'])
    print("\nRegression Predictions:")
    for target, value in predictions['regression_predictions'].items():
        if value is not None:
            print(f"  {target}: {value:.4f}")
    print("\nClassification Predictions:")
    for target, value in predictions['classification_predictions'].items():
        if value is not None:
            print(f"  {target}: Class {value['class']} (Probability: {value['probability']:.2%})")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
