"""
Classification model training and evaluation module.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from optuna.pruners import MedianPruner

from config import (
    MODELS_CLF, CV_FOLDS, SEED, USE_HYPERPARAMETER_TUNING,
    OPTUNA_N_TRIALS, CLASSIFICATION_METRICS, CATEGORICAL_FEATURES
)
from utils import get_logger
from evaluate import calculate_classification_metrics

logger = get_logger(__name__)


def create_classification_models():
    """
    Create base classification models without hyperparameter tuning.
    
    Returns:
        Dictionary of models
    """
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, class_weight='balanced'),
        'SVM': SVC(kernel='rbf', probability=True, random_state=SEED, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, verbosity=0, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, verbosity=-1, class_weight='balanced'),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=SEED, verbose=False)
    }
    
    return models


def optimize_hyperparameters(model_name, X_train, y_train):
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        model_name: Name of the model
        X_train: Training features
        y_train: Training target
        
    Returns:
        Best model
    """
    logger.info(f"Starting hyperparameter optimization for {model_name}...")
    
    def objective(trial):
        if model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'class_weight': 'balanced',
                'random_state': SEED,
                'n_jobs': -1
            }
            model = RandomForestClassifier(**params)
            
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': SEED,
                'n_jobs': -1,
                'verbosity': 0,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            model = xgb.XGBClassifier(**params)
            
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'class_weight': 'balanced',
                'random_state': SEED,
                'n_jobs': -1,
                'verbosity': -1
            }
            model = lgb.LGBMClassifier(**params)
            
        elif model_name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': SEED,
                'verbose': False
            }
            model = CatBoostClassifier(**params)
            
        elif model_name == 'SVM':
            params = {
                'C': trial.suggest_float('C', 0.1, 100, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly']),
                'class_weight': 'balanced',
                'probability': True,
                'random_state': SEED
            }
            model = SVC(**params)
            
        elif model_name == 'LogisticRegression':
            params = {
                'C': trial.suggest_float('C', 0.01, 10, log=True),
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': SEED,
                'n_jobs': -1
            }
            model = LogisticRegression(**params)
            
        else:
            return None
        
        # Cross-validation with stratified k-fold
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        scores = cross_validate(model, X_train, y_train, cv=skf, scoring='f1_weighted', n_jobs=-1)
        
        return scores['test_score'].mean()
    
    study = optuna.create_study(direction='maximize', pruner=MedianPruner())
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)
    
    logger.info(f"Best {model_name} F1-Weighted score: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Recreate best model
    best_params = study.best_params
    
    if model_name == 'RandomForest':
        best_params['class_weight'] = 'balanced'
        best_params['n_jobs'] = -1
        best_params['random_state'] = SEED
        best_model = RandomForestClassifier(**best_params)
    elif model_name == 'XGBoost':
        best_params['n_jobs'] = -1
        best_params['random_state'] = SEED
        best_params['verbosity'] = 0
        best_params['use_label_encoder'] = False
        best_params['eval_metric'] = 'logloss'
        best_model = xgb.XGBClassifier(**best_params)
    elif model_name == 'LightGBM':
        best_params['class_weight'] = 'balanced'
        best_params['n_jobs'] = -1
        best_params['random_state'] = SEED
        best_params['verbosity'] = -1
        best_model = lgb.LGBMClassifier(**best_params)
    elif model_name == 'CatBoost':
        best_params['random_state'] = SEED
        best_params['verbose'] = False
        best_model = CatBoostClassifier(**best_params)
    elif model_name == 'SVM':
        best_params['probability'] = True
        best_params['random_state'] = SEED
        best_model = SVC(**best_params)
    elif model_name == 'LogisticRegression':
        best_params['max_iter'] = 1000
        best_params['class_weight'] = 'balanced'
        best_params['random_state'] = SEED
        best_params['n_jobs'] = -1
        best_model = LogisticRegression(**best_params)
    else:
        return None
    
    return best_model


def train_classification_models(X_train, y_train, X_val, y_val, target_name, optimize=USE_HYPERPARAMETER_TUNING):
    """
    Train classification models and evaluate on validation set.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        target_name: Name of the target variable
        optimize: Whether to optimize hyperparameters
        
    Returns:
        Dictionary with trained models and results
    """
    logger.info("=" * 80)
    logger.info(f"Training classification models for target: {target_name}")
    logger.info("=" * 80)
    
    results = {}
    trained_models = {}
    
    # Get base models
    base_models = create_classification_models()
    
    for model_name, model in base_models.items():
        logger.info(f"\nTraining {model_name}...")
        
        try:
            # Optimize hyperparameters if enabled
            if optimize and model_name in ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'SVM', 'LogisticRegression']:
                model = optimize_hyperparameters(model_name, X_train, y_train)
                if model is None:
                    logger.warning(f"Skipping {model_name} due to optimization error")
                    continue
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred_val = model.predict(X_val)
            
            # Get probabilities for ROC-AUC
            y_proba_val = None
            if hasattr(model, 'predict_proba'):
                y_proba_val = model.predict_proba(X_val)
            
            # Calculate metrics
            metrics = calculate_classification_metrics(y_val, y_pred_val, y_proba_val)
            results[model_name] = metrics
            trained_models[model_name] = model
            
            logger.info(f"{model_name} - F1-Weighted: {metrics['F1_Weighted']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")
            
            # Save model
            model_path = MODELS_CLF / f"{target_name}_{model_name}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Model saved to: {model_path}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    logger.info("=" * 80)
    logger.info("Classification model training completed")
    logger.info("=" * 80)
    
    return results, trained_models


def select_best_classification_model(results):
    """
    Select the best classification model based on F1-Weighted score.
    
    Args:
        results: Dictionary with metrics for each model
        
    Returns:
        Name of the best model
    """
    best_model = max(results.items(), key=lambda x: x[1]['F1_Weighted'])
    return best_model[0]


if __name__ == "__main__":
    # This would be called from main.py
    pass
