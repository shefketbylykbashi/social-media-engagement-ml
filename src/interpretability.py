"""
Model interpretability and feature importance analysis module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import shap
from sklearn.inspection import permutation_importance

from config import RESULTS_FIGURES, RESULTS_TABLES, TOP_N_FEATURES, USE_SHAP, USE_PERMUTATION_IMPORTANCE
from utils import get_logger, save_figure, save_table

logger = get_logger(__name__)


def calculate_feature_importance(model, model_name):
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: Trained model
        model_name: Name of the model
        
    Returns:
        Dictionary with feature importance or None if not available
    """
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    elif hasattr(model, 'coef_'):
        return np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
    else:
        logger.warning(f"Model {model_name} does not have feature importance attribute")
        return None


def calculate_permutation_importance(model, X_val, y_val, feature_names):
    """
    Calculate permutation importance.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        feature_names: List of feature names
        
    Returns:
        DataFrame with permutation importance
    """
    if USE_PERMUTATION_IMPORTANCE:
        logger.info("Calculating permutation importance...")
        
        # Determine scoring function
        if hasattr(model, 'predict_proba'):
            scoring = 'f1_weighted'  # Classification
        else:
            scoring = 'r2'  # Regression
        
        perm_importance = permutation_importance(model, X_val, y_val, n_repeats=10, 
                                                 random_state=42, n_jobs=-1, scoring=scoring)
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    return None


def plot_feature_importance(importance_array, feature_names, target_name, model_name, n_features=TOP_N_FEATURES):
    """
    Plot feature importance.
    
    Args:
        importance_array: Array of importance values
        feature_names: List of feature names
        target_name: Name of the target variable
        model_name: Name of the model
        n_features: Number of top features to show
        
    Returns:
        matplotlib figure
    """
    # Sort features by importance
    indices = np.argsort(importance_array)[-n_features:][::-1]
    top_features = [feature_names[i] for i in indices]
    top_importances = importance_array[indices]
    
    fig, ax = plt.subplots(figsize=(10, max(6, n_features * 0.3)))
    
    bars = ax.barh(range(len(top_features)), top_importances, color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Feature Importance: {target_name} ({model_name})', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_importances)):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='x')
    
    return fig, ax


def create_shap_explanation(model, X_val, feature_names, target_name, model_name):
    """
    Create SHAP explanation for model predictions.
    
    Args:
        model: Trained model
        X_val: Validation features
        feature_names: List of feature names
        target_name: Name of the target variable
        model_name: Name of the model
        
    Returns:
        Dictionary with SHAP results
    """
    if not USE_SHAP:
        return None
    
    logger.info(f"Calculating SHAP values for {model_name}...")
    
    try:
        # Create explainer based on model type
        if hasattr(model, 'predict_proba'):
            # Classification model
            explainer = shap.TreeExplainer(model)
        else:
            # Regression model
            explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_val)
        
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'X_val': X_val,
            'feature_names': feature_names
        }
    
    except Exception as e:
        logger.warning(f"Error calculating SHAP values: {str(e)}")
        return None


def plot_shap_summary(shap_results, target_name, model_name, plot_type='bar'):
    """
    Plot SHAP summary plot.
    
    Args:
        shap_results: Dictionary with SHAP results
        target_name: Name of the target variable
        model_name: Name of the model
        plot_type: Type of plot ('bar', 'beeswarm', or 'violin')
        
    Returns:
        matplotlib figure
    """
    if shap_results is None:
        return None, None
    
    shap_values = shap_results['shap_values']
    X_val = shap_results['X_val']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    try:
        if plot_type == 'bar':
            shap.summary_plot(shap_values, X_val, plot_type='bar', show=False)
        else:
            shap.summary_plot(shap_values, X_val, plot_type=plot_type, show=False)
        
        ax.set_title(f'SHAP Summary: {target_name} ({model_name})', fontsize=14, fontweight='bold')
        
    except Exception as e:
        logger.warning(f"Error plotting SHAP summary: {str(e)}")
        return None, None
    
    return fig, ax


def create_feature_importance_summary(importance_results, output_file):
    """
    Create a summary table of feature importances across models.
    
    Args:
        importance_results: Dictionary with importance results
        output_file: Path to save the CSV file
    """
    rows = []
    
    for target, models_results in importance_results.items():
        for model_name, importance_data in models_results.items():
            if importance_data is not None:
                if isinstance(importance_data, pd.DataFrame):
                    # Permutation importance
                    for idx, row in importance_data.iterrows():
                        rows.append({
                            'Target': target,
                            'Model': model_name,
                            'Feature': row['Feature'],
                            'Importance': row['Importance'],
                            'Std': row.get('Std', None),
                            'Type': 'Permutation'
                        })
                else:
                    # Feature importance - just get top N
                    importance_array = importance_data
                    # We need feature names which aren't passed here
                    # This is handled separately in the analysis
                    pass
    
    if rows:
        summary_df = pd.DataFrame(rows)
        summary_df = summary_df.sort_values(['Target', 'Model', 'Importance'], ascending=[True, True, False])
        summary_df.to_csv(output_file, index=False)
        logger.info(f"Feature importance summary saved to: {output_file}")


def analyze_model_interpretability(trained_models, X_val, y_val, feature_names, 
                                   targets, output_prefix='model'):
    """
    Comprehensive interpretability analysis for all models.
    
    Args:
        trained_models: Dictionary with trained models
        X_val: Validation features
        y_val: Validation target
        feature_names: List of feature names
        targets: Dictionary with target information
        output_prefix: Prefix for output files
    """
    logger.info("=" * 80)
    logger.info("STARTING INTERPRETABILITY ANALYSIS")
    logger.info("=" * 80)
    
    importance_results = {}
    
    for target_name, model_dict in trained_models.items():
        logger.info(f"\nAnalyzing {target_name}...")
        importance_results[target_name] = {}
        
        for model_name, model in model_dict.items():
            logger.info(f"  - {model_name}")
            
            # Feature importance
            importance_array = calculate_feature_importance(model, model_name)
            
            if importance_array is not None:
                # Plot feature importance
                fig, ax = plot_feature_importance(
                    importance_array, feature_names, target_name, model_name
                )
                filename = f"feature_importance_{target_name}_{model_name}.png"
                save_figure(fig, filename)
                
                importance_results[target_name][model_name] = importance_array
            
            # Permutation importance
            perm_importance_df = calculate_permutation_importance(
                model, X_val, y_val, feature_names
            )
            
            if perm_importance_df is not None:
                # Save permutation importance
                filename = f"permutation_importance_{target_name}_{model_name}.csv"
                save_table(perm_importance_df, filename)
    
    logger.info("=" * 80)
    logger.info("INTERPRETABILITY ANALYSIS COMPLETED")
    logger.info("=" * 80)
    
    return importance_results


if __name__ == "__main__":
    # This would be called from main.py
    pass
