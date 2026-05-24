"""
Model evaluation and metrics calculation module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from config import RESULTS_TABLES, RESULTS_FIGURES, REGRESSION_METRICS, CLASSIFICATION_METRICS
from utils import get_logger, save_table, save_figure

logger = get_logger(__name__)


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) if np.all(y_true != 0) else np.nan,
        'Median_Absolute_Error': median_absolute_error(y_true, y_pred)
    }
    
    return metrics


def calculate_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_proba: Predicted probabilities (for ROC-AUC)
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1_Macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'F1_Weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # ROC-AUC for binary classification
    if len(np.unique(y_true)) == 2 and y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['ROC_AUC'] = np.nan
    
    return metrics


def plot_actual_vs_predicted(y_true, y_pred, target_name, filename=None):
    """
    Plot actual vs predicted values for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        target_name: Name of the target variable
        filename: Optional filename to save
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'Actual vs Predicted: {target_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if filename:
        save_figure(fig, filename)
    
    return fig, ax


def plot_residuals(y_true, y_pred, target_name, filename=None):
    """
    Plot residuals for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        target_name: Name of the target variable
        filename: Optional filename to save
        
    Returns:
        matplotlib figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k')
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title(f'Residuals vs Predicted: {target_name}', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Residuals Distribution: {target_name}', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    if filename:
        save_figure(fig, filename)
    
    return fig, axes


def plot_confusion_matrix(y_true, y_pred, target_name, filename=None):
    """
    Plot confusion matrix for classification.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        target_name: Name of the target variable
        filename: Optional filename to save
        
    Returns:
        matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix: {target_name}', fontsize=14, fontweight='bold')
    
    if filename:
        save_figure(fig, filename)
    
    return fig, ax


def plot_roc_curve(y_true, y_proba, target_name, filename=None):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True values
        y_proba: Predicted probabilities
        target_name: Name of the target variable
        filename: Optional filename to save
        
    Returns:
        matplotlib figure
    """
    if len(np.unique(y_true)) != 2:
        logger.warning("ROC curve only supports binary classification.")
        return None, None
    
    # Get probabilities for positive class
    if len(y_proba.shape) == 2:
        y_proba_positive = y_proba[:, 1]
    else:
        y_proba_positive = y_proba
    
    fpr, tpr, _ = roc_curve(y_true, y_proba_positive)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve: {target_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if filename:
        save_figure(fig, filename)
    
    return fig, ax


def create_metrics_summary(results_dict, output_file):
    """
    Create a summary table of all metrics across models and targets.
    
    Args:
        results_dict: Dictionary with results
        output_file: Path to save the CSV file
    """
    rows = []
    
    for target, models_results in results_dict.items():
        for model_name, metrics in models_results.items():
            row = {'Target': target, 'Model': model_name}
            row.update(metrics)
            rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(['Target', 'R2' if 'R2' in summary_df.columns else 'F1_Weighted'], ascending=False)
    
    save_table(summary_df, output_file)
    
    return summary_df


if __name__ == "__main__":
    # This would be called from training scripts
    pass
