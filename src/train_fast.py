import os
import math
import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from catboost import CatBoostRegressor, CatBoostClassifier, Pool
except Exception:
    CatBoostRegressor = None
    CatBoostClassifier = None
    Pool = None

logger = logging.getLogger("train_fast")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def _is_classification(y):
    if y is None:
        return False
    # treat as classification if integer labels with few unique values or dtype is object
    try:
        unique = np.unique(y)
        if y.dtype == object or len(unique) <= 20 and np.all(np.equal(np.mod(unique, 1), 0)):
            return True
        return False
    except Exception:
        return False


def _train_lightgbm(X_train, X_val, y_train, y_val, params=None):
    if lgb is None:
        raise RuntimeError("lightgbm is not installed")
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    params = params or {"objective": "regression", "metric": "rmse", "verbosity": -1, "n_jobs": -1}
    model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dval], early_stopping_rounds=50, verbose_eval=False)
    return model


def _train_catboost(X_train, X_val, y_train, y_val, is_classification=False):
    if CatBoostRegressor is None:
        raise RuntimeError("catboost is not installed")
    if is_classification:
        model = CatBoostClassifier(iterations=1000, learning_rate=0.05, eval_metric="AUC", early_stopping_rounds=50, verbose=False)
    else:
        model = CatBoostRegressor(iterations=1000, learning_rate=0.05, eval_metric="RMSE", early_stopping_rounds=50, verbose=False)
    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_val, y_val)
    model.fit(train_pool, eval_set=eval_pool)
    return model


def _save_model(model, out_path: Path, model_name: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_model") and str(out_path).endswith(".cbm"):
        model.save_model(str(out_path))
    else:
        joblib.dump(model, str(out_path))


def train_models_for_targets(X_reg: Optional[np.ndarray] = None,
                             y_reg: Optional[Dict[str, np.ndarray]] = None,
                             X_clf: Optional[np.ndarray] = None,
                             y_clf: Optional[Dict[str, np.ndarray]] = None,
                             optuna_trials: int = 10,
                             n_jobs: int = -1,
                             output_dir: Path = Path("models/fast")):
    """
    Train LightGBM and CatBoost quickly for provided targets.

    Parameters:
    - X_reg: feature matrix for regression (numpy or pandas)
    - y_reg: dict mapping target name -> array
    - X_clf: feature matrix for classification
    - y_clf: dict mapping target name -> array
    - optuna_trials: not used here (kept for API compatibility)
    - n_jobs: parallel jobs (currently not used for per-target parallelization)
    - output_dir: where to save models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Helper to train per target
    def _train_target(X, y, name, is_class=False):
        logger.info(f"Training target {name} (classification={is_class})")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

        # LightGBM
        try:
            lgb_model = _train_lightgbm(X_train, X_val, y_train, y_val, params={"objective": "binary" if is_class else "regression", "metric": "auc" if is_class else "rmse", "verbosity": -1, "n_jobs": -1})
            lgb_path = output_dir / f"{name}_lightgbm.pkl"
            _save_model(lgb_model, lgb_path, "lightgbm")
            logger.info(f"Saved LightGBM model to {lgb_path}")
        except Exception as e:
            logger.warning(f"LightGBM training failed for {name}: {e}")

        # CatBoost
        try:
            cb_model = _train_catboost(X_train, X_val, y_train, y_val, is_classification=is_class)
            cb_path = output_dir / f"{name}_catboost.cbm"
            _save_model(cb_model, cb_path, "catboost")
            logger.info(f"Saved CatBoost model to {cb_path}")
        except Exception as e:
            logger.warning(f"CatBoost training failed for {name}: {e}")

    # Regression targets
    if y_reg and X_reg is not None:
        for name, y in y_reg.items():
            _train_target(X_reg, y, name, is_class=False)

    # Classification targets
    if y_clf and X_clf is not None:
        for name, y in y_clf.items():
            _train_target(X_clf, y, name, is_class=True)

    logger.info("All fast trainings complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/cleaned_dataset.csv")
    args = parser.parse_args()

    # Minimal runner: loads preprocessed CSV and attempts to run using known column names
    if not Path(args.processed).exists():
        raise SystemExit("Processed dataset not found. Run data loading first.")

    import pandas as pd

    df = pd.read_csv(args.processed)
    # This runner is conservative: attempts to find target columns automatically
    reg_targets = [c for c in ["likes", "comments", "shares", "impressions", "reach", "engagement_rate"] if c in df.columns]
    clf_targets = [c for c in ["viral", "high_engagement", "sentiment"] if c in df.columns]

    feature_cols = [c for c in df.columns if c not in reg_targets + clf_targets + ["post_content", "post_id", "campaign_id", "influencer_id"]]
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0).values

    y_reg = {t: df[t].values for t in reg_targets}
    y_clf = {t: df[t].values for t in clf_targets}

    train_models_for_targets(X_reg=X, y_reg=y_reg, X_clf=X, y_clf=y_clf, output_dir=Path("models/fast"))
