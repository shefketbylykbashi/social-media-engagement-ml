import os
import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

from src import config

try:
    from src import data_loader, preprocessing
except Exception:
    data_loader = None
    preprocessing = None

from src.train_fast import train_models_for_targets


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("main_fast")


def load_or_create_cleaned(path: Path):
    if path.exists():
        logger.info(f"Loading processed dataset from {path}")
        return pd.read_csv(path)

    if data_loader and hasattr(data_loader, "load_raw_dataset"):
        logger.info("Processed dataset not found — running data_loader.load_raw_dataset()")
        df = data_loader.load_raw_dataset()
        cleaned_path = path.parent
        cleaned_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df

    raise FileNotFoundError(f"Processed dataset not found at {path} and data_loader unavailable")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fast pipeline orchestrator for SentimentAnalysis")
    parser.add_argument("--processed", default="data/processed/cleaned_dataset.csv", help="Path to cleaned dataset CSV")
    parser.add_argument("--experiment", choices=["A", "B"], default="A", help="Experiment variant to run")
    parser.add_argument("--skip-training", action="store_true", help="Only run preprocessing and exit")
    parser.add_argument("--optuna-trials", type=int, default=10, help="Reduced Optuna trials for speed (if used)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs for training (-1 = all cores)")
    args = parser.parse_args(argv)

    processed_path = Path(args.processed)

    # Load or create cleaned dataset
    df = load_or_create_cleaned(processed_path)

    # Preprocessing (cached)
    if preprocessing and hasattr(preprocessing, "preprocess_dataset"):
        logger.info("Running preprocessing (cached)")
        # many preprocessing implementations accept df and return X,y dicts; attempt to call defensively
        try:
            prep = preprocessing.preprocess_dataset(df.copy(), experiment=args.experiment, cache_dir=".cache")
        except TypeError:
            # fallback to simple preprocess_dataset(df)
            prep = preprocessing.preprocess_dataset(df.copy())
    else:
        raise RuntimeError("preprocessing.preprocess_dataset not available in src/preprocessing.py")

    # Expect prep to be a dict with keys: X_reg, y_reg (dict of targets), X_clf, y_clf (dict)
    X_reg = prep.get("X_reg") if isinstance(prep, dict) else prep
    y_reg = prep.get("y_reg") if isinstance(prep, dict) else None
    X_clf = prep.get("X_clf") if isinstance(prep, dict) else None
    y_clf = prep.get("y_clf") if isinstance(prep, dict) else None

    if args.skip_training:
        logger.info("Skipping training as requested. Preprocessing complete.")
        return

    # Train models (fast flows)
    logger.info("Starting fast model training")
    train_models_for_targets(X_reg=X_reg, y_reg=y_reg, X_clf=X_clf, y_clf=y_clf,
                             optuna_trials=args.optuna_trials, n_jobs=args.n_jobs,
                             output_dir=Path("models/fast"))

    logger.info("Fast pipeline complete. Models saved to models/fast/")


if __name__ == "__main__":
    main()
