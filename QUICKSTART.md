# QUICK START GUIDE

## Sentiment Analysis and Engagement Prediction Project

This guide will help you get the project up and running in 5 minutes.

---

## Prerequisites

- Python 3.8 or higher
- pip package manager
- ~2GB disk space for data and models
- ~4GB RAM recommended for model training

---

## Installation (Windows/Mac/Linux)

### Step 1: Navigate to Project Directory

```bash
cd c:\Users\shefk\source\repos\Python\SentimentAnalysis
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Expected Output:**
```
[nltk_data] Downloading package punkt to C:\Users\...\nltk_data...
[nltk_data] Downloading package stopwords to C:\Users\...\nltk_data...
```

---

## Running the Project

### Option 1: Run Complete Pipeline (Recommended)

This will execute the entire machine learning workflow:

```bash
python main.py
```

**What happens:**
1. Loads and cleans the dataset
2. Performs exploratory data analysis
3. Engineers advanced features
4. Creates train/validation/test splits
5. Trains regression models
6. Trains classification models
7. Performs hyperparameter optimization
8. Generates interpretability analysis
9. Creates methodology report
10. Saves all results

**Expected Runtime:** 30-60 minutes (first time)
**Output:** 
- Cleaned datasets in `data/processed/`
- Trained models in `models/`
- Evaluation metrics in `results/`
- Visualizations in `results/figures/`
- Summary tables in `results/tables/`
- Methodology report in `reports/`

### Option 2: Run Exploratory Data Analysis

To explore and understand the data:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

This notebook includes:
- Data loading and cleaning
- Distribution analysis
- Platform performance comparison
- Correlation analysis
- Target variable creation

---

## Making Predictions

### Predict Performance for a New Post

Once models are trained, predict engagement for a new unpublished post:

```python
# In Python script or Jupyter notebook:
from src.predict_new_post import PostPerformancePredictor

# Initialize predictor
predictor = PostPerformancePredictor()

# Define new post
new_post = {
    "platform": "Instagram",
    "post_type": "Image",
    "post_content": "Check out our new collection! #Fashion #Summer",
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

# Make prediction
predictions = predictor.predict(new_post)

# Display results
print(predictions['summary'])
```

**Output includes:**
- Predicted likes, comments, shares
- Predicted impressions and reach
- Predicted engagement rate
- Viral probability
- Sentiment analysis
- Actionable recommendations

---

## Project Structure

```
social_media_reaction_prediction/
├── data/
│   ├── raw/                          # Original Excel file
│   └── processed/                    # Cleaned datasets
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb  (in development)
│   ├── 03_model_training.ipynb       (in development)
│   └── 04_model_interpretability.ipynb (in development)
├── src/
│   ├── config.py                     # All configuration parameters
│   ├── data_loader.py                # Data loading and cleaning
│   ├── preprocessing.py              # Feature engineering
│   ├── train_regression.py           # Regression models
│   ├── train_classification.py       # Classification models
│   ├── evaluate.py                   # Metrics and evaluation
│   ├── interpretability.py           # SHAP and feature importance
│   ├── predict_new_post.py           # Prediction system
│   └── utils.py                      # Utility functions
├── models/
│   ├── regression/                   # Saved regression models
│   └── classification/               # Saved classification models
├── results/
│   ├── figures/                      # Visualizations (PNG)
│   ├── tables/                       # Summary tables (CSV)
│   ├── metrics_regression.csv
│   ├── metrics_classification.csv
│   └── feature_importance.csv
├── reports/
│   └── methodology_summary.md        # Scientific methodology
├── main.py                           # Main orchestrator
├── requirements.txt                  # Dependencies
└── README.md                         # Full documentation
```

---

## Key Features

### ✅ Data Leakage Prevention
- Only uses pre-publication information
- Removes post-engagement metrics from features
- Validates feature engineering

### ✅ Advanced Feature Engineering
- Text analysis (length, sentiment, hashtags, etc.)
- Temporal features (hour, day, weekday)
- TF-IDF features from post content
- Categorical encoding

### ✅ Multiple Models
**Regression:** Linear, Ridge, Random Forest, XGBoost, LightGBM, CatBoost, SVR
**Classification:** Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, CatBoost

### ✅ Hyperparameter Optimization
- Optuna-based Bayesian optimization
- 50 trials per model
- 5-fold cross-validation

### ✅ Interpretability
- Feature importance analysis
- Permutation importance
- SHAP values and plots
- Publication-ready visualizations

### ✅ Experimental Comparison
- Experiment A: Without campaign/influencer IDs
- Experiment B: With campaign/influencer IDs
- Performance comparison

---

## Results and Outputs

After running `python main.py`, you'll have:

### Datasets
- `data/processed/cleaned_dataset.csv` - Clean data with targets
- `data/processed/train_dataset.csv` - Training set
- `data/processed/val_dataset.csv` - Validation set
- `data/processed/test_dataset.csv` - Test set

### Models
- `models/regression/` - 7 regression models × 6 targets = 42 models
- `models/classification/` - 6 classification models × 3 targets = 18 models

### Metrics
- `results/metrics_regression.csv` - MAE, RMSE, R², MAPE
- `results/metrics_classification.csv` - Accuracy, F1, ROC-AUC
- `results/feature_importance.csv` - Top features per target

### Visualizations
- Distribution plots
- Correlation heatmaps
- Platform performance comparisons
- Feature importance plots
- SHAP analysis plots

### Documentation
- `reports/methodology_summary.md` - Academic methodology
- README.md - Complete documentation

---

## Configuration

All parameters can be modified in `src/config.py`:

### Data Paths
```python
DATASET_RAW = "data/raw/social_media_engagement_data.xlsx"
DATASET_CLEANED = "data/processed/cleaned_dataset.csv"
```

### Train/Test Split
```python
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
```

### Feature Engineering
```python
TFIDF_MAX_FEATURES = 500
SENTIMENT_MODEL = "vader"  # or "textblob"
```

### Hyperparameter Tuning
```python
USE_HYPERPARAMETER_TUNING = True
OPTUNA_N_TRIALS = 50
CV_FOLDS = 5
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'xgboost'"

**Solution:**
```bash
pip install --upgrade xgboost lightgbm catboost optuna shap
```

### Issue: "NLTK data not found"

**Solution:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Issue: Memory error during training

**Solution:**
- Reduce number of TF-IDF features in config.py: `TFIDF_MAX_FEATURES = 300`
- Reduce Optuna trials: `OPTUNA_N_TRIALS = 25`
- Train one target at a time instead of all targets

### Issue: Excel file not found

**Solution:**
Ensure `social_media_engagement_data.xlsx` is in `data/raw/` directory:
```
data/
└── raw/
    └── social_media_engagement_data.xlsx
```

---

## Performance Tips

1. **First Run**: Full pipeline takes 30-60 minutes
2. **GPU Support**: Install GPU versions of XGBoost/LightGBM for faster training:
   ```bash
   pip install xgboost[gpu] lightgbm[gpu]
   ```
3. **Parallel Processing**: Models use `n_jobs=-1` (all cores)
4. **Reduce Hyperparameter Trials**: Set `OPTUNA_N_TRIALS = 25` for faster optimization

---

## Next Steps

1. **Review Results**: Check `results/` for metrics and visualizations
2. **Read Methodology**: See `reports/methodology_summary.md` for academic details
3. **Make Predictions**: Use `src/predict_new_post.py` for new posts
4. **Modify Configuration**: Customize `src/config.py` for your needs
5. **Extend Analysis**: Add more notebooks in `notebooks/` directory

---

## Project Timeline

| Step | Time | Output |
|------|------|--------|
| Data Loading & Cleaning | 1-2 min | Cleaned dataset |
| Feature Engineering | 5-10 min | Engineered features |
| Model Training | 15-30 min | Trained models |
| Hyperparameter Optimization | 10-20 min | Optimized models |
| Evaluation & Interpretability | 5-10 min | Metrics & visualizations |
| **Total** | **30-60 min** | **Complete ML pipeline** |

---

## Support and Documentation

- **Full README**: See `README.md` for comprehensive documentation
- **Code Comments**: All source files have detailed comments
- **Methodology**: See `reports/methodology_summary.md` for academic approach
- **EDA Notebook**: See `notebooks/01_exploratory_data_analysis.ipynb` for data exploration

---

## Citation

If you use this project, please cite:

```
@software{sentiment_engagement_2026,
  title={Sentiment Analysis and Engagement Prediction in Social Media Data},
  year={2026},
  note={Research Project}
}
```

---

## License

MIT License - Feel free to use and modify for your research.

---

**Ready to start?**

```bash
python main.py
```

Happy analyzing! 🚀
