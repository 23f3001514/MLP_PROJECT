# MLP_PROJECT

<div align="center">

# 🎬 Cinema Audience Forecasting

### Predicting daily cinema audience counts using machine learning

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Best Model:** Gradient Boosting &nbsp;|&nbsp; **Validation RMSE:** `18.03` &nbsp;|&nbsp; **Validation R²:** `0.656`

</div>

---

## 📌 Overview

This project forecasts the **number of people who will attend a cinema show** on a given day, for a given theater, using historical booking and visit data. It's framed as a regression problem: given a theater and a future date, predict `audience_count`.

The pipeline combines **time-series style feature engineering** (lags, rolling averages, theater statistics) with **classical ML regressors**, and finishes with a **blended, bounded prediction strategy** designed to produce realistic, submission-ready forecasts.

## 🗂️ Repository Structure

```
MLP_PROJECT/
├── Models/         # Saved/serialized model artifacts
├── Notebook/        # Main analysis & modeling notebook
├── scripts/         # Standalone Python scripts
├── src/              # Source modules
└── README.md
```

## 📊 Dataset

The project uses **7 datasets** describing cinema visits, theaters, bookings, and show dates:

| Dataset | Description |
|---|---|
| `booknow_visits` | Audience count per show, per theater |
| `booknow_theaters` | Theater metadata (type, area, etc.) |
| `date_info` | Calendar information for show dates |
| `booknow_booking` | Individual booking / ticket records |
| `cinePOS_booking` | Point-of-sale booking records |
| `cinePOS_theaters` | POS theater metadata |
| `sample_submission` | Submission template |

## 🔧 Pipeline

<table>
<tr><td><b>1. Data Loading</b></td><td>Load and merge all 7 datasets; parse datetime fields; sanity-check shapes and null values.</td></tr>
<tr><td><b>2. EDA</b></td><td>6 visualizations covering audience distribution, theater types, day-of-week patterns, and audience trends over time.</td></tr>
<tr><td><b>3. Feature Engineering</b></td><td>16 engineered features — date parts, per-theater statistics, lag features (1/3/7/14 days), rolling means & std (3/7/14 days), booking aggregates, and label-encoded categoricals.</td></tr>
<tr><td><b>4. Train/Val Split</b></td><td>80/20 split with missing and infinite values handled before training.</td></tr>
<tr><td><b>5. Model Training</b></td><td>4 candidate regressors trained and evaluated: Linear Regression, Ridge Regression, Random Forest, Gradient Boosting.</td></tr>
<tr><td><b>6. Hyperparameter Tuning</b></td><td>GridSearchCV (3-fold) tuning of the Gradient Boosting model on a data subset for faster search.</td></tr>
<tr><td><b>7. Model Comparison</b></td><td>RMSE, R², actual-vs-predicted, and feature-importance plots across all models.</td></tr>
<tr><td><b>8. Prediction Blending</b></td><td>Final forecasts blend model output (70%) with theater historical average (30%), apply a day-of-week adjustment, and clip to the 5th–95th percentile range for realism.</td></tr>
<tr><td><b>9. Submission</b></td><td>Final predictions written to <code>submission.csv</code>.</td></tr>
</table>

## 🧠 Models & Results

| Model | Notes |
|---|---|
| Linear Regression | Baseline linear model |
| Ridge Regression | Regularized linear model (α = 1.0) |
| Random Forest | 150 estimators, max depth 20 |
| **Gradient Boosting** ⭐ | 150 estimators, max depth 6 — **best performer**, further tuned via GridSearchCV |

**Best model:** Gradient Boosting Regressor

| Metric | Score |
|---|---|
| Validation RMSE | **18.0298** |
| Validation R² | **0.6555** |

## ✨ Key Features Engineered

- **Date features:** day of week, month, quarter, week of year, weekend flag
- **Theater statistics:** mean, median, std, min, max, count of historical audience
- **Lag features:** audience count 1, 3, 7, and 14 days prior
- **Rolling statistics:** 3/7/14-day rolling mean, 7/14-day rolling std
- **Booking features:** total tickets, average tickets, booking count per theater/day
- **Encoded categoricals:** theater type and theater area

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/23f3001514/MLP_PROJECT.git
cd MLP_PROJECT

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Launch the notebook
jupyter notebook Notebook/
```

## 🛠️ Tech Stack

`Python` · `NumPy` · `Pandas` · `Matplotlib` · `Seaborn` · `scikit-learn`

## 📁 Outputs

- `eda_analysis.png` — exploratory data analysis visualizations
- `model_comparison.png` — model performance comparison plots
- `submission.csv` — final audience count predictions

## 📈 Future Improvements

- Incorporate holiday/event calendars as additional signal
- Experiment with gradient boosting libraries (XGBoost, LightGBM, CatBoost)
- Add cross-validation across time windows (walk-forward validation) instead of a single split
- Explore ensembling/stacking across the four trained models

## 📄 License

This project is available under the MIT License.

---

<div align="center">
Made with 🍿 for cinema audience forecasting
</div>
