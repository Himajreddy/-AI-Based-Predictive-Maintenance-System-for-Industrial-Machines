# 🔧 AI-Based Predictive Maintenance System

> *Predict machine failure before it happens — saving downtime, money, and lives.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Business Impact](#business-impact)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Models Used](#models-used)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Deployment](#deployment)
11. [Future Improvements](#future-improvements)

---

## 1. Problem Statement

Industrial machines fail unexpectedly, causing **unplanned downtime**, costly emergency repairs, and in severe cases, safety incidents. Traditional maintenance is either:

- **Reactive** — fix it after it breaks (expensive)
- **Scheduled** — replace parts on a fixed calendar (wasteful)

This project builds a **condition-based, AI-driven predictive maintenance system** that analyses real-time sensor data to:

1. **Predict whether a machine will fail** (binary classification)
2. **Estimate Remaining Useful Life** in hours (regression)

---

## 2. Business Impact

| Metric | Industry Benchmark |
|---|---|
| Unplanned downtime cost | $260,000/hour (manufacturing) |
| Reduction in maintenance cost with PdM | 10–25% |
| Increase in equipment uptime | 10–20% |
| ROI of predictive maintenance | 10× on average |

By accurately flagging at-risk machines hours or days in advance, maintenance teams can:
- Schedule targeted repairs during planned stops
- Reduce spare parts inventory
- Prevent catastrophic failures and safety incidents

---

## 3. Dataset

**Source:** [Kaggle — AI4I Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) / UCI Machine Learning Repository

**Alternative:** The project includes a synthetic dataset generator (`src/data_loader.py → generate_synthetic_dataset`) so you can run the full pipeline without a Kaggle account.

### Input Features

| Feature | Type | Description |
|---|---|---|
| temperature | float | Machine surface temperature (°C) |
| vibration | float | Vibration amplitude (mm/s) |
| pressure | float | Operating pressure (bar) |
| voltage | float | Supply voltage (V) |
| runtime_hours | float | Cumulative operating hours |
| humidity | float | Ambient humidity (%) — optional |
| rotational_speed | float | Shaft RPM — optional |
| torque | float | Motor torque (Nm) — optional |
| wear_level | float | Component wear 0–1 — optional |

### Targets

| Target | Type | Description |
|---|---|---|
| `failure` | int {0, 1} | Binary failure label |
| `rul` | float | Remaining Useful Life (hours) |

---

## 4. Methodology

```
Raw CSV → Data Loading & Validation
        → Exploratory Data Analysis
        → Feature Engineering
        → Preprocessing (Scaling, SMOTE)
        → Model Training (CV + GridSearch)
        → Evaluation & Explainability
        → Streamlit Dashboard
```

### Key techniques

- **SMOTE** to handle class imbalance in failure labels
- **GridSearchCV** with stratified cross-validation for hyperparameter tuning
- **CalibratedClassifierCV** for reliable failure probability estimates
- **Threshold optimisation** — maximises F1 score for maintenance alerting
- **SHAP** for global and local model explanations

---

## 5. Models Used

### Classification (Failure Prediction)

| Model | Notes |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | Ensemble, robust to noise |
| XGBoost | Gradient boosting, top performer |
| LightGBM | Fast gradient boosting |
| SVM (RBF kernel) | Good for non-linear boundaries |

### Regression (RUL Estimation)

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Random Forest Regressor | Robust ensemble |
| XGBoost Regressor | Top performer |
| LightGBM Regressor | Fast & accurate |

---

## 6. Results

> Results below are from the synthetic dataset (5,000 samples). Your results on real data may differ.

### Classification Metrics (Best Model: XGBoost)

| Metric | Score |
|---|---|
| Accuracy | ~0.94 |
| Precision | ~0.91 |
| Recall | ~0.89 |
| F1 Score | ~0.90 |
| ROC-AUC | ~0.97 |

### Regression Metrics (Best Model: XGBoost Regressor)

| Metric | Score |
|---|---|
| MAE | ~180 hours |
| RMSE | ~310 hours |
| R² | ~0.92 |

---

## 7. Project Structure

```
predictive-maintenance/
│
├── data/
│   ├── raw/                    # Original CSV files
│   └── processed/              # Cleaned & feature-engineered data
│
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
│
├── src/
│   ├── data_loader.py          # Load, validate, clean data
│   ├── features.py             # Feature engineering
│   ├── preprocessing.py        # Scaling, encoding, SMOTE, splits
│   ├── train.py                # Model training + hyperparameter tuning
│   ├── evaluate.py             # Metrics, confusion matrix, ROC curves
│   ├── explain.py              # SHAP explainability
│   └── utils.py                # Shared utilities
│
├── models/                     # Saved models (gitignored)
│   ├── classifier.pkl
│   ├── regressor.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_columns.pkl
│
├── reports/
│   └── figures/                # Auto-generated plots
│
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 8. Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/predictive-maintenance.git
cd predictive-maintenance

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download dataset
# Place your CSV at data/raw/sensor_data.csv
# OR let the pipeline auto-generate a synthetic dataset
```

---

## 9. Usage

### A) Train the full pipeline (one command)

```bash
python src/train.py
```

This will:
1. Generate/load raw data
2. Clean and engineer features
3. Preprocess with SMOTE
4. Train all classifiers and regressors
5. Select and tune the best models
6. Save all artefacts to `models/`

### B) Evaluate models

```bash
python src/evaluate.py
```

### C) Generate SHAP explanations

```bash
python src/explain.py
```

### D) Run the interactive dashboard

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### E) Run EDA notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

---

## 10. Deployment

### Local Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python src/train.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t predictive-maintenance .
docker run -p 8501:8501 predictive-maintenance
```

### Cloud (Streamlit Community Cloud)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the entry point
4. Add `requirements.txt` — Streamlit handles the rest

---

## 11. Future Improvements

- **LSTM / Transformer** models for time-series RUL prediction
- **Anomaly detection** (Isolation Forest, Autoencoder) for unsupervised failure signals
- **MLflow** integration for experiment tracking
- **REST API** with FastAPI for real-time scoring
- **Kafka / MQTT** integration for live sensor stream ingestion
- **Federated learning** for multi-factory deployment with privacy preservation
- **Drift detection** (Evidently AI) for production model monitoring
- **AutoML** search over a wider model space

---

## License

MIT License — see `LICENSE` for details.

---

*Built as a Master's in AI portfolio project demonstrating end-to-end ML engineering capabilities.*
