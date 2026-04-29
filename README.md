# Predictive Maintenance System for Industrial Equipment

## Project Summary

Industrial equipment rarely fails without warning. In most cases, machines show measurable signs of degradation long before breakdown. The challenge is identifying those patterns early enough to act.

This project builds a machine learning-based predictive maintenance system designed to estimate equipment health from sensor data and support maintenance decisions before failure occurs.

The system performs two separate but connected tasks:

* **Failure Risk Detection** → predicts whether a machine is approaching failure
* **Remaining Useful Life Estimation (RUL)** → estimates how long the machine can continue operating before maintenance becomes critical

The project uses a real benchmark dataset and combines preprocessing, feature engineering, model benchmarking, explainability, and deployment into a complete production-style workflow.

---

## Problem Statement

Maintenance strategies in industrial environments usually fall into two categories:

**Reactive maintenance**
Machines are repaired only after failure.

**Scheduled maintenance**
Machines are serviced at fixed intervals whether necessary or not.

Both approaches create inefficiencies.

Predictive maintenance introduces a smarter alternative by using historical sensor data to predict degradation patterns and optimize maintenance timing.

The objective of this system is to convert raw operational signals into actionable maintenance intelligence.

---

## Dataset Information

This system is trained using the NASA CMAPSS turbofan degradation dataset.

The dataset simulates progressive equipment degradation under operational conditions and is widely used for predictive maintenance research.

### Input Signals

Sensor-based features used in this project include:

* Temperature
* Vibration
* Pressure
* Voltage
* Runtime duration

### Output Targets

The system learns two targets:

* Failure probability (classification)
* Remaining useful life (regression)

---

## Workflow Design

System pipeline:

Raw Data Collection
↓
Data Validation
↓
Cleaning & Normalization
↓
Feature Engineering
↓
Feature Scaling
↓
Class Balancing (SMOTE)
↓
Model Training
↓
Performance Evaluation
↓
Model Explainability
↓
Interactive Dashboard

---

## Core Components

### Data Processing

Responsible for:

* schema validation
* null handling
* duplicate removal
* outlier capping
* type conversion

---

### Feature Engineering

Transforms raw machine signals into enriched features for stronger predictive performance.

Includes:

* operational indicators
* derived sensor interactions
* engineered degradation signals

---

### Classification Pipeline

Purpose:

Identify machines at high failure risk.

Models tested:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* Support Vector Machine

Final selected model:

**Random Forest Classifier**

---

### Regression Pipeline

Purpose:

Estimate remaining operational lifespan.

Models tested:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* LightGBM Regressor

Final selected model:

**Random Forest Regressor**

---

### Explainability Module

To improve transparency, SHAP is integrated to interpret model behavior.

This helps identify:

* which sensor features matter most
* what drives failure predictions
* what affects RUL estimation

---

## Performance Results

## Failure Detection Performance

Best model: Random Forest

Accuracy: 97.17%
Precision: 84.99%
Recall: 87.62%
F1 Score: 86.28%
ROC-AUC: 0.9896

---

## Remaining Useful Life Prediction

Best model: Random Forest

MAE: 24.92
RMSE: 35.47
R² Score: 0.7246

---

## Technical Stack

Programming Language:

Python

Libraries:

* scikit-learn
* XGBoost
* LightGBM
* Pandas
* NumPy
* Matplotlib
* SHAP
* Streamlit

---

## Repository Structure

predictive-maintenance/

app.py
prepare_cmapss.py
requirements.txt

data/
models/
reports/
notebooks/

src/
├── data_loader.py
├── preprocessing.py
├── features.py
├── train.py
├── evaluate.py
├── explain.py
├── utils.py

---

## Setup Instructions

Clone repository:

git clone <repository-url>

Enter project directory:

cd predictive-maintenance

Create environment:

python -m venv venv

Activate environment:

venv\Scripts\activate

Install packages:

pip install -r requirements.txt

---

## Run Training

python src/train.py

---

## Run Evaluation

python src/evaluate.py

---

## Launch Dashboard

streamlit run app.py

---

## Why This Project Matters

Industrial downtime is expensive.

A practical predictive maintenance system can:

* reduce emergency failures
* improve maintenance planning
* increase equipment reliability
* reduce operational cost
* improve asset utilization

This project demonstrates how machine learning can directly support industrial decision-making.

---

## Future Development

Planned improvements:

* sequence modeling using LSTM
* transformer-based predictive maintenance
* live IoT integration
* edge deployment
* anomaly detection expansion

---

## Author

Himaj Reddy

Machine Learning | Applied AI | Industrial Analytics
