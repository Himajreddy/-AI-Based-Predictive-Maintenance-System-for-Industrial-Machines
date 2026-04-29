"""
app.py
------
Streamlit dashboard for the Predictive Maintenance AI System.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

from src.utils import failure_probability_to_risk, rul_to_urgency, SENSOR_SCHEMA
from src.features import FeatureEngineer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🔧 Predictive Maintenance AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load artefacts (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    try:
        clf = joblib.load("models/classifier.pkl")
        reg = joblib.load("models/regressor.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_columns = joblib.load("models/feature_columns.pkl")
        return clf, reg, scaler, feature_columns
    except FileNotFoundError:
        return None, None, None, None


classifier, regressor, scaler, feature_columns = load_models()
models_loaded = classifier is not None

# ---------------------------------------------------------------------------
# Sidebar — Input panel
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Sensor Input Panel")
st.sidebar.markdown("Adjust the sensor readings for the machine you want to diagnose.")
st.sidebar.markdown("---")

temperature = st.sidebar.slider(
    "🌡️ Temperature (°C)",
    min_value=0.0, max_value=150.0, value=70.0, step=0.5,
)
vibration = st.sidebar.slider(
    "📳 Vibration (mm/s)",
    min_value=0.0, max_value=5.0, value=0.5, step=0.01,
)
pressure = st.sidebar.slider(
    "💨 Pressure (bar)",
    min_value=0.0, max_value=200.0, value=100.0, step=1.0,
)
voltage = st.sidebar.slider(
    "⚡ Voltage (V)",
    min_value=180.0, max_value=260.0, value=220.0, step=1.0,
)
runtime_hours = st.sidebar.slider(
    "⏱️ Runtime Hours (h)",
    min_value=0.0, max_value=20000.0, value=3000.0, step=50.0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Optional / Advanced Sensors**")
show_advanced = st.sidebar.checkbox("Show advanced sensors", value=False)

humidity = 55.0
rotational_speed = 1500.0
torque = 80.0
wear_level = 0.3

if show_advanced:
    humidity = st.sidebar.slider("💧 Humidity (%)", 0.0, 100.0, 55.0, 1.0)
    rotational_speed = st.sidebar.slider("🔄 Rotational Speed (RPM)", 500.0, 5000.0, 1500.0, 50.0)
    torque = st.sidebar.slider("🔩 Torque (Nm)", 0.0, 500.0, 80.0, 5.0)
    wear_level = st.sidebar.slider("🪛 Wear Level (0–1)", 0.0, 1.0, 0.3, 0.01)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("🔧 Predictive Maintenance AI System")
st.markdown(
    """
    An end-to-end machine learning system that predicts **machine failure probability**
    and estimates **Remaining Useful Life (RUL)** from real-time sensor data.
    """
)

# Warning if models not loaded
if not models_loaded:
    st.warning(
        "⚠️ Trained models not found in `models/`. "
        "Please run the training pipeline first:\n\n"
        "```bash\npython src/train.py\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Build input DataFrame and apply feature engineering
# ---------------------------------------------------------------------------

raw_input = pd.DataFrame([{
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "voltage": voltage,
    "runtime_hours": runtime_hours,
    "humidity": humidity,
    "rotational_speed": rotational_speed,
    "torque": torque,
    "wear_level": wear_level,
    "failure": 0,        # placeholder — not used for inference
}])

fe = FeatureEngineer()
input_featured = fe.fit_transform(raw_input)

# Align columns to training schema
for col in feature_columns:
    if col not in input_featured.columns:
        input_featured[col] = 0.0
input_featured = input_featured[feature_columns]

X_input = scaler.transform(input_featured.values)

# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

failure_prob = float(classifier.predict_proba(X_input)[0, 1])
rul_pred = float(max(0, regressor.predict(X_input)[0]))
risk_label = failure_probability_to_risk(failure_prob)
urgency_label = rul_to_urgency(rul_pred)

# ---------------------------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💥 Failure Probability", f"{failure_prob * 100:.1f}%")
    st.markdown(f"**Risk Level:** {risk_label}")

with col2:
    st.metric("⏳ Remaining Useful Life", f"{rul_pred:,.0f} hours")
    st.markdown(f"**Status:** {urgency_label}")

with col3:
    days_remaining = rul_pred / 24
    st.metric("📅 Days Remaining", f"{days_remaining:,.1f} days")
    st.markdown(f"**Weeks:** {days_remaining / 7:.1f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Gauge chart — failure probability
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📊 Risk Gauge", "🧠 Feature Importance", "📋 Sensor Summary"])

with tab1:
    fig, ax = plt.subplots(figsize=(7, 3.5), subplot_kw={"projection": "polar"})
    theta_range = np.linspace(np.pi, 0, 300)
    # Background arc
    ax.plot(theta_range, np.ones_like(theta_range), "lightgrey", lw=20, alpha=0.5)
    # Coloured arc
    theta_fill = np.linspace(np.pi, np.pi * (1 - failure_prob), 300)
    colour = "#d62728" if failure_prob >= 0.75 else "#ff7f0e" if failure_prob >= 0.50 else "#ffdd57" if failure_prob >= 0.25 else "#2ca02c"
    ax.plot(theta_fill, np.ones_like(theta_fill), colour, lw=20)
    ax.set_ylim(0, 1.3)
    ax.axis("off")
    ax.text(0, -0.15, f"{failure_prob * 100:.1f}%", ha="center", va="center",
            fontsize=30, fontweight="bold", transform=ax.transAxes)
    ax.text(0, -0.28, "Failure Probability", ha="center", va="center",
            fontsize=13, transform=ax.transAxes, color="grey")
    for label, angle in zip(["0%", "25%", "50%", "75%", "100%"], [np.pi, 3*np.pi/4, np.pi/2, np.pi/4, 0]):
        x = np.cos(angle) * 1.12
        y = np.sin(angle) * 1.12
        ax.text(angle, 1.15, label, ha="center", va="center", fontsize=8, color="grey")
    st.pyplot(fig, use_container_width=False)

with tab2:
    try:
        # SHAP for the input instance
        inner_clf = classifier
        if hasattr(classifier, "calibrated_classifiers_"):
            inner_clf = classifier.calibrated_classifiers_[0].estimator

        try:
            ex = shap.TreeExplainer(inner_clf)
            shap_vals = ex.shap_values(X_input)
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]  # positive class
            elif shap_vals.ndim == 3:
                sv = shap_vals[0, :, 1]
            else:
                sv = shap_vals[0]
        except Exception:
            bg = np.zeros((1, X_input.shape[1]))
            ex = shap.KernelExplainer(
                lambda x: classifier.predict_proba(x)[:, 1], bg
            )
            sv = ex.shap_values(X_input)[0]

        feat_importance = pd.DataFrame(
            {"Feature": feature_columns, "SHAP Value": sv}
        ).sort_values("SHAP Value", key=abs, ascending=True).tail(12)

        colours = ["#d62728" if v > 0 else "#1f77b4" for v in feat_importance["SHAP Value"]]
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.barh(feat_importance["Feature"], feat_importance["SHAP Value"], color=colours)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set(xlabel="SHAP Value (impact on failure probability)",
                title="Feature Contribution to This Prediction")
        plt.tight_layout()
        st.pyplot(fig2)
        st.caption("🔴 Red = increases failure risk  |  🔵 Blue = decreases failure risk")

    except Exception as exc:
        st.warning(f"SHAP explanation unavailable: {exc}")
        # Fallback: simple bar chart
        feat_df = pd.DataFrame({
            "Feature": feature_columns,
            "Value": X_input[0],
        }).sort_values("Value", key=abs, ascending=False).head(10)
        st.bar_chart(feat_df.set_index("Feature")["Value"])

with tab3:
    sensor_data = {
        "Sensor": ["Temperature", "Vibration", "Pressure", "Voltage", "Runtime Hours",
                   "Humidity", "Rotational Speed", "Torque", "Wear Level"],
        "Value": [temperature, vibration, pressure, voltage, runtime_hours,
                  humidity, rotational_speed, torque, wear_level],
        "Unit": ["°C", "mm/s", "bar", "V", "h", "%", "RPM", "Nm", "—"],
        "Normal Range": [
            "20–90 °C", "0–0.8 mm/s", "80–120 bar", "210–230 V", "0–10,000 h",
            "30–70 %", "1,000–3,000 RPM", "20–200 Nm", "0–0.6"
        ],
    }
    st.dataframe(pd.DataFrame(sensor_data), use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<small>🤖 Predictive Maintenance AI System | Built with Python, scikit-learn, XGBoost, LightGBM & Streamlit</small>",
    unsafe_allow_html=True,
)
