"""
features.py
-----------
Domain-driven feature engineering for predictive maintenance sensor data.
All transformations are reproducible and stateless (no data leakage).
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
COL_TEMP = "temperature"
COL_VIB = "vibration"
COL_PRES = "pressure"
COL_VOLT = "voltage"
COL_RUNTIME = "runtime_hours"
COL_HUMIDITY = "humidity"
COL_ROT_SPD = "rotational_speed"
COL_TORQUE = "torque"
COL_WEAR = "wear_level"


class FeatureEngineer:
    """Applies all feature engineering steps to a cleaned DataFrame."""

    def __init__(
        self,
        rolling_window: int = 5,
        add_optional: bool = True,
    ) -> None:
        self.rolling_window = rolling_window
        self.add_optional = add_optional
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to *df* (in-place copy).
        Call this on training data.
        """
        df = df.copy()
        df = self._core_features(df)
        if self.add_optional:
            df = self._optional_features(df)
        df = self._interaction_features(df)
        df = self._runtime_segments(df)
        self.feature_names = self._collect_feature_names(df)
        logger.info("Feature engineering complete. New shape: %s", df.shape)
        return df

    # alias so callers can use .transform() on test data
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit_transform(df)

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features derived from the mandatory sensor columns."""

        # 1. Temperature × Pressure ratio — indicates thermal-mechanical stress
        if COL_TEMP in df.columns and COL_PRES in df.columns:
            df["temp_pressure_ratio"] = df[COL_TEMP] / (df[COL_PRES] + 1e-6)
            logger.debug("Added temp_pressure_ratio")

        # 2. Voltage stability score — deviation from nominal 220 V
        if COL_VOLT in df.columns:
            df["voltage_deviation"] = (df[COL_VOLT] - 220).abs()
            df["voltage_stability"] = 1 / (1 + df["voltage_deviation"])
            logger.debug("Added voltage_deviation, voltage_stability")

        # 3. Rolling average vibration (proxy for vibration trend)
        if COL_VIB in df.columns:
            df["vibration_rolling_mean"] = (
                df[COL_VIB]
                .rolling(window=self.rolling_window, min_periods=1)
                .mean()
            )
            df["vibration_rolling_std"] = (
                df[COL_VIB]
                .rolling(window=self.rolling_window, min_periods=1)
                .std()
                .fillna(0)
            )
            logger.debug("Added vibration rolling features")

        # 4. Machine stress index — combined thermal & vibration load
        if COL_VIB in df.columns and COL_TEMP in df.columns:
            df["stress_index"] = df[COL_VIB] * df[COL_TEMP]
            logger.debug("Added stress_index")

        # 5. Runtime-based degradation proxy (log-scaled)
        if COL_RUNTIME in df.columns:
            df["log_runtime"] = np.log1p(df[COL_RUNTIME])
            df["runtime_squared"] = df[COL_RUNTIME] ** 2
            logger.debug("Added log_runtime, runtime_squared")

        return df

    def _optional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features from optional sensor columns (added only if present)."""

        # Humidity-temperature interaction
        if COL_HUMIDITY in df.columns and COL_TEMP in df.columns:
            df["heat_index"] = df[COL_TEMP] + 0.33 * df[COL_HUMIDITY] - 4
            logger.debug("Added heat_index")

        # Mechanical power estimate: P ∝ torque × rotational_speed
        if COL_TORQUE in df.columns and COL_ROT_SPD in df.columns:
            df["mechanical_power"] = df[COL_TORQUE] * df[COL_ROT_SPD] / 9550
            logger.debug("Added mechanical_power")

        # Wear acceleration: wear normalised by runtime
        if COL_WEAR in df.columns and COL_RUNTIME in df.columns:
            df["wear_rate"] = df[COL_WEAR] / (df[COL_RUNTIME] + 1)
            logger.debug("Added wear_rate")

        # Bearing load index
        if COL_VIB in df.columns and COL_ROT_SPD in df.columns:
            df["bearing_load_index"] = df[COL_VIB] * df[COL_ROT_SPD]
            logger.debug("Added bearing_load_index")

        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Higher-order cross-sensor interactions."""

        # Thermal-electrical stress
        if COL_TEMP in df.columns and COL_VOLT in df.columns:
            df["thermal_electrical_stress"] = df[COL_TEMP] * df["voltage_deviation"] if "voltage_deviation" in df.columns else df[COL_TEMP] * (df[COL_VOLT] - 220).abs()

        # Overall health score (heuristic — lower = healthier)
        health_components = []
        if COL_VIB in df.columns:
            health_components.append(_minmax_norm(df[COL_VIB]))
        if COL_TEMP in df.columns:
            health_components.append(_minmax_norm(df[COL_TEMP]))
        if "voltage_deviation" in df.columns:
            health_components.append(_minmax_norm(df["voltage_deviation"]))
        if COL_RUNTIME in df.columns:
            health_components.append(_minmax_norm(df[COL_RUNTIME]))

        if health_components:
            df["health_score"] = np.mean(health_components, axis=0)

        return df

    def _runtime_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode machine lifecycle phase as a categorical feature."""
        if COL_RUNTIME not in df.columns:
            return df

        bins = [0, 2000, 5000, 8000, np.inf]
        labels = [0, 1, 2, 3]          # early / mid / mature / end-of-life
        df["lifecycle_phase"] = pd.cut(
            df[COL_RUNTIME], bins=bins, labels=labels, right=False
        ).astype(int)
        logger.debug("Added lifecycle_phase")
        return df

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _collect_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Return list of engineered column names (excludes target cols)."""
        exclude = {"failure", "rul"}
        return [c for c in df.columns if c not in exclude]

    def get_feature_names(self) -> List[str]:
        return self.feature_names


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _minmax_norm(series: pd.Series) -> pd.Series:
    min_v, max_v = series.min(), series.max()
    if max_v == min_v:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_v) / (max_v - min_v)


def get_classification_features(df: pd.DataFrame) -> List[str]:
    """Return feature column names for the classification task."""
    exclude = {"failure", "rul"}
    return [c for c in df.columns if c not in exclude]


def get_regression_features(df: pd.DataFrame) -> List[str]:
    """Return feature column names for the RUL regression task."""
    exclude = {"failure", "rul"}
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_data, generate_synthetic_dataset, save_processed
    import os

    raw_path = "data/raw/sensor_data.csv"
    os.makedirs("data/raw", exist_ok=True)

    if not __import__("pathlib").Path(raw_path).exists():
        df_raw = generate_synthetic_dataset()
        df_raw.to_csv(raw_path, index=False)

    df = load_data(raw_path)
    fe = FeatureEngineer()
    df_feat = fe.fit_transform(df)
    save_processed(df_feat, "data/processed/featured_data.csv")
    print(df_feat.head())
    print("Engineered features:", fe.get_feature_names())
