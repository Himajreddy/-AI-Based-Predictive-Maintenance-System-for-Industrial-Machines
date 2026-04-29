"""
data_loader.py
--------------
Handles loading, validating, cleaning, and preparing raw sensor data
for the Predictive Maintenance pipeline.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "voltage",
    "runtime_hours",
    "failure",
]

OPTIONAL_COLUMNS = [
    "humidity",
    "rotational_speed",
    "torque",
    "wear_level",
]

NUMERIC_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "voltage",
    "runtime_hours",
    "humidity",
    "rotational_speed",
    "torque",
    "wear_level",
]


class DataLoader:
    """Loads and validates raw sensor data from CSV."""

    def __init__(self, filepath: str) -> None:
        self.filepath = Path(filepath)
        self.df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Full pipeline: load → validate → clean → return DataFrame."""
        self._read_csv()
        self._normalise_columns()
        self._validate_schema()
        self._handle_missing_values()
        self._remove_duplicates()
        self._detect_and_cap_outliers()
        self._cast_types()
        logger.info("Data loading complete. Shape: %s", self.df.shape)
        return self.df.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_csv(self) -> None:
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.filepath}")
        logger.info("Reading CSV from %s", self.filepath)
        self.df = pd.read_csv(self.filepath)
        logger.info("Raw shape: %s", self.df.shape)

    def _normalise_columns(self) -> None:
        """Lower-case and strip whitespace from column names."""
        self.df.columns = [c.strip().lower().replace(" ", "_") for c in self.df.columns]

    def _validate_schema(self) -> None:
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        logger.info("Schema validation passed.")

    def _handle_missing_values(self) -> None:
        before = self.df.isnull().sum().sum()
        if before == 0:
            logger.info("No missing values found.")
            return

        numeric_cols = [c for c in NUMERIC_COLUMNS if c in self.df.columns]
        for col in numeric_cols:
            if self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                logger.info("Filled %s NaNs in '%s' with median (%.4f)", self.df[col].isnull().sum(), col, median_val)

        # Target column: drop rows with missing label
        if self.df["failure"].isnull().any():
            n_drop = self.df["failure"].isnull().sum()
            self.df.dropna(subset=["failure"], inplace=True)
            logger.warning("Dropped %d rows with missing 'failure' label.", n_drop)

        after = self.df.isnull().sum().sum()
        logger.info("Missing values handled: %d → %d", before, after)

    def _remove_duplicates(self) -> None:
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)
        if removed:
            logger.info("Removed %d duplicate rows.", removed)
        else:
            logger.info("No duplicates found.")

    def _detect_and_cap_outliers(self, z_thresh: float = 3.5) -> None:
        """Cap extreme outliers using the modified Z-score method."""
        numeric_cols = [c for c in NUMERIC_COLUMNS if c in self.df.columns]
        total_capped = 0
        for col in numeric_cols:
            median = self.df[col].median()
            mad = np.median(np.abs(self.df[col] - median))
            if mad == 0:
                continue
            modified_z = 0.6745 * (self.df[col] - median) / mad
            mask = np.abs(modified_z) > z_thresh
            n_outliers = mask.sum()
            if n_outliers:
                lower = self.df.loc[~mask, col].min()
                upper = self.df.loc[~mask, col].max()
                self.df[col] = self.df[col].clip(lower, upper)
                total_capped += n_outliers
        logger.info("Outlier capping complete. Total cells affected: %d", total_capped)

    def _cast_types(self) -> None:
        numeric_cols = [c for c in NUMERIC_COLUMNS if c in self.df.columns]
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df["failure"] = self.df["failure"].astype(int)


# ------------------------------------------------------------------
# Standalone helpers
# ------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Convenience wrapper for the DataLoader class."""
    return DataLoader(filepath).load()


def save_processed(df: pd.DataFrame, output_path: str = "data/processed/clean_data.csv") -> None:
    """Persist cleaned DataFrame to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Processed data saved to %s", output_path)


def generate_synthetic_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic predictive-maintenance dataset.
    Useful for local testing without a Kaggle download.
    """
    rng = np.random.default_rng(seed)

    runtime_hours = rng.uniform(0, 10_000, n_samples)
    temperature = 60 + 0.003 * runtime_hours + rng.normal(0, 5, n_samples)
    vibration = 0.5 + 0.00005 * runtime_hours + rng.normal(0, 0.1, n_samples)
    pressure = 100 + rng.normal(0, 8, n_samples)
    voltage = 220 + rng.normal(0, 10, n_samples)
    humidity = rng.uniform(30, 80, n_samples)
    rotational_speed = rng.uniform(1000, 3000, n_samples)
    torque = rng.uniform(20, 200, n_samples)
    wear_level = runtime_hours / 10_000 + rng.uniform(0, 0.1, n_samples)

    # Failure probability increases with high wear, temperature, vibration
    failure_score = (
        0.3 * (wear_level > 0.85).astype(float)
        + 0.3 * (temperature > 80).astype(float)
        + 0.2 * (vibration > 0.7).astype(float)
        + 0.2 * rng.uniform(0, 1, n_samples)
    )
    failure = (failure_score > 0.6).astype(int)

    # RUL: hours until failure (0 if already failed)
    rul = np.where(failure == 0, np.maximum(0, 10_000 - runtime_hours), 0)
    rul = rul + rng.normal(0, 50, n_samples)
    rul = np.maximum(0, rul)

    df = pd.DataFrame(
        {
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "voltage": voltage,
            "runtime_hours": runtime_hours,
            "humidity": humidity,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "wear_level": wear_level,
            "failure": failure,
            "rul": rul,
        }
    )
    logger.info("Synthetic dataset generated. Shape: %s  Failure rate: %.2f%%", df.shape, failure.mean() * 100)
    return df


if __name__ == "__main__":
    raw_path = "data/raw/sensor_data.csv"
    os.makedirs("data/raw", exist_ok=True)

    if not Path(raw_path).exists():
        logger.info("No raw CSV found — generating synthetic dataset.")
        df_syn = generate_synthetic_dataset()
        df_syn.to_csv(raw_path, index=False)

    df = load_data(raw_path)
    save_processed(df)
    print(df.head())
    print(df.info())
