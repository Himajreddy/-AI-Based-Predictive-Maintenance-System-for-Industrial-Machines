"""
preprocessing.py
----------------
Label encoding, feature scaling, train-test split, and sklearn Pipeline
construction for both the classification and regression tasks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSIFICATION_TARGET = "failure"
REGRESSION_TARGET = "rul"
TEST_SIZE = 0.20
RANDOM_STATE = 42

EXCLUDE_COLS = {CLASSIFICATION_TARGET, REGRESSION_TARGET}

CATEGORICAL_COLS: List[str] = []   # add column names if categorical cols exist


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class DataPreprocessor:
    """
    Handles all preprocessing steps for the predictive maintenance dataset.

    Usage (training)
    ----------------
    pp = DataPreprocessor()
    splits = pp.fit_transform(df)

    Usage (inference)
    -----------------
    pp = DataPreprocessor.load("models/")
    X_scaled = pp.transform_inference(raw_input_df)
    """

    def __init__(
        self,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        use_smote: bool = True,
        smote_k: int = 5,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.use_smote = use_smote
        self.smote_k = smote_k

        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Training pipeline
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Dict[str, Tuple]:
        """
        Full preprocessing for training.

        Returns
        -------
        dict with keys:
            'classification' → (X_train, X_test, y_train, y_test)
            'regression'     → (X_train_r, X_test_r, y_train_r, y_test_r)
        """
        df = df.copy()
        df = self._encode_categoricals(df, fit=True)

        # ---- Feature columns ----
        self.feature_columns = [c for c in df.columns if c not in EXCLUDE_COLS]
        X = df[self.feature_columns].values
        y_cls = df[CLASSIFICATION_TARGET].values.astype(int)

        # ---- Scale ----
        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True

        # ---- Train / test split ----
        X_train, X_test, y_train_cls, y_test_cls = train_test_split(
            X_scaled, y_cls, test_size=self.test_size,
            random_state=self.random_state, stratify=y_cls
        )

        logger.info(
            "Classification split — train: %d  test: %d  failure rate train: %.2f%%",
            len(X_train), len(X_test), y_train_cls.mean() * 100,
        )

        # ---- SMOTE on training set only ----
        if self.use_smote:
            X_train, y_train_cls = self._apply_smote(X_train, y_train_cls)

        # ---- Regression (RUL) ----
        if REGRESSION_TARGET in df.columns:
            y_reg = df[REGRESSION_TARGET].values.astype(float)
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                X_scaled, y_reg, test_size=self.test_size,
                random_state=self.random_state
            )
            logger.info(
                "Regression split — train: %d  test: %d  mean RUL: %.1f",
                len(X_train_r), len(X_test_r), y_train_r.mean(),
            )
            reg_split = (X_train_r, X_test_r, y_train_r, y_test_r)
        else:
            reg_split = None
            logger.warning("'rul' column not found — skipping regression split.")

        return {
            "classification": (X_train, X_test, y_train_cls, y_test_cls),
            "regression": reg_split,
        }

    # ------------------------------------------------------------------
    # Inference pipeline
    # ------------------------------------------------------------------

    def transform_inference(self, df: pd.DataFrame) -> np.ndarray:
        """Scale a single-row (or multi-row) inference DataFrame."""
        if not self._fitted:
            raise RuntimeError("DataPreprocessor must be fitted before calling transform_inference.")
        df = df.copy()
        df = self._encode_categoricals(df, fit=False)
        # Ensure column order matches training
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_columns].values
        return self.scaler.transform(X)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, model_dir: str = "models") -> None:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{model_dir}/label_encoders.pkl")
        joblib.dump(self.feature_columns, f"{model_dir}/feature_columns.pkl")
        logger.info("Preprocessor artefacts saved to %s/", model_dir)

    @classmethod
    def load(cls, model_dir: str = "models") -> "DataPreprocessor":
        pp = cls.__new__(cls)
        pp.scaler = joblib.load(f"{model_dir}/scaler.pkl")
        pp.label_encoders = joblib.load(f"{model_dir}/label_encoders.pkl")
        pp.feature_columns = joblib.load(f"{model_dir}/feature_columns.pkl")
        pp._fitted = True
        logger.info("Preprocessor artefacts loaded from %s/", model_dir)
        return pp

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    continue
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda v: v if v in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        return df

    def _apply_smote(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        minority_count = int(y.sum())
        if minority_count < self.smote_k + 1:
            logger.warning(
                "Too few minority samples (%d) for SMOTE (k=%d). Skipping.",
                minority_count, self.smote_k,
            )
            return X, y
        sm = SMOTE(k_neighbors=self.smote_k, random_state=self.random_state)
        X_res, y_res = sm.fit_resample(X, y)
        logger.info(
            "SMOTE applied: %d → %d samples (failure rate: %.2f%%)",
            len(y), len(y_res), y_res.mean() * 100,
        )
        return X_res, y_res


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def preprocess(
    df: pd.DataFrame,
    model_dir: str = "models",
    use_smote: bool = True,
) -> Tuple[Dict, "DataPreprocessor"]:
    """One-shot preprocessing. Returns splits dict and fitted preprocessor."""
    pp = DataPreprocessor(use_smote=use_smote)
    splits = pp.fit_transform(df)
    pp.save(model_dir)
    return splits, pp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_data, generate_synthetic_dataset
    from src.features import FeatureEngineer
    import os

    os.makedirs("data/raw", exist_ok=True)
    raw_path = "data/raw/sensor_data.csv"
    if not Path(raw_path).exists():
        generate_synthetic_dataset().to_csv(raw_path, index=False)

    df = load_data(raw_path)
    df = FeatureEngineer().fit_transform(df)
    splits, pp = preprocess(df)

    X_tr, X_te, y_tr, y_te = splits["classification"]
    print(f"Train: {X_tr.shape}  Test: {X_te.shape}")
    print(f"Failure rate — train: {y_tr.mean():.2%}  test: {y_te.mean():.2%}")
