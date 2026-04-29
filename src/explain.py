"""
explain.py
----------
Model explainability using SHAP.
Provides global and local feature importance for both tasks.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

PLOTS_DIR = Path("reports/figures")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class ModelExplainer:
    """
    Generates SHAP-based explanations for any sklearn-compatible model.

    Usage
    -----
    explainer = ModelExplainer(model, X_train, feature_names)
    explainer.global_summary(X_test)
    explainer.local_explanation(X_test[0])
    """

    def __init__(
        self,
        model: Any,
        X_background: np.ndarray,
        feature_names: List[str],
        model_name: str = "model",
        task: str = "clf",        # 'clf' or 'reg'
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.model_name = model_name
        self.task = task

        logger.info("Building SHAP explainer for: %s", model_name)
        self.explainer = self._build_explainer(X_background)
        self.shap_values: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute and cache SHAP values for a dataset."""
        shap_vals = self.explainer(X)
        if self.task == "clf" and hasattr(shap_vals, "values"):
            # For binary classifiers take the positive class
            values = shap_vals.values
            if values.ndim == 3:
                values = values[:, :, 1]
            self.shap_values = values
        else:
            self.shap_values = shap_vals.values if hasattr(shap_vals, "values") else shap_vals
        return self.shap_values

    def global_summary(
        self, X: np.ndarray, max_display: int = 15, save: bool = True
    ) -> pd.DataFrame:
        """
        Plot global feature importance (bar + beeswarm) and return
        a ranked DataFrame of mean |SHAP| values.
        """
        shap_vals = self.compute_shap_values(X)

        # --- Bar chart (mean |SHAP|) ---
        mean_abs = np.abs(shap_vals).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)

        fig, ax = plt.subplots(figsize=(9, 6))
        top = importance_df.head(max_display)
        ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="steelblue")
        ax.set(xlabel="Mean |SHAP value|", title=f"Global Feature Importance — {self.model_name}")
        plt.tight_layout()
        if save:
            path = PLOTS_DIR / f"shap_global_{self.model_name}.png"
            plt.savefig(path, dpi=150)
            logger.info("Global SHAP plot saved: %s", path)
        plt.show()

        # --- Beeswarm (requires shap library plotting) ---
        try:
            shap_exp = shap.Explanation(
                values=shap_vals,
                feature_names=self.feature_names,
            )
            plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(shap_exp, max_display=max_display, show=False)
            plt.title(f"SHAP Beeswarm — {self.model_name}")
            plt.tight_layout()
            if save:
                path = PLOTS_DIR / f"shap_beeswarm_{self.model_name}.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
                logger.info("Beeswarm plot saved: %s", path)
            plt.show()
        except Exception as exc:
            logger.warning("Beeswarm plot failed: %s", exc)

        return importance_df

    def local_explanation(
        self, x_instance: np.ndarray, save: bool = True, instance_idx: int = 0
    ) -> pd.DataFrame:
        """
        Waterfall plot for a single prediction (local explanation).
        Returns a DataFrame with per-feature SHAP contributions.
        """
        if x_instance.ndim == 1:
            x_instance = x_instance.reshape(1, -1)

        shap_exp = self.explainer(x_instance)
        shap_val = shap_exp.values[0]
        if shap_val.ndim > 1:
            shap_val = shap_val[:, 1]  # positive class

        local_df = pd.DataFrame(
            {"feature": self.feature_names, "shap_value": shap_val}
        ).sort_values("shap_value", key=abs, ascending=False)

        try:
            single_exp = shap.Explanation(
                values=shap_val,
                base_values=shap_exp.base_values[0] if shap_exp.base_values.ndim > 0 else shap_exp.base_values,
                data=x_instance[0],
                feature_names=self.feature_names,
            )
            plt.figure(figsize=(10, 5))
            shap.plots.waterfall(single_exp, show=False)
            plt.title(f"Local Explanation — instance #{instance_idx}")
            plt.tight_layout()
            if save:
                path = PLOTS_DIR / f"shap_local_{self.model_name}_inst{instance_idx}.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
                logger.info("Local SHAP plot saved: %s", path)
            plt.show()
        except Exception as exc:
            logger.warning("Waterfall plot failed: %s", exc)

        return local_df

    def dependence_plot(
        self, feature_name: str, X: np.ndarray, save: bool = True
    ) -> None:
        """SHAP dependence plot for a specific feature."""
        if self.shap_values is None:
            self.compute_shap_values(X)

        if feature_name not in self.feature_names:
            logger.error("Feature '%s' not found.", feature_name)
            return

        feat_idx = self.feature_names.index(feature_name)
        feat_vals = X[:, feat_idx]

        plt.figure(figsize=(8, 5))
        plt.scatter(feat_vals, self.shap_values[:, feat_idx], alpha=0.3, s=10)
        plt.axhline(0, color="red", linestyle="--", linewidth=0.8)
        plt.xlabel(feature_name)
        plt.ylabel(f"SHAP value for {feature_name}")
        plt.title(f"SHAP Dependence Plot — {feature_name}")
        plt.tight_layout()
        if save:
            path = PLOTS_DIR / f"shap_dep_{feature_name}_{self.model_name}.png"
            plt.savefig(path, dpi=150)
        plt.show()

    def top_features(self, n: int = 10) -> List[str]:
        """Return list of top-n feature names by mean |SHAP|."""
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:n]
        return [self.feature_names[i] for i in top_idx]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_explainer(self, X_background: np.ndarray) -> shap.Explainer:
        """Select the appropriate SHAP explainer based on model type."""
        model_type = type(self.model).__name__.lower()

        try:
            if any(kw in model_type for kw in ["xgb", "lgbm", "randomforest", "gradientboosting"]):
                # Try TreeExplainer first — fast & exact
                inner = self._unwrap_calibrated(self.model)
                return shap.TreeExplainer(inner)
        except Exception:
            pass

        try:
            # LinearExplainer for linear models
            if any(kw in model_type for kw in ["logistic", "linear", "ridge", "lasso"]):
                inner = self._unwrap_calibrated(self.model)
                return shap.LinearExplainer(inner, X_background)
        except Exception:
            pass

        # Fallback: KernelExplainer (model-agnostic, slower)
        logger.info("Using KernelExplainer (slower but model-agnostic).")
        if self.task == "clf":
            predict_fn = lambda x: self.model.predict_proba(x)[:, 1]
        else:
            predict_fn = self.model.predict
        background = shap.kmeans(X_background, 50)
        return shap.KernelExplainer(predict_fn, background)

    @staticmethod
    def _unwrap_calibrated(model: Any) -> Any:
        """Unwrap CalibratedClassifierCV to get the base estimator."""
        if hasattr(model, "calibrated_classifiers_"):
            return model.calibrated_classifiers_[0].estimator
        if hasattr(model, "estimator"):
            return model.estimator
        return model


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def explain_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    model_name: str = "model",
    task: str = "clf",
    n_samples: int = 300,
) -> ModelExplainer:
    """
    High-level function: build explainer, compute SHAP values, plot global summary.
    """
    # Use a subsample for speed
    idx = np.random.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)
    X_sample = X_test[idx]

    explainer = ModelExplainer(model, X_train, feature_names, model_name, task)
    explainer.global_summary(X_sample)
    explainer.local_explanation(X_sample[0], instance_idx=0)
    return explainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, joblib
    sys.path.insert(0, ".")
    from src.data_loader import load_data, generate_synthetic_dataset
    from src.features import FeatureEngineer, get_classification_features
    from src.preprocessing import DataPreprocessor

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    raw_path = "data/raw/sensor_data.csv"
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    if not Path(raw_path).exists():
        generate_synthetic_dataset().to_csv(raw_path, index=False)

    df = load_data(raw_path)
    fe = FeatureEngineer()
    df_feat = fe.fit_transform(df)

    pp = DataPreprocessor.load("models")
    X_test_scaled = pp.transform_inference(df_feat.head(500))
    feature_names = pp.feature_columns

    clf = joblib.load("models/classifier.pkl")
    exp = explain_model(clf, X_test_scaled, X_test_scaled, feature_names, "best_classifier", "clf")
    print("Top features:", exp.top_features(5))
