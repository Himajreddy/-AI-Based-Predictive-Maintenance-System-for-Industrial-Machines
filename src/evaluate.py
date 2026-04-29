"""
evaluate.py
-----------
Comprehensive model evaluation for classification and regression tasks.
Generates metric tables, confusion matrices, and ROC curves.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

PLOTS_DIR = Path("reports/figures")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------


class ClassificationEvaluator:
    """Evaluates one or more classifier models."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.results: Dict[str, Dict] = {}

    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        optimize_threshold: bool = True,
    ) -> Dict:
        """Compute all classification metrics for a single model."""
        y_proba = model.predict_proba(X_test)[:, 1]

        if optimize_threshold:
            threshold = self._find_best_threshold(y_test, y_proba)
            logger.info("Optimal threshold for %s: %.3f", model_name, threshold)
        else:
            threshold = self.threshold

        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "threshold": threshold,
        }

        logger.info(
            "%s → Acc: %.4f  Prec: %.4f  Rec: %.4f  F1: %.4f  AUC: %.4f",
            model_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        )
        self.results[model_name] = metrics
        return metrics

    def evaluate_all(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """Evaluate a dict of {name: model} and return comparison DataFrame."""
        for name, model in models.items():
            self.evaluate(model, X_test, y_test, model_name=name)
        return self.comparison_table()

    def comparison_table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.results).T
        df = df.sort_values("roc_auc", ascending=False)
        return df

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
        model_name: str = "model", save: bool = True
    ) -> None:
        y_proba = model.predict_proba(X_test)[:, 1]
        threshold = self.results.get(model_name, {}).get("threshold", self.threshold)
        y_pred = (y_proba >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"],
            xlabel="Predicted", ylabel="Actual",
            title=f"Confusion Matrix — {model_name}",
        )
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save:
            path = PLOTS_DIR / f"confusion_matrix_{model_name}.png"
            plt.savefig(path, dpi=150)
            logger.info("Confusion matrix saved: %s", path)
        plt.show()

    def plot_roc_curves(
        self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray,
        save: bool = True
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, model in models.items():
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title="ROC Curves — All Classifiers")
        ax.legend(loc="lower right")
        plt.tight_layout()
        if save:
            path = PLOTS_DIR / "roc_curves.png"
            plt.savefig(path, dpi=150)
            logger.info("ROC curves saved: %s", path)
        plt.show()

    # ------------------------------------------------------------------
    # Threshold optimisation
    # ------------------------------------------------------------------

    @staticmethod
    def _find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Maximise F1 score over candidate thresholds."""
        thresholds = np.linspace(0.05, 0.95, 100)
        f1_scores = [
            f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
            for t in thresholds
        ]
        best_idx = int(np.argmax(f1_scores))
        return float(thresholds[best_idx])


# ---------------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------------


class RegressionEvaluator:
    """Evaluates one or more RUL regressor models."""

    def __init__(self) -> None:
        self.results: Dict[str, Dict] = {}

    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
    ) -> Dict:
        y_pred = model.predict(X_test)
        y_pred = np.maximum(0, y_pred)  # RUL cannot be negative

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        metrics = {"mae": mae, "rmse": rmse, "r2": r2}
        logger.info(
            "%s → MAE: %.2f  RMSE: %.2f  R²: %.4f",
            model_name, mae, rmse, r2,
        )
        self.results[model_name] = metrics
        return metrics

    def evaluate_all(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        for name, model in models.items():
            self.evaluate(model, X_test, y_test, model_name=name)
        return self.comparison_table()

    def comparison_table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.results).T
        return df.sort_values("r2", ascending=False)

    def plot_predictions(
        self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
        model_name: str = "model", save: bool = True, n: int = 200
    ) -> None:
        y_pred = np.maximum(0, model.predict(X_test))
        idx = np.random.choice(len(y_test), size=min(n, len(y_test)), replace=False)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Actual vs predicted scatter
        axes[0].scatter(y_test[idx], y_pred[idx], alpha=0.4, edgecolors="k", linewidths=0.3)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        axes[0].plot(lims, lims, "r--", lw=1.5)
        axes[0].set(xlabel="Actual RUL (h)", ylabel="Predicted RUL (h)",
                    title=f"Actual vs Predicted — {model_name}")

        # Residuals
        residuals = y_test[idx] - y_pred[idx]
        axes[1].scatter(y_pred[idx], residuals, alpha=0.4, edgecolors="k", linewidths=0.3)
        axes[1].axhline(0, color="r", linestyle="--", lw=1.5)
        axes[1].set(xlabel="Predicted RUL (h)", ylabel="Residual",
                    title="Residual Plot")

        plt.tight_layout()
        if save:
            path = PLOTS_DIR / f"rul_predictions_{model_name}.png"
            plt.savefig(path, dpi=150)
            logger.info("RUL prediction plot saved: %s", path)
        plt.show()


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def evaluate_classifiers(
    models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[pd.DataFrame, ClassificationEvaluator]:
    ev = ClassificationEvaluator()
    table = ev.evaluate_all(models, X_test, y_test)
    return table, ev


def evaluate_regressors(
    models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[pd.DataFrame, RegressionEvaluator]:
    ev = RegressionEvaluator()
    table = ev.evaluate_all(models, X_test, y_test)
    return table, ev


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_data, generate_synthetic_dataset
    from src.features import FeatureEngineer
    from src.preprocessing import preprocess
    from src.train import train_all

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    raw_path = "data/raw/sensor_data.csv"
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    if not Path(raw_path).exists():
        generate_synthetic_dataset().to_csv(raw_path, index=False)

    df = load_data(raw_path)
    df = FeatureEngineer().fit_transform(df)
    splits, _ = preprocess(df)

    trainer = train_all(splits, tune_best=False)

    clf_split = splits["classification"]
    X_tr, X_te, y_tr, y_te = clf_split
    clf_table, clf_ev = evaluate_classifiers(
        {trainer.best_clf_name: trainer.best_classifier}, X_te, y_te
    )
    print("\n=== CLASSIFICATION METRICS ===")
    print(clf_table.to_string())

    reg_split = splits["regression"]
    if reg_split:
        X_tr, X_te, y_tr, y_te = reg_split
        reg_table, _ = evaluate_regressors(
            {trainer.best_reg_name: trainer.best_regressor}, X_te, y_te
        )
        print("\n=== REGRESSION METRICS ===")
        print(reg_table.to_string())
