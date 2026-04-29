"""
train.py
--------
Model training module for:
  - Task 1: Binary failure classification
  - Task 2: Remaining Useful Life (RUL) regression

Includes hyperparameter tuning (GridSearchCV), cross-validation,
and model serialisation.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
MODEL_DIR = Path("models")

# ---------------------------------------------------------------------------
# Model registries
# ---------------------------------------------------------------------------

CLASSIFIERS: Dict[str, Any] = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        n_estimators=200, use_label_encoder=False,
        eval_metric="logloss", random_state=RANDOM_STATE,
        verbosity=0,
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=200, random_state=RANDOM_STATE,
        verbose=-1,
    ),
    "svm": SVC(
        kernel="rbf", probability=True, random_state=RANDOM_STATE
    ),
}

REGRESSORS: Dict[str, Any] = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "xgboost": XGBRegressor(
        n_estimators=200, random_state=RANDOM_STATE, verbosity=0
    ),
    "lightgbm": LGBMRegressor(
        n_estimators=200, random_state=RANDOM_STATE, verbose=-1
    ),
}

# ---------------------------------------------------------------------------
# Hyperparameter grids (used in grid-search fine-tuning)
# ---------------------------------------------------------------------------

CLASSIFIER_PARAM_GRIDS: Dict[str, Dict] = {
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    },
    "lightgbm": {
        "n_estimators": [100, 200],
        "num_leaves": [31, 63],
        "learning_rate": [0.05, 0.1],
    },
}

REGRESSOR_PARAM_GRIDS: Dict[str, Dict] = {
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    },
    "lightgbm": {
        "n_estimators": [100, 200],
        "num_leaves": [31, 63],
        "learning_rate": [0.05, 0.1],
    },
}


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------


class ModelTrainer:
    """
    Trains, cross-validates, optionally tunes, and saves all models.

    Parameters
    ----------
    tune_best : bool
        If True, run GridSearchCV on the best-scoring model for each task.
    cv_folds : int
        Number of cross-validation folds.
    """

    def __init__(self, tune_best: bool = True, cv_folds: int = 5) -> None:
        self.tune_best = tune_best
        self.cv_folds = cv_folds
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # populated after training
        self.clf_results: Dict[str, Dict] = {}
        self.reg_results: Dict[str, Dict] = {}
        self.best_classifier = None
        self.best_regressor = None
        self.best_clf_name: str = ""
        self.best_reg_name: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_classifiers(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict]:
        """Train all classifiers and return performance dict."""
        logger.info("=" * 60)
        logger.info("TRAINING CLASSIFIERS")
        logger.info("=" * 60)

        for name, clf in CLASSIFIERS.items():
            self.clf_results[name] = self._train_single(
                name, clf, X_train, y_train, X_test, y_test, task="clf"
            )

        self.best_clf_name, self.best_classifier = self._select_best(
            self.clf_results, metric="roc_auc"
        )

        if self.tune_best and self.best_clf_name in CLASSIFIER_PARAM_GRIDS:
            logger.info("Tuning best classifier: %s", self.best_clf_name)
            self.best_classifier = self._grid_search(
                CLASSIFIERS[self.best_clf_name],
                CLASSIFIER_PARAM_GRIDS[self.best_clf_name],
                X_train, y_train, scoring="roc_auc", task="clf"
            )
            self.best_classifier.fit(X_train, y_train)

        # Probability calibration for reliable failure probabilities
        self.best_classifier = CalibratedClassifierCV(
            self.best_classifier, cv=3, method="isotonic"
        )
        self.best_classifier.fit(X_train, y_train)

        self._save_model(self.best_classifier, "classifier.pkl")
        logger.info("Best classifier: %s", self.best_clf_name)
        return self.clf_results

    def train_regressors(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict]:
        """Train all RUL regressors and return performance dict."""
        logger.info("=" * 60)
        logger.info("TRAINING REGRESSORS (RUL)")
        logger.info("=" * 60)

        for name, reg in REGRESSORS.items():
            self.reg_results[name] = self._train_single(
                name, reg, X_train, y_train, X_test, y_test, task="reg"
            )

        self.best_reg_name, self.best_regressor = self._select_best(
            self.reg_results, metric="r2"
        )

        if self.tune_best and self.best_reg_name in REGRESSOR_PARAM_GRIDS:
            logger.info("Tuning best regressor: %s", self.best_reg_name)
            self.best_regressor = self._grid_search(
                REGRESSORS[self.best_reg_name],
                REGRESSOR_PARAM_GRIDS[self.best_reg_name],
                X_train, y_train, scoring="r2", task="reg"
            )
            self.best_regressor.fit(X_train, y_train)

        self._save_model(self.best_regressor, "regressor.pkl")
        logger.info("Best regressor: %s", self.best_reg_name)
        return self.reg_results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_single(
        self,
        name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task: str,
    ) -> Dict:
        logger.info("Training: %s", name)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        if task == "clf":
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_STATE)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            metric_name = "roc_auc"
            score_train = cv_scores.mean()
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring="r2", n_jobs=-1)
            metric_name = "r2"
            score_train = cv_scores.mean()

        result = {
            "model": model,
            metric_name: score_train,
            "cv_std": cv_scores.std(),
            "fit_time_s": elapsed,
        }
        logger.info(
            "  %s → %s: %.4f ± %.4f  (%.1fs)",
            name, metric_name, score_train, cv_scores.std(), elapsed,
        )
        return result

    def _select_best(
        self, results: Dict[str, Dict], metric: str
    ) -> Tuple[str, Any]:
        best_name = max(results, key=lambda n: results[n].get(metric, -np.inf))
        return best_name, results[best_name]["model"]

    def _grid_search(
        self,
        model: Any,
        param_grid: Dict,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str,
        task: str,
    ) -> Any:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE) if task == "clf" else 3
        gs = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, refit=True, verbose=0,
        )
        gs.fit(X, y)
        logger.info("Best params: %s  →  %s: %.4f", gs.best_params_, scoring, gs.best_score_)
        return gs.best_estimator_

    def _save_model(self, model: Any, filename: str) -> None:
        path = MODEL_DIR / filename
        joblib.dump(model, path)
        logger.info("Model saved: %s", path)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def train_all(
    splits: Dict,
    tune_best: bool = True,
    cv_folds: int = 5,
) -> ModelTrainer:
    """
    High-level entry-point used by the main training script.

    Parameters
    ----------
    splits : dict
        Output of DataPreprocessor.fit_transform()
    """
    trainer = ModelTrainer(tune_best=tune_best, cv_folds=cv_folds)

    clf_split = splits.get("classification")
    if clf_split is not None:
        X_tr, X_te, y_tr, y_te = clf_split
        trainer.train_classifiers(X_tr, y_tr, X_te, y_te)

    reg_split = splits.get("regression")
    if reg_split is not None:
        X_tr, X_te, y_tr, y_te = reg_split
        trainer.train_regressors(X_tr, y_tr, X_te, y_te)

    return trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_data, generate_synthetic_dataset
    from src.features import FeatureEngineer
    from src.preprocessing import preprocess
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    os.makedirs("data/raw", exist_ok=True)
    raw_path = "data/raw/sensor_data.csv"
    if not Path(raw_path).exists():
        generate_synthetic_dataset().to_csv(raw_path, index=False)

    df = load_data(raw_path)
    df = FeatureEngineer().fit_transform(df)
    splits, _ = preprocess(df)

    trainer = train_all(splits, tune_best=True)
    print("\nClassifier results:")
    for name, res in trainer.clf_results.items():
        print(f"  {name}: roc_auc={res['roc_auc']:.4f} ± {res['cv_std']:.4f}")
    print(f"\nBest classifier: {trainer.best_clf_name}")

    if trainer.reg_results:
        print("\nRegressor results:")
        for name, res in trainer.reg_results.items():
            print(f"  {name}: r2={res['r2']:.4f} ± {res['cv_std']:.4f}")
        print(f"Best regressor: {trainer.best_reg_name}")
