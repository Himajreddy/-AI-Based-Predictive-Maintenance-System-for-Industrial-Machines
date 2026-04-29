"""
utils.py
--------
Shared utility functions used across the predictive maintenance pipeline.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logger with console (and optional file) handler."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------


def save_artifact(obj: Any, path: str) -> None:
    """Persist any Python object with joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved artifact: %s", path)


def load_artifact(path: str) -> Any:
    """Load a joblib-serialised artifact."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------


def timer(func):
    """Decorator that logs wall-clock time of any function call."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("%s completed in %.3f s", func.__qualname__, elapsed)
        return result

    return wrapper


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Extended describe() including skewness and kurtosis."""
    desc = df.describe().T
    desc["skew"] = df.skew(numeric_only=True)
    desc["kurt"] = df.kurt(numeric_only=True)
    desc["missing"] = df.isnull().sum()
    desc["missing_%"] = (df.isnull().sum() / len(df) * 100).round(2)
    return desc


def class_balance(df: pd.DataFrame, target_col: str = "failure") -> pd.DataFrame:
    """Return a DataFrame summarising class counts and proportions."""
    counts = df[target_col].value_counts()
    pct = df[target_col].value_counts(normalize=True) * 100
    return pd.DataFrame({"count": counts, "pct": pct.round(2)})


def check_memory(df: pd.DataFrame) -> str:
    """Return a human-readable memory usage string for a DataFrame."""
    mem_bytes = df.memory_usage(deep=True).sum()
    for unit in ["B", "KB", "MB", "GB"]:
        if mem_bytes < 1024:
            return f"{mem_bytes:.1f} {unit}"
        mem_bytes /= 1024
    return f"{mem_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# Risk labelling
# ---------------------------------------------------------------------------


def failure_probability_to_risk(prob: float) -> str:
    """Map a failure probability to a human-readable risk level."""
    if prob >= 0.75:
        return "🔴 CRITICAL"
    elif prob >= 0.50:
        return "🟠 HIGH"
    elif prob >= 0.25:
        return "🟡 MEDIUM"
    else:
        return "🟢 LOW"


def rul_to_urgency(rul_hours: float) -> str:
    """Map Remaining Useful Life (hours) to a maintenance urgency label."""
    if rul_hours < 100:
        return "⚠️  Immediate maintenance required"
    elif rul_hours < 500:
        return "🔧 Schedule maintenance soon"
    elif rul_hours < 2000:
        return "📅 Plan maintenance in advance"
    else:
        return "✅ Machine is healthy"


# ---------------------------------------------------------------------------
# Config / versioning
# ---------------------------------------------------------------------------


def create_run_id() -> str:
    """Generate a unique run identifier based on current timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_run_metadata(meta: Dict, path: str = "models/run_metadata.json") -> None:
    """Save training run metadata as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Run metadata saved: %s", path)


def load_run_metadata(path: str = "models/run_metadata.json") -> Dict:
    """Load training run metadata from JSON."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seeds(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    logger.info("Random seeds set to %d", seed)


# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

SENSOR_SCHEMA = {
    "temperature": {"min": 0, "max": 150, "unit": "°C", "description": "Machine surface temperature"},
    "vibration": {"min": 0, "max": 5, "unit": "mm/s", "description": "Vibration amplitude"},
    "pressure": {"min": 0, "max": 200, "unit": "bar", "description": "Operating pressure"},
    "voltage": {"min": 180, "max": 260, "unit": "V", "description": "Supply voltage"},
    "runtime_hours": {"min": 0, "max": 50000, "unit": "h", "description": "Cumulative operating hours"},
    "humidity": {"min": 0, "max": 100, "unit": "%", "description": "Ambient humidity"},
    "rotational_speed": {"min": 0, "max": 10000, "unit": "RPM", "description": "Shaft rotational speed"},
    "torque": {"min": 0, "max": 500, "unit": "Nm", "description": "Motor torque"},
    "wear_level": {"min": 0, "max": 1, "unit": "—", "description": "Component wear (0=new, 1=fully worn)"},
}


def validate_input(data: Dict[str, float]) -> List[str]:
    """Validate a single inference input dict against sensor schema."""
    errors: List[str] = []
    for key, spec in SENSOR_SCHEMA.items():
        if key not in data:
            continue  # optional fields
        val = data[key]
        if val < spec["min"] or val > spec["max"]:
            errors.append(
                f"'{key}' = {val} is outside valid range [{spec['min']}, {spec['max']}] {spec['unit']}"
            )
    return errors
