from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src.config import FORM_WINDOW, MODELS_DIR, ensure_dirs
from src.features import FEATURE_COLUMNS, FeatureBuilder


@dataclass
class TrainMetrics:
    model_type: str
    form_window: int
    n_samples: int
    n_train: int
    n_test: int
    mae: float
    rmse: float
    r2: float
    brier: float
    accuracy_at_050: float
    roc_auc: float
    baseline_elo_rmse: float
    baseline_elo_brier: float
    faceit_baseline_brier: float | None


def _elo_baseline_proba(delta_elo: pd.Series) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-delta_elo.to_numpy(dtype=float) / 400.0))


def _clip_proba(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), 0.0, 1.0)


class WinChanceRegressor:
    """LightGBM regressor that predicts continuous win chance in [0, 1]."""

    def __init__(self, model: lgb.LGBMRegressor, calibrator: IsotonicRegression | None = None):
        self.model = model
        self.calibrator = calibrator

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw = self.model.predict(x)
        if self.calibrator is not None:
            raw = self.calibrator.predict(raw)
        return _clip_proba(np.asarray(raw))


def train_model(
    *,
    min_prior_matches: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    calibrate: bool = True,
    model_name: str = "win_model",
    form_window: int = FORM_WINDOW,
) -> tuple[Path, TrainMetrics]:
    ensure_dirs()
    builder = FeatureBuilder(form_window=form_window)
    try:
        df = builder.build_dataset(min_prior_matches=min_prior_matches)
    finally:
        builder.close()

    if df.empty or len(df) < 50:
        raise RuntimeError(
            f"Not enough training rows ({len(df)}). Collect more matches first."
        )

    df = df.sort_values("started_at").reset_index(drop=True)
    split_idx = int(len(df) * (1.0 - test_size))
    split_idx = max(1, min(split_idx, len(df) - 1))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"].astype(float)
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"].astype(float)

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        verbosity=-1,
        objective="regression",
    )

    calibrator: IsotonicRegression | None = None
    if calibrate and len(train_df) >= 200:
        cal_n = max(30, int(len(train_df) * 0.15))
        x_fit = x_train.iloc[:-cal_n]
        y_fit = y_train.iloc[:-cal_n]
        x_cal = x_train.iloc[-cal_n:]
        y_cal = y_train.iloc[-cal_n:]
        model.fit(x_fit, y_fit)
        raw_cal = model.predict(x_cal)
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw_cal, y_cal)
    else:
        model.fit(x_train, y_train)

    estimator = WinChanceRegressor(model, calibrator)
    pred = estimator.predict(x_test)
    hard = (pred >= 0.5).astype(int)
    y_true = y_test.to_numpy(dtype=int)

    elo_proba = _elo_baseline_proba(test_df["delta_avg_elo"])

    faceit_brier = None
    if test_df["faceit_win_prob_f1"].notna().sum() >= 20:
        faceit_proba = test_df["faceit_win_prob_f1"].fillna(0.5).to_numpy(dtype=float)
        faceit_brier = float(brier_score_loss(y_true, faceit_proba))

    metrics = TrainMetrics(
        model_type="regression",
        form_window=form_window,
        n_samples=len(df),
        n_train=len(train_df),
        n_test=len(test_df),
        mae=float(mean_absolute_error(y_true, pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, pred))),
        r2=float(r2_score(y_true, pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        brier=float(brier_score_loss(y_true, pred)),
        accuracy_at_050=float(accuracy_score(y_true, hard)),
        roc_auc=float(roc_auc_score(y_true, pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        baseline_elo_rmse=float(np.sqrt(mean_squared_error(y_true, elo_proba))),
        baseline_elo_brier=float(brier_score_loss(y_true, elo_proba)),
        faceit_baseline_brier=faceit_brier,
    )

    model_path = MODELS_DIR / f"{model_name}.joblib"
    meta_path = MODELS_DIR / f"{model_name}.meta.json"
    joblib.dump(
        {
            "model": estimator,
            "model_type": "regression",
            "feature_columns": FEATURE_COLUMNS,
            "min_prior_matches": min_prior_matches,
            "form_window": form_window,
        },
        model_path,
    )
    meta_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    try:
        df.to_parquet(MODELS_DIR / f"{model_name}_dataset.parquet", index=False)
    except Exception:
        df.to_csv(MODELS_DIR / f"{model_name}_dataset.csv", index=False)

    return model_path, metrics
