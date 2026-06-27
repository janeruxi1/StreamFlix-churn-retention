"""Model training utilities for the StreamFlix churn pipeline.

Three trainers + one prep helper:
    prepare_features()        -- raw df -> (X, y) model-ready matrix
    train_logistic_regression -- regularized LR baseline
    train_xgboost             -- gradient-boosted trees
    calibrate_xgboost         -- isotonic post-hoc calibration

Design principles:
    - Stateless: no hidden train/test leakage. The one piece of state
      (one-hot column order) is returned as `feature_names` so the
      Streamlit app can align inference rows identically.
    - Calibration kept SEPARATE from the base model. We don't use
      `scale_pos_weight` -- it sacrifices calibration for ranking,
      and the Phase 6 decision rule needs calibrated probabilities.
    - LR and XGBoost share the same input matrix so the comparison is
      apples-to-apples.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# Feature prep
# ---------------------------------------------------------------------
DROP_COLS = ["subscriber_id", "monthly_revenue", "churned_next_30d"]
# monthly_revenue is collinear with plan_tier AND is the LTV input the
# Phase 6 decision rule consumes -- exclude from training features.

CATEGORICAL_COLS = [
    "plan_tier", "billing_cycle", "country", "payment_method",
    "engagement_cohort", "tenure_bucket",
]
BOOLEAN_COLS = [
    "auto_renew", "multi_profile", "promo_active",
    "is_trial_drop_window", "is_anniversary_window",
    "recent_plan_change_flag", "promo_expiring_soon_flag",
    "high_risk_segment_flag",
]


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Convert an engineered subscriber DataFrame into (X, y).

    - Drops IDs, target, and the LTV column (kept aside for decision rule)
    - One-hot encodes categoricals
    - Casts booleans to int
    - Preserves all other numerics as-is
    """
    y = df["churned_next_30d"].astype(int)

    feat = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Cast booleans to int (XGBoost wants numeric; LR doesn't care)
    for col in BOOLEAN_COLS:
        if col in feat.columns:
            feat[col] = feat[col].astype(int)

    # One-hot encode categoricals
    feat = pd.get_dummies(feat, columns=CATEGORICAL_COLS, drop_first=False)

    # Cast any remaining bool dummy columns to int
    for col in feat.columns:
        if feat[col].dtype == bool:
            feat[col] = feat[col].astype(int)

    return feat, y


# ---------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------
def train_logistic_regression(X_train: pd.DataFrame,
                              y_train: pd.Series,
                              random_state: int = 42) -> Pipeline:
    """Regularized LR baseline with feature standardization.

    StandardScaler -> LogisticRegression with L2.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=2000,
            class_weight=None,    # keep calibration; let policy handle imbalance
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  random_state: int = 42) -> XGBClassifier:
    """XGBoost classifier.

    No scale_pos_weight -- preserves calibration potential.
    Modest depth + many rounds + early shrinkage = standard tabular setup.
    """
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",        # primary metric matches imbalanced setting
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def calibrate_xgboost(base_model: XGBClassifier,
                      X_calib: pd.DataFrame,
                      y_calib: pd.Series,
                      method: str = "sigmoid") -> CalibratedClassifierCV:
    """Wrap a prefit XGBoost with Platt (sigmoid) or isotonic calibration.

    Fits the calibrator on a held-out calibration set without retraining
    the base XGBoost. Compatible with both legacy sklearn (<1.6) using
    `cv='prefit'` and modern sklearn (>=1.6) using `FrozenEstimator`.

    Default 'sigmoid' (Platt): monotonic transform, preserves the model's
    ranking (so PR-AUC and ROC-AUC are unchanged), and smooths
    probabilities for the decision-rule's expected-value calc downstream.

    Isotonic is the alternative -- non-parametric and more flexible, but
    produces piecewise-constant probabilities which create ties and can
    measurably hurt PR-AUC on small positive classes.
    """
    try:
        # sklearn >= 1.6
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method=method,
        )
    except ImportError:
        # sklearn < 1.6 (legacy prefit pattern)
        calibrated = CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv="prefit",
        )
    calibrated.fit(X_calib, y_calib)
    return calibrated
