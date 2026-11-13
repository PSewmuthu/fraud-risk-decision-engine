"""Probability calibration helpers (Platt scaling = sigmoid, isotonic regression)."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

from .config import SEED, N_SPLITS


def ece(y, p, n_bins=10):
    """Expected calibration error with equal-frequency bins."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) / len(p) * abs(y[b].mean() - p[b].mean()) for b in bins))


def calibration_slope_intercept(y, p):
    """Logistic recalibration: slope 1 / intercept 0 means perfectly calibrated."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def oof_calibrated(pipe, X, y, method, seed=SEED, inner_cv=3):
    """Out-of-fold calibrated probabilities. The calibrator is fitted inside each
    outer training fold (inner CV), never on the fold being scored."""
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(X, y):
        if method == "none":
            m = clone(pipe).fit(X.iloc[tr], y[tr])
        else:
            m = CalibratedClassifierCV(clone(pipe), method=method, cv=inner_cv).fit(X.iloc[tr], y[tr])
        p[va] = m.predict_proba(X.iloc[va])[:, 1]
    return p


def calibration_report(y, p):
    slope, intercept = calibration_slope_intercept(y, p)
    return {"brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
            "ece": ece(y, p), "cal_slope": slope, "cal_intercept": intercept,
            "mean_pred": float(np.mean(p)), "observed_rate": float(np.mean(y))}


def compare_methods(pipe, X, y, methods=("none", "sigmoid", "isotonic")):
    probs, rows = {}, []
    for m in methods:
        probs[m] = oof_calibrated(pipe, X, y, m)
        r = calibration_report(y, probs[m])
        r["method"] = m
        rows.append(r)
    return pd.DataFrame(rows).set_index("method"), probs


def reliability_points(y, p, n_bins=8):
    frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return mean, frac


def fit_calibrated(pipe, X, y, method, cv=5):
    return CalibratedClassifierCV(clone(pipe), method=method, cv=cv).fit(X, y)
