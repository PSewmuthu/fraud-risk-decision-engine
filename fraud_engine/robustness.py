"""Stress tests (distribution shift, missingness, extremes) and drift metrics (PSI)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from .config import SEED, MISSING_NUMERIC
from .costs import three_way_cost


def psi(expected, actual, bins=10):
    """Population Stability Index with quantile bins from the reference distribution."""
    e, a = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    if e.nunique() <= 1:
        return 0.0
    if e.nunique() <= bins:
        cats = sorted(set(e.unique()) | set(a.unique()))
        pe = e.value_counts(normalize=True).reindex(cats).fillna(0) + 1e-4
        pa = a.value_counts(normalize=True).reindex(cats).fillna(0) + 1e-4
    else:
        edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        pe = pd.Series(np.histogram(e, edges)[0] / len(e)) + 1e-4
        pa = pd.Series(np.histogram(a, edges)[0] / len(a)) + 1e-4
    return float(((pa - pe) * np.log(pa / pe)).sum())


def psi_table(ref, cur, cols):
    rows = []
    for c in cols:
        v = psi(ref[c], cur[c]) if pd.api.types.is_numeric_dtype(
            ref[c]) else psi(ref[c].fillna("Missing"), cur[c].fillna("Missing"))
        rows.append({"feature": c, "psi": v})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


# ---------------------------------------------------------------- shifts
def shift_amount(X, f):
    X = X.copy()
    X["transaction_amount"] = X["transaction_amount"] * f
    return X


def shift_velocity(X, add1=1, add24=3):
    X = X.copy()
    X["transactions_last_1h"] += add1
    X["transactions_last_24h"] += add24
    return X


def shift_risk(X, d=0.1):
    X = X.copy()
    for c in ["merchant_risk_score", "ip_risk_score"]:
        X[c] = (X[c] + d).clip(0, 1)
    return X


def shift_channel_mix(X, y, target_share=0.45, seed=SEED):
    """Re-sample rows so E-commerce is `target_share` of volume (channel-mix shift; base rate may move)."""
    rng = np.random.default_rng(seed)
    ec = (X["channel"] == "E-commerce").values
    n = len(X)
    idx = np.concatenate([rng.choice(np.where(ec)[0], int(n * target_share)),
                          rng.choice(np.where(~ec)[0], n - int(n * target_share))])
    return X.iloc[idx].reset_index(drop=True), y[idx]


def inject_missing(X, rate, seed=SEED):
    rng = np.random.default_rng(seed)
    X = X.copy()
    for c in MISSING_NUMERIC:
        X.loc[rng.random(len(X)) < rate, c] = np.nan
    return X


def inject_extremes(X, frac=0.02, seed=SEED):
    rng = np.random.default_rng(seed)
    X = X.copy()
    m = rng.random(len(X)) < frac
    X.loc[m, "transaction_amount"] = X.loc[m,
                                           "transaction_amount"].fillna(100) * 10
    X.loc[m, "distance_from_usual_location_km"] = 2500.0
    X.loc[m, "transactions_last_24h"] = X.loc[m, "transactions_last_24h"] + 120
    return X


def evaluate_scenario(model, X, y, amount_fill, thr_review, thr_block, sc, label):
    p = model.predict_proba(X)[:, 1]
    amt = X["transaction_amount"].fillna(amount_fill).values
    d = three_way_cost(p, y, amt, thr_review, thr_block, sc, detail=True)
    return {"scenario": label, "n": len(y), "fraud_rate": float(np.mean(y)), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "mean_score": float(p.mean()), "review_rate": d["review_rate"], "block_rate": d["block_rate"],
            "fraud_capture": d["fraud_capture_rate"], "cost_per_1000_txn": d["cost"] / len(y) * 1000}, p


def stress_suite(model, X, y, amount_fill, thr_review, thr_block, sc):
    rows = []

    def add(label, Xs, ys=None):
        r, _ = evaluate_scenario(
            model, Xs, y if ys is None else ys, amount_fill, thr_review, thr_block, sc, label)
        rows.append(r)
    add("Baseline (test)", X)
    add("Amounts x1.5", shift_amount(X, 1.5))
    add("Amounts x2", shift_amount(X, 2.0))
    add("Velocity +1 (1h) / +3 (24h)", shift_velocity(X))
    add("Merchant & IP risk +0.10", shift_risk(X, 0.10))
    Xc, yc = shift_channel_mix(X, y, 0.45)
    add("E-commerce share 45%", Xc, yc)
    Xc, yc = shift_channel_mix(X, y, 0.10)
    add("E-commerce share 10%", Xc, yc)
    for r in (0.05, 0.10, 0.20, 0.40):
        add(f"Extra missing values {int(r * 100)}%", inject_missing(X, r))
    add("2% extreme observations", inject_extremes(X, 0.02))
    add("5% extreme observations", inject_extremes(X, 0.05))
    return pd.DataFrame(rows)
