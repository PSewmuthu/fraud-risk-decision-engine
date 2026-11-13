"""SHAP explanations and reason codes. Explanations describe the model, not causal effects."""
from .models import cv_splitter
from .config import FIG, SEED
from sklearn.base import clone
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")


BLUE, ORANGE = "#2a78d6", "#eb6834"


def transform_features(fitted_pipe, X):
    """Run everything before the final estimator (feature engineering + preprocessing)."""
    Z = X
    for name, step in fitted_pipe.steps[:-1]:
        if name == "sampler":
            continue
        Z = step.transform(Z)
    return Z


def shap_values(fitted_pipe, X, background=None):
    """Return (Explanation-like arrays, feature frame). Uses TreeExplainer when possible."""
    Z = transform_features(fitted_pipe, X)
    model = fitted_pipe.steps[-1][1]
    from sklearn.linear_model import LogisticRegression
    if isinstance(model, LogisticRegression):
        Zb = transform_features(
            fitted_pipe, background if background is not None else X)
        exp = shap.LinearExplainer(model, Zb)
        return np.asarray(exp.shap_values(Z)), Z, float(np.atleast_1d(exp.expected_value)[-1])
    try:
        exp = shap.TreeExplainer(model)
        sv = exp.shap_values(Z)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        base = exp.expected_value
        base = float(np.atleast_1d(base)[-1])
    except Exception:
        bg = transform_features(fitted_pipe, background if background is not None else X).sample(
            100, random_state=SEED)
        exp = shap.Explainer(lambda a: model.predict_proba(
            pd.DataFrame(a, columns=Z.columns))[:, 1], bg)
        e = exp(Z)
        sv, base = e.values, float(np.mean(e.base_values))
    return sv, Z, base


def display_frame(fitted_pipe, Z):
    """Undo standard scaling so that explanations show real values (e.g. 4 transactions, not +1.9 sd)."""
    prep = fitted_pipe.named_steps["prep"]
    scaler = prep.named_transformers_["num"].named_steps.get("scale")
    if scaler is None:
        return Z
    num_cols = list(prep.transformers_[0][2])
    out = Z.copy()
    out[num_cols] = scaler.inverse_transform(Z[num_cols])
    return out


def global_importance(sv, Z, top=20):
    imp = pd.Series(np.abs(sv).mean(
        0), index=Z.columns).sort_values(ascending=False)
    return imp.head(top)


def plot_global(sv, Z, name="fig13_shap_global.png", top=15):
    """Two figures: mean |SHAP| bar chart and the beeswarm. Returns both paths."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    imp = global_importance(sv, Z, top)[::-1]
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_title("Global importance: mean |SHAP| (log-odds)")
    fig.tight_layout()
    p1 = FIG / name
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    plt.figure()
    shap.summary_plot(sv, Z, max_display=top, show=False)
    plt.title("SHAP beeswarm (colour = feature value)")
    p2 = FIG / name.replace(".png", "_beeswarm.png")
    plt.savefig(p2, bbox_inches="tight", dpi=110)
    plt.close("all")
    return p1, p2


def plot_dependence(sv, Z, features, name="fig14_shap_dependence.png"):
    n = len(features)
    fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 3.6))
    axs = np.atleast_1d(axs)
    for ax, f in zip(axs, features):
        plt.sca(ax)
        shap.dependence_plot(f, sv, Z, ax=ax, show=False,
                             interaction_index="auto")
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    return FIG / name


def interaction_strengths(fitted_pipe, X, n=500, top=10):
    """Mean |SHAP interaction value| for feature pairs (tree models only)."""
    Z = transform_features(fitted_pipe, X).sample(
        min(n, len(X)), random_state=SEED)
    model = fitted_pipe.steps[-1][1]
    iv = shap.TreeExplainer(model).shap_interaction_values(Z)
    if isinstance(iv, list):
        iv = iv[1]
    iv = np.asarray(iv)
    if iv.ndim == 4:
        iv = iv[..., 1]
    m = np.abs(iv).mean(0)
    np.fill_diagonal(m, 0)
    rows = [(Z.columns[i], Z.columns[j], m[i, j] * 2)
            for i in range(len(m)) for j in range(i + 1, len(m))]
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "mean_abs_interaction"]).sort_values(
        "mean_abs_interaction", ascending=False).head(top).reset_index(drop=True)


def reason_codes(sv_row, Z_row, k=3):
    idx = np.argsort(-np.abs(sv_row))[:k]
    parts = []
    for i in idx:
        v = Z_row.iloc[i]
        vs = f"{v:.3g}" if isinstance(
            v, (int, float, np.floating, np.integer)) else str(v)
        parts.append(
            f"{Z_row.index[i]}={vs} ({'raises' if sv_row[i] > 0 else 'lowers'} risk, {sv_row[i]:+.2f})")
    return "; ".join(parts)


def explain_transactions(sv, Z, ids, proba, y, idxs, labels, base):
    rows = []
    for i, lab in zip(idxs, labels):
        rows.append({"case": lab, "transaction_id": ids[i], "calibrated_prob": float(proba[i]), "actual_fraud": int(y[i]),
                     "reason_codes": reason_codes(sv[i], Z.iloc[i])})
    return pd.DataFrame(rows)


def plot_waterfalls(sv, Z, idxs, labels, base, name="fig15_local_explanations.png", top=8):
    n = len(idxs)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(13, 3.3 * rows))
    axs = np.atleast_1d(axs).ravel()
    for ax, i, lab in zip(axs, idxs, labels):
        order = np.argsort(-np.abs(sv[i]))[:top][::-1]
        vals = sv[i][order]
        ax.barh([f"{Z.columns[j]}={Z.iloc[i, j]:.3g}" if np.issubdtype(type(Z.iloc[i, j]), np.number) else Z.columns[j] for j in order],
                vals, color=[ORANGE if v > 0 else BLUE for v in vals])
        ax.axvline(0, color="#333", lw=.8)
        ax.set_title(lab, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axs[n:]:
        ax.axis("off")
    fig.suptitle(
        "Local SHAP explanations (orange raises risk, blue lowers it; log-odds units)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    return FIG / name


def direction_check(sv, Z, X_raw, y_dev):
    """Flag counter-intuitive explanations: SHAP-vs-value slope disagreeing with the univariate fraud relationship."""
    rows = []
    for c in Z.columns:
        z = Z[c].values
        if np.nanstd(z) == 0:
            continue
        r_shap = spearmanr(
            z, sv[:, list(Z.columns).index(c)], nan_policy="omit")[0]
        r_uni = spearmanr(z, y_dev, nan_policy="omit")[0]
        rows.append({"feature": c, "spearman_value_vs_shap": r_shap, "spearman_value_vs_fraud": r_uni,
                     "mean_abs_shap": np.abs(sv[:, list(Z.columns).index(c)]).mean()})
    d = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    d["direction_conflict"] = (np.sign(d.spearman_value_vs_shap) != np.sign(d.spearman_value_vs_fraud)) & (
        d.spearman_value_vs_fraud.abs() > 0.02) & (d.spearman_value_vs_shap.abs() > 0.1)
    return d


def fold_stability(pipe, X, y, top=10):
    """Fit on each CV training fold and compute SHAP on the validation fold: how stable are importances?"""
    imps = {}
    for k, (tr, va) in enumerate(cv_splitter().split(X, y)):
        m = clone(pipe).fit(X.iloc[tr], y[tr])
        sv, Z, _ = shap_values(m, X.iloc[va], background=X.iloc[tr])
        imps[f"fold{k + 1}"] = pd.Series(np.abs(sv).mean(0), index=Z.columns)
    df = pd.DataFrame(imps)
    ranks = df.rank(ascending=False)
    cons = ranks.mean(axis=1).sort_values()
    summary = pd.DataFrame({"mean_abs_shap": df.mean(axis=1), "cv_of_importance": df.std(axis=1) / df.mean(axis=1),
                            "mean_rank": ranks.mean(axis=1), "rank_std": ranks.std(axis=1),
                            "in_top10_folds": (ranks <= top).sum(axis=1)}).loc[cons.index]
    corrs = [spearmanr(df.iloc[:, i], df.iloc[:, j])[0]
             for i in range(df.shape[1]) for j in range(i + 1, df.shape[1])]
    return summary, df, float(np.mean(corrs))
