"""Exploratory analysis figures. Every function saves a PNG and returns its path."""
from .config import FIG, TARGET, MISSING_NUMERIC
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")


BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "axes.titlesize": 11, "axes.labelsize": 9,
                     "font.size": 9, "axes.titleweight": "bold"})


def _rate(g):
    n, k = len(g), int(g[TARGET].sum())
    lo, hi = proportion_confint(
        k, n, method="wilson") if n else (np.nan, np.nan)
    return pd.Series({"rate": k / n if n else np.nan, "lo": lo, "hi": hi, "n": n})


def rate_by(df, key):
    return df.groupby(key, observed=True).apply(_rate, include_groups=False)


def qbins(s, q=6, digits=1):
    return pd.qcut(s, q, duplicates="drop").astype(str) if False else pd.qcut(s, q, duplicates="drop")


def _bar(ax, tab, title, color=BLUE, base=None, rot=0, xlabel=""):
    x = np.arange(len(tab))
    ax.bar(x, tab["rate"] * 100, color=color, width=.65)
    ax.errorbar(x, tab["rate"] * 100, yerr=[(tab["rate"] - tab["lo"]) * 100, (tab["hi"] - tab["rate"]) * 100],
                fmt="none", ecolor="#444", lw=1, capsize=2)
    labels = [str(i) if not isinstance(i, pd.Interval)
              else f"{i.left:.3g}-{i.right:.3g}" for i in tab.index]
    ax.set_xticks(x, labels, rotation=rot, ha="right" if rot else "center")
    if base is not None:
        ax.axhline(base * 100, color=ORANGE, ls="--", lw=1.2)
    ax.set_title(title)
    ax.set_ylabel("fraud rate (%)")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0)


def _save(fig, name):
    p = FIG / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_class_balance_missing(df):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    vc = df[TARGET].value_counts().sort_index()
    ax[0].bar(["Legitimate", "Fraud"], vc.values,
              color=[BLUE, ORANGE], width=.55)
    for i, v in enumerate(vc.values):
        ax[0].text(i, v, f"{v:,} ({v / len(df):.1%})",
                   ha="center", va="bottom")
    ax[0].set_title(
        f"Class balance (1 fraud : {vc[0] / vc[1]:.1f} legitimate)")
    ax[0].set_ylim(0, vc.max() * 1.15)
    cols = MISSING_NUMERIC + ["authentication_level"]
    rows = []
    for c in cols:
        m = df[c].isna()
        rows.append((c.replace("distance_from_usual_location_km", "distance").replace("_score", "").replace("transaction_", "").replace("login_to_transaction_minutes", "login_gap"),
                     df.loc[m, TARGET].mean(), df.loc[~m, TARGET].mean(), int(m.sum())))
    x = np.arange(len(rows))
    ax[1].bar(x - .2, [r[1] * 100 for r in rows], .38,
              color=ORANGE, label="value missing")
    ax[1].bar(x + .2, [r[2] * 100 for r in rows], .38,
              color=BLUE, label="value observed")
    ax[1].set_xticks(x, [f"{r[0]}\n(n={r[3]})" for r in rows], fontsize=7.5)
    ax[1].set_ylabel("fraud rate (%)")
    ax[1].set_title("Fraud rate when a field is missing")
    ax[1].legend(frameon=False)
    ax[1].grid(axis="x", alpha=0)
    return _save(fig, "fig01_class_balance_missingness.png")


def fig_categoricals(df):
    base = df[TARGET].mean()
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2))
    spec = [("channel", "Channel"), ("merchant_category", "Merchant category"), ("authentication_level", "Authentication level"),
            ("customer_tier", "Customer tier"), ("device_type", "Device type"), ("country", "Country")]
    d = df.copy()
    d["authentication_level"] = d["authentication_level"].fillna("Missing")
    for ax, (c, t) in zip(axs.ravel(), spec):
        tab = rate_by(d, c).sort_values("rate", ascending=False)
        _bar(ax, tab, t, base=base, rot=35 if len(tab) > 4 else 0)
    return _save(fig, "fig02_fraud_rate_by_categories.png")


def fig_time(df):
    base = df[TARGET].mean()
    d = df.copy()
    d["period"] = pd.cut(d.hour, [-1, 5, 11, 17, 23],
                         labels=["00-05", "06-11", "12-17", "18-23"])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    _bar(ax[0], rate_by(d, "period"), "By 6-hour period",
         base=base, xlabel="hour of day")
    hr = rate_by(d, "hour")
    ax[1].plot(hr.index, hr["rate"] * 100,
               color=BLUE, lw=1.6, marker="o", ms=3)
    ax[1].fill_between(hr.index, hr["lo"] * 100, hr["hi"]
                       * 100, color=BLUE, alpha=.15)
    ax[1].axhline(base * 100, color=ORANGE, ls="--", lw=1.2)
    ax[1].set_title("By hour (95% Wilson band)")
    ax[1].set_xlabel("hour")
    ax[1].set_ylabel("fraud rate (%)")
    _bar(ax[2], rate_by(d, "day_of_week"),
         "By day of week (0-6)", base=base, xlabel="day_of_week")
    return _save(fig, "fig03_fraud_rate_by_time.png")


def fig_nonlinear(df, cols, title, name, q=6):
    base = df[TARGET].mean()
    fig, axs = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 3.5))
    axs = np.atleast_1d(axs)
    for ax, (c, lab) in zip(axs, cols):
        d = df[[c, TARGET]].dropna()
        d["bin"] = pd.qcut(d[c], q, duplicates="drop")
        _bar(ax, rate_by(d, "bin"), lab, base=base, rot=35)
    fig.suptitle(title, fontweight="bold", y=1.02)
    return _save(fig, name)


def fig_amount(df):
    base = df[TARGET].mean()
    d = df[["transaction_amount", TARGET]].dropna().copy()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    d["bin"] = pd.qcut(d.transaction_amount, 8, duplicates="drop")
    _bar(ax[0], rate_by(d, "bin"), "Amount octiles", base=base, rot=40)
    d["top"] = pd.cut(d.transaction_amount, [0, d.transaction_amount.quantile(.9), d.transaction_amount.quantile(.95),
                                             d.transaction_amount.quantile(.99), d.transaction_amount.max()],
                      labels=["<P90", "P90-95", "P95-99", ">P99"])
    _bar(ax[1], rate_by(d, "top"), "Tail of the amount distribution", base=base)
    for v, c, l in [(0, BLUE, "legitimate"), (1, ORANGE, "fraud")]:
        ax[2].hist(np.log10(d.loc[d[TARGET] == v, "transaction_amount"]),
                   bins=30, density=True, alpha=.55, color=c, label=l)
    ax[2].set_title("log10(amount) distribution")
    ax[2].legend(frameon=False)
    return _save(fig, "fig04_amount_nonlinear.png")


def _heat(ax, d, rowcol, colcol, title, rl, cl, min_n=15, vmax=None):
    tab = d.groupby([rowcol, colcol], observed=True)[
        TARGET].agg(["mean", "size"]).reset_index()
    piv = tab.pivot(index=rowcol, columns=colcol, values="mean") * 100
    cnt = tab.pivot(index=rowcol, columns=colcol, values="size")
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto",
                   vmin=0, vmax=vmax or np.nanmax(piv.values))
    ax.set_xticks(range(piv.shape[1]), [str(c)
                  for c in piv.columns], rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]), [str(c) for c in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v, n = piv.values[i, j], cnt.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}%\nn={int(n)}" if n >= min_n else f"n={int(n)}", ha="center", va="center",
                        fontsize=7.5, color="white" if v > np.nanmax(piv.values) * .6 and n >= min_n else "#222")
    ax.set_title(title)
    ax.set_xlabel(cl)
    ax.set_ylabel(rl)
    ax.grid(False)
    return im


def fig_interactions(df):
    out = []
    d = df.copy()
    d["distance bin"] = pd.qcut(
        d.distance_from_usual_location_km, 4, duplicates="drop").astype(str)
    d["new_device_lab"] = d.new_device.map(
        {0: "known device", 1: "new device"})
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    _heat(ax, d.dropna(subset=["distance_from_usual_location_km"]), "new_device_lab", "distance bin",
          "Interaction: new device x distance from usual location", "", "distance quartile (km)")
    out.append(_save(fig, "fig08_interaction_newdevice_distance.png"))

    d["amount bin"] = pd.qcut(d.transaction_amount, 4,
                              duplicates="drop").astype(str)
    d["merchant risk"] = pd.qcut(
        d.merchant_risk_score, 3, labels=["low", "mid", "high"])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    _heat(ax, d.dropna(subset=["transaction_amount", "merchant_risk_score"]), "merchant risk", "amount bin",
          "Interaction: amount x merchant risk", "merchant risk tercile", "amount quartile (USD)")
    out.append(_save(fig, "fig09_interaction_amount_merchantrisk.png"))

    d["auth"] = d.authentication_level.fillna("Missing")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    _heat(ax, d, "auth", "channel",
          "Interaction: authentication level x channel", "authentication", "channel")
    out.append(_save(fig, "fig10_interaction_auth_channel.png"))

    d["ip tercile"] = pd.qcut(d.ip_risk_score, 3, labels=[
                              "low IP risk", "mid", "high IP risk"])
    d["velocity"] = pd.cut(d.transactions_last_1h, [-1, 1, 3, 20],
                           labels=["0-1 /1h", "2-3 /1h", "4+ /1h"])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _heat(ax, d.dropna(subset=["ip_risk_score"]), "ip tercile", "velocity",
          "Interaction: IP risk x 1-hour velocity", "", "transactions in last hour")
    out.append(_save(fig, "fig11_interaction_iprisk_velocity.png"))
    return out


def fig_correlation(df):
    num = df.select_dtypes("number").drop(columns=["hour"])
    c = num.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-.5, vmax=.5)
    ax.set_xticks(range(len(c)), c.columns,
                  rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(len(c)), c.columns, fontsize=7)
    ax.grid(False)
    ax.set_title("Spearman correlation (numeric fields incl. target)")
    fig.colorbar(im, ax=ax, shrink=.7)
    return _save(fig, "fig12_correlation.png")


def run_all(df):
    p = [fig_class_balance_missing(df), fig_categoricals(
        df), fig_time(df), fig_amount(df)]
    p.append(fig_nonlinear(df, [("transactions_last_1h", "Transactions last 1h"), ("transactions_last_24h", "Transactions last 24h")],
                           "Velocity effects", "fig05_velocity.png", q=5))
    p.append(fig_nonlinear(df, [("distance_from_usual_location_km", "Distance from usual location (km)"),
                                ("login_to_transaction_minutes", "Login-to-transaction gap (min)")],
                           "Distance and session timing", "fig06_distance_login.png"))
    p.append(fig_nonlinear(df, [("account_age_days", "Account age (days)"), ("merchant_risk_score", "Merchant risk score"),
                                ("ip_risk_score", "IP risk score")], "Account age and risk scores", "fig07_age_risk.png"))
    p += fig_interactions(df)
    p.append(fig_correlation(df))
    return p
