"""Data forensics: schema, missingness, duplicates, impossible values, outliers, leakage register."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

from .config import TARGET, ID_COL, MISSING_NUMERIC, SEED


def schema_table(df):
    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s)
        rows.append({"column": c, "dtype": str(s.dtype), "n_unique": s.nunique(),
                     "missing_n": int(s.isna().sum()), "missing_pct": round(100 * s.isna().mean(), 2),
                     "min": s.min() if num else None, "median": s.median() if num else None,
                     "max": s.max() if num else None,
                     "skew": round(float(s.skew()), 2) if num else None,
                     "top_value": s.value_counts().index[0] if not num else None})
    return pd.DataFrame(rows)


def duplicates(df):
    return {"full_row_duplicates": int(df.duplicated().sum()),
            "duplicate_ids": int(df[ID_COL].duplicated().sum()),
            "duplicates_ignoring_id": int(df.drop(columns=[ID_COL]).duplicated().sum())}


def impossible_values(df):
    chk = {
        "hour outside 0-23": (~df.hour.between(0, 23)).sum(),
        "day_of_week outside 0-6": (~df.day_of_week.between(0, 6)).sum(),
        "non-positive amount": (df.transaction_amount <= 0).sum(),
        "negative distance": (df.distance_from_usual_location_km < 0).sum(),
        "transactions_last_1h > transactions_last_24h": (df.transactions_last_1h > df.transactions_last_24h).sum(),
        "risk score outside [0,1]": ((df.merchant_risk_score < 0) | (df.merchant_risk_score > 1) | (df.ip_risk_score < 0) | (df.ip_risk_score > 1)).sum(),
        "account_age_days <= 0": (df.account_age_days <= 0).sum(),
        "ATM channel with card not present (logical inconsistency)": ((df.channel == "ATM") & (df.card_presence == "Not present")).sum(),
        "device older than account (possible device re-use)": (df.device_age_days > df.account_age_days).sum(),
        "login_to_transaction_minutes at 120 (possible cap / censoring)": (df.login_to_transaction_minutes == 120).sum(),
        "binary flags not in {0,1}": (~df.new_device.isin([0, 1]) | ~df.billing_address_mismatch.isin([0, 1]) | ~df.fraud.isin([0, 1])).sum(),
    }
    return pd.Series({k: int(v) for k, v in chk.items()}, name="count").to_frame()


def outlier_table(df, cols=None):
    cols = cols or ["transaction_amount", "distance_from_usual_location_km", "transactions_last_24h",
                    "transactions_last_1h", "login_to_transaction_minutes", "account_age_days"]
    rows = []
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        hi = q3 + 3 * (q3 - q1)
        out = df[c] > hi
        rows.append({"column": c, "p99": s.quantile(.99), "max": s.max(), "extreme_threshold_3IQR": hi,
                     "n_extreme": int(out.sum()), "fraud_rate_extreme": df.loc[out, TARGET].mean() if out.any() else np.nan,
                     "fraud_rate_other": df.loc[~out & df[c].notna(), TARGET].mean()})
    return pd.DataFrame(rows)


def missingness_signal(df):
    """Does missingness carry information about fraud, and what drives it?"""
    base = df.drop(columns=[ID_COL])
    rows = []
    cols = MISSING_NUMERIC + ["authentication_level"]
    for c in cols:
        m = df[c].isna()
        tab = [[int((m & (df[TARGET] == 1)).sum()), int((m & (df[TARGET] == 0)).sum())],
               [int((~m & (df[TARGET] == 1)).sum()), int((~m & (df[TARGET] == 0)).sum())]]
        _, p_fisher = fisher_exact(tab)
        # can the missing flag be predicted from the other observed columns? (MCAR check, AUC ~0.5 => MCAR-like)
        other = base.drop(columns=[c, TARGET]).copy()
        for k in other.select_dtypes(exclude="number").columns:
            other[k] = other[k].astype("category").cat.codes
        pr = cross_val_predict(HistGradientBoostingClassifier(max_iter=60, max_depth=3, random_state=SEED),
                               other, m.astype(int), cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                               method="predict_proba")[:, 1]
        rows.append({"column": c, "n_missing": int(m.sum()), "pct_missing": round(100 * m.mean(), 2),
                     "fraud_rate_missing": df.loc[m, TARGET].mean(), "fraud_rate_observed": df.loc[~m, TARGET].mean(),
                     "fisher_p_value": p_fisher, "auc_predicting_missingness": roc_auc_score(m, pr)})
    return pd.DataFrame(rows)


LEAKAGE_REGISTER = pd.DataFrame([
    ("transaction_id", "Identifier (sequential). Could encode time/ordering or label-correlated batches.",
     "NOT permitted as a feature. Dropped."),
    ("transactions_last_1h / _24h", "Velocity counters. Leak if the window includes the transaction being scored or later events.",
     "Permitted if computed strictly before the transaction timestamp. Must be verified with data engineering."),
    ("prior_chargebacks", "Chargebacks are reported weeks after the fraud happens. Counts that include chargebacks reported after the scoring time leak the future.",
     "Permitted only as a point-in-time count (chargebacks already reported at scoring time)."),
    ("merchant_risk_score / ip_risk_score", "If merchant score is refreshed using confirmed-fraud labels that include the present period it leaks the target.",
     "Permitted if the score is versioned and as-of the transaction time (vendor score or lagged rate)."),
    ("authentication_level", "May be the outcome of a risk-based step-up triggered by a pre-existing risk decision, or only recorded for completed transactions. Missing in 9.6% of rows with a higher fraud rate (19.0% vs 11.4%).",
     "Permitted only if the field is known before the decision. Flagged as the highest leakage risk; ablation reported."),
    ("login_to_transaction_minutes",
     "Known at scoring time (session telemetry).", "Permitted."),
    ("failed_logins_24h",
     "Known at scoring time if the window ends at the transaction.", "Permitted."),
    ("account_age_days / device_age_days / email_age_days",
     "Known at scoring time. Must be computed relative to the transaction date, not the extraction date.", "Permitted (verify reference date)."),
    ("hour / day_of_week", "Known at scoring time. No full timestamp exists, so a true out-of-time split is impossible.",
     "Permitted. Temporal leakage cannot be tested; see limitation."),
    ("fraud (target)", "Label is confirmed later (chargeback / investigation).",
     "Target only. Labels arrive with delay: monitoring must handle label lag."),
], columns=["field", "leakage risk", "decision"])


TREATMENT_LOG = pd.DataFrame([
    ("Missing numeric values (amount, distance, merchant risk, IP risk, login gap)", "1.5-2.5% each", "Median imputation inside the pipeline (fit on training folds only). Explicit <col>_missing flags were tested and NOT adopted.",
     "No signal after Bonferroni correction (6 tests, alpha 0.0083; smallest p = 0.028) and no CV gain. Rows are never dropped."),
    ("Missing authentication_level", "9.6%", "Kept as its own category 'Missing'.",
     "Fraud rate is 19.0% vs 11.4% when observed (Fisher p < 0.001): the only missingness that is informative."),
    ("Heavy right tails (amount, distance, 24h velocity)", "max amount 5,538 vs median 88",
     "No rows removed or capped. Log transforms and tree models handle tails; stress test adds extremes.", "Instructions forbid removing difficult observations."),
    ("ATM + card not present", "121 rows", "Kept; flagged with atm_cnp_inconsistent.",
     "Logical inconsistency in source data; report to data owner."),
    ("login_to_transaction_minutes = 120", "6 rows",
     "Kept (possible cap at 120).", "Immaterial."),
    ("Duplicates", "0", "None needed.",
     "Checked full rows, ids, and rows ignoring id."),
    ("transaction_id", "unique", "Dropped from the model; retained in the processed file for traceability.",
     "Leakage / no generalisable signal."),
    ("Categorical variables", "2-10 levels", "One-hot encoding (unknown levels ignored).",
     "Low cardinality: no target encoding needed."),
], columns=["issue", "scale", "treatment", "rationale"])
