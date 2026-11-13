"""Feature engineering as a scikit-learn transformer.

Everything that *learns* from data (group medians, quantile cut-offs, z-score
statistics) is learned in ``fit`` on training data only, so the transformer is
safe inside cross-validation. No target information is ever used.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import MISSING_NUMERIC, HOME_COUNTRY

# name -> (description, availability at prediction time)
FEATURE_DOCS = {
    "velocity_ratio_1h_24h": ("transactions_last_1h / (transactions_last_24h + 1): share of the day's activity that is happening right now (burst detection).", "Available if the velocity counters exclude the current transaction."),
    "burst_flag": ("1 if 4+ transactions in the last hour.", "Same as above."),
    "amount_log": ("log1p(transaction_amount): tames the heavy right tail (max about 5.5k vs median 88).", "Available."),
    "log_amount_to_category_median": ("log1p(amount / training median amount of the merchant category): is this purchase large for this kind of merchant?", "Available; medians fitted on training data only."),
    "amount_z_tier": ("z-score of log-amount within customer tier (training mean/std).", "Available; statistics fitted on training data only."),
    "amount_extreme_flag": ("1 if amount above the training 95th percentile.", "Available; cut-off fitted on training data only."),
    "risk_product": ("merchant_risk_score * ip_risk_score: both risk signals elevated together.", "Available if both scores are computed as-of the transaction."),
    "risk_max": ("max(merchant_risk_score, ip_risk_score).", "Same."),
    "risk_mean": ("mean of merchant and IP risk scores.", "Same."),
    "amount_x_merchant_risk": ("amount_log * merchant_risk_score: large purchases at risky merchants.", "Available."),
    "newdev_x_distance": ("new_device * log1p(distance): unfamiliar device used far from home.", "Available."),
    "strongauth_x_ecom": ("1 if E-commerce channel with Strong authentication.", "CAUTION: authentication_level may be recorded after a risk-based step-up decision; see leakage register."),
    "cnp_x_ecom": ("1 if card not present on the E-commerce channel.", "Available."),
    "is_night": ("1 if hour in 00:00-05:59.", "Available."),
    "is_weekend": ("1 if day_of_week in {5, 6} (assumes 0 = Monday).", "Available."),
    "hour_sin": ("sin(2*pi*hour/24): cyclical hour.", "Available."),
    "hour_cos": ("cos(2*pi*hour/24): cyclical hour.", "Available."),
    "distance_log": ("log1p(distance from usual location).", "Available."),
    "distance_z": ("z-score of log-distance (training mean/std): behavioural deviation from the population norm.", "Available; fitted on training data only."),
    "far_from_home_flag": ("1 if distance above the training 90th percentile.", "Available; cut-off fitted on training data only."),
    "login_quick_flag": ("1 if the transaction follows login by under 2 minutes (scripted / account-takeover pattern).", "Available."),
    "login_gap_log": ("log1p(login_to_transaction_minutes).", "Available."),
    "account_age_log": ("log1p(account_age_days).", "Available."),
    "new_account_flag": ("1 if account younger than 90 days.", "Available."),
    "log_device_to_account_age": ("log1p(device_age_days / (account_age_days + 1)); log because brand-new accounts push the raw ratio above 1,000.", "Available."),
    "device_older_than_account": ("1 if the device is older than the account (device re-use across accounts).", "Available."),
    "log_email_to_account_age": ("log1p(email_age_days / (account_age_days + 1)); log for the same reason.", "Available."),
    "failed_login_flag": ("1 if any failed login in the last 24h.", "Available."),
    "chargeback_flag": ("1 if the customer has any prior chargeback.", "Available only for chargebacks already reported before the transaction (labels lag)."),
    "foreign_country_flag": (f"1 if country is not {HOME_COUNTRY}.", "Available."),
    "atm_cnp_inconsistent": ("1 if channel is ATM but card_presence says 'Not present' (data inconsistency).", "Available."),
    "n_missing": ("number of missing values among the 6 incomplete fields.", "Available."),
}
MISSING_IND_DOCS = {
    f"{c}_missing": f"1 if {c} is missing (data-quality treatment, not engineered)." for c in MISSING_NUMERIC}
MISSING_IND_DOCS[
    "authentication_level"] = "missing authentication_level is kept as its own category 'Missing' (informative: 19.0% vs 11.4% fraud)."


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean raw columns, add missing indicators and (optionally) engineered features.

    Parameters
    ----------
    engineer : bool
        Add the derived features. False gives the raw-feature baseline.
    missing_indicators : bool
        Add ``<col>_missing`` flags for the five numeric columns with gaps. Missing
        ``authentication_level`` is always kept as its own category 'Missing'.
        Default False: the flags showed no signal after multiple-testing correction
        and no CV gain (see notebook sections A and C).
    """

    def __init__(self, engineer=True, missing_indicators=False):
        self.engineer = engineer
        self.missing_indicators = missing_indicators

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        amt = X["transaction_amount"]
        self.cat_median_ = amt.groupby(
            X["merchant_category"]).median().to_dict()
        self.global_median_amt_ = float(amt.median())
        la = np.log1p(amt)
        g = la.groupby(X["customer_tier"])
        self.tier_mean_ = g.mean().to_dict()
        self.tier_std_ = g.std().replace(0, 1).to_dict()
        self.global_mean_la_ = float(la.mean())
        self.global_std_la_ = float(la.std())
        self.amt_p95_ = float(amt.quantile(0.95))
        ld = np.log1p(X["distance_from_usual_location_km"])
        self.ld_mean_, self.ld_std_ = float(ld.mean()), float(ld.std())
        self.dist_p90_ = float(
            X["distance_from_usual_location_km"].quantile(0.90))
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = X.drop(
            columns=[c for c in ["transaction_id", "fraud"] if c in X.columns])
        miss = out[MISSING_NUMERIC].isna()
        auth_missing = out["authentication_level"].isna()
        if self.missing_indicators:
            for c in MISSING_NUMERIC:
                out[f"{c}_missing"] = miss[c].astype(int)
        out["authentication_level"] = out["authentication_level"].fillna(
            "Missing").astype(str)
        if not self.engineer:
            return out

        amt, dist = out["transaction_amount"], out["distance_from_usual_location_km"]
        t1, t24 = out["transactions_last_1h"], out["transactions_last_24h"]
        acc, dev, em = out["account_age_days"], out["device_age_days"], out["email_age_days"]
        mr, ipr = out["merchant_risk_score"], out["ip_risk_score"]
        la = np.log1p(amt)
        ecom = out["channel"].eq("E-commerce")
        new = pd.DataFrame(index=out.index)
        new["velocity_ratio_1h_24h"] = t1 / (t24 + 1)
        new["burst_flag"] = (t1 >= 4).astype(int)
        new["amount_log"] = la
        med = out["merchant_category"].map(
            self.cat_median_).fillna(self.global_median_amt_)
        new["log_amount_to_category_median"] = np.log1p(amt / med)
        tm = out["customer_tier"].map(
            self.tier_mean_).fillna(self.global_mean_la_)
        ts = out["customer_tier"].map(
            self.tier_std_).fillna(self.global_std_la_)
        new["amount_z_tier"] = (la - tm) / ts
        new["amount_extreme_flag"] = (
            amt > self.amt_p95_).astype(int).where(amt.notna())
        new["risk_product"] = mr * ipr
        new["risk_max"] = np.fmax(mr, ipr)
        new["risk_mean"] = (mr + ipr) / 2
        new["amount_x_merchant_risk"] = la * mr
        new["newdev_x_distance"] = out["new_device"] * np.log1p(dist)
        new["strongauth_x_ecom"] = (
            ecom & out["authentication_level"].eq("Strong")).astype(int)
        new["cnp_x_ecom"] = (
            ecom & out["card_presence"].eq("Not present")).astype(int)
        h = out["hour"]
        new["is_night"] = (h < 6).astype(int)
        new["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
        new["hour_sin"] = np.sin(2 * np.pi * h / 24)
        new["hour_cos"] = np.cos(2 * np.pi * h / 24)
        ld = np.log1p(dist)
        new["distance_log"] = ld
        new["distance_z"] = (ld - self.ld_mean_) / self.ld_std_
        new["far_from_home_flag"] = (
            dist > self.dist_p90_).astype(int).where(dist.notna())
        lg = out["login_to_transaction_minutes"]
        new["login_quick_flag"] = (lg < 2).astype(int).where(lg.notna())
        new["login_gap_log"] = np.log1p(lg)
        new["account_age_log"] = np.log1p(acc)
        new["new_account_flag"] = (acc < 90).astype(int)
        new["log_device_to_account_age"] = np.log1p(dev / (acc + 1))
        new["device_older_than_account"] = (dev > acc).astype(int)
        new["log_email_to_account_age"] = np.log1p(em / (acc + 1))
        new["failed_login_flag"] = (out["failed_logins_24h"] > 0).astype(int)
        new["chargeback_flag"] = (out["prior_chargebacks"] > 0).astype(int)
        new["foreign_country_flag"] = (
            out["country"] != HOME_COUNTRY).astype(int)
        new["atm_cnp_inconsistent"] = (out["channel"].eq(
            "ATM") & out["card_presence"].eq("Not present")).astype(int)
        new["n_missing"] = miss.sum(axis=1) + auth_missing.astype(int)
        return pd.concat([out, new], axis=1)
