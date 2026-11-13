"""Operational decision engine: calibrated probability -> Approve / Review / Block + explanation."""
import json
import joblib
import numpy as np
import pandas as pd

from .explain import shap_values, reason_codes, display_frame


class FraudDecisionEngine:
    """Wraps the calibrated model, the cost-derived thresholds and the explainer.

    Parameters
    ----------
    calibrated_model : fitted CalibratedClassifierCV (04_calibrated_model.joblib)
    base_pipeline    : fitted uncalibrated pipeline used for SHAP (03_fraud_model_pipeline.joblib)
    policy           : dict with ``t_review`` and ``t_block``
    background       : small raw-feature DataFrame used as SHAP background
    """

    def __init__(self, calibrated_model, base_pipeline, policy, background):
        self.model, self.base, self.policy, self.background = calibrated_model, base_pipeline, policy, background

    @classmethod
    def load(cls, model_path, base_path, policy_path, background):
        with open(policy_path) as f:
            policy = json.load(f)
        return cls(joblib.load(model_path), joblib.load(base_path), policy, background)

    def decide(self, X, explain=True):
        p = self.model.predict_proba(X)[:, 1]
        t_r, t_b = self.policy["t_review"], self.policy["t_block"]
        decision = np.where(p >= t_b, "Block", np.where(
            p >= t_r, "Review", "Approve"))
        out = pd.DataFrame(
            {"fraud_probability": p, "decision": decision}, index=X.index)
        out["explanation"] = ""
        if explain and (decision != "Approve").any():
            idx = np.where(decision != "Approve")[0]
            sv, Z, _ = shap_values(
                self.base, X.iloc[idx], background=self.background)
            Zd = display_frame(self.base, Z)
            out.iloc[idx, out.columns.get_loc("explanation")] = [
                reason_codes(sv[i], Zd.iloc[i]) for i in range(len(idx))]
        return out
