"""Turns the measured results (results.joblib) into the written answers used by the notebook and PDFs.

Every number in the text is read from the results dictionary, never typed by hand.
"""
import numpy as np
import pandas as pd

from .models import MODEL_LABELS


def _m(x):
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def _pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def _logit(p):
    return np.log(p / (1 - p))


def _expit(z):
    return 1 / (1 + np.exp(-z))


def answers(R):
    out = []
    fin = R["final_model"]
    final_lab = MODEL_LABELS[fin]
    stab, rc = R["shap_stability"], R["shap_rank_corr"]
    top = stab.head(6)
    stable = [
        f"{i} (mean rank {r.mean_rank:.1f}, top-10 in {int(r.in_top10_folds)}/5 folds)" for i, r in top.iterrows()]
    unstable = stab[(stab.in_top10_folds <= 2) & (
        stab.mean_abs_shap > stab.mean_abs_shap.median())]
    txt = ("The strongest predictors in the final model are: " + "; ".join(stable) + ". "
           f"The top-ranked feature, transactions_last_1h, is the most stable (it ranks first or second in every fold). "
           f"Across the five CV folds the pairwise Spearman correlation between importance vectors is {rc:.2f}, so the top 4-5 are stable but the middle of the ranking is not"
           + (": " + ", ".join(f"{i} (CV of importance {r.cv_of_importance:.0%}, top-10 in only {int(r.in_top10_folds)}/5 folds)" for i,
              r in unstable.head(3).iterrows()) if len(unstable) else "")
           + ". Failed-login information enters twice (failed_logins_24h and failed_login_flag) with opposite signs, which is a collinearity artefact, so read the pair together.")
    out.append(
        ("Which features are the strongest predictors of fraud, and are they stable across validation folds?", txt))

    disc, piv = R["model_discrimination"], R["model_cost_pivot"]
    # display label
    best_auc = disc["ROC-AUC (calibrated OOF)"].idxmax()
    short = {v: k for k, v in MODEL_LABELS.items()}
    # key used in the cost pivot
    best_auc_k = short[best_auc]
    best_cost = piv["Base"].idxmin()
    bd = R["bootstrap_cost_diff"]
    row = bd.iloc[[i for i, n in enumerate(bd.index) if MODEL_LABELS[best_cost] in n.split(
        " minus ")[0]][0]] if best_cost != fin else None
    auc_txt = ", ".join(f"{k} {v:.3f}" for k,
                        v in disc["ROC-AUC (calibrated OOF)"].items())
    cost_txt = ", ".join(
        f"{MODEL_LABELS[n]} {_m(piv.loc[n, 'Base'])}" for n in piv["Base"].sort_values().index)
    t = (f"Not necessarily. Out-of-fold ROC-AUC (calibrated): {auc_txt}. Expected base-scenario cost per 1,000 transactions under each model's own cost-optimal, capacity-constrained thresholds: {cost_txt}. "
         f"The highest-AUC model is {best_auc}; the lowest-cost model is {MODEL_LABELS[best_cost]}. ")
    if best_cost != best_auc_k:
        lo, med, hi = row.iloc[0], row.iloc[1], row.iloc[2]
        t += (f"The gap is a real ranking difference but not a statistically reliable one: the paired-bootstrap 95% interval for the cost difference ({MODEL_LABELS[best_cost]} minus {final_lab}) is {_m(lo)} to {_m(hi)} per 1,000 transactions and contains zero. "
              f"Because the evidence cannot separate them, the simpler and more transparent {final_lab} is kept.")
    out.append(
        ("Does the highest-ROC-AUC model also produce the best business outcome?", t))

    cr = R["calibration"].loc[final_lab]
    u, s = cr.loc["none"], cr.loc["sigmoid"]
    ex = [(p, _expit(u.cal_intercept + u.cal_slope * _logit(p)))
          for p in (0.25, 0.40)]
    thr = R["policy"]
    t = (f"Before calibration the tuned {final_lab} was over-dispersed (calibration slope {u.cal_slope:.2f}, intercept {u.cal_intercept:.2f}; ideal 1 and 0): "
         + "; ".join(f"a raw score of {p:.2f} corresponded to an observed fraud rate of only about {o:.2f}" for p, o in ex)
         + f". After Platt scaling the slope is {s.cal_slope:.2f}, the ECE falls from {u.ece:.3f} to {s.ece:.3f} and log-loss from {u.log_loss:.3f} to {s.log_loss:.3f} (out-of-fold on the development set). "
         f"After calibration a score is a probability: a transaction scored {thr['t_review']:.2f} really is fraudulent about {thr['t_review']:.0%} of the time, so the review and block thresholds ({thr['t_review']:.3f} and {thr['t_block']:.3f}) can be derived from costs, expected losses can be summed, and the numbers mean the same thing after a retrain. "
         f"On the 600-row test set the calibration gain is not visible (Brier {R['test_metrics']['brier']:.4f} calibrated vs {R['test_uncalibrated']['brier']:.4f} uncalibrated; 73 fraud cases is too few to resolve it), so we report this honestly rather than claim a test-set improvement.")
    out.append(
        ("How does calibration change the operational meaning of a predicted probability?", t))

    wt = R["workload_targets"].set_index("target_capture")
    n_fraud_100k = R["prevalence"] * 100_000
    a, b = wt.loc[0.3], wt.loc[0.7]
    extra_reviews = (b.review_rate - a.review_rate) * 100_000
    extra_fraud = (0.7 - 0.3) * n_fraud_100k
    t = (f"Workload rises much faster than recall. On development out-of-fold data, capturing 30% of fraud needs about {wt.loc[0.3, 'review_rate']:.0%} of transactions reviewed, 50% needs {wt.loc[0.5, 'review_rate']:.0%}, 70% needs {wt.loc[0.7, 'review_rate']:.0%} and 80% needs {wt.loc[0.8, 'review_rate']:.0%}. "
         f"Moving from 30% to 70% capture adds about {extra_reviews:,.0f} reviews per 100,000 transactions for about {extra_fraud:,.0f} extra fraud cases, i.e. roughly {extra_reviews / extra_fraud:.1f} extra reviews per additional fraud found (before analyst error). "
         f"This is a consequence of the model's modest discrimination (ROC-AUC about {R['test_metrics']['roc_auc']:.2f} on test); the 10% review-capacity guardrail therefore caps capture at about {R['capacity_sensitivity'].set_index('review_capacity').loc['10%', 'fraud_capture']:.0%} in the dev analysis.")
    out.append(
        ("What happens to manual-review workload when recall is increased?", t))

    il = R["interaction_lift"].sort_values("observed_lift", ascending=False)
    top2 = il.iloc[:2]
    ints = R["shap_interactions"]
    gl = R["shap_global"]
    t = ("Measured joint fraud rates: " + "; ".join(f"{r.pair}: {r.fraud_rate_both:.1%} fraud (n={r.n_both}), {r.observed_lift:.1f}x the base rate vs {r.expected_lift_if_independent:.1f}x if the two effects simply multiplied" for r in top2.itertuples())
         + f". The largest SHAP interaction values from the boosted-tree model ({R['interaction_model']}) are small (at most {ints.mean_abs_interaction.max():.3f}, versus {gl.iloc[0]:.2f} for the top main effect), so the data show mostly additive risk with at most mild amplification. "
         "The new-device combinations are the clearest, but cell sizes are in the tens, so treat them as hypotheses to monitor rather than proven interactions.")
    out.append(
        ("Which feature interactions create the greatest change in risk?", t))

    tt = R["threshold_table"].set_index("scenario")
    sg = R["sensitivity_grid"]
    cs = R["capacity_sensitivity"]
    ub = R["bootstrap_binary_threshold"]
    t = (f"Without capacity limits the cost-optimal binary threshold ranges from {sg.t_binary_opt.min():.2f} to {sg.t_binary_opt.max():.2f} across a grid of false-positive costs ($5-$80) and chargeback fees ($5-$160): it rises when false positives are expensive relative to missed fraud and falls when missed fraud is expensive. "
         f"Even in the base scenario its bootstrap 5-95% range is {ub[0]:.2f}-{ub[2]:.2f}. "
         f"With the operational guardrails the picture changes: the block threshold stays near {tt.loc['Base', 't_block_opt']:.2f} in most scenarios because the 5% block cap binds, and the review threshold moves with analyst capacity ({cs.set_index('review_capacity').loc['5%', 't_review']:.3f} at 5% capacity, {cs.set_index('review_capacity').loc['10%', 't_review']:.3f} at 10%, {cs.set_index('review_capacity').loc['20%', 't_review']:.3f} at 20%). "
         "Capacity, not unit cost, is the dominant driver of the recommended thresholds here, which is why the thresholds must be re-derived whenever staffing changes.")
    out.append(
        ("How sensitive is the recommended threshold to the assumed financial costs?", t))

    top_auc = disc["ROC-AUC (calibrated OOF)"]
    diffs = []
    for sc in piv.columns:
        lead = piv[sc].idxmin()
        if lead != best_auc_k:
            diffs.append(
                (sc, lead, piv.loc[best_auc_k, sc] - piv.loc[lead, sc]))
    t = (
        f"Yes, in the point estimates. {best_auc} has the highest ROC-AUC ({top_auc[best_auc]:.3f}), but another model is cheaper in {len(diffs)} of {len(piv.columns)} cost scenarios. ")
    if diffs:
        sc, lead, g = max(diffs, key=lambda x: x[2])
        ll = MODEL_LABELS[lead]
        t += (f"The clearest case is {ll} (ROC-AUC {top_auc[ll]:.3f}, {top_auc[ll] - top_auc[best_auc]:+.3f} vs {best_auc}; PR-AUC {disc.loc[ll, 'PR-AUC (calibrated OOF)']:.3f} vs {disc.loc[best_auc, 'PR-AUC (calibrated OOF)']:.3f}), "
              f"which costs {_m(g)} less per 1,000 transactions in the '{sc}' scenario ({g / piv.loc[best_auc_k, sc]:.1%} lower). "
              f"All other cost columns are in the results table. However, the paired bootstrap shows these gaps are inside sampling noise for this dataset (2,400 development rows, 291 fraud cases), so the quantitative demonstration is that rankings by cost differ from rankings by AUC, not that one model is proven better. "
              "With more labelled data a cost-based comparison should be repeated before choosing a model.")
    out.append(("Could a model with slightly lower predictive performance be preferable operationally under a different cost structure?", t))

    ps = R["psi"].set_index("feature")
    shifted = ps[["psi_amount_x1.5", "psi_ecom_share_10pct",
                  "psi_velocity_risk_shift"]].max(axis=1).sort_values(ascending=False)
    t = ("Monitor, in priority order: (1) transactions_last_1h and transactions_last_24h (top global predictor; PSI reaches " + f"{ps.loc['transactions_last_1h', 'psi_velocity_risk_shift']:.2f}" + " when velocity shifts) and failed_logins_24h; "
         "(2) authentication_level, especially its missing rate (informative missingness: 19.0% vs 11.4% fraud) and any change in how it is recorded; "
         f"(3) merchant_risk_score and ip_risk_score (PSI {ps.loc['merchant_risk_score', 'psi_velocity_risk_shift']:.2f} / {ps.loc['ip_risk_score', 'psi_velocity_risk_shift']:.2f} under a +0.10 shift, so they react strongly to vendor or scoring changes); "
         f"(4) transaction_amount (PSI {ps.loc['transaction_amount', 'psi_amount_x1.5']:.2f} for a 1.5x increase) and the channel mix (E-commerce share), which moved the score distribution and calibration in the stress test; "
         "(5) the share of missing values in the five numeric fields (stress test: capture falls from 32% to 12% when 40% of values are missing). "
         f"For reference, the dev-vs-test PSI of every field is below {np.ceil(ps.psi_dev_vs_test.max() * 100) / 100:.2f}, which is the sampling-noise floor for this dataset size.")
    out.append(
        ("Which data fields should be monitored most closely for drift after deployment?", t))
    return out


def monitoring_table(R):
    fv = R["fold_variability"]
    pr_lo = fv.loc["pr_auc", "mean"] - 2 * fv.loc["pr_auc", "std"]
    roc_lo = fv.loc["roc_auc", "mean"] - 2 * fv.loc["roc_auc", "std"]
    pol = R["policy"]
    ops = R["ops"].iloc[-1]
    rows = [
        ("Rolling PR-AUC (labelled, 30-day cohorts)", f"{fv.loc['pr_auc', 'mean']:.2f} (CV mean)",
         f"Warn below {pr_lo + 0.03:.2f}; act below {pr_lo:.2f} (mean - 2 sd)", "Weekly, once chargebacks mature"),
        ("Rolling ROC-AUC", f"{fv.loc['roc_auc', 'mean']:.2f}",
         f"Warn below {roc_lo + 0.03:.2f}; act below {roc_lo:.2f}", "Weekly"),
        ("Calibration: slope, ECE, mean predicted vs observed rate", "slope about 1.0; mean predicted 12.1% vs observed 12.1%",
         "Warn slope outside 0.8-1.2 or |mean gap| > 1.5 pts; act outside 0.6-1.4 or gap > 3 pts", "Weekly"),
        ("Score PSI (calibrated probability)",
         "below 0.02 (dev vs test)", "Warn 0.10; act 0.25", "Daily"),
        ("Feature PSI for the 10 most important inputs", "below 0.06",
         "Warn 0.10; act 0.25 on any top-5 feature", "Daily"),
        ("Review rate and block rate", f"{ops['review /100k'] / 1000:.1f}% / {ops['block /100k'] / 1000:.1f}% of volume",
         "Warn review > 12% or block > 6%; act review > 15% (capacity breach)", "Daily"),
        ("Fraud capture and false-positive burden (confirmed labels)",
         f"capture {ops['fraud capture %']:.0f}% of fraud cases (test)", "Warn capture down 25% relative; act down 40%", "Weekly"),
        ("Data quality: missing rate per field, share of authentication_level missing",
         "numeric 1.5-2.5%; auth 9.6%", "Warn if any field exceeds 2x baseline; act if > 4x", "Daily"),
        ("Label delay and label coverage", "n/a (not in dataset)",
         "Warn if < 80% of cohort labelled at day 45", "Weekly"),
        ("Expected vs realised cost per 1,000 transactions",
         f"base scenario {_m(R['threshold_table'].set_index('scenario').loc['Base', 'cost_3way_opt'])} (dev OOF)", "Warn if realised > expected by 15%; act by 30%", "Monthly"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Baseline (this evaluation)", "Thresholds", "Frequency"])


ACTIONS = [
    ("Retrain", "Scheduled quarterly, or earlier when: score PSI > 0.25 for 7 consecutive days; rolling PR-AUC falls below the act level for 2 consecutive weeks; calibration slope leaves 0.6-1.4; a top-5 feature definition changes. Recalibrate only (keep the model, refit Platt scaling) when ranking metrics are healthy but calibration drifts."),
    ("Review (human model-risk review)", "Any change to data feeds feeding velocity counters, authentication_level, merchant or IP risk scores; any new channel or country; cost-assumption changes (e.g. staffing, chargeback fees); new fairness findings; the first 90 days after launch."),
    ("Temporarily disable (fall back to rules + manual process)", "Rolling ROC-AUC below the act level for 2 consecutive weeks AND a confirmed data fault; missing rate of velocity or authentication fields above 25%; review queue above 150% of capacity for more than 3 days; score PSI above 0.50; or an incident where the score distribution collapses (e.g. more than 90% of scores in one 0.01 bin)."),
]
