# Fraud-Risk Intelligence Pipeline

Calibrated fraud probability + Approve / Review / Block decision + reason codes, for `transaction_fraud_risk_advanced_dataset.csv` (3,000 rows, 12.1 % fraud).

## Deliverables (this folder)

| File                          | Content                                                                                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fraud_risk_analysis.ipynb`   | Complete executed notebook, sections A–K plus answers to the guiding questions                                                                                      |
| `processed_dataset.csv`       | Engineered feature table (all rows, `split` column = train/test, target `fraud`); transformer fitted on development rows only; NaNs left for in-pipeline imputation |
| `fraud_model_pipeline.joblib` | Uncalibrated preprocessing + logistic-regression pipeline (fit on the 80 % development set)                                                                         |
| `calibrated_model.joblib`     | Production candidate: same pipeline with 5-fold cross-fitted Platt scaling                                                                                          |
| `threshold_cost_analysis.csv` | Threshold sweep (0.01–0.99) for 11 cost scenarios: TP/FP/FN/TN, precision, recall, F1, alert rate, cost, optimum flags                                              |
| `model_card.pdf`              | Model governance document (3 pages)                                                                                                                                 |
| `business_fraud_strategy.pdf` | Management report (9 pages)                                                                                                                                         |
| `decision_policy.json`        | Thresholds (review 0.179, block 0.240), base costs, seed                                                                                                            |
| `supplementary/`              | Data-quality tables, ablation, calibration, scenario costs, stress tests, PSI, monitoring plan, `results.joblib` (every number used in the PDFs)                    |
| `figures/`                    | 20 figures (12 EDA incl. 4 interaction maps)                                                                                                                        |

## Environment

Python 3.13; scikit-learn 1.9, imbalanced-learn, LightGBM, SHAP, pandas, numpy, scipy, matplotlib, joblib, reportlab, Pillow, nbconvert.
`pip install scikit-learn imbalanced-learn lightgbm shap pandas numpy scipy matplotlib joblib reportlab pillow jupyter`

## Run

From the project root (where `src/` and the CSV live):

```
jupyter nbconvert --to notebook --execute --inplace fraud_risk_analysis.ipynb   # regenerates outputs/
python build_reports.py                                                          # regenerates both PDFs
```

Slow hyper-parameter searches are cached in `outputs/cache/`; delete it to retrain from scratch (several minutes). Code is modular in `src/fraud_engine/` (config, data_quality, eda, features, models, calibration, costs, explain, robustness, reporting, engine).

## Reproducibility and leakage controls

- Seed 42 everywhere; 80/20 stratified split; 5-fold (5×3 repeated) stratified CV.
- Test set (600 rows, 73 fraud) used once at the end – not for feature choice, tuning, thresholds, calibration or resampling.
- Imputation, encoders, group statistics, scalers, SMOTE / under-sampling live inside `sklearn`/`imblearn` pipelines, refit per fold.
- No observation removed; `transaction_id` excluded.

## Key assumptions

- **All cost figures are hypothetical** (see `src/fraud_engine/config.py` and the report): fraud loss = amount + $15; false-positive block $20; review $4 (+$2 friction), analysts catch 90 %; at most 10 % of volume reviewed and 5 % blocked. Ten sensitivity scenarios included.
- Velocity counters, authentication result, risk scores, chargebacks and ages are assumed point-in-time values available before the decision.
- No timestamp in the data → no out-of-time validation; results are optimistic. 12 % prevalence is probably higher than production.
- 100,000-transaction figures are scaled from the 600-row test set and carry wide intervals.

## Main findings

Test ROC-AUC 0.71, PR-AUC 0.24 (random 0.12). Proposed three-way policy flags ~32 % of fraud, reviewing ~10 % and blocking ~5 % of volume, cutting hypothetical cost ~42 % versus approving everything (95 % CI wide). Default 0.50 threshold flags nothing. Tree models were not statistically better than the logistic baseline; engineered features gave no measurable lift.

## Loading the models

The joblib files reference the custom transformer in `src/fraud_engine`, so put it on the path first:

```python
import sys; sys.path.insert(0, "src")
import joblib; model = joblib.load("outputs/calibrated_model.joblib")
p = model.predict_proba(raw_dataframe_without_fraud_column)[:, 1]
```

`src/fraud_engine/engine.py::FraudDecisionEngine` adds the Approve/Review/Block decision and reason codes.
