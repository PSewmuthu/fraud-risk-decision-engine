"""Pipelines, hyper-parameter search spaces, tuning and cross-validation helpers."""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier

from .config import SEED, TARGET, TEST_SIZE, N_SPLITS, N_REPEATS
from .features import FeatureEngineer

warnings.filterwarnings("ignore")


def split_data(df):
    """Stratified dev/test split. The test set is touched only for final evaluation."""
    dev, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEED)
    return dev.reset_index(drop=True), test.reset_index(drop=True)


def xy(df):
    return df.drop(columns=[TARGET]), df[TARGET].values


def make_preprocessor(scale):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    from sklearn.pipeline import Pipeline as SkPipe
    return ColumnTransformer(
        [("num", SkPipe(num_steps), make_column_selector(dtype_include=np.number)),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
          make_column_selector(dtype_exclude=np.number))],
        verbose_feature_names_out=False).set_output(transform="pandas")


def make_model(name, **kw):
    if name == "logreg":
        return LogisticRegression(max_iter=3000, random_state=SEED, **kw)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1, **kw)
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **kw)
    if name == "lgbm":
        return LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1, **kw)
    raise ValueError(name)


def make_sampler(kind):
    if kind in (None, "none", "class_weight"):
        return None
    if kind == "smote":
        return SMOTE(random_state=SEED, k_neighbors=5)
    if kind == "undersample":
        return RandomUnderSampler(sampling_strategy=0.5, random_state=SEED)
    raise ValueError(kind)


def build_pipeline(name, engineer=True, missing_indicators=False, sampler=None, model_params=None):
    """FeatureEngineer -> preprocessor -> (optional sampler) -> model.

    The sampler is an imblearn step, so resampling happens only on the training
    part of each CV fold and never on validation or test data.
    """
    scale = name == "logreg"
    steps = [("fe", FeatureEngineer(engineer=engineer, missing_indicators=missing_indicators)),
             ("prep", make_preprocessor(scale))]
    s = make_sampler(sampler)
    if s is not None:
        steps.append(("sampler", s))
    steps.append(("model", make_model(name, **(model_params or {}))))
    return Pipeline(steps)


SEARCH_SPACES = {
    "logreg": {"model__C": loguniform(1e-3, 1e2), "model__class_weight": [None, "balanced"]},
    "rf": {"model__n_estimators": [200, 400], "model__max_depth": [3, 5, 8, 12, None],
           "model__min_samples_leaf": [1, 3, 5, 10, 20], "model__max_features": ["sqrt", 0.3, 0.5],
           "model__class_weight": [None, "balanced_subsample"]},
    "hgb": {"model__learning_rate": loguniform(0.01, 0.2), "model__max_depth": [2, 3, 4, None],
            "model__max_leaf_nodes": [8, 15, 31], "model__min_samples_leaf": [10, 20, 40],
            "model__l2_regularization": loguniform(1e-3, 10), "model__max_iter": [100, 200, 300],
            "model__class_weight": [None, "balanced"]},
    "lgbm": {"model__n_estimators": [100, 200, 400], "model__learning_rate": loguniform(0.01, 0.15),
             "model__num_leaves": [7, 15, 31], "model__min_child_samples": [10, 20, 40],
             "model__subsample": uniform(0.6, 0.4), "model__subsample_freq": [1],
             "model__colsample_bytree": uniform(0.5, 0.5), "model__reg_lambda": loguniform(1e-2, 20),
             "model__class_weight": [None, "balanced"]},
}
MODEL_LABELS = {"logreg": "Logistic Regression", "rf": "Random Forest",
                "hgb": "HistGradientBoosting", "lgbm": "LightGBM"}


def cv_splitter():
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def tune(name, X, y, n_iter=30, **pipe_kw):
    """RandomizedSearchCV optimising PR-AUC (average precision) on the dev set only."""
    search = RandomizedSearchCV(
        build_pipeline(name, **pipe_kw), SEARCH_SPACES[name], n_iter=n_iter,
        scoring="average_precision", cv=cv_splitter(), random_state=SEED, n_jobs=2,
        refit=True, return_train_score=False)
    search.fit(X, y)
    return search


def metrics_at(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0)}


def cv_scores(pipe, X, y, repeated=True):
    """Per-fold metrics (repeated stratified CV). Returns a DataFrame, one row per fold."""
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS if repeated else 1, random_state=SEED)
    rows = []
    for i, (tr, va) in enumerate(cv.split(X, y)):
        m = clone(pipe).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        r = metrics_at(y[va], p)
        r.update(fold=i, repeat=i // N_SPLITS)
        rows.append(r)
    return pd.DataFrame(rows)


def oof_proba(pipe, X, y, seed=SEED):
    """Out-of-fold probabilities: each row is scored by a model that never saw it."""
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(X, y):
        p[va] = clone(pipe).fit(X.iloc[tr], y[tr]
                                ).predict_proba(X.iloc[va])[:, 1]
    return p


def summarize(df, cols=("roc_auc", "pr_auc", "brier", "recall", "precision", "f1")):
    return df[list(cols)].agg(["mean", "std"]).T
