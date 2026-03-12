#!/usr/bin/env python3
"""Train and optimize RF, XGBoost, CatBoost, and MLP models for delayed-shipment prediction.

Default inputs map to this project:
- model_ready_tree.csv  -> RandomForest, XGBoost
- model_ready_nn.csv    -> MLP
- model_ready_catboost.csv -> CatBoost

Metrics emphasized:
- ROC-AUC
- Precision (delayed class = 1)
- Recall (delayed class = 1)
"""

import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


@dataclass
class ModelResult:
    model: str
    roc_auc: float
    precision_delayed: float
    recall_delayed: float
    f1_delayed: float
    threshold: float



def delayed_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    delayed = report.get("1", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision_delayed": float(delayed["precision"]),
        "recall_delayed": float(delayed["recall"]),
        "f1_delayed": float(delayed["f1-score"]),
    }



def find_best_threshold(y_true: pd.Series, y_prob: np.ndarray, start: float = 0.20, stop: float = 0.80, step: float = 0.01) -> Tuple[float, float]:
    thresholds = np.arange(start, stop + 1e-9, step)
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])



def split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return X, y



def train_rf(tree_df: pd.DataFrame, target: str) -> ModelResult:
    X, y = split_xy(tree_df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE)

    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    params = {
        "n_estimators": [300, 500, 700, 900],
        "max_depth": [None, 8, 12, 16, 22],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "max_features": ["sqrt", "log2", 0.5, 0.8],
        "class_weight": ["balanced", "balanced_subsample", None],
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=params,
        n_iter=20,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    search.fit(X_tr, y_tr)

    best_model = search.best_estimator_
    val_prob = best_model.predict_proba(X_val)[:, 1]
    best_thr, _ = find_best_threshold(y_val, val_prob)

    test_prob = best_model.predict_proba(X_test)[:, 1]
    m = delayed_metrics(y_test, test_prob, best_thr)

    print("\n[RandomForest] best params:", search.best_params_)
    return ModelResult("RandomForest", m["roc_auc"], m["precision_delayed"], m["recall_delayed"], m["f1_delayed"], best_thr)



def train_xgb(tree_df: pd.DataFrame, target: str) -> ModelResult:
    from xgboost import XGBClassifier

    X, y = split_xy(tree_df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE)

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    scale_pos_weight = float(neg / max(pos, 1))

    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    params = {
        "n_estimators": [300, 500, 700, 900],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.3, 0.5],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=params,
        n_iter=20,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    search.fit(X_tr, y_tr)

    best_model = XGBClassifier(
        **search.best_params_,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    best_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_prob = best_model.predict_proba(X_val)[:, 1]
    best_thr, _ = find_best_threshold(y_val, val_prob)

    test_prob = best_model.predict_proba(X_test)[:, 1]
    m = delayed_metrics(y_test, test_prob, best_thr)

    print("\n[XGBoost] best params:", search.best_params_)
    return ModelResult("XGBoost", m["roc_auc"], m["precision_delayed"], m["recall_delayed"], m["f1_delayed"], best_thr)



def train_catboost(cat_df: pd.DataFrame, target: str) -> ModelResult:
    from catboost import CatBoostClassifier

    X, y = split_xy(cat_df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE)

    cat_cols = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_features = [X_tr.columns.get_loc(c) for c in cat_cols]

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    class_weights = [1.0, float(neg / max(pos, 1))]

    param_dist = {
        "iterations": [400, 700, 1000],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.08],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "bagging_temperature": [0, 0.5, 1, 2],
        "random_strength": [0.5, 1, 2, 5],
        "border_count": [64, 128, 254],
    }

    best_auc = -np.inf
    best_params = None
    best_model = None
    best_val_prob = None

    for params in ParameterSampler(param_dist, n_iter=16, random_state=RANDOM_STATE):
        model = CatBoostClassifier(
            **params,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            class_weights=class_weights,
            allow_writing_files=False,
            verbose=False,
        )

        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_features,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=100,
        )

        val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        if val_auc > best_auc:
            best_auc = val_auc
            best_params = params
            best_model = model
            best_val_prob = val_prob

    best_thr, _ = find_best_threshold(y_val, best_val_prob)

    test_prob = best_model.predict_proba(X_test)[:, 1]
    m = delayed_metrics(y_test, test_prob, best_thr)

    print("\n[CatBoost] best params:", best_params)
    return ModelResult("CatBoost", m["roc_auc"], m["precision_delayed"], m["recall_delayed"], m["f1_delayed"], best_thr)



def train_mlp(nn_df: pd.DataFrame, target: str) -> ModelResult:
    X, y = split_xy(nn_df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    activation="relu",
                    solver="adam",
                    max_iter=120,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    params = {
        "mlp__hidden_layer_sizes": [(128, 64), (256, 128), (256, 128, 64), (512, 256)],
        "mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3],
        "mlp__batch_size": [256, 512, 1024],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=12,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    search.fit(X_tr, y_tr)

    best_model = search.best_estimator_
    val_prob = best_model.predict_proba(X_val)[:, 1]
    best_thr, _ = find_best_threshold(y_val, val_prob)

    test_prob = best_model.predict_proba(X_test)[:, 1]
    m = delayed_metrics(y_test, test_prob, best_thr)

    print("\n[MLP] best params:", search.best_params_)
    return ModelResult("NeuralNet_MLP", m["roc_auc"], m["precision_delayed"], m["recall_delayed"], m["f1_delayed"], best_thr)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize RF, XGB, CatBoost and MLP for delayed prediction.")
    parser.add_argument("--tree-data", default="model_ready_tree.csv", help="CSV for RandomForest and XGBoost")
    parser.add_argument("--nn-data", default="model_ready_nn.csv", help="CSV for MLP")
    parser.add_argument("--catboost-data", default="model_ready_catboost.csv", help="CSV for CatBoost")
    parser.add_argument("--target", default="delayed", help="Target column name")
    parser.add_argument("--output", default="model_comparison_results.csv", help="Where to save summary metrics")
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    tree_df = pd.read_csv(args.tree_data)
    nn_df = pd.read_csv(args.nn_data)
    cat_df = pd.read_csv(args.catboost_data)

    results = [
        train_rf(tree_df, args.target),
        train_xgb(tree_df, args.target),
        train_catboost(cat_df, args.target),
        train_mlp(nn_df, args.target),
    ]

    summary = pd.DataFrame([r.__dict__ for r in results]).sort_values(
        ["roc_auc", "recall_delayed", "precision_delayed"], ascending=False
    )

    print("\n=== Model Leaderboard ===")
    print(summary.to_string(index=False))

    summary.to_csv(args.output, index=False)
    print(f"\nSaved metrics to: {args.output}")


if __name__ == "__main__":
    main()
