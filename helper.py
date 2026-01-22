# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.preprocessing import label_binarize

import plots as p
from models import (
    ClassificationMetrics,
    RocCurveData,
    ClassificationReportRow,
    BaseMetrics,
    RegressionMetrics,
    MultipleModelResults,
)


# ============================================================
# Common helpers
# ============================================================

def divide_data(data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split data into features and target."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def _is_multiclass(y: Any) -> bool:
    y_arr = np.asarray(y)
    uniq = np.unique(y_arr)
    return len(uniq) > 2


def _coerce_proba_for_binary(y_probs: Any) -> Optional[np.ndarray]:
    """
    For binary ROC curve we need a 1D score:
      - if proba is (N,2) -> take [:,1]
      - if proba is (N,) -> use as-is
      - else -> None
    """
    if y_probs is None:
        return None
    yp = np.asarray(y_probs)
    if yp.ndim == 1:
        return yp.astype(float)
    if yp.ndim == 2 and yp.shape[1] == 2:
        return yp[:, 1].astype(float)
    return None


# ============================================================
# Classification metrics (multi-class aware)
# ============================================================

def _calculate_classification_metrics(
    y_test: Any,
    y_pred: Any,
    y_probs: Optional[Any] = None,
    class_names: Optional[List[str]] = None,
) -> ClassificationMetrics:
    """Supports binary and multiclass. Matches current ClassificationMetrics fields."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve,
    )

    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_test, y_pred]))
    n_classes = int(len(classes))

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    # macro
    pr_macro, rc_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    pr_macro, rc_macro, f1_macro = float(pr_macro), float(rc_macro), float(f1_macro)

    # weighted (обязательные поля у тебя есть!)
    pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    pr_w, rc_w, f1_w = float(pr_w), float(rc_w), float(f1_w)

    # AUC
    roc_auc_ovr_macro: Optional[float] = None
    if y_probs is not None:
        y_probs = np.asarray(y_probs)
        try:
            if n_classes <= 2:
                # binary: y_probs (N,) or (N,2)
                if y_probs.ndim == 2 and y_probs.shape[1] == 2:
                    y_probs_1d = y_probs[:, 1]
                else:
                    y_probs_1d = y_probs
                roc_auc_ovr_macro = float(roc_auc_score(y_test, y_probs_1d))
            else:
                # multiclass: y_probs (N,C)
                if y_probs.ndim != 2:
                    raise ValueError("For multiclass AUC y_probs must be 2D (N, C)")
                roc_auc_ovr_macro = float(
                    roc_auc_score(y_test, y_probs, multi_class="ovr", average="macro")
                )
        except Exception:
            roc_auc_ovr_macro = None

    # payload строго под твою модель (поля ты уже вывел)
    payload = dict(
        name="Model",
        training_time=None,
        estimators=None,

        accuracy=acc,

        f1_macro=f1_macro,
        precision_macro=pr_macro,
        recall_macro=rc_macro,

        f1_weighted=f1_w,
        precision_weighted=pr_w,
        recall_weighted=rc_w,

        roc_auc_ovr_macro=roc_auc_ovr_macro,

        confusion_matrix=cm,
        classification_report=None,
        roc_curve_micro=None,
        roc_curves_by_class=None,

        # backward-compat
        roc_auc=roc_auc_ovr_macro,
        f1_score=f1_macro,
        precision=pr_macro,
        recall=rc_macro,
        roc_curve=None,
    )

    metrics_model = ClassificationMetrics(**payload)

    # classification_report по классам
    if hasattr(metrics_model, "classification_report"):
        pr_by, rc_by, _, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=classes, average=None, zero_division=0
        )

        if class_names is not None and len(class_names) == len(classes):
            labels_for_rows = class_names
        else:
            labels_for_rows = [str(c) for c in classes]

        p_c, r_c, f1_c, s_c = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, average=None, zero_division=0
    )

    rows: List[ClassificationReportRow] = []
    for i, lbl in enumerate(labels_for_rows):
        rows.append(
            _build_report_row(
                class_label=str(lbl),
                precision=float(p_c[i]),
                recall=float(r_c[i]),
                f1=float(f1_c[i]),
                support=int(s_c[i]),
            )
        )
    metrics_model.classification_report = rows

    # roc_curve только для бинарного (и только если поле есть)
    if y_probs is not None and n_classes <= 2 and hasattr(metrics_model, "roc_curve"):
        if y_probs.ndim == 2 and y_probs.shape[1] == 2:
            y_probs_1d = y_probs[:, 1]
        else:
            y_probs_1d = y_probs
        fpr, tpr, thresholds = roc_curve(y_test, y_probs_1d)
        metrics_model.roc_curve = RocCurveData(fpr=fpr, tpr=tpr, thresholds=thresholds)

    return metrics_model


def _build_report_row(
    class_label: str,
    precision: float,
    recall: float,
    f1: float,
    support: int,
) -> ClassificationReportRow:
    """
    Build ClassificationReportRow compatible with your models.py schema.
    """
    fields = getattr(ClassificationReportRow, "model_fields", None)  # pydantic v2
    if fields is None:
        fields = getattr(ClassificationReportRow, "__fields__", {})  # pydantic v1

    payload = {"class_label": class_label}

    # Always add what exists
    if "precision" in fields:
        payload["precision"] = float(precision)
    if "recall" in fields:
        payload["recall"] = float(recall)

    # These are required in your current schema
    if "f1" in fields:
        payload["f1"] = float(f1)
    if "f1_score" in fields:
        payload["f1_score"] = float(f1)
    if "support" in fields:
        payload["support"] = int(support)

    return ClassificationReportRow(**payload)



def evaluate_classification(
    y_test: Any,
    y_pred: Any,
    y_probs: Optional[Any] = None,
    model_name: str = "Model",
    enable_plot: bool = True,
    class_names: Optional[List[str]] = None,
) -> ClassificationMetrics:
    """Evaluate performance and optionally plot/print using the Pydantic model."""
    metrics_model = _calculate_classification_metrics(
        y_test=y_test,
        y_pred=y_pred,
        y_probs=y_probs,
        class_names=class_names,
    )

    if enable_plot:
        p.plot_classification_results(metrics_model, model_name)
        p.print_classification_report(metrics_model.to_report_dict(), model_name)

    return metrics_model


# ============================================================
# Regression metrics (как в примере)
# ============================================================

def _calculate_regression_metrics(y_test: Any, y_pred: Any) -> RegressionMetrics:
    """Calculate regression performance metrics into a Pydantic model."""
    mae_value = float(mean_absolute_error(y_test, y_pred))
    mse_value = float(mean_squared_error(y_test, y_pred))
    rmse_value = float(mse_value ** 0.5)
    r2_value = float(r2_score(y_test, y_pred))
    explained_variance_value = float(explained_variance_score(y_test, y_pred))

    metrics_model = RegressionMetrics(
        mae=mae_value,
        mse=mse_value,
        rmse=rmse_value,
        r2=r2_value,
        explained_variance=explained_variance_value,
    )
    return metrics_model


def evaluate_regression(
    y_test: Any,
    y_pred: Any,
    model_name: str = "Model",
    enable_plot: bool = True
) -> RegressionMetrics:
    """Evaluate regression performance and optionally plot/print using the Pydantic model."""
    metrics_model = _calculate_regression_metrics(y_test, y_pred)

    if enable_plot:
        p.plot_regression_results(metrics_model, model_name)
        p.print_regression_report(metrics_model.get_numeric_metrics(), model_name)

    return metrics_model


# ============================================================
# CV aggregation helpers (как в примере)
# ============================================================

def aggregate_regression_cv_metrics(
    *,
    mae: Optional[float] = None,
    mse: Optional[float] = None,
    rmse: Optional[float] = None,
    r2: Optional[float] = None,
    explained_variance: Optional[float] = None,
    training_time: Optional[float] = None,
    name: Optional[str] = None,
) -> RegressionMetrics:
    computed_rmse = rmse
    if computed_rmse is None and mse is not None:
        try:
            computed_rmse = float(mse ** 0.5)
        except Exception:
            computed_rmse = None

    metrics = RegressionMetrics(
        mae=float(mae) if mae is not None else float("nan"),
        mse=float(mse) if mse is not None else float("nan"),
        rmse=float(computed_rmse) if computed_rmse is not None else float("nan"),
        r2=float(r2) if r2 is not None else float("nan"),
        explained_variance=float(explained_variance) if explained_variance is not None else float("nan"),
        training_time=float(training_time) if training_time is not None else None,
        name=name,
    )
    return metrics


def aggregate_classification_cv_metrics(
    *,
    accuracy: Optional[float] = None,
    precision: Optional[float] = None,
    recall: Optional[float] = None,
    f1_score_value: Optional[float] = None,
    roc_auc: Optional[float] = None,
    training_time: Optional[float] = None,
    name: Optional[str] = None,
    y_true: Optional[Union[np.ndarray, Any]] = None,
    y_pred: Optional[Union[np.ndarray, Any]] = None,
    y_probs: Optional[Union[np.ndarray, Any]] = None,
    class_names: Optional[List[str]] = None,
) -> ClassificationMetrics:
    """
    Build a ClassificationMetrics object from CV summaries and optional OOF data.
    Multiclass-aware (roc_auc = OVR macro; roc_curve = micro-average).
    """
    cm = None
    roc_curve_data = None

    if y_true is not None and y_pred is not None:
        cm = confusion_matrix(y_true, y_pred)

    if y_true is not None and y_probs is not None:
        y_true_arr = np.asarray(y_true)
        y_probs_arr = np.asarray(y_probs)

        if _is_multiclass(y_true_arr):
            # micro-average ROC curve
            try:
                classes = np.unique(y_true_arr)
                Y = label_binarize(y_true_arr, classes=classes)
                if y_probs_arr.ndim == 2 and Y.shape[1] == y_probs_arr.shape[1]:
                    fpr, tpr, thresholds = roc_curve(Y.ravel(), y_probs_arr.ravel())
                    roc_curve_data = RocCurveData(fpr=fpr, tpr=tpr, thresholds=thresholds)
            except Exception:
                roc_curve_data = None

            if roc_auc is None:
                try:
                    roc_auc = float(
                        roc_auc_score(y_true_arr, y_probs_arr, multi_class="ovr", average="macro")
                    )
                except Exception:
                    roc_auc = None
        else:
            y_score = _coerce_proba_for_binary(y_probs_arr)
            if y_score is not None:
                try:
                    fpr, tpr, thresholds = roc_curve(y_true_arr, y_score)
                    roc_curve_data = RocCurveData(fpr=fpr, tpr=tpr, thresholds=thresholds)
                except Exception:
                    roc_curve_data = None
                if roc_auc is None:
                    try:
                        roc_auc = float(roc_auc_score(y_true_arr, y_score))
                    except Exception:
                        roc_auc = None

    metrics = _build_classification_metrics_model(
        name=name or "Model",
        accuracy=float(accuracy) if accuracy is not None else float("nan"),
        precision_macro=float(precision) if precision is not None else float("nan"),
        recall_macro=float(recall) if recall is not None else float("nan"),
        f1_macro=float(f1_score_value) if f1_score_value is not None else float("nan"),
        roc_auc=(None if roc_auc is None or (isinstance(roc_auc, float) and np.isnan(roc_auc)) else float(roc_auc)),
        confusion_matrix_value=cm,
        training_time=training_time,
    )


    # add classification report 
    if y_true is not None and y_pred is not None:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        labels = np.unique(y_true_arr)

        p_c, r_c, f1_c, s_c = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0
        )

        rows: List[ClassificationReportRow] = []
        for i, lab in enumerate(labels):
            name_lab = str(lab)
            if class_names is not None and int(lab) >= 0 and int(lab) < len(class_names):
                name_lab = str(class_names[int(lab)])

            rows.append(
                _build_report_row(
                    class_label=name_lab,
                    precision=float(p_c[i]),
                    recall=float(r_c[i]),
                    f1=float(f1_c[i]),
                    support=int(s_c[i]),
                )
            )

        metrics.classification_report = rows


    if roc_curve_data is not None:
        metrics.roc_curve = roc_curve_data

    return metrics


def _set_model_random_state(model: BaseEstimator, seed: Optional[int]) -> None:
    """Set random state for model if supported."""
    if seed is None:
        return
    try:
        if hasattr(model, "random_state"):
            model.set_params(random_state=seed)
        elif hasattr(model, "seed"):
            model.set_params(seed=seed)
    except Exception:
        pass


# ============================================================
# Heatmap comparison (как в примере)
# ============================================================

def plot_metrics_heatmap(
    metrics: List[BaseMetrics],
    title: str = "Model Evaluation Metrics Comparison",
    figsize: Tuple[int, int] = (8, 4),
) -> None:
    """Plot heatmap of model metrics with per-column color scaling."""
    rows: Dict[str, Dict[str, float]] = {}
    for i, m in enumerate(metrics):
        row_name = m.name if getattr(m, "name", None) else f"Model {i+1}"
        rows[row_name] = m.get_numeric_metrics()
    KEEP_COLS = ["ROC AUC", "F1 Score", "Precision", "Recall", "Accuracy", "Training Time (s)"]

    metrics_df = pd.DataFrame.from_dict(rows, orient="index")
    metrics_df = metrics_df[[c for c in KEEP_COLS if c in metrics_df.columns]]


    plt.figure(figsize=figsize)

    normalized_df = metrics_df.copy()
    for col in metrics_df.columns:
        col_min = metrics_df[col].min()
        col_max = metrics_df[col].max()
        if col_max != col_min:
            normalized_df[col] = (metrics_df[col] - col_min) / (col_max - col_min)
        else:
            normalized_df[col] = 0.5

    sns.heatmap(
        normalized_df,
        cmap="RdBu_r",
        annot=metrics_df,
        fmt=".3f",
        cbar_kws={"label": "Normalized Score (0-1 per metric)"},
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


def _evaluate_multiple_models_pydantic(
    models: List[Tuple[str, BaseEstimator]],
    evaluation_func: Callable,
    task_type: str,
    *args,
    **kwargs,
) -> MultipleModelResults:
    metrics_objects: List[BaseMetrics] = []

    for model_name, model in models:
        current_model = clone(model)
        eval_result = evaluation_func(current_model, model_name, *args, **kwargs)

        if isinstance(eval_result, BaseMetrics):
            eval_result.name = model_name
            metrics_objects.append(eval_result)
        else:
            from models import GenericMetrics
            generic_metrics = GenericMetrics(values=eval_result, name=model_name)
            metrics_objects.append(generic_metrics)

    plot_metrics_heatmap(metrics_objects)
    return MultipleModelResults(results=metrics_objects, task_type=task_type)


# ============================================================
# Single model train/eval (hold-out) — оставлено как в примере, но multiclass-safe
# ============================================================

def train_evaluate_model(
    model: BaseEstimator,
    model_name: str,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    seed: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Train and evaluate a single model."""
    _set_model_random_state(model, seed)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_probs = None
    if hasattr(model, "predict_proba"):
        try:
            y_probs = model.predict_proba(X_test)
        except Exception:
            y_probs = None

    metrics_model = evaluate_classification(
        y_test=y_test,
        y_pred=y_pred,
        y_probs=y_probs,
        model_name=model_name,
        enable_plot=False,
        class_names=class_names,
    )
    return metrics_model.to_compact_dict()


# ============================================================
# CV evaluation — ключевая часть для baseline ноутбука
# ============================================================

def train_evaluate_model_cv(
    model: BaseEstimator,
    model_name: str,
    X: Any,
    y: Any,
    preprocessor: Optional[Any] = None,
    cv: Any = 5,
    seed: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    plot_feature_importance: bool = True,
    task_type: str = "classification",
    class_names: Optional[List[str]] = None,
) -> BaseMetrics:
    """
    Train and evaluate a model using cross-validation (classification/regression).

    Fixes vs your current version:
    - Uses sklearn>=1.3 scoring API (response_method instead of needs_proba).
    - Computes OOF predictions to enable confusion matrix and ROC plots.
    - Safely handles models without predict_proba (ROC metrics/curves become None).
    - Removes duplicated/unsafe roc_mean computation.
    - Keeps your existing plotting + feature-importance hooks.
    """
    _set_model_random_state(model, seed)

    # ---- build pipeline ----
    if isinstance(preprocessor, Pipeline):
        steps = preprocessor.steps.copy()
        steps.append(("model", model))
        pipeline = Pipeline(steps)
    elif preprocessor is not None:
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    else:
        pipeline = model

    y_arr = np.asarray(y)

    # ---- scoring ----
    if task_type == "classification":
        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, average="macro", zero_division=0),
            "recall": make_scorer(recall_score, average="macro", zero_division=0),
            "f1": make_scorer(f1_score, average="macro", zero_division=0),
        }

        # ROC AUC scorer:
        # - multiclass: OVR macro on predict_proba
        # - binary: try predict_proba/decision_function
        if _is_multiclass(y_arr):
            scoring["roc_auc"] = make_scorer(
                roc_auc_score,
                response_method="predict_proba",
                multi_class="ovr",
                average="macro",
            )
        else:
            scoring["roc_auc"] = make_scorer(
                roc_auc_score,
                response_method=("predict_proba", "decision_function"),
            )

    elif task_type == "regression":
        scoring = {
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2",
            "explained_variance": "explained_variance",
        }
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")

    # ---- CV ----
    start_time = time.time()
    try:
        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            return_estimator=True,
        )
    except Exception:
        # fallback: if roc_auc scorer fails (e.g., estimator has no proba/decision)
        if task_type == "classification" and "roc_auc" in scoring:
            scoring2 = dict(scoring)
            scoring2.pop("roc_auc", None)
            cv_results = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring2,
                return_train_score=False,
                return_estimator=True,
            )
        else:
            raise
    training_time = time.time() - start_time

    # ---- build metrics object ----
    if task_type == "classification":
        # mean ROC AUC (safe)
        roc_mean = None
        if "test_roc_auc" in cv_results:
            vals = cv_results["test_roc_auc"]
            if vals is not None and len(vals) > 0:
                m = float(np.nanmean(vals))
                if np.isfinite(m):
                    roc_mean = m

        # OOF predictions for confusion matrix + ROC curves
        y_pred_oof = cross_val_predict(pipeline, X, y, cv=cv, method="predict")

        y_proba_oof = None
        try:
            y_proba_oof = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")
        except Exception:
            y_proba_oof = None

        acc_mean = float(np.nanmean(cv_results["test_accuracy"]))
        prec_mean = float(np.nanmean(cv_results["test_precision"]))
        rec_mean = float(np.nanmean(cv_results["test_recall"]))
        f1_mean = float(np.nanmean(cv_results["test_f1"]))

        cv_metrics = aggregate_classification_cv_metrics(
            accuracy=acc_mean,
            precision=prec_mean,
            recall=rec_mean,
            f1_score_value=f1_mean,
            roc_auc=roc_mean,
            training_time=float(training_time),
            name=model_name,
            y_true=y,
            y_pred=y_pred_oof,
            y_probs=y_proba_oof,
            class_names=class_names,
        )
        cv_metrics.estimators = cv_results.get("estimator")

        p.plot_classification_results(cv_metrics, model_name)

    else:
        cv_metrics = aggregate_regression_cv_metrics(
            mae=float(-np.nanmean(cv_results["test_mae"])),
            mse=float(-np.nanmean(cv_results["test_mse"])),
            r2=float(np.nanmean(cv_results["test_r2"])),
            explained_variance=float(np.nanmean(cv_results["test_explained_variance"])),
            training_time=float(training_time),
            name=model_name,
        )
        cv_metrics.estimators = cv_results.get("estimator")
        p.plot_regression_results(cv_metrics, model_name)

    # ---- feature importance ----
    if plot_feature_importance:
        _plot_feature_importance_cv(pipeline, model_name, feature_names, X, y)

    return cv_metrics


def _unwrap_estimator(est):
    # unwrap Pipeline
    if hasattr(est, "steps"):
        est = est.steps[-1][1]

    # unwrap OneVsRestClassifier
    if hasattr(est, "estimator") and est.__class__.__name__ in ("OneVsRestClassifier",):
        est = est.estimator

    # unwrap CalibratedClassifierCV
    if hasattr(est, "base_estimator"):
        est = est.base_estimator

    return est


def _plot_feature_importance_cv(pipeline, model_name, feature_names, X, y):
    from types import SimpleNamespace
    import numpy as np
    from sklearn.pipeline import Pipeline

    try:
        pipeline.fit(X, y)

        # 1) final estimator
        final_model = pipeline.steps[-1][1] if isinstance(pipeline, Pipeline) else pipeline

        # 2) unwrap search
        if hasattr(final_model, "best_estimator_"):
            final_model = final_model.best_estimator_

        # 3) unwrap calibration
        if hasattr(final_model, "base_estimator_"):
            final_model = final_model.base_estimator_
        elif hasattr(final_model, "base_estimator"):
            final_model = final_model.base_estimator

        # 4) feature importances
        pseudo_model = None
        importances = None

        if hasattr(final_model, "feature_importances_"):
            importances = np.asarray(final_model.feature_importances_)
            pseudo_model = final_model

        elif hasattr(final_model, "coef_"):
            coef = np.asarray(final_model.coef_)
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
            pseudo_model = SimpleNamespace(feature_importances_=importances)

        else:
            print(f"Warning: Model {model_name} does not support feature importance")
            return

        n_features = int(len(importances))

        actual_feature_names = _extract_feature_names_from_pipeline(
            pipeline=pipeline,
            X=X,
            n_features=n_features,
            provided_feature_names=feature_names
        )

        p.plot_feature_importance(pseudo_model, actual_feature_names, top_n=20)

    except Exception as e:
        print(f"Warning: Could not plot feature importance for {model_name}: {str(e)}")




def _extract_feature_names_from_pipeline(
    pipeline: Any,
    X: Any,
    n_features: int,
    provided_feature_names: Optional[List[str]] = None,
) -> List[str]:
    if provided_feature_names is not None and len(provided_feature_names) == n_features:
        return provided_feature_names

    if not isinstance(pipeline, Pipeline) or len(pipeline.steps) == 1:
        if hasattr(X, "columns") and len(X.columns) == n_features:
            return list(X.columns)
        return [f"feature_{i}" for i in range(n_features)]

    feature_names: List[str] = []
    preprocessor_steps = pipeline.steps[:-1]

    if len(preprocessor_steps) == 1:
        step_name, transformer = preprocessor_steps[0]
        feature_names = _extract_feature_names_from_transformer(transformer, step_name)
    else:
        for step_name, transformer in preprocessor_steps:
            feature_names.extend(_extract_feature_names_from_transformer(transformer, step_name))

    if len(feature_names) != n_features:
        if hasattr(X, "columns") and len(getattr(X, "columns")) == n_features:
            feature_names = list(X.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(n_features)]

    return feature_names


def _extract_feature_names_from_transformer(transformer: Any, step_name: str) -> List[str]:
    try:
        if hasattr(transformer, "transformers_"):
            names: List[str] = []
            for name, trans, columns in transformer.transformers_:
                if hasattr(trans, "get_feature_names_out"):
                    trans_names = list(trans.get_feature_names_out())
                    prefix = f"{name}_"
                    names.extend([f"{prefix}{n}" for n in trans_names])
            return names

        if hasattr(transformer, "get_feature_names_out"):
            return list(transformer.get_feature_names_out())

        if hasattr(transformer, "vocabulary_"):
            return list(transformer.vocabulary_.keys())

        if hasattr(transformer, "feature_names_in_"):
            return list(transformer.feature_names_in_)

        return []
    except Exception:
        return []


# ============================================================
# Multiple models CV — это то, что тебе нужно в baseline
# ============================================================

def train_evaluate_models_cv(
    models: Union[List[Tuple[str, BaseEstimator]], Dict[str, BaseEstimator]],
    X: Any,
    y: Any,
    preprocessor: Optional[Any] = None,
    cv: Any = 5,
    seed: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    plot_feature_importance: bool = True,
    task_type: str = "classification",
    class_names: Optional[List[str]] = None,
) -> MultipleModelResults:
    """
    Train and evaluate multiple models using cross-validation.

    Совместимо с двумя форматами:
    - list[('name', estimator), ...]  (как в примере)
    - dict[name] = estimator          (как часто удобно в ноутбуке)
    """
    if isinstance(models, dict):
        models_list = list(models.items())
    else:
        models_list = list(models)

    def _cv_wrapper(model: BaseEstimator, model_name: str) -> BaseMetrics:
        current_preprocessor = clone(preprocessor) if preprocessor is not None else None
        return train_evaluate_model_cv(
            model=model,
            model_name=model_name,
            X=X,
            y=y,
            preprocessor=current_preprocessor,
            cv=cv,
            seed=seed,
            feature_names=feature_names,
            plot_feature_importance=plot_feature_importance,
            task_type=task_type,
            class_names=class_names,
        )

    return _evaluate_multiple_models_pydantic(models_list, _cv_wrapper, task_type)


# ============================================================
# Hold-out evaluation for multiple models (оставлено как в примере)
# ============================================================

def train_evaluate_models(
    models: List[Tuple[str, BaseEstimator]],
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    seed: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    all_metrics: Dict[str, Dict[str, Any]] = {}
    metrics_objects: List[BaseMetrics] = []

    for model_name, model in models:
        current_model = clone(model)
        res = train_evaluate_model(
            current_model, model_name, X_train, y_train, X_test, y_test, seed=seed, class_names=class_names
        )
        from models import GenericMetrics
        gm = GenericMetrics(values=res, name=model_name)
        metrics_objects.append(gm)
        all_metrics[model_name] = res

    plot_metrics_heatmap(metrics_objects)
    return pd.DataFrame.from_dict(all_metrics, orient="index")


# ============================================================
# Outliers
# ============================================================

def winsorize_outliers(
    df: pd.DataFrame,
    column_name: str,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> pd.DataFrame:
    df = df.copy()

    if lower_bound is not None:
        df.loc[df[column_name] < lower_bound, column_name] = lower_bound
    if upper_bound is not None:
        df.loc[df[column_name] > upper_bound, column_name] = upper_bound

    return df

def _build_classification_metrics_model(
    *,
    name: str,
    accuracy: float,
    precision_macro: float,
    recall_macro: float,
    f1_macro: float,
    roc_auc: Optional[float],
    confusion_matrix_value: Any = None,
    training_time: Optional[float] = None,
):
    fields = getattr(ClassificationMetrics, "model_fields", None) 
    if fields is None:
        fields = getattr(ClassificationMetrics, "__fields__", {})

    payload = {"name": name, "training_time": training_time}

    # два возможных формата models.py
    if "precision_macro" in fields:
        payload.update({
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "roc_auc_ovr_macro": roc_auc,
            "confusion_matrix": confusion_matrix_value,
        })
    else:
        payload.update({
            "accuracy": accuracy,
            "precision": precision_macro,
            "recall": recall_macro,
            "f1_score": f1_macro,
            "roc_auc": roc_auc,
            "confusion_matrix": confusion_matrix_value,
        })

    return ClassificationMetrics(**payload)
