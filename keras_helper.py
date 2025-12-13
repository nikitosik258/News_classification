from __future__ import annotations

from typing import Any, Callable, List, MutableMapping, Optional, Sequence, Union

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from time import perf_counter
from helper import evaluate_classification, aggregate_classification_cv_metrics
from models import ClassificationMetrics
import plots as p

def _safe_index(data: Any, indices: Union[Sequence[int], np.ndarray]) -> Any:
    """Index arrays, pandas objects, or sequences safely by integer indices."""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    try:
        return data[indices]
    except Exception:
        return np.asarray(data)[indices]

 


def cross_validate_model(
    model_builder: Callable[[], Any],
    X: Any,
    y: Any,
    *,
    cv: StratifiedKFold,
    fit_params: Optional[MutableMapping[str, Any]] = None,
    preprocessor: Optional[Any] = None,
) -> ClassificationMetrics:
    """Classification cross-validation for Keras models reusing helper evaluation.

    Returns ClassificationMetrics with mean metrics across folds and estimators.
    """

    fit_times: List[float] = []
    accs: List[float] = []
    f1s: List[float] = []
    precs: List[float] = []
    recalls: List[float] = []
    aucs: List[float] = []
    estimators: List[Any] = []

    X_array = X
    y_array = y

    # Out-of-fold containers for post-CV evaluation/plotting
    n_samples = len(y_array)
    oof_pred: np.ndarray = np.empty(n_samples, dtype=int)
    oof_probs: np.ndarray = np.empty(n_samples, dtype=float)
    has_probs: bool = True

    # Iterate folds
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_array, y_array)):
        print(f"Fold {fold_idx + 1}")


        # Split
        X_train, y_train = _safe_index(X_array, train_idx), _safe_index(y_array, train_idx)
        X_test, y_test = _safe_index(X_array, test_idx), _safe_index(y_array, test_idx)

        # Preprocess per fold
        fitted_pre = clone(preprocessor)
        if fitted_pre is not None:
            X_train_proc = fitted_pre.fit_transform(X_train, y_train)
            X_test_proc = fitted_pre.transform(X_test)
        else:
            X_train_proc, X_test_proc = X_train, X_test

        # Build model per fold
        model = model_builder()

        model.summary()

        # Fit
        start_fit = float(perf_counter())
        if fit_params:
            model.fit(X_train_proc, y_train, **fit_params)
        else:
            model.fit(X_train_proc, y_train)
        fit_times.append(float(perf_counter() - start_fit))

        # Score (test)
        y_score = np.asarray(model.predict(X_test_proc))
        y_pred = _labels_from_score(y_score)

        # Use helper evaluation for per-fold metrics (no plots)
        y_probs_binary = _positive_class_probabilities(y_score)
        fold_metrics: ClassificationMetrics = evaluate_classification(
            y_test=y_test,
            y_pred=y_pred,
            y_probs=y_probs_binary,
            model_name=f"Fold {fold_idx + 1}",
            enable_plot=False,
        )

        accs.append(float(fold_metrics.accuracy))
        f1s.append(float(fold_metrics.f1_score))
        precs.append(float(fold_metrics.precision))
        recalls.append(float(fold_metrics.recall))
        aucs.append(float(fold_metrics.roc_auc) if fold_metrics.roc_auc is not None else float('nan'))

        estimators.append({"preprocessor": fitted_pre, "model": model})

        # Collect OOF predictions
        oof_pred[test_idx] = y_pred
        if y_probs_binary is not None:
            oof_probs[test_idx] = y_probs_binary
        else:
            has_probs = False

    # Create final ClassificationMetrics using shared aggregator
    final_metrics = aggregate_classification_cv_metrics(
        accuracy=float(np.nanmean(accs)) if len(accs) else float('nan'),
        precision=float(np.nanmean(precs)) if len(precs) else float('nan'),
        recall=float(np.nanmean(recalls)) if len(recalls) else float('nan'),
        f1_score_value=float(np.nanmean(f1s)) if len(f1s) else float('nan'),
        roc_auc=float(np.nanmean(aucs)) if len(aucs) else None,
        training_time=float(np.nansum(fit_times)),
        name="Keras CV",
        y_true=y_array,
        y_pred=oof_pred,
        y_probs=oof_probs if has_probs else None,
    )
    # Attach estimators list for downstream use
    final_metrics.estimators = estimators

    # Plot results
    p.plot_classification_results(final_metrics, model_name="Keras CV")
    
    return final_metrics


def _labels_from_score(y_score: np.ndarray) -> np.ndarray:
    if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
        return (y_score.ravel() >= 0.5).astype(int)
    return np.argmax(y_score, axis=1)


def _positive_class_probabilities(y_score: np.ndarray) -> Optional[np.ndarray]:
    """Return probabilities for the positive class for binary outputs; otherwise None.

    Supports shapes (n,), (n,1), or (n,2). Returns None for multi-class (>2 columns).
    """
    if y_score.ndim == 1:
        return y_score.ravel()
    if y_score.ndim == 2 and y_score.shape[1] == 1:
        return y_score.ravel()
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        return y_score[:, 1]
    return None

__all__ = ["cross_validate_model"]


 