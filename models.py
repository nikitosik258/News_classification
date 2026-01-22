from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator


class BaseMetrics(BaseModel):
    """
    Базовый контейнер метрик для любой модели.
    """
    name: Optional[str] = None
    training_time: Optional[float] = Field(default=None, description="Training time in seconds")
    estimators: Optional[Any] = Field(default=None, description="Trained estimators from cross-validation")

    def get_numeric_metrics(self) -> Dict[str, float]:
        raise NotImplementedError


# ============================================================
# Classification: multiclass-friendly
# ============================================================

class ClassificationReportRow(BaseModel):
    """
    Строка отчёта по одному классу.
    """
    class_label: str
    precision: float
    recall: float
    f1: float
    support: int


class RocCurveData(BaseModel):
    """
    Данные ROC-кривой (обычно для бинарной или OVR-формы).
    """
    fpr: Any
    tpr: Any
    thresholds: Any


class ClassificationMetrics(BaseMetrics):
    """
    Метрики мультиклассовой классификации.

    ВАЖНО:
    - поля f1_macro/precision_macro/recall_macro — основные для multi-class;
    - поля f1_score/precision/recall оставлены для совместимости (они == macro).
    """

    # --- Primary multiclass metrics ---
    accuracy: float

    f1_macro: float
    precision_macro: float
    recall_macro: float
    # --- Optional weighted metrics ---
    f1_weighted: Optional[float] = None
    precision_weighted: Optional[float] = None
    recall_weighted: Optional[float] = None

    # Proper multiclass ROC-AUC
    roc_auc_ovr_macro: Optional[float] = Field(
        default=None,
        description="ROC AUC for multiclass using one-vs-rest, macro-averaged"
    )

    # Confusion matrix (json-friendly)
    confusion_matrix: Optional[List[List[int]]] = None

    # Per-class report (precision/recall/f1/support)
    classification_report: Optional[List[ClassificationReportRow]] = None

    # Optional ROC curves:
    # - micro-average curve
    roc_curve_micro: Optional[RocCurveData] = None
    # - per-class OVR curves, key can be label like "0", "1", "Sports", etc.
    roc_curves_by_class: Optional[Dict[str, RocCurveData]] = None

    # --- Backward-compatible aliases (macro) ---
    roc_auc: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    roc_curve: Optional[RocCurveData] = None  # legacy single-curve slot

    # ---------------- Validators ----------------

    @validator(
        "accuracy",
        "f1_macro", "precision_macro", "recall_macro",
        "f1_weighted", "precision_weighted", "recall_weighted",
        "roc_auc_ovr_macro",
        "roc_auc", "f1_score", "precision", "recall",
        pre=True
    )
    def _ensure_float(cls, v: Any) -> float:  # type: ignore[override]
        if v is None:
            return float("nan")
        try:
            return float(v)
        except Exception:
            return float("nan")

    @validator("confusion_matrix", pre=True)
    def _ensure_confusion_matrix(cls, v: Any) -> Optional[List[List[int]]]:  # type: ignore[override]
        if v is None:
            return None
        # numpy array -> list
        if isinstance(v, np.ndarray):
            v = v.tolist()
        # pandas -> numpy -> list
        if isinstance(v, (pd.DataFrame, pd.Series)):
            v = np.asarray(v).tolist()

        # validate nested structure
        try:
            out: List[List[int]] = []
            for row in v:
                out.append([int(x) for x in row])
            return out
        except Exception:
            # fallback: try best-effort
            return None

    @validator("classification_report", pre=True)
    def _ensure_report_rows(cls, v: Any) -> Any:  # type: ignore[override]
        # allow passing sklearn dict or df-like structures; keep as-is if already correct
        return v

    @validator("roc_auc", always=True)
    def _fill_legacy_roc_auc(cls, v: Any, values: Dict[str, Any]) -> Any:
        """
        Если roc_auc (legacy) не задан, берём roc_auc_ovr_macro.
        """
        if v is None or (isinstance(v, float) and np.isnan(v)):
            auc = values.get("roc_auc_ovr_macro", None)
            return auc
        return v

    @validator("f1_score", always=True)
    def _fill_legacy_f1(cls, v: Any, values: Dict[str, Any]) -> Any:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return values.get("f1_macro", None)
        return v

    @validator("precision", always=True)
    def _fill_legacy_precision(cls, v: Any, values: Dict[str, Any]) -> Any:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return values.get("precision_macro", None)
        return v

    @validator("recall", always=True)
    def _fill_legacy_recall(cls, v: Any, values: Dict[str, Any]) -> Any:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return values.get("recall_macro", None)
        return v

    # ---------------- Public API ----------------

    def get_numeric_metrics(self) -> Dict[str, float]:
        """
        Возвращаем словарь числовых метрик.
        ВАЖНО: оставляем старые ключи (F1 Score/Precision/Recall/ROC AUC) как macro,
        чтобы не ломать старый код, и добавляем новые явные ключи.
        """
        metrics: Dict[str, float] = {
            # Backward-compatible keys (interpreted as macro)
            "ROC AUC": float(self.roc_auc) if self.roc_auc is not None else float("nan"),
            "F1 Score": float(self.f1_score) if self.f1_score is not None else float("nan"),
            "Precision": float(self.precision) if self.precision is not None else float("nan"),
            "Recall": float(self.recall) if self.recall is not None else float("nan"),
            "Accuracy": float(self.accuracy),
        }

        if self.training_time is not None:
            metrics["Training Time (s)"] = float(self.training_time)

        return metrics

    def to_compact_dict(self) -> Dict[str, Any]:
        """
        Минимальный словарь (без тяжёлых массивов).
        """
        return self.get_numeric_metrics()

    def to_plot_dict(self) -> Dict[str, Any]:
        """
        Словарь под plotting.
        Пока возвращаем конфьюжн-матрицу и (если есть) micro ROC + legacy ROC.
        Пер-класс ROC-кривые тоже кладем, но они потребуют мультикласс-plotter.
        """
        out: Dict[str, Any] = {}

        if self.confusion_matrix is not None:
            out["Confusion Matrix"] = self.confusion_matrix

        # legacy single-curve
        if self.roc_curve is not None and self.roc_auc is not None:
            out["ROC Curve"] = {
                "fpr": self.roc_curve.fpr,
                "tpr": self.roc_curve.tpr,
                "thresholds": self.roc_curve.thresholds,
            }
            out["ROC AUC"] = self.roc_auc

        # micro-average curve (multiclass)
        if self.roc_curve_micro is not None and self.roc_auc_ovr_macro is not None:
            out["ROC Curve (micro)"] = {
                "fpr": self.roc_curve_micro.fpr,
                "tpr": self.roc_curve_micro.tpr,
                "thresholds": self.roc_curve_micro.thresholds,
            }
            out["ROC AUC OVR Macro"] = self.roc_auc_ovr_macro

        # per-class curves (OVR)
        if self.roc_curves_by_class is not None:
            out["ROC Curves (per class)"] = {
                k: {"fpr": v.fpr, "tpr": v.tpr, "thresholds": v.thresholds}
                for k, v in self.roc_curves_by_class.items()
            }

        return out

    def to_report_dict(self) -> Dict[str, Any]:
        """
        Словарь для печати/табличного отчёта.
        """
        report: Dict[str, Any] = {
            "Accuracy": self.accuracy,
            "F1 Macro": self.f1_macro,
            "Precision Macro": self.precision_macro,
            "Recall Macro": self.recall_macro,
            "ROC AUC OVR Macro": self.roc_auc_ovr_macro,
        }

        if self.f1_weighted is not None:
            report["F1 Weighted"] = self.f1_weighted
        if self.precision_weighted is not None:
            report["Precision Weighted"] = self.precision_weighted
        if self.recall_weighted is not None:
            report["Recall Weighted"] = self.recall_weighted

        if self.classification_report is not None:
            report["Classification Report"] = {
                "Class": [row.class_label for row in self.classification_report],
                "Precision": [row.precision for row in self.classification_report],
                "Recall": [row.recall for row in self.classification_report],
                "F1": [row.f1 for row in self.classification_report],
                "Support": [row.support for row in self.classification_report],
            }

        return report


# ============================================================
# Regression (оставлено для совместимости, но проекту не мешает)
# ============================================================

class RegressionMetrics(BaseMetrics):
    mae: float
    mse: float
    rmse: float
    r2: float
    explained_variance: Optional[float] = None

    def _as_float(self, v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    def get_numeric_metrics(self) -> Dict[str, float]:
        data: Dict[str, float] = {
            "MAE": self._as_float(self.mae),
            "MSE": self._as_float(self.mse),
            "RMSE": self._as_float(self.rmse),
            "R2": self._as_float(self.r2),
        }
        if self.explained_variance is not None:
            data["Explained Variance"] = self._as_float(self.explained_variance)
        if self.training_time is not None:
            data["Training Time (s)"] = float(self.training_time)
        return data


class GenericMetrics(BaseMetrics):
    values: Dict[str, float]

    def get_numeric_metrics(self) -> Dict[str, float]:
        return dict(self.values)


# ============================================================
# Multiple results container
# ============================================================

class MultipleModelResults(BaseModel):
    """
    Контейнер для набора результатов по моделям.
    """
    results: List[BaseMetrics] = Field(description="List of model evaluation results")
    task_type: str = Field(description="Type of task: 'classification' or 'regression'")

    def to_dataframe(self) -> pd.DataFrame:
        all_metrics: Dict[str, Dict[str, float]] = {}
        for result in self.results:
            model_name = result.name if result.name else "Unknown Model"
            all_metrics[model_name] = result.get_numeric_metrics()
        return pd.DataFrame.from_dict(all_metrics, orient="index")

    def get_model_names(self) -> List[str]:
        return [result.name for result in self.results if result.name]

    def get_best_model(self, metric: str = None) -> Optional[BaseMetrics]:
        """
        Для классификации по умолчанию выбираем по F1 Macro (а не ROC AUC),
        т.к. это более устойчиво и обычно основной KPI для multi-class текста.
        """
        if not self.results:
            return None

        if metric is None:
            metric = "F1 Macro" if self.task_type == "classification" else "R2"

        best_model = None
        best_score = float("-inf")

        for result in self.results:
            metrics = result.get_numeric_metrics()
            if metric in metrics:
                score = metrics[metric]
                if not np.isnan(score) and score > best_score:
                    best_score = score
                    best_model = result

        return best_model
