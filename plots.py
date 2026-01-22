# plots.py
# Plotting utilities for your text classification project (multi-class first).
# Works with your updated models.py (ClassificationMetrics, RegressionMetrics).

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional deps (keep project runnable even if missing)
try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None  # type: ignore

try:
    from wordcloud import WordCloud
except Exception:  # pragma: no cover
    WordCloud = None  # type: ignore

from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import BaseEstimator
from sklearn.tree import plot_tree

from models import ClassificationMetrics, RegressionMetrics


# ============================================================
# Common config
# ============================================================

DEFAULT_FIGSIZE = (8, 6)
DEFAULT_PALETTE = "viridis"
DEFAULT_GRID_ALPHA = 0.3


def _require_seaborn() -> None:
    if sns is None:
        raise ImportError("seaborn is required for this plot. Install it via: pip install seaborn")


def _setup_plot_style(figsize: Tuple[int, int] = DEFAULT_FIGSIZE) -> None:
    plt.figure(figsize=figsize)
    plt.grid(alpha=DEFAULT_GRID_ALPHA)


def _finalize_plot(title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    if title:
        plt.title(title, fontsize=14, fontweight="bold")
    if xlabel:
        plt.xlabel(xlabel, fontsize=12)
    if ylabel:
        plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================
# EDA plots
# ============================================================

def plot_hist_numeric(
    data: pd.DataFrame,
    feature: str,
    figsize: Tuple[int, int] = (8, 4),
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> None:
    _require_seaborn()
    filtered = data.copy()
    if x_min is not None:
        filtered = filtered[filtered[feature] >= x_min]
    if x_max is not None:
        filtered = filtered[filtered[feature] <= x_max]

    _setup_plot_style(figsize)
    sns.histplot(filtered[feature], kde=True)
    _finalize_plot(f"Distribution of {feature}", feature, "Frequency")


def barplot(
    category_counts: pd.Series,
    title: str,
    ylabel: str,
    figsize: Tuple[int, int] = (4, 6),
    top_n: Optional[int] = None,
    color_palette: str = DEFAULT_PALETTE,
) -> None:
    _require_seaborn()
    if top_n is not None and len(category_counts) > top_n:
        plot_data = category_counts.nlargest(top_n)
    else:
        plot_data = category_counts

    plt.figure(figsize=figsize)
    plt.grid(axis="x", alpha=DEFAULT_GRID_ALPHA)
    sns.barplot(
        x=plot_data.values,
        y=plot_data.index,
        hue=plot_data.index,
        palette=color_palette,
        orient="h",
        legend=False,
        dodge=False,
    )
    _finalize_plot(title, "Frequency", ylabel)


def plot_hist_categorical(
    data: pd.DataFrame,
    feature: str,
    figsize: Tuple[int, int] = (4, 4),
) -> None:
    counts = data[feature].value_counts().sort_values(ascending=False)
    barplot(counts, f"Distribution of {feature}", feature, figsize)


def plot_categorical_relationship(df: pd.DataFrame, col1: str, col2: str) -> None:
    _require_seaborn()
    count_crosstab = pd.crosstab(df[col1], df[col2])
    row_prop = pd.crosstab(df[col1], df[col2], normalize="index")
    col_prop = pd.crosstab(df[col1], df[col2], normalize="columns")

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    sns.heatmap(count_crosstab, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f"Абсолютные значения\n{col1} vs {col2}")
    axes[0].set_xlabel(col2)
    axes[0].set_ylabel(col1)

    sns.heatmap(row_prop, annot=True, fmt=".2f", cmap="Greens", ax=axes[1])
    axes[1].set_title(f"Доли внутри {col1} (по строкам)")
    axes[1].set_xlabel(col2)
    axes[1].set_ylabel(col1)

    sns.heatmap(col_prop, annot=True, fmt=".2f", cmap="Oranges", ax=axes[2])
    axes[2].set_title(f"Доли внутри {col2} (по столбцам)")
    axes[2].set_xlabel(col2)
    axes[2].set_ylabel(col1)

    plt.tight_layout()
    plt.show()


def plot_numeric_relationship(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    target_col: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> None:
    """
    Scatter plot y_col vs x_col.
    If target_col is provided -> color by target (supports multi-class).
    """
    _require_seaborn()

    if not pd.api.types.is_numeric_dtype(df[x_col]):
        raise TypeError(f"{x_col} не является числовой переменной.")
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise TypeError(f"{y_col} не является числовой переменной.")

    plt.figure(figsize=figsize)

    if target_col is not None:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=target_col)
        plt.legend(title=target_col)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col)

    if x_min is not None or x_max is not None:
        plt.xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)

    plt.title(f"Зависимость {y_col} от {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# Classification plots (MULTI-CLASS)
# ============================================================

def _infer_class_labels(
    metrics: ClassificationMetrics,
    class_names: Optional[List[str]] = None,
) -> List[str]:
    """
    Priority:
      1) explicit class_names if provided
      2) classification_report rows (class_label)
      3) numeric indices from confusion_matrix size
    """
    if class_names is not None and len(class_names) > 0:
        return [str(x) for x in class_names]

    if metrics.classification_report is not None and len(metrics.classification_report) > 0:
        return [str(r.class_label) for r in metrics.classification_report]

    if metrics.confusion_matrix is not None:
        k = len(metrics.confusion_matrix)
        return [str(i) for i in range(k)]

    return []


def plot_classification_results(
    metrics: ClassificationMetrics,
    model_name: str = "Model",
    *,
    class_names: Optional[List[str]] = None,
    show_roc: bool = True,
    roc_max_classes: int = 8,
    figsize: Tuple[int, int] = (16, 6),
) -> None:
    """
    Multi-class friendly:
      - Confusion matrix with correct labels
      - ROC: OVR per class (if available) + micro curve (optional)
    """
    _require_seaborn()

    labels = _infer_class_labels(metrics, class_names=class_names)

    has_cm = metrics.confusion_matrix is not None
    has_multiclass_roc = metrics.roc_curves_by_class is not None and len(metrics.roc_curves_by_class) > 0
    has_micro_roc = metrics.roc_curve_micro is not None
    has_legacy_roc = metrics.roc_curve is not None and metrics.roc_auc is not None

    want_roc = bool(show_roc) and (has_multiclass_roc or has_micro_roc or has_legacy_roc)

    if has_cm and want_roc:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    elif has_cm and not want_roc:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    elif (not has_cm) and want_roc:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    else:
        print("Nothing to plot: confusion_matrix and ROC curves are missing.")
        return

    # --- Confusion matrix ---
    if has_cm:
        ax = axes[0]
        cm = np.asarray(metrics.confusion_matrix, dtype=int)

        tick = labels if len(labels) == cm.shape[0] else [str(i) for i in range(cm.shape[0])]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=tick, yticklabels=tick)
        ax.set_title(f"{model_name} - Confusion Matrix", fontsize=14)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)

    # --- ROC ---
    if want_roc:
        ax = axes[1] if has_cm and len(axes) > 1 else axes[0]

        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"{model_name} - ROC (OVR)", fontsize=14)

        # Multiclass curves (per class)
        if has_multiclass_roc:
            items = list(metrics.roc_curves_by_class.items())
            if len(items) > roc_max_classes:
                items = items[:roc_max_classes]

            for cls_name, curve in items:
                ax.plot(curve.fpr, curve.tpr, linewidth=2, label=f"{cls_name}")

        # Micro-average
        if has_micro_roc:
            curve = metrics.roc_curve_micro
            ax.plot(curve.fpr, curve.tpr, linewidth=2, label="micro-average")

        # Legacy binary curve (fallback)
        if (not has_multiclass_roc) and (not has_micro_roc) and has_legacy_roc:
            curve = metrics.roc_curve
            ax.plot(curve.fpr, curve.tpr, linewidth=2, label=f"AUC={metrics.roc_auc:.3f}")

        # AUC text
        auc_text = None
        if metrics.roc_auc_ovr_macro is not None and not np.isnan(metrics.roc_auc_ovr_macro):
            auc_text = f"AUC(OVR macro)={metrics.roc_auc_ovr_macro:.3f}"
        elif metrics.roc_auc is not None and not np.isnan(metrics.roc_auc):
            auc_text = f"AUC={metrics.roc_auc:.3f}"
        if auc_text is not None:
            ax.text(0.60, 0.05, auc_text, transform=ax.transAxes)

        ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.show()


def print_classification_report(
    metrics: Union[ClassificationMetrics, Dict[str, Any]],
    model_name: str = "Model",
) -> None:
    """
    Accepts:
      - ClassificationMetrics object
      - dict from metrics.to_report_dict()
    """
    if isinstance(metrics, ClassificationMetrics):
        report = metrics.to_report_dict()
        num = metrics.get_numeric_metrics()
    else:
        report = metrics
        num = metrics  # best-effort

    main_rows = []
    for k in ["Accuracy", "F1 Macro", "Precision Macro", "Recall Macro", "ROC AUC OVR Macro", "F1 Weighted"]:
        if k in report and report[k] is not None:
            v = report[k]
            if isinstance(v, (int, float, np.floating)) and not np.isnan(v):
                main_rows.append((k, float(v)))

    # Backward compatible keys (if someone passes dict from get_numeric_metrics)
    for k in ["F1 Score", "Precision", "Recall", "ROC AUC"]:
        if k in num and num[k] is not None:
            v = num[k]
            if isinstance(v, (int, float, np.floating)) and not np.isnan(v):
                # avoid duplicates
                if not any(r[0] == k for r in main_rows):
                    main_rows.append((k, float(v)))

    metrics_df = pd.DataFrame(main_rows, columns=["Metric", "Value"])
    if not metrics_df.empty:
        metrics_df["Value"] = metrics_df["Value"].map(lambda x: f"{x:.4f}")

    class_df = None
    if "Classification Report" in report and report["Classification Report"] is not None:
        class_df = pd.DataFrame(report["Classification Report"])

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} EVALUATION".center(70))
    print("=" * 70)

    if metrics_df.empty:
        print("\nMAIN METRICS: N/A")
    else:
        print("\nMAIN METRICS:")
        print(metrics_df.to_string(index=False))

    if class_df is not None and not class_df.empty:
        print("\n\nPER-CLASS REPORT:")
        print(class_df.to_string(index=False))

    print("\n" + "=" * 70)


def plot_training_history(
    history: Union[List[Dict[str, Any]], pd.DataFrame],
    *,
    title: str = "Training History",
    figsize: Tuple[int, int] = (10, 5),
    metrics_to_plot: Optional[List[str]] = None,
) -> None:
    """
    history: list of dicts (как в fit() из pytorch helper) или DataFrame.
    По умолчанию рисуем train_loss и val_f1_macro / val_accuracy если есть.
    """
    if isinstance(history, list):
        df = pd.DataFrame(history)
    else:
        df = history.copy()

    if df.empty:
        print("Empty history.")
        return

    if metrics_to_plot is None:
        candidates = ["train_loss", "val_F1 Macro", "val_Accuracy", "val_F1 Score", "val_Accuracy"]
        metrics_to_plot = [c for c in candidates if c in df.columns]
        # common alt keys from our pytorch helper output
        if "val_f1_macro" in df.columns and "val_F1 Macro" not in df.columns:
            metrics_to_plot.append("val_f1_macro")
        if "val_accuracy" in df.columns and "val_Accuracy" not in df.columns:
            metrics_to_plot.append("val_accuracy")

    plt.figure(figsize=figsize)
    for col in metrics_to_plot:
        plt.plot(df["epoch"], df[col], marker="o", label=col)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Feature importance / Trees / Hyperparam tuning (sklearn baselines)
# ============================================================

def plot_feature_importance(
    model: BaseEstimator,
    feature_names: List[str],
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6),
    model_type: str = "auto",
) -> pd.DataFrame:
    _require_seaborn()

    if model_type == "auto":
        if hasattr(model, "feature_importances_"):
            model_type = "tree"
        elif hasattr(model, "coef_"):
            model_type = "linear"
        else:
            raise ValueError("Could not determine model type. Specify 'tree' or 'linear'.")

    if model_type == "tree":
        importances = model.feature_importances_
        importance_label = "Feature Importance"
    elif model_type == "linear":
        if len(model.coef_.shape) > 1:  # multi-class
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            importances = np.abs(model.coef_[0])
        importance_label = "Absolute Coefficient"
    else:
        raise ValueError("model_type must be either 'tree' or 'linear'")

    feature_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
        "Importance", ascending=False
    )

    if top_n is not None:
        feature_imp = feature_imp.head(top_n)

    plt.figure(figsize=figsize)
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_imp,
        hue="Feature",
        palette="viridis",
        legend=False,
    )
    plt.title(f"Feature Importances ({model_type} model)")
    plt.xlabel(importance_label)
    plt.tight_layout()
    plt.show()

    return feature_imp


def visualize_decision_tree(
    model: BaseEstimator,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (20, 10),
    max_depth: Optional[int] = None,
) -> None:
    plt.figure(figsize=figsize)
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=True,
        max_depth=max_depth,
    )
    plt.title("Decision Tree Visualization")
    plt.show()


def plot_hyperparam_search_results(
    results: Union[Dict[str, Any], pd.DataFrame],
    score_key: str = "mean_test_score",
    title: str = "Hyperparameter Tuning Results",
    xtick_step: int = 5,
) -> pd.DataFrame:
    if isinstance(results, dict):
        params = results.get("params")
        scores = results.get(score_key)
        if params is None or scores is None:
            raise ValueError(f"'params' and '{score_key}' must exist in results dict.")
        df = pd.DataFrame(params)
        df[score_key] = scores
    elif isinstance(results, pd.DataFrame):
        if "params" in results.columns:
            df = pd.DataFrame(results["params"].tolist())
            df[score_key] = results[score_key].values
        else:
            raise ValueError("DataFrame input must have a 'params' column.")
    else:
        raise TypeError("results must be a dict (like cv_results_) or a DataFrame.")

    df = df.reset_index().rename(columns={"index": "Set #"})
    best_idx = df[score_key].idxmax()
    best_score = df.loc[best_idx, score_key]

    plt.figure(figsize=(12, 6))
    x = df["Set #"]
    y = df[score_key]
    plt.plot(x, y, marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel("Hyperparameter Set #")
    plt.ylabel(score_key)
    plt.grid(True)
    plt.xticks(ticks=x[::xtick_step])

    plt.plot(df.loc[best_idx, "Set #"], best_score, "ro", label=f"Best: {best_score:.4f}")
    plt.annotate(
        f"Best\n{best_score:.4f}",
        xy=(df.loc[best_idx, "Set #"], best_score),
        xytext=(df.loc[best_idx, "Set #"], best_score + 0.02),
        arrowprops=dict(facecolor="red", shrink=0.05),
        ha="center",
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    return df


# ============================================================
# Compare metrics heatmap (baseline vs NN etc.)
# ============================================================

def compare_metrics_heatmap(
    df1: Any,
    df2: Any,
    df1_name: str = "DF1",
    df2_name: str = "DF2",
    figsize: Tuple[int, int] = (8, 4),
    annot_fontsize: int = 10,
    title: str = "Comparison of ML Metrics",
    lower_is_better_metrics: Optional[List[str]] = None,
) -> Tuple[Any, pd.DataFrame]:
    """
    Compares two metrics DataFrames and draws a semantic delta heatmap:
      - green = improvement, red = degradation

    Accepts:
      - pd.DataFrame
      - MultipleModelResults-like objects (pydantic) returned by helper.py
    """
    _require_seaborn()

    def _to_metrics_df(obj: Any) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            return obj

        # 1) Common direct attributes
        for attr in ("metrics_df", "df", "dataframe", "summary_df", "table"):
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if isinstance(val, pd.DataFrame):
                    return val

        # 2) If it stores per-model metrics list
        for attr in ("results", "items", "models", "metrics_list"):
            if hasattr(obj, attr):
                lst = getattr(obj, attr)
                try:
                    rows = []
                    for x in lst:
                        if hasattr(x, "get_numeric_metrics"):
                            rows.append(x.get_numeric_metrics())
                        elif hasattr(x, "model_dump"):
                            rows.append(x.model_dump())
                        elif isinstance(x, dict):
                            rows.append(x)
                    df = pd.DataFrame(rows)

                    # Make sure model names appear as index if possible
                    if "Model Name" in df.columns:
                        df = df.set_index("Model Name")
                    elif "name" in df.columns:
                        df = df.set_index("name")

                    return df
                except Exception:
                    pass

        # 3) Fallback: try pydantic dump of the outer object
        if hasattr(obj, "model_dump"):
            d = obj.model_dump()
            # sometimes the list is nested
            for key in ("results", "items", "models", "metrics"):
                if isinstance(d, dict) and key in d and isinstance(d[key], list):
                    rows = []
                    for x in d[key]:
                        if isinstance(x, dict):
                            rows.append(x)
                    df = pd.DataFrame(rows)
                    if "Model Name" in df.columns:
                        df = df.set_index("Model Name")
                    elif "name" in df.columns:
                        df = df.set_index("name")
                    return df

        raise TypeError("compare_metrics_heatmap expects a pandas DataFrame or MultipleModelResults-like object.")

    df1 = _to_metrics_df(df1)
    df2 = _to_metrics_df(df2)

    # keep only numeric columns for heatmap
    df1 = df1.select_dtypes(include=[np.number]).copy()
    df2 = df2.select_dtypes(include=[np.number]).copy()

    # align by rows/models
    common_idx = df1.index.intersection(df2.index)
    df1 = df1.loc[common_idx]
    df2 = df2.loc[common_idx]

    if lower_is_better_metrics is None:
        patterns = ["time", "loss", "error", "cost", "latency", "duration", "mse", "mae", "rmse", "runtime", "seconds"]
        lower_is_better_metrics = [c for c in df1.columns if any(p in c.lower() for p in patterns)]

    common_cols = [c for c in df1.columns if c in df2.columns]
    df1c = df1[common_cols].copy()
    df2c = df2[common_cols].copy()

    delta = df2c - df1c

    semantic_delta = delta.copy()
    for col in lower_is_better_metrics:
        if col in semantic_delta.columns:
            semantic_delta[col] = -semantic_delta[col]

    normalized = semantic_delta.copy()
    for col in semantic_delta.columns:
        col_values = semantic_delta[col]
        col_min = col_values.min()
        col_max = col_values.max()
        if col_max != col_min:
            norm_col = col_values.copy()
            pos = col_values > 0
            neg = col_values < 0
            if col_max > 0 and pos.any():
                norm_col[pos] = col_values[pos] / col_max
            if col_min < 0 and neg.any():
                norm_col[neg] = col_values[neg] / abs(col_min)
            normalized[col] = norm_col
        else:
            normalized[col] = 0

    colors = ["#ff2700", "#ffffff", "#00b975"]
    cmap = LinearSegmentedColormap.from_list("rwg", colors)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        normalized,
        annot=delta,
        fmt=".3f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": annot_fontsize},
        cbar_kws={"label": "Improvement (Green) ← → Degradation (Red)"},
    )

    ax.set_title(title, pad=20, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    return fig, delta


# ============================================================
# Token count analysis plots (from nlp.count_based_analysis output)
# ============================================================

def plot_count_based_analysis(
    df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    plot_type: str = "heatmap",
    sort_by_class: Optional[Union[str, int]] = None,
) -> None:
    _require_seaborn()

    if "token" not in df.columns:
        raise ValueError("DataFrame must contain 'token' column")

    count_cols = [c for c in df.columns if c.startswith("count_") and c != "total_count"]
    freq_cols = [c for c in df.columns if c.startswith("freq_")]

    if not count_cols:
        raise ValueError("No count columns found in DataFrame")

    classes = [c.replace("count_", "") for c in count_cols]

    if sort_by_class is not None:
        sort_col = f"freq_{sort_by_class}"
        if sort_col not in df.columns:
            raise ValueError(f"Class '{sort_by_class}' not found. Available classes: {classes}")
        sorted_df = df.sort_values(sort_col, ascending=False).copy()
    else:
        sorted_df = df.sort_values("total_count", ascending=False).copy() if "total_count" in df.columns else df.copy()

    top_df = sorted_df.head(top_n).copy()

    if plot_type == "heatmap":
        _plot_count_heatmap(top_df, freq_cols, classes, figsize, sort_by_class)
    elif plot_type == "bar":
        _plot_count_bars(top_df, count_cols, classes, figsize, sort_by_class)
    elif plot_type == "stacked_bar":
        _plot_stacked_bars(top_df, count_cols, classes, figsize, sort_by_class)
    else:
        raise ValueError("plot_type must be one of: 'heatmap', 'bar', 'stacked_bar'")


def _plot_count_heatmap(
    df: pd.DataFrame,
    freq_cols: List[str],
    classes: List[str],
    figsize: Tuple[int, int],
    sort_by_class: Optional[Union[str, int]] = None,
) -> None:
    _require_seaborn()

    plot_df = df[["token"] + freq_cols].set_index("token")
    plt.figure(figsize=figsize)
    sns.heatmap(plot_df, cmap="YlGnBu", linewidths=0.5)
    title = "Token Frequency Across Classes"
    if sort_by_class is not None:
        title += f" (sorted by Class {sort_by_class})"
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Tokens", fontsize=12)
    plt.xlabel("Classes", fontsize=12)
    plt.tight_layout()
    plt.show()


def _plot_count_bars(
    df: pd.DataFrame,
    count_cols: List[str],
    classes: List[str],
    figsize: Tuple[int, int],
    sort_by_class: Optional[Union[str, int]] = None,
) -> None:
    tokens = df["token"].tolist()
    data = df[count_cols].copy()

    y = np.arange(len(tokens))
    height = 0.8 / max(1, len(classes))

    plt.figure(figsize=figsize)
    for i, (col, cls) in enumerate(zip(count_cols, classes)):
        plt.barh(y + i * height, data[col], height, label=f"Class {cls}", alpha=0.85)

    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Tokens", fontsize=12)

    title = "Token Counts Across Classes"
    if sort_by_class is not None:
        title += f" (sorted by Class {sort_by_class})"
    plt.title(title, fontsize=14, fontweight="bold")

    plt.yticks(y + height * (len(classes) - 1) / 2, tokens)
    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_stacked_bars(
    df: pd.DataFrame,
    count_cols: List[str],
    classes: List[str],
    figsize: Tuple[int, int],
    sort_by_class: Optional[Union[str, int]] = None,
) -> None:
    tokens = df["token"].tolist()
    data = df[count_cols].copy()

    plt.figure(figsize=figsize)
    left = np.zeros(len(tokens))

    for col, cls in zip(count_cols, classes):
        plt.barh(tokens, data[col], left=left, label=f"Class {cls}", alpha=0.85)
        left += data[col].values

    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Tokens", fontsize=12)

    title = "Stacked Token Counts Across Classes"
    if sort_by_class is not None:
        title += f" (sorted by Class {sort_by_class})"
    plt.title(title, fontsize=14, fontweight="bold")

    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# Wordcloud
# ============================================================

def plot_wordcloud(
    token_counts: pd.Series,
    title: str = "Word Cloud",
    figsize: Tuple[int, int] = (12, 8),
    max_words: int = 100,
    background_color: str = "white",
    colormap: str = "viridis",
    width: int = 800,
    height: int = 400,
) -> None:
    if WordCloud is None:
        raise ImportError("wordcloud is required for this plot. Install it via: pip install wordcloud")

    word_freq = token_counts.to_dict()
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
        relative_scaling=0.5,
        random_state=42,
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=figsize)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()


# ============================================================
# Regression plots (kept for compatibility)
# ============================================================

def plot_regression_results(metrics: RegressionMetrics, model_name: str = "Model") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{model_name} - Regression Results", fontsize=16, fontweight="bold")

    data = metrics.get_numeric_metrics()
    main = {k: v for k, v in data.items() if k != "Training Time (s)"}

    names = list(main.keys())
    vals = list(main.values())

    axes[0, 0].bar(names, vals, alpha=0.7)
    axes[0, 0].set_title("Regression Metrics")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].tick_params(axis="x", rotation=45)

    for i, v in enumerate(vals):
        axes[0, 0].text(i, v + (max(vals) * 0.01 if len(vals) else 0.01), f"{v:.3f}", ha="center", va="bottom")

    r2 = float(data.get("R2", 0))
    axes[0, 1].bar(["R²"], [r2], alpha=0.7)
    axes[0, 1].set_title("R² Score")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].text(0, r2 + 0.02, f"{r2:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    err_names = ["MAE", "MSE", "RMSE"]
    err_vals = [float(data.get(k, 0)) for k in err_names]
    axes[1, 0].bar(err_names, err_vals, alpha=0.7)
    axes[1, 0].set_title("Error Metrics")
    axes[1, 0].tick_params(axis="x", rotation=45)

    for i, v in enumerate(err_vals):
        axes[1, 0].text(i, v + (max(err_vals) * 0.01 if len(err_vals) else 0.01), f"{v:.3f}", ha="center", va="bottom")

    if "Training Time (s)" in data:
        tt = float(data["Training Time (s)"])
        axes[1, 1].bar(["Training Time (s)"], [tt], alpha=0.7)
        axes[1, 1].set_title("Training Time")
        axes[1, 1].text(0, tt + (tt * 0.02 if tt else 0.02), f"{tt:.1f}", ha="center", va="bottom")
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()
