# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Callable, List, MutableMapping, Optional, Sequence, Union, Dict

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from time import perf_counter
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

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


class _NumpyLikeDataset(Dataset):
    """
    Минимальный Dataset для (X, y), где X и y индексируются по integer.
    По умолчанию пытается конвертировать X в torch.Tensor, если это возможно.
    Для сложных X (например, список токенов разной длины) рекомендуется передать
    custom dataset_cls/collate_fn через fit_params.
    """

    def __init__(self, X: Any, y: Any):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        x = _safe_index(self.X, [idx])
        # _safe_index с list вернёт "срез" из 1 элемента; приведём к одному элементу
        if hasattr(x, "__len__") and len(x) == 1:
            x = x[0]
        y = _safe_index(self.y, [idx])
        if hasattr(y, "__len__") and len(y) == 1:
            y = y[0]

        # Попытка "по-умолчанию" превратить в тензоры
        x_t = x
        if not torch.is_tensor(x):
            try:
                x_t = torch.as_tensor(x)
            except Exception:
                # Оставляем как есть (например, строка или список переменной длины)
                x_t = x

        y_t = y
        if not torch.is_tensor(y):
            try:
                y_t = torch.as_tensor(y)
            except Exception:
                y_t = y

        return x_t, y_t


def _default_collate(batch):
    """
    Дефолтный collate: если элементы уже тензоры одинаковой формы — stack.
    Если X не тензоры/разные формы — возвращаем списком (тогда модель должна уметь).
    """
    xs, ys = zip(*batch)

    # y почти всегда можно в тензор
    try:
        ys_t = torch.stack([y if torch.is_tensor(y) else torch.as_tensor(y) for y in ys]).squeeze()
    except Exception:
        ys_t = ys

    # x: пытаемся стекать, иначе списком
    try:
        xs_t = torch.stack([x if torch.is_tensor(x) else torch.as_tensor(x) for x in xs])
    except Exception:
        xs_t = list(xs)

    return xs_t, ys_t


def _infer_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    """
    Приводим выход модели к вероятностям.
    - binary: (n,), (n,1) -> sigmoid
    - multiclass: (n,C) -> softmax
    """
    if logits.ndim == 1:
        # binary logits
        return 1.0 / (1.0 + np.exp(-logits))
    if logits.ndim == 2 and logits.shape[1] == 1:
        z = logits[:, 0]
        return 1.0 / (1.0 + np.exp(-z))
    if logits.ndim == 2:
        # softmax
        z = logits - np.max(logits, axis=1, keepdims=True)
        expz = np.exp(z)
        return expz / np.sum(expz, axis=1, keepdims=True)
    # fallback
    return logits


def cross_validate_model(
    model_builder: Callable[[], Any],
    X: Any,
    y: Any,
    *,
    cv: StratifiedKFold,
    fit_params: Optional[MutableMapping[str, Any]] = None,
    preprocessor: Optional[Any] = None,
) -> ClassificationMetrics:
    """
    Classification cross-validation for PyTorch models reusing helper evaluation.

    Возвращает ClassificationMetrics со средними метриками по фолдам и estimators.
    Поддерживает EarlyStopping через fit_params["early_stopping"] (monitor='val_loss').
    """

    fit_params = dict(fit_params) if fit_params else {}

    # --- Hyperparams / hooks ---
    epochs: int = int(fit_params.get("epochs", 3))
    batch_size: int = int(fit_params.get("batch_size", 64))
    lr: float = float(fit_params.get("lr", 1e-3))
    weight_decay: float = float(fit_params.get("weight_decay", 0.0))
    device: torch.device = _infer_device(fit_params.get("device"))
    num_workers: int = int(fit_params.get("num_workers", 0))
    pin_memory: bool = bool(fit_params.get("pin_memory", device.type == "cuda"))
    grad_clip: Optional[float] = fit_params.get("grad_clip", None)
    verbose: int = int(fit_params.get("verbose", 1))

    # Dataset/Collate (для текста часто важно)
    dataset_cls = fit_params.get("dataset_cls", _NumpyLikeDataset)
    collate_fn = fit_params.get("collate_fn", _default_collate)

    # Optimizer/Loss можно переопределить
    optimizer_builder = fit_params.get("optimizer_builder", None)  # callable(model_params)->optim
    criterion = fit_params.get("criterion", None)  # nn.Module

    # --- Containers ---
    fit_times: List[float] = []
    accs: List[float] = []
    f1s: List[float] = []
    precs: List[float] = []
    recalls: List[float] = []
    aucs: List[float] = []
    estimators: List[Any] = []

    X_array = X
    y_array = y

    n_samples = len(y_array)
    oof_pred: np.ndarray = np.empty(n_samples, dtype=int)
    oof_probs: Optional[np.ndarray] = None
    has_probs: bool = True

    # ---------------------------
    # helper: loss per batch
    # ---------------------------
    def _batch_loss(logits, yb, crit: nn.Module) -> torch.Tensor:
        if isinstance(crit, nn.CrossEntropyLoss):
            # y: Long [B]
            if torch.is_tensor(yb):
                yb_ce = yb.long().view(-1)
            else:
                yb_ce = torch.as_tensor(yb, device=device).long().view(-1)
            return crit(logits, yb_ce)
        else:
            # BCEWithLogits: y float [B], logits [B] (или [B,1])
            if torch.is_tensor(yb):
                yb_bce = yb.float().view(-1)
            else:
                yb_bce = torch.as_tensor(yb, device=device).float().view(-1)
            logits_bce = logits.view(-1)
            return crit(logits_bce, yb_bce)

    # Iterate folds
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_array, y_array)):
        print(f"Fold {fold_idx + 1}")

        # Split
        X_train, y_train = _safe_index(X_array, train_idx), _safe_index(y_array, train_idx)
        X_test, y_test = _safe_index(X_array, test_idx), _safe_index(y_array, test_idx)

        # Preprocess per fold
        fitted_pre = clone(preprocessor) if preprocessor is not None else None
        if fitted_pre is not None:
            X_train_proc = fitted_pre.fit_transform(X_train, y_train)
            X_test_proc = fitted_pre.transform(X_test)
        else:
            X_train_proc, X_test_proc = X_train, X_test

        # If sklearn transformer returns sparse, optionally densify
        if fit_params.get("sparse_to_dense", False):
            if hasattr(X_train_proc, "toarray"):
                X_train_proc = X_train_proc.toarray()
            if hasattr(X_test_proc, "toarray"):
                X_test_proc = X_test_proc.toarray()

        # Build model per fold
        model = model_builder()
        if not isinstance(model, nn.Module):
            raise TypeError("model_builder() must return a torch.nn.Module for PyTorch helper.")
        model.to(device)

        # Infer criterion if not provided
        if criterion is None:
            y_train_np = _to_numpy(y_train)
            n_unique = len(np.unique(y_train_np))
            if n_unique > 2:
                criterion_fold: nn.Module = nn.CrossEntropyLoss()
            else:
                criterion_fold = nn.BCEWithLogitsLoss()
        else:
            criterion_fold = criterion

        # Optimizer
        if optimizer_builder is not None:
            optimizer = optimizer_builder(model.parameters())
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # ---------------------------
        # EarlyStopping: внутренний val-split в рамках train фолда
        # ---------------------------
        early_stopping_cfg = fit_params.get("early_stopping", None)

        if early_stopping_cfg is not None:
            import copy
            from sklearn.model_selection import StratifiedShuffleSplit

            patience = int(early_stopping_cfg.get("patience", 2))
            min_delta = float(early_stopping_cfg.get("min_delta", 0.0))
            monitor = str(early_stopping_cfg.get("monitor", "val_loss"))
            mode = str(early_stopping_cfg.get("mode", "min"))
            val_fraction = float(early_stopping_cfg.get("val_fraction", 0.1))
            es_random_state = int(early_stopping_cfg.get("random_state", 42))

            if monitor != "val_loss":
                raise ValueError("early_stopping сейчас поддерживает только monitor='val_loss'.")
            if mode not in ("min", "max"):
                raise ValueError("early_stopping.mode должен быть 'min' или 'max'.")
            if not (0.0 < val_fraction < 1.0):
                raise ValueError("early_stopping.val_fraction должен быть в (0, 1).")

            y_train_np = _to_numpy(y_train)
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=val_fraction, random_state=es_random_state
            )
            tr_sub_idx, val_idx = next(splitter.split(np.zeros(len(y_train_np)), y_train_np))

            X_tr_sub = _safe_index(X_train_proc, tr_sub_idx)
            y_tr_sub = _safe_index(y_train, tr_sub_idx)
            X_val_sub = _safe_index(X_train_proc, val_idx)
            y_val_sub = _safe_index(y_train, val_idx)
        else:
            patience = 0
            min_delta = 0.0
            mode = "min"
            X_tr_sub, y_tr_sub = X_train_proc, y_train
            X_val_sub, y_val_sub = None, None

        # ---------------------------
        # DataLoaders (train/val/test)
        # ---------------------------
        train_ds = dataset_cls(X_tr_sub, y_tr_sub)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        if X_val_sub is not None:
            val_ds = dataset_cls(X_val_sub, y_val_sub)
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            )
        else:
            val_loader = None

        test_ds = dataset_cls(X_test_proc, y_test)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        # ---------------------------
        # Fit (+ early stopping)
        # ---------------------------
        def _eval_val_loss() -> float:
            assert val_loader is not None
            model.eval()
            total = 0.0
            n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    if torch.is_tensor(xb):
                        xb = xb.to(device, non_blocking=True)
                    if torch.is_tensor(yb):
                        yb = yb.to(device, non_blocking=True)
                    logits = model(xb)
                    loss = _batch_loss(logits, yb, criterion_fold)
                    bs = int(len(yb)) if hasattr(yb, "__len__") else int(xb.shape[0])
                    total += float(loss.item()) * bs
                    n += bs
            model.train()
            return total / max(n, 1)

        start_fit = float(perf_counter())
        model.train()

        best_val = float("inf") if mode == "min" else -float("inf")
        best_state = None
        bad_epochs = 0

        for ep in range(epochs):
            for xb, yb in train_loader:
                optimizer.zero_grad(set_to_none=True)

                if torch.is_tensor(xb):
                    xb = xb.to(device, non_blocking=True)
                if torch.is_tensor(yb):
                    yb = yb.to(device, non_blocking=True)

                logits = model(xb)
                loss = _batch_loss(logits, yb, criterion_fold)

                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()

            # Early stopping check
            if val_loader is not None:
                val_loss = _eval_val_loss()
                improved = (val_loss < best_val - min_delta) if mode == "min" else (val_loss > best_val + min_delta)

                if improved:
                    best_val = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if verbose:
                    print(
                        f"  epoch={ep+1}/{epochs}  val_loss={val_loss:.6f}  "
                        f"best={best_val:.6f}  bad={bad_epochs}/{patience}"
                    )

                if bad_epochs >= patience:
                    if verbose:
                        print(f"  EarlyStopping: stop on epoch {ep+1}, best_val={best_val:.6f}")
                    break

        # restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        fit_times.append(float(perf_counter() - start_fit))

        # ---------------------------
        # Score (test): logits -> probs -> preds
        # ---------------------------
        model.eval()
        all_logits: List[np.ndarray] = []
        with torch.no_grad():
            for xb, _yb in test_loader:
                if torch.is_tensor(xb):
                    xb = xb.to(device, non_blocking=True)
                out = model(xb)
                all_logits.append(_to_numpy(out))

        logits_test = np.concatenate(all_logits, axis=0) if len(all_logits) else np.empty((0,))
        probs_test = _probabilities_from_logits(logits_test)

        y_pred = _labels_from_score(probs_test)

        # Для multiclass: y_probs (N,C); для binary: (N,) prob positive
        if probs_test is None:
            y_probs_for_eval = None
        elif hasattr(probs_test, "ndim") and probs_test.ndim == 2 and probs_test.shape[1] > 2:
            y_probs_for_eval = probs_test
        else:
            y_probs_for_eval = _positive_class_probabilities(probs_test)

        fold_metrics: ClassificationMetrics = evaluate_classification(
            y_test=y_test,
            y_pred=y_pred,
            y_probs=y_probs_for_eval,
            model_name=f"Fold {fold_idx + 1}",
            enable_plot=False,
        )

        accs.append(float(fold_metrics.accuracy))
        f1s.append(float(fold_metrics.f1_macro))
        precs.append(float(fold_metrics.precision_macro))
        recalls.append(float(fold_metrics.recall_macro))

        aucs.append(
            float(fold_metrics.roc_auc_ovr_macro)
            if fold_metrics.roc_auc_ovr_macro is not None
            else float("nan")
        )

        # Save estimator (переносим на CPU)
        model_cpu = model.to("cpu")
        estimators.append(
            {
                "preprocessor": fitted_pre,
                "model": model_cpu,
                "optimizer": optimizer.__class__.__name__,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "early_stopping": early_stopping_cfg,
            }
        )

        # OOF
        oof_pred[test_idx] = y_pred

        if y_probs_for_eval is None:
            has_probs = False
        else:
            y_probs_np = _to_numpy(y_probs_for_eval)
            if oof_probs is None:
                if y_probs_np.ndim == 1:
                    oof_probs = np.empty(n_samples, dtype=float)
                else:
                    oof_probs = np.empty((n_samples, y_probs_np.shape[1]), dtype=float)
            oof_probs[test_idx] = y_probs_np

    # ---------------------------
    # Final aggregation
    # ---------------------------
    mean_auc = float(np.nanmean(aucs)) if len(aucs) else float("nan")
    final_roc = None
    if np.isfinite(mean_auc):
        roc_auc_value = mean_auc
    else:
        roc_auc_value = None

    final_metrics = ClassificationMetrics(
        name="PyTorch CV",
        training_time=float(np.nansum(fit_times)),
        estimators=None,

        accuracy=float(np.nanmean(accs)) if len(accs) else float("nan"),

        f1_macro=float(np.nanmean(f1s)) if len(f1s) else float("nan"),
        precision_macro=float(np.nanmean(precs)) if len(precs) else float("nan"),
        recall_macro=float(np.nanmean(recalls)) if len(recalls) else float("nan"),

        # weighted (если не считаешь отдельно — кладем macro, чтобы модель не падала)
        f1_weighted=float(np.nanmean(f1s)) if len(f1s) else float("nan"),
        precision_weighted=float(np.nanmean(precs)) if len(precs) else float("nan"),
        recall_weighted=float(np.nanmean(recalls)) if len(recalls) else float("nan"),

        roc_auc_ovr_macro=roc_auc_value,

        confusion_matrix=None,
        classification_report=None,
        roc_curve_micro=None,
        roc_curves_by_class=None,

        # backward-compat
        roc_auc=roc_auc_value,
        f1_score=float(np.nanmean(f1s)) if len(f1s) else float("nan"),
        precision=float(np.nanmean(precs)) if len(precs) else float("nan"),
        recall=float(np.nanmean(recalls)) if len(recalls) else float("nan"),
        roc_curve=final_roc,
    )

    final_metrics.estimators = estimators

    p.plot_classification_results(final_metrics, model_name="PyTorch CV")

    return final_metrics



def _labels_from_score(y_score: np.ndarray) -> np.ndarray:
    """
    Тот же контракт, что в keras_helper.py:
    - binary: порог 0.5
    - multiclass: argmax
    """
    if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
        return (y_score.ravel() >= 0.5).astype(int)
    return np.argmax(y_score, axis=1)


def _positive_class_probabilities(y_score: np.ndarray) -> Optional[np.ndarray]:
    """
    Как в keras_helper.py:
    - (n,), (n,1) -> 1D
    - (n,2) -> proba класса 1
    - multiclass (>2) -> None
    """
    if y_score.ndim == 1:
        return y_score.ravel()
    if y_score.ndim == 2 and y_score.shape[1] == 1:
        return y_score.ravel()
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        return y_score[:, 1]
    return None

def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # строгая воспроизводимость (может замедлить)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_categorical(y, num_classes: int, device: torch.device | None = None):
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    return F.one_hot(y, num_classes=num_classes).to(torch.float32)

def l2_reg_value_to_weight_decay(l2_reg_value: float) -> float:
    return float(l2_reg_value)

# лоссы
BinaryCrossentropy = nn.BCEWithLogitsLoss
CategoricalCrossentropy = nn.CrossEntropyLoss  

def init_constant_(tensor, value: float = 0.0):
    nn.init.constant_(tensor, value)
    return tensor

def list_physical_devices():
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = float(min_delta)
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.best: float | None = None
        self.bad_epochs = 0
        self.best_state_dict: dict[str, torch.Tensor] | None = None

    def step(self, value: float, model: nn.Module) -> bool:
        value = float(value)

        if self.best is None:
            self.best = value
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False

        if self.mode == "min":
            improved = value < (self.best - self.min_delta)
        else:
            improved = value > (self.best + self.min_delta)

        if improved:
            self.best = value
            self.bad_epochs = 0
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, rnn_units: int, num_classes: int = 4,
                 dropout: float = 0.3, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=rnn_units,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * rnn_units, num_classes)

    def forward(self, x):
        x = x.long()                               # [B, L]
        emb = self.embedding(x)                    # [B, L, D]
        out, _ = self.rnn(emb)                     # [B, L, 2H]

        # lengths = число НЕ-pad токенов
        lengths = (x != self.pad_id).sum(dim=1).clamp(min=1)          # [B]
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2)) # [B,1,2H]

        last = out.gather(1, idx).squeeze(1)        # [B, 2H] — last real token
        last = self.dropout(last)
        logits = self.fc(last)                     # [B, C]
        return logits


# ===== PyTorch TextCNN =====
class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_filters: int, kernel_size: int,
                 num_classes: int, dropout: float = 0.0, pad_id: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        # x: [B, L] token ids
        x = x.long()
        e = self.embed(x)              # [B, L, E]
        e = e.transpose(1, 2)          # [B, E, L]
        c = F.relu(self.conv(e))       # [B, F, L-K+1]
        p = F.max_pool1d(c, c.size(2)).squeeze(2)  # [B, F]
        p = self.dropout(p)
        logits = self.fc(p)            # [B, C]
        return logits
    

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_classes: int, dropout: float = 0.0, pad_id: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        x = x.long()              # [B, L]
        e = self.embed(x)         # [B, L, E]
        out, (h_n, c_n) = self.lstm(e)
        h_fwd = h_n[-2]           # [B, H]
        h_bwd = h_n[-1]           # [B, H]
        h = torch.cat([h_fwd, h_bwd], dim=1)   # [B, 2H]
        h = self.dropout(h)
        logits = self.fc(h)       # [B, C]
        return logits
    
class TextCNNGloveFrozen(nn.Module):
    def __init__(self, glove_weights: np.ndarray, num_filters: int, kernel_size: int,
                 num_classes: int, dropout: float = 0.0, pad_id: int = 0):
        super().__init__()
        V, D = glove_weights.shape
        self.embed = nn.Embedding(V, D, padding_idx=pad_id)
        self.embed.weight.data.copy_(torch.from_numpy(glove_weights))
        self.embed.weight.requires_grad = False  # замораживаем

        self.conv = nn.Conv1d(D, num_filters, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = x.long()                 # [B, L]
        e = self.embed(x)            # [B, L, D]
        e = e.transpose(1, 2)        # [B, D, L]
        c = F.relu(self.conv(e))     # [B, F, L-K+1]
        p = F.max_pool1d(c, c.size(2)).squeeze(2)  # [B, F]
        p = self.dropout(p)
        logits = self.fc(p)          # [B, C]
        return logits

__all__ = ["cross_validate_model"]