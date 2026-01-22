# transformers.py
# sklearn-style transformers for text pipelines:
# - NgramFeatureSelector: choose top-k ngrams by score (sparse, scalable)
# - SequenceVectorizer: convert token sequences to padded integer arrays (OOV mapped, not dropped)

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from nlp import tokenize_words, compute_ngram_scores


# ============================================================
# NgramFeatureSelector (for baselines / analysis)
# ============================================================

class NgramFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Выбирает top_k n-грамм по score (chi2/anova_f/mi) на train.
    Возвращает тексты в виде списка токенов (ngram strings) или списка строк,
    в зависимости от input.

    Практически: чаще полезно для baseline (CountVectorizer/TFIDF),
    а не для RNN/LSTM/CNN+GloVe.
    """

    def __init__(
        self,
        top_k: int = 50_000,
        n: int = 1,
        metric: str = "chi2",
        min_df: Union[int, float] = 5,
        max_df: Union[int, float] = 1.0,
        max_features: Optional[int] = 200_000,
        tokenizer: Optional[Callable[[str], List[str]]] = tokenize_words,
        output: str = "text",  # 'text' or 'tokens'
    ):
        self.top_k = top_k
        self.n = n
        self.metric = metric
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.tokenizer = tokenizer
        self.output = output

        self.selected_: Optional[set] = None

    def fit(self, X: Any, y: Optional[Sequence[int]] = None):
        texts = _coerce_texts(X)

        if y is None:
            # частотный top_k (fallback)
            c = Counter()
            for t in texts:
                toks = self.tokenizer(t) if self.tokenizer is not None else t.split()
                # n-grams как строки
                ng = _to_ngrams(toks, self.n)
                c.update(ng)
            self.selected_ = set([w for w, _ in c.most_common(self.top_k)])
            return self

        scores_df = compute_ngram_scores(
            texts,
            y,
            n=self.n,
            metric=self.metric,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            tokenizer=self.tokenizer,
        )
        top = scores_df.head(self.top_k)["token"].astype(str).tolist()
        self.selected_ = set(top)
        return self

    def transform(self, X: Any):
        if self.selected_ is None:
            raise RuntimeError("NgramFeatureSelector is not fitted")

        texts = _coerce_texts(X)
        out_tokens: List[List[str]] = []

        for t in texts:
            toks = self.tokenizer(t) if self.tokenizer is not None else t.split()
            ng = _to_ngrams(toks, self.n)
            kept = [g for g in ng if g in self.selected_]
            out_tokens.append(kept)

        if self.output == "tokens":
            return out_tokens
        # output == 'text'
        return [" ".join(toks) for toks in out_tokens]


def _to_ngrams(tokens: Sequence[str], n: int) -> List[str]:
    if n <= 0:
        return []
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _coerce_texts(X: Any) -> List[str]:
    """
    Приводим вход к list[str].
    Поддержка:
      - pd.Series
      - pd.DataFrame (берём колонку 'text' если есть, иначе 1-ю)
      - list[str]
      - np.ndarray 1d
    """
    if isinstance(X, pd.DataFrame):
        if "text" in X.columns:
            return X["text"].astype(str).tolist()
        return X.iloc[:, 0].astype(str).tolist()
    if isinstance(X, pd.Series):
        return X.astype(str).tolist()
    if isinstance(X, np.ndarray):
        if X.ndim != 1:
            raise ValueError("Expected 1d array for texts")
        return [str(v) for v in X.tolist()]
    if isinstance(X, list):
        if len(X) == 0:
            return []
        if isinstance(X[0], str):
            return [str(v) for v in X]
        # list of tokens -> join
        if isinstance(X[0], (list, tuple)):
            return [" ".join(map(str, v)) for v in X]
        return [str(v) for v in X]
    return [str(X)]


# ============================================================
# SequenceVectorizer (for NN inputs if you want pre-vectorized arrays)
# ============================================================

class SequenceVectorizer(BaseEstimator, TransformerMixin):
    """
    Преобразует тексты/токены -> padded sequences of ids (numpy int32).
    Важно:
    - OOV НЕ выкидывается, а мапится в unk_id (по умолчанию 1).
    - pad_id=0.
    - max_len фиксируется: либо задаёшь явно, либо выбирается по percentile.

    Это удобно для:
    - Keras (если бы ты делал)
    - быстрых прототипов
    Для PyTorch-пайплайна у тебя уже есть Dataset/Collate в helper.py — он чаще лучше.
    """

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = tokenize_words,
        max_len: Optional[int] = 250,
        infer_max_len_percentile: float = 0.95,
        cap_max_len: int = 512,
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
        pad_left: bool = False,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.infer_max_len_percentile = infer_max_len_percentile
        self.cap_max_len = cap_max_len
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.pad_left = pad_left
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.stoi_: Optional[Dict[str, int]] = None
        self.itos_: Optional[List[str]] = None
        self.pad_id_: int = 0
        self.unk_id_: int = 1
        self.max_len_: Optional[int] = None

    def fit(self, X: Any, y: Any = None):
        texts = _coerce_texts(X)

        tokenized = [self._tokenize(t) for t in texts]
        lengths = np.asarray([len(t) for t in tokenized], dtype=int)

        # determine max_len_
        if self.max_len is not None:
            max_len_ = int(self.max_len)
        else:
            # percentile-based, capped
            q = float(self.infer_max_len_percentile)
            q = min(max(q, 0.5), 1.0)
            max_len_ = int(np.quantile(lengths, q)) if len(lengths) else 0
            max_len_ = min(max_len_, int(self.cap_max_len))
            max_len_ = max(1, max_len_)

        self.max_len_ = max_len_

        # build vocab
        c = Counter()
        for toks in tokenized:
            c.update(toks)

        items = [(w, f) for w, f in c.items() if f >= int(self.min_freq)]
        items.sort(key=lambda x: x[1], reverse=True)

        itos = [self.pad_token, self.unk_token]
        if self.max_vocab_size is not None:
            lim = max(0, int(self.max_vocab_size) - len(itos))
            items = items[:lim]

        for w, _ in items:
            if w in (self.pad_token, self.unk_token):
                continue
            itos.append(w)

        stoi = {w: i for i, w in enumerate(itos)}
        self.itos_ = itos
        self.stoi_ = stoi
        self.pad_id_ = stoi[self.pad_token]
        self.unk_id_ = stoi[self.unk_token]
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.stoi_ is None or self.max_len_ is None:
            raise RuntimeError("SequenceVectorizer is not fitted")

        texts = _coerce_texts(X)
        seqs = []
        for t in texts:
            toks = self._tokenize(t)
            ids = [self.stoi_.get(w, self.unk_id_) for w in toks]
            ids = ids[: self.max_len_]

            pad_len = self.max_len_ - len(ids)
            if pad_len > 0:
                pads = [self.pad_id_] * pad_len
                ids = pads + ids if self.pad_left else ids + pads

            seqs.append(ids)

        return np.asarray(seqs, dtype=np.int32)

    def get_vocab(self) -> Dict[str, Any]:
        if self.stoi_ is None or self.itos_ is None:
            raise RuntimeError("SequenceVectorizer is not fitted")
        return {
            "stoi": self.stoi_,
            "itos": self.itos_,
            "pad_id": self.pad_id_,
            "unk_id": self.unk_id_,
            "max_len": self.max_len_,
        }

    def _tokenize(self, text: str) -> List[str]:
        if self.tokenizer is None:
            return str(text).split()
        return self.tokenizer(str(text))
