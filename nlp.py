# -*- coding: utf-8 -*-
# nlp.py
# Utilities for tokenization and multi-class token/ngram analysis (sparse, scalable).

from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Any, Set
from functools import lru_cache
import html

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif


# ============================================================
# Tokenization
# ============================================================

_WORD_RE = re.compile(r"[^a-z0-9\s]+")


def tokenize_words(text: str) -> List[str]:
    """
    Простая word-level токенизация под RNN/CNN/LSTM/GloVe.
    - lower
    - схлопывает пробелы
    """
    if not isinstance(text, str):
        return []
    t = text.lower()
    t = _WORD_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.split() if t else []


def detokenize(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


# ============================================================
# Stopwords / punctuation
# ============================================================

def get_stopwords_set(extra: Optional[Iterable[str]] = None) -> set:
    s = set(ENGLISH_STOP_WORDS)
    if extra is not None:
        s |= {str(x).lower() for x in extra}
    return s


def get_punctuation_set(extra: Optional[Iterable[str]] = None) -> set:
    s = set(string.punctuation)
    if extra is not None:
        s |= set(extra)
    return s


def filter_tokens(
    tokens: Sequence[str],
    *,
    to_lower: bool = True,
    remove_stopwords: bool = True,
    remove_punctuation: bool = True,
    stopwords: Optional[set] = None,
    punctuation: Optional[set] = None,
) -> List[str]:
    if stopwords is None:
        stopwords = get_stopwords_set()
    if punctuation is None:
        punctuation = get_punctuation_set()

    out: List[str] = []
    for tok in tokens:
        t = str(tok)
        if to_lower:
            t = t.lower()

        if remove_punctuation and (t in punctuation):
            continue
        if remove_stopwords and (t in stopwords):
            continue
        if t.strip() == "":
            continue

        out.append(t)
    return out


# ============================================================
# Counts
# ============================================================

def token_counts(
    tokenized_texts: Sequence[Sequence[str]],
    *,
    min_len: int = 1,
) -> Counter:
    c = Counter()
    for toks in tokenized_texts:
        for t in toks:
            if len(t) >= min_len:
                c[t] += 1
    return c


def filter_tokens_by_frequency(
    tokenized_texts: Sequence[Sequence[str]],
    *,
    min_freq: int = 2,
    max_freq: Optional[int] = None,
) -> List[List[str]]:
    """
    Удаляет токены, встречающиеся слишком редко/слишком часто (если max_freq задан).
    """
    c = token_counts(tokenized_texts)
    out: List[List[str]] = []
    for toks in tokenized_texts:
        nt = []
        for t in toks:
            f = c.get(t, 0)
            if f < min_freq:
                continue
            if max_freq is not None and f > max_freq:
                continue
            nt.append(t)
        out.append(nt)
    return out


# ============================================================
# N-grams
# ============================================================

def generate_ngrams(tokens: Sequence[str], n: int) -> List[str]:
    if n <= 0:
        return []
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def texts_to_ngrams(
    tokenized_texts: Sequence[Sequence[str]],
    n: int,
) -> List[List[str]]:
    return [generate_ngrams(toks, n) for toks in tokenized_texts]


# ============================================================
# Sparse multi-class n-gram scoring
# ============================================================

def compute_ngram_scores(
    texts: Sequence[Union[str, Sequence[str]]],
    y: Sequence[int],
    *,
    n: int = 1,
    metric: str = "chi2",  # 'chi2' | 'anova_f' | 'mutual_info'
    min_df: Union[int, float] = 5,
    max_df: Union[int, float] = 1.0,
    max_features: Optional[int] = 200_000,
    tokenizer: Optional[callable] = None,
    lowercase: bool = True,
) -> pd.DataFrame:
    """
    Возвращает DataFrame: token, score
    Счёт делается на sparse матрице (OK для 120k строк).

    texts:
      - list[str] (сырые тексты)  ИЛИ
      - list[list[str]] (уже токенизировано)

    Если list[list[str]], то мы объединяем токены пробелом и применяем CountVectorizer
    с analyzer='word'.
    """
    if len(texts) != len(y):
        raise ValueError("texts and y must have same length")

    # привести к list[str]
    if len(texts) == 0:
        return pd.DataFrame({"token": [], "score": []})

    if isinstance(texts[0], (list, tuple)):
        docs = [" ".join(map(str, t)) for t in texts]  # type: ignore[arg-type]
        tok = None
    else:
        docs = [str(t) for t in texts]  # type: ignore[assignment]
        tok = tokenizer

    vec = CountVectorizer(
        lowercase=lowercase,
        tokenizer=tok,
        ngram_range=(n, n),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )

    X = vec.fit_transform(docs)  # sparse
    y_arr = np.asarray(y, dtype=int)

    metric = metric.lower().strip()
    if metric == "chi2":
        scores, _ = chi2(X, y_arr)
    elif metric in ("anova_f", "f", "f_classif"):
        scores, _ = f_classif(X, y_arr)
    elif metric in ("mutual_info", "mi"):
        # MI в sklearn работает с dense/sparse, но может быть медленнее; оставляем как опцию
        scores = mutual_info_classif(X, y_arr, discrete_features=True, random_state=42)
    else:
        raise ValueError("metric must be one of: 'chi2', 'anova_f', 'mutual_info'")

    feats = np.asarray(vec.get_feature_names_out())
    df = pd.DataFrame({"token": feats, "score": scores})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def count_based_analysis(
    tokenized_texts: Sequence[Sequence[str]],
    y: Sequence[int],
    *,
    top_k: int = 300,
    metric: str = "chi2",
    n: int = 1,
    min_df: Union[int, float] = 5,
    max_df: Union[int, float] = 1.0,
    max_features: Optional[int] = 200_000,
) -> pd.DataFrame:
    """
    Возвращает DataFrame формата, который ожидает твой plot_count_based_analysis:
      token, score, total_count,
      count_<class>, freq_<class> для каждого класса.

    Важно: всё строится на sparse CountVectorizer, без dense матриц.
    """
    if len(tokenized_texts) != len(y):
        raise ValueError("tokenized_texts and y must have same length")

    if len(tokenized_texts) == 0:
        return pd.DataFrame()

    # превращаем токены обратно в документы
    docs = [" ".join(map(str, toks)) for toks in tokenized_texts]
    y_arr = np.asarray(y, dtype=int)
    classes = sorted(np.unique(y_arr).tolist())

    # sparse ngram matrix
    vec = CountVectorizer(
        lowercase=True,
        ngram_range=(n, n),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    X = vec.fit_transform(docs)  # (N, V), sparse
    feats = np.asarray(vec.get_feature_names_out())

    # score
    metric_l = metric.lower().strip()
    if metric_l == "chi2":
        scores, _ = chi2(X, y_arr)
    elif metric_l in ("anova_f", "f", "f_classif"):
        scores, _ = f_classif(X, y_arr)
    elif metric_l in ("mutual_info", "mi"):
        scores = mutual_info_classif(X, y_arr, discrete_features=True, random_state=42)
    else:
        raise ValueError("metric must be one of: 'chi2', 'anova_f', 'mutual_info'")

    # total counts per feature
    total_count = np.asarray(X.sum(axis=0)).ravel().astype(int)

    # per-class counts
    class_counts: Dict[int, np.ndarray] = {}
    for c in classes:
        Xc = X[y_arr == c]
        class_counts[c] = np.asarray(Xc.sum(axis=0)).ravel().astype(int)

    # build df
    df = pd.DataFrame({"token": feats, "score": scores, "total_count": total_count})
    for c in classes:
        df[f"count_{c}"] = class_counts[c]
        denom = max(1, int((y_arr == c).sum()))
        # "freq" как среднее количество n-грамм на документ класса
        df[f"freq_{c}"] = df[f"count_{c}"] / denom

    df = df.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    return df

@lru_cache(maxsize=1)
def get_default_resources():
    """
    Кэшируем, чтобы стоп-слова/пунктуация создавались один раз за процесс.
    """
    stopwords = get_stopwords_set()
    punctuation = get_punctuation_set()
    return stopwords, punctuation

def ensure_tokens(x):
    """
    Приводит значение к list[str] токенов.
    - list -> как есть
    - str  -> tokenize_words(str)
    - другое/NaN -> []
    """
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return tokenize_words(x)
    return []


# --------------- настройки мусора ---------------
BAD_EXACT = {
    "#name?", "#name", "nan", "none", "null", "n/a", "na", "undefined", "error"
}
BAD_CONTAINS = {
    "#name?",  # иногда встречается в составе
}

ARTIFACT_TOKENS = {"lt", "gt", "amp", "quot", "nbsp", "apos"}  # мусор после html

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+")

# "битые" html сущности: "#39;" вместо "&#39;"
BROKEN_ENTITY_RE = re.compile(r"#(\d{1,4});")

ARTIFACT_TOKENS_RE = re.compile(r"\b(?:lt|gt|amp|quot|nbsp|apos)\b", flags=re.IGNORECASE)


def is_garbage_text(x: object) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    if not s:
        return True

    low = s.lower()
    if low in BAD_EXACT:
        return True

    if any(b in low for b in BAD_CONTAINS):
        return True

    # частый маркер мусорной категории
    if low == "unknown":
        return True

    return False


def clean_text_html_artifacts(x: object) -> str:
    """Очистка HTML-артефактов + нормализация пробелов."""
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""

    # лечим "битые" сущности вида "#39;" -> "'"
    def _fix_broken_entity(m):
        code = int(m.group(1))
        if 0 <= code <= 0x10FFFF:
            try:
                return chr(code)
            except Exception:
                return " "
        return " "

    s = BROKEN_ENTITY_RE.sub(_fix_broken_entity, s)

    # стандартное декодирование html сущностей: &amp; &#39; &lt; ...
    s = html.unescape(s)

    # убрать html-теги
    s = TAG_RE.sub(" ", s)

    # убрать control chars
    s = CTRL_RE.sub(" ", s)

    # убрать отдельные мусорные токены lt/gt/amp/...
    s = ARTIFACT_TOKENS_RE.sub(" ", s)

    # нормализовать пробелы
    s = WS_RE.sub(" ", s).strip()
    return s


def clean_and_dedup_df(
    df: pd.DataFrame,
    text_col: str = "Description",
    label_col: str = "Class Index",
    also_clean_cols: tuple = ("Title",),
    drop_unknown_label: bool = True,
) -> pd.DataFrame:
    df2 = df.copy()

    # 1) убрать явный мусор по тексту
    df2[text_col] = df2[text_col].astype("string")
    df2 = df2.loc[~df2[text_col].apply(is_garbage_text)].copy()

    # 2) очистить HTML-артефакты
    df2[text_col] = df2[text_col].apply(clean_text_html_artifacts)

    for c in also_clean_cols:
        if c in df2.columns:
            df2[c] = df2[c].astype("string").apply(clean_text_html_artifacts)

    # 3) после очистки снова убрать пустые
    df2 = df2.loc[df2[text_col].str.len() > 0].copy()

    # 4) (опционально) убрать мусорный класс Unknown, если он у тебя есть в label_col
    if drop_unknown_label and label_col in df2.columns:
        df2 = df2.loc[df2[label_col].astype(str).str.lower() != "unknown"].copy()

    # 5) дедуп по Description с обработкой конфликтов классов
    if label_col in df2.columns:
        nunique_per_text = df2.groupby(text_col)[label_col].nunique(dropna=False)
        conflict_texts = nunique_per_text[nunique_per_text > 1].index

        n_conflicts = len(conflict_texts)
        if n_conflicts > 0:
            # удаляем все конфликтные описания
            df2 = df2.loc[~df2[text_col].isin(conflict_texts)].copy()

        # теперь оставляем по одному экземпляру каждого Description
        before = len(df2)
        df2 = df2.drop_duplicates(subset=[text_col], keep="first").copy()
        after = len(df2)

        print(f"[DEDUP] conflicts removed: {n_conflicts}")
        print(f"[DEDUP] duplicates removed: {before - after}")

    else:
        # если лейбла нет, просто drop_duplicates
        before = len(df2)
        df2 = df2.drop_duplicates(subset=[text_col], keep="first").copy()
        print(f"[DEDUP] duplicates removed: {before - len(df2)}")

    df2 = df2.reset_index(drop=True)
    return df2