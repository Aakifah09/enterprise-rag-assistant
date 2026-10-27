"""A small, transparent Okapi BM25 implementation for keyword ranking.

score(q, d) = sum over query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
with idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5)) (always positive).
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")
STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it
    its me my no not of on or our should so than that the their them then there these they this to was
    we were what when where which who whom why will with would you your about after before any all also
    must may per via""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (keeping things like '4.2' or 'it-ops'), minus stopwords."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_freqs = [Counter(toks) for toks in self.doc_tokens]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.n_docs = len(documents)
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0
        df: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        self.idf = {t: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = [0.0] * self.n_docs
        if not terms or not self.n_docs:
            return out
        for i, freqs in enumerate(self.doc_freqs):
            norm = self.k1 * (1 - self.b + self.b * self.doc_lens[i] / (self.avgdl or 1.0))
            s = 0.0
            for t in terms:
                tf = freqs.get(t)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            out[i] = s
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        """(row, score) for the best ``k`` rows with a positive score."""
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
