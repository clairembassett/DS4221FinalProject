"""BM25 keyword retrieval: the non-vector baseline.

    python3 ask.py --retriever bm25 "When was Harvard founded?"
    python3 evaluate.py --retriever bm25

BM25 is the standard search-engine ranking function. For each query word w
found in a chunk:

    score += idf(w) * tf * (K1 + 1) / (tf + K1 * (1 - B + B * len / avg_len))

    tf       how often w occurs in the chunk
    idf(w)   log(1 + (N - df + 0.5) / (df + 0.5)): rare words count more
    K1, B    repeats saturate (K1) and long chunks are gently penalized (B)

Chunks are stored as sparse vectors {word_id: weight} in an inverted index
(word -> chunks containing it), so a query only touches chunks that share a
word with it. Scores are exact, the same a brute-force pass would give.

History, for the report: earlier versions used TF-IDF cosine. With every word
hashed into 2048 buckets, unrelated words collided ("virginia" and "johns"
shared a bucket, so Johns Hopkins chunks matched Virginia questions).
Exact TF-IDF fixed that but still let short chunks that repeat a query word
win. BM25 fixed both. On the first 15 questions (passage hit@3 / MRR@10):
TF-IDF hashed 0.40 / 0.33, TF-IDF exact 0.60 / 0.40, BM25 0.80 / 0.58.
"""

from __future__ import annotations

import heapq
import math
import re

from rag import STOPWORDS, fetch_chunks

TOKEN_RE = re.compile(r"[a-z0-9]+")
K1 = 1.2
B = 0.75


def tokens(text: str) -> list[str]:
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in STOPWORDS]


class BM25Retriever:
    def __init__(self, con, config: str):
        self.con, self.kind, self.config, self.model = con, "bm25", config, None
        rows = con.execute("SELECT chunk_id, text FROM chunks WHERE config = ? ORDER BY chunk_id",
                           [config]).fetchall()
        if not rows:
            raise SystemExit(f"No chunks with config {config!r}. Run ingest.py first.")
        self.ids = [r[0] for r in rows]
        docs = [tokens(r[1]) for r in rows]

        df: dict[str, int] = {}
        for toks in docs:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        avg_len = sum(len(t) for t in docs) / n
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

        # Inverted index: word -> [(chunk position, BM25 weight of the word in that chunk)]
        self.postings: dict[str, list[tuple[int, float]]] = {}
        for i, toks in enumerate(docs):
            counts: dict[str, int] = {}
            for w in toks:
                counts[w] = counts.get(w, 0) + 1
            length_norm = K1 * (1 - B + B * len(toks) / avg_len)
            for w, tf in counts.items():
                weight = self.idf[w] * tf * (K1 + 1) / (tf + length_norm)
                self.postings.setdefault(w, []).append((i, weight))

    def search(self, question: str, k: int) -> list[tuple[float, str]]:
        """Top k as (score, chunk_id), best first."""
        scores: dict[int, float] = {}
        for w in tokens(question):
            for i, weight in self.postings.get(w, ()):
                scores[i] = scores.get(i, 0.0) + weight
        best = heapq.nlargest(k, scores.items(), key=lambda item: item[1])
        return [(score, self.ids[i]) for i, score in best]

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        hits = self.search(question, k)
        details = fetch_chunks(self.con, [cid for _, cid in hits])
        return [{**details[cid], "score": score} for score, cid in hits]
