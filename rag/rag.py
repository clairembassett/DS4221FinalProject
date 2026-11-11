"""College RAG — the Lab 10 micro-rag pipeline pointed at the college corpus.

Same stages as lab10/micro_rag.py, with the course corpus swapped out:

    chunk_corpus()   college JSON files -> section chunks (chunker.py, <= 600 chars)
    embedder         text -> unit vector. Three choices (see make_embedder):
                       "tfidf"   hashed bag-of-words + IDF weighting  (default)
                       "hashed"  the lab's original embed(), unchanged
                       "minilm"  all-MiniLM-L6-v2 sentence embeddings
                                 (needs: pip install sentence-transformers)
    Retriever        chunks -> BruteForceIndex -> top-k (same as your lab)
    build_prompt()   question + chunks -> prompt (cites chunk ids)
    answer()         prompt -> text (echo mode, or a real LLM via LLM_* env vars)

Why "tfidf" is the default: the lab's embed() weights every word equally, so on
108 college articles words like "university", "students" and "campus" (in
nearly every chunk) swamp the words that matter, and years are dropped
entirely. IDF down-weights words that appear everywhere, and keeping digits
lets "1890" match "1890". On evaluate.py's questions that took hit@3 from
0.00 to 0.40 with no model and no new dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re

from chunker import DEFAULT_DATA_DIR, MAX_CHARS, chunk_corpus  # noqa: F401 (re-exported)
from microvector import BruteForceIndex

DIM = 2048

STOPWORDS = set("""a an and are as at be because by can each for from has have
how in is it its like more most no not of on one only or so than that the
their them then there these they this to was what when where which while who
why will with you your""".split())


def _bucket(word: str) -> int:
    return int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM


def _unit(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


# ------------------------------------------------- embedders

def embed(text: str) -> list[float]:
    """The lab's original hashed bag-of-words embedder, unchanged."""
    counts: dict[str, int] = {}
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    v = [0.0] * DIM
    for word, count in counts.items():
        v[_bucket(word)] += 1.0 / (1.0 + math.log(count))
    return _unit(v)


class HashedEmbedder:
    """Lab 10's embed(): no corpus statistics, letters only."""

    def fit(self, texts: list[str]) -> None:
        pass

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [embed(t) for t in texts]


class TfidfEmbedder:
    """Hashed bag-of-words with IDF weights learned from the chunks.

    weight(word) = (1 + log count_in_text) * idf(word)
    idf(word)    = log((N + 1) / (chunks_containing_word + 1)) + 1
    Words never seen in the corpus get the highest idf. Digits count as
    words, so years and numbers can match.
    """

    TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self):
        self.idf: dict[str, float] = {}
        self.unseen_idf = 1.0

    def _tokens(self, text: str) -> list[str]:
        return [w for w in self.TOKEN_RE.findall(text.lower()) if w not in STOPWORDS]

    def fit(self, texts: list[str]) -> None:
        df: dict[str, int] = {}
        for t in texts:
            for w in set(self._tokens(t)):
                df[w] = df.get(w, 0) + 1
        n = len(texts)
        self.idf = {w: math.log((n + 1) / (d + 1)) + 1 for w, d in df.items()}
        self.unseen_idf = math.log(n + 1) + 1

    def _one(self, text: str) -> list[float]:
        counts: dict[str, int] = {}
        for w in self._tokens(text):
            counts[w] = counts.get(w, 0) + 1
        v = [0.0] * DIM
        for w, c in counts.items():
            v[_bucket(w)] += (1.0 + math.log(c)) * self.idf.get(w, self.unseen_idf)
        return _unit(v)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class MiniLMEmbedder:
    """A learned sentence-embedding model (the one lab10/measure_real.py
    compares against). Downloads ~90 MB on first use; encoding all chunks
    takes a few minutes on a laptop CPU, much less on a GPU."""

    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise SystemExit("The minilm embedder needs:  pip install sentence-transformers") from e
        self.model = SentenceTransformer(self.MODEL)

    def fit(self, texts: list[str]) -> None:
        pass

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=64,
                                 show_progress_bar=len(texts) > 100)
        return vecs.tolist()


EMBEDDERS = {"tfidf": TfidfEmbedder, "hashed": HashedEmbedder, "minilm": MiniLMEmbedder}


def make_embedder(name: str = "tfidf"):
    return EMBEDDERS[name]()


# ------------------------------------------------- retrieval

class Retriever:
    """Embeds every chunk at build time; answers top-k at query time."""

    def __init__(self, chunks: list[dict], embedder: str = "tfidf"):
        self.chunks = chunks
        self.embedder = make_embedder(embedder)
        texts = [c["text"] for c in self.chunks]
        self.embedder.fit(texts)
        # index id i <-> self.chunks[i]
        self.index = BruteForceIndex(self.embedder.encode(texts))

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        """The k best chunks, best first, each a copy with a "score" added."""
        qv = self.embedder.encode([question])[0]
        results = []
        for score, i in self.index.search(qv, k):
            chunk = dict(self.chunks[i])
            chunk["score"] = score
            results.append(chunk)
        return results


# ------------------------------------------------- prompt

HEADER = "Answer using ONLY the sources below. Cite as [chunk_id]."


def build_prompt(question: str, retrieved: list[dict]) -> str:
    """Same shape as the lab:

        Answer using ONLY the sources below. Cite as [chunk_id].

        [wp:32127#0007] University of Chicago — History > Early years: In 1890, ...
        [wp:18879#0001] Massachusetts Institute of Technology — Introduction: ...

        Question: <the question>

    One line per chunk: "[chunk_id] heading: body". The heading is the
    college name and section path, so the model knows where each fact is from.
    """
    lines = []
    for c in retrieved:
        body = c["text"].split("\n", 1)[-1].replace("\n", " ")
        lines.append(f"[{c['chunk_id']}] {c['heading']}: {body}")
    return f"{HEADER}\n\n" + "\n".join(lines) + f"\n\nQuestion: {question}"


# ------------------------------------------------- generation (from lab 10)

def answer(question: str, retriever: Retriever, k: int = 3) -> str:
    """Retrieve, build the prompt, generate. Without LLM_BASE_URL set this
    runs in ECHO mode and returns the prompt, so you can see what retrieval
    chose. With LLM_BASE_URL / LLM_API_KEY / LLM_MODEL set, it calls any
    OpenAI-compatible /chat/completions endpoint."""
    retrieved = retriever.retrieve(question, k)
    prompt = build_prompt(question, retrieved)
    base = os.environ.get("LLM_BASE_URL")
    if not base:
        cited = ", ".join(c["chunk_id"] for c in retrieved)
        return (prompt + "\n\n[echo mode — no LLM configured. "
                f"A real model would answer from: {cited}]")
    import urllib.request
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}",
                 "Content-Type": "application/json"},
        data=json.dumps({
            "model": os.environ.get("LLM_MODEL", ""),
            "messages": [{"role": "user", "content": prompt}],
        }).encode())
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]
