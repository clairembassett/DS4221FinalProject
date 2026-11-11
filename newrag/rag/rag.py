"""College RAG: embedders, retrievers, prompt and LLM call.

The pipeline, and where each stage lives:

    1 INGEST    ingest.py    data/*.json -> documents, chunks tables
    2 EMBED     embed.py     chunks -> embeddings table (MiniLM by default)
    3 INDEX     index.py     embeddings -> Lab 9 IVFIndex, saved in DuckDB
    4 RETRIEVE  rag.py       question -> top-k chunk_ids -> chunks JOIN documents
    5 GENERATE  rag.py       question + chunks -> prompt -> LLM -> cited answer
    6 EVALUATE  evaluate.py  eval_questions -> recall@k, MRR, per-question results
                bench.py     build time, queries per second, recall vs. speed

Retrievers (make_retriever):
    "ivf"    Lab 9 IVFIndex over sentence embeddings: the product
    "brute"  Lab 9 BruteForceIndex over the same embeddings: the exact baseline
    "bm25"   keyword search (keyword.py), a non-vector baseline and the
             keyword half of the hybrid-retrieval stretch goal
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re

DEFAULT_CONFIG = "sec600"      # chunking configuration made by: python3 ingest.py
DEFAULT_PROBE = 8              # IVF lists searched per query (see bench.py)

# ------------------------------------------------- embedders

# Short names accepted on the command line -> the name stored in embeddings.model
MODELS = {
    "minilm":   "sentence-transformers/all-MiniLM-L6-v2",       # default; the model lab10 uses
    "multi-qa": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # same size, trained for Q&A search
    "hashed":   "lab10-hashed-2048",                            # the lab's bag-of-words embed()
}


def resolve_model(name: str) -> str:
    return MODELS.get(name, name)


STOPWORDS = set("""a an and are as at be because by can each for from has have
how in is it its like more most no not of on one only or so than that the
their them then there these they this to was what when where which while who
why will with you your""".split())


class HashedEmbedder:
    """Lab 10's embed(): hashed bag of words, no learned model. Kept as a
    baseline; it is not a sentence-embedding model."""

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self.name = f"lab10-hashed-{dim}"

    def _one(self, text: str) -> list[float]:
        counts: dict[str, int] = {}
        for word in re.findall(r"[a-z]+", text.lower()):
            if word not in STOPWORDS:
                counts[word] = counts.get(word, 0) + 1
        v = [0.0] * self.dim
        for word, count in counts.items():
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            v[bucket] += 1.0 / (1.0 + math.log(count))
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class SentenceEmbedder:
    """A sentence-transformers model, e.g. all-MiniLM-L6-v2 (384 dimensions).
    Downloads the model (~90 MB) the first time. Vectors are normalized to
    unit length, so dot product = cosine similarity, as microvector expects."""

    def __init__(self, name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise SystemExit("Sentence embeddings need:  pip install sentence-transformers") from e
        self.name = name
        self.model = SentenceTransformer(name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=64,
                                 show_progress_bar=len(texts) > 500)
        return vecs.tolist()


def make_embedder(name: str = "minilm"):
    name = resolve_model(name)
    m = re.fullmatch(r"lab10-hashed-(\d+)", name)
    if m:
        return HashedEmbedder(int(m.group(1)))
    return SentenceEmbedder(name)


# ------------------------------------------------- retrieval

def fetch_chunks(con, chunk_ids: list[str]) -> dict[str, dict]:
    """Chunk text plus its source document, via the chunks -> documents join.
    This join is what lets every retrieved chunk cite a document."""
    if not chunk_ids:
        return {}
    rows = con.execute("""
        SELECT c.chunk_id, c.doc_id, c.heading, c.text, d.title, d.url
        FROM chunks c JOIN documents d USING (doc_id)
        WHERE c.chunk_id IN (SELECT unnest(?))""", [chunk_ids]).fetchall()
    cols = ["chunk_id", "doc_id", "heading", "text", "title", "url"]
    return {r[0]: dict(zip(cols, r)) for r in rows}


class VectorRetriever:
    """Embed the question, search a Lab 9 index, look the hits up in DuckDB.

    kind="ivf"   IVFIndex loaded from index.py's saved build (the product)
    kind="brute" BruteForceIndex over the same vectors (exact baseline)
    """

    def __init__(self, con, kind: str = "ivf", model: str = "minilm",
                 config: str = DEFAULT_CONFIG, probe: int = DEFAULT_PROBE):
        from index import load_brute, load_ivf
        self.con, self.kind, self.config, self.probe = con, kind, config, probe
        self.model = resolve_model(model)
        self.embedder = make_embedder(self.model)
        loader = {"ivf": load_ivf, "brute": load_brute}[kind]
        self.index, self.ids = loader(con, self.model, config)

    def encode_query(self, question: str) -> list[float]:
        return self.embedder.encode([question])[0]

    def search_vector(self, qv: list[float], k: int) -> list[tuple[float, str]]:
        """Top k as (score, chunk_id), best first."""
        if self.kind == "ivf":
            hits = self.index.search(qv, k, probe=self.probe)
        else:
            hits = self.index.search(qv, k)
        return [(score, self.ids[i]) for score, i in hits]

    def search(self, question: str, k: int) -> list[tuple[float, str]]:
        return self.search_vector(self.encode_query(question), k)

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        """The k best chunks, best first: chunk fields + document title/url + score."""
        hits = self.search(question, k)
        details = fetch_chunks(self.con, [cid for _, cid in hits])
        return [{**details[cid], "score": score} for score, cid in hits]


def make_retriever(con, kind: str = "ivf", model: str = "minilm",
                   config: str = DEFAULT_CONFIG, probe: int = DEFAULT_PROBE):
    if kind == "bm25":
        from keyword_search import BM25Retriever
        return BM25Retriever(con, config)
    return VectorRetriever(con, kind, model, config, probe)


# ------------------------------------------------- prompt

HEADER = ("Answer the question using ONLY the sources below. Cite the source id "
          "in brackets after each fact, e.g. [wp:32127@1a2b3c4d/sec600#0007]. "
          "If the sources do not contain the answer, say so.")


def build_prompt(question: str, retrieved: list[dict]) -> str:
    """One line per chunk: "[chunk_id] heading: body". The heading names the
    college and section, so the model knows where each fact is from."""
    lines = []
    for c in retrieved:
        body = c["text"].split("\n", 1)[-1].replace("\n", " ")
        lines.append(f"[{c['chunk_id']}] {c['heading']}: {body}")
    return f"{HEADER}\n\n" + "\n".join(lines) + f"\n\nQuestion: {question}"


# ------------------------------------------------- LLM settings
# Default: Gemma on the Gemini API, called through its native generateContent
# endpoint. To use the course-provided endpoint instead (OpenAI-compatible):
#     export LLM_PROVIDER=openai LLM_BASE_URL=... LLM_MODEL=...
# The API key is read from rag/secrets_local.py (kept out of git by
# .gitignore), or from the LLM_API_KEY environment variable.
# With no key, answer() runs in echo mode and returns the prompt.

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma-4-26b-a4b-it")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
try:
    from secrets_local import LLM_API_KEY
except ImportError:
    LLM_API_KEY = os.environ.get("LLM_API_KEY", "")


def _post_json(url: str, headers: dict, body: dict, retries: int = 3) -> dict:
    """POST a JSON body and return the JSON reply. Retries server errors
    (500-504, usually temporary) with a short wait; any other error stops
    with the server's message."""
    import time
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={**headers, "Content-Type": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (500, 502, 503, 504) and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise SystemExit(f"LLM error {e.code} (model {LLM_MODEL}): "
                             f"{e.read().decode()[:500]}")


def call_llm(prompt: str) -> str:
    if LLM_PROVIDER == "openai":
        data = _post_json(LLM_BASE_URL.rstrip("/") + "/chat/completions",
                          {"Authorization": f"Bearer {LLM_API_KEY}"},
                          {"model": LLM_MODEL,
                           "messages": [{"role": "user", "content": prompt}]})
        return data["choices"][0]["message"]["content"]
    data = _post_json(GEMINI_URL.format(model=LLM_MODEL),
                      {"x-goog-api-key": LLM_API_KEY},
                      {"contents": [{"role": "user", "parts": [{"text": prompt}]}]})
    candidates = data.get("candidates") or []
    if not candidates or "content" not in candidates[0]:
        return f"[no answer returned: {json.dumps(data)[:300]}]"
    parts = candidates[0]["content"].get("parts", [])
    # Skip the model's thinking parts, if any; keep only the answer text.
    return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()


def llm_configured() -> bool:
    return bool(LLM_API_KEY) and (LLM_PROVIDER != "openai" or bool(LLM_BASE_URL))


def answer(question: str, retriever, k: int = 3) -> dict:
    """Retrieve, build the prompt, generate.
    Returns {"answer": text or None, "prompt": prompt, "sources": chunks}.
    answer is None in echo mode (no LLM configured)."""
    sources = retriever.retrieve(question, k)
    prompt = build_prompt(question, sources)
    text = call_llm(prompt) if llm_configured() else None
    return {"answer": text, "prompt": prompt, "sources": sources}
