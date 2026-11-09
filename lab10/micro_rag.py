"""micro-rag — retrieval-augmented generation, small enough to trust.

Lab 10 of Advanced Databases for Data Science (DS 6XXX, Fall 2026).
Runs on Lab 9's microvector (reference ships in this folder).

The pipeline, one function per stage:

    chunk_corpus()   documents -> chunks              (provided)
    embed(text)      text -> 2048-dim unit vector     (provided; see note)
    Retriever        chunks -> index -> top-k         (YOUR JOB)
    build_prompt()   question + chunks -> a prompt    (YOUR JOB)
    answer()         prompt -> text                   (provided; offline echo
                                                       mode, or a real LLM via
                                                       LLM_BASE_URL/KEY/MODEL)

EMBEDDING NOTE, read this: real systems use a learned sentence-embedding
model. This lab uses hashed bag-of-words — each word hashes to a dimension,
counts get tf-scaled, the vector is normalized. It is deterministic,
offline, and honest about being the weakest link: words shared between
question and chunk drive similarity, synonyms don't. The RETRIEVAL
machinery above it is exactly what you'd run under a real model (and your
project will, with this file as the skeleton).

Run the tests any time:   python3 test_rag.py
Talk to it:               python3 ask.py "why do databases read whole blocks?"
"""

from __future__ import annotations

import hashlib
import math
import os
import re

from corpus import DOCS
from microvector import BruteForceIndex

DIM = 2048

# Words so common they carry no signal — without this list, every doc looks
# like every other doc because they all share "the", "of", "and"...
STOPWORDS = set("""a an and are as at be because by can each for from has have
how in is it its like more most no not of on one only or so than that the
their them then there these they this to was what when where which while who
why will with you your""".split())


# ------------------------------------------------- provided: chunk + embed

def chunk_corpus(max_words: int = 60) -> list[dict]:
    """Split each doc at sentence boundaries, targeting `max_words` words.
    A single sentence longer than the target stays intact. Returns [{"doc": id, "title": t, "text": ...}].
    With this corpus most docs yield 1-2 chunks — chunking earns its keep
    on real corpora; the interface is what matters."""
    chunks = []
    for doc_id, title, text in DOCS:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        current: list[str] = []
        for s in sentences:
            if current and len(" ".join(current + [s]).split()) > max_words:
                chunks.append({"doc": doc_id, "title": title,
                               "text": " ".join(current)})
                current = []
            current.append(s)
        if current:
            chunks.append({"doc": doc_id, "title": title,
                           "text": " ".join(current)})
    return chunks


def embed(text: str) -> list[float]:
    """Hashed bag-of-words ("the hashing trick"): each non-stopword hashes
    to one of DIM dimensions and bumps it, scaled by 1/(1+log(count));
    the result is unit-normalized so dot = cosine. Two words can collide
    on a dimension — DIM=2048 makes collisions rare enough (the lab's
    measurement shows hit@3 collapse when you shrink it)."""
    counts: dict[str, int] = {}
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    v = [0.0] * DIM
    for word, count in counts.items():
        dim = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
        v[dim] += 1.0 / (1.0 + math.log(count))
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


# ------------------------------------------------- YOUR JOB: retrieval

class Retriever:
    """Embeds every chunk at build time; answers top-k at query time."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        # ---------------- YOUR JOB starts here. ----------------
        # Embed every chunk's text and build a BruteForceIndex over the
        # vectors (store it as self.index). Keep the order aligned:
        # index id i must correspond to self.chunks[i].
        
        # raise NotImplementedError
#
        # look at chunks in stored order
        vectors = []
        for chunk in self.chunks:
            # turn this chunk's text into vector 
            vectors.append(embed(chunk["text"]))

        # wrap the aligned vectors in BFI 
        self.index = BruteForceIndex(vectors)
 

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        """The k best chunks for the question, each as
        {"doc", "title", "text", "score"} — score is the similarity,
        results best-first.

        Sketch: embed the question; self.index.search(qv, k); attach
        scores to copies of the matching chunk dicts."""
        # TODO: 
        # raise NotImplementedError

        # embed question :: 
        # Turn the question into the same kind of unit vector the index stores.
        qv = embed(question)

        # Ask the index for the k nearest neighbors: [(score, chunk_id), ...]
        # already sorted best-first. Returns fewer than k if the corpus is small.
        hits = self.index.search(qv, k)

        results = []
        for score, i in hits:
            # Copy so attaching "score" cannot mutate the stored chunk dict
            chunk = dict(self.chunks[i])
            # record cosine similarity (dot of two unit vectors) score on the copy 
            chunk["score"] = score
            results.append(chunk)

        # returns one dict per hit, still in best-first order from the index
        return results 

    # ---------------- YOUR JOB ends here (class). ----------------


def build_prompt(question: str, retrieved: list[dict]) -> str:
    """The context-assembly step — where RAG actually happens.

    ---------------- YOUR JOB ----------------
    Produce exactly this shape (the tests parse it):

        Answer using ONLY the sources below. Cite as [doc_id].

        [blocks] Blocks and pages: Disks move data in fixed-size...
        [fsync] Durability and fsync: A normal write only reaches...

        Question: <the question>

    One line per retrieved chunk: "[" + doc + "] " + title + ": " + text.
    Blank lines between the header, sources, and question sections."""
    #  TODO
    # raise NotImplementedError

    # instruction the model must follow: answer only from the sources below.
    header = "Answer using ONLY the sources below. Cite as [doc_id]."

    # One source line per retrieved chunk, in the order retrieve() returned them.
    source_lines = []
    for chunk in retrieved:
        # Format: [doc_id] title: text  (tests parse this exact shape).
        source_lines.append(f"[{chunk['doc']}] {chunk['title']}: {chunk['text']}")

    # Join source lines with newlines. Empty retrieved -> empty string.
    sources = "\n".join(source_lines)

    # Blank lines separate header, sources, and the question.
    return f"{header}\n\n{sources}\n\nQuestion: {question}"


# ------------------------------------------------- provided: generation

def answer(question: str, retriever: Retriever, k: int = 3) -> str:
    """Retrieve, build the prompt, generate. Without LLM_* env vars this
    runs in ECHO mode: it returns the prompt plus a note — retrieval is
    the graded part, and the prompt IS the retrieval made visible."""
    retrieved = retriever.retrieve(question, k)
    prompt = build_prompt(question, retrieved)
    base = os.environ.get("LLM_BASE_URL")
    if not base:
        cited = ", ".join(c["doc"] for c in retrieved)
        return (prompt + "\n\n[echo mode — no LLM configured. "
                f"A real model would answer from: {cited}]")
    # Real mode: OpenAI-compatible chat endpoint (course endpoint in class).
    import json
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
