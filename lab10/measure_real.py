"""Lab 10, Part 4: the same pipeline on 4,000 real abstracts. Run after the
tests pass:  python3 measure_real.py   (about 30 s: pure Python over 4,000 x 2048)

Parts 1 to 3 used a 24-document corpus about this course, small enough to judge
by eye. This script points YOUR Retriever at 4,000 real arXiv abstracts (the
papers Lab 9 embedded) and compares two embedders on them:

    hashed     your micro_rag.embed(): a hashed bag of words, no model
    MiniLM     a real sentence-embedding model (all-MiniLM-L6-v2), whose
               vectors for every abstract and every question ship precomputed
               in data/, so no download or GPU is needed

Three views: what each embedder retrieves for the same question; how much
the two agree; and a labeled mini-eval, ten questions written to paraphrase
one specific paper without reusing its title words, scored with hit@3 and MRR@10.
"""

import gzip
import json
import math
import os
import struct

import micro_rag

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DIM_MINILM = 384


def load_docs() -> list[dict]:
    with gzip.open(os.path.join(DATA, "arxiv_4000.jsonl.gz"), "rt") as f:
        return [json.loads(line) for line in f]


def read_f16(name: str, dim: int) -> list[list[float]]:
    raw = open(os.path.join(DATA, name), "rb").read()
    flat = struct.unpack(f"<{len(raw) // 2}e", raw)
    return [list(flat[i:i + dim]) for i in range(0, len(flat), dim)]


def top_k(query: list[float], vectors: list[list[float]], k: int) -> list[int]:
    scored = sorted(((sum(a * b for a, b in zip(query, v)), i) for i, v in enumerate(vectors)), reverse=True)
    return [i for _, i in scored[:k]]


def main():
    docs = load_docs()
    chunks = [{"doc": d["id"], "title": d["title"], "text": d["title"] + ". " + d["abstract"]} for d in docs]
    print(f"Embedding {len(chunks)} abstracts with your hashed embedder (DIM={micro_rag.DIM})...")
    retriever = micro_rag.Retriever(chunks)                     # YOUR code, real corpus
    minilm_docs = read_f16("arxiv_4000_384.f16", DIM_MINILM)  # precomputed model vectors

    questions = [l.strip() for l in open(os.path.join(DATA, "queries.txt")) if l.strip()]
    minilm_q = read_f16("queries_384.f16", DIM_MINILM)

    print("\n1. Same question, two embedders (top-3 titles)")
    for qi in (2, 9, 13):
        print(f"\n   Q: {questions[qi]}")
        print("   hashed:")
        for r in retriever.retrieve(questions[qi], k=3):
            print(f"      {r['score']:.3f}  {r['title'][:72]}")
        print("   MiniLM:")
        for i in top_k(minilm_q[qi], minilm_docs, 3):
            print(f"      {sum(a * b for a, b in zip(minilm_q[qi], minilm_docs[i])):.3f}  {docs[i]['title'][:72]}")

    print("\n2. Labeled mini-eval: 10 questions, each paraphrasing one paper without its title words")
    ev = [json.loads(l) for l in open(os.path.join(DATA, "real_eval.jsonl"))]
    ev_q = read_f16("real_eval_384.f16", DIM_MINILM)
    id_of = {d["id"]: i for i, d in enumerate(docs)}
    rows = []
    for e, qv in zip(ev, ev_q):
        want = e["relevant"]
        got_h = [r["doc"] for r in retriever.retrieve(e["question"], k=10)]
        got_m = [docs[i]["id"] for i in top_k(qv, minilm_docs, 10)]
        rows.append((want, got_h, got_m, e["question"]))

    def score(rank_lists):
        hit3 = sum(1 for want, got in rank_lists if want in got[:3]) / len(rank_lists)
        mrr = sum(1 / (got.index(want) + 1) if want in got else 0 for want, got in rank_lists) / len(rank_lists)
        return hit3, mrr
    h3h, mrrh = score([(w, gh) for w, gh, _, _ in rows])
    h3m, mrrm = score([(w, gm) for w, _, gm, _ in rows])
    print(f"   {'embedder':>8}  {'hit@3':>6}  {'MRR@10':>6}")
    print(f"   {'hashed':>8}  {h3h:>6.2f}  {mrrh:>6.2f}")
    print(f"   {'MiniLM':>8}  {h3m:>6.2f}  {mrrm:>6.2f}")
    print("\n   question by question (rank of the right paper, - if not in the top 10):")
    for want, gh, gm, q in rows:
        rh = gh.index(want) + 1 if want in gh else "-"
        rm = gm.index(want) + 1 if want in gm else "-"
        print(f"   hashed {rh!s:>2}  MiniLM {rm!s:>2}   {q[:70]}")

    print("\nThink about these for class:")
    print("  1. Where the hashed embedder finds the paper, which words did the question")
    print("     share with the abstract? Where it misses, what did the paraphrase change?")
    print("  2. This lab uses a pretrained MiniLM model without fine-tuning it on the")
    print("     evaluation questions. How can learned representations help with")
    print("     paraphrases, and what does encoding a new question cost?")
    print("  3. Your project corpus: which of the two failure modes will it show more,")
    print("     and how many labeled questions would convince you either way?")


if __name__ == "__main__":
    main()
