"""Lab 10 measurement — run AFTER test_rag.py passes:  python3 measure_rag.py

Three exhibits:

    1. The DIM sweep — shrink the embedding and watch retrieval collapse
       as hash collisions merge unrelated words onto shared dimensions.
    2. The eval table — every question, what came back, at what rank.
    3. The synonym wall — the same information need phrased with matching
       vs. mismatched vocabulary. Bag-of-words has no idea "popular pages
       in memory" means "buffer pool". This failure is WHY learned
       embedding models exist; your project swaps one in.
"""

import micro_rag
from corpus import EVAL
from micro_rag import Retriever, chunk_corpus

CHUNKS = chunk_corpus()


def eval_scores(retriever):
    hits, rr_total = 0, 0.0
    for question, relevant in EVAL:
        docs = [c["doc"] for c in retriever.retrieve(question, k=3)]
        ranks = [r for r, d in enumerate(docs, start=1) if d in relevant]
        if ranks:
            hits += 1
            rr_total += 1.0 / ranks[0]
    return hits / len(EVAL), rr_total / len(EVAL)


print("1. Embedding width vs. retrieval quality (same corpus, same questions)")
print(f"   {'DIM':>6}  {'hit@3':>6}  {'MRR@3':>6}")
for dim in (64, 256, 2048):
    micro_rag.DIM = dim              # embed() reads the module global
    hit3, mrr = eval_scores(Retriever(CHUNKS))
    print(f"   {dim:>6}  {hit3:>6.2f}  {mrr:>6.2f}")
micro_rag.DIM = 2048

print()
print("2. The eval set, question by question (DIM=2048)")
r = Retriever(CHUNKS)
for question, relevant in EVAL:
    docs = [c["doc"] for c in r.retrieve(question, k=3)]
    ranks = [i for i, d in enumerate(docs, start=1) if d in relevant]
    rank = f"rank {ranks[0]}" if ranks else "MISS  "
    print(f"   {rank}  {', '.join(docs):<32}  {question}")

print()
print("3. The synonym wall (both questions mean 'the buffer pool')")
for question in (
    "Why can't the OS page cache replace the database's own cache?",
    "What keeps the most popular pages sitting in memory?",
):
    got = [(c["doc"], c["score"]) for c in r.retrieve(question, k=3)]
    pretty = ", ".join(f"{d} ({s:.2f})" for d, s in got)
    print(f"   {question}\n      -> {pretty}")
print("   Shared words drive similarity; meaning does not. A learned")
print("   embedding model maps both phrasings near the same point.")
