"""Score retrieval on a small labeled set of college questions.

    python3 evaluate.py                       # default tfidf embedder
    python3 evaluate.py --embedder hashed     # the lab's original embedder
    python3 evaluate.py --embedder minilm     # learned model (pip install sentence-transformers)

Each EVAL item is (question, doc_id of the right college, regex the answer
chunk must contain). A retrieved chunk counts as correct only if it comes
from that college AND contains the answer, so "found the right college but
the wrong section" is a miss for hit@3 / MRR, and is reported separately as
"college@3".

Add your own questions to EVAL as the project grows; ~30 is a reasonable
target before trusting the numbers.
"""

import argparse
import re
import time

from rag import EMBEDDERS, Retriever, chunk_corpus

EVAL = [
    ("When was the University of Chicago founded?",               "wp:32127",    r"1890|1857"),
    ("What are Georgia Tech's sports teams called?",              "wp:28486339", r"Yellow Jackets"),
    ("Who founded MIT?",                                          "wp:18879",    r"Rogers"),
    ("When was Stanford University founded and by whom?",         "wp:26977",    r"1885"),
    ("When was Harvard founded?",                                 "wp:18426501", r"1636"),
    ("What are Notre Dame's athletic teams known as?",            "wp:146269",   r"Fighting Irish"),
    ("When did Johns Hopkins University open?",                   "wp:38420",    r"1876"),
    ("Who was the first president of Johns Hopkins University?",  "wp:38420",    r"Gilman"),
    ("How did William Marsh Rice die?",                           "wp:25813",    r"murder"),
    ("Who established Carnegie Mellon University?",               "wp:48093",    r"Andrew Carnegie"),
    ("How large is Yale's endowment?",                            "wp:34273",    r"54\.5 billion"),
    ("How much did Duke spend on research in 2024?",              "wp:53273",    r"1\.5 billion"),
    ("How many students showed up on Texas A&M's first day of classes?", "wp:29927", r"six students"),
    ("When did Texas A&M end mandatory participation in the Corps of Cadets?", "wp:29927", r"ended the requirement"),
    ("Who donated money to found the new University of Chicago in 1890?", "wp:32127", r"Rockefeller"),
]


def is_hit(chunk, doc_id, pattern):
    return chunk["doc"] == doc_id and re.search(pattern, chunk["text"]) is not None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedder", choices=sorted(EMBEDDERS), default="tfidf")
    p.add_argument("-q", "--quiet", action="store_true", help="scores only")
    args = p.parse_args()

    t0 = time.time()
    retriever = Retriever(chunk_corpus(), embedder=args.embedder)
    print(f"[{args.embedder}] indexed {len(retriever.chunks)} chunks "
          f"in {time.time() - t0:.1f}s\n")

    hit3 = college3 = 0
    rr_total = 0.0
    rows = []
    for question, doc_id, pattern in EVAL:
        got = retriever.retrieve(question, k=10)
        ranks = [r for r, c in enumerate(got, start=1) if is_hit(c, doc_id, pattern)]
        first = ranks[0] if ranks else None
        hit3 += bool(first and first <= 3)
        college3 += any(c["doc"] == doc_id for c in got[:3])
        rr_total += 1 / first if first else 0
        rows.append((first, got[0]["heading"], question))

    n = len(EVAL)
    print(f"   hit@3     {hit3 / n:.2f}   (answer chunk in the top 3)")
    print(f"   MRR@10    {rr_total / n:.2f}")
    print(f"   college@3 {college3 / n:.2f}   (right college anywhere in the top 3)\n")
    if args.quiet:
        return
    print("   rank  question  ->  top result")
    for first, top, question in rows:
        rank = f"{first:>4}" if first else "   -"
        print(f"   {rank}  {question}\n         -> {top}")


if __name__ == "__main__":
    main()
