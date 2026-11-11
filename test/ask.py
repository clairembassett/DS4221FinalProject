"""Ask the college RAG pipeline a question from the command line.

    python3 ask.py "When was the University of Chicago founded?"
    python3 ask.py -k 5 "Which colleges are in Boston?"
    python3 ask.py --embedder minilm "Who founded MIT?"

Without LLM_* environment variables this prints the assembled prompt (echo
mode), so you can see exactly which chunks retrieval chose.
"""

import argparse
import time

from rag import DEFAULT_DATA_DIR, EMBEDDERS, Retriever, answer, chunk_corpus

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="+")
    p.add_argument("-k", type=int, default=3, help="how many chunks to retrieve (default 3)")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help="folder of college JSON files (default: ../data)")
    p.add_argument("--embedder", choices=sorted(EMBEDDERS), default="tfidf",
                   help="tfidf (default), hashed (the lab's original), or minilm")
    args = p.parse_args()

    t0 = time.time()
    chunks = chunk_corpus(args.data_dir)
    retriever = Retriever(chunks, embedder=args.embedder)
    print(f"[indexed {len(chunks)} chunks in {time.time() - t0:.1f}s]\n")
    print(answer(" ".join(args.question), retriever, k=args.k))
