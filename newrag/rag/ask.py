"""Ask the college RAG pipeline a question from the command line.

    python3 ask.py "When was the University of Chicago founded?"
    python3 ask.py -k 5 "What are Notre Dame's athletic teams known as?"
    python3 ask.py --retrieve-only "Who founded MIT?"      # top-k chunks, no LLM
    python3 ask.py --retriever brute "Who founded MIT?"    # exact baseline
    python3 ask.py --retriever bm25 "Who founded MIT?"     # keyword baseline
    python3 ask.py                                         # interactive: load once, ask many

Interactive mode is the one to use for the live demo: the model and index
load once, then each question takes about a second.

Prints the answer with [chunk_id] citations, then the cited chunks with
their college, section and Wikipedia URL. Without an API key (see rag.py)
it prints the prompt it would have sent instead (echo mode).
"""

from __future__ import annotations

import argparse
import time

from db import connect
from rag import DEFAULT_CONFIG, DEFAULT_PROBE, answer, make_retriever


def show_chunks(chunks: list[dict], full: bool = False) -> None:
    for rank, c in enumerate(chunks, start=1):
        body = c["text"].split("\n", 1)[-1].replace("\n", " ")
        if not full and len(body) > 220:
            body = body[:220] + "…"
        print(f"  {rank}. [{c['chunk_id']}]  score {c['score']:.3f}")
        print(f"     {c['heading']}")
        print(f"     {c['url']}")
        print(f"     {body}\n")


def run(question: str, retriever, k: int, retrieve_only: bool) -> None:
    t0 = time.time()
    if retrieve_only:
        chunks = retriever.retrieve(question, k)
        print(f"\nTop {k} chunks for: {question}   ({(time.time() - t0) * 1000:.0f} ms)\n")
        show_chunks(chunks, full=True)
        return
    result = answer(question, retriever, k)
    if result["answer"] is None:
        print("\n[echo mode: no LLM API key configured. This is the prompt that would be sent.]\n")
        print(result["prompt"])
        return
    print(f"\n{result['answer']}\n\n({time.time() - t0:.1f}s)  Sources:\n")
    show_chunks(result["sources"])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="*", help="leave out for interactive mode")
    p.add_argument("-k", type=int, default=3, help="chunks to retrieve (default 3)")
    p.add_argument("--retriever", choices=["ivf", "brute", "bm25"], default="ivf")
    p.add_argument("--model", default="minilm", help="embedding model (default minilm)")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="chunking config (default sec600)")
    p.add_argument("--probe", type=int, default=DEFAULT_PROBE,
                   help=f"IVF lists to search (default {DEFAULT_PROBE})")
    p.add_argument("--retrieve-only", action="store_true",
                   help="show the top-k chunks without calling the LLM")
    args = p.parse_args()

    con = connect()
    t0 = time.time()
    retriever = make_retriever(con, args.retriever, args.model, args.config, args.probe)
    print(f"[{args.retriever} retriever, {args.config}, loaded in {time.time() - t0:.1f}s]")

    if args.question:
        run(" ".join(args.question), retriever, args.k, args.retrieve_only)
        return
    print("Ask a question (empty line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            break
        if not question:
            break
        run(question, retriever, args.k, args.retrieve_only)


if __name__ == "__main__":
    main()
