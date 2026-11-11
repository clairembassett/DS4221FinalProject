"""Retrieval milestone (Nov 12): run example queries and save the top chunks
and the LLM's cited answer for each.

    python3 milestone.py                  # writes ../milestone_examples.md
    python3 milestone.py --retriever brute
    python3 milestone.py --no-llm         # retrieved chunks only

Edit QUERIES to the five examples your team wants to show.
The output file records the retriever, model, index and timing alongside
each query's retrieved chunks, so it can be handed in as is.
"""

from __future__ import annotations

import argparse
import os
import time

from db import connect
from rag import (DEFAULT_CONFIG, DEFAULT_PROBE, LLM_MODEL, build_prompt, call_llm,
                 llm_configured, make_retriever)

QUERIES = [
    "When was the University of Chicago founded?",
    "What are Notre Dame's athletic teams known as?",
    "Who was the first president of Johns Hopkins University?",
    "How large is Yale's endowment?",
    "Which university did Thomas Jefferson found?",
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--retriever", choices=["ivf", "brute", "bm25"], default="ivf")
    p.add_argument("--model", default="minilm")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--probe", type=int, default=DEFAULT_PROBE)
    p.add_argument("--out", default=os.path.join(here, "..", "milestone_examples.md"))
    p.add_argument("--no-llm", action="store_true", help="retrieval only, no generated answers")
    args = p.parse_args()
    use_llm = not args.no_llm and llm_configured()
    if not args.no_llm and not use_llm:
        print("No LLM API key configured (see rag.py): saving retrieved chunks only.")

    con = connect()
    n_docs, = con.execute("SELECT count(*) FROM documents").fetchone()
    n_chunks, = con.execute("SELECT count(*) FROM chunks WHERE config = ?",
                            [args.config]).fetchone()
    t0 = time.time()
    retriever = make_retriever(con, args.retriever, args.model, args.config, args.probe)
    load_s = time.time() - t0

    setup = f"`{args.retriever}`"
    if args.retriever != "bm25":
        setup += f" over `{retriever.model}`"
        if args.retriever == "ivf":
            setup += f", {len(retriever.index.lists)} lists, probe {args.probe}"
    lines = ["# Retrieval milestone: example queries", "",
             f"- Corpus: {n_docs} documents, {n_chunks} chunks (config `{args.config}`), in DuckDB",
             f"- Retriever: {setup}",
             f"- Index loaded in {load_s:.1f}s; top {args.k} per query",
             f"- Answers: {LLM_MODEL if use_llm else 'not generated'}", ""]
    for n, q in enumerate(QUERIES, start=1):
        t = time.time()
        chunks = retriever.retrieve(q, args.k)
        ms = (time.time() - t) * 1000
        lines += [f"## {n}. {q}", ""]
        if use_llm:
            t = time.time()
            reply = call_llm(build_prompt(q, chunks))
            llm_s = time.time() - t
            lines += ["**Answer:**", "", *[f"> {line}" for line in reply.splitlines()], "",
                      f"_retrieval {ms:.0f} ms, answer {llm_s:.1f} s_", "",
                      "**Retrieved chunks:**", ""]
        else:
            lines += [f"_retrieval {ms:.0f} ms_", ""]
        for rank, c in enumerate(chunks, start=1):
            body = c["text"].split("\n", 1)[-1].replace("\n", " ")
            lines += [f"{rank}. **{c['heading']}** · score {c['score']:.3f} · "
                      f"`{c['chunk_id']}`  ", f"   {body}", ""]
        print(f"{n}. {q}  ->  {chunks[0]['heading'] if chunks else '(nothing)'}")
        if use_llm:
            print(f"   {reply.splitlines()[0] if reply else '(empty answer)'}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {os.path.relpath(args.out)}")


if __name__ == "__main__":
    main()