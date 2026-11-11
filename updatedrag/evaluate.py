"""Score retrieval on the evaluation questions.

    python3 evaluate.py                          # IVF over MiniLM (the product)
    python3 evaluate.py --retriever brute        # exact baseline, same vectors
    python3 evaluate.py --retriever bm25         # keyword baseline
    python3 evaluate.py --probe 2                # a faster, less exact IVF setting
    python3 evaluate.py --team partner           # only the partner team's questions

Questions come from eval_questions.csv (one row per question) and are loaded
into the eval_questions table on every run:

    q_id              own-01, partner-01, ...
    question          the question as a user would ask it
    relevant_doc_ids  doc_ids that answer it, separated by ";"   (wp:32127)
    author            the team member who read the article and wrote the question
    source_team       own | partner
    answer_pattern    optional regex the answering passage contains (1890|1857)

Scores, each averaged over questions:

    recall@k     share of the relevant documents that appear in the top k
                 chunks (the assignment's document-level labels)
    MRR@10       1 / rank of the first chunk from a relevant document
    passage@3    the top 3 contain a chunk from a relevant document that also
                 matches answer_pattern: "found the right college but the
                 wrong section" counts as a miss here (questions with a
                 pattern only)
    passage MRR  as MRR@10, but for that stricter match

Every run's per-question results go into the eval_results table, so the
failure analysis can be done with SQL afterwards. Misses are also printed.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import re
import time

from db import connect
from rag import DEFAULT_CONFIG, DEFAULT_PROBE, fetch_chunks, make_retriever

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_CSV = os.path.join(HERE, "eval_questions.csv")
DEPTH = 10            # chunks retrieved per question
RECALL_KS = (1, 3, 5, 10)


def load_questions(con, path: str = QUESTIONS_CSV) -> None:
    """Replace the eval_questions table with the CSV's contents, checking
    that every relevant doc_id exists in documents."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    known = {r[0] for r in con.execute("SELECT doc_id FROM documents").fetchall()}
    records, problems, no_author = [], [], []
    for r in rows:
        doc_ids = [d.strip() for d in r["relevant_doc_ids"].split(";") if d.strip()]
        unknown = [d for d in doc_ids if d not in known]
        if not doc_ids or unknown:
            problems.append(f"{r['q_id']}: unknown or missing doc ids {unknown or doc_ids}")
        if not r["author"].strip():
            no_author.append(r["q_id"])
        records.append((r["q_id"], r["question"], doc_ids, r["author"].strip(),
                        r["source_team"].strip() or "own", r.get("answer_pattern") or None))
    if problems:
        raise SystemExit("eval_questions.csv has bad rows:\n  " + "\n  ".join(problems))
    if no_author:
        print(f"WARNING: {len(no_author)} questions have no author ({', '.join(no_author)}).\n"
              "  The assignment requires questions written by a team member who read the\n"
              "  source; fill in the author column (or replace the question).\n")
    con.execute("DELETE FROM eval_questions")
    con.executemany("INSERT INTO eval_questions VALUES (?, ?, ?, ?, ?, ?)", records)


def first_rank(flags: list[bool]) -> int | None:
    return next((rank for rank, ok in enumerate(flags, start=1) if ok), None)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--retriever", choices=["ivf", "brute", "bm25"], default="ivf")
    p.add_argument("--model", default="minilm")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--probe", type=int, default=DEFAULT_PROBE)
    p.add_argument("--team", choices=["own", "partner", "all"], default="all")
    p.add_argument("--questions", default=QUESTIONS_CSV)
    p.add_argument("-q", "--quiet", action="store_true", help="scores only")
    args = p.parse_args()

    con = connect()
    load_questions(con, args.questions)
    retriever = make_retriever(con, args.retriever, args.model, args.config, args.probe)
    where = "" if args.team == "all" else f"WHERE source_team = '{args.team}'"
    questions = con.execute(f"""SELECT q_id, question, relevant_doc_ids, source_team,
                                       answer_pattern FROM eval_questions {where}
                                ORDER BY q_id""").fetchall()

    label = args.retriever + ("" if args.retriever == "bm25" else f" {retriever.model}")
    if args.retriever == "ivf":
        label += f" probe={args.probe}"
    run_id = f"{datetime.datetime.now().isoformat(timespec='seconds')} {label} {args.config}"

    results = []   # per question: (q_id, team, recalls, doc_rank, passage_rank, has_pattern, ...)
    for q_id, question, relevant, team, pattern in questions:
        t0 = time.time()
        hits = retriever.search(question, DEPTH)
        seconds = time.time() - t0
        ids = [cid for _, cid in hits]
        chunks = fetch_chunks(con, ids)
        docs = [chunks[cid]["doc_id"] for cid in ids]
        rel = set(relevant)
        recalls = {k: len(rel & set(docs[:k])) / len(rel) for k in RECALL_KS}
        doc_rank = first_rank([d in rel for d in docs])
        passage_rank = None
        if pattern:
            passage_rank = first_rank([d in rel and re.search(pattern, chunks[cid]["text"]) is not None
                                       for d, cid in zip(docs, ids)])
        results.append((q_id, team, recalls, doc_rank, passage_rank, bool(pattern),
                        question, chunks[ids[0]]["heading"] if ids else "(nothing)"))
        con.execute("INSERT INTO eval_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [run_id, q_id, args.retriever, retriever.model, args.config,
                     args.probe if args.retriever == "ivf" else None,
                     ids, doc_rank, passage_rank, seconds])

    print(f"run: {run_id}\n")
    print(f"   {'questions':<10}{'n':>4}" + "".join(f"{'R@' + str(k):>7}" for k in RECALL_KS)
          + f"{'MRR@10':>8}{'pass@3':>8}{'passMRR':>9}")
    teams = sorted({r[1] for r in results})
    for team in (["all"] + teams if len(teams) > 1 else teams):
        rows = [r for r in results if team == "all" or r[1] == team]
        n = len(rows)
        line = f"   {team:<10}{n:>4}"
        for k in RECALL_KS:
            line += f"{sum(r[2][k] for r in rows) / n:>7.2f}"
        line += f"{sum(1 / r[3] for r in rows if r[3]) / n:>8.2f}"
        with_pattern = [r for r in rows if r[5]]
        if with_pattern:
            m = len(with_pattern)
            line += f"{sum(1 for r in with_pattern if r[4] and r[4] <= 3) / m:>8.2f}"
            line += f"{sum(1 / r[4] for r in with_pattern if r[4]) / m:>9.2f}"
        print(line)

    if args.quiet:
        return
    misses = [r for r in results if not r[3] or r[3] > 3 or (r[5] and (not r[4] or r[4] > 3))]
    print(f"\n   {len(misses)} questions where the answer is not in the top 3:")
    for q_id, team, recalls, doc_rank, passage_rank, has_pattern, question, top in misses:
        print(f"   {q_id}  doc rank {doc_rank or '-'}, passage rank "
              f"{(passage_rank or '-') if has_pattern else 'n/a'}:  {question}\n"
              f"           top result: {top}")
    print("\n   Per-question results are in the eval_results table, run_id above.")


if __name__ == "__main__":
    main()
