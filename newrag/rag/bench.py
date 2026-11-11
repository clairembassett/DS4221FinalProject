"""Measure the index: build time, queries per second, recall vs. speed.

    python3 bench.py                         # every saved IVF build for MiniLM
    python3 bench.py --build 32 64 216       # first build IVF with these list counts

For each IVF build and each probe setting, compared with brute force on the
same vectors and the same queries (the eval_questions):

    recall@k     share of brute force's exact top k that IVF also returned
                 (microvector.recall_at_k): what the speed costs
    QPS          index searches per second. Query embedding is done once up
                 front and excluded, so this measures the index alone
    comparisons  dot products per query (brute force: one per chunk)

Build times come from index_builds. Results go to the bench_results table.
"""

from __future__ import annotations

import argparse
import datetime
import time

from db import connect
from evaluate import load_questions
from index import build_ivf, load_brute, load_ivf
from microvector import recall_at_k
from rag import DEFAULT_CONFIG, make_embedder, resolve_model

PROBES = (1, 2, 4, 8, 16, 32)


def timed_searches(index, queries, k, **kwargs):
    """Run every query; return (results, queries per second, comparisons per query)."""
    index.comparisons = 0
    t0 = time.perf_counter()
    results = [index.search(q, k, **kwargs) for q in queries]
    seconds = time.perf_counter() - t0
    return results, len(queries) / seconds, index.comparisons / len(queries)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="minilm")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--build", type=int, nargs="*", default=[],
                   help="IVF list counts to build before measuring")
    args = p.parse_args()

    con = connect()
    model = resolve_model(args.model)
    for n_lists in args.build:
        _, _, seconds = build_ivf(con, model, args.config, n_lists)
        print(f"built L={n_lists} in {seconds:.1f}s")

    load_questions(con)
    questions = [q for (q,) in con.execute("SELECT question FROM eval_questions").fetchall()]
    t0 = time.perf_counter()
    queries = make_embedder(model).encode(questions)
    embed_ms = (time.perf_counter() - t0) * 1000 / len(queries)

    run_id = f"{datetime.datetime.now().isoformat(timespec='seconds')} {model} {args.config}"
    brute, _ = load_brute(con, model, args.config)
    exact, brute_qps, brute_cmp = timed_searches(brute, queries, args.k)
    rows = [("brute", None, 1.0, brute_qps, brute_cmp)]

    builds = con.execute("""SELECT n_lists, build_seconds FROM index_builds
                            WHERE model = ? AND config = ? ORDER BY n_lists""",
                         [model, args.config]).fetchall()
    for n_lists, build_s in builds:
        ivf, _ = load_ivf(con, model, args.config, n_lists)
        name = f"IVF L={n_lists} (built in {build_s:.0f}s)"
        for probe in PROBES:
            if probe > n_lists:
                continue
            approx, qps, cmp = timed_searches(ivf, queries, args.k, probe=probe)
            recall = sum(recall_at_k(a, e, args.k) for a, e in zip(approx, exact)) / len(queries)
            rows.append((name, probe, recall, qps, cmp))

    print(f"\n{len(queries)} queries, k={args.k}, {model}, {args.config}; "
          f"query embedding {embed_ms:.0f} ms each (not included below)\n")
    print(f"   {'index':<28}{'probe':>6}{'recall@' + str(args.k):>11}{'QPS':>9}"
          f"{'speedup':>9}{'dots/query':>12}")
    for name, probe, recall, qps, cmp in rows:
        print(f"   {name:<28}{probe or '-':>6}{recall:>11.2f}{qps:>9.1f}"
              f"{qps / brute_qps:>8.1f}x{cmp:>12.0f}")
        con.execute("INSERT INTO bench_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [run_id, name.split(" (")[0], probe or 0, args.k, len(queries),
                     recall, qps, cmp])


if __name__ == "__main__":
    main()
