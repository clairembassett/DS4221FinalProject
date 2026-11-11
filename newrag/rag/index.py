"""Build and load the vector indexes from the embeddings table.

    python3 index.py                        # IVF over MiniLM vectors, sqrt(n) lists
    python3 index.py --lists 64             # a different number of lists
    python3 index.py --model multi-qa       # index a second model's vectors

The product index is Lab 9's IVFIndex (microvector.py, unchanged):
k-means the vectors into L lists, then at query time search only the
`probe` lists whose centroids are nearest the query. BruteForceIndex is the
exact baseline it is measured against.

k-means is the slow part, so the result is saved in DuckDB:
    index_builds     one row per build: model, config, L, build time
    ivf_centroids    the L centroid vectors
    ivf_assignments  which list each chunk_id was assigned to
load_ivf() rebuilds the IVFIndex object from those rows in seconds.
"""

from __future__ import annotations

import argparse
import datetime
import math
import time

from db import connect
from microvector import BruteForceIndex, IVFIndex


def load_vectors(con, model: str, config: str) -> tuple[list[str], list[list[float]]]:
    """chunk_ids and their vectors, in chunk_id order. Position i in both
    lists is index id i, which is how a search result (score, i) is turned
    back into a chunk_id."""
    rows = con.execute("""
        SELECT e.chunk_id, e.vector
        FROM embeddings e JOIN chunks c USING (chunk_id)
        WHERE e.model = ? AND c.config = ?
        ORDER BY e.chunk_id""", [model, config]).fetchall()
    if not rows:
        raise SystemExit(f"No embeddings for model {model!r} and chunks {config!r}. "
                         "Run embed.py first.")
    return [r[0] for r in rows], [r[1] for r in rows]


def index_name(model: str, config: str, n_lists: int) -> str:
    return f"ivf/{model}/{config}/L{n_lists}"


def default_lists(n_vectors: int) -> int:
    """The usual starting point for IVF: about sqrt(n) lists."""
    return max(1, round(math.sqrt(n_vectors)))


def build_ivf(con, model: str, config: str, n_lists: int | None = None):
    """Run k-means, save the result, return (IVFIndex, chunk_ids, seconds)."""
    ids, vectors = load_vectors(con, model, config)
    n_lists = n_lists or default_lists(len(ids))
    t0 = time.time()
    index = IVFIndex(vectors, n_clusters=n_lists)
    seconds = time.time() - t0

    name = index_name(model, config, n_lists)
    for table in ("ivf_assignments", "ivf_centroids", "index_builds"):
        con.execute(f"DELETE FROM {table} WHERE index_name = ?", [name])
    con.execute("INSERT INTO index_builds VALUES (?, ?, ?, ?, ?, ?, ?)",
                [name, model, config, n_lists, len(ids), seconds, datetime.datetime.now()])
    con.executemany("INSERT INTO ivf_centroids VALUES (?, ?, ?)",
                    [(name, list_no, c) for list_no, c in enumerate(index.centroids)])
    con.executemany("INSERT INTO ivf_assignments VALUES (?, ?, ?)",
                    [(name, ids[i], list_no)
                     for list_no, members in enumerate(index.lists) for i in members])
    return index, ids, seconds


def load_ivf(con, model: str, config: str, n_lists: int | None = None):
    """Rebuild a saved IVFIndex without re-running k-means.
    Uses the most recent build for this model and config unless n_lists is
    given. Returns (IVFIndex, chunk_ids)."""
    query = "SELECT index_name, n_vectors FROM index_builds WHERE model = ? AND config = ?"
    params = [model, config]
    if n_lists:
        query += " AND n_lists = ?"
        params.append(n_lists)
    row = con.execute(query + " ORDER BY built_at DESC LIMIT 1", params).fetchone()
    if row is None:
        raise SystemExit(f"No IVF index for {model!r} / {config!r}. Run index.py first.")
    name, n_vectors = row

    ids, vectors = load_vectors(con, model, config)
    position = {chunk_id: i for i, chunk_id in enumerate(ids)}
    centroids = [c for (c,) in con.execute(
        "SELECT centroid FROM ivf_centroids WHERE index_name = ? ORDER BY list_no",
        [name]).fetchall()]
    lists: list[list[int]] = [[] for _ in centroids]
    for chunk_id, list_no in con.execute(
            "SELECT chunk_id, list_no FROM ivf_assignments WHERE index_name = ?",
            [name]).fetchall():
        lists[list_no].append(position[chunk_id])
    if n_vectors != len(ids) or sum(len(l) for l in lists) != len(ids):
        raise SystemExit(f"{name} was built on {n_vectors} vectors but there are now "
                         f"{len(ids)}. Re-run index.py.")

    # Same object IVFIndex.__init__ would produce, minus the k-means run.
    index = IVFIndex.__new__(IVFIndex)
    index.vectors = vectors
    index.centroids = centroids
    index.lists = [sorted(l) for l in lists]
    index.comparisons = 0
    return index, ids


def load_brute(con, model: str, config: str):
    """The exact baseline. Returns (BruteForceIndex, chunk_ids)."""
    ids, vectors = load_vectors(con, model, config)
    return BruteForceIndex(vectors), ids


def main():
    from rag import DEFAULT_CONFIG, resolve_model
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="minilm")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--lists", type=int, default=None,
                   help="number of IVF lists (default: about sqrt of the number of chunks)")
    args = p.parse_args()

    con = connect()
    model = resolve_model(args.model)
    index, ids, seconds = build_ivf(con, model, args.config, args.lists)
    sizes = sorted(len(l) for l in index.lists)
    print(f"built {index_name(model, args.config, len(index.lists))}")
    print(f"  {len(ids)} vectors -> {len(index.lists)} lists in {seconds:.1f}s "
          f"(list sizes: smallest {sizes[0]}, median {sizes[len(sizes) // 2]}, "
          f"largest {sizes[-1]})")


if __name__ == "__main__":
    main()
