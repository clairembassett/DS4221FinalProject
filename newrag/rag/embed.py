"""Embed chunks and store the vectors in the embeddings table.

    python3 embed.py                     # all-MiniLM-L6-v2, chunks from config sec600
    python3 embed.py --model multi-qa    # a second model, for the comparison
    python3 embed.py --config sec400     # chunks from another chunking config

Only chunks that have no vector yet for this model are embedded, so
re-running is cheap and an interrupted run picks up where it stopped.
On a laptop CPU, MiniLM takes a few minutes for ~11,600 chunks.
"""

from __future__ import annotations

import argparse
import time

from db import connect
from rag import DEFAULT_CONFIG, make_embedder

BATCH = 512   # chunks embedded and written per round


def embed_missing(con, model: str, config: str) -> int:
    embedder = make_embedder(model)
    todo = con.execute("""
        SELECT c.chunk_id, c.text
        FROM chunks c
        WHERE c.config = ?
          AND NOT EXISTS (SELECT 1 FROM embeddings e
                          WHERE e.chunk_id = c.chunk_id AND e.model = ?)
        ORDER BY c.chunk_id""", [config, embedder.name]).fetchall()
    print(f"{len(todo)} chunks to embed with {embedder.name}")
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        vectors = embedder.encode([text for _, text in batch])
        con.executemany("INSERT INTO embeddings VALUES (?, ?, ?, ?)",
                        [(chunk_id, embedder.name, len(v), v)
                         for (chunk_id, _), v in zip(batch, vectors)])
        print(f"  {min(start + BATCH, len(todo))}/{len(todo)}")
    return len(todo)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="minilm")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    args = p.parse_args()

    con = connect()
    t0 = time.time()
    n = embed_missing(con, args.model, args.config)
    print(f"embedded {n} chunks in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
