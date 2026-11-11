"""Load the college JSON files into the documents and chunks tables.

    python3 ingest.py                    # ../data, 600-character section chunks
    python3 ingest.py --max-chars 400    # add a second chunking config (both are kept)

Safe to re-run. Documents already in the table are never rewritten, and
documents already chunked with this configuration are skipped.

Chunk ids are stable across runs and across re-chunking:

    wp:32127@1a2b3c4d/sec600#0007
    |        |        |      ordinal: position within the document
    |        |        chunking configuration (section chunker, 600 chars)
    |        version: first 8 hex digits of sha1 of the article text
    doc_id

so an id can never point at different text, and results computed with one
chunk size are never confused with results for another.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import time

from chunker import DEFAULT_DATA_DIR, MAX_CHARS, chunk_document, load_documents
from db import connect


def doc_version(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def config_name(max_chars: int) -> str:
    return f"sec{max_chars}"


def make_chunk_id(doc_id: str, version: str, config: str, ordinal: int) -> str:
    return f"{doc_id}@{version}/{config}#{ordinal:04d}"


def count_tokens(text: str) -> int:
    return len(text.split())


def ingest_documents(con, data_dir: str) -> int:
    """Insert documents not already in the table. Returns how many were added."""
    stored = dict(con.execute("SELECT doc_id, version FROM documents").fetchall())
    now = datetime.datetime.now()
    rows = []
    for d in load_documents(data_dir):
        version = doc_version(d["text"])
        if d["doc_id"] in stored:
            if stored[d["doc_id"]] != version:
                print(f"  note: {d['title']} changed since it was ingested; "
                      "keeping the stored version")
            continue
        rows.append((d["doc_id"], d.get("source", "wikipedia"), d["title"], d.get("url"),
                     version, len(d["text"]), d["text"], now))
    if rows:
        con.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def ingest_chunks(con, max_chars: int) -> int:
    """Chunk every stored document not yet chunked with this configuration.
    Chunks are made from the stored text (not the JSON files), so their ids
    always match the version in the documents table."""
    config = config_name(max_chars)
    done = {r[0] for r in con.execute(
        "SELECT DISTINCT doc_id FROM chunks WHERE config = ?", [config]).fetchall()}
    rows = []
    for doc_id, title, url, version, text in con.execute(
            "SELECT doc_id, title, url, version, text FROM documents ORDER BY doc_id").fetchall():
        if doc_id in done:
            continue
        doc = {"doc_id": doc_id, "title": title, "url": url, "text": text}
        for ordinal, c in enumerate(chunk_document(doc, max_chars)):
            rows.append((make_chunk_id(doc_id, version, config, ordinal), doc_id, config,
                         ordinal, c["section"], c["subsection"], c["heading"], c["text"],
                         count_tokens(c["text"])))
    if rows:
        con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--max-chars", type=int, default=MAX_CHARS)
    args = p.parse_args()

    con = connect()
    t0 = time.time()
    n_docs = ingest_documents(con, args.data_dir)
    n_chunks = ingest_chunks(con, args.max_chars)
    config = config_name(args.max_chars)
    total_docs, = con.execute("SELECT count(*) FROM documents").fetchone()
    total, avg, longest = con.execute(
        "SELECT count(*), avg(length(text)), max(length(text)) FROM chunks WHERE config = ?",
        [config]).fetchone()
    print(f"added {n_docs} documents and {n_chunks} chunks in {time.time() - t0:.1f}s")
    print(f"documents: {total_docs}   chunks ({config}): {total} "
          f"(avg {avg or 0:.0f} chars, longest {longest or 0})")


if __name__ == "__main__":
    main()
