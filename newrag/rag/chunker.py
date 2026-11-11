"""Section-aware chunker for the college Wikipedia files.

Reads the per-college JSON files written by pull_wikipedia.py and splits each
article's text into chunks that:

  * never cross a section boundary. Headings in the text look like
        == History ==            section      (level 2)
        === Early years ===      subsection   (level 3)
        ==== Founding ====       deeper levels are kept as part of the path
  * are at most MAX_CHARS (600) characters long, INCLUDING a one-line prefix
        "University of Chicago — History > Early years"
    so every chunk says which college and section it came from (with a
    bag-of-words embedder, the college name in the chunk is what lets
    "When was UChicago founded?" find the right college).

Sections longer than the limit are split at paragraph breaks, then sentence
ends, then spaces; small pieces are packed back together up to the limit.

Run it on its own to write chunks.jsonl (one chunk per line) and print stats:
    python3 chunker.py                       # reads ../data
    python3 chunker.py --data-dir path/to/json/folder --out chunks.jsonl
    python3 chunker.py --data-dir ../documents.jsonl   # one-file format also works
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

MAX_CHARS = 600
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, "..", "data")

# "== Title ==", "=== Title ===", ... on a line of their own.
HEADING_RE = re.compile(r"^\s*(={2,6})\s*(.+?)\s*=+\s*$")
PARAGRAPH_RE = re.compile(r"\n+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
SPACE_RE = re.compile(r"\s+")

# Reference-list sections: citations and link lists, not facts about the college.
SKIP_SECTIONS = {
    "references", "external links", "notes", "see also", "further reading",
    "explanatory notes", "citations", "bibliography", "footnotes", "sources",
    "notes and references", "references and notes", "works cited",
}

INTRO = "Introduction"   # name given to the text before the first heading


# ------------------------------------------------------------- loading

def load_documents(data_dir: str = DEFAULT_DATA_DIR) -> list[dict]:
    """Every *.json file in data_dir, sorted by file name (stable chunk order).
    data_dir may also be a single .jsonl file with one document per line
    (the older pull_wikipedia.py output, documents.jsonl)."""
    if data_dir.endswith(".jsonl") and os.path.isfile(data_dir):
        with open(data_dir, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            docs.append(json.load(f))
    if not docs:
        raise SystemExit(f"No .json files found in {os.path.abspath(data_dir)}")
    return docs


# ------------------------------------------------------------- sections

def split_sections(text: str):
    """Yield (path, body) for each block of text, where path is the list of
    headings above it, e.g. ["History", "Early years"]."""
    path: list[str] = [INTRO]
    lines: list[str] = []

    def flush():
        body = "\n".join(lines).strip()
        if body:
            yield list(path), body

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            yield from flush()
            lines = []
            level = len(m.group(1))               # 2 = section, 3 = subsection, ...
            path = path[: level - 2] + [m.group(2).strip()]
        else:
            lines.append(line)
    yield from flush()


# ------------------------------------------------------------- splitting

def _split_units(text: str, limit: int) -> list[str]:
    """Break text into pieces no longer than limit, preferring natural breaks."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    for splitter in (PARAGRAPH_RE, SENTENCE_RE, SPACE_RE):
        parts = [p.strip() for p in splitter.split(text) if p.strip()]
        if len(parts) > 1:
            out: list[str] = []
            for p in parts:
                out.extend(_split_units(p, limit))
            return out
    # One unbreakable token longer than the limit: hard cut.
    return [text[i:i + limit] for i in range(0, len(text), limit)]


def _pack(units: list[str], limit: int) -> list[str]:
    """Greedily join consecutive units with a space, up to limit characters."""
    packed: list[str] = []
    current = ""
    for u in units:
        candidate = f"{current} {u}" if current else u
        if len(candidate) <= limit:
            current = candidate
        else:
            packed.append(current)
            current = u
    if current:
        packed.append(current)
    return packed


# ------------------------------------------------------------- chunking

def chunk_document(doc: dict, max_chars: int = MAX_CHARS,
                   skip_sections: set[str] = SKIP_SECTIONS) -> list[dict]:
    """Chunks for one college document (a dict loaded from its JSON file)."""
    chunks = []
    for path, body in split_sections(doc["text"]):
        if path[0].lower() in skip_sections:
            continue
        heading = f"{doc['title']} — {' > '.join(path)}"
        budget = max_chars - len(heading) - 1          # -1 for the newline
        if budget < 100:                               # absurdly long heading
            heading = heading[: max_chars - 101].rstrip() + "…"
            budget = max_chars - len(heading) - 1
        pieces = _pack(_split_units(body, budget), budget)
        for i, piece in enumerate(pieces, start=1):
            chunks.append({
                "chunk_id": None,                      # filled in below
                "doc": doc["doc_id"],
                "title": doc["title"],
                "url": doc.get("url"),
                "section": path[0],
                "subsection": path[1] if len(path) > 1 else None,
                "heading": heading,
                "part": i,
                "parts": len(pieces),
                "text": f"{heading}\n{piece}",
            })
    for n, c in enumerate(chunks):
        c["chunk_id"] = f"{doc['doc_id']}#{n:04d}"
    return chunks


def chunk_corpus(data_dir: str = DEFAULT_DATA_DIR,
                 max_chars: int = MAX_CHARS) -> list[dict]:
    """All chunks for every college file in data_dir, in a stable order."""
    chunks = []
    for doc in load_documents(data_dir):
        chunks.extend(chunk_document(doc, max_chars))
    return chunks


# ------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--max-chars", type=int, default=MAX_CHARS)
    p.add_argument("--out", default=os.path.join(HERE, "chunks.jsonl"))
    args = p.parse_args()

    chunks = chunk_corpus(args.data_dir, args.max_chars)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    lengths = [len(c["text"]) for c in chunks]
    assert max(lengths) <= args.max_chars, "a chunk exceeded the limit"
    n_docs = len({c["doc"] for c in chunks})
    print(f"{n_docs} colleges -> {len(chunks)} chunks "
          f"(avg {sum(lengths) / len(lengths):.0f} chars, longest {max(lengths)}, "
          f"limit {args.max_chars})")
    print(f"wrote {os.path.relpath(args.out)}")


if __name__ == "__main__":
    main()
