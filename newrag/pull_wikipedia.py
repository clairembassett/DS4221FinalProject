"""Download Wikipedia articles about US colleges, one JSON file per college.

Each college is saved as its own file in DS4221/project/documents/, named after
the article title (e.g. University_of_Virginia.json). Each file holds one row
for the `documents` table (plus the article text):
  doc_id, source, title, url, college_id, college_name, revision_id,
  revision_timestamp, license, ingested_at, requested_title, text

Usage:
  # Specific colleges: one Wikipedia title or URL per line
  python pull_wikipedia.py --titles colleges.txt

  # Many colleges: get the list from Wikidata, take a fixed random sample
  python pull_wikipedia.py --sparql --limit 600
pytho
  # Save somewhere else
  python pull_wikipedia.py --sparql --limit 600 --out-dir some/other/folder

Re-running is safe: colleges already saved in the folder are skipped, so an
interrupted download picks up where it stopped.
"""
import argparse
import datetime
import glob
import json
import os
import random
import re
import time
from urllib.parse import unquote

import requests

# Wikipedia asks every client to identify itself with a contact address.
USER_AGENT = "UVA-DS-RAG-class-project (qxm6fm@virginia.edu)"
API = "https://en.wikipedia.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"
DEFAULT_OUT_DIR = os.path.join("DS4221", "project", "documents")

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT

# US universities (and subclasses of university) that have an English Wikipedia article.
COLLEGE_QUERY = """
SELECT DISTINCT ?title WHERE {
  ?college wdt:P31/wdt:P279* wd:Q3918 ;
           wdt:P17 wd:Q30 .
  ?article schema:about ?college ;
           schema:isPartOf <https://en.wikipedia.org/> ;
           schema:name ?title .
}
"""


def titles_from_file(path):
    titles = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("http"):                     # accept full URLs too
                line = unquote(line.split("/wiki/", 1)[1])
            titles.append(line.replace("_", " "))
    return titles


def titles_from_sparql(limit, seed):
    r = session.get(SPARQL, params={"query": COLLEGE_QUERY, "format": "json"}, timeout=180)
    r.raise_for_status()
    titles = sorted({b["title"]["value"] for b in r.json()["results"]["bindings"]})
    print(f"Wikidata returned {len(titles)} college articles")
    if limit and limit < len(titles):
        titles = sorted(random.Random(seed).sample(titles, limit))  # reproducible sample
    return titles


def get_json(params, tries=3):
    for attempt in range(tries):
        try:
            r = session.get(API, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == tries - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  request failed ({e}); retrying in {wait}s")
            time.sleep(wait)


def fetch_article(title):
    data = get_json({
        "action": "query",
        "titles": title,
        "redirects": 1,                       # follow redirects to the real article
        "prop": "extracts|revisions|info|pageprops",
        "explaintext": 1,                     # plain text, no markup
        "exsectionformat": "wiki",            # keep "== History ==" section headings
        "rvprop": "ids|timestamp",
        "inprop": "url",
        "ppprop": "wikibase_item",            # the Wikidata id, used as college_id
        "format": "json",
        "formatversion": 2,
    })
    page = data["query"]["pages"][0]
    if page.get("missing") or "extract" not in page:
        return None
    rev = page["revisions"][0]
    return {
        "doc_id": f"wp:{page['pageid']}",
        "source": "wikipedia",
        "title": page["title"],
        "url": page["fullurl"],
        "college_id": page.get("pageprops", {}).get("wikibase_item"),
        "college_name": page["title"],
        "revision_id": rev["revid"],
        "revision_timestamp": rev["timestamp"],
        "license": "CC BY-SA 4.0",
        "ingested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "requested_title": title,
        "text": page["extract"],
    }


def safe_filename(title):
    """Turn an article title into a safe file name, e.g. 'Texas A&M University' -> 'Texas_A&M_University'."""
    name = re.sub(r'[\\/:*?"<>|]+', "-", title)    # characters not allowed in file names
    name = re.sub(r"\s+", "_", name.strip())
    return name[:150] or "untitled"


def load_existing(out_dir):
    """Return (requested titles already saved, doc_ids already saved) from the output folder."""
    requested, doc_ids = set(), set()
    for path in glob.glob(os.path.join(out_dir, "*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"  warning: could not read {path}; it will be re-downloaded if requested")
            continue
        requested.add(doc.get("requested_title"))
        doc_ids.add(doc.get("doc_id"))
    return requested, doc_ids


def save_doc(doc, out_dir):
    """Write one college to its own JSON file and return the path."""
    path = os.path.join(out_dir, safe_filename(doc["title"]) + ".json")
    if os.path.exists(path):
        # Two different titles cleaned up to the same file name: keep both by adding the page id.
        path = os.path.join(out_dir, f"{safe_filename(doc['title'])}_{doc['doc_id'].split(':')[1]}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)          # only a complete file ever appears under the final name
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--titles", help="file with one Wikipedia title or URL per line")
    p.add_argument("--sparql", action="store_true", help="get the college list from Wikidata")
    p.add_argument("--limit", type=int, default=0, help="sample this many colleges (with --sparql)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-chars", type=int, default=0, help="skip stub articles shorter than this")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help=f"folder for the per-college JSON files (default: {DEFAULT_OUT_DIR})")
    args = p.parse_args()

    if args.titles:
        titles = titles_from_file(args.titles)
    elif args.sparql:
        titles = titles_from_sparql(args.limit, args.seed)
    else:
        p.error("give --titles FILE or --sparql")

    os.makedirs(args.out_dir, exist_ok=True)
    done, seen_doc_ids = load_existing(args.out_dir)
    todo = [t for t in titles if t not in done]
    print(f"{len(titles)} titles, {len(done)} already saved in {args.out_dir}, {len(todo)} to fetch")

    for i, title in enumerate(todo, 1):
        doc = fetch_article(title)
        if doc is None:
            print(f"[{i}/{len(todo)}] MISSING  {title}")
        elif len(doc["text"]) < args.min_chars:
            print(f"[{i}/{len(todo)}] STUB     {title} ({len(doc['text'])} chars)")
        elif doc["doc_id"] in seen_doc_ids:
            print(f"[{i}/{len(todo)}] DUPLICATE {title} -> {doc['title']}")
        else:
            seen_doc_ids.add(doc["doc_id"])
            path = save_doc(doc, args.out_dir)
            n_sections = doc["text"].count("\n== ")
            print(f"[{i}/{len(todo)}] ok       {doc['title']} -> {os.path.basename(path)} "
                  f"(rev {doc['revision_id']}, {len(doc['text'])} chars, {n_sections} sections)")
        time.sleep(0.5)   # be polite: one request at a time, small pause


if __name__ == "__main__":
    main()