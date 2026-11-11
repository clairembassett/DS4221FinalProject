# DS4221 Final Project — College RAG

A retrieval-augmented generation (RAG) pipeline that answers questions about
US colleges from their Wikipedia articles. It is the Lab 10 `micro_rag`
pipeline, adapted to our own data and chunked by article section.

```
pull_wikipedia.py   Wikipedia  ->  data/<College>.json        (one file per college)
rag/chunker.py      data/*.json ->  section chunks, <= 600 characters each
rag/rag.py          chunks -> embeddings -> BruteForceIndex -> top-k -> prompt -> LLM
rag/ask.py          ask a question from the command line
rag/evaluate.py     score retrieval on labeled questions (hit@3, MRR)
```

## Repository layout

| Path | What it is |
|---|---|
| `colleges.txt` | The colleges to download, one Wikipedia title or URL per line |
| `pull_wikipedia.py` | Downloads each article into its own JSON file |
| `data/` | The downloaded college JSON files (108 colleges) |
| `rag/chunker.py` | Splits articles into section chunks of at most 600 characters |
| `rag/rag.py` | Embedders, retriever, prompt builder and LLM call |
| `rag/microvector.py` | Lab 9's `BruteForceIndex` / `IVFIndex`, unchanged |
| `rag/ask.py` | Command-line question answering |
| `rag/evaluate.py` | Labeled eval questions and retrieval scores |
| `lab10/` | The original lab code, kept unchanged so its tests still pass |

## Requirements

- Python 3.9 or newer
- `requests`, only for downloading data: `pip install requests`
- Optional: `sentence-transformers`, for the learned-embedding option (step 6)

The pipeline itself (steps 2–5) uses only the Python standard library.

## Running the pipeline

All `rag/` commands are run from inside the `rag/` folder:

```bash
cd rag
```

### 1. Download the college articles (skip if `data/` is already filled)

From the repository root:

```bash
python pull_wikipedia.py --titles colleges.txt --out-dir data
```

This writes one JSON file per college to `data/`. Re-running it is safe:
colleges already saved are skipped. Before running it, put your own contact
email in `USER_AGENT` at the top of the script, since Wikipedia asks every
client to identify itself.

### 2. Chunk the articles (optional: lets you inspect the chunks)

```bash
python3 chunker.py
```

```
108 colleges -> 11575 chunks (avg 457 chars, longest 600, limit 600)
wrote chunks.jsonl
```

`chunks.jsonl` holds one chunk per line, ready to load into a `chunks` table.
You don't need this file for steps 3–5, which chunk the data in memory each
run (it takes well under a second).

### 3. Ask a question

```bash
python3 ask.py "When was Harvard founded?"
python3 ask.py -k 5 "What are Notre Dame's athletic teams known as?"
```

With no LLM configured, `ask.py` runs in **echo mode**: it prints the prompt
it would send, which shows exactly which chunks retrieval chose. Building the
index takes about 3 seconds; each question then takes about 1 second.

### 4. Evaluate retrieval

```bash
python3 evaluate.py                    # default tfidf embedder
python3 evaluate.py --embedder hashed  # the lab's original embedder, for comparison
python3 evaluate.py -q                 # scores only
```

Each question in `EVAL` (in `evaluate.py`) has two labels: the college it is
about, and a regular expression the answer chunk must contain. The script
reports three scores:

- **hit@3**: the answer chunk is in the top 3
- **MRR@10**: the average of 1 / rank of the first answer chunk
- **college@3**: some chunk from the right college is in the top 3

Current results on the 15 questions:

| embedder | hit@3 | MRR@10 | college@3 |
|---|---|---|---|
| `hashed` (lab 10 original) | 0.00 | 0.02 | 0.73 |
| `tfidf` (default) | 0.40 | 0.33 | 0.93 |
| `minilm` | run it yourself (step 6) | | |

Add more questions to `EVAL` as you go. Around 30 is a reasonable minimum
before trusting the comparison.

### 5. Generate real answers with an LLM

Set the same environment variables the lab uses, for any OpenAI-compatible
endpoint:

```bash
export LLM_BASE_URL="https://your-endpoint/v1"
export LLM_API_KEY="your-key"
export LLM_MODEL="model-name"
python3 ask.py "Who founded MIT?"
```

The model is told to answer only from the retrieved chunks and to cite them
as `[chunk_id]`.

### 6. Optional: use a learned embedding model

```bash
pip install sentence-transformers
python3 evaluate.py --embedder minilm
python3 ask.py --embedder minilm "Who donated the money to start the University of Chicago?"
```

This uses `all-MiniLM-L6-v2`, the model `lab10/measure_real.py` compares
against. It downloads about 90 MB the first time, and encoding all chunks
takes a few minutes on a laptop CPU. It handles paraphrases that bag-of-words
misses, such as "founded" vs. "incorporated".

## How chunking works

Article text keeps Wikipedia's section headings:

```
== History ==
=== Early years ===
In 1890, the American Baptist Education Society ...
```

`chunker.py` follows these rules:

1. **Splits at every heading.** `==` starts a section and `===` a
   subsection. Deeper `====` headings are kept in the path (for example
   `Academics > Undergraduate schools > Oxford College`). A chunk never
   crosses a heading.
2. **Labels each chunk with the college and section path.** The first line
   of every chunk is a heading such as
   `University of Chicago — History > Early years`. Without it, a chunk that
   says "In 1890, the society incorporated a new university…" never mentions
   which college it is about.
3. **Caps every chunk at 600 characters, heading line included.** A section
   that fits stays whole. A longer section is split at line breaks, then at
   sentence ends, then at spaces, and the small pieces are packed back
   together up to the limit.
4. **Skips reference-list sections** (References, External links, Notes,
   See also, Further reading, …). These hold citations, not facts. The full
   list is `SKIP_SECTIONS` in `chunker.py`.
5. **Names text before the first heading `Introduction`.**

You can change the limit with `python3 chunker.py --max-chars 800`, or with
`chunk_corpus(max_chars=800)` in code.

Each chunk is a dict with these fields:

| Field | Example |
|---|---|
| `chunk_id` | `wp:32127#0007` (document id + position) |
| `doc` | `wp:32127`, the `doc_id` from the documents table |
| `title` | `University of Chicago` |
| `url` | `https://en.wikipedia.org/wiki/University_of_Chicago` |
| `section` | `History` |
| `subsection` | `Early years` (or `null`) |
| `heading` | `University of Chicago — History > Early years` |
| `part`, `parts` | `1`, `2`: this chunk's position when a section was split |
| `text` | heading line + newline + chunk body (≤ 600 characters) |

## What changed from Lab 10's `micro_rag.py`

- **Corpus:** `chunk_corpus()` reads `data/*.json` instead of the 24 course
  documents in `corpus.py`, and chunks by section with a 600-character cap
  instead of by 60 words.
- **Embedder:** the default is now `tfidf`. This is the lab's hashed
  bag-of-words with two changes: IDF weighting, so words that appear in
  nearly every chunk ("university", "students", "campus") count for less,
  and digits kept as words, so years like 1890 can match. The original
  `embed()` is still available as `--embedder hashed`.
- **Prompt:** sources are cited by `chunk_id`, and each source line begins
  with the college and section path.
- **Unchanged:** `Retriever`, `BruteForceIndex`, `answer()` and the LLM
  environment variables all work the same way as in the lab.

## Troubleshooting

- **`No .json files found`:** run from inside `rag/`, or pass
  `--data-dir path/to/folder` to `ask.py` or `chunker.py`. `chunker.py`
  also accepts a single `.jsonl` file (`--data-dir ../documents.jsonl`).
- **Retrieval finds the right college but the wrong section:** try `-k 5`,
  or the `minilm` embedder. With bag-of-words, a question has to share
  words with the chunk that answers it.

Wikipedia text is licensed CC BY-SA 4.0.
