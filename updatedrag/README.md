# DS4221 Final Project — College RAG

A retrieval-augmented generation (RAG) pipeline that answers questions about US colleges from their Wikipedia articles. Documents, chunks, embeddings, evaluation questions, the vector index and every evaluation run are stored in a DuckDB database. Retrieval uses the IVF index from Lab 9, measured against brute force.

| Stage | File | Reads → writes |
|---|---|---|
| 1 Ingest | `pull_wikipedia.py`, `rag/ingest.py` | Wikipedia → `data/*.json` → `documents`, `chunks` |
| 2 Embed | `rag/embed.py` | `chunks` → `embeddings` (all-MiniLM-L6-v2) |
| 3 Index | `rag/index.py` | `embeddings` → Lab 9 `IVFIndex`, saved in `index_builds`, `ivf_centroids`, `ivf_assignments` |
| 4 Retrieve | `rag/rag.py` | question → top-k chunk ids → `chunks` JOIN `documents` |
| 5 Generate | `rag/rag.py`, `rag/ask.py` | question + chunks → LLM → answer citing `[chunk_id]` |
| 6 Evaluate | `rag/evaluate.py`, `rag/bench.py` | `eval_questions` → `eval_results`, `bench_results` |

## Repository layout

| Path | What it is |
|---|---|
| `colleges.txt` | The colleges to download, one Wikipedia title or URL per line |
| `pull_wikipedia.py` | Downloads each article into its own JSON file |
| `data/` | The downloaded college JSON files (108 colleges) |
| `rag/schema.sql` | The table design (read this first) |
| `rag/db.py` | Opens `college_rag.duckdb` and creates the tables |
| `rag/ingest.py` | JSON files → `documents` and `chunks` |
| `rag/chunker.py` | Splits an article into section chunks of at most 600 characters |
| `rag/embed.py` | Chunks → vectors in `embeddings` |
| `rag/index.py` | Builds, saves and loads the IVF index |
| `rag/microvector.py` | Lab 9's `BruteForceIndex` / `IVFIndex`, unchanged |
| `rag/rag.py` | Embedders, retrievers, prompt, LLM call |
| `rag/keyword_search.py` | BM25 keyword retrieval (baseline) |
| `rag/ask.py` | Command-line and interactive question answering |
| `rag/milestone.py` | Example queries for the retrieval milestone → `milestone_examples.md` |
| `rag/eval_questions.csv` | The evaluation questions, written by the team |
| `rag/evaluate.py` | Retrieval quality: recall@k, MRR |
| `rag/bench.py` | Index build time, queries per second, recall vs. speed |
| `lab10/` | The original lab code, unchanged |

## Tables

Full definitions, with comments, are in `rag/schema.sql`.

| Table | Key | Purpose |
|---|---|---|
| `documents` | `doc_id` | One row per article: title, URL, version (hash of the text), full text, `ingested_at`. Written once, never rewritten. |
| `chunks` | `chunk_id`; `doc_id` → `documents` | One row per chunk: `ordinal`, section path, text, `n_tokens`, chunking `config`. |
| `embeddings` | (`chunk_id`, `model`); `chunk_id` → `chunks` | One vector per chunk per model, so two models can be compared on the same chunks. |
| `eval_questions` | `q_id` | Question, `relevant_doc_ids`, `author`, `source_team` (own / partner), optional `answer_pattern`. |
| `index_builds`, `ivf_centroids`, `ivf_assignments` | `index_name` | The saved IVF index and how long each build took. |
| `eval_results`, `bench_results` | `run_id` | Every evaluation and benchmark run, per question / per setting. |

**Chunk ids are stable.** `wp:32127@1a2b3c4d/sec600#0007` = document `wp:32127`, text version `1a2b3c4d` (sha1 of the article), chunking config `sec600` (section chunker, 600 characters), ordinal 7. Re-chunking with another size adds rows with different ids rather than replacing these, so earlier results still point at the text they were computed on.

**Evaluation labels are document-level** (`relevant_doc_ids`), as the assignment specifies, so re-chunking never invalidates a label. Because every question here names its college, document-level recall is easy to max out. Each question can therefore also carry an `answer_pattern` (a regex the answering passage contains, e.g. `1890|1857`), and `evaluate.py` reports a stricter passage-level score from it.

## Setup

Python 3.9 or newer.

```bash
pip install -r requirements.txt     # duckdb, sentence-transformers, requests
```

## Running the pipeline

Run all `rag/` commands from inside the `rag/` folder:

```bash
cd rag
```

### 1. Download and ingest the articles

```bash
python ../pull_wikipedia.py --titles ../colleges.txt --out-dir ../data   # skip if data/ is filled
python3 ingest.py
```
```
added 108 documents and 11575 chunks in 13.2s
documents: 108   chunks (sec600): 11575 (avg 457 chars, longest 600)
```
This creates `college_rag.duckdb` in the repository root. Re-running is safe: nothing already stored is rewritten. Before downloading, put your contact email in `USER_AGENT` at the top of `pull_wikipedia.py` (Wikipedia asks every client to identify itself).

### 2. Embed the chunks

```bash
python3 embed.py
```
Downloads all-MiniLM-L6-v2 (~90 MB) the first time, then embeds every chunk: a few minutes on a laptop CPU. Interrupted runs pick up where they stopped.

### 3. Build the index

```bash
python3 index.py
```
Runs Lab 9's k-means to split the vectors into about √n = 108 lists and saves the result in DuckDB (about 1–2 minutes). Later commands load the saved index in under a second.

### 4. Ask questions

```bash
python3 ask.py "When was Harvard founded?"
python3 ask.py --retrieve-only -k 5 "Who founded MIT?"   # chunks only, no LLM
python3 ask.py                                            # interactive mode (use this for the demo)
```
`--retriever brute` uses the exact baseline and `--retriever bm25` keyword search. `--probe N` sets how many IVF lists are searched.

### 5. Retrieval milestone examples

```bash
python3 milestone.py        # writes ../milestone_examples.md
```
Edit `QUERIES` at the top of `milestone.py` to choose the five examples.

### 6. Evaluate

```bash
python3 evaluate.py                     # IVF over MiniLM (the product)
python3 evaluate.py --retriever brute   # exact baseline on the same vectors
python3 evaluate.py --retriever bm25    # keyword baseline
python3 evaluate.py --team partner      # the partner team's questions only
python3 bench.py --build 32 64 216      # build time, QPS, recall vs. speed
```
`evaluate.py` reports, averaged over questions:

- **recall@k:** share of the relevant documents found in the top k chunks
- **MRR@10:** 1 / rank of the first chunk from a relevant document
- **passage@3** and **passage MRR:** the same, but the chunk must also match `answer_pattern`

It lists every question whose answer is not in the top 3 and saves all per-question results to `eval_results`.

`bench.py` compares each IVF setting with brute force on the same queries: recall@10 against brute force's exact top 10, searches per second, and dot products per query.

**Adding questions:** add rows to `eval_questions.csv`. Each question must be written by a team member who read the source article, named in `author`. Partner-team questions use `source_team` = `partner`. `relevant_doc_ids` are doc ids from the `documents` table, separated by `;`. Find one with:

```bash
python3 -c "import db; print(db.connect().execute(\"SELECT doc_id, title FROM documents WHERE title ILIKE '%virginia%'\").fetchall())"
```

### Configurations to compare

| Comparison | How |
|---|---|
| IVF vs. brute force | `evaluate.py` and `evaluate.py --retriever brute`; `bench.py` |
| IVF probe (recall vs. speed) | `bench.py`; `evaluate.py --probe 2` |
| Embedding model | `embed.py --model multi-qa`, `index.py --model multi-qa`, `evaluate.py --model multi-qa` |
| Chunk size | `ingest.py --max-chars 400`, then `embed.py`, `index.py` and `evaluate.py` with `--config sec400` |
| Vector vs. keyword | `evaluate.py --retriever bm25` |

## Generating answers with an LLM

By default `ask.py` calls Gemma (`gemma-4-26b-a4b-it`) through the Gemini API. Put the API key in `rag/secrets_local.py`:

```python
LLM_API_KEY = "your-key"
```
That file is in `.gitignore`; never commit it. Without a key, `ask.py` runs in echo mode and prints the prompt it would send.

To use the course-provided endpoint (OpenAI-compatible) instead:
```bash
export LLM_PROVIDER=openai LLM_BASE_URL="https://.../v1" LLM_MODEL="model-name"
```

The model is told to answer only from the retrieved chunks and to cite them as `[chunk_id]`. `ask.py` prints each cited chunk's college, section and Wikipedia URL.

## How chunking works

Article text keeps Wikipedia's section headings:

```
== History ==
=== Early years ===
In 1890, the American Baptist Education Society ...
```

`chunker.py` follows these rules:

- **Splits at every heading.** `==` starts a section and `===` a subsection. Deeper `====` headings are kept in the path (for example `Academics > Undergraduate schools > Oxford College`). A chunk never crosses a heading.
- **Labels each chunk with the college and section path.** The first line of every chunk is a heading such as `University of Chicago — History > Early years`. Without it, a chunk that says "In 1890, the society incorporated a new university…" never mentions which college it is about.
- **Caps every chunk at 600 characters, heading line included.** A section that fits stays whole. A longer section is split at line breaks, then at sentence ends, then at spaces, and the small pieces are packed back together up to the limit.
- **Skips reference-list sections** (References, External links, Notes, See also, Further reading, …). These hold citations, not facts. The full list is `SKIP_SECTIONS` in `chunker.py`.
- **Names text before the first heading** `Introduction`.

## Troubleshooting

- **`No embeddings for model …`** or **`No IVF index …`**: run `embed.py` and then `index.py` (with the same `--model` and `--config`).
- **`… was built on N vectors but there are now M`**: chunks or embeddings changed since the index was built. Re-run `index.py`.
- **Database is locked**: another `ask.py` or `evaluate.py` is still running. DuckDB allows one writer at a time.
- **Right college, wrong section**: try `-k 5`, a higher `--probe`, or compare with `--retriever bm25`.

## AI assistance

Parts of this code were written with help from Claude (Anthropic), as allowed by the course with attribution. *(Edit this note to describe what was AI-assisted.)* The evaluation questions are written by team members.

Wikipedia text is licensed CC BY-SA 4.0.
