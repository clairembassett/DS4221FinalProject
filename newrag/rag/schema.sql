-- College RAG: table design (DuckDB).
--
-- The four tables the assignment asks for (documents, chunks, embeddings,
-- eval_questions), plus tables that persist the IVF index and record every
-- index build and evaluation run, so each number in the report can be traced
-- back to the exact data it was computed on.
--
-- db.connect() runs this file every time; IF NOT EXISTS makes that safe.

-- One row per Wikipedia article. Written once at ingestion, never rewritten.
CREATE TABLE IF NOT EXISTS documents (
    doc_id      VARCHAR PRIMARY KEY,    -- 'wp:32127' = Wikipedia page id 32127
    source      VARCHAR NOT NULL,       -- 'wikipedia'
    title       VARCHAR NOT NULL,       -- 'University of Chicago'
    url         VARCHAR,
    version     VARCHAR NOT NULL,       -- first 8 hex digits of sha1(text): which text we ingested
    n_chars     INTEGER NOT NULL,
    text        VARCHAR NOT NULL,       -- full article text, so re-chunking never needs the JSON files
    ingested_at TIMESTAMP NOT NULL
);

-- One row per chunk. Several chunking configurations can live side by side
-- (config 'sec600' = section chunker, 600-character limit), so changing the
-- chunk size adds rows instead of overwriting the ones earlier results used.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    VARCHAR PRIMARY KEY,    -- '{doc_id}@{version}/{config}#{ordinal}', e.g. 'wp:32127@1a2b3c4d/sec600#0007'
    doc_id      VARCHAR NOT NULL REFERENCES documents (doc_id),
    config      VARCHAR NOT NULL,       -- chunking configuration
    ordinal     INTEGER NOT NULL,       -- position of the chunk within its document, from 0
    section     VARCHAR NOT NULL,       -- 'History'
    subsection  VARCHAR,                -- 'Early years' (NULL if none)
    heading     VARCHAR NOT NULL,       -- 'University of Chicago — History > Early years'
    text        VARCHAR NOT NULL,       -- heading line + newline + body, at most max_chars characters
    n_tokens    INTEGER NOT NULL,       -- whitespace-separated words in text
    UNIQUE (doc_id, config, ordinal)
);

-- One row per (chunk, embedding model). The model column is what lets two
-- embedders be compared on the very same chunks.
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id    VARCHAR NOT NULL REFERENCES chunks (chunk_id),
    model       VARCHAR NOT NULL,       -- 'sentence-transformers/all-MiniLM-L6-v2'
    dim         INTEGER NOT NULL,       -- 384 for MiniLM
    vector      FLOAT[] NOT NULL,       -- unit length, so dot product = cosine similarity
    PRIMARY KEY (chunk_id, model)
);

-- Evaluation questions, written by team members who read the source article.
-- Labeled at the document level, so re-chunking never invalidates a label.
-- Every id in relevant_doc_ids must be a doc_id in documents (checked by
-- evaluate.py when it loads the questions; DuckDB cannot put a foreign key on
-- a list column).
CREATE TABLE IF NOT EXISTS eval_questions (
    q_id             VARCHAR PRIMARY KEY,   -- 'own-01', 'partner-03'
    question         VARCHAR NOT NULL,
    relevant_doc_ids VARCHAR[] NOT NULL,    -- ['wp:32127']
    author           VARCHAR NOT NULL,      -- team member who read the source and wrote the question
    source_team      VARCHAR NOT NULL,      -- 'own' or 'partner' (the question swap)
    answer_pattern   VARCHAR                -- optional regex the answering passage contains, for passage-level scoring
);

-- ---------------------------------------------------------------- the index
-- Lab 9's IVFIndex, persisted: k-means centroids and the list each chunk was
-- assigned to. ask.py reloads these instead of re-running k-means.
-- index_name is not declared as a foreign key: rebuilding an index replaces
-- its rows, and DuckDB refuses to delete a row that a foreign key points at.

CREATE TABLE IF NOT EXISTS index_builds (
    index_name    VARCHAR PRIMARY KEY,  -- 'ivf/sentence-transformers/all-MiniLM-L6-v2/sec600/L108'
    model         VARCHAR NOT NULL,
    config        VARCHAR NOT NULL,
    n_lists       INTEGER NOT NULL,     -- number of k-means clusters
    n_vectors     INTEGER NOT NULL,
    build_seconds DOUBLE NOT NULL,      -- k-means + assignment time (report: index build time)
    built_at      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS ivf_centroids (
    index_name  VARCHAR NOT NULL,
    list_no     INTEGER NOT NULL,
    centroid    FLOAT[] NOT NULL,
    PRIMARY KEY (index_name, list_no)
);

CREATE TABLE IF NOT EXISTS ivf_assignments (
    index_name  VARCHAR NOT NULL,
    chunk_id    VARCHAR NOT NULL REFERENCES chunks (chunk_id),
    list_no     INTEGER NOT NULL,
    PRIMARY KEY (index_name, chunk_id)
);

-- ---------------------------------------------------------------- results
-- One row per (evaluation run, question): what was retrieved and where the
-- first relevant result landed. The failure analysis is a query on this table.
-- q_id is not a foreign key so that eval_questions can be reloaded from
-- eval_questions.csv after results exist (same DuckDB restriction as above).

CREATE TABLE IF NOT EXISTS eval_results (
    run_id         VARCHAR NOT NULL,     -- '2026-12-01T14:03:22 bm25' etc.
    q_id           VARCHAR NOT NULL,
    retriever      VARCHAR NOT NULL,     -- 'ivf', 'brute' or 'bm25'
    model          VARCHAR,              -- NULL for bm25
    config         VARCHAR NOT NULL,
    probe          INTEGER,              -- IVF lists probed (NULL unless ivf)
    retrieved      VARCHAR[] NOT NULL,   -- chunk_ids, best first (top 10)
    doc_rank       INTEGER,              -- rank of the first chunk from a relevant doc (NULL = not in top 10)
    passage_rank   INTEGER,              -- rank of the first chunk that also matches answer_pattern
    seconds        DOUBLE NOT NULL,      -- retrieval time for this question
    PRIMARY KEY (run_id, q_id)
);

-- One row per (benchmark run, IVF setting): the recall-vs-speed tradeoff.
CREATE TABLE IF NOT EXISTS bench_results (
    run_id        VARCHAR NOT NULL,
    index_name    VARCHAR NOT NULL,     -- 'brute' or an ivf/... name from index_builds
    probe         INTEGER NOT NULL,     -- lists searched; 0 for brute force
    k             INTEGER NOT NULL,
    n_queries     INTEGER NOT NULL,
    recall_at_k   DOUBLE NOT NULL,      -- share of brute force's exact top k that IVF also found
    qps           DOUBLE NOT NULL,      -- index searches per second (query embedding excluded)
    comparisons   DOUBLE NOT NULL,      -- dot products per query
    PRIMARY KEY (run_id, index_name, probe)
);
