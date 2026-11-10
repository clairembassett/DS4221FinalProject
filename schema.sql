CREATE TABLE IF NOT EXISTS documents (
    doc_id VARCHAR PRIMARY KEY,
    source VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    url VARCHAR NOT NULL,
    ingested_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR PRIMARY KEY,
    doc_id VARCHAR NOT NULL REFERENCES documents(doc_id),
    ordinal INTEGER NOT NULL,
    section_path VARCHAR NOT NULL,
    text VARCHAR NOT NULL,
    n_tokens INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id VARCHAR REFERENCES chunks(chunk_id),
    model VARCHAR,
    dim INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, model)
);

CREATE TABLE IF NOT EXISTS eval_questions (
    q_id VARCHAR PRIMARY KEY,
    question VARCHAR NOT NULL,
    relevant_doc_ids VARCHAR[] NOT NULL,
    author VARCHAR NOT NULL
);
