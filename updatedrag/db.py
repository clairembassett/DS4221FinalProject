"""Open the project's DuckDB database, creating the tables in schema.sql if needed.

The database file is college_rag.duckdb in the repository root. Set the
RAG_DB environment variable to use a different file.

DuckDB allows one writing process at a time: if a command says the database
is locked, another ask.py / evaluate.py is still running.
"""

import os

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("RAG_DB", os.path.join(HERE, "..", "college_rag.duckdb"))
SCHEMA_PATH = os.path.join(HERE, "schema.sql")


def connect(path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(path)
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        con.execute(f.read())
    return con
