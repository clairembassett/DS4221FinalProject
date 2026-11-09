"""Lab 10 — the corpus: this course, as documents. Provided complete.

24 short documents summarizing DS 6XXX topics — so you can judge retrieval
quality yourself (you took the class the corpus describes). Each doc has an
id, a title, and a few sentences.

EVAL holds 12 questions with the doc ids a correct retrieval should
surface. Your retriever is scored on hit@3 and MRR against them.
"""

DOCS = [
    ("blocks", "Blocks and pages",
     "Disks move data in fixed-size blocks because a storage trip costs the same "
     "regardless of size. microdb uses 4 KB blocks; a page is the in-memory copy of a "
     "block. Block number k of a file begins at byte k times the block size. "
     "Reading whole blocks makes neighboring data nearly free to fetch."),
    ("fsync", "Durability and fsync",
     "A normal write only reaches the operating system's cache; fsync forces bytes to "
     "durable storage and costs ten to a thousand times more. Databases ration fsync "
     "carefully. Lab 1 measured buffered versus durable writes per second, and the "
     "ratio motivates the entire design of transaction logging."),
    ("bufferpool", "The buffer pool",
     "The buffer pool is the database's own cache: fixed frames each holding one "
     "block, with a pin and unpin protocol so pages in use are never evicted. LRU "
     "chooses eviction victims by recency. Databases build their own pool rather "
     "than trusting the OS cache because they need pinning and control over write "
     "ordering for recovery."),
    ("cliff", "The sequential flooding cliff",
     "A sequential scan through an LRU pool one frame smaller than the file achieves "
     "a zero percent hit rate: each block is evicted just before it is needed again. "
     "Real engines defend with ring buffers or midpoint insertion. Hot-set workloads, "
     "whose working set fits the pool, hit above ninety percent."),
    ("slots", "Record slots and layout",
     "Rows live in fixed-size slots: a four-byte in-use flag then fields at offsets "
     "computed from the schema. Fixed slots make addressing arithmetic, updates "
     "in-place, and deletes a flag flip. The waste inside reserved space is internal "
     "fragmentation, the deliberate rent paid for O(1) access."),
    ("tombstone", "Tombstones and deletion",
     'In microdb, deleting a record marks its slot empty; a later insert can reuse the slot. The old bytes may remain until overwritten. Production engines use different deletion schemes, including MVCC versions or tombstones, and may need vacuum or compaction to reclaim space.'),
    ("rid", "Record identifiers",
     "A RID is a row's physical address: block number and slot number. In microdb, the address stays stable while the row exists. An index maps field values to RIDs and must remove stale entries when a row is deleted or its slot is reused."),
    ("catalog", "The system catalog",
     "The catalog stores every table's schema and layout in ordinary tables like "
     "field_catalog. The catalog's own layout is hardcoded at startup to break the "
     "circular dependency, a move called bootstrapping. In Postgres the psql "
     "backslash-d command is just a catalog query."),
    ("iterator", "The iterator model",
     'Every query operator implements before_first, next, get_val, has_field, and close. Pipelined scans pass rows on demand with little buffering. Sorts, hash joins, and some aggregations retain data, so the iterator interface alone does not guarantee constant memory.'),
    ("product", "Products and joins",
     "A join is a cartesian product filtered by a predicate. The nested-loop "
     "product rewinds its right input for every left row, so pairing costs the "
     "product of the table sizes. Hash joins and merge joins exist to avoid "
     "building pairs that will be discarded."),
    ("pushdown", "Predicate pushdown",
     "Filtering each input before a join instead of filtering joined pairs "
     "afterward can cut work by orders of magnitude with an identical answer, "
     "guaranteed by relational algebra. The same idea reaches into Parquet files "
     "as chunk skipping and across clusters in distributed engines."),
    ("parsing", "Parsing SQL",
     "A lexer turns characters into tokens; a recursive-descent parser turns tokens "
     "into a description, one method per grammar rule using peek, next, match, and "
     "expect. Parsing to plain data lets the planner be a separate, swappable "
     "stage, and expect() produces helpful errors by construction."),
    ("planner", "Planning and naivety",
     "The naive planner builds table scans, folds products left to right, then adds "
     "one select and one project. It is always correct and often slow, because it "
     "never pushes filters down; doing that safely requires knowing which fields "
     "each term references and how many rows will survive."),
    ("btree", "B+ trees",
     "A B+ tree keeps sorted keys in leaves linked left to right, with internal "
     "nodes holding routing keys. A full node splits at its middle key, which is "
     "copied up from a leaf or moved up from an internal node; the root splitting "
     "grows the tree and keeps every leaf at the same depth. Fan-out around two "
     "hundred makes a hundred-million-row tree four levels tall."),
    ("selectivity", "Selectivity and index choice",
     "Selectivity is the fraction of rows a predicate keeps. Indexes shine on "
     "selective predicates like an equality on a unique id and lose to a plain "
     "scan on unselective ones, where random jumps cost more than one smooth read. "
     "Optimizers estimate selectivity from statistics to choose."),
    ("wal", "Write-ahead logging",
     "In this lab's undo log, the old value must reach durable storage before the changed data page can. Commit flushes data pages before syncing the commit record. This FORCE/STEAL design needs earlier log and page syncs as well as the final commit sync."),
    ("recovery", "Crash recovery",
     "Recovery reads the log newest first, so each transaction's fate is known "
     "before its writes are encountered. Writes of unfinished transactions are "
     "undone by restoring old values. Recovery must be idempotent because a crash "
     "can interrupt recovery itself; rollback receipts make reruns skip cleanly."),
    ("locks", "Two-phase locking",
     "Shared locks allow many readers; exclusive locks allow one writer and no "
     "readers. Strict two-phase locking holds all locks until commit, which "
     "guarantees serializability and prevents dirty reads, at the price of "
     "waiting and deadlocks, which engines break by aborting a victim."),
    ("mvcc", "Multi-version concurrency control",
     'Updates create new row versions. A snapshot determines which committed versions a reader can see: per statement at READ COMMITTED, or per transaction at REPEATABLE READ in PostgreSQL. Ordinary snapshot reads do not block row writers, but explicit locks can conflict. Vacuum reclaims versions no snapshot needs.'),
    ("columnar", "Columnar storage",
     "Analytics touches all rows but few columns, so column stores keep each "
     "column's values contiguous: queries read only named columns, and runs of "
     "same-typed values compress with run-length and dictionary encoding. Row "
     "stores win point lookups; column stores win scans."),
    ("parquet", "Parquet and partitioning",
     "Parquet stores columns in chunks with min-max statistics so engines skip "
     "data without reading it. Hive partitioning puts the partition column in "
     "folder names like month equals twelve, letting a filter skip whole "
     "directories. DuckDB reports scanning one of twelve files."),
    ("vectors", "Vector search and IVF",
     "Embeddings turn similarity into geometry: nearest neighbors by cosine. "
     "Exact search compares against everything; IVF clusters vectors with k-means "
     "and probes only the lists nearest the query. The probe parameter trades "
     "recall against comparisons, and recall at k measures what fraction of true "
     "neighbors were found."),
    ("warehouse", "Cloud warehouses",
     "Cloud warehouses separate storage from compute: data rests in object "
     "storage while query clusters spin up on demand and bill by bytes scanned. "
     "Naming only needed columns and partitioning on filtered keys directly "
     "reduces cost, because the columnar lessons become line items."),
    ("lakehouse", "The lakehouse",
     "A lakehouse adds a transactional metadata layer such as Delta or Iceberg "
     "over Parquet files in object storage: a manifest records which files form "
     "the table, enabling ACID commits, schema evolution, and time travel. "
     "Structurally it is a catalog plus a log conferring table-hood on a lake."),
]

# question, {doc ids that count as a correct retrieval}
EVAL = [
    ("Why do disks move data in whole blocks rather than single values?", {"blocks"}),
    ("What forces written bytes all the way to durable storage?", {"fsync", "wal"}),
    ("Why can't the OS page cache replace the database's own cache?", {"bufferpool"}),
    ("Why does a scan get zero cache hits with a nearly big enough pool?", {"cliff"}),
    ("Where inside a record's slot does each field live?", {"slots", "catalog"}),
    ("What happens to a row's bytes when it is deleted?", {"tombstone"}),
    ("How does an index refer to the rows it points at?", {"rid", "btree"}),
    ("Why is recovery safe to run twice after a crash?", {"recovery"}),
    ("How can readers avoid blocking writers entirely?", {"mvcc"}),
    ("Why are analytics queries faster on columns than rows?", {"columnar", "parquet"}),
    ("What does the probe setting trade away in vector search?", {"vectors"}),
    ("How do cloud warehouses charge for queries?", {"warehouse"}),
]
