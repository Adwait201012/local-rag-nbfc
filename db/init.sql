-- Schema for hybrid (dense + lexical) retrieval in a single Postgres instance.
-- bge-m3 emits 1024-dimensional vectors.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    title       TEXT,
    sha256      TEXT NOT NULL,
    n_chunks    INT  NOT NULL DEFAULT 0,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id        BIGSERIAL PRIMARY KEY,
    doc_id    BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord       INT    NOT NULL,
    heading   TEXT,
    page      INT,
    text      TEXT   NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    tsv       TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

-- Dense index. Cosine distance because we store L2-normalised vectors.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Lexical index: catches error codes, SKUs, proper nouns that dense search misses.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_id  ON chunks (doc_id);

-- Reciprocal Rank Fusion over the two candidate lists.
-- k = 60 is the standard constant from the original RRF paper.
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding VECTOR(1024),
    query_text      TEXT,
    match_limit     INT DEFAULT 40,
    rrf_k           INT DEFAULT 60
)
RETURNS TABLE (
    chunk_id BIGINT,
    doc_id   BIGINT,
    source   TEXT,
    heading  TEXT,
    page     INT,
    text     TEXT,
    score    DOUBLE PRECISION
)
LANGUAGE sql STABLE AS $$
WITH sem AS (
    SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.embedding <=> query_embedding) AS rnk
    FROM chunks c
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_limit
),
lex AS (
    SELECT c.id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, q) DESC) AS rnk
    FROM chunks c, websearch_to_tsquery('english', query_text) q
    WHERE c.tsv @@ q
    ORDER BY ts_rank_cd(c.tsv, q) DESC
    LIMIT match_limit
),
fused AS (
    SELECT COALESCE(sem.id, lex.id) AS id,
           COALESCE(1.0 / (rrf_k + sem.rnk), 0.0)
         + COALESCE(1.0 / (rrf_k + lex.rnk), 0.0) AS score
    FROM sem FULL OUTER JOIN lex ON sem.id = lex.id
)
SELECT c.id, c.doc_id, d.source_path, c.heading, c.page, c.text, f.score
FROM fused f
JOIN chunks c    ON c.id = f.id
JOIN documents d ON d.id = c.doc_id
ORDER BY f.score DESC
LIMIT match_limit;
$$;
-- Generated questions: a third retrieval signal.
--
-- Regulations are written in formal RBI wording ("period of public deposit");
-- people ask in industry wording ("deposit tenor"). For every passage, a local
-- model writes a few questions the way a practitioner would phrase them, and
-- those questions are searched alongside the passages themselves.
--
-- They live in their own table rather than being pasted into the passage text,
-- because appended text adds noise to the passage's embedding. A weak generated
-- question can only add a candidate to the pool; the reranker still judges the
-- real passage text against the question the person actually asked.

CREATE TABLE IF NOT EXISTS chunk_questions (
    id        BIGSERIAL PRIMARY KEY,
    chunk_id  BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    question  TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunk_questions_embedding_hnsw
    ON chunk_questions USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunk_questions_chunk_id ON chunk_questions (chunk_id);

-- Same signature as before, so the API needs no change. With the questions
-- table empty, the third list contributes nothing and results are identical
-- to the two-way search.
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding VECTOR(1024),
    query_text      TEXT,
    match_limit     INT DEFAULT 40,
    rrf_k           INT DEFAULT 60
)
RETURNS TABLE (
    chunk_id BIGINT,
    doc_id   BIGINT,
    source   TEXT,
    heading  TEXT,
    page     INT,
    text     TEXT,
    score    DOUBLE PRECISION
)
LANGUAGE sql STABLE AS $$
WITH sem AS (
    SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.embedding <=> query_embedding) AS rnk
    FROM chunks c
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_limit
),
lex AS (
    SELECT c.id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, q) DESC) AS rnk
    FROM chunks c, websearch_to_tsquery('english', query_text) q
    WHERE c.tsv @@ q
    ORDER BY ts_rank_cd(c.tsv, q) DESC
    LIMIT match_limit
),
-- Several questions can point at one passage; a passage ranks by its best one.
q_raw AS (
    SELECT cq.chunk_id, cq.embedding <=> query_embedding AS dist
    FROM chunk_questions cq
    ORDER BY cq.embedding <=> query_embedding
    LIMIT match_limit * 3
),
qsem AS (
    SELECT q_raw.chunk_id AS id,
           ROW_NUMBER() OVER (ORDER BY min(q_raw.dist)) AS rnk
    FROM q_raw
    GROUP BY q_raw.chunk_id
    ORDER BY min(q_raw.dist)
    LIMIT match_limit
),
fused AS (
    SELECT u.id, sum(1.0 / (rrf_k + u.rnk)) AS score
    FROM (
        SELECT id, rnk FROM sem
        UNION ALL SELECT id, rnk FROM lex
        UNION ALL SELECT id, rnk FROM qsem
    ) u
    GROUP BY u.id
)
SELECT c.id, c.doc_id, d.source_path, c.heading, c.page, c.text, f.score::double precision
FROM fused f
JOIN chunks c    ON c.id = f.id
JOIN documents d ON d.id = c.doc_id
ORDER BY f.score DESC
LIMIT match_limit;
$$;

-- Areas: separate bodies of law (RBI, income tax, GST, Companies Act...)
--
-- Every document belongs to one area. Files in the corpus root belong to the
-- default area ('rbi'); files in a subfolder belong to the area named by that
-- folder, e.g. corpus/income_tax/x.pdf -> 'income_tax'.
--
-- Searching within one area keeps a GST question from competing with 45 RBI
-- rulebooks. Searching with no area behaves exactly as before.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS area TEXT NOT NULL DEFAULT 'rbi';
CREATE INDEX IF NOT EXISTS documents_area ON documents (area);

-- The previous four-argument version must go first. Adding a fifth argument
-- creates a second overload rather than replacing the first, and calls with
-- three arguments would then be ambiguous.
DROP FUNCTION IF EXISTS hybrid_search(vector, text, integer, integer);

-- Unfiltered search: identical to the previous hybrid_search, unchanged.
CREATE OR REPLACE FUNCTION _hybrid_all(
    query_embedding VECTOR(1024), query_text TEXT, match_limit INT, rrf_k INT
)
RETURNS TABLE (chunk_id BIGINT, doc_id BIGINT, source TEXT, heading TEXT,
               page INT, text TEXT, score DOUBLE PRECISION)
LANGUAGE sql STABLE AS $$
WITH sem AS (
    SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.embedding <=> query_embedding) AS rnk
    FROM chunks c
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_limit
),
lex AS (
    SELECT c.id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, q) DESC) AS rnk
    FROM chunks c, websearch_to_tsquery('english', query_text) q
    WHERE c.tsv @@ q
    ORDER BY ts_rank_cd(c.tsv, q) DESC
    LIMIT match_limit
),
q_raw AS (
    SELECT cq.chunk_id, cq.embedding <=> query_embedding AS dist
    FROM chunk_questions cq
    ORDER BY cq.embedding <=> query_embedding
    LIMIT match_limit * 3
),
qsem AS (
    SELECT q_raw.chunk_id AS id,
           ROW_NUMBER() OVER (ORDER BY min(q_raw.dist)) AS rnk
    FROM q_raw
    GROUP BY q_raw.chunk_id
    ORDER BY min(q_raw.dist)
    LIMIT match_limit
),
fused AS (
    SELECT u.id, sum(1.0 / (rrf_k + u.rnk)) AS score
    FROM (
        SELECT id, rnk FROM sem
        UNION ALL SELECT id, rnk FROM lex
        UNION ALL SELECT id, rnk FROM qsem
    ) u
    GROUP BY u.id
)
SELECT c.id, c.doc_id, d.source_path, c.heading, c.page, c.text, f.score::double precision
FROM fused f
JOIN chunks c    ON c.id = f.id
JOIN documents d ON d.id = c.doc_id
ORDER BY f.score DESC
LIMIT match_limit;
$$;

-- Filtered search. The vector index is deliberately bypassed ("+ 0" stops the
-- planner from using HNSW): an approximate index filtered afterwards can return
-- too few rows when an area is small. An exact scan over one area is cheap at
-- this corpus size and never misses a passage.
CREATE OR REPLACE FUNCTION _hybrid_scoped(
    query_embedding VECTOR(1024), query_text TEXT, match_limit INT, rrf_k INT,
    area_filter TEXT
)
RETURNS TABLE (chunk_id BIGINT, doc_id BIGINT, source TEXT, heading TEXT,
               page INT, text TEXT, score DOUBLE PRECISION)
LANGUAGE sql STABLE AS $$
WITH sem AS (
    SELECT c.id, ROW_NUMBER() OVER (ORDER BY (c.embedding <=> query_embedding) + 0) AS rnk
    FROM chunks c JOIN documents d ON d.id = c.doc_id
    WHERE d.area = area_filter
    ORDER BY (c.embedding <=> query_embedding) + 0
    LIMIT match_limit
),
lex AS (
    SELECT c.id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, q) DESC) AS rnk
    FROM chunks c JOIN documents d ON d.id = c.doc_id,
         websearch_to_tsquery('english', query_text) q
    WHERE d.area = area_filter AND c.tsv @@ q
    ORDER BY ts_rank_cd(c.tsv, q) DESC
    LIMIT match_limit
),
q_raw AS (
    SELECT cq.chunk_id, (cq.embedding <=> query_embedding) + 0 AS dist
    FROM chunk_questions cq
    JOIN chunks c    ON c.id = cq.chunk_id
    JOIN documents d ON d.id = c.doc_id
    WHERE d.area = area_filter
    ORDER BY (cq.embedding <=> query_embedding) + 0
    LIMIT match_limit * 3
),
qsem AS (
    SELECT q_raw.chunk_id AS id,
           ROW_NUMBER() OVER (ORDER BY min(q_raw.dist)) AS rnk
    FROM q_raw
    GROUP BY q_raw.chunk_id
    ORDER BY min(q_raw.dist)
    LIMIT match_limit
),
fused AS (
    SELECT u.id, sum(1.0 / (rrf_k + u.rnk)) AS score
    FROM (
        SELECT id, rnk FROM sem
        UNION ALL SELECT id, rnk FROM lex
        UNION ALL SELECT id, rnk FROM qsem
    ) u
    GROUP BY u.id
)
SELECT c.id, c.doc_id, d.source_path, c.heading, c.page, c.text, f.score::double precision
FROM fused f
JOIN chunks c    ON c.id = f.id
JOIN documents d ON d.id = c.doc_id
ORDER BY f.score DESC
LIMIT match_limit;
$$;

-- Public entry point. With no area (NULL) it runs the unchanged search, so
-- every existing caller and evaluation behaves exactly as before.
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding VECTOR(1024),
    query_text      TEXT,
    match_limit     INT DEFAULT 40,
    rrf_k           INT DEFAULT 60,
    area_filter     TEXT DEFAULT NULL
)
RETURNS TABLE (chunk_id BIGINT, doc_id BIGINT, source TEXT, heading TEXT,
               page INT, text TEXT, score DOUBLE PRECISION)
LANGUAGE plpgsql STABLE AS $$
BEGIN
    IF area_filter IS NULL THEN
        RETURN QUERY SELECT * FROM _hybrid_all(query_embedding, query_text, match_limit, rrf_k) h
                     ORDER BY h.score DESC;
    ELSE
        RETURN QUERY SELECT * FROM _hybrid_scoped(query_embedding, query_text, match_limit, rrf_k, area_filter) h
                     ORDER BY h.score DESC;
    END IF;
END;
$$;
