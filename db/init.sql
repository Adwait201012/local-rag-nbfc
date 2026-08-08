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
