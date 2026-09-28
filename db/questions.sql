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
