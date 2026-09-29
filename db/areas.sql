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
