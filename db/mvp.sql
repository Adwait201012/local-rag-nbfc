CREATE TABLE IF NOT EXISTS jobs (
    id         BIGSERIAL PRIMARY KEY,
    filename   TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'queued',
    message    TEXT,
    n_chunks   INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
