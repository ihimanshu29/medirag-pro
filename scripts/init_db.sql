-- MediRAG Pro — Database Schema
-- Runs on first Postgres container start

-- Session memory: stores conversation history per session
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL,
    role        VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON chat_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON chat_sessions(created_at);

-- User feedback: thumbs up/down per response → feeds evaluation loop
CREATE TABLE IF NOT EXISTS feedback (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL,
    query       TEXT NOT NULL,
    answer      TEXT NOT NULL,
    rating      SMALLINT CHECK (rating IN (-1, 1)),  -- -1 = thumbs down, 1 = up
    comment     TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);

-- Document registry: tracks ingested documents
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        VARCHAR(512) NOT NULL UNIQUE,
    chunks_created  INT DEFAULT 0,
    tables_extracted INT DEFAULT 0,
    ingested_at     TIMESTAMPTZ DEFAULT NOW()
);
