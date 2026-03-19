-- Gajashastra Sanskrit Intelligence Platform
-- Supabase Migration - Run in SQL Editor
-- Requires: pgvector extension (enabled by default on Supabase)

CREATE EXTENSION IF NOT EXISTS vector;

-- Texts
CREATE TABLE IF NOT EXISTS texts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    title_devanagari VARCHAR(500),
    author VARCHAR(255),
    author_devanagari VARCHAR(255),
    language VARCHAR(50) DEFAULT 'sanskrit',
    description TEXT,
    source_url VARCHAR(1000),
    total_pages INTEGER DEFAULT 0,
    total_chapters INTEGER DEFAULT 0,
    total_verses INTEGER DEFAULT 0,
    metadata_json JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chapters
CREATE TABLE IF NOT EXISTS chapters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text_id UUID REFERENCES texts(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    title_devanagari VARCHAR(500),
    "order" INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    summary TEXT,
    metadata_json JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_chapters_text ON chapters(text_id);

-- Verses
CREATE TABLE IF NOT EXISTS verses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chapter_id UUID REFERENCES chapters(id) ON DELETE CASCADE,
    verse_number VARCHAR(50),
    "order" INTEGER NOT NULL,
    sanskrit_devanagari TEXT NOT NULL,
    sanskrit_iast TEXT,
    english_translation TEXT,
    english_summary TEXT,
    commentary TEXT,
    verse_type VARCHAR(50) DEFAULT 'verse',
    meter VARCHAR(100),
    page_number INTEGER,
    source_file VARCHAR(255),
    extraction_model VARCHAR(100),
    extraction_confidence FLOAT,
    metadata_json JSONB DEFAULT '{}',
    is_indexed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_verses_chapter ON verses(chapter_id);
CREATE INDEX idx_verses_order ON verses("order");

-- Verse Embeddings (pgvector)
CREATE TABLE IF NOT EXISTS verse_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verse_id UUID REFERENCES verses(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    context_prefix TEXT,
    embedding vector(3072) NOT NULL,
    embedding_model VARCHAR(100) NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    metadata_json JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_verse_emb_verse ON verse_embeddings(verse_id);
CREATE INDEX idx_verse_emb_hnsw ON verse_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Insights
CREATE TABLE IF NOT EXISTS insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verse_id UUID REFERENCES verses(id) ON DELETE SET NULL,
    insight_type VARCHAR(50) NOT NULL,
    title VARCHAR(500),
    content TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.0,
    teaching TEXT,
    application TEXT,
    cross_references JSONB DEFAULT '[]',
    embedding vector(3072),
    embedding_model VARCHAR(100),
    source_model VARCHAR(100),
    is_approved BOOLEAN DEFAULT FALSE,
    metadata_json JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Patterns
CREATE TABLE IF NOT EXISTS patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    description TEXT,
    pattern_type VARCHAR(50),
    when_to_use TEXT,
    what_to_do TEXT,
    how_to_apply TEXT,
    pitfalls TEXT,
    source_verse_ids JSONB DEFAULT '[]',
    application_count INTEGER DEFAULT 0,
    metadata_json JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Cross References
CREATE TABLE IF NOT EXISTS cross_references (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_verse_id UUID REFERENCES verses(id) ON DELETE CASCADE,
    target_verse_id UUID REFERENCES verses(id) ON DELETE CASCADE,
    reference_type VARCHAR(50),
    description TEXT,
    confidence FLOAT DEFAULT 0.0,
    is_verified BOOLEAN DEFAULT FALSE,
    metadata_json JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Protocols (Zoo/Veterinary)
CREATE TABLE IF NOT EXISTS protocols (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    title_devanagari VARCHAR(500),
    protocol_type VARCHAR(100) NOT NULL,
    category VARCHAR(100),
    condition TEXT NOT NULL,
    symptoms JSONB DEFAULT '[]',
    diagnosis TEXT,
    treatment TEXT,
    herbs JSONB DEFAULT '[]',
    procedure_steps JSONB DEFAULT '[]',
    precautions TEXT,
    body_part VARCHAR(100),
    season VARCHAR(50),
    severity VARCHAR(50),
    source_verse_id UUID REFERENCES verses(id) ON DELETE SET NULL,
    source_reference VARCHAR(255),
    is_verified BOOLEAN DEFAULT FALSE,
    metadata_json JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Assistant Sessions
CREATE TABLE IF NOT EXISTS assistant_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500),
    context_mode VARCHAR(50) DEFAULT 'general',
    message_count INTEGER DEFAULT 0,
    last_query TEXT,
    metadata_json JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Assistant Messages
CREATE TABLE IF NOT EXISTS assistant_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES assistant_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    sources JSONB DEFAULT '[]',
    token_count INTEGER DEFAULT 0,
    metadata_json JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Search Indexes (tracking)
CREATE TABLE IF NOT EXISTS search_indexes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    total_items INTEGER DEFAULT 0,
    processed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    metadata_json JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Words (for word-by-word breakdown)
CREATE TABLE IF NOT EXISTS words (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verse_id UUID REFERENCES verses(id) ON DELETE CASCADE,
    word_devanagari VARCHAR(255) NOT NULL,
    word_iast VARCHAR(255),
    root_devanagari VARCHAR(255),
    root_iast VARCHAR(255),
    "order" INTEGER NOT NULL,
    part_of_speech VARCHAR(50),
    gender VARCHAR(20),
    "number" VARCHAR(20),
    "case" VARCHAR(50),
    tense VARCHAR(50),
    voice VARCHAR(50),
    compound_type VARCHAR(50),
    compound_parts JSONB DEFAULT '[]',
    meaning_english VARCHAR(500),
    meaning_hindi VARCHAR(500),
    notes TEXT,
    metadata_json JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_words_verse ON words(verse_id);
