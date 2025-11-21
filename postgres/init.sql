-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create candidates table
CREATE TABLE candidates (
    id SERIAL PRIMARY KEY,
    candidate_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(20),
    location JSONB,
    professional_summary TEXT,
    raw_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create candidate embeddings table
CREATE TABLE candidate_embeddings (
    id SERIAL PRIMARY KEY,
    candidate_id VARCHAR(255) NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    embedding vector(1536),  -- For OpenAI embeddings (1536 dimensions)
    embedding_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(candidate_id)
);

-- Create jobs table
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    company VARCHAR(255),
    description TEXT NOT NULL,
    requirements TEXT,
    raw_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create job embeddings table
CREATE TABLE job_embeddings (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    embedding vector(1536),  -- For OpenAI embeddings
    embedding_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_id)
);

-- Create recommendations table (for audit trail)
CREATE TABLE recommendations (
    id SERIAL PRIMARY KEY,
    candidate_id VARCHAR(255) NOT NULL REFERENCES candidates(candidate_id),
    job_id VARCHAR(255) NOT NULL REFERENCES jobs(job_id),
    match_score FLOAT,
    skills_match FLOAT,
    experience_match FLOAT,
    education_match FLOAT,
    semantic_similarity FLOAT,
    matched_skills TEXT[],
    missing_skills TEXT[],
    explanation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indices for performance
CREATE INDEX idx_candidate_id ON candidates(candidate_id);
CREATE INDEX idx_job_id ON jobs(job_id);
CREATE INDEX idx_candidate_embeddings_candidate_id ON candidate_embeddings(candidate_id);
CREATE INDEX idx_job_embeddings_job_id ON job_embeddings(job_id);
CREATE INDEX idx_recommendations_candidate_job ON recommendations(candidate_id, job_id);

-- Create vector indices for similarity search (IVFFLAT for speed)
CREATE INDEX idx_candidate_embeddings_vector ON candidate_embeddings 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_job_embeddings_vector ON job_embeddings 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Sample data for testing
INSERT INTO candidates (candidate_id, name, email, phone, location, raw_data) 
VALUES (
    'cand_001',
    'John Doe',
    'john@example.com',
    '+1-234-567-8900',
    '{"city": "San Francisco", "country": "US"}'::jsonb,
    '{}'::jsonb
);

INSERT INTO jobs (job_id, title, company, description, raw_data) 
VALUES (
    'job_001',
    'Senior AI Engineer',
    'Tech Corp',
    'Looking for an experienced AI engineer with expertise in NLP and LLMs',
    '{}'::jsonb
);