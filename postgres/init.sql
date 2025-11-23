-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create candidates table (IF NOT EXISTS makes it safe to run multiple times)
CREATE TABLE IF NOT EXISTS candidates (
    id SERIAL PRIMARY KEY,
    candidate_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(20),
    location JSONB,
    professional_summary TEXT,
    profession VARCHAR(100),
    work_experience JSONB,
    education JSONB,
    skills_technical TEXT[],
    skills_soft TEXT[],
    certifications JSONB,
    achievements JSONB,
    languages JSONB,
    raw_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create candidate embeddings table
CREATE TABLE IF NOT EXISTS candidate_embeddings (
    id SERIAL PRIMARY KEY,
    candidate_id VARCHAR(255) NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    embedding vector(768),  -- For sentence-transformers/all-mpnet-base-v2 (768 dimensions)
    embedding_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(candidate_id)
);

-- Create jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    job_title VARCHAR(255) NOT NULL,
    company VARCHAR(255),
    description TEXT NOT NULL,
    responsibilities TEXT,
    skills_technical TEXT[],
    skills_soft TEXT[],
    experience_years_min INTEGER,
    experience_years_max INTEGER,
    seniority_level VARCHAR(50),
    education_required VARCHAR(100),
    education_field VARCHAR(100),
    certifications TEXT[],
    languages JSONB,
    location JSONB,
    posted_date DATE,
    application_deadline DATE,
    raw_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create job embeddings table
CREATE TABLE IF NOT EXISTS job_embeddings (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    embedding vector(768),  -- For sentence-transformers/all-mpnet-base-v2 (768 dimensions)
    embedding_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_id)
);

-- Create recommendations table 
CREATE TABLE IF NOT EXISTS recommendations (
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
CREATE INDEX IF NOT EXISTS idx_candidate_id ON candidates(candidate_id);
CREATE INDEX IF NOT EXISTS idx_job_id ON jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_candidate_embeddings_candidate_id ON candidate_embeddings(candidate_id);
CREATE INDEX IF NOT EXISTS idx_job_embeddings_job_id ON job_embeddings(job_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_candidate_job ON recommendations(candidate_id, job_id);

-- Create vector indices for similarity search (IVFFLAT for speed)
CREATE INDEX IF NOT EXISTS idx_candidate_embeddings_vector ON candidate_embeddings 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_job_embeddings_vector ON job_embeddings 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- -- Create GIN index on skills_technical array
-- CREATE INDEX idx_skills_technical ON candidates USING GIN(skills_technical);
-- -- Create GIN index on skills_soft array
-- CREATE INDEX idx_skills_soft ON candidates USING GIN(skills_soft);