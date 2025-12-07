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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(candidate_id, job_id)
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

-- ============================================================================
-- PERFORMANCE MONITORING TABLES
-- ============================================================================

-- Performance metrics table - stores operation timing with dataset size context
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    operation_type VARCHAR(100) NOT NULL,
    entity_id VARCHAR(255),
    duration_seconds FLOAT NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    metadata JSONB,
    dataset_size_cvs INTEGER,
    dataset_size_jobs INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Query performance table - tracks database query metrics
CREATE TABLE IF NOT EXISTS query_performance (
    id SERIAL PRIMARY KEY,
    query_type VARCHAR(100) NOT NULL,
    duration_ms FLOAT NOT NULL,
    rows_affected INTEGER,
    index_used BOOLEAN,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System metrics table - resource usage snapshots
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    cpu_percent FLOAT,
    memory_mb FLOAT,
    disk_io_mb FLOAT,
    active_workers INTEGER,
    throughput_per_min FLOAT,
    dataset_size_cvs INTEGER,
    dataset_size_jobs INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing sessions table - tracks each processing run
CREATE TABLE IF NOT EXISTS processing_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    session_type VARCHAR(50) NOT NULL,  -- 'cv_processing', 'job_processing', 'recommendation_generation'
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_seconds FLOAT,
    items_processed INTEGER DEFAULT 0,
    items_success INTEGER DEFAULT 0,
    items_failed INTEGER DEFAULT 0,
    items_skipped INTEGER DEFAULT 0,
    total_cvs_in_db INTEGER,
    total_jobs_in_db INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create suspicious_resumes table for security monitoring
CREATE TABLE IF NOT EXISTS suspicious_resumes (
    id SERIAL PRIMARY KEY,
    candidate_id VARCHAR(255) NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    threats_detected TEXT[],
    anomalies_detected TEXT[],
    threat_count INTEGER DEFAULT 0,
    anomaly_count INTEGER DEFAULT 0,
    severity VARCHAR(20) CHECK (severity IN ('none', 'low', 'medium', 'high', 'critical')),
    requires_manual_review BOOLEAN DEFAULT FALSE,
    reviewed BOOLEAN DEFAULT FALSE,
    reviewed_by VARCHAR(255),
    reviewed_at TIMESTAMP,
    review_notes TEXT,
    false_positive BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    UNIQUE(candidate_id)
);

-- Create indices for performance queries
CREATE INDEX IF NOT EXISTS idx_perf_operation_type ON performance_metrics(operation_type);
CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_type ON query_performance(query_type);
CREATE INDEX IF NOT EXISTS idx_query_timestamp ON query_performance(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_session_type ON processing_sessions(session_type);
CREATE INDEX IF NOT EXISTS idx_session_timestamp ON processing_sessions(start_time);

-- Create indices for security table
CREATE INDEX IF NOT EXISTS idx_suspicious_severity ON suspicious_resumes(severity);
CREATE INDEX IF NOT EXISTS idx_suspicious_timestamp ON suspicious_resumes(detection_timestamp);
CREATE INDEX IF NOT EXISTS idx_suspicious_review ON suspicious_resumes(requires_manual_review, reviewed);

-- ============================================================================
-- EVALUATION TRACKING TABLES (Optional - for development use only)
-- ============================================================================

-- Evaluation sessions table - tracks each evaluation run
CREATE TABLE IF NOT EXISTS evaluation_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    split_name VARCHAR(20) NOT NULL,  -- 'train', 'val', 'test'
    top_k INTEGER,
    k_values INTEGER[],
    num_candidates INTEGER,
    num_jobs INTEGER,
    use_existing_recommendations BOOLEAN,
    duration_seconds FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evaluation metrics table - stores aggregate metrics for each session
CREATE TABLE IF NOT EXISTS evaluation_metrics (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL REFERENCES evaluation_sessions(session_id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,  -- 'main_system', 'random', 'popularity', etc.
    metric_name VARCHAR(50) NOT NULL,  -- 'precision@10', 'ndcg@5', 'mrr', etc.
    metric_value FLOAT NOT NULL,
    k_value INTEGER,  -- NULL for metrics like MRR that don't use K
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, model_name, metric_name, k_value)
);

-- Create indices for evaluation queries
CREATE INDEX IF NOT EXISTS idx_eval_session_id ON evaluation_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_eval_split ON evaluation_sessions(split_name);
CREATE INDEX IF NOT EXISTS idx_eval_date ON evaluation_sessions(evaluation_date);
CREATE INDEX IF NOT EXISTS idx_eval_metrics_session ON evaluation_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_eval_metrics_model ON evaluation_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_eval_metrics_name ON evaluation_metrics(metric_name);