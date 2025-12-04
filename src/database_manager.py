"""
Database Manager for CV Parser and Recommender System
Handles all PostgreSQL operations with pgvector support
"""
import logging
import time
import psycopg2
from psycopg2.extras import Json, execute_values
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.performance_monitor import track_time, track_performance, track_query, get_monitor

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage PostgreSQL database operations for candidates and embeddings"""
    
    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize database connection
        
        Args:
            db_config: Database configuration dictionary
        """
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.monitor = None  # Will be set lazily via get_monitor() when needed
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('database'),
                user=self.db_config.get('user'),
                password=self.db_config.get('password')
            )
            self.cursor = self.conn.cursor()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def candidate_exists(self, candidate_id: str) -> bool:
        """
        Check if candidate already exists in database
        
        Args:
            candidate_id: Unique candidate identifier
            
        Returns:
            True if candidate exists, False otherwise
        """
        try:
            self.cursor.execute(
                "SELECT COUNT(*) FROM candidates WHERE candidate_id = %s",
                (candidate_id,)
            )
            count = self.cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            logger.error(f"Error checking candidate existence: {e}")
            return False
    
    @track_performance('cv_db_insert')
    def insert_candidate(self, candidate_id: str, resume_json: dict, profession: str) -> bool:
        """
        Insert candidate data into candidates table
        
        Args:
            candidate_id: Unique candidate identifier
            resume_json: Parsed resume data
            profession: Candidate's profession category
            
        Returns:
            True if successful, False otherwise
        """
        try:
            basics = resume_json.get('basics', {})
            
            # Extract structured data
            name = basics.get('name', '')
            email = basics.get('email', '')
            phone = basics.get('phone', '')
            location = basics.get('location', {})
            summary = resume_json.get('summary', '')
            
            # Extract arrays and JSON data
            work_experience = resume_json.get('work', [])
            education = resume_json.get('education', [])
            skills = resume_json.get('skills', {})
            tech_skills = skills.get('technical', [])
            soft_skills = skills.get('soft', [])
            certifications = resume_json.get('certifications', [])
            achievements = resume_json.get('achievements', [])
            languages = resume_json.get('languages', [])
            
            # Insert query
            insert_query = """
                INSERT INTO candidates (
                    candidate_id, name, email, phone, location, 
                    professional_summary, profession,
                    work_experience, education, 
                    skills_technical, skills_soft,
                    certifications, achievements, languages,
                    raw_data, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (candidate_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    email = EXCLUDED.email,
                    phone = EXCLUDED.phone,
                    location = EXCLUDED.location,
                    professional_summary = EXCLUDED.professional_summary,
                    profession = EXCLUDED.profession,
                    work_experience = EXCLUDED.work_experience,
                    education = EXCLUDED.education,
                    skills_technical = EXCLUDED.skills_technical,
                    skills_soft = EXCLUDED.skills_soft,
                    certifications = EXCLUDED.certifications,
                    achievements = EXCLUDED.achievements,
                    languages = EXCLUDED.languages,
                    raw_data = EXCLUDED.raw_data,
                    updated_at = EXCLUDED.updated_at
            """
            
            now = datetime.now()
            
            self.cursor.execute(insert_query, (
                candidate_id, name, email, phone, Json(location),
                summary, profession,
                Json(work_experience), Json(education),
                tech_skills, soft_skills,
                Json(certifications), Json(achievements), Json(languages),
                Json(resume_json), now, now
            ))
            
            self.conn.commit()
            logger.info(f"Inserted/Updated candidate: {candidate_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting candidate {candidate_id}: {e}")
            self.conn.rollback()
            return False
    
    @track_performance('cv_embedding_db_insert')
    def insert_embedding(
        self, 
        candidate_id: str, 
        embedding: List[float], 
        model_name: str
    ) -> bool:
        """
        Insert embedding vector for candidate
        
        Args:
            candidate_id: Unique candidate identifier
            embedding: Embedding vector (768 dimensions)
            model_name: Name of the embedding model used
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert list to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            insert_query = """
                INSERT INTO candidate_embeddings (
                    candidate_id, embedding, embedding_model, created_at
                ) VALUES (
                    %s, %s::vector, %s, %s
                )
                ON CONFLICT (candidate_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    created_at = EXCLUDED.created_at
            """
            
            self.cursor.execute(insert_query, (
                candidate_id, 
                embedding_str,
                model_name,
                datetime.now()
            ))
            
            self.conn.commit()
            logger.info(f"Inserted/Updated embedding for: {candidate_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting embedding for {candidate_id}: {e}")
            self.conn.rollback()
            return False
    
    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve candidate data by ID
        
        Args:
            candidate_id: Unique candidate identifier
            
        Returns:
            Candidate data dictionary or None
        """
        try:
            with track_query('get_candidate'):
                self.cursor.execute(
                    "SELECT * FROM candidates WHERE candidate_id = %s",
                    (candidate_id,)
                )
                result = self.cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in self.cursor.description]
                return dict(zip(columns, result))
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving candidate {candidate_id}: {e}")
            return None
    
    def get_all_candidate_ids(self, profession: Optional[str] = None) -> List[str]:
        """
        Retrieve all candidate IDs (memory efficient)
        
        Args:
            profession: Optional profession filter
            
        Returns:
            List of candidate IDs
        """
        try:
            if profession:
                query = "SELECT candidate_id FROM candidates WHERE profession = %s"
                self.cursor.execute(query, (profession,))
            else:
                query = "SELECT candidate_id FROM candidates"
                self.cursor.execute(query)
            
            results = self.cursor.fetchall()
            return [row[0] for row in results]
            
        except Exception as e:
            logger.error(f"Error retrieving candidate IDs: {e}")
            return []
    
    def get_all_candidates(self, profession: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all candidates, optionally filtered by profession
        
        Args:
            profession: Optional profession filter
            
        Returns:
            List of candidate dictionaries
        """
        # TODO: make this more efficient for large datasets (pagination?), all candidates may not fit in memory
        try:
            if profession:
                query = "SELECT * FROM candidates WHERE profession = %s"
                self.cursor.execute(query, (profession,))
            else:
                query = "SELECT * FROM candidates"
                self.cursor.execute(query)
            
            results = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            logger.error(f"Error retrieving candidates: {e}")
            return []
    
    def get_candidate_count(self) -> int:
        """
        Get total count of candidates in database
        
        Returns:
            Number of candidates
        """
        try:
            self.cursor.execute("SELECT COUNT(*) FROM candidates")
            return self.cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting candidate count: {e}")
            return 0
    
    def get_profession_counts(self) -> Dict[str, int]:
        """
        Get count of candidates by profession
        
        Returns:
            Dictionary mapping profession to count
        """
        try:
            self.cursor.execute(
                "SELECT profession, COUNT(*) FROM candidates GROUP BY profession"
            )
            results = self.cursor.fetchall()
            return {row[0]: row[1] for row in results}
        except Exception as e:
            logger.error(f"Error getting profession counts: {e}")
            return {}
    
    # ========== JOB-RELATED METHODS ==========
    
    def job_exists(self, job_id: str) -> bool:
        """
        Check if job already exists in database
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            True if job exists, False otherwise
        """
        try:
            self.cursor.execute(
                "SELECT COUNT(*) FROM jobs WHERE job_id = %s",
                (job_id,)
            )
            count = self.cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            logger.error(f"Error checking job existence: {e}")
            return False
    
    def insert_job(self, job_id: str, job_json: dict) -> bool:
        """
        Insert job data into jobs table
        
        Args:
            job_id: Unique job identifier
            job_json: Parsed job data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract structured data
            job_title = job_json.get('job_title', '')
            company = job_json.get('company', '')
            description = job_json.get('description', '')
            responsibilities = job_json.get('responsibilities', '')
            
            # Extract skills
            skills_technical = job_json.get('skills_technical', [])
            skills_soft = job_json.get('skills_soft', [])
            
            # Extract experience requirements
            experience_years_min = job_json.get('experience_years_min')
            experience_years_max = job_json.get('experience_years_max')
            seniority_level = job_json.get('seniority_level', '')
            
            # Extract education requirements
            education_required = job_json.get('education_required', '')
            education_field = job_json.get('education_field', '')
            
            # Extract certifications and languages
            certifications = job_json.get('certifications', [])
            languages = job_json.get('languages', [])
            location = job_json.get('location', {})
            
            # Extract dates
            posted_date = job_json.get('posted_date')
            application_deadline = job_json.get('application_deadline')
            
            # Insert query
            insert_query = """
                INSERT INTO jobs (
                    job_id, job_title, company, description, responsibilities,
                    skills_technical, skills_soft,
                    experience_years_min, experience_years_max, seniority_level,
                    education_required, education_field,
                    certifications, languages, location,
                    posted_date, application_deadline,
                    raw_data, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (job_id) DO UPDATE SET
                    job_title = EXCLUDED.job_title,
                    company = EXCLUDED.company,
                    description = EXCLUDED.description,
                    responsibilities = EXCLUDED.responsibilities,
                    skills_technical = EXCLUDED.skills_technical,
                    skills_soft = EXCLUDED.skills_soft,
                    experience_years_min = EXCLUDED.experience_years_min,
                    experience_years_max = EXCLUDED.experience_years_max,
                    seniority_level = EXCLUDED.seniority_level,
                    education_required = EXCLUDED.education_required,
                    education_field = EXCLUDED.education_field,
                    certifications = EXCLUDED.certifications,
                    languages = EXCLUDED.languages,
                    location = EXCLUDED.location,
                    posted_date = EXCLUDED.posted_date,
                    application_deadline = EXCLUDED.application_deadline,
                    raw_data = EXCLUDED.raw_data,
                    updated_at = EXCLUDED.updated_at
            """
            
            now = datetime.now()
            
            self.cursor.execute(insert_query, (
                job_id, job_title, company, description, responsibilities,
                skills_technical, skills_soft,
                experience_years_min, experience_years_max, seniority_level,
                education_required, education_field,
                certifications, Json(languages), Json(location),
                posted_date, application_deadline,
                Json(job_json), now, now
            ))
            
            self.conn.commit()
            logger.info(f"Inserted/Updated job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting job {job_id}: {e}")
            self.conn.rollback()
            return False
    
    def insert_job_embedding(
        self, 
        job_id: str, 
        embedding: List[float], 
        model_name: str
    ) -> bool:
        """
        Insert embedding vector for job
        
        Args:
            job_id: Unique job identifier
            embedding: Embedding vector (768 dimensions)
            model_name: Name of the embedding model used
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert list to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            insert_query = """
                INSERT INTO job_embeddings (
                    job_id, embedding, embedding_model, created_at
                ) VALUES (
                    %s, %s::vector, %s, %s
                )
                ON CONFLICT (job_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    created_at = EXCLUDED.created_at
            """
            
            self.cursor.execute(insert_query, (
                job_id, 
                embedding_str,
                model_name,
                datetime.now()
            ))
            
            self.conn.commit()
            logger.info(f"Inserted/Updated embedding for job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting job embedding for {job_id}: {e}")
            self.conn.rollback()
            return False
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve job data by ID
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Job data dictionary or None
        """
        try:
            with track_query('get_job'):
                self.cursor.execute(
                    "SELECT * FROM jobs WHERE job_id = %s",
                    (job_id,)
                )
                result = self.cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in self.cursor.description]
                return dict(zip(columns, result))
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving job {job_id}: {e}")
            return None
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """
        Retrieve all jobs from database
        
        Returns:
            List of job dictionaries
        """
        try:
            query = "SELECT * FROM jobs"
            self.cursor.execute(query)
            
            results = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            logger.error(f"Error retrieving jobs: {e}")
            return []
    
    def get_job_count(self) -> int:
        """
        Get total count of jobs in database
        
        Returns:
            Number of jobs
        """
        try:
            self.cursor.execute("SELECT COUNT(*) FROM jobs")
            return self.cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting job count: {e}")
            return 0
    
    # ========== EMBEDDING RETRIEVAL METHODS ==========
    
    def get_candidate_embedding(self, candidate_id: str) -> Optional[List[float]]:
        """
        Retrieve embedding vector for a candidate
        
        Args:
            candidate_id: Unique candidate identifier
            
        Returns:
            Embedding vector as list of floats, or None if not found
        """
        try:
            with track_query('get_candidate_embedding'):
                self.cursor.execute(
                    "SELECT embedding FROM candidate_embeddings WHERE candidate_id = %s",
                    (candidate_id,)
                )
                result = self.cursor.fetchone()
            
            if result and result[0]:
                # Convert PostgreSQL vector to list of floats
                return result[0]
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving embedding for candidate {candidate_id}: {e}")
            return None
    
    def get_job_embedding(self, job_id: str) -> Optional[List[float]]:
        """
        Retrieve embedding vector for a job
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Embedding vector as list of floats, or None if not found
        """
        try:
            with track_query('get_job_embedding'):
                self.cursor.execute(
                    "SELECT embedding FROM job_embeddings WHERE job_id = %s",
                    (job_id,)
                )
                result = self.cursor.fetchone()
            
            if result and result[0]:
                # Convert PostgreSQL vector to list of floats
                return result[0]
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving embedding for job {job_id}: {e}")
            return None
    
    def get_all_candidate_embeddings(self) -> Dict[str, List[float]]:
        """
        Retrieve all candidate embeddings
        
        Returns:
            Dictionary mapping candidate_id to embedding vector
        """
        try:
            self.cursor.execute(
                "SELECT candidate_id, embedding FROM candidate_embeddings"
            )
            results = self.cursor.fetchall()
            
            return {row[0]: row[1] for row in results if row[1]}
            
        except Exception as e:
            logger.error(f"Error retrieving all candidate embeddings: {e}")
            return {}
    
    def get_all_job_embeddings(self) -> Dict[str, List[float]]:
        """
        Retrieve all job embeddings
        
        Returns:
            Dictionary mapping job_id to embedding vector
        """
        try:
            self.cursor.execute(
                "SELECT job_id, embedding FROM job_embeddings"
            )
            results = self.cursor.fetchall()
            
            return {row[0]: row[1] for row in results if row[1]}
            
        except Exception as e:
            logger.error(f"Error retrieving all job embeddings: {e}")
            return {}
    
    def get_all_jobs_with_similarity_for_candidate(
        self, 
        candidate_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get ALL jobs with pre-computed semantic similarity for a candidate.
        Uses PostgreSQL pgvector for efficient similarity computation.
        
        This is the OPTIMIZED method that:
        1. Computes similarity directly in PostgreSQL (fast C code)
        2. Uses IVFFLAT vector index for speed
        3. Returns all jobs with similarity already computed
        4. Avoids loading embeddings into Python memory
        
        Args:
            candidate_id: Unique candidate identifier
            
        Returns:
            List of job dictionaries with 'semantic_similarity' field added
        """
        try:
            # Query that joins jobs with their embeddings and computes similarity
            # using pgvector's cosine distance operator (<=>)
            query = """
                SELECT 
                    j.job_id,
                    j.job_title,
                    j.company,
                    j.description,
                    j.responsibilities,
                    j.skills_technical,
                    j.skills_soft,
                    j.experience_years_min,
                    j.experience_years_max,
                    j.seniority_level,
                    j.education_required,
                    j.education_field,
                    j.certifications,
                    j.languages,
                    j.location,
                    1 - (je.embedding <=> ce.embedding) AS semantic_similarity
                FROM 
                    jobs j
                    INNER JOIN job_embeddings je ON j.job_id = je.job_id
                    CROSS JOIN candidate_embeddings ce
                WHERE 
                    ce.candidate_id = %s
                    AND je.embedding IS NOT NULL
                    AND ce.embedding IS NOT NULL
            """
            
            # Track query performance - THIS IS CRITICAL FOR TASK 4
            with track_query('vector_similarity_search'):
                self.cursor.execute(query, (candidate_id,))
                results = self.cursor.fetchall()
            
            columns = [desc[0] for desc in self.cursor.description]
            jobs_with_similarity = [dict(zip(columns, row)) for row in results]
            
            logger.info(
                f"Retrieved {len(jobs_with_similarity)} jobs with pre-computed "
                f"similarity for candidate {candidate_id}"
            )
            
            return jobs_with_similarity
            
        except Exception as e:
            logger.error(
                f"Error retrieving jobs with similarity for candidate {candidate_id}: {e}"
            )
            return []
    
    def get_top_k_jobs_by_similarity(
        self, 
        candidate_id: str,
        top_k: int = 50,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        TWO-STAGE RETRIEVAL - STAGE 1: Fast filtering using vector similarity only.
        
        Get top-K most semantically similar jobs for a candidate using ONLY pgvector.
        This is much faster than computing all similarities + full scoring.
        
        Use this for Stage 1 filtering, then apply full scoring (Stage 2) to only these candidates.
        
        Performance benefits:
        - For 500 jobs: Filters down to 50 jobs (90% reduction in Python computation)
        - For 10,000 jobs: Filters down to 50 jobs (99.5% reduction!)
        - Database does heavy lifting using optimized C code and vector index
        
        Args:
            candidate_id: Unique candidate identifier
            top_k: Number of top similar jobs to return (default: 50)
            similarity_threshold: Minimum similarity score (0.0 to 1.0, default: 0.0)
            
        Returns:
            List of top-K job dictionaries with 'semantic_similarity' field, 
            ordered by similarity (highest first)
        """
        try:
            # STAGE 1 QUERY: Vector similarity only, with LIMIT for efficiency
            # The ORDER BY + LIMIT allows PostgreSQL to use the vector index optimally
            query = """
                SELECT 
                    j.job_id,
                    j.job_title,
                    j.company,
                    j.description,
                    j.responsibilities,
                    j.skills_technical,
                    j.skills_soft,
                    j.experience_years_min,
                    j.experience_years_max,
                    j.seniority_level,
                    j.education_required,
                    j.education_field,
                    j.certifications,
                    j.languages,
                    j.location,
                    1 - (je.embedding <=> ce.embedding) AS semantic_similarity
                FROM 
                    jobs j
                    INNER JOIN job_embeddings je ON j.job_id = je.job_id
                    CROSS JOIN candidate_embeddings ce
                WHERE 
                    ce.candidate_id = %s
                    AND je.embedding IS NOT NULL
                    AND ce.embedding IS NOT NULL
                    AND (1 - (je.embedding <=> ce.embedding)) >= %s
                ORDER BY 
                    je.embedding <=> ce.embedding ASC
                LIMIT %s
            """
            
            # Track query performance for Stage 1
            with track_query('vector_similarity_stage1_topk'):
                self.cursor.execute(query, (candidate_id, similarity_threshold, top_k))
                results = self.cursor.fetchall()
            
            columns = [desc[0] for desc in self.cursor.description]
            top_jobs = [dict(zip(columns, row)) for row in results]
            
            logger.info(
                f"Stage 1 filtered: Retrieved top-{len(top_jobs)} jobs "
                f"(threshold >= {similarity_threshold:.2f}) for candidate {candidate_id}"
            )
            
            return top_jobs
            
        except Exception as e:
            logger.error(
                f"Error in Stage 1 retrieval for candidate {candidate_id}: {e}"
            )
            return []
    
    # ========== RECOMMENDATION METHODS ==========
    
    def check_recommendation_exists(self, candidate_id: str, job_id: str) -> bool:
        """
        Check if recommendation already exists for a candidate-job pair
        
        Args:
            candidate_id: Unique candidate identifier
            job_id: Unique job identifier
            
        Returns:
            True if recommendation exists, False otherwise
        """
        try:
            self.cursor.execute(
                "SELECT COUNT(*) FROM recommendations WHERE candidate_id = %s AND job_id = %s",
                (candidate_id, job_id)
            )
            count = self.cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            logger.error(f"Error checking recommendation existence: {e}")
            return False
    
    def get_existing_recommendation_job_ids(self, candidate_id: str) -> set:
        """
        Get set of job IDs that already have recommendations for a candidate
        
        Args:
            candidate_id: Unique candidate identifier
            
        Returns:
            Set of job IDs with existing recommendations
        """
        try:
            with track_query('get_existing_recommendations'):
                self.cursor.execute(
                    "SELECT job_id FROM recommendations WHERE candidate_id = %s",
                    (candidate_id,)
                )
                results = self.cursor.fetchall()
            
            return {row[0] for row in results}
        except Exception as e:
            logger.error(f"Error getting existing recommendations: {e}")
            return set()
    
    def get_jobs_without_recommendations(
        self,
        candidate_id: str,
        use_two_stage: bool = True,
        stage1_top_k: int = 50,
        stage1_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Get jobs that DON'T have recommendations yet for a candidate.
        Combines vector similarity filtering with existence check.
        
        Args:
            candidate_id: Unique candidate identifier
            use_two_stage: Use two-stage filtering
            stage1_top_k: Top-K for Stage 1 filtering
            stage1_threshold: Similarity threshold
            
        Returns:
            List of job dictionaries without existing recommendations
        """
        try:
            # Get jobs with similarity (using chosen strategy)
            if use_two_stage:
                all_jobs = self.get_top_k_jobs_by_similarity(
                    candidate_id, stage1_top_k, stage1_threshold
                )
            else:
                all_jobs = self.get_all_jobs_with_similarity_for_candidate(candidate_id)
            
            # Get existing recommendation job IDs
            existing_job_ids = self.get_existing_recommendation_job_ids(candidate_id)
            
            # Filter out jobs that already have recommendations
            new_jobs = [job for job in all_jobs if job['job_id'] not in existing_job_ids]
            
            logger.info(
                f"Candidate {candidate_id}: {len(all_jobs)} jobs retrieved, "
                f"{len(existing_job_ids)} already have recommendations, "
                f"{len(new_jobs)} new jobs to process"
            )
            
            return new_jobs
            
        except Exception as e:
            logger.error(f"Error getting jobs without recommendations: {e}")
            return []
    
    def save_recommendation(
        self,
        candidate_id: str,
        job_id: str,
        match_score: float,
        skills_match: float,
        experience_match: float,
        education_match: float,
        semantic_similarity: float,
        matched_skills: List[str],
        missing_skills: List[str],
        explanation: str
    ) -> bool:
        """
        Save a job recommendation for a candidate
        
        Args:
            candidate_id: Unique candidate identifier
            job_id: Unique job identifier
            match_score: Overall match score
            skills_match: Skills match score
            experience_match: Experience match score
            education_match: Education match score
            semantic_similarity: Semantic similarity score
            matched_skills: List of matched skills
            missing_skills: List of missing skills
            explanation: Human-readable explanation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            insert_query = """
                INSERT INTO recommendations (
                    candidate_id, job_id, match_score,
                    skills_match, experience_match, education_match, semantic_similarity,
                    matched_skills, missing_skills, explanation, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (candidate_id, job_id) 
                DO UPDATE SET
                    match_score = EXCLUDED.match_score,
                    skills_match = EXCLUDED.skills_match,
                    experience_match = EXCLUDED.experience_match,
                    education_match = EXCLUDED.education_match,
                    semantic_similarity = EXCLUDED.semantic_similarity,
                    matched_skills = EXCLUDED.matched_skills,
                    missing_skills = EXCLUDED.missing_skills,
                    explanation = EXCLUDED.explanation,
                    created_at = EXCLUDED.created_at
            """
            
            self.cursor.execute(insert_query, (
                candidate_id, job_id, match_score,
                skills_match, experience_match, education_match, semantic_similarity,
                matched_skills, missing_skills, explanation,
                datetime.now()
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error saving recommendation for {candidate_id} -> {job_id}: {e}")
            self.conn.rollback()
            return False
    
    def save_recommendations_batch(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> int:
        """
        Save multiple recommendations in batch using execute_values for efficiency
        
        Args:
            recommendations: List of recommendation dictionaries with keys:
                candidate_id, job_id, match_score, skills_match, experience_match,
                education_match, semantic_similarity, matched_skills, missing_skills, explanation
            
        Returns:
            Number of successfully saved recommendations
        """
        if not recommendations:
            return 0
        
        try:
            # Prepare data for batch insert
            values = [
                (
                    rec['candidate_id'],
                    rec['job_id'],
                    rec['match_score'],
                    rec['skills_match'],
                    rec['experience_match'],
                    rec['education_match'],
                    rec['semantic_similarity'],
                    rec['matched_skills'],
                    rec['missing_skills'],
                    rec['explanation'],
                    datetime.now()
                )
                for rec in recommendations
            ]
            
            # Batch insert with execute_values (much faster than individual inserts)
            insert_query = """
                INSERT INTO recommendations (
                    candidate_id, job_id, match_score,
                    skills_match, experience_match, education_match, semantic_similarity,
                    matched_skills, missing_skills, explanation, created_at
                ) VALUES %s
                ON CONFLICT (candidate_id, job_id) 
                DO UPDATE SET
                    match_score = EXCLUDED.match_score,
                    skills_match = EXCLUDED.skills_match,
                    experience_match = EXCLUDED.experience_match,
                    education_match = EXCLUDED.education_match,
                    semantic_similarity = EXCLUDED.semantic_similarity,
                    matched_skills = EXCLUDED.matched_skills,
                    missing_skills = EXCLUDED.missing_skills,
                    explanation = EXCLUDED.explanation,
                    created_at = EXCLUDED.created_at
            """
            
            execute_values(
                self.cursor,
                insert_query,
                values,
                template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            )
            
            self.conn.commit()
            success_count = len(recommendations)
            logger.info(f"Batch saved {success_count} recommendations")
            return success_count
            
        except Exception as e:
            logger.error(f"Error batch saving recommendations: {e}")
            self.conn.rollback()
            return 0
    
    def get_recommendations_for_candidate(
        self,
        candidate_id: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve saved recommendations for a candidate
        
        Args:
            candidate_id: Unique candidate identifier
            top_k: Number of top recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            if top_k:
                query = """
                    SELECT * FROM recommendations 
                    WHERE candidate_id = %s 
                    ORDER BY match_score DESC 
                    LIMIT %s
                """
                self.cursor.execute(query, (candidate_id, top_k))
            else:
                query = """
                    SELECT * FROM recommendations 
                    WHERE candidate_id = %s 
                    ORDER BY match_score DESC
                """
                self.cursor.execute(query, (candidate_id,))
            
            results = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            logger.error(f"Error retrieving recommendations for {candidate_id}: {e}")
            return []
    
    def get_recommendations_for_job(
        self,
        job_id: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve saved recommendations for a job (best candidates)
        
        Args:
            job_id: Unique job identifier
            top_k: Number of top candidates to return
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            if top_k:
                query = """
                    SELECT * FROM recommendations 
                    WHERE job_id = %s 
                    ORDER BY match_score DESC 
                    LIMIT %s
                """
                self.cursor.execute(query, (job_id, top_k))
            else:
                query = """
                    SELECT * FROM recommendations 
                    WHERE job_id = %s 
                    ORDER BY match_score DESC
                """
                self.cursor.execute(query, (job_id,))
            
            results = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            logger.error(f"Error retrieving recommendations for job {job_id}: {e}")
            return []
    
    def get_recommendation_count(self) -> int:
        """
        Get total count of recommendations in database
        
        Returns:
            Number of recommendations
        """
        try:
            self.cursor.execute("SELECT COUNT(*) FROM recommendations")
            return self.cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting recommendation count: {e}")
            return 0
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")
    
    # ========== PERFORMANCE METRICS METHODS ==========
    
    def insert_performance_metric(self, metric) -> bool:
        """
        Insert performance metric into database
        
        Args:
            metric: PerformanceMetric object
            
        Returns:
            True if successful
        """
        try:
            insert_query = """
                INSERT INTO performance_metrics (
                    operation_type, entity_id, duration_seconds, success,
                    error_message, metadata, dataset_size_cvs, dataset_size_jobs, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(insert_query, (
                metric.operation_type,
                metric.entity_id,
                metric.duration_seconds,
                metric.success,
                metric.error_message,
                Json(metric.metadata),
                metric.dataset_size_cvs,
                metric.dataset_size_jobs,
                metric.timestamp
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.debug(f"Failed to insert performance metric: {e}")
            self.conn.rollback()
            return False
    
    def insert_query_metric(self, metric) -> bool:
        """
        Insert query performance metric into database
        
        Args:
            metric: QueryMetric object
            
        Returns:
            True if successful
        """
        try:
            insert_query = """
                INSERT INTO query_performance (
                    query_type, duration_ms, rows_affected, index_used, timestamp
                ) VALUES (%s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(insert_query, (
                metric.query_type,
                metric.duration_ms,
                metric.rows_affected,
                metric.index_used,
                metric.timestamp
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.debug(f"Failed to insert query metric: {e}")
            self.conn.rollback()
            return False
    
    def insert_system_metric(self, metric) -> bool:
        """
        Insert system metric into database
        
        Args:
            metric: SystemMetric object
            
        Returns:
            True if successful
        """
        try:
            insert_query = """
                INSERT INTO system_metrics (
                    cpu_percent, memory_mb, disk_io_mb, active_workers,
                    throughput_per_min, dataset_size_cvs, dataset_size_jobs, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(insert_query, (
                metric.cpu_percent,
                metric.memory_mb,
                metric.disk_io_mb,
                metric.active_workers,
                metric.throughput_per_min,
                metric.dataset_size_cvs,
                metric.dataset_size_jobs,
                metric.timestamp
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.debug(f"Failed to insert system metric: {e}")
            self.conn.rollback()
            return False
    
    def insert_processing_session(self, session) -> bool:
        """
        Insert processing session into database
        
        Args:
            session: ProcessingSession object
            
        Returns:
            True if successful
        """
        try:
            insert_query = """
                INSERT INTO processing_sessions (
                    session_id, session_type, start_time, end_time, duration_seconds,
                    items_processed, items_success, items_failed, items_skipped,
                    total_cvs_in_db, total_jobs_in_db, metadata, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(insert_query, (
                session.session_id,
                session.session_type,
                session.start_time,
                session.end_time,
                session.duration_seconds,
                session.items_processed,
                session.items_success,
                session.items_failed,
                session.items_skipped,
                session.total_cvs_in_db,
                session.total_jobs_in_db,
                Json(session.metadata),
                datetime.now()
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.debug(f"Failed to insert processing session: {e}")
            self.conn.rollback()
            return False
    
    def load_performance_metrics(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load performance metrics from database
        
        Args:
            limit: Maximum number of metrics to load (None = all)
            
        Returns:
            List of performance metrics as dictionaries
        """
        try:
            query = """
                SELECT operation_type, entity_id, duration_seconds, success,
                       error_message, metadata, dataset_size_cvs, dataset_size_jobs, timestamp
                FROM performance_metrics
                ORDER BY timestamp DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            
            metrics = []
            for row in rows:
                metrics.append({
                    'operation_type': row[0],
                    'entity_id': row[1],
                    'duration_seconds': row[2],
                    'success': row[3],
                    'error_message': row[4],
                    'metadata': row[5] or {},
                    'dataset_size_cvs': row[6],
                    'dataset_size_jobs': row[7],
                    'timestamp': row[8].isoformat() if row[8] else None
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to load performance metrics: {e}")
            return []
    
    def load_processing_sessions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load processing sessions from database
        
        Args:
            limit: Maximum number of sessions to load (None = all)
            
        Returns:
            List of processing sessions as dictionaries
        """
        try:
            query = """
                SELECT session_id, session_type, start_time, end_time, duration_seconds,
                       items_processed, items_success, items_failed, items_skipped,
                       total_cvs_in_db, total_jobs_in_db, metadata
                FROM processing_sessions
                ORDER BY start_time DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            
            sessions = []
            for row in rows:
                sessions.append({
                    'session_id': row[0],
                    'session_type': row[1],
                    'start_time': row[2].isoformat() if row[2] else None,
                    'end_time': row[3].isoformat() if row[3] else None,
                    'duration_seconds': row[4],
                    'items_processed': row[5],
                    'items_success': row[6],
                    'items_failed': row[7],
                    'items_skipped': row[8],
                    'total_cvs_in_db': row[9],
                    'total_jobs_in_db': row[10],
                    'metadata': row[11] or {}
                })
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to load processing sessions: {e}")
            return []
    
    def load_query_metrics(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load query performance metrics from database
        
        Args:
            limit: Maximum number of metrics to load (None = all)
            
        Returns:
            List of query metrics as dictionaries
        """
        try:
            query = """
                SELECT query_type, duration_ms, rows_affected, index_used, timestamp
                FROM query_performance
                ORDER BY timestamp DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            
            metrics = []
            for row in rows:
                metrics.append({
                    'query_type': row[0],
                    'duration_ms': row[1],
                    'rows_affected': row[2],
                    'index_used': row[3],
                    'timestamp': row[4].isoformat() if row[4] else None
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to load query metrics: {e}")
            return []
    
    def load_system_metrics(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load system resource metrics from database
        
        Args:
            limit: Maximum number of metrics to load (None = all)
            
        Returns:
            List of system metrics as dictionaries
        """
        try:
            query = """
                SELECT cpu_percent, memory_mb, disk_io_mb, active_workers,
                       throughput_per_min, dataset_size_cvs, dataset_size_jobs, timestamp
                FROM system_metrics
                ORDER BY timestamp DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            
            metrics = []
            for row in rows:
                metrics.append({
                    'cpu_percent': row[0],
                    'memory_mb': row[1],
                    'disk_io_mb': row[2],
                    'active_workers': row[3],
                    'throughput_per_min': row[4],
                    'dataset_size_cvs': row[5],
                    'dataset_size_jobs': row[6],
                    'timestamp': row[7].isoformat() if row[7] else None
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to load system metrics: {e}")
            return []
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
