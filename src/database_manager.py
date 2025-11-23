"""
Database Manager for CV Parser and Recommender System
Handles all PostgreSQL operations with pgvector support
"""
import logging
import psycopg2
from psycopg2.extras import Json, execute_values
from typing import Dict, Any, List, Optional
from datetime import datetime

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
    
    def get_all_candidates(self, profession: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all candidates, optionally filtered by profession
        
        Args:
            profession: Optional profession filter
            
        Returns:
            List of candidate dictionaries
        """
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
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
