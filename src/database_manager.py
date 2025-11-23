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
