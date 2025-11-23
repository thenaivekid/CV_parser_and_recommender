"""
Embedding Generator for CV Parser and Recommender System
Uses SentenceTransformer to generate embeddings from parsed resume JSON
"""
import logging
from typing import Tuple, List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings from resume data using SentenceTransformers"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")
    
    def prepare_text(self, resume_json: dict) -> str:
        """
        Concatenate relevant fields from resume JSON for embedding
        
        Args:
            resume_json: Parsed resume data
            
        Returns:
            Concatenated text string for embedding
        """
        parts = []
        
        # Note: I am skipping the name field because I do not think it adds value to the embedding
        # name = resume_json.get('basics', {}).get('name', '')
        # if name:
        #     parts.append(f"Name: {name}")
        
        # Professional Summary
        summary = resume_json.get('summary', '')
        if summary:
            parts.append(f"Summary: {summary}")
        
        # Technical Skills
        tech_skills = resume_json.get('skills', {}).get('technical', [])
        if tech_skills:
            parts.append(f"Technical Skills: {', '.join(tech_skills)}")
        
        # Soft Skills
        soft_skills = resume_json.get('skills', {}).get('soft', [])
        if soft_skills:
            parts.append(f"Soft Skills: {', '.join(soft_skills)}")
        
        # Work Experience
        work_experiences = []
        for work in resume_json.get('work', []):
            company = work.get('company', '')
            position = work.get('position', '')
            summary = work.get('summary', '')
            work_experiences.append(f"{position} at {company}: {summary}")
        
        if work_experiences:
            parts.append(f"Experience: {' | '.join(work_experiences)}")
        
        # Education
        education_list = []
        for edu in resume_json.get('education', []):
            degree = edu.get('studyType', '')
            field = edu.get('area', '')
            institution = edu.get('institution', '')
            education_list.append(f"{degree} in {field} from {institution}")
        
        if education_list:
            parts.append(f"Education: {' | '.join(education_list)}")
        
        # Certifications
        certs = []
        for cert in resume_json.get('certifications', []):
            cert_name = cert.get('name', '')
            issuer = cert.get('issuer', '')
            certs.append(f"{cert_name} ({issuer})")
        
        if certs:
            parts.append(f"Certifications: {', '.join(certs)}")
        
        # Join all parts
        text = ' | '.join(parts)
        return text
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector from text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding, using default text")
            text = "No information available"
        
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def process_resume(self, resume_json: dict) -> Tuple[str, List[float]]:
        """
        Process resume JSON and generate embedding
        
        Args:
            resume_json: Parsed resume data
            
        Returns:
            Tuple of (text_used_for_embedding, embedding_vector)
        """
        text = self.prepare_text(resume_json)
        embedding = self.generate_embedding(text)
        
        logger.debug(f"Generated embedding of dimension {len(embedding)}")
        return text, embedding
    
    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch (more efficient)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]
