"""
Embedding Generator for CV Parser and Recommender System
Uses SentenceTransformer to generate embeddings from parsed resume JSON
"""
import logging
import re
from typing import Tuple, List
from collections import Counter
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer

from src.performance_monitor import track_performance

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
        
        # Skill normalization mappings
        self.skill_synonyms = {
            'python': ['python3', 'python 3.x', 'python 2.x'],
            'javascript': ['js', 'ecmascript', 'es6', 'es2015'],
            'aws': ['amazon web services', 'amazon aws'],
            'gcp': ['google cloud platform', 'google cloud'],
            'docker': ['docker container', 'docker engine'],
            'kubernetes': ['k8s', 'k8'],
            'machine learning': ['ml', 'machine-learning'],
            'artificial intelligence': ['ai'],
            'react': ['reactjs', 'react.js'],
            'node': ['nodejs', 'node.js'],
            'postgresql': ['postgres', 'psql'],
            'mongodb': ['mongo', 'mongo db'],
        }
    
    def normalize_text_for_embedding(self, text: str) -> str:
        """
        Normalize and clean text before embedding to prevent manipulation
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Normalized text
        """
        # Remove zero-width and invisible characters
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Detect and cap repetition (keyword stuffing)
        words = text.split()
        if len(words) > 10:
            word_counts = Counter(w.lower() for w in words)
            
            # Flag if any word appears > 10 times
            max_repetitions = max(word_counts.values()) if word_counts else 0
            if max_repetitions > 10:
                logger.warning(f"Keyword stuffing in embedding text: word repeated {max_repetitions}x")
                # Deduplicate excessive repetition
                seen = set()
                deduped_words = []
                for word in words:
                    word_lower = word.lower()
                    # Allow common words to repeat, but cap technical terms
                    common_words = {'the', 'a', 'an', 'and', 'or', 'in', 'at', 'to', 'for', 'of', 'with', 'on'}
                    if word_lower in common_words or word_lower not in seen:
                        deduped_words.append(word)
                        seen.add(word_lower)
                text = ' '.join(deduped_words)
        
        return text.strip()
    
    def normalize_skill(self, skill: str) -> str:
        """
        Normalize skill to canonical form (lowercase, handle synonyms)
        
        Args:
            skill: Raw skill string
            
        Returns:
            Normalized skill
        """
        skill_lower = skill.lower().strip()
        
        # Remove special characters but keep spaces and hyphens
        skill_lower = re.sub(r'[^a-z0-9\s\-+#]', '', skill_lower)
        
        # Check if it's a known synonym
        for canonical, synonyms in self.skill_synonyms.items():
            if skill_lower in synonyms or skill_lower == canonical:
                return canonical
        
        return skill_lower
    
    def deduplicate_skills(self, skills: List[str]) -> List[str]:
        """
        Remove duplicate and similar skills using fuzzy matching
        
        Args:
            skills: List of skills
            
        Returns:
            Deduplicated list (max 50 skills)
        """
        if not skills:
            return []
        
        # First pass: normalize to canonical forms
        normalized_map = {}
        for skill in skills:
            normalized = self.normalize_skill(skill)
            if normalized not in normalized_map:
                normalized_map[normalized] = skill  # Keep original casing for first occurrence
        
        # Second pass: fuzzy deduplication
        unique_skills = []
        normalized_list = list(normalized_map.keys())
        
        for i, norm_skill in enumerate(normalized_list):
            is_duplicate = False
            
            # Check against already selected skills
            for existing_norm in unique_skills:
                # Use SequenceMatcher for fuzzy matching
                similarity = SequenceMatcher(None, norm_skill, existing_norm).ratio()
                
                # 80% similarity threshold
                if similarity > 0.8:
                    is_duplicate = True
                    logger.debug(f"Skill '{norm_skill}' is {similarity:.2%} similar to '{existing_norm}', deduplicating")
                    break
            
            if not is_duplicate:
                unique_skills.append(norm_skill)
        
        # Cap at 50 skills to prevent manipulation
        if len(unique_skills) > 50:
            logger.warning(f"Capping skills from {len(unique_skills)} to 50 for embedding")
            unique_skills = unique_skills[:50]
        
        return unique_skills
    
    def prepare_text(self, resume_json: dict) -> str:
        """
        Concatenate relevant fields from resume JSON for embedding with normalization
        
        Args:
            resume_json: Parsed resume data
            
        Returns:
            Concatenated and normalized text string for embedding
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
        
        # Technical Skills - Deduplicate and normalize
        tech_skills = resume_json.get('skills', {}).get('technical', [])
        if tech_skills:
            deduped_tech = self.deduplicate_skills(tech_skills)
            if len(deduped_tech) != len(tech_skills):
                logger.info(f"Deduplicated technical skills: {len(tech_skills)} -> {len(deduped_tech)}")
            parts.append(f"Technical Skills: {', '.join(deduped_tech)}")
        
        # Soft Skills - Deduplicate (less aggressive normalization)
        soft_skills = resume_json.get('skills', {}).get('soft', [])
        if soft_skills:
            # Remove exact duplicates only for soft skills
            unique_soft = list(dict.fromkeys([s.lower().strip() for s in soft_skills]))[:50]
            parts.append(f"Soft Skills: {', '.join(unique_soft)}")
        
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
        
        # Normalize the final text to prevent embedding manipulation
        normalized_text = self.normalize_text_for_embedding(text)
        
        # Log if significant changes were made
        if len(normalized_text) < len(text) * 0.8:
            logger.warning(f"Embedding text reduced by {100 - len(normalized_text)/len(text)*100:.1f}% after normalization")
        
        return normalized_text
    
    @track_performance('embedding_generation')
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
