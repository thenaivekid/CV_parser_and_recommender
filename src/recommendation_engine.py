"""
Recommendation Engine for CV Parser and Recommender System
Matches candidates with job vacancies based on multiple factors
"""
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta

from src.performance_monitor import track_performance, track_time

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Generate job recommendations for candidates based on:
    - Skills match (technical and soft skills)
    - Experience match (years and relevance)
    - Education match (degree and field)
    - Semantic similarity (vector embeddings)
    
    Supports TWO-STAGE RETRIEVAL for efficient large-scale recommendations.
    """
    
    def __init__(
        self, 
        weights: Optional[Dict[str, float]] = None,
        use_two_stage: bool = True,
        stage1_top_k: int = 50,
        stage1_threshold: float = 0.3
    ):
        """
        Initialize recommendation engine with configurable weights
        
        Args:
            weights: Dictionary of weights for different matching factors
                     Default: {'skills': 0.35, 'experience': 0.25, 'education': 0.15, 'semantic': 0.25}
            use_two_stage: Enable two-stage retrieval (Stage 1: vector filtering, Stage 2: full scoring)
            stage1_top_k: Number of candidates to retrieve in Stage 1 (default: 50)
            stage1_threshold: Minimum similarity threshold for Stage 1 (default: 0.3)
        """
        self.weights = weights or {
            'skills': 0.35,
            'experience': 0.25,
            'education': 0.15,
            'semantic': 0.25
        } # TODO: tune the weights
        
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        # Two-stage retrieval configuration
        self.use_two_stage = use_two_stage
        self.stage1_top_k = stage1_top_k
        self.stage1_threshold = stage1_threshold
        
        logger.info(
            f"Recommendation engine initialized with weights: {self.weights}, "
            f"two-stage: {use_two_stage}, stage1_k: {stage1_top_k}, threshold: {stage1_threshold}"
        )
    
    def calculate_skills_match(
        self, 
        candidate_tech_skills: List[str],
        candidate_soft_skills: List[str],
        job_tech_skills: List[str],
        job_soft_skills: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """
        TODO: Implement more robust skills matching logic that is not sensitive to slight variations in skill naming.
        Calculate skills match score
        
        Args:
            candidate_tech_skills: Candidate's technical skills
            candidate_soft_skills: Candidate's soft skills
            job_tech_skills: Job's required technical skills
            job_soft_skills: Job's required soft skills
            
        Returns:
            Tuple of (match_score, matched_skills, missing_skills)
        """
        # Normalize skills to lowercase for comparison
        cand_tech = set([s.lower().strip() for s in candidate_tech_skills])
        cand_soft = set([s.lower().strip() for s in candidate_soft_skills])
        job_tech = set([s.lower().strip() for s in job_tech_skills])
        job_soft = set([s.lower().strip() for s in job_soft_skills])
        
        # Calculate technical skills match (weighted 70%)
        if job_tech:
            matched_tech = cand_tech.intersection(job_tech)
            tech_score = len(matched_tech) / len(job_tech)
        else:
            matched_tech = set()
            tech_score = 1.0  # No tech requirements means perfect match
        
        # Calculate soft skills match (weighted 30%)
        if job_soft:
            matched_soft = cand_soft.intersection(job_soft)
            soft_score = len(matched_soft) / len(job_soft)
        else:
            matched_soft = set()
            soft_score = 1.0  # No soft requirements means perfect match
        
        # Combine scores
        skills_score = 0.7 * tech_score + 0.3 * soft_score
        
        # Find matched and missing skills (preserve original casing)
        matched_skills = []
        missing_skills = []
        
        for skill in job_tech_skills:
            if skill.lower().strip() in matched_tech:
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)
        
        for skill in job_soft_skills:
            if skill.lower().strip() in matched_soft:
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)
        
        return skills_score, matched_skills, missing_skills
    
    def calculate_experience_match(
        self,
        candidate_work_experience: List[Dict[str, Any]],
        job_exp_min: Optional[int],
        job_exp_max: Optional[int],
        job_seniority: str
    ) -> Tuple[float, float]:
        """
        Calculate experience match score
        
        Args:
            candidate_work_experience: List of work experience entries
            job_exp_min: Minimum years of experience required
            job_exp_max: Maximum years of experience required
            job_seniority: Seniority level (entry, mid, senior, lead, etc.)
            
        Returns:
            Tuple of (experience_match_score, total_years_of_experience)
        """
        # NOTE: If no experience data, assume 0 years
        total_years = 0.0
        
        for work in candidate_work_experience:
            start_date = work.get('startDate')
            end_date = work.get('endDate', 'Present')
            
            if start_date:
                try:
                    # Parse dates
                    start = datetime.strptime(start_date, '%Y-%m-%d')
                    if end_date and end_date.lower() != 'present':
                        end = datetime.strptime(end_date, '%Y-%m-%d')
                    else:
                        end = datetime.now()
                    
                    # Calculate years
                    delta = relativedelta(end, start)
                    years = delta.years + delta.months / 12.0
                    total_years += years
                except:
                    # Fallback: assume 2 years per position if dates are unclear
                    total_years += 2.0
        
        # Fallback if no dates but has experience entries
        if total_years == 0 and candidate_work_experience:
            total_years = len(candidate_work_experience) * 2.0
        
        # Score based on experience range
        if job_exp_min is None and job_exp_max is None:
            # No experience requirements specified
            return 1.0, total_years
        
        if job_exp_min is not None and job_exp_max is not None:
            # Range specified
            if job_exp_min <= total_years <= job_exp_max:
                return 1.0, total_years
            elif total_years < job_exp_min:
                # Underqualified: gradual penalty
                diff = job_exp_min - total_years
                return max(0.0, 1.0 - (diff * 0.15)), total_years  # -15% per year short
            else:
                # Overqualified: treat as qualified
                diff = total_years - job_exp_max
                return max(0.8, 1.0 - (diff * 0.05)), total_years  # -5% per year over, min 0.8
        
        elif job_exp_min is not None:
            # Only minimum specified
            if total_years >= job_exp_min:
                return 1.0, total_years
            else:
                diff = job_exp_min - total_years
                return max(0.0, 1.0 - (diff * 0.15)), total_years
        
        else:
            # Only maximum specified (rare)
            if total_years <= job_exp_max:
                return 1.0, total_years
            else:
                diff = total_years - job_exp_max
                return max(0.5, 1.0 - (diff * 0.05)), total_years # penalty for overqualified
    
    def calculate_education_match(
        self,
        candidate_education: List[Dict[str, Any]],
        job_education_required: str,
        job_education_field: str
    ) -> float:
        """
        Calculate education match score
        
        Args:
            candidate_education: List of education entries
            job_education_required: Required degree level
            job_education_field: Required field of study
            
        Returns:
            Education match score (0.0 to 1.0)
        """
        if not job_education_required and not job_education_field:
            return 1.0  # No education requirements
        
        if not candidate_education:
            return 0.0  # Job requires education but candidate has none listed
        
        # Define education hierarchy
        edu_hierarchy = {
            'high school': 1,
            'diploma': 2,
            'associate': 3,
            'bachelor': 4,
            'master': 5,
            'phd': 6,
            'doctorate': 6
        }
        
        # Find candidate's highest education level
        candidate_highest = 0
        field_match = False
        
        for edu in candidate_education:
            degree = edu.get('studyType', '').lower()
            field = edu.get('area', '').lower()
            
            # Check degree level
            for key, level in edu_hierarchy.items():
                if key in degree:
                    candidate_highest = max(candidate_highest, level)
                    break
            
            # Check field match
            if job_education_field and field:
                if job_education_field.lower() in field or field in job_education_field.lower():
                    field_match = True
        
        # Calculate degree level score
        job_edu_level = 0
        if job_education_required:
            for key, level in edu_hierarchy.items():
                if key in job_education_required.lower():
                    job_edu_level = level
                    break
        
        # Degree level score (70% weight)
        if job_edu_level == 0:
            degree_score = 1.0  # No specific degree required
        elif candidate_highest >= job_edu_level:
            degree_score = 1.0  # Meets or exceeds requirements
        elif candidate_highest == 0:
            degree_score = 0.3  # No formal education listed
        else:
            # Underqualified: partial credit
            degree_score = 0.5 + (candidate_highest / job_edu_level) * 0.3
        
        # Field match score (30% weight)
        if not job_education_field:
            field_score = 1.0  # No field requirement
        elif field_match:
            field_score = 1.0
        else:
            field_score = 0.5  # Different field, partial credit
        
        return 0.7 * degree_score + 0.3 * field_score # TODO: tune the weights
    
    def calculate_semantic_similarity(
        self,
        candidate_embedding: List[float],
        job_embedding: List[float]
    ) -> float:
        """
        Calculate cosine similarity between candidate and job embeddings
        
        Args:
            candidate_embedding: Candidate's embedding vector
            job_embedding: Job's embedding vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not candidate_embedding or not job_embedding:
            logger.warning("Missing embeddings for similarity calculation")
            return 0.5  # Neutral score if embeddings missing
        
        # Convert to numpy arrays
        cand_vec = np.array(candidate_embedding)
        job_vec = np.array(job_embedding)
        
        # Calculate cosine similarity
        dot_product = np.dot(cand_vec, job_vec)
        norm_cand = np.linalg.norm(cand_vec)
        norm_job = np.linalg.norm(job_vec)
        
        if norm_cand == 0 or norm_job == 0:
            return 0.0
        
        similarity = dot_product / (norm_cand * norm_job)
        
        # Normalize to 0-1 range (cosine similarity is -1 to 1)
        normalized_similarity = (similarity + 1) / 2
        
        return float(normalized_similarity)
    
    def calculate_overall_match(
        self,
        skills_score: float,
        experience_score: float,
        education_score: float,
        semantic_score: float
    ) -> float:
        """
        Calculate weighted overall match score
        
        Args:
            skills_score: Skills match score
            experience_score: Experience match score
            education_score: Education match score
            semantic_score: Semantic similarity score
            
        Returns:
            Overall match score (0.0 to 1.0)
        """
        overall = (
            self.weights['skills'] * skills_score +
            self.weights['experience'] * experience_score +
            self.weights['education'] * education_score +
            self.weights['semantic'] * semantic_score
        )
        
        return overall
    
    def generate_explanation(
        self,
        candidate_name: str,
        job_title: str,
        skills_score: float,
        experience_score: float,
        education_score: float,
        semantic_score: float,
        matched_skills: List[str],
        missing_skills: List[str],
        candidate_years_exp: float,
        job_exp_min: Optional[int],
        job_exp_max: Optional[int]
    ) -> str:
        """
        Generate human-readable explanation for the match
        
        Args:
            Various match factors and scores
            
        Returns:
            Explanation string
        """
        explanations = []
        
        # Skills explanation
        if skills_score >= 0.8:
            explanations.append(f"Excellent skills match with {len(matched_skills)} matching skills")
        elif skills_score >= 0.6:
            explanations.append(f"Good skills alignment with {len(matched_skills)} relevant skills")
        elif skills_score >= 0.4:
            explanations.append(f"Partial skills match with {len(matched_skills)} matching skills")
        else:
            explanations.append(f"Limited skills overlap, missing key requirements")
        
        # Experience explanation
        if experience_score >= 0.9:
            explanations.append(f"Perfect experience fit with {candidate_years_exp:.1f} years")
        elif experience_score >= 0.7:
            explanations.append(f"Strong experience match with {candidate_years_exp:.1f} years")
        elif experience_score >= 0.5:
            explanations.append(f"Adequate experience level ({candidate_years_exp:.1f} years)")
        else:
            if job_exp_min and candidate_years_exp < job_exp_min:
                explanations.append(f"Below required experience ({candidate_years_exp:.1f} vs {job_exp_min}+ years)")
            else:
                explanations.append(f"Experience level may not align with role requirements")
        
        # Education explanation
        if education_score >= 0.8:
            explanations.append("Meets education requirements")
        elif education_score >= 0.6:
            explanations.append("Education partially meets requirements")
        
        # Semantic explanation
        if semantic_score >= 0.8:
            explanations.append("Strong overall profile alignment")
        elif semantic_score >= 0.6:
            explanations.append("Good overall fit based on profile analysis")
        
        return " | ".join(explanations)
    
    @track_performance('recommendation_generation')
    def rank_jobs_for_candidate(
        self,
        candidate: Dict[str, Any],
        jobs_with_similarity: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Rank all jobs for a candidate and return recommendations.
        OPTIMIZED: Expects jobs with pre-computed semantic_similarity from database.
        
        Args:
            candidate: Candidate data dictionary
            jobs_with_similarity: List of job dictionaries WITH 'semantic_similarity' field
                                  (pre-computed by database using pgvector)
            top_k: Number of top recommendations to return (None = all)
            
        Returns:
            Recommendations dictionary with ranked jobs
        """
        candidate_id = candidate['candidate_id']
        candidate_name = candidate.get('name', 'Unknown')
        
        recommendations = []
        
        for job in jobs_with_similarity:
            job_id = job['job_id']
            
            # Get pre-computed semantic similarity from database (already normalized 0-1)
            semantic_score = float(job.get('semantic_similarity', 0.5))
            
            # Calculate individual match scores (skills, experience, education in Python)
            skills_score, matched_skills, missing_skills = self.calculate_skills_match(
                candidate.get('skills_technical', []) or [],
                candidate.get('skills_soft', []) or [],
                job.get('skills_technical', []) or [],
                job.get('skills_soft', []) or []
            )
            
            experience_score, candidate_years_exp = self.calculate_experience_match(
                candidate.get('work_experience', []) or [],
                job.get('experience_years_min'),
                job.get('experience_years_max'),
                job.get('seniority_level', '')
            )
            
            education_score = self.calculate_education_match(
                candidate.get('education', []) or [],
                job.get('education_required', ''),
                job.get('education_field', '')
            )
            
            # Calculate overall match score using pre-computed semantic similarity
            match_score = self.calculate_overall_match(
                skills_score,
                experience_score,
                education_score,
                semantic_score
            )
            
            # Generate explanation
            explanation = self.generate_explanation(
                candidate_name,
                job.get('job_title', ''),
                skills_score,
                experience_score,
                education_score,
                semantic_score,
                matched_skills,
                missing_skills,
                candidate_years_exp,
                job.get('experience_years_min'),
                job.get('experience_years_max')
            )
            
            # Create recommendation entry
            recommendation = {
                'job_id': job_id,
                'job_title': job.get('job_title', ''),
                'company': job.get('company', ''),
                'match_score': round(match_score, 4),
                'matching_factors': {
                    'skills_match': round(skills_score, 4),
                    'experience_match': round(experience_score, 4),
                    'education_match': round(education_score, 4),
                    'semantic_similarity': round(semantic_score, 4)
                },
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
                'explanation': explanation
            }
            
            recommendations.append(recommendation)
        
        # Sort by match score (descending)
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Limit to top_k if specified
        if top_k:
            recommendations = recommendations[:top_k]
        
        return {
            'candidate_id': candidate_id,
            'candidate_name': candidate_name,
            'recommendations': recommendations,
            'total_jobs_evaluated': len(jobs_with_similarity),
            'generated_at': datetime.now().isoformat()
        }
