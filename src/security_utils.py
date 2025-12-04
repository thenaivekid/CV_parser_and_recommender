"""
Security Utilities for CV Parser
Implements defenses against adversarial attacks including:
- LLM prompt injection
- Embedding poisoning
- Keyword stuffing
- Data validation
"""
import re
import logging
from typing import Dict, Any, List, Tuple
from collections import Counter
from datetime import datetime
import unicodedata

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Validates and sanitizes resume data to prevent adversarial attacks"""
    
    # Maximum allowed lengths
    MAX_RESUME_WORDS = 1000
    MAX_RESUME_CHARS = 10000
    MAX_FIELD_LENGTH = 5000
    MAX_SKILLS_COUNT = 50
    MAX_WORK_ENTRIES = 15
    MAX_EDUCATION_ENTRIES = 10
    
    # Suspicious patterns for prompt injection
    INJECTION_PATTERNS = [
        r'(?i)(ignore|disregard|forget|override)\s+(all|any|previous|prior|above|earlier)\s+(instruction|prompt|rule|direction|command)',
        r'(?i)system\s*(prompt|instruction|override|message|role)',
        r'(?i)(new|updated|revised|different)\s*(instruction|prompt|rule|system|role)',
        r'(?i)(high|top|critical|urgent)\s*priority\s*(resume|candidate|application)',
        r'(?i)you\s+(are|must|should|will)\s+(now|be|become)\s+(an?|the)\s*(expert|assistant|parser|system)',
        r'(?i)(act|behave|pretend|respond)\s+(as|like)\s+(if|an?)',
        r'(?i)(execute|run|perform)\s+(this|these|the|following)\s*(command|instruction|task)',
        r'(?i)(extract|set|assign|mark)\s+(all|every|each)\s+(skill|experience|education)',
        r'(?i)(PhD|Ph\.?D\.?|doctorate|master).*(MIT|Stanford|Harvard|Oxford|Cambridge|Berkeley)',
        r'(?i)15\+?\s*years',
        r'(?i)<\s*(RESUME|SYSTEM|INSTRUCTION|PROMPT)\s*>',
        r'(?i)</\s*(RESUME|SYSTEM|INSTRUCTION|PROMPT)\s*>',
    ]
    
    @classmethod
    def sanitize_resume_text(cls, text: str) -> Tuple[str, List[str]]:
        """
        Sanitize resume text before LLM processing
        
        Args:
            text: Raw resume text
            
        Returns:
            Tuple of (sanitized_text, detected_threats)
        """
        detected_threats = []
        
        # Detect injection attempts
        for pattern in cls.INJECTION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_threats.append(f"Injection pattern: {pattern[:50]}...")
                logger.warning(f"Detected potential prompt injection: {matches[0][:100]}")
        
        # Remove injection patterns (redact, don't completely remove to preserve context)
        cleaned_text = text
        for pattern in cls.INJECTION_PATTERNS:
            cleaned_text = re.sub(pattern, '[REDACTED]', cleaned_text, flags=re.IGNORECASE)
        
        # Remove zero-width and invisible characters
        cleaned_text = cls._remove_invisible_chars(cleaned_text)
        
        # Detect and flag excessive repetition
        repetition_score = cls._detect_keyword_stuffing(cleaned_text)
        if repetition_score > 0.3:  # >30% repetition
            detected_threats.append(f"Keyword stuffing detected: {repetition_score:.2%}")
            logger.warning(f"High keyword repetition detected: {repetition_score:.2%}")
        
        # Word limit enforcement
        words = cleaned_text.split()
        if len(words) > cls.MAX_RESUME_WORDS:
            detected_threats.append(f"Excessive length: {len(words)} words (max: {cls.MAX_RESUME_WORDS})")
            logger.warning(f"Resume exceeds word limit: {len(words)} words")
            cleaned_text = ' '.join(words[:cls.MAX_RESUME_WORDS]) + "\n[TRUNCATED]"
        
        # Character limit enforcement
        if len(cleaned_text) > cls.MAX_RESUME_CHARS:
            detected_threats.append(f"Excessive chars: {len(cleaned_text)} (max: {cls.MAX_RESUME_CHARS})")
            cleaned_text = cleaned_text[:cls.MAX_RESUME_CHARS] + "\n[TRUNCATED]"
        
        # Normalize whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
        
        return cleaned_text.strip(), detected_threats
    
    @staticmethod
    def _remove_invisible_chars(text: str) -> str:
        """Remove zero-width and invisible Unicode characters"""
        # Zero-width characters
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff\u180e]', '', text)
        
        # Remove other invisible/control characters except common whitespace
        cleaned = []
        for char in text:
            cat = unicodedata.category(char)
            # Keep letters, numbers, punctuation, space, tab, newline
            if cat[0] not in ['C'] or char in [' ', '\t', '\n', '\r']:
                cleaned.append(char)
        
        return ''.join(cleaned)
    
    @staticmethod
    def _detect_keyword_stuffing(text: str) -> float:
        """
        Detect keyword stuffing by analyzing word repetition
        
        Returns:
            Repetition score (0.0 = no repetition, 1.0 = extreme repetition)
        """
        words = [w.lower() for w in text.split() if len(w) > 3]  # Ignore short words
        if len(words) < 10:
            return 0.0
        
        word_counts = Counter(words)
        
        # Calculate repetition score
        max_count = max(word_counts.values()) if word_counts else 0
        unique_words = len(word_counts)
        total_words = len(words)
        
        # High max_count or low uniqueness indicates stuffing
        max_repetition = max_count / total_words if total_words > 0 else 0
        uniqueness = unique_words / total_words if total_words > 0 else 1
        
        # Combined score
        repetition_score = (max_repetition * 0.7) + ((1 - uniqueness) * 0.3)
        
        return repetition_score
    
    @classmethod
    def validate_parsed_output(cls, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate parsed resume data for anomalies
        
        Args:
            data: Parsed resume data dictionary
            
        Returns:
            Tuple of (validated_data, anomalies_detected)
        """
        anomalies = []
        validated_data = data.copy()
        
        # Validate skills count
        tech_skills = data.get('skills', {}).get('technical', [])
        soft_skills = data.get('skills', {}).get('soft', [])
        
        if len(tech_skills) > cls.MAX_SKILLS_COUNT:
            anomalies.append(f"Excessive technical skills: {len(tech_skills)} (max: {cls.MAX_SKILLS_COUNT})")
            logger.warning(f"Capping technical skills from {len(tech_skills)} to {cls.MAX_SKILLS_COUNT}")
            validated_data['skills']['technical'] = tech_skills[:cls.MAX_SKILLS_COUNT]
        
        if len(soft_skills) > cls.MAX_SKILLS_COUNT:
            anomalies.append(f"Excessive soft skills: {len(soft_skills)} (max: {cls.MAX_SKILLS_COUNT})")
            logger.warning(f"Capping soft skills from {len(soft_skills)} to {cls.MAX_SKILLS_COUNT}")
            validated_data['skills']['soft'] = soft_skills[:cls.MAX_SKILLS_COUNT]
        
        # Deduplicate skills
        original_tech_count = len(tech_skills)
        deduped_tech = cls._deduplicate_skills(tech_skills)
        if len(deduped_tech) < original_tech_count:
            anomalies.append(f"Duplicate skills removed: {original_tech_count - len(deduped_tech)}")
            validated_data['skills']['technical'] = deduped_tech
        
        # Validate work experience
        work = data.get('work', [])
        if len(work) > cls.MAX_WORK_ENTRIES:
            anomalies.append(f"Excessive work entries: {len(work)} (max: {cls.MAX_WORK_ENTRIES})")
            validated_data['work'] = work[:cls.MAX_WORK_ENTRIES]
        
        # Calculate total experience
        total_years = cls._calculate_experience_years(work)
        if total_years > 50:
            anomalies.append(f"Unrealistic experience: {total_years} years")
            logger.error(f"Suspicious: {total_years} years of experience claimed")
        
        # Detect overlapping work periods
        overlaps = cls._detect_overlapping_work(work)
        if overlaps:
            anomalies.append(f"Overlapping work periods detected: {len(overlaps)} overlaps")
            logger.warning(f"Detected {len(overlaps)} overlapping work periods")
        
        # Validate education
        education = data.get('education', [])
        if len(education) > cls.MAX_EDUCATION_ENTRIES:
            anomalies.append(f"Excessive education entries: {len(education)}")
            validated_data['education'] = education[:cls.MAX_EDUCATION_ENTRIES]
        
        # Flag prestigious institutions (requires manual verification)
        prestigious = ['MIT', 'Stanford', 'Harvard', 'Oxford', 'Cambridge', 'Berkeley', 'Yale', 'Princeton']
        for edu in education:
            institution = edu.get('institution', '')
            if any(univ in institution for univ in prestigious):
                anomalies.append(f"Prestigious institution claimed: {institution}")
                logger.info(f"Flagged for verification: {institution}")
        
        # Validate field lengths
        for field_name, max_len in [('summary', cls.MAX_FIELD_LENGTH)]:
            field_value = data.get(field_name, '')
            if isinstance(field_value, str) and len(field_value) > max_len:
                anomalies.append(f"Field '{field_name}' exceeds max length")
                validated_data[field_name] = field_value[:max_len]
        
        return validated_data, anomalies
    
    @staticmethod
    def _deduplicate_skills(skills: List[str]) -> List[str]:
        """Remove duplicate and similar skills"""
        from difflib import SequenceMatcher
        
        unique_skills = []
        for skill in skills:
            is_duplicate = False
            skill_lower = skill.lower().strip()
            
            for existing in unique_skills:
                existing_lower = existing.lower().strip()
                
                # Exact match
                if skill_lower == existing_lower:
                    is_duplicate = True
                    break
                
                # Fuzzy match (>85% similar)
                similarity = SequenceMatcher(None, skill_lower, existing_lower).ratio()
                if similarity > 0.85:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_skills.append(skill)
        
        return unique_skills
    
    @staticmethod
    def _calculate_experience_years(work_list: List[Dict[str, Any]]) -> float:
        """Calculate total years of work experience"""
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        
        total_years = 0.0
        
        for work in work_list:
            start_date = work.get('startDate', '')
            end_date = work.get('endDate', 'Present')
            
            if not start_date:
                continue
            
            try:
                # Parse dates (handle YYYY-MM or YYYY-MM-DD formats)
                if len(start_date) == 7:  # YYYY-MM
                    start = datetime.strptime(start_date, '%Y-%m')
                else:
                    start = datetime.strptime(start_date, '%Y-%m-%d')
                
                if end_date and end_date.lower() != 'present':
                    if len(end_date) == 7:
                        end = datetime.strptime(end_date, '%Y-%m')
                    else:
                        end = datetime.strptime(end_date, '%Y-%m-%d')
                else:
                    end = datetime.now()
                
                delta = relativedelta(end, start)
                years = delta.years + delta.months / 12.0
                total_years += years
                
            except Exception as e:
                logger.debug(f"Error parsing dates: {e}")
                # Fallback: assume 2 years per position
                total_years += 2.0
        
        return total_years
    
    @staticmethod
    def _detect_overlapping_work(work_list: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """Detect overlapping work periods"""
        from datetime import datetime
        
        overlaps = []
        
        for i, work1 in enumerate(work_list):
            for j, work2 in enumerate(work_list[i+1:], start=i+1):
                try:
                    start1 = work1.get('startDate', '')
                    end1 = work1.get('endDate', 'Present')
                    start2 = work2.get('startDate', '')
                    end2 = work2.get('endDate', 'Present')
                    
                    if not start1 or not start2:
                        continue
                    
                    # Parse dates
                    s1 = datetime.strptime(start1[:7], '%Y-%m')
                    s2 = datetime.strptime(start2[:7], '%Y-%m')
                    
                    if end1.lower() == 'present':
                        e1 = datetime.now()
                    else:
                        e1 = datetime.strptime(end1[:7], '%Y-%m')
                    
                    if end2.lower() == 'present':
                        e2 = datetime.now()
                    else:
                        e2 = datetime.strptime(end2[:7], '%Y-%m')
                    
                    # Check for overlap
                    if (s1 <= s2 <= e1) or (s1 <= e2 <= e1) or (s2 <= s1 <= e2):
                        overlaps.append((i, j))
                
                except Exception:
                    pass
        
        return overlaps
    
    @classmethod
    def create_security_report(
        cls, 
        candidate_id: str,
        threats: List[str],
        anomalies: List[str]
    ) -> Dict[str, Any]:
        """
        Create a security report for suspicious resumes
        
        Args:
            candidate_id: Candidate identifier
            threats: List of detected threats
            anomalies: List of detected anomalies
            
        Returns:
            Security report dictionary
        """
        severity = cls._calculate_severity(threats, anomalies)
        
        return {
            'candidate_id': candidate_id,
            'threats_detected': threats,
            'anomalies_detected': anomalies,
            'threat_count': len(threats),
            'anomaly_count': len(anomalies),
            'severity': severity,
            'requires_manual_review': severity in ['high', 'critical'],
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def _calculate_severity(threats: List[str], anomalies: List[str]) -> str:
        """Calculate severity level based on threats and anomalies"""
        threat_count = len(threats)
        anomaly_count = len(anomalies)
        
        # Check for critical patterns
        critical_keywords = ['injection', 'stuffing', 'unrealistic', 'excessive']
        critical_count = sum(1 for t in threats + anomalies 
                            if any(k in t.lower() for k in critical_keywords))
        
        if critical_count >= 3 or threat_count >= 2:
            return 'critical'
        elif critical_count >= 2 or threat_count >= 1 or anomaly_count >= 5:
            return 'high'
        elif anomaly_count >= 3:
            return 'medium'
        elif anomaly_count >= 1:
            return 'low'
        else:
            return 'none'
