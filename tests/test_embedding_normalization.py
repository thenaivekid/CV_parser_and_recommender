#!/usr/bin/env python3
"""
Test embedding normalization and skill deduplication
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding_generator import EmbeddingGenerator

def test_skill_deduplication():
    """Test skill deduplication with synonyms and fuzzy matching"""
    print("="*60)
    print("TEST: Skill Deduplication")
    print("="*60)
    
    emb_gen = EmbeddingGenerator()
    
    # Test case: Many variations of same skills
    skills = [
        "Python", "python", "PYTHON", "Python3", "Python 3.x",
        "AWS", "aws", "Amazon Web Services", "Amazon AWS",
        "Docker", "docker", "DOCKER", "Docker Container",
        "JavaScript", "JS", "js", "ECMAScript",
        "React", "ReactJS", "React.js",
        "Machine Learning", "ML", "machine-learning"
    ]
    
    print(f"\nInput skills ({len(skills)}):")
    for skill in skills:
        print(f"  - {skill}")
    
    deduped = emb_gen.deduplicate_skills(skills)
    
    print(f"\nDeduplicated skills ({len(deduped)}):")
    for skill in deduped:
        print(f"  - {skill}")
    
    print(f"\nReduction: {len(skills)} -> {len(deduped)} ({100-len(deduped)/len(skills)*100:.1f}% reduction)")
    print()

def test_keyword_stuffing():
    """Test keyword stuffing detection"""
    print("="*60)
    print("TEST: Keyword Stuffing Detection")
    print("="*60)
    
    emb_gen = EmbeddingGenerator()
    
    # Normal text
    normal_text = "Experienced software engineer with Python, JavaScript, and Docker skills"
    print("\nNormal text:")
    print(f"  Input:  {normal_text}")
    normalized = emb_gen.normalize_text_for_embedding(normal_text)
    print(f"  Output: {normalized}")
    print(f"  Changed: {'No' if normal_text == normalized else 'Yes'}")
    
    # Stuffed text
    stuffed_text = "Python " * 20 + "JavaScript " * 20 + "Docker " * 20
    print(f"\nStuffed text ({len(stuffed_text.split())} words):")
    print(f"  Input:  {stuffed_text[:100]}...")
    normalized = emb_gen.normalize_text_for_embedding(stuffed_text)
    print(f"  Output: {normalized[:100]}...")
    print(f"  Reduced: {len(stuffed_text.split())} words -> {len(normalized.split())} words")
    print()

def test_resume_embedding():
    """Test full resume processing"""
    print("="*60)
    print("TEST: Full Resume Embedding with Normalization")
    print("="*60)
    
    emb_gen = EmbeddingGenerator()
    
    # Resume with duplicated skills
    resume = {
        "basics": {"name": "John Doe", "email": "john@test.com"},
        "summary": "Software engineer",
        "skills": {
            "technical": [
                "Python", "python", "Python3",
                "AWS", "Amazon Web Services",
                "Docker", "Docker Container",
                "JavaScript", "JS"
            ],
            "soft": ["Communication", "Teamwork", "communication"]
        },
        "work": [{
            "company": "Tech Corp",
            "position": "Engineer",
            "summary": "Developed applications"
        }],
        "education": [],
        "certifications": [],
        "achievements": [],
        "languages": []
    }
    
    print(f"\nInput technical skills ({len(resume['skills']['technical'])}):")
    for skill in resume['skills']['technical']:
        print(f"  - {skill}")
    
    text = emb_gen.prepare_text(resume)
    
    print(f"\nPrepared text for embedding:")
    print(f"  {text}")
    
    embedding = emb_gen.generate_embedding(text)
    print(f"\nEmbedding generated: {len(embedding)} dimensions")
    print(f"Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, ..., {embedding[-1]:.4f}]")
    print()

if __name__ == "__main__":
    test_skill_deduplication()
    test_keyword_stuffing()
    test_resume_embedding()
    
    print("="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
