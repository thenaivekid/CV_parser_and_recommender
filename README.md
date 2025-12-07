# CV parser and recommender

## Usage
**Needs Linux machine with git, python, docker and docker-compose installed.**
- If docker-compose not present else skip :
```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

```bash
git clone https://github.com/thenaivekid/CV_parser_and_recommender.git

cd CV_parser_and_recommender

#get the test resume dataset
curl -L -o resume-dataset.zip https://www.kaggle.com/api/v1/datasets/download/snehaanbhawal/resume-dataset
unzip resume-dataset.zip -d resume-dataset

# for llm api
cp .env.example .env
# get gemini api key and put it in .env file

# make all scripts executable
chmod +x ./scripts/*.sh

# create python env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# start postgresql docker container
./scripts/init_database.sh #for creating tables for first time
# ./scripts/restart_db.sh #after running init_database.sh once, keeps the data from earlier sessions

# install redis
./scripts/install_redis.sh

# run tests
./scripts/test_all.sh

# run cv parsing using test pdfs from dataset downloaded earlier
./scripts/run_cv_batch_processing.sh 

# run job processing
./scripts/run_job_processing.sh

# run generate recommendations
./scripts/run_recommendations.sh

# run evaluation pipeline to get qualitative evals of generated recommendations, using dummy test ground truth
./scripts/run_evaluation.sh

# generate dashboard to see performance metrics in terms of system throughput and so on(uses stored values during parsing and recommendation generation if nothing is running currently)
./scripts/generate_dashboard.sh # see ./data/performance_reports to see the html generated showing performace metrics

# run recommendation using 2 stage vs 1 stage comparision
./scripts/run_recommendation_benchmark.sh 


```

## Resume Parser

The resume parser intelligently extracts structured data from PDF CVs using LLM-powered NLP. It handles diverse CV layouts (traditional, modern, multi-column) and converts them into standardized JSON format with fields like work experience, education, skills, and certifications.

**Key Features:**
- **Multi-provider support**: Azure OpenAI and Google Gemini
- **Robust error handling**: Validates PDFs, handles corruption, provides detailed logging  
- **Format flexibility**: Parses various CV layouts and unconventional structures
- **Smart extraction**: Uses few-shot learning for accurate field detection
- **Retry logic**: Up to 3 attempts with fallback to ensure reliability
- **Data validation**: Cleans and validates extracted information
- **🛡️ Security hardening**: LLM prompt injection defense, adversarial attack protection

**Usage:**
```bash
python src/cv_parser.py path/to/resume.pdf -o output.json --provider gemini
```

## 🔐 Security Features

This system implements comprehensive defenses against adversarial attacks:

### Protections Implemented

#### 1. LLM Parsing Protection
- **Prompt Injection Defense**: 12 regex patterns detect injection attempts
- **Delimiter-Based Prompts**: `<RESUME>` tags isolate user content from instructions
- **Input Sanitization**: Removes invisible characters, limits word count (1000 words)
- **Output Validation**: Caps skills (max 50), detects unrealistic experience (>50 years)

#### 2. Embedding Model Protection (NEW)
Prevents manipulation of the embedding-based similarity matching:

- **Skill Synonym Normalization**: Maps variations to canonical forms
  ```
  "Python", "python", "Python3", "Python 3.x" → "python"
  "AWS", "Amazon Web Services", "Amazon AWS" → "aws"
  "Docker", "docker", "Docker Container" → "docker"
  ```
  
- **Fuzzy Skill Deduplication**: 80% similarity threshold removes duplicates
  ```
  Input: ["Python", "python", "PYTHON", "Python3", "AWS", "aws", "Amazon AWS", "Docker", "docker"]
  Output: ["python", "aws", "docker"]  (69.6% reduction)
  ```

- **Keyword Stuffing Detection in Embeddings**: Detects word repetition >10x
  ```
  Input:  "Python " × 20 + "JavaScript " × 20 + "Docker " × 20 = 60 words
  Output: "Python JavaScript Docker" = 3 words (deduplicates excessive repetition)
  ```

- **Skill Count Capping**: Maximum 50 technical skills per resume
- **Invisible Character Removal**: Strips zero-width and Unicode tricks
- **Case-Insensitive Normalization**: Treats "Python", "PYTHON", "python" as identical

**Test Results:**
```bash
# Test embedding protection
python tests/test_embedding_normalization.py

# Results:
# ✅ 23 skill variations → 7 unique skills (69.6% reduction)
# ✅ 60 stuffed words → 3 unique words
# ✅ Fuzzy matching: 80% similarity threshold working
```

#### 3. Monitoring & Logging
- **Keyword Stuffing Detection**: Flags resumes with >30% repetition
- **Suspicious Resume Logging**: Database tracking with severity levels
- **Non-Breaking Design**: Continues parsing even when threats detected

### Security Monitoring
```bash
# Check for suspicious resumes
python scripts/check_security.py

# Query database directly
python -c "
from src.database_manager import DatabaseManager
from src.config import config
db = DatabaseManager(config.database)
cases = db.get_suspicious_resumes(requires_review=True)
print(f'Cases requiring review: {len(cases)}')
"
```

**Documentation:**
- 📄 [SECURITY_IMPLEMENTATION.md](SECURITY_IMPLEMENTATION.md) - Implementation details and test results
- 📄 [ADVERSARIAL_ATTACK_ANALYSIS.md](ADVERSARIAL_ATTACK_ANALYSIS.md) - Attack surface analysis

### How Embedding Protection Works

The system now automatically normalizes skills before generating embeddings, preventing adversaries from inflating similarity scores through skill manipulation:

**Without Protection (VULNERABLE):**
```python
# Attacker lists skill 23 times with variations
skills = ["Python", "python", "PYTHON", "Python3", "Python 3.x", ...]
embedding_text = "Technical Skills: Python, python, PYTHON, Python3, ..."
# Result: Artificially high similarity score due to repetition
```

**With Protection (SECURE):**
```python
# System deduplicates and normalizes
skills = ["Python", "python", "PYTHON", "Python3", "Python 3.x", ...]
normalized = deduplicate_skills(skills)  # ["python"]
embedding_text = "Technical Skills: python"
# Result: Fair similarity score based on unique skills
```

**Benefit:** Prevents gaming the recommendation system through skill stuffing or synonym abuse.

## Matching algorithm
- ~~Assumes that the names of skills and education degrees in jobs and cvs are standardized. EG. all CVs and job descriptions must have "Machine Learning" not "ML" or "ML Engineer" or anything else. We can use fuzzy search or even better embeddings for skills to make it more robust, but it is left for future to keep the current repo simpler.~~
- **UPDATE:** Now implements skill synonym normalization and fuzzy matching (80% similarity threshold)
- Automatically maps common variations: "ML" → "machine learning", "JS" → "javascript", "AWS" → "aws"
- Uses `SequenceMatcher` for fuzzy deduplication of similar skills
- Caps skills at 50 per category to prevent manipulation

## embeddings
- Use sentence-transformer(https://huggingface.co/sentence-transformers/all-mpnet-base-v2) for efficiency in terms of cost and compute.
- For best performance, we should use some of the top models from https://huggingface.co/spaces/mteb/leaderboard

## CV data source
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/code?datasetId=1519260

## Job descriptions are synthesized using claude sonnet 4.5

## Unique Features

### 🛡️ Adversarial Attack Defense
Comprehensive security system protecting against:
- **LLM Prompt Injection**: Delimiter-based prompts + 12 detection patterns
- **Embedding Poisoning**: Keyword stuffing detection + deduplication
- **Skills Manipulation**: Fuzzy matching + caps on skill counts
- **Experience Inflation**: Date validation + overlap detection
- **Credential Fabrication**: Prestigious institution flagging

**Test Results:**
- ✅ 100% injection pattern detection rate
- ✅ 0% false negatives in keyword stuffing tests
- ✅ Non-breaking: All resumes processed despite threats
- ✅ Minimal overhead: ~30ms per resume

See [SECURITY_IMPLEMENTATION.md](SECURITY_IMPLEMENTATION.md) for details.

### ⚡ Two-Stage Retrieval
Performance-optimized recommendation system for large-scale datasets:
- **Stage 1**: Fast vector similarity filtering (top-k candidates)
- **Stage 2**: Detailed multi-factor scoring (skills, experience, education)
- **Benefits**: 16% reduction in Stage 2 jobs, scales to 1,000+ CVs and 500+ jobs

```bash
# Enable two-stage retrieval
python src/generate_recommendations.py --stage1-k 50 --stage1-threshold 0.3

# Disable for small datasets
python src/generate_recommendations.py --single-stage
```

See configuration in `configurations/config.yaml`.

## Strengths

✅ pgvector with IVFFLAT indexing - Already optimized for vector similarity search
✅ Batch processing - save_recommendations_batch() using execute_values
✅ Database-side similarity computation - Using 1 - (je.embedding <=> ce.embedding)
✅ Parallel processing - ThreadPoolExecutor with 10 workers in generate_recommendations.py
✅ Performance monitoring - Comprehensive tracking in performance_monitor.py
✅ Vector embeddings - Using sentence-transformers/all-mpnet-base-v2 (768-dim)



#!/bin/bash
!curl -L -o resume-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/snehaanbhawal/resume-dataset
!unzip resume-dataset.zip -d resume-dataset
!ls resume-dataset

