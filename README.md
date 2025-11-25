# CV parser and recommender

## Resume Parser

The resume parser intelligently extracts structured data from PDF CVs using LLM-powered NLP. It handles diverse CV layouts (traditional, modern, multi-column) and converts them into standardized JSON format with fields like work experience, education, skills, and certifications.

**Key Features:**
- **Multi-provider support**: Azure OpenAI and Google Gemini
- **Robust error handling**: Validates PDFs, handles corruption, provides detailed logging  
- **Format flexibility**: Parses various CV layouts and unconventional structures
- **Smart extraction**: Uses few-shot learning for accurate field detection
- **Retry logic**: Up to 3 attempts with fallback to ensure reliability
- **Data validation**: Cleans and validates extracted information

**Usage:**
```bash
python src/resume_parser.py path/to/resume.pdf -o output.json --provider gemini
```

## Matching algorithm
- Assumes that the names of skills and education degrees in jobs and cvs are standardized. EG. all CVs and job descriptions must have "Machine Learning" not "ML" or "ML Engineer" or anything else. We can use fuzzy search or even better embeddings for skills to make it more robust, but it is left for future to keep the current repo simpler.

## embeddings
- Use sentence-transformer(https://huggingface.co/sentence-transformers/all-mpnet-base-v2) for efficiency in terms of cost and compute.
- For best performance, we should use some of the top models from https://huggingface.co/spaces/mteb/leaderboard

## CV data source
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/code?datasetId=1519260