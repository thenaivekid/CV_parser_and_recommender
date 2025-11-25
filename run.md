<!-- create dir structure -->
mkdir -p postgres
mkdir -p data/sample_cvs
mkdir -p data/sample_jobs
mkdir -p data/outputs
mkdir -p scripts


<!-- start pg docker -->
docker-compose up -d
docker-compose down


<!-- test conn -->
docker exec -it cv-job-pgvector psql -U cv_user -d cv_job_db -c "\dx"

docker exec -it cv-job-pgvector psql -U cv_user -d cv_job_db -c "\dt"

docker exec -it cv-job-pgvector psql -U cv_user -d cv_job_db -c "\d candidates"

docker exec -it cv-job-pgvector   psql -U cv_user -d cv_job_db   -c "SELECT count(*) FROM recommendations;"

docker exec -it cv-job-pgvector   psql -U cv_user -d cv_job_db   -c "SELECT candidate_id FROM candidate_embeddings;"

docker exec -it cv-job-pgvector   psql -U cv_user -d cv_job_db   -c "SELECT * FROM jobs;"

<!-- create venv -->
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

<!-- test db -->
python tests/test_connection.py 


<!-- push to github -->
git add .
git commit -m "Setup: Docker + pgvector + Python env"
git push


<!-- parser -->
python src/resume_parser.py /workspaces/CV_parser_and_recommender/resume-dataset/data/data/ENGINEERING/10030015.pdf -o data/outputs/ENGINEERING_10030015.pdf.json
python src/resume_parser.py /workspaces/CV_parser_and_recommender/resume-dataset/data/data/ENGINEERING/10030015.pdf -o data/outputs/ENGINEERING_10030015.pdf.json --provider gemini