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

<!-- create venv -->
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

<!-- test db -->
python scripts/tests/test_connection.py 


<!-- push to github -->
git add .
git commit -m "Setup: Docker + pgvector + Python env"
git push