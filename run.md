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

docker exec -it cv-job-pgvector   psql -U cv_user -d cv_job_db   -c "SELECT count(*) FROM processing_sessions;"

docker exec -it cv-job-pgvector \
  psql -U cv_user -d cv_job_db \
  -c "TRUNCATE TABLE performance_metrics;"


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
python src/cv_parser.py /workspaces/CV_parser_and_recommender/resume-dataset/data/data/ENGINEERING/10030015.pdf -o data/outputs/ENGINEERING_10030015.pdf.json
python src/cv_parser.py /workspaces/CV_parser_and_recommender/resume-dataset/data/data/ENGINEERING/10030015.pdf -o data/outputs/ENGINEERING_10030015.pdf.json --provider gemini

<!-- test perf monitoring -->
python tests/test_performance_monitoring.py

<!-- install redis and test -->
./scripts/install_redis.sh
sudo service redis-server start && sleep 2 && redis-cli ping
python -c 'import redis; r = redis.Redis(); print("Connected!" if r.ping() else "Failed")'

PYTHONPATH=/workspaces/CV_parser_and_recommender python test_backward_compatibility_without_redis.py #set configs to not use the redis cache

PYTHONPATH=/workspaces/CV_parser_and_recommender python src/generate_recommendations.py --workers 2 --top-k 5 --no-save-db --force-recalculate #set the configs to use redis

PYTHONPATH=/workspaces/CV_parser_and_recommender python test_redis_recommendations.py

<!-- next prompt -->
batch Recommendation Generation with Connection Pooling ⭐
Current: Each worker creates own DB connection (overhead)

Enhancement:

Connection pool (psycopg2.pool.ThreadedConnectionPool)
Batch candidate fetching (fetch 100 candidates at once)
Vectorized similarity computation for candidate batches
Benefits:

Reduces connection overhead by 80%
Enables true batch processing
Better resource utilization

