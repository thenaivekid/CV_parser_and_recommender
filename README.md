# CV parser and recommender
This repository implements a CV/Resume parsing and job recommendation system focused on: NLP / LLM-driven parsing, vector embeddings & semantic search (pgvector + Redis cache), explainable matching, and performance-aware retrieval and long with qualitative and performance evaluation metrics.

**The technical report is in [DOCUMENTATION.md](https://github.com/thenaivekid/CV_parser_and_recommender/blob/main/DOCUMENTATION.md)**
## Usage
**Needs Linux machine with git, python, docker and docker-compose installed. Also, python venv should be supported.**
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
pip install uv
uv pip install -r requirements.txt


# start postgresql docker container
./scripts/init_database.sh #for creating tables for first time
# ./scripts/restart_db.sh #after running init_database.sh once, keeps the data from earlier sessions

# install redis
./scripts/install_redis.sh

# run tests
./scripts/test_all.sh

# run cv parsing using test pdfs from dataset downloaded earlier and put them to db
./scripts/run_cv_batch_processing.sh #replace resume_base_path configurations/config.yaml to use the custom pdfs

# run job processing to put job descriptions to db
./scripts/run_job_processing.sh # uses sample jsons 

./scripts/run_job_pdf_processing.sh data/sample_jd_pdf #pass path to custom jd pdfs

# run generate recommendations and put to db
./scripts/run_recommendations.sh

# run evaluation pipeline to get qualitative evals of generated recommendations, using dummy test ground truth
./scripts/run_evaluation.sh

# generate dashboard to see performance metrics in terms of system throughput and so on(uses stored values during parsing and recommendation generation if nothing is running currently)
./scripts/generate_dashboard.sh # see ./data/performance_reports to see the html generated showing performace metrics

# run recommendation using 2 stage vs 1 stage comparision
./scripts/run_recommendation_benchmark.sh 


```