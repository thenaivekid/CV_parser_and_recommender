#!/bin/bash
# Database initialization script for fresh setup

set -e  # Exit on any error

echo "=================================================="
echo "CV Parser Database Initialization"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Stop and remove existing containers (including volumes)
echo -e "${YELLOW}Step 1: Stopping existing containers...${NC}"
docker-compose down
echo -e "${GREEN}✓ Containers stopped and volumes removed${NC}"
echo ""

# Step 2: Start PostgreSQL with auto-initialization
echo -e "${YELLOW}Step 2: Starting PostgreSQL container...${NC}"
docker-compose up -d
echo -e "${GREEN}✓ PostgreSQL container started${NC}"
echo ""

# Step 3: Wait for PostgreSQL to be ready
echo -e "${YELLOW}Step 3: Waiting for PostgreSQL to be ready...${NC}"
max_attempts=30
attempt=0

until docker exec cv-job-pgvector pg_isready -U cv_user -d cv_job_db > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        echo -e "${RED}✗ PostgreSQL failed to start after $max_attempts seconds${NC}"
        exit 1
    fi
    echo -e "  Waiting... ($attempt/$max_attempts)"
    sleep 1
done

echo -e "${GREEN}✓ PostgreSQL is ready${NC}"
echo ""



# Check the exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo -e "${GREEN}✓ Database initialization complete!${NC}"
    echo "=================================================="
    echo ""
    echo "Database is ready to use."
    
else
    echo ""
    echo "=================================================="
    echo -e "${RED}✗ Database initialization failed${NC}"
    echo "=================================================="
    exit 1
fi
