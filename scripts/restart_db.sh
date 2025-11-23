#!/bin/bash
# Restart database without losing data - for normal development workflow

set -e  # Exit on any error

echo "=================================================="
echo "Restarting PostgreSQL (Data will persist)"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Stop containers (keep volumes)
echo -e "${YELLOW}Stopping PostgreSQL container...${NC}"
docker-compose down
echo -e "${GREEN}✓ Container stopped (data preserved)${NC}"
echo ""

# Step 2: Start PostgreSQL
echo -e "${YELLOW}Starting PostgreSQL container...${NC}"
docker-compose up -d
echo -e "${GREEN}✓ PostgreSQL container started${NC}"
echo ""

# Step 3: Wait for PostgreSQL to be ready
echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
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
