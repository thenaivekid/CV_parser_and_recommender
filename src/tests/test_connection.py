import os
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine, text

load_dotenv()

# Test basic psycopg2 connection
try:
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )
    cur = conn.cursor()
    cur.execute("SELECT version();")
    db_version = cur.fetchone()
    print(f"✓ PostgreSQL connected: {db_version[0]}")
    
    # Check for pgvector extension
    cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
    if cur.fetchone():
        print("✓ pgvector extension installed")
    
    cur.close()
    conn.close()
except Exception as e:
    print(f"✗ Connection failed: {e}")

# Test SQLAlchemy connection
try:
    engine = create_engine(os.getenv("DATABASE_URL"))
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM candidates;"))
        count = result.scalar()
        print(f"✓ SQLAlchemy connected. Candidates in DB: {count}")
except Exception as e:
    print(f"✗ SQLAlchemy failed: {e}")

print("\n✓ Setup complete! Ready to start development.")