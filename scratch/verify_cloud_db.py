import os
import sys
import redis
from sqlalchemy import create_engine, text

def test_redis():
    print("--- 1. Testing Redis Connection ---")
    redis_url = os.getenv("REDIS_URL")
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    
    try:
        if redis_url:
            print(f"Connecting via REDIS_URL environment variable...")
            r = redis.Redis.from_url(redis_url, socket_timeout=3)
        else:
            print(f"Connecting via host/port: {redis_host}:{redis_port}...")
            r = redis.Redis(host=redis_host, port=redis_port, socket_timeout=3)
            
        r.ping()
        print("✔ Redis connection successful!")
        
        # Test write/read
        test_key = "iris:connection_test"
        r.setex(test_key, 10, "connection_ok")
        val = r.get(test_key)
        if val and val.decode('utf-8') == "connection_ok":
            print("✔ Redis write/read test passed!")
        else:
            print("❌ Redis read value mismatch.")
            
    except Exception as e:
        print(f"❌ Redis connection failed: {str(e)}")
        print("Note: Backend will automatically fall back to an in-memory dictionary cache.")

def test_postgresql():
    print("\n--- 2. Testing PostgreSQL / SQL Database Connection ---")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("No DATABASE_URL set. Defaulting to local SQLite (sqlite:///./iris.db).")
        database_url = "sqlite:///./iris.db"
    else:
        print(f"Connecting via DATABASE_URL environment variable...")
        
    try:
        # Prevent check_same_thread for SQLite test
        connect_args = {"check_same_thread": False} if "sqlite" in database_url else {}
        engine = create_engine(database_url, connect_args=connect_args)
        
        # Try a quick connection test
        with engine.connect() as conn:
            # Query version or current time
            if "sqlite" in database_url:
                res = conn.execute(text("SELECT sqlite_version();")).scalar()
                print(f"✔ Local SQLite connection successful! Version: {res}")
            else:
                res = conn.execute(text("SELECT version();")).scalar()
                print(f"✔ Remote PostgreSQL connection successful! Version: {res}")
                
            # Perform table creation check
            conn.execute(text("CREATE TABLE IF NOT EXISTS iris_temp_test (id SERIAL PRIMARY KEY, test_val VARCHAR(20));"))
            conn.execute(text("INSERT INTO iris_temp_test (test_val) VALUES ('db_ok');"))
            val = conn.execute(text("SELECT test_val FROM iris_temp_test ORDER BY id DESC LIMIT 1;")).scalar()
            conn.execute(text("DROP TABLE iris_temp_test;"))
            
            if val == "db_ok":
                print("✔ Database write/read/drop test passed!")
            else:
                print("❌ Database test value mismatch.")
                
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        if "postgresql" in database_url:
            print("Check that your Supabase password is correct and your connection URL allows incoming connections.")

if __name__ == "__main__":
    print("====================================================")
    print("Project IRIS - Cloud Database Connectivity Diagnostics")
    print("====================================================\n")
    test_redis()
    test_postgresql()
    print("\n====================================================")
