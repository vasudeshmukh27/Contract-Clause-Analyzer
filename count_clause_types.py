# count_clause_types.py
import os
import psycopg2

# Load DB config from env or hardcode
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "contract_analyzer"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "YourPwd123"),
}

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT clause_type, COUNT(*) 
        FROM clause_spans
        GROUP BY clause_type
        ORDER BY COUNT DESC;
    """)
    rows = cur.fetchall()
    print("Clause Type Counts:")
    for clause_type, count in rows:
        print(f"{clause_type:40} {count}")
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
