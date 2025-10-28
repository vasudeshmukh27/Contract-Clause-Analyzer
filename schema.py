import psycopg2
from pathlib import Path

DDL = Path("schema.sql").read_text(encoding="utf-8")

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="contract_analyzer",
    user="postgres",
    password="YourPwd123",
)
conn.autocommit = False
cur = conn.cursor()

# Simple splitter: ensure statements end with semicolons and no procedural blocks
statements = [s.strip() for s in DDL.split(";") if s.strip()]
for stmt in statements:
    cur.execute(stmt + ";")

conn.commit()
cur.close()
conn.close()
print("Schema created.")
