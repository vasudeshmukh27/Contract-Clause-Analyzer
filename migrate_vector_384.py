import psycopg2

conn = psycopg2.connect(
    host="localhost", port=5432,
    database="contract_analyzer",
    user="postgres", password="YourPwd123"
)
conn.autocommit = True  # DDL convenience
cur = conn.cursor()

ddl = """
DROP TABLE IF EXISTS span_embeddings;
CREATE TABLE span_embeddings (
  span_id INT PRIMARY KEY REFERENCES clause_spans(span_id) ON DELETE CASCADE,
  embedding vector(384)
);
"""
cur.execute(ddl)
cur.close(); conn.close()
print("span_embeddings recreated with vector(384)")