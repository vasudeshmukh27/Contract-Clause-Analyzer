import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect(host="localhost", port=5432,
                        database="contract_analyzer",
                        user="postgres", password="YourPwd123")
register_vector(conn)
cur = conn.cursor()

cur.execute("SELECT extname FROM pg_extension WHERE extname='vector';")
print("Vector extension:", cur.fetchone())

cur.execute("""
SELECT table_name FROM information_schema.tables
WHERE table_schema='public' ORDER BY table_name;
""")
print("Tables:", [r[0] for r in cur.fetchall()])

cur.close(); conn.close()