from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector

DB = dict(
    host="localhost",
    port=5432,
    database="contract_analyzer",
    user="postgres",
    password="YourPwd123",
)

# Use a 384-d model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
QUERY = "data sharing and processing terms"

def main():
    # Build query embedding
    model = SentenceTransformer(MODEL_NAME)  # outputs 384-d embeddings
    qvec = model.encode([QUERY])[0]  # numpy array is acceptable

    # Connect to Postgres and ensure pgvector adaptation
    conn = psycopg2.connect(**DB)
    register_vector(conn)
    cur = conn.cursor()

    # Cosine distance query with explicit ::vector cast on the parameter
    # Note: lower distance = more similar
    sql = """
    SELECT
        cs.span_id,
        cs.heading,
        cs.page,
        LEFT(cs.text, 200) AS excerpt,
        (se.embedding <=> %s::vector) AS cosine_distance
    FROM span_embeddings se
    JOIN clause_spans cs ON cs.span_id = se.span_id
    ORDER BY se.embedding <=> %s::vector
    LIMIT 5;
    """
    cur.execute(sql, (qvec, qvec))
    rows = cur.fetchall()

    for r in rows:
        span_id, heading, page, excerpt, dist = r
        print(f"[span_id={span_id}] (page={page}) {heading} | dist={dist:.6f}")
        print(f"  {excerpt}\n")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()