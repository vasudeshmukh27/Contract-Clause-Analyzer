# ingest_one.py
# Purpose: Ingest a single PDF contract into the database with clause spans and embeddings

import os
from pathlib import Path
import argparse
import psycopg2
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector
from unstructured.partition.pdf import partition_pdf
import json

# Database configuration (from .env or defaults)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "contract_analyzer"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "YourPwd123"),
}

# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_contract(pdf_path: str, vendor: str, title: str) -> None:
    # 1. Parse PDF into spans
    elements = partition_pdf(filename=pdf_path)
    # Extract spans (e.g., by section headings or paragraphs)
    spans = []
    for elem in elements:
        text = elem.text.strip()
        page = getattr(elem.metadata, "page_number", None)
        if text:
            spans.append((text, page))

    # 2. Connect to DB and insert document record
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (vendor, title) VALUES (%s, %s) RETURNING doc_id",
        (vendor, title),
    )
    doc_id = cur.fetchone()[0]
    conn.commit()

    # 3. Insert spans and compute embeddings
    model = SentenceTransformer(MODEL_NAME)
    register_vector(conn)
    span_ids = []

    for text, page in spans:
        # Insert span record
        cur.execute(
            "INSERT INTO clause_spans (doc_id, text, heading, page, clause_type, attributes) "
            "VALUES (%s, %s, %s, %s, 'Unknown', '{}'::jsonb) RETURNING span_id",
            (doc_id, text[:1000], None, page),
        )
        span_id = cur.fetchone()[0]
        span_ids.append(span_id)

        # Compute embedding
        embedding = model.encode([text])[0]
        # Insert embedding
        cur.execute(
            "INSERT INTO span_embeddings (span_id, embedding) VALUES (%s, %s)",
            (span_id, embedding.tolist()),
        )

    conn.commit()
    cur.close()
    conn.close()

    # Final confirmation print (ASCII only)
    print(f"Done. Document {doc_id} with {len(span_ids)} spans and embeddings inserted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a contract PDF")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--vendor", required=True, help="Vendor name")
    parser.add_argument("--title", required=True, help="Document title")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    ingest_contract(str(pdf_path), args.vendor, args.title)
