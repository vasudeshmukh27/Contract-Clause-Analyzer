CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  doc_id SERIAL PRIMARY KEY,
  vendor TEXT NOT NULL,
  title TEXT NOT NULL,
  file_path TEXT,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS clause_spans (
  span_id SERIAL PRIMARY KEY,
  doc_id INT REFERENCES documents(doc_id) ON DELETE CASCADE,
  clause_type TEXT NOT NULL,
  heading TEXT,
  text TEXT NOT NULL,
  page INTEGER,
  start_char INTEGER,
  end_char INTEGER,
  attributes JSONB,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS span_embeddings (
  span_id INT PRIMARY KEY REFERENCES clause_spans(span_id) ON DELETE CASCADE,
  embedding vector(768)
);