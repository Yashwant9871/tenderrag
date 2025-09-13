CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY,
  filename TEXT NOT NULL,
  status TEXT NOT NULL,
  uploaded_at TIMESTAMP DEFAULT now(),
  pages INTEGER DEFAULT 0,
  chunks INTEGER DEFAULT 0,
  error TEXT NULL
);
