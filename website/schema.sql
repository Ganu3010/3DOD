DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS experiments;

CREATE TABLE IF NOT EXISTS user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  input_file_path TEXT NOT NULL,
  preprocessed_file_path TEXT,
  output_file_path TEXT,
  dataset TEXT NOT NULL,
  model TEXT NOT NULL,
  UNIQUE(input_file_path, dataset, model)
);

