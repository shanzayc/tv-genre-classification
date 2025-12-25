import sqlite3
import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "netflix_titles.csv"
DB_PATH = DATA_DIR / "netflix.db"

# -----------------------------
# Load CSV
# -----------------------------
print("Loading CSV dataset...")
df = pd.read_csv(CSV_PATH)

# -----------------------------
# Select and clean columns
# -----------------------------
# Keep only the columns we care about
df = df[[
    "title",
    "description",
    "listed_in",
    "release_year",
    "type"
]]

# Rename columns for SQL clarity
df = df.rename(columns={
    "listed_in": "genre"
})

# Drop rows with missing text or genre
df = df.dropna(subset=["description", "genre"])

print(f"Rows after cleaning: {len(df)}")

# -----------------------------
# Connect to SQLite database
# -----------------------------
print("Connecting to SQLite database...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# -----------------------------
# Create table
# -----------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS titles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    genre TEXT,
    release_year INTEGER,
    type TEXT
)
""")

# -----------------------------
# Clear existing data (optional but clean)
# -----------------------------
cursor.execute("DELETE FROM titles")
conn.commit()

# -----------------------------
# Insert data into table
# -----------------------------
print("Inserting data into database...")
df.to_sql(
    "titles",
    conn,
    if_exists="append",
    index=False
)

# -----------------------------
# Verify insertion
# -----------------------------
count = cursor.execute("SELECT COUNT(*) FROM titles").fetchone()[0]
print(f"Inserted {count} rows into database.")

# -----------------------------
# Close connection
# -----------------------------
conn.close()
print("Database setup complete.")
