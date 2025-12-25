import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "netflix.db"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "genre_model.joblib"

# -----------------------------
# Load data from SQLite using SQL
# -----------------------------
def load_from_sqlite(
    db_path: Path,
    only_type: str | None = None,       # "TV Show" or "Movie"
    min_year: int | None = None,
    limit: int | None = None
) -> pd.DataFrame:
    """
    Loads (description, genre) from SQLite using a SQL query.
    Optional filters:
      - only_type: restrict to "TV Show" or "Movie"
      - min_year: restrict to titles released >= min_year
      - limit: restrict number of rows for quick testing
    """
    conn = sqlite3.connect(db_path)

    where_clauses = [
        "description IS NOT NULL",
        "genre IS NOT NULL",
        "TRIM(description) != ''",
        "TRIM(genre) != ''",
    ]
    params: list = []

    if only_type:
        where_clauses.append("type = ?")
        params.append(only_type)

    if min_year:
        where_clauses.append("release_year >= ?")
        params.append(min_year)

    where_sql = " AND ".join(where_clauses)

    query = f"""
    SELECT
        description,
        genre
    FROM titles
    WHERE {where_sql}
    """

    if limit:
        query += " LIMIT ?"
        params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Keep primary genre only if genre contains multiple (comma-separated)
    # Example: "Dramas, International Movies" -> "Dramas"
    df["genre"] = df["genre"].astype(str).str.split(",").str[0].str.strip()

    return df


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run: python src/load_to_db.py"
        )

    # -----------------------------
    # Load dataset
    # -----------------------------
    # Choose one:
    #  - only_type="TV Show"  (genre classification for TV only)
    #  - only_type="Movie"    (movies only)
    #  - only_type=None       (both)
    df = load_from_sqlite(DB_PATH, only_type=None, min_year=None, limit=None)

    # Optional: filter out very rare genres to stabilize training
    # (Helps when some genres have almost no examples.)
    min_samples_per_genre = 20
    genre_counts = df["genre"].value_counts()
    keep_genres = genre_counts[genre_counts >= min_samples_per_genre].index
    df = df[df["genre"].isin(keep_genres)].copy()

    X = df["description"].astype(str)
    y = df["genre"].astype(str)

    # -----------------------------
    # Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,   # reproducible results
        stratify=y         # keeps genre proportions similar in train/test
    )

    # -----------------------------
    # Build pipeline: TF-IDF -> Logistic Regression
    # -----------------------------
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=50000
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            n_jobs=None
        ))
    ])

    # -----------------------------
    # Train
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluate
    # -----------------------------
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    labels = sorted(y.unique().tolist())

    print("\n=== TV/Movie Genre Classification Results (SQL-backed) ===")
    print(f"Samples used: {len(df)}")
    print(f"Number of genres: {len(labels)}")
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels))

    # -----------------------------
    # Save model (vectorizer + classifier together)
    # -----------------------------
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
