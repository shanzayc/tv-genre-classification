import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = os.path.join("data", "netflix_titles.csv")
MODEL_PATH = os.path.join("models", "genre_model.joblib")


def extract_primary_genre(listed_in: str) -> str:
    if not isinstance(listed_in, str) or not listed_in.strip():
        return ""
    return listed_in.split(",")[0].strip()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Make sure netflix_titles.csv is in /data."
        )

    df = pd.read_csv(DATA_PATH)

    df["primary_genre"] = df["listed_in"].apply(extract_primary_genre)
    df["description"] = df["description"].apply(clean_text)

    df = df[(df["primary_genre"] != "") & (df["description"] != "")]
    df = df.dropna(subset=["primary_genre", "description"])

    # Remove extremely rare genres to reduce noise
    min_samples = 40
    genre_counts = df["primary_genre"].value_counts()
    df = df[df["primary_genre"].isin(genre_counts[genre_counts >= min_samples].index)]

    X = df["description"]
    y = df["primary_genre"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=25000
            )),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n=== TV Genre Classification Results ===")
    print(f"Samples used: {len(df)}")
    print(f"Number of genres: {df['primary_genre'].nunique()}")
    print(f"Accuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
