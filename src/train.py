import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# Load the dataset
# --------------------------------------------------
# This CSV contains Netflix movies and TV shows
# Each row has a description and a list of genres
df = pd.read_csv("data/netflix_titles.csv")


# --------------------------------------------------
# Data cleaning
# --------------------------------------------------
# Remove rows where description or genre is missing
df = df.dropna(subset=["description", "listed_in"])

# Keep only the FIRST genre to simplify classification
# Example: "Dramas, Thrillers" → "Dramas"
df["genre"] = df["listed_in"].apply(lambda x: x.split(",")[0])


# --------------------------------------------------
# Define features (X) and labels (y)
# --------------------------------------------------
X = df["description"]   # input text
y = df["genre"]         # genre we want to predict


# --------------------------------------------------
# Split data into training and testing sets
# --------------------------------------------------
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# Build ML pipeline
# --------------------------------------------------
# Step 1: Convert text → numerical features (TF-IDF)
# Step 2: Train a classifier (Logistic Regression)
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])


# --------------------------------------------------
# Train the model
# --------------------------------------------------
model.fit(X_train, y_train)


# --------------------------------------------------
# Evaluate the model
# --------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n=== TV Genre Classification Results ===")
print(f"Samples used: {len(df)}")
print(f"Number of genres: {df['genre'].nunique()}")
print(f"Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# Save the trained model
# --------------------------------------------------
joblib.dump(model, "models/genre_model.joblib")
print("\nModel saved to models/genre_model.joblib")
