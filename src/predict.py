import os
import sys
import joblib
import numpy as np

MODEL_PATH = os.path.join("models", "genre_model.joblib")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Run: python src/train.py")
        sys.exit(1)

    # Get the input text from command line
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "your show description here"')
        sys.exit(1)

    description = " ".join(sys.argv[1:]).strip()
    if not description:
        print("Please provide a non-empty description.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    # Predict label
    pred_label = model.predict([description])[0]

    print("\n=== Prediction ===")
    print(f"Input: {description}")
    print(f"Predicted genre: {pred_label}")

    # If the classifier supports probabilities, print top 3
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([description])[0]
        classes = model.classes_

        top_k = 3
        top_idx = np.argsort(probs)[::-1][:top_k]

        print("\nTop 3 predictions:")
        for i in top_idx:
            print(f"- {classes[i]} ({probs[i]:.2f})")
    else:
        print("\n(No probability output available for this model.)")

if __name__ == "__main__":
    main()
