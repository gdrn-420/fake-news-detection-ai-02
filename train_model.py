"""
Train a TF-IDF + Logistic Regression text classifier and evaluate it honestly.

Key points:
  * Duplicate sentences are removed.
  * The data is split BY TEMPLATE (GroupShuffleSplit), so test sentences come from
    sentence patterns the model never saw in training. Scores are therefore not
    inflated by memorisation.
  * Results are also checked with grouped cross-validation.
  * Metrics are saved to evaluation_results.json by this script, so the numbers
    in your logbook always match your code.

Run from the project folder:
    python generate_data.py
    python train_model.py
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "news_samples.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def main():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["text", "label", "template_id"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df = df[df["label"].isin(["REAL", "FAKE"])].drop_duplicates(subset="text")
    if len(df) < 10 or df["label"].nunique() != 2:
        raise ValueError("Dataset must contain at least 10 rows and both REAL and FAKE labels.")

    print(f"Rows after removing duplicates: {len(df)}  | unique templates: {df['template_id'].nunique()}")
    print(df["label"].value_counts().to_string(), "\n")

    # Split by template so test patterns are unseen
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(df["text"], df["label"], groups=df["template_id"]))
    X_train, X_test = df["text"].iloc[train_idx], df["text"].iloc[test_idx]
    y_train, y_test = df["label"].iloc[train_idx], df["label"].iloc[test_idx]
    overlap = set(X_train) & set(X_test)

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, pos_label="FAKE", zero_division=0)
    rec = recall_score(y_test, predictions, pos_label="FAKE", zero_division=0)
    f1 = f1_score(y_test, predictions, pos_label="FAKE", zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=["FAKE", "REAL"])

    print("TEST SET RESULTS (unseen sentence patterns, small synthetic dataset only)")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)} | Sentences shared by train and test: {len(overlap)}")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}  (FAKE)")
    print(f"Recall   : {rec:.3f}  (FAKE)")
    print(f"F1-score : {f1:.3f}  (FAKE)")
    print("Confusion matrix (rows = actual, columns = predicted; order: FAKE, REAL):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    cv = cross_val_score(build_model(), df["text"], df["label"], groups=df["template_id"],
                         cv=GroupKFold(n_splits=5), scoring="accuracy")
    print(f"5-fold grouped cross-validation accuracy: {cv.mean():.3f} (+/- {cv.std():.3f})")

    # Fit the final model on ALL data for the app, keep evaluation numbers from the held-out test above
    final_model = build_model().fit(df["text"], df["label"])
    joblib.dump(final_model, MODEL_DIR / "fake_news_pipeline.joblib")

    results = {
        "dataset_rows": int(len(df)),
        "unique_templates": int(df["template_id"].nunique()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "sentences_shared_between_train_and_test": int(len(overlap)),
        "split_method": "GroupShuffleSplit by template_id (test patterns unseen)",
        "accuracy": round(float(acc), 3),
        "precision_fake": round(float(prec), 3),
        "recall_fake": round(float(rec), 3),
        "f1_fake": round(float(f1), 3),
        "confusion_matrix_fake_real": cm.tolist(),
        "grouped_cv_accuracy_mean": round(float(cv.mean()), 3),
        "grouped_cv_accuracy_std": round(float(cv.std()), 3),
        "note": "Scores are for an invented, template-based dataset. They do not measure real-world fake-news detection.",
    }
    (ROOT / "evaluation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved model to {MODEL_DIR / 'fake_news_pipeline.joblib'}")
    print("Saved metrics to evaluation_results.json")
    print("Reminder: these scores measure performance on invented examples, not real-world fact-checking.")


if __name__ == "__main__":
    main()
