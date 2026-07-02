"""
train.py

Loads captured hand-landmark data, trains a classifier to recognize
ASL alphabet letters, and saves the trained model.

Usage:
    python3 train.py

Expects:
    data/raw/landmarks.csv   (produced by capture.py)

Produces:
    model/asl_classifier.pkl
    A printed accuracy score + confusion matrix showing which letters
    get confused with which.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = "data/raw/landmarks.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "asl_classifier.pkl")


def main():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"Couldn't find {DATA_PATH}. Run capture.py first to collect data."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} letters.")

    # Show how many samples per letter -- flags any letter that's thin on data
    print("\nSamples per letter:")
    print(df["label"].value_counts().sort_index())

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples...")

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.2%}")

    print("\nPer-letter performance:")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels_sorted = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    header = "     " + " ".join(f"{l.upper():>3}" for l in labels_sorted)
    print(header)
    for label, row in zip(labels_sorted, cm):
        row_str = " ".join(f"{v:>3}" for v in row)
        print(f"{label.upper():>3}  {row_str}")

    # Flag the most-confused letter pairs so you know what to watch for
    print("\nMost confused pairs (actual -> predicted, count):")
    confusions = []
    for i, actual in enumerate(labels_sorted):
        for j, predicted in enumerate(labels_sorted):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], actual, predicted))
    confusions.sort(reverse=True)
    for count, actual, predicted in confusions[:10]:
        print(f"  {actual.upper()} -> {predicted.upper()}: {count} times")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
