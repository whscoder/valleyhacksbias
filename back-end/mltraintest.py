from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.pipeline import FeatureUnion, Pipeline


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "fact_opinion"
    / "processed"
    / "fact_opinion.csv"
)
MODEL_PATH = DATA_PATH.with_name("fact_opinion_classifier.pkl")
TARGET_SCORE = 0.90

df = pd.read_csv(DATA_PATH)

# Keep related sentences in the same split.
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=2026)
train_index, test_index = next(
    splitter.split(df["text"], df["label"], groups=df["group_id"])
)
train_df = df.iloc[train_index]
test_df = df.iloc[test_index]

# Keep validation separate from both grid-search training and final testing.
validation_splitter = GroupShuffleSplit(
    test_size=0.2, n_splits=1, random_state=2027
)
fit_index, validation_index = next(
    validation_splitter.split(
        train_df["text"], train_df["label"], groups=train_df["group_id"]
    )
)
fit_df = train_df.iloc[fit_index]
validation_df = train_df.iloc[validation_index]

# Add character TF-IDF to the original word TF-IDF + Logistic Regression logic.
tfidf = FeatureUnion(
    [
        (
            "word",
            TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, sublinear_tf=True
            ),
        ),
        (
            "character",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=100_000,
                sublinear_tf=True,
            ),
        ),
    ]
)

# TF-IDF is fitted only on each training fold because it is inside the pipeline.
pipeline = Pipeline(
    [
        ("tfidf", tfidf),
        (
            "classifier",
            LogisticRegression(max_iter=2_000, random_state=42),
        ),
    ]
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid={
        "classifier__C": [2.0, 4.0, 8.0],
        "classifier__class_weight": [None, "balanced"],
    },
    scoring="f1_macro",
    cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=1,
)

search.fit(
    fit_df["text"],
    fit_df["label"],
    groups=fit_df["group_id"],
)

# Pick the lowest threshold where every class metric reaches 90%.
validation_probabilities = search.predict_proba(validation_df["text"])
validation_predictions = search.classes_[validation_probabilities.argmax(axis=1)]
validation_confidence = validation_probabilities.max(axis=1)
confidence_threshold = 0.95

for threshold in np.arange(0.50, 0.96, 0.01):
    accepted = validation_confidence >= threshold
    if accepted.sum() == 0 or len(np.unique(validation_df["label"][accepted])) < 2:
        continue

    precision, recall, f1, _ = precision_recall_fscore_support(
        validation_df["label"][accepted],
        validation_predictions[accepted],
        labels=["fact", "opinion"],
        zero_division=0,
    )
    if min(*precision, *recall, *f1) >= TARGET_SCORE:
        confidence_threshold = float(round(threshold, 2))
        break

# Refit the selected pipeline on all non-test data, then test it once.
model = clone(search.best_estimator_)
model.fit(train_df["text"], train_df["label"])
model.confidence_threshold_ = confidence_threshold

test_probabilities = model.predict_proba(test_df["text"])
test_predictions = model.classes_[test_probabilities.argmax(axis=1)]
test_confidence = test_probabilities.max(axis=1)
accepted = test_confidence >= confidence_threshold

with MODEL_PATH.open("wb") as model_file:
    pickle.dump(model, model_file)

print("Best parameters:", search.best_params_)
print("Confidence threshold:", confidence_threshold)
print("OpenAI fallback rate:", f"{1 - accepted.mean():.1%}")
print("\nAll test predictions")
print(classification_report(test_df["label"], test_predictions, digits=3))
print("Accepted local predictions")
print(
    classification_report(
        test_df["label"][accepted], test_predictions[accepted], digits=3
    )
)
print("Saved model:", MODEL_PATH)
