"""
Train, tune and evaluate the SMS Spam Detector.

Run:  python train.py
Creates: model.joblib, metrics.json and charts in images/
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # draw charts without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_FILE = "data/sms.tsv"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["label", "text"])
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)   # same SMS in train AND test would be cheating
    df["is_spam"] = (df["label"] == "spam").astype(int)
    df["length"] = df["text"].str.len()
    print(f"Loaded {before} messages, kept {len(df)} after removing {before - len(df)} duplicates")
    print(df["label"].value_counts().to_string(), "\n")
    return df


def make_pipeline(classifier):
    tfidf = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                            min_df=2, sublinear_tf=True)
    return Pipeline([("tfidf", tfidf), ("clf", classifier)])


def candidate_models():
    """name -> (pipeline, small hyper-parameter grid)"""
    return {
        "Naive Bayes": (make_pipeline(MultinomialNB()),
                        {"clf__alpha": [0.05, 0.1, 0.3, 1.0]}),
        "Logistic Regression": (make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
                                {"clf__C": [1, 5, 10, 30]}),
        "Linear SVM": (make_pipeline(CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)),
                       {"clf__estimator__C": [0.3, 1, 3]}),
    }


def scores(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}


def save_eda_chart(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df["label"].value_counts()
    bars = axes[0].bar(counts.index, counts.values, color=["#4C78A8", "#E45756"])
    axes[0].bar_label(bars)
    axes[0].set_title("Messages per class")
    for label, color in [("ham", "#4C78A8"), ("spam", "#E45756")]:
        axes[1].hist(df[df["label"] == label]["length"], bins=40, alpha=0.65, label=label, color=color)
    axes[1].set_title("Message length (characters)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("images/eda.png", dpi=120)
    plt.close()


def save_top_words_chart(lr_pipeline, n=12):
    words = np.array(lr_pipeline.named_steps["tfidf"].get_feature_names_out())
    coefs = lr_pipeline.named_steps["clf"].coef_[0]
    order = np.argsort(coefs)
    spam_idx, ham_idx = order[-n:], order[:n]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].barh(words[spam_idx], coefs[spam_idx], color="#E45756")
    axes[0].set_title("Words pushing towards SPAM")
    axes[1].barh(words[ham_idx], -coefs[ham_idx], color="#4C78A8")
    axes[1].set_title("Words pushing towards HAM (normal)")
    plt.tight_layout()
    plt.savefig("images/top_words.png", dpi=120)
    plt.close()
    return list(words[spam_idx][::-1][:8])


def main():
    df = load_data()
    save_eda_chart(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["is_spam"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["is_spam"])
    print(f"Train: {len(X_train)} messages | Test: {len(X_test)} messages\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = {}, {}
    for name, (pipe, grid) in candidate_models().items():
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1).fit(X_train, y_train)
        test_scores = scores(y_test, search.predict(X_test))
        results[name] = {"best_params": {k.split("__", 1)[1]: v for k, v in search.best_params_.items()},
                         "cv_f1": round(search.best_score_, 4),
                         **{f"test_{k}": round(v, 4) for k, v in test_scores.items()}}
        fitted[name] = search.best_estimator_

    table = pd.DataFrame(results).T[["cv_f1", "test_accuracy", "test_precision", "test_recall", "test_f1"]]
    print(table.to_string(), "\n")

    best_name = max(results, key=lambda n: results[n]["cv_f1"])   # choose using CV, not the test set
    best = fitted[best_name]
    y_pred = best.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"BEST MODEL: {best_name}  {results[best_name]['best_params']}")
    print(f"Test set -> correct ham: {tn}, ham wrongly flagged as spam: {fp}, spam missed: {fn}, spam caught: {tp}")

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["ham", "spam"]).plot(
        cmap="Blues", colorbar=False)
    plt.title(f"Confusion matrix - {best_name}")
    plt.grid(False)
    plt.savefig("images/confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    spam_words = save_top_words_chart(fitted["Logistic Regression"])
    print("Strongest spam words:", spam_words)

    joblib.dump(best, "model.joblib")
    wrong = X_test[y_pred != y_test]
    with open("metrics.json", "w") as f:
        json.dump({"best_model": best_name, "models": results,
                   "confusion": {"true_ham": int(tn), "false_spam": int(fp), "missed_spam": int(fn), "caught_spam": int(tp)},
                   "top_spam_words": spam_words, "misclassified_examples": wrong.head(5).tolist()}, f, indent=2)
    print("\nSaved model.joblib, metrics.json and charts in images/")


if __name__ == "__main__":
    main()
