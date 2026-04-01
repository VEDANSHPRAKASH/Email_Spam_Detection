"""
train_model.py
==============
Standalone training script for SpamShield AI.

HOW TO USE:
-----------
1. Put your CSV file inside the  backend/data/  folder.
   The CSV must have exactly two columns:
       - Category  →  "spam" or "ham"
       - Message   →  the email/SMS text

   Common alternative column names are handled automatically:
       v1 / v2  (raw UCI SMS Spam dataset)
       label / text
       class / message

2. Run from the backend/ directory:
       python train_model.py

3. (Optional) point to a different CSV:
       python train_model.py --data path/to/your_file.csv

4. After training, the model is saved to backend/model/ and
   app.py will load it automatically on next start.

CSV EXAMPLE:
------------
Category,Message
ham,"Hey, are you coming to dinner tonight?"
spam,"FREE prize! Click here to claim £1000 NOW!"
ham,"Can we reschedule our meeting to Thursday?"
spam,"URGENT: Your account will be suspended. Verify immediately."
"""

import argparse
import os
import sys
import json
import logging
import re
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(SCRIPT_DIR, "data", "spam.csv")   # ← PUT YOUR CSV HERE
MODEL_DIR    = os.path.join(SCRIPT_DIR, "model")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TEXT PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_text(text: str) -> str:
    """
    Clean and normalise raw email / SMS text.
    Applied before vectorisation so the TF-IDF vocabulary stays clean.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Replace URLs with a generic token so "http://win-now.com" and
    # "https://freemoney.net" both become "URL" in the vocabulary.
    text = re.sub(r"https?://\S+|www\.\S+", "URL", text)

    # Replace e-mail addresses with a token
    text = re.sub(r"\S+@\S+", "EMAIL", text)

    # Replace phone numbers with a token
    text = re.sub(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]", "PHONE", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Keep only word characters + a few meaningful punctuation marks
    text = re.sub(r"[^\w\s!?$£€%]", "", text)

    return text


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
COLUMN_ALIASES = {
    # UCI / Kaggle SMS Spam Collection uses v1 / v2
    "v1": "Category",
    "v2": "Message",
    # Other common names
    "label":   "Category",
    "class":   "Category",
    "target":  "Category",
    "text":    "Message",
    "sms":     "Message",
    "content": "Message",
    "email":   "Message",
    "body":    "Message",
}


def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV and normalise column names.
    Raises FileNotFoundError / ValueError with a helpful message.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"\n\n  CSV not found: {path}\n"
            f"  ► Place your CSV file at:  backend/data/spam.csv\n"
            f"    Or pass a custom path:   python train_model.py --data /path/to/file.csv\n"
        )

    log.info(f"Loading dataset from: {path}")
    try:
        df = pd.read_csv(path, encoding="latin-1")
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    # Rename columns using aliases
    df = df.rename(columns={c: COLUMN_ALIASES[c]
                             for c in df.columns if c in COLUMN_ALIASES})

    # Check required columns exist
    missing = {"Category", "Message"} - set(df.columns)
    if missing:
        raise ValueError(
            f"\n\n  Required columns not found: {missing}\n"
            f"  Columns found in your CSV: {list(df.columns)}\n"
            f"  Make sure your CSV has columns named:\n"
            f"    Category  →  'spam' or 'ham'\n"
            f"    Message   →  the email/SMS text\n"
            f"  (or use the aliases listed at the top of this file)\n"
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Category", "Message"]].copy()
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df["Category"] = df["Category"].str.strip().str.lower()

    invalid = df[~df["Category"].isin(["spam", "ham"])]["Category"].unique()
    if len(invalid):
        log.warning(f"Dropping rows with unknown labels: {invalid}")
    df = df[df["Category"].isin(["spam", "ham"])]

    if df.empty:
        raise ValueError("Dataset is empty after cleaning. Check your CSV labels.")

    df["ProcessedMessage"] = df["Message"].apply(preprocess_text)
    df["Spam"] = (df["Category"] == "spam").astype(int)

    spam_n = df["Spam"].sum()
    ham_n  = len(df) - spam_n
    log.info(f"Clean dataset: {len(df):,} rows  |  spam={spam_n:,}  ham={ham_n:,}  "
             f"ratio=1:{ham_n//max(spam_n,1)}")

    if spam_n < 10 or ham_n < 10:
        raise ValueError(
            f"Not enough examples — spam={spam_n}, ham={ham_n}. "
            "Need at least 10 of each class."
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4.  BUILD PIPELINES
# ══════════════════════════════════════════════════════════════════════════════
def build_pipelines() -> dict:
    """
    Three pipelines that will be trained and compared.
    All use TF-IDF with unigrams + bigrams (better than raw CountVectorizer).
    """

    def tfidf():
        return TfidfVectorizer(
            max_features=15_000,
            ngram_range=(1, 2),       # unigrams + bigrams
            sublinear_tf=True,        # log(1+tf)  — reduces impact of frequent terms
            min_df=2,                 # ignore terms appearing in only 1 document
            strip_accents="unicode",
            analyzer="word",
            stop_words="english",
        )

    return {
        "Naive Bayes": Pipeline([
            ("tfidf", tfidf()),
            ("clf",   MultinomialNB(alpha=0.1)),
        ]),

        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf",   LogisticRegression(
                C=5.0,
                max_iter=1_000,
                class_weight="balanced",  # handles class imbalance
                solver="liblinear",
            )),
        ]),

        "SVM": Pipeline([
            ("tfidf", tfidf()),
            # LinearSVC doesn't natively produce probabilities, so we wrap it
            ("clf",   CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", max_iter=2_000),
                cv=3,
            )),
        ]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING + EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(df: pd.DataFrame, cv_folds: int = 5) -> dict:
    """
    Train all three pipelines, print a full evaluation report,
    pick the best model by F1 score, and return artefacts.
    """
    X = df["ProcessedMessage"]
    y = df["Spam"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    pipelines  = build_pipelines()
    all_metrics = {}
    best_name   = None
    best_f1     = -1.0
    best_pipe   = None

    separator = "─" * 60

    for name, pipe in pipelines.items():
        log.info(f"\n{separator}")
        log.info(f"Training: {name}")

        # ── Fit ──────────────────────────────────────────────────────────────
        pipe.fit(X_train, y_train)

        # ── Predictions ──────────────────────────────────────────────────────
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        # ── Cross-validation ─────────────────────────────────────────────────
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv,
                                scoring="f1", n_jobs=-1)
        log.info(f"Cross-val F1 ({cv_folds}-fold): "
                 f"{cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

        # ── Hold-out metrics ─────────────────────────────────────────────────
        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred),          4),
            "precision": round(precision_score(y_test, y_pred),         4),
            "recall":    round(recall_score(y_test, y_pred),             4),
            "f1_score":  round(f1_score(y_test, y_pred),                 4),
            "roc_auc":   round(roc_auc_score(y_test, y_prob),            4),
            "cv_f1_mean": round(float(cv_f1.mean()),                     4),
            "cv_f1_std":  round(float(cv_f1.std()),                      4),
        }
        all_metrics[name] = metrics

        log.info(
            f"  Accuracy : {metrics['accuracy']:.4f}\n"
            f"  Precision: {metrics['precision']:.4f}\n"
            f"  Recall   : {metrics['recall']:.4f}\n"
            f"  F1 Score : {metrics['f1_score']:.4f}\n"
            f"  ROC-AUC  : {metrics['roc_auc']:.4f}"
        )
        log.info("Classification Report:\n" +
                 classification_report(y_test, y_pred,
                                       target_names=["Ham", "Spam"]))

        cm = confusion_matrix(y_test, y_pred)
        log.info(f"Confusion Matrix:\n"
                 f"             Pred Ham   Pred Spam\n"
                 f"  True Ham   {cm[0,0]:<10} {cm[0,1]}\n"
                 f"  True Spam  {cm[1,0]:<10} {cm[1,1]}")

        if metrics["f1_score"] > best_f1:
            best_f1   = metrics["f1_score"]
            best_name = name
            best_pipe = pipe

    log.info(f"\n{separator}")
    log.info(f"✅  Best model: {best_name}  (F1 = {best_f1:.4f})")

    return {
        "best_model_name": best_name,
        "best_pipeline":   best_pipe,
        "all_pipelines":   pipelines,      # every trained pipeline
        "metrics":         all_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SAVE ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════════
def save_artefacts(result: dict, df: pd.DataFrame) -> None:
    """
    Persist the best pipeline + metadata so app.py can load them.

    Saved files
    -----------
    model/best_pipeline.pkl   – the winning sklearn Pipeline (joblib)
    model/all_pipelines.pkl   – all three trained pipelines
    model/metadata.json       – metrics, training stats, column info
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Pipelines ─────────────────────────────────────────────────────────────
    joblib.dump(result["best_pipeline"],
                os.path.join(MODEL_DIR, "best_pipeline.pkl"))
    log.info(f"Saved best_pipeline.pkl  ({result['best_model_name']})")

    joblib.dump(result["all_pipelines"],
                os.path.join(MODEL_DIR, "all_pipelines.pkl"))
    log.info("Saved all_pipelines.pkl")

    # ── Metadata ──────────────────────────────────────────────────────────────
    spam_n = int(df["Spam"].sum())
    ham_n  = len(df) - spam_n

    metadata = {
        "best_model":    result["best_model_name"],
        "metrics":       result["metrics"],
        "training_stats": {
            "total_samples":   len(df),
            "spam_count":      spam_n,
            "ham_count":       ham_n,
            "imbalance_ratio": round(ham_n / max(spam_n, 1), 2),
            "trained_at":      datetime.now().isoformat(),
        },
    }

    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    log.info(f"Saved metadata.json  →  {meta_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Train SpamShield AI models from a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py                          # uses backend/data/spam.csv
  python train_model.py --data data/myfile.csv  # custom path
  python train_model.py --cv 3                  # 3-fold cross-validation
        """,
    )
    parser.add_argument(
        "--data", "-d",
        default=DEFAULT_DATA,
        help=f"Path to the CSV file  (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds  (default: 5)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  SpamShield AI — Model Training")
    log.info("=" * 60)
    log.info(f"CSV path : {args.data}")
    log.info(f"CV folds : {args.cv}")
    log.info(f"Model dir: {MODEL_DIR}")

    try:
        df     = load_csv(args.data)
        df     = clean_data(df)
        result = train_and_evaluate(df, cv_folds=args.cv)
        save_artefacts(result, df)
    except (FileNotFoundError, ValueError) as exc:
        log.error(str(exc))
        sys.exit(1)

    log.info("\n🎉  Training complete!  Run  python app.py  to start the API.")


if __name__ == "__main__":
    main()