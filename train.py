import argparse
import csv
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.data import download_spamassassin
from src.model import build_pipeline

def load_csv(path, text_col, label_col):
    texts, labels = [], []
    label_mapping = {"1": "spam", "0": "ham", "not spam": "ham", "legitimate": "ham"}
    with open(path, encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or text_col not in reader.fieldnames or label_col not in reader.fieldnames:
            raise ValueError(f"CSV must contain '{text_col}' and '{label_col}' columns.")
        for row in reader:
            text = row.get(text_col)
            label = row.get(label_col)
            if not text or not label or not text.strip() or not label.strip():
                continue
            label = label.strip().lower()
            label = label_mapping.get(label, label)
            if label in ("spam", "ham"):
                texts.append(text)
                labels.append(label)
    return texts, labels

def main():
    parser = argparse.ArgumentParser(description="Train an email spam classifier.")
    parser.add_argument("--csv", help="Optional CSV dataset path.")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()

    if args.csv:
        texts, labels = load_csv(args.csv, args.text_column, args.label_column)
    else:
        print("Downloading and parsing Apache SpamAssassin public email corpus...")
        texts, labels = download_spamassassin()

    if len(set(labels)) < 2:
        raise ValueError("Training data must contain both spam and ham examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    model = build_pipeline()
    print(f"Training on {len(X_train):,} emails...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nTest accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nClassification report:\n", classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix [ham, spam]:\n", confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    print("\nRunning 5-fold cross-validation (may take a few minutes)...")
    cv = cross_val_score(build_pipeline(), texts, labels, cv=5, scoring="f1_macro", n_jobs=-1)
    print("CV macro-F1:", [round(float(x), 4) for x in cv])
    print("Mean CV macro-F1:", round(float(cv.mean()), 4))

    out = Path("models")
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "spam_model.joblib")
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    (report_dir / "evaluation.txt").write_text(
        f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n\n"
        + classification_report(y_test, predictions, zero_division=0),
        encoding="utf-8"
    )
    print("\nSaved model to models/spam_model.joblib")

if __name__ == "__main__":
    main()
