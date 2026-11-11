# Email Spam Detection (TF-IDF + Logistic Regression)

A complete, editable Python project for classifying email text as **Spam** or **Ham (Not Spam)**.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train.py
streamlit run app.py
```

The trained model is included at `models/spam_model.joblib`, so the app runs without downloading or training a model. `train.py` can download the public Apache SpamAssassin email corpus, parse MIME email messages, train a TF-IDF + Logistic Regression pipeline, evaluate it, and save a replacement model.

## Deploy the live app

This project is a Python Streamlit app. Deploy it with [Streamlit Community Cloud](https://share.streamlit.io/):

1. Sign in with GitHub and select **Create app**.
2. Select this repository, the `main` branch, and `app.py` as the app file.
3. Choose **Deploy**. Streamlit installs the dependencies from `requirements.txt` and loads the included model.

GitHub Pages only hosts static websites and cannot run the Streamlit/Python app. You can link to the deployed Streamlit app from the repository or a GitHub Pages landing page.

If you already have a CSV dataset, use:
```bash
python train.py --csv path/to/emails.csv --text-column text --label-column label
```
Labels should contain `spam` and `ham` (or `1` and `0`). The default downloader is the Apache SpamAssassin public corpus.

## Features
- Email subject/body text input and prediction.
- Spam probability and model confidence display.
- MIME-aware corpus parsing.
- TF-IDF word and character features.
- Logistic Regression with balanced class weights.
- Train/test split, classification report, confusion matrix and cross-validation.
- Saved model pipeline for reuse.

## Dataset
Apache SpamAssassin Public Corpus:
https://spamassassin.apache.org/old/publiccorpus/

The training script downloads the easy-ham, hard-ham and spam archives directly from that source. Dataset terms and availability are controlled by the source provider. Use the dataset's documentation for research and redistribution terms.

## Project structure
- `app.py` — Streamlit interface
- `train.py` — download, preprocess, train and evaluate
- `src/data.py` — corpus download and parsing
- `src/model.py` — pipeline construction
- `models/` — trained model output
- `reports/` — evaluation output

## Notes
This is an educational classifier, not a production mail gateway. Do not use its score as the sole basis for deleting messages. Real-world performance depends on the training corpus and changing spam patterns.
