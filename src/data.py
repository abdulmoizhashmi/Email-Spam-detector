from pathlib import Path
from urllib.request import urlretrieve
import tarfile
import email
from email import policy
from email.message import Message
import re

BASE = "https://spamassassin.apache.org/old/publiccorpus/"
ARCHIVES = {
    "easy_ham": "20030228_easy_ham.tar.bz2",
    "hard_ham": "20030228_hard_ham.tar.bz2",
    "spam": "20030228_spam.tar.bz2",
}

def _decode_part(part: Message) -> str:
    try:
        payload = part.get_content()
        if isinstance(payload, str):
            return payload
    except Exception:
        pass
    try:
        raw = part.get_payload(decode=True)
        if raw:
            return raw.decode(part.get_content_charset() or "utf-8", errors="replace")
    except Exception:
        pass
    return ""

def extract_email_text(path: Path) -> str:
    try:
        msg = email.message_from_bytes(path.read_bytes(), policy=policy.default)
        subject = str(msg.get("subject", ""))
        chunks = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain" and part.get_content_disposition() != "attachment":
                    chunks.append(_decode_part(part))
        elif msg.get_content_type() == "text/plain":
            chunks.append(_decode_part(msg))
        body = "\n".join(chunks)
        # Remove URLs and collapse excessive whitespace without stripping useful wording.
        text = f"{subject}\n{body}"
        text = re.sub(r"https?://\S+|www\.\S+", " URL ", text, flags=re.I)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""

def download_spamassassin(data_dir: str = "data") -> tuple[list[str], list[str]]:
    data = Path(data_dir)
    raw_dir = data / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    texts, labels = [], []
    for label, archive in ARCHIVES.items():
        archive_path = raw_dir / archive
        if not archive_path.exists():
            urlretrieve(BASE + archive, archive_path)
        extract_dir = raw_dir / label
        extract_dir.mkdir(exist_ok=True)
        # Extract only files from the expected archive into its own folder.
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(extract_dir, filter="data")
        for p in extract_dir.rglob("*"):
            if p.is_file() and not p.name.startswith("."):
                text = extract_email_text(p)
                if len(text) >= 20:
                    texts.append(text)
                    labels.append("spam" if label == "spam" else "ham")
    if not texts:
        raise RuntimeError("No emails were parsed from the corpus.")
    return texts, labels
