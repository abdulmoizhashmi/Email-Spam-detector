from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_pipeline():
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            max_features=100_000, sublinear_tf=True, min_df=2,
            token_pattern=r"(?u)\b\w+\b"
        )),
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), max_features=60_000,
            min_df=2, sublinear_tf=True
        ))
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=1200, class_weight="balanced", C=3.0,
            solver="liblinear", random_state=42
        ))
    ])
