# ============================================================
# baselines/classical.py
# Baseline 3: Naive Bayes + TF-IDF
# Baseline 4: Random Forest + Stylometric Features
#
# These classical ML baselines are used in Papers 1, 2, and 3
# of the literature review. They provide interpretability:
# NB shows WHICH words discriminate human vs AI text,
# RF shows WHICH stylometric features matter most.
# ============================================================

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import SEED
from utils.data_utils import extract_stylometric_features, STYLO_FEATURE_NAMES


def run_naive_bayes(combined_df, exp_name):
    """
    Naive Bayes with TF-IDF features (1-2 grams, top-50k).
    Also prints top discriminating words for thesis analysis.

    Returns: (accuracy, f1_score)
    """
    texts  = combined_df['generated_review'].astype(str).tolist()
    labels = combined_df['label'].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED
    )
    clf = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2), max_features=50000,
            sublinear_tf=True, min_df=2
        )),
        ('nb', MultinomialNB(alpha=0.1))
    ])
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)

    # Print top discriminating words — useful for thesis analysis
    feature_names = clf.named_steps['tfidf'].get_feature_names_out()
    log_diff      = (clf.named_steps['nb'].feature_log_prob_[1]
                   - clf.named_steps['nb'].feature_log_prob_[0])
    print(f'  Top AI words    : {list(feature_names[log_diff.argsort()[-10:][::-1]])}')
    print(f'  Top Human words : {list(feature_names[log_diff.argsort()[:10]])}')

    return accuracy_score(y_te, preds), f1_score(y_te, preds)


def run_random_forest(combined_df, exp_name):
    """
    Random Forest with 14 stylometric features.
    Same feature set as Papers 1, 2, 3 in the literature review.
    Also prints feature importances for thesis analysis.

    Returns: (accuracy, f1_score)
    """
    texts  = combined_df['generated_review'].astype(str).tolist()
    labels = combined_df['label'].tolist()
    X = extract_stylometric_features(texts)
    y = np.array(labels)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)
    clf    = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2,
        random_state=SEED, n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)

    # Feature importances — useful for thesis
    imp     = clf.feature_importances_
    top_idx = imp.argsort()[-5:][::-1]
    print(f'  Top features: {[(STYLO_FEATURE_NAMES[i], round(imp[i],3)) for i in top_idx]}')

    return accuracy_score(y_te, preds), f1_score(y_te, preds)
