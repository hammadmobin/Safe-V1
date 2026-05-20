# ============================================================
# models/safe_textonly.py
# Proposed Method 3: SAFE Text-Only
#
# Key difference from DeBERTa fine-tuned (Method 2):
#   Method 2: trains a SEPARATE model per experiment
#             (e.g., Human vs Zero-Shot separately)
#             → may overfit to one generation style
#
#   SAFE Text-Only: trains ONE model on ALL 4 strategies combined
#             (Zero-Shot + Few-Shot + Facet-Aware + Replication)
#             → generalizable detector
#
# More realistic for real-world deployment: you don't know
# which prompting strategy was used to generate a fake review.
#
# Training: done ONCE per product category
# Evaluation: done separately for each prompting strategy
# ============================================================

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import SEED, DEBERTA_NAME, SAMPLE_SIZE, AI_STRATEGIES
from utils.data_utils import ReviewDataset, _make_dl, _initialize_head
from utils.train_utils import train_loop, eval_loop
import pandas as pd

# Global cache: category_name → (model, tokenizer)
_safe_textonly_models = {}


def train_safe_textonly(splits, category_name, device):
    """
    Train one DeBERTa on all AI strategies combined.
    Call ONCE per category before per-strategy evaluation.

    Args:
        splits:        dict from load_category() — {strategy: DataFrame}
        category_name: product category name (used as cache key)
        device:        torch device
    """
    print(f'  Training SAFE Text-Only [{category_name}] ...')
    human_df = splits.get('Human', pd.DataFrame())
    ai_parts = [splits[s] for s in AI_STRATEGIES
                if s in splits and not splits[s].empty]
    if not ai_parts:
        print('  [SKIP] No AI data.'); return

    ai_combined = pd.concat([
        p.sample(n=min(len(p), SAMPLE_SIZE), random_state=SEED)
        for p in ai_parts
    ])
    human_combined = human_df.sample(
        n=min(len(human_df), len(ai_combined)), random_state=SEED
    )
    combined = pd.concat([human_combined, ai_combined]).sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)
    print(f'    Size: {len(combined)} '
          f'({combined["label"].sum()} AI / '
          f'{(combined["label"]==0).sum()} Human)')

    texts  = combined['generated_review'].astype(str).tolist()
    labels = combined['label'].tolist()
    X_tr, _, y_tr, _ = train_test_split(
        texts, labels, test_size=0.1, random_state=SEED
    )

    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_NAME)
    train_dl  = _make_dl(ReviewDataset(X_tr, y_tr, tokenizer), shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        DEBERTA_NAME, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    _initialize_head(model)   # Fix DeBERTa NaN

    opt, sch = train_loop(model, train_dl, device)
    model.eval()
    _safe_textonly_models[category_name] = (model, tokenizer)
    del opt, sch
    torch.cuda.empty_cache()
    print(f'  Stored [{category_name}].')


def eval_safe_textonly(combined_df, category_name, exp_name, device):
    """
    Evaluate the stored SAFE Text-Only model on one strategy.

    Args:
        combined_df:   DataFrame with 'generated_review' and 'label'
        category_name: must match what was used in train_safe_textonly
        exp_name:      experiment label for display
        device:        torch device

    Returns:
        (accuracy, f1_score)
    """
    if category_name not in _safe_textonly_models:
        print(f'  [SKIP] SAFE Text-Only not trained for {category_name}')
        return 0.0, 0.0

    model, tokenizer = _safe_textonly_models[category_name]
    texts  = combined_df['generated_review'].astype(str).tolist()
    labels = combined_df['label'].tolist()
    test_dl = _make_dl(ReviewDataset(texts, labels, tokenizer), shuffle=False)
    all_labels, all_preds = eval_loop(model, test_dl, device)
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds)
