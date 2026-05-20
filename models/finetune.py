# ============================================================
# models/finetune.py
# Proposed Methods 1 & 2: Fine-tuned RoBERTa and DeBERTa
#
# Method 1: roberta-base fine-tuned      (~85-90% expected)
# Method 2: microsoft/deberta-v3-base    (~88-93% expected)
#
# Key fixes applied (prevents DeBERTa NaN):
#   - FP32 throughout (no BF16)
#   - _initialize_head() with normal(0, 0.02)
#   - Dual LR: backbone=2e-5, head=1e-4
#   - NaN/Inf guard per batch
#   - num_workers=0
# ============================================================

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import SEED
from utils.data_utils import ReviewDataset, _make_dl, _initialize_head
from utils.train_utils import train_loop, eval_loop


def fine_tune_and_eval(combined_df, model_name, method_label, device):
    """
    Fine-tune a transformer classifier (roberta-base or deberta-v3-base).

    Args:
        combined_df:  DataFrame with 'generated_review' and 'label'
        model_name:   HuggingFace model name string
        method_label: display label for progress output
        device:       torch device

    Returns:
        (accuracy, f1_score)
    """
    print(f'  Fine-tuning {method_label} ...')
    texts  = combined_df['generated_review'].astype(str).tolist()
    labels = combined_df['label'].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dl = _make_dl(ReviewDataset(X_tr, y_tr, tokenizer), shuffle=True)
    test_dl  = _make_dl(ReviewDataset(X_te, y_te, tokenizer), shuffle=False)

    # Load in FP32 — no BF16 cast
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    # Fix DeBERTa NaN: reinitialize head with normal(0, 0.02)
    _initialize_head(model)

    opt, sch = train_loop(model, train_dl, device)
    all_labels, all_preds = eval_loop(model, test_dl, device)

    del model, opt, sch
    torch.cuda.empty_cache()

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    print(f'    Result: acc={acc:.4f}  f1={f1:.4f}')
    return acc, f1
