# ============================================================
# baselines/gltr.py
# Baseline 1: Extended GLTR (8-feature version)
#
# Reference: Gehrmann et al. 2019 — original used 4 bin features.
# Extended version adds mean rank, std rank, entropy, top-1 prob
# for stronger discrimination against GPT-4.1-mini outputs.
#
# Reference model: EleutherAI/gpt-neo-1.3B
# WHY: GPT-2 (~50% accuracy) assigns similar probabilities to both
# human and GPT-4.1-mini text. GPT-Neo 1.3B (trained on The Pile,
# same corpus family as GPT-3/4) provides a measurable signal.
#
# Features:
#   [0] frac tokens in top-10      (AI: higher — more predictable)
#   [1] frac tokens in top-11..100
#   [2] frac tokens in top-101..1000
#   [3] frac tokens above rank-1000 (human: higher — more varied)
#   [4] mean rank                   (AI: lower)
#   [5] std of ranks                (human: higher)
#   [6] mean token entropy          (human: higher)
#   [7] mean top-1 probability      (AI: higher)
# ============================================================

import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import SEED, GLTR_BATCH, TOPK, MAX_LEN


def _gltr_features_batch(texts, glm_model, glm_tokenizer, device):
    enc = glm_tokenizer(
        [str(t) for t in texts],
        return_tensors='pt', padding=True,
        truncation=True, max_length=MAX_LEN
    ).to(device)
    ids, mask = enc['input_ids'], enc['attention_mask']
    with torch.inference_mode():
        logits = glm_model(input_ids=ids, attention_mask=mask).logits

    feats = []
    for b in range(ids.shape[0]):
        seq_len = mask[b].sum().item()
        ranks, entropies, top1_probs = [], [], []
        for i in range(1, seq_len):
            probs      = torch.softmax(logits[b, i-1], dim=-1)
            _, top_ids = torch.topk(probs, k=TOPK)
            tok        = ids[b, i].item()
            pos        = (top_ids == tok).nonzero(as_tuple=True)
            ranks.append(pos[0].item() if len(pos[0]) else TOPK + 1)
            p = probs.float()
            entropies.append(-(p * (p + 1e-10).log()).sum().item())
            top1_probs.append(probs.max().item())
        if not ranks:
            feats.append([0.0] * 8)
            continue
        ranks = np.array(ranks)
        feats.append([
            float(np.mean(ranks < 10)),
            float(np.mean((ranks >= 10)  & (ranks < 100))),
            float(np.mean((ranks >= 100) & (ranks < 1000))),
            float(np.mean(ranks >= 1000)),
            float(np.mean(ranks)),
            float(np.std(ranks)),
            float(np.mean(entropies)),
            float(np.mean(top1_probs)),
        ])
    return feats


def run_gltr(combined_df, exp_name, glm_model, glm_tokenizer, device):
    """
    Run GLTR baseline.
    Args:
        combined_df: DataFrame with 'generated_review' and 'label' columns
        exp_name:    experiment label for tqdm display
        glm_model:   loaded gpt-neo-1.3B model
        glm_tokenizer: corresponding tokenizer
        device:      torch device
    Returns:
        (accuracy, f1_score)
    """
    texts  = combined_df['generated_review'].tolist()
    labels = combined_df['label'].tolist()
    feats  = []
    for i in tqdm(range(0, len(texts), GLTR_BATCH),
                  desc=f'GLTR [{exp_name}]', leave=False):
        feats.extend(_gltr_features_batch(
            texts[i:i+GLTR_BATCH], glm_model, glm_tokenizer, device
        ))
    X, y = np.array(feats), np.array(labels)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)
    preds  = LogisticRegression(max_iter=1000, C=1.0).fit(X_tr, y_tr).predict(X_te)
    return accuracy_score(y_te, preds), f1_score(y_te, preds)
