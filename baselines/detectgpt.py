# ============================================================
# baselines/detectgpt.py
# Baseline 2: DetectGPT with span-deletion perturbation
#
# Reference: Mitchell et al. 2023 (ICML)
#
# WHY span-deletion instead of mask-fill:
#   The HuggingFace fill-mask pipeline fails silently with
#   multiple [MASK] tokens per sentence — it only fills the first
#   and returns the original for the rest.
#   Result: original ≈ perturbed → curvature score ≈ 0 → ~50% accuracy.
#   Span-deletion removes contiguous word spans or whole sentences,
#   producing meaningful perturbations without a second model.
#
# Scoring model: EleutherAI/gpt-neo-1.3B (shared with GLTR)
# ============================================================

import random
import re
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

from config import SEED, GLTR_BATCH, DETECTGPT_PERTURB, SPAN_DROP_PROB, MAX_LEN


def _batch_log_probs(texts, glm_model, glm_tokenizer, device):
    """Mean negative log-prob per text under gpt-neo-1.3B."""
    import torch.nn as nn
    enc = glm_tokenizer(
        [str(t) for t in texts],
        return_tensors='pt', padding=True,
        truncation=True, max_length=MAX_LEN
    ).to(device)
    with torch.inference_mode():
        logits = glm_model(**enc, labels=enc['input_ids']).logits
    shift_log = logits[:, :-1, :]
    shift_ids = enc['input_ids'][:, 1:]
    loss_fn   = nn.CrossEntropyLoss(reduction='none')
    losses    = loss_fn(
        shift_log.reshape(-1, shift_log.size(-1)),
        shift_ids.reshape(-1)
    ).view(shift_ids.size())
    return (-losses.mean(dim=1)).cpu().numpy()


def _span_perturb(text, drop_prob=SPAN_DROP_PROB):
    """Delete a random contiguous span (sentences or words)."""
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    if len(sentences) > 2:
        n_drop   = max(1, int(len(sentences) * drop_prob))
        drop_idx = set(random.sample(range(len(sentences)),
                                     min(n_drop, len(sentences)-1)))
        kept = [s for i, s in enumerate(sentences) if i not in drop_idx]
        return ' '.join(kept) if kept else text
    else:
        words = str(text).split()
        if len(words) < 5:
            return text
        span_len   = max(1, int(len(words) * drop_prob))
        start      = random.randint(0, max(0, len(words) - span_len))
        kept_words = words[:start] + words[start + span_len:]
        return ' '.join(kept_words) if len(kept_words) >= 3 else text


def _detectgpt_scores_batch(texts, glm_model, glm_tokenizer, device):
    orig_scores  = _batch_log_probs(texts, glm_model, glm_tokenizer, device)
    all_perturbs, counts = [], []
    for t in texts:
        perturbs = [_span_perturb(t) for _ in range(DETECTGPT_PERTURB)]
        all_perturbs.extend(perturbs)
        counts.append(len(perturbs))
    pert_scores = _batch_log_probs(all_perturbs, glm_model, glm_tokenizer, device)
    out, idx = [], 0
    for os, cnt in zip(orig_scores, counts):
        out.append(os - np.mean(pert_scores[idx:idx+cnt]))
        idx += cnt
    return out


def run_detectgpt(combined_df, exp_name, glm_model, glm_tokenizer, device):
    """
    Run DetectGPT baseline.
    Args:
        combined_df:   DataFrame with 'generated_review' and 'label'
        exp_name:      experiment label for tqdm
        glm_model:     loaded gpt-neo-1.3B
        glm_tokenizer: corresponding tokenizer
        device:        torch device
    Returns:
        (accuracy, f1_score)
    """
    texts  = combined_df['generated_review'].tolist()
    labels = combined_df['label'].tolist()
    scores = []
    for i in tqdm(range(0, len(texts), GLTR_BATCH),
                  desc=f'DetectGPT [{exp_name}]', leave=False):
        scores.extend(_detectgpt_scores_batch(
            texts[i:i+GLTR_BATCH], glm_model, glm_tokenizer, device
        ))
    threshold = np.median(scores)
    preds     = [1 if s > threshold else 0 for s in scores]
    return accuracy_score(labels, preds), f1_score(labels, preds)
