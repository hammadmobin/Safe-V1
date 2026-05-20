# ============================================================
# baselines/openai_detector.py
# Baseline 5: OpenAI Detector (roberta-base-openai-detector)
#
# Zero-shot accuracy: <50% (worse than random)
# WHY: The model was trained to detect GPT-2 outputs.
#      GPT-4.1-mini generates much more human-like text.
#
# Fix: Freeze the RoBERTa backbone. Fine-tune only the
# 2-class classifier head on our data. This adapts the
# output layer to GPT-4.1-mini patterns while keeping
# the pre-trained text representations intact.
#
# Expected after fix: ~72-78%
# ============================================================

import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import SEED, OPENAI_DET_MODEL
from utils.data_utils import ReviewDataset, _make_dl


def load_openai_detector(device):
    """Load the OpenAI detector model and tokenizer."""
    print(f'Loading {OPENAI_DET_MODEL} ...')
    tokenizer = AutoTokenizer.from_pretrained(OPENAI_DET_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        OPENAI_DET_MODEL
    ).to(device)
    print(f'Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
    return model, tokenizer


def run_openai_detector(combined_df, exp_name, model, tokenizer, device):
    """
    Fine-tune only the classifier head (backbone frozen).
    Args:
        combined_df: DataFrame with 'generated_review' and 'label'
        exp_name:    experiment label for display
        model:       loaded openai-detector model
        tokenizer:   corresponding tokenizer
        device:      torch device
    Returns:
        (accuracy, f1_score)
    """
    texts  = combined_df['generated_review'].astype(str).tolist()
    labels = combined_df['label'].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED
    )

    # Freeze backbone — only train classifier head
    for name, param in model.named_parameters():
        param.requires_grad = ('classifier' in name)

    train_dl = _make_dl(ReviewDataset(X_tr, y_tr, tokenizer), shuffle=True)
    test_dl  = _make_dl(ReviewDataset(X_te, y_te, tokenizer), shuffle=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, eps=1e-8
    )
    model.train()
    for epoch in range(2):   # 2 epochs sufficient for head-only
        total_loss, steps = 0, 0
        for batch in tqdm(train_dl,
                          desc=f'  OpenAI-Det ep{epoch+1}', leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch['labels'] = batch['labels'].long()
            loss = model(**batch).loss
            if torch.isnan(loss):
                optimizer.zero_grad(); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
            total_loss += loss.item(); steps += 1
        print(f'    Epoch {epoch+1} loss: {total_loss/max(steps,1):.4f}')

    # Re-enable all params for next experiment
    for param in model.parameters():
        param.requires_grad = True

    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for batch in test_dl:
            lbls  = batch.pop('labels').numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds); all_labels.extend(lbls)

    torch.cuda.empty_cache()
    return (accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, zero_division=0))
