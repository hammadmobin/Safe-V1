# ============================================================
# models/safe_stylometric.py
# SAFE+Stylometric Model Architecture
#
# Architecture:
#   text → DeBERTa-v3-base encoder → [CLS] token (768-dim)
#                                            ↓
#   text → 14 stylometric features → MLP (14→64→32)
#                                            ↓
#   concat [CLS(768) + stylo(32)] = 800-dim
#                                            ↓
#   Classifier (800→256→2)
#
# Why DeBERTa + stylometric fusion:
#   DeBERTa captures deep semantic/syntactic patterns.
#   Stylometric features (readability, lexical diversity,
#   punctuation) capture surface-level writing style differences.
#   Paper 1 (DetectAIRev) showed RoBERTa + stylometric features
#   outperformed RoBERTa alone. SAFE+Stylometric extends this
#   idea using DeBERTa-v3 as the stronger backbone.
# ============================================================

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import SEED, DEBERTA_NAME, EPOCHS, WARMUP_RATIO, GRAD_CLIP
from utils.data_utils import (
    StyloDataset, _make_dl, _initialize_head,
    extract_stylometric_features, STYLO_FEATURE_NAMES
)
from utils.train_utils import get_optimizer


class SAFEStyloModel(nn.Module):
    """
    DeBERTa [CLS] + stylometric MLP fusion classifier.

    Args:
        backbone:   AutoModelForSequenceClassification (deberta-v3-base)
        stylo_dim:  number of stylometric features (default 14)
        hidden:     hidden size of fusion classifier (default 256)
    """
    def __init__(self, backbone, stylo_dim=14, hidden=256):
        super().__init__()
        self.encoder    = backbone.deberta
        self.dropout    = nn.Dropout(0.1)
        hidden_size     = self.encoder.config.hidden_size   # 768

        self.stylo_mlp = nn.Sequential(
            nn.Linear(stylo_dim, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 32),        nn.GELU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 32, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, stylo, labels=None):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls     = self.dropout(enc_out.last_hidden_state[:, 0, :].float())
        stylo_h = self.stylo_mlp(stylo.float())
        logits  = self.classifier(torch.cat([cls, stylo_h], dim=-1))
        loss    = self.loss_fn(logits, labels) if labels is not None else None
        return logits, loss


def run_safe_stylometric(combined_df, exp_name, device):
    """
    Train and evaluate SAFE+Stylometric on one experiment.
    Args:
        combined_df: DataFrame with 'generated_review' and 'label'
        exp_name:    experiment label
        device:      torch device
    Returns:
        (accuracy, f1_score)
    """
    print(f'  SAFE+Stylometric [{exp_name}] ...')
    texts  = combined_df['generated_review'].astype(str).tolist()
    labels = combined_df['label'].tolist()

    # Extract and scale stylometric features
    stylo_all = extract_stylometric_features(texts)
    idx = list(range(len(texts)))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=SEED)
    scaler   = StandardScaler()
    stylo_tr = scaler.fit_transform(stylo_all[tr_idx])
    stylo_te = scaler.transform(stylo_all[te_idx])

    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_NAME)
    train_dl  = _make_dl(
        StyloDataset([texts[i] for i in tr_idx], stylo_tr,
                     [labels[i] for i in tr_idx], tokenizer), shuffle=True
    )
    test_dl = _make_dl(
        StyloDataset([texts[i] for i in te_idx], stylo_te,
                     [labels[i] for i in te_idx], tokenizer), shuffle=False
    )

    backbone   = AutoModelForSequenceClassification.from_pretrained(
        DEBERTA_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    safe_model = SAFEStyloModel(
        backbone, stylo_dim=len(STYLO_FEATURE_NAMES)
    ).to(device)

    _initialize_head(safe_model)   # Fix DeBERTa NaN

    optimizer   = get_optimizer(safe_model)
    total_steps = len(train_dl) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps
    )

    safe_model.train()
    for epoch in range(EPOCHS):
        total_loss, steps = 0, 0
        for batch in tqdm(train_dl,
                          desc=f'    Epoch {epoch+1}/{EPOCHS}', leave=False):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            stylo = batch['stylo'].to(device)
            lbl   = batch['labels'].to(device)
            _, loss = safe_model(ids, mask, stylo, labels=lbl)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(safe_model.parameters(), GRAD_CLIP)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item(); steps += 1
        print(f'    Epoch {epoch+1} loss: {total_loss/max(steps,1):.4f}'
              f'  ({steps}/{len(train_dl)} valid)')

    safe_model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for batch in test_dl:
            lbl   = batch['labels'].numpy()
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            stylo = batch['stylo'].to(device)
            logits, _ = safe_model(ids, mask, stylo)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds); all_labels.extend(lbl)

    del safe_model, backbone, optimizer, scheduler
    torch.cuda.empty_cache()

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    print(f'    Result: acc={acc:.4f}  f1={f1:.4f}')
    return acc, f1
