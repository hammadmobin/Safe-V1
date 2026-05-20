# ============================================================
# utils/train_utils.py
# Core training/evaluation utilities shared by all fine-tuning methods
# ============================================================

import torch
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from config import (
    BACKBONE_LR, HEAD_LR, WEIGHT_DECAY, EPOCHS,
    WARMUP_RATIO, GRAD_CLIP
)


def get_optimizer(model):
    """
    Dual learning rate optimizer.
    - Backbone (pre-trained): BACKBONE_LR = 2e-5
    - Classifier/pooler head (randomly initialized): HEAD_LR = 1e-4

    WHY dual LR: The classifier head is randomly initialized (MISSING in
    HuggingFace load report). It needs a higher LR to learn quickly in
    3 epochs. The backbone needs a low LR to preserve pre-trained weights.
    """
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if any(k in name for k in ['classifier', 'pooler', 'stylo_mlp']):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': BACKBONE_LR},
        {'params': head_params,     'lr': HEAD_LR},
    ], weight_decay=WEIGHT_DECAY, eps=1e-8)


def train_loop(model, train_dl, device, n_epochs=EPOCHS):
    """
    Standard training loop with all stability fixes:
    - FP32 throughout (no BF16 during training)
    - Linear warmup (WARMUP_RATIO of total steps)
    - NaN/Inf guard: skips batch if loss is bad
    - Gradient clipping at GRAD_CLIP=1.0
    - Reports valid batch count per epoch

    Returns: (optimizer, scheduler) for cleanup
    """
    optimizer   = get_optimizer(model)
    total_steps = len(train_dl) * n_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps
    )
    model.train()
    for epoch in range(n_epochs):
        total_loss, steps = 0, 0
        for batch in tqdm(train_dl,
                          desc=f'    Epoch {epoch+1}/{n_epochs}', leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch['labels'] = batch['labels'].long()
            loss = model(**batch).loss
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            steps += 1
        print(f'    Epoch {epoch+1} loss: {total_loss/max(steps,1):.4f}'
              f'  ({steps}/{len(train_dl)} valid batches)')
    return optimizer, scheduler


def eval_loop(model, test_dl, device):
    """Standard evaluation loop. Returns (all_labels, all_preds)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for batch in test_dl:
            lbls  = batch.pop('labels').numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls)
    return all_labels, all_preds
