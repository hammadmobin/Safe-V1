# ============================================================
# utils/data_utils.py
# Data loading, dataset classes, and stylometric features
# ============================================================

import re
import random
import numpy as np
import pandas as pd
import textstat
import torch
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader

from config import (
    SEED, MAX_LEN, SAMPLE_SIZE, FINETUNE_BATCH, EVAL_BATCH,
    PROMPT_TYPES, AI_STRATEGIES
)

# ── Stylometric feature names ─────────────────────────────────
STYLO_FEATURE_NAMES = [
    'word_count', 'char_count', 'avg_word_len', 'unique_word_ratio',
    'avg_sentence_len', 'sentence_count',
    'flesch_ease', 'fk_grade', 'gunning_fog',
    'exclamation_ratio', 'question_ratio',
    'digit_ratio', 'upper_ratio', 'punct_ratio'
]


def load_category(csv_path):
    """Load a category CSV and split by prompt_type."""
    df = pd.read_csv(csv_path)
    splits = {}
    for raw_key, short_key in PROMPT_TYPES.items():
        sub = df[df['prompt_type'] == raw_key][['generated_review']].dropna().copy()
        sub['label'] = 0 if short_key == 'Human' else 1
        splits[short_key] = sub
        print(f'  {short_key:28s}: {len(sub):>5} rows')
    return splits


def build_binary_dataset(human_df, ai_df, sample_size=SAMPLE_SIZE):
    """Balance human vs AI reviews and combine."""
    n = min(len(human_df), len(ai_df), sample_size)
    h = human_df.sample(n=n, random_state=SEED)
    a = ai_df.sample(n=n, random_state=SEED)
    return pd.concat([h, a]).sample(frac=1, random_state=SEED).reset_index(drop=True)


def extract_stylometric_features(texts):
    """
    Extract 14 handcrafted stylometric features per review.

    Features:
      Lexical:     word_count, char_count, avg_word_len,
                   unique_word_ratio, avg_sentence_len, sentence_count
      Readability: flesch_ease, fk_grade, gunning_fog
      Stylistic:   exclamation_ratio, question_ratio,
                   digit_ratio, upper_ratio, punct_ratio
    """
    records = []
    for text in texts:
        t       = str(text)
        words   = t.split()
        n_words = len(words)
        n_chars = len(t)
        sents   = re.split(r'[.!?]+', t)
        n_sents = max(len([s for s in sents if s.strip()]), 1)
        try:
            fe = textstat.flesch_reading_ease(t)
            fk = textstat.flesch_kincaid_grade(t)
            gf = textstat.gunning_fog(t)
        except Exception:
            fe = fk = gf = 0.0
        records.append([
            n_words, n_chars,
            np.mean([len(w) for w in words]) if words else 0,
            len(set(w.lower() for w in words)) / (n_words + 1),
            n_words / n_sents, n_sents,
            fe, fk, gf,
            t.count('!') / (n_words + 1),
            t.count('?') / (n_words + 1),
            sum(c.isdigit() for c in t) / (n_chars + 1),
            sum(c.isupper() for c in t) / (n_chars + 1),
            sum(not c.isalnum() and not c.isspace() for c in t) / (n_chars + 1),
        ])
    return np.array(records, dtype=np.float32)


def _make_dl(dataset, shuffle):
    """
    DataLoader factory — always num_workers=0.
    Fixes Python 3.12 + PyTorch multiprocessing crash on Colab.
    """
    return DataLoader(
        dataset,
        batch_size=FINETUNE_BATCH if shuffle else EVAL_BATCH,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )


def _initialize_head(model):
    """
    Reinitialize classifier/pooler/stylo_mlp layers with normal(0, 0.02).

    WHY: HuggingFace initializes MISSING classifier heads with kaiming_uniform_.
    For DeBERTa (hidden_size=768), this can produce large initial logits
    → NaN loss on first backward pass.
    normal(0, 0.02) matches DeBERTa pretraining initialization.
    """
    for name, module in model.named_modules():
        if any(k in name for k in ['classifier', 'pooler', 'stylo_mlp']):
            if hasattr(module, 'weight') and module.weight is not None:
                init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                init.zeros_(module.bias)


class ReviewDataset(Dataset):
    """Standard text classification dataset."""
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            [str(t) for t in texts],
            truncation=True, padding='max_length',
            max_length=MAX_LEN, return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids'      : self.enc['input_ids'][idx],
            'attention_mask' : self.enc['attention_mask'][idx],
            'labels'         : self.labels[idx]
        }


class StyloDataset(Dataset):
    """Text + stylometric features dataset for SAFE+Stylometric."""
    def __init__(self, texts, stylo_feats, labels, tokenizer):
        self.enc = tokenizer(
            [str(t) for t in texts],
            truncation=True, padding='max_length',
            max_length=MAX_LEN, return_tensors='pt'
        )
        self.stylo  = torch.tensor(stylo_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids'      : self.enc['input_ids'][idx],
            'attention_mask' : self.enc['attention_mask'][idx],
            'stylo'          : self.stylo[idx],
            'labels'         : self.labels[idx]
        }
