#!/usr/bin/env python3
# ============================================================
# main.py — SAFE Framework Entry Point
#
# Run all experiments:
#   python main.py
#
# Run specific categories only:
#   python main.py --categories Books Electronics
#
# Skip specific methods:
#   python main.py --skip gltr detectgpt
#
# Quick test (1 category, 1 strategy, 50 samples):
#   python main.py --quick
# ============================================================

import os
import random
import argparse
import warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from config import (
    SEED, DEVICE,
    CATEGORY_FILES, AI_STRATEGIES,
    DEBERTA_NAME, GLM_NAME,
    RUN_GLTR, RUN_DETECTGPT, RUN_NB, RUN_RF,
    RUN_OPENAI, RUN_ROBERTA, RUN_DEBERTA,
    RUN_SAFE_TEXTONLY, RUN_SAFE_STYLO,
    RESULTS_DIR, RESULTS_LIVE_CSV
)
from utils.data_utils import load_category, build_binary_dataset
from baselines.gltr import run_gltr
from baselines.detectgpt import run_detectgpt
from baselines.classical import run_naive_bayes, run_random_forest
from baselines.openai_detector import load_openai_detector, run_openai_detector
from models.finetune import fine_tune_and_eval
from models.safe_textonly import train_safe_textonly, eval_safe_textonly
from models.safe_stylometric import run_safe_stylometric
from results.evaluate import save_results, print_tables, plot_results, print_analysis


def setup():
    """Set seeds and device."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if DEVICE is not None:
        device = torch.device(DEVICE)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')
        print(f'VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return device


def load_reference_model(device):
    """Load gpt-neo-1.3B for GLTR and DetectGPT."""
    if not (RUN_GLTR or RUN_DETECTGPT):
        return None, None
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f'Loading {GLM_NAME} (~2.6GB) ...')
    tokenizer = AutoTokenizer.from_pretrained(GLM_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model     = AutoModelForCausalLM.from_pretrained(
        GLM_NAME, torch_dtype=torch.float16
    ).to(device).eval()
    print(f'Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='SAFE Detection Framework')
    parser.add_argument('--categories', nargs='+', default=None,
                        help='Run only these categories (e.g. Books Electronics)')
    parser.add_argument('--skip', nargs='+', default=[],
                        choices=['gltr','detectgpt','nb','rf','openai',
                                 'roberta','deberta','safe_textonly','safe_stylo'],
                        help='Skip specific methods')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 1 category, Zero-Shot only, 50 samples')
    return parser.parse_args()


def main():
    args   = setup_args()  # parse_args() alias below
    device = setup()

    # Method flags (respect config.py + CLI --skip)
    run = {
        'gltr'          : RUN_GLTR          and 'gltr'          not in args.skip,
        'detectgpt'     : RUN_DETECTGPT     and 'detectgpt'     not in args.skip,
        'nb'            : RUN_NB            and 'nb'            not in args.skip,
        'rf'            : RUN_RF            and 'rf'            not in args.skip,
        'openai'        : RUN_OPENAI        and 'openai'        not in args.skip,
        'roberta'       : RUN_ROBERTA       and 'roberta'       not in args.skip,
        'deberta'       : RUN_DEBERTA       and 'deberta'       not in args.skip,
        'safe_textonly' : RUN_SAFE_TEXTONLY and 'safe_textonly' not in args.skip,
        'safe_stylo'    : RUN_SAFE_STYLO    and 'safe_stylo'    not in args.skip,
    }

    # Quick mode overrides
    if args.quick:
        import config
        config.SAMPLE_SIZE = 50
        print('[QUICK MODE] sample_size=50, Books only, Zero-Shot only')

    # Load reference model once (shared by GLTR + DetectGPT)
    glm_model, glm_tokenizer = load_reference_model(device)

    # Load OpenAI detector once (reused across experiments)
    openai_model = openai_tokenizer = None
    if run['openai']:
        openai_model, openai_tokenizer = load_openai_detector(device)

    # Select categories
    cat_files = CATEGORY_FILES
    if args.categories:
        cat_files = {k: v for k, v in CATEGORY_FILES.items()
                     if k in args.categories}
    if args.quick:
        cat_files = {'Books': CATEGORY_FILES['Books']}

    all_results = []

    def record(cat, method, comparison, acc, f1):
        print(f'    {method:34s}  acc={acc:.4f}  f1={f1:.4f}')
        all_results.append({
            'Category'  : cat,
            'Method'    : method,
            'Comparison': comparison,
            'Accuracy'  : round(acc, 4),
            'F1'        : round(f1,  4)
        })
        # Live save after every result
        pd.DataFrame(all_results).to_csv(RESULTS_LIVE_CSV, index=False)

    strategies = ['Zero-Shot'] if args.quick else AI_STRATEGIES

    for cat_name, csv_path in cat_files.items():
        if not os.path.exists(csv_path):
            print(f'[SKIP] {cat_name}: {csv_path} not found')
            continue

        print(f'\n{"="*70}')
        print(f'  CATEGORY: {cat_name}')
        print(f'{"="*70}')
        splits   = load_category(csv_path)
        human_df = splits.get('Human', pd.DataFrame())
        if human_df.empty:
            print('  [SKIP] No human reviews.'); continue

        # Train SAFE Text-Only ONCE per category
        if run['safe_textonly']:
            train_safe_textonly(splits, cat_name, device)

        for strategy in strategies:
            ai_df = splits.get(strategy, pd.DataFrame())
            if ai_df.empty:
                print(f'  [SKIP] {strategy}: no data.'); continue

            exp_name = f'Human vs {strategy}'
            combined = build_binary_dataset(human_df, ai_df)
            print(f'\n  ── {exp_name}  (n={len(combined)}) ──')

            if run['gltr']:
                record(cat_name, 'GLTR (gpt-neo-1.3B)', exp_name,
                       *run_gltr(combined, exp_name, glm_model, glm_tokenizer, device))

            if run['detectgpt']:
                record(cat_name, 'DetectGPT (span-deletion)', exp_name,
                       *run_detectgpt(combined, exp_name, glm_model, glm_tokenizer, device))

            if run['nb']:
                record(cat_name, 'Naive Bayes + TF-IDF', exp_name,
                       *run_naive_bayes(combined, exp_name))

            if run['rf']:
                record(cat_name, 'Random Forest + Stylometric', exp_name,
                       *run_random_forest(combined, exp_name))

            if run['openai']:
                record(cat_name, 'OpenAI Detector (head FT)', exp_name,
                       *run_openai_detector(combined, exp_name,
                                            openai_model, openai_tokenizer, device))

            if run['roberta']:
                record(cat_name, 'RoBERTa-base (fine-tuned)', exp_name,
                       *fine_tune_and_eval(combined, 'roberta-base', 'RoBERTa-base', device))

            if run['deberta']:
                record(cat_name, 'DeBERTa-v3-base (fine-tuned)', exp_name,
                       *fine_tune_and_eval(combined, DEBERTA_NAME, 'DeBERTa-v3-base', device))

            if run['safe_textonly']:
                record(cat_name, 'SAFE Text-Only', exp_name,
                       *eval_safe_textonly(combined, cat_name, exp_name, device))

            if run['safe_stylo']:
                record(cat_name, 'SAFE+Stylometric', exp_name,
                       *run_safe_stylometric(combined, exp_name, device))

    print('\n' + '='*70)
    print('  ALL EXPERIMENTS COMPLETE')
    print('='*70)

    if all_results:
        results_df, _ = save_results(all_results)
        print_tables(results_df)
        plot_results(results_df)
        print_analysis(results_df)
    else:
        print('No results to save.')


def setup_args():
    """Wrapper so main() can call parse_args safely."""
    try:
        return parse_args()
    except SystemExit:
        # Running interactively (e.g. in a notebook importing main)
        import argparse
        return argparse.Namespace(categories=None, skip=[], quick=False)


if __name__ == '__main__':
    main()
