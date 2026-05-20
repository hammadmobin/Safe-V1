# ============================================================
# config.py — SAFE Framework: All hyperparameters and paths
# Edit this file to change settings without touching other code
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
SEED = 42

# ── Device ───────────────────────────────────────────────────
# Set to 'cpu' to force CPU (auto-detected at runtime otherwise)
DEVICE = None   # None = auto-detect

# ── Token / Model Settings ───────────────────────────────────
MAX_LEN           = 256    # max tokens for all transformer models
DEBERTA_NAME      = 'microsoft/deberta-v3-base'
GLM_NAME          = 'EleutherAI/gpt-neo-1.3B'   # GLTR + DetectGPT reference model
OPENAI_DET_MODEL  = 'roberta-base-openai-detector'

# ── Sampling ──────────────────────────────────────────────────
SAMPLE_SIZE = 5000   # reviews per class per experiment 

# ── Batch Sizes (tuned for H100 80GB) ────────────────────────
GLTR_BATCH     = 64    # inference batch for GLTR / DetectGPT
FINETUNE_BATCH = 32    # training batch for all fine-tuning
EVAL_BATCH     = 128   # evaluation batch

# ── Training ──────────────────────────────────────────────────
EPOCHS      = 3
BACKBONE_LR = 2e-5   # LR for pre-trained backbone weights
HEAD_LR     = 1e-4   # LR for randomly initialized classifier head (kept low to avoid NaN)
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.15  # fraction of total steps used for linear warmup
GRAD_CLIP    = 1.0

# ── GLTR Settings ────────────────────────────────────────────
TOPK = 1000   # number of top tokens considered for rank computation

# ── DetectGPT Settings ───────────────────────────────────────
DETECTGPT_PERTURB = 5     # perturbations per review
SPAN_DROP_PROB    = 0.20  # fraction of words/sentences dropped per perturbation

# ── Data Paths ────────────────────────────────────────────────
# Update these paths to match where your CSV files are located.
# For Google Colab: '/content/GPT_Generated_Books_review.csv'
# For local: 'data/GPT_Generated_Books_review.csv'
DATA_DIR = '/content'

CATEGORY_FILES = {
    'Books'                      : os.path.join(DATA_DIR, 'GPT_Generated_Books_review.csv'),
    'Electronics'                : os.path.join(DATA_DIR, 'GPT_Generated_Electronics.csv'),
    'Grocery and Gourmet Food'   : os.path.join(DATA_DIR, 'GPT_Generated_Grocery_and_Gourmet_Food.csv'),
    'Home and Kitchen'           : os.path.join(DATA_DIR, 'GPT_Generated_Home_and_Kitchen.csv'),
    'Tools and Home Improvement' : os.path.join(DATA_DIR, 'GPT_Generated_Tools_and_Home_Improvement.csv'),
}

PROMPT_TYPES = {
    'Human'                        : 'Human',
    'Zero-Shot Prompting'          : 'Zero-Shot',
    'Few-Shot Prompting'           : 'Few-Shot',
    'Review Replication Prompting' : 'Review Replication',
    'Facet-Aware Prompting'        : 'Facet-Aware',
}

AI_STRATEGIES = ['Zero-Shot', 'Few-Shot', 'Facet-Aware', 'Review Replication']

# ── Output Paths ──────────────────────────────────────────────
RESULTS_DIR      = 'results'
RESULTS_LIVE_CSV = os.path.join(RESULTS_DIR, 'SAFE_results_live.csv')
RESULTS_FULL_CSV = os.path.join(RESULTS_DIR, 'SAFE_results_complete.csv')
RESULTS_AVG_CSV  = os.path.join(RESULTS_DIR, 'SAFE_results_by_method.csv')
PLOT_F1          = os.path.join(RESULTS_DIR, 'SAFE_f1_overall.png')
PLOT_STRATEGY    = os.path.join(RESULTS_DIR, 'SAFE_heatmap_strategy.png')
PLOT_CATEGORY    = os.path.join(RESULTS_DIR, 'SAFE_heatmap_category.png')

# ── Method Run Flags ──────────────────────────────────────────
# Set any to False to skip that method
RUN_GLTR          = True
RUN_DETECTGPT     = True
RUN_NB            = True
RUN_RF            = True
RUN_OPENAI        = True
RUN_ROBERTA       = True
RUN_DEBERTA       = True
RUN_SAFE_TEXTONLY = True
RUN_SAFE_STYLO    = True
