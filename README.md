# SAFE: Suspicious vs. Authentic Feedback Evaluation

Detecting LLM-generated product reviews on e-commerce platforms using text features.

## Project Structure

```
SAFE_Detection/
├── config.py                   ← All hyperparameters and file paths (edit this)
├── main.py                     ← Entry point — run everything from here
├── requirements.txt
│
├── baselines/
│   ├── gltr.py                 ← Baseline 1: Extended GLTR (8 features, gpt-neo-1.3B)
│   ├── detectgpt.py            ← Baseline 2: DetectGPT (span-deletion perturbation)
│   ├── classical.py            ← Baseline 3: Naive Bayes + TF-IDF
│   │                              Baseline 4: Random Forest + Stylometric
│   └── openai_detector.py      ← Baseline 5: OpenAI Detector (head fine-tuned)
│
├── models/
│   ├── finetune.py             ← Proposed 1: RoBERTa-base fine-tuned
│   │                              Proposed 2: DeBERTa-v3-base fine-tuned
│   ├── safe_textonly.py        ← Proposed 3: SAFE Text-Only (DeBERTa on all strategies)
│   └── safe_stylometric.py     ← Proposed 4: SAFE+Stylometric (DeBERTa + MLP fusion)
│
├── utils/
│   ├── data_utils.py           ← Data loading, stylometric features, Dataset classes
│   └── train_utils.py          ← Training loop, optimizer, eval loop
│
└── results/
    └── evaluate.py             ← Tables, visualizations, thesis interpretation
```

## Methods

### Baselines (5)
| Method | Type |
|--------|------|
| GLTR (gpt-neo-1.3B) | Statistical | 
| DetectGPT (span-deletion) | Statistical | 
| Naive Bayes + TF-IDF | Classical ML |
| Random Forest + Stylometric | Classical ML | 
| OpenAI Detector (head FT) | Pre-trained | 

### Proposed Methods (2)
| Method | Type |
|--------|------|
| RoBERTa-base fine-tuned | Supervised | 
| DeBERTa-v3-base fine-tuned | Supervised | 


## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Run all experiments
```bash
python main.py
```

### Run specific categories only
```bash
python main.py --categories Books Electronics
```

### Skip slow methods during testing
```bash
python main.py --skip detectgpt safe_stylo
```

### Quick test (50 samples, Books only, Zero-Shot only)
```bash
python main.py --quick
```

## Configuration

Edit `config.py` to change:
- `DATA_DIR` — path to your CSV files
- `SAMPLE_SIZE` — reviews per class (default 300)
- `EPOCHS` — fine-tuning epochs (default 3)
- `RUN_*` flags — enable/disable specific methods

## Data Format

CSV files must have two columns:
- `prompt_type` — one of: `Human`, `Zero-Shot Prompting`, `Few-Shot Prompting`, `Review Replication Prompting`, `Facet-Aware Prompting`
- `generated_review` — the review text

## Key Engineering Notes

**DeBERTa NaN loss fix:** `_initialize_head()` in `utils/data_utils.py` reinitializes the randomly initialized classifier head with `normal(0, 0.02)` instead of PyTorch's default `kaiming_uniform_`. This prevents NaN loss on the first backward pass.

**DataLoader:** All DataLoaders use `num_workers=0` to avoid a Python 3.12 + PyTorch multiprocessing crash on Google Colab.

**Dual LR:** The AdamW optimizer uses two learning rate groups — `2e-5` for the pre-trained backbone and `1e-4` for the randomly initialized head.

## Results Output

Results are saved to the `results/` folder:
- `SAFE_results_live.csv` — updates after every single experiment (crash protection)
- `SAFE_results_complete.csv` — full results table
- `SAFE_results_by_method.csv` — average per method
- `SAFE_f1_overall.png` — F1 bar chart
- `SAFE_heatmap_strategy.png` — F1 heatmap by prompting strategy
- `SAFE_heatmap_category.png` — F1 heatmap by product category

