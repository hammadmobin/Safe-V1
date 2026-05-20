# ============================================================
# results/evaluate.py
# Results tables, visualizations, and thesis interpretation
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import (
    RESULTS_FULL_CSV, RESULTS_AVG_CSV,
    PLOT_F1, PLOT_STRATEGY, PLOT_CATEGORY, RESULTS_DIR
)

METHOD_ORDER = [
    # Baselines
    'GLTR (gpt-neo-1.3B)',
    'DetectGPT (span-deletion)',
    'Naive Bayes + TF-IDF',
    'Random Forest + Stylometric',
    'OpenAI Detector (head FT)',
    # Proposed
    'RoBERTa-base (fine-tuned)',
    'DeBERTa-v3-base (fine-tuned)',
    'SAFE Text-Only',
    'SAFE+Stylometric',
]

BASELINE_METHODS = [
    'GLTR (gpt-neo-1.3B)',
    'DetectGPT (span-deletion)',
    'Naive Bayes + TF-IDF',
    'Random Forest + Stylometric',
    'OpenAI Detector (head FT)',
]


def save_results(all_results):
    """Save full results and method-average CSV files."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    present    = [m for m in METHOD_ORDER if m in results_df['Method'].unique()]
    avg_method = (
        results_df.groupby('Method')[['Accuracy', 'F1']]
        .mean().round(4).reindex(present)
    )
    results_df.to_csv(RESULTS_FULL_CSV, index=False)
    avg_method.to_csv(RESULTS_AVG_CSV)
    print(f'Saved: {RESULTS_FULL_CSV}')
    print(f'Saved: {RESULTS_AVG_CSV}')
    return results_df, avg_method


def print_tables(results_df):
    """Print summary tables to stdout."""
    present = [m for m in METHOD_ORDER if m in results_df['Method'].unique()]

    print('\n=== Average by Method ===')
    avg = (results_df.groupby('Method')[['Accuracy','F1']]
           .mean().round(4).reindex(present))
    print(avg.to_string())

    print('\n=== Average by Strategy ===')
    print(results_df.groupby('Comparison')[['Accuracy','F1']].mean().round(4).to_string())

    print('\n=== Average by Category ===')
    print(results_df.groupby('Category')[['Accuracy','F1']].mean().round(4).to_string())


def plot_results(results_df):
    """Generate and save all three visualizations."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    present = [m for m in METHOD_ORDER if m in results_df['Method'].unique()]
    colors  = ['#7B9EC7' if m in BASELINE_METHODS else '#E07B54' for m in present]

    # Plot 1: Overall F1 bar chart
    fig, ax = plt.subplots(figsize=(13, 5))
    avg_f1  = results_df.groupby('Method')['F1'].mean().reindex(present)
    bars    = ax.barh(present, avg_f1.values, color=colors, edgecolor='white', height=0.6)
    ax.axvline(0.5, color='red',    linestyle='--', linewidth=1.2)
    ax.axvline(0.6, color='orange', linestyle=':',  linewidth=1.2)
    for bar, val in zip(bars, avg_f1.values):
        ax.text(val+0.005, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.legend(handles=[
        mpatches.Patch(color='#7B9EC7', label='Baseline'),
        mpatches.Patch(color='#E07B54', label='Proposed'),
        mpatches.Patch(color='red',     label='Random (0.5)'),
        mpatches.Patch(color='orange',  label='Target (~0.6)'),
    ], loc='lower right', fontsize=9)
    ax.set_xlabel('Average F1'); ax.set_xlim(0.3, 1.05)
    ax.set_title('SAFE — Average F1 by Method')
    plt.tight_layout()
    plt.savefig(PLOT_F1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {PLOT_F1}')

    # Plot 2: Method × Strategy heatmap
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot   = results_df.groupby(['Method','Comparison'])['F1'].mean().unstack().reindex(present)
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.45, vmax=1.0, linewidths=0.5, ax=ax)
    ax.set_title('F1: Method × Prompting Strategy')
    ax.set_xlabel(''); ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(PLOT_STRATEGY, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {PLOT_STRATEGY}')

    # Plot 3: Method × Category heatmap
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot2  = results_df.groupby(['Method','Category'])['F1'].mean().unstack().reindex(present)
    sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.45, vmax=1.0, linewidths=0.5, ax=ax)
    ax.set_title('F1: Method × Product Category')
    ax.set_xlabel(''); ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(PLOT_CATEGORY, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {PLOT_CATEGORY}')


def print_analysis(results_df):
    """Print complete thesis interpretation to stdout."""
    sep = '=' * 70
    print(sep)
    print('  SAFE FRAMEWORK — COMPLETE RESULTS ANALYSIS')
    print(sep)

    avg   = results_df.groupby('Method')[['Accuracy','F1']].mean().round(4)
    strat = results_df.groupby('Comparison')[['Accuracy','F1']].mean().round(4)
    cats  = results_df.groupby('Category')['F1'].mean().round(4)

    def get(m, col='F1'):
        return avg.loc[m, col] if m in avg.index else None

    print('\n─── 1. Statistical Baselines ───────────────────────────────')
    for m in ['GLTR (gpt-neo-1.3B)', 'DetectGPT (span-deletion)']:
        v = get(m)
        if v: print(f'  {m:40s} F1={v:.4f}')
    print('  GLTR uses 8 token rank features from gpt-neo-1.3B.')
    print('  GPT-4.1-mini prefers high-probability tokens, creating a')
    print('  measurable rank signal vs human writing.')
    print('  DetectGPT measures log-prob curvature via span-deletion.')

    print('\n─── 2. Classical ML Baselines ──────────────────────────────')
    for m in ['Naive Bayes + TF-IDF', 'Random Forest + Stylometric']:
        v = get(m)
        if v: print(f'  {m:40s} F1={v:.4f}')
    print('  NB exploits vocabulary differences (AI: generic promotional')
    print('  phrases; human: informal, experience-driven words).')
    print('  RF + 14 stylometric features replicates Papers 1, 2, 3.')

    print('\n─── 3. OpenAI Detector ─────────────────────────────────────')
    v = get('OpenAI Detector (head FT)')
    if v:
        print(f'  OpenAI Detector (head FT)              F1={v:.4f}')
        print('  Zero-shot <50% — trained on GPT-2, not GPT-4.1-mini.')
        print('  Head fine-tuning adapts to our task with minimal compute.')

    print('\n─── 4. Proposed Methods ────────────────────────────────────')
    for m in ['RoBERTa-base (fine-tuned)', 'DeBERTa-v3-base (fine-tuned)',
              'SAFE Text-Only', 'SAFE+Stylometric']:
        v = get(m)
        if v: print(f'  {m:40s} F1={v:.4f}')
    rob    = get('RoBERTa-base (fine-tuned)')
    deb    = get('DeBERTa-v3-base (fine-tuned)')
    safe_t = get('SAFE Text-Only')
    safe_s = get('SAFE+Stylometric')
    if rob and deb:
        print(f'  DeBERTa {deb-rob:+.4f} vs RoBERTa.')
    if safe_t and deb:
        print(f'  SAFE Text-Only {safe_t-deb:+.4f} vs per-strategy DeBERTa.')
    if safe_s and safe_t:
        print(f'  SAFE+Stylometric {safe_s-safe_t:+.4f} vs SAFE Text-Only.')
        if safe_s <= safe_t:
            print('  Text-derived features did not improve further.')
            print('  Real metadata (verified_purchase, helpful_vote) is future work.')

    print('\n─── 5. Prompting Strategy Difficulty ───────────────────────')
    hardest = strat['F1'].idxmin()
    easiest = strat['F1'].idxmax()
    print(f'  Easiest: {easiest}  (F1={strat.loc[easiest,"F1"]:.4f})')
    print(f'  Hardest: {hardest} (F1={strat.loc[hardest,"F1"]:.4f})')
    print('  Review Replication hardness motivates full SAFE+metadata.')

    print('\n─── 6. Category Analysis ───────────────────────────────────')
    print(f'  Easiest: {cats.idxmax()}  (F1={cats.max():.4f})')
    print(f'  Hardest: {cats.idxmin()} (F1={cats.min():.4f})')
    print('  Books: sequential generation → more uniform → easier to detect.')

    print('\n─── 7. Final Rankings ───────────────────────────────────────')
    print(avg.sort_values('F1', ascending=False).to_string())
    print(sep)
