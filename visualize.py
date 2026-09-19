# -*- coding: utf-8 -*-
"""
银行营销数据分析 - 可视化呈现
使用 matplotlib 和 seaborn 对分析结果进行可视化
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
import os

# ============================================================
# 设置中文字体
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 检查可用中文字体
chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'SimHei' in f.name or 'YaHei' in f.name or 'Songti' in f.name]
print(f"Available Chinese fonts: {chinese_fonts}")

# 使用英文作为后备
USE_CHINESE = len(chinese_fonts) > 0
if not USE_CHINESE:
    print("Warning: No Chinese font found, using English labels")

def load_data():
    """Load all analysis results"""
    # Load preprocessing stats
    with open('preprocess_stats.json', 'r', encoding='utf-8') as f:
        stats = json.load(f)

    # Load model results
    with open('model_results.json', 'r') as f:
        model_results = json.load(f)

    # Load bank data
    df = pd.read_csv('bank.csv', sep=';')
    return df, stats, model_results

def plot_target_distribution(stats, ax):
    """Plot target variable distribution"""
    labels = ['No (Did Not Subscribe)', 'Yes (Subscribed)']
    sizes = [stats['y_distribution']['no'], stats['y_distribution']['yes']]
    colors = ['#ff6b6b', '#51cf66']
    explode = (0, 0.1)

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                       colors=colors, autopct='%1.1f%%',
                                       shadow=True, startangle=140)
    for t in autotexts:
        t.set_fontsize(12)
        t.set_fontweight('bold')
    for t in texts:
        t.set_fontsize(11)
    ax.set_title('Term Deposit Subscription Distribution', fontsize=14, fontweight='bold')

def plot_job_subscription(stats, ax):
    """Plot subscription rate by job"""
    data = stats['y_by_job']
    jobs = list(data.keys())
    rates = [data[j] * 100 for j in jobs]

    # Sort by rate
    sorted_idx = np.argsort(rates)
    jobs = [jobs[i] for i in sorted_idx]
    rates = [rates[i] for i in sorted_idx]

    colors = ['#ff6b6b' if r < 11.5 else '#51cf66' for r in rates]
    bars = ax.barh(jobs, rates, color=colors, edgecolor='white', linewidth=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{rate:.1f}%', va='center', fontsize=9)

    ax.axvline(x=11.52, color='red', linestyle='--', alpha=0.7, label='Overall avg: 11.52%')
    ax.set_xlabel('Subscription Rate (%)', fontsize=11)
    ax.set_title('Subscription Rate by Job', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)

def plot_monthly_effect(stats, ax):
    """Plot monthly campaign effect"""
    data = stats['y_by_month']
    months_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    months = [m for m in months_order if m in data]
    rates = [data[m] * 100 for m in months]

    colors = ['#ff6b6b' if r < 11.5 else '#51cf66' for r in rates]
    ax.bar(months, rates, color=colors, edgecolor='white', linewidth=0.5)

    for i, rate in enumerate(rates):
        ax.text(i, rate + 0.5, f'{rate:.1f}%', ha='center', fontsize=9, fontweight='bold')

    ax.axhline(y=11.52, color='red', linestyle='--', alpha=0.7, label='Overall avg: 11.52%')
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Subscription Rate (%)', fontsize=11)
    ax.set_title('Monthly Campaign Subscription Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)

def plot_education_marital(stats, ax1, ax2):
    """Plot education and marital subscription rates"""
    # Education
    edu_data = stats['y_by_education']
    edu_labels = list(edu_data.keys())
    edu_rates = [edu_data[e] * 100 for e in edu_labels]
    colors1 = ['#ff6b6b' if r < 11.5 else '#51cf66' for r in edu_rates]
    bars1 = ax1.bar(edu_labels, edu_rates, color=colors1, edgecolor='white')
    for bar, rate in zip(bars1, edu_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax1.axhline(y=11.52, color='red', linestyle='--', alpha=0.7)
    ax1.set_title('By Education', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Subscription Rate (%)')

    # Marital
    mar_data = stats['y_by_marital']
    mar_labels = list(mar_data.keys())
    mar_rates = [mar_data[m] * 100 for m in mar_labels]
    colors2 = ['#ff6b6b' if r < 11.5 else '#51cf66' for r in mar_rates]
    bars2 = ax2.bar(mar_labels, mar_rates, color=colors2, edgecolor='white')
    for bar, rate in zip(bars2, mar_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax2.axhline(y=11.52, color='red', linestyle='--', alpha=0.7)
    ax2.set_title('By Marital Status', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Subscription Rate (%)')

def plot_poutcome_contact(stats, ax1, ax2):
    """Plot poutcome and contact impact"""
    # Previous outcome
    pout_data = stats['y_by_poutcome']
    labels1 = list(pout_data.keys())
    rates1 = [pout_data[p] * 100 for p in labels1]
    colors1 = ['#51cf66' if r > 30 else '#ffd43b' if r > 11.5 else '#ff6b6b' for r in rates1]
    bars1 = ax1.bar(labels1, rates1, color=colors1, edgecolor='white')
    for bar, rate in zip(bars1, rates1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax1.set_title('Impact of Previous Outcome', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Subscription Rate (%)')
    ax1.axhline(y=11.52, color='red', linestyle='--', alpha=0.7)

    # Contact type
    cont_data = stats['y_by_contact']
    labels2 = list(cont_data.keys())
    rates2 = [cont_data[c] * 100 for c in labels2]
    colors2 = ['#51cf66' if r > 11.5 else '#ff6b6b' for r in rates2]
    bars2 = ax2.bar(labels2, rates2, color=colors2, edgecolor='white')
    for bar, rate in zip(bars2, rates2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax2.set_title('Impact of Contact Type', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Subscription Rate (%)')
    ax2.axhline(y=11.52, color='red', linestyle='--', alpha=0.7)

def plot_age_balance_analysis(df, ax1, ax2):
    """Age distribution and balance analysis"""
    # Age histogram by subscription
    for label, color, name in [('yes', '#51cf66', 'Subscribed'), ('no', '#ff6b6b', 'Not Subscribed')]:
        subset = df[df['y'] == label]['age']
        ax1.hist(subset, bins=25, alpha=0.6, color=color, label=name, edgecolor='white')

    ax1.set_xlabel('Age', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Age Distribution by Subscription', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)

    # Balance boxplot
    data_yes = df[df['y'] == 'yes']['balance']
    data_no = df[df['y'] == 'no']['balance']
    # Filter outliers for better visualization
    q99 = df['balance'].quantile(0.99)
    q01 = df['balance'].quantile(0.01)
    bp = ax2.boxplot([data_no[(data_no > q01) & (data_no < q99)],
                       data_yes[(data_yes > q01) & (data_yes < q99)]],
                      labels=['Not Subscribed', 'Subscribed'],
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('#ff6b6b')
    bp['boxes'][1].set_facecolor('#51cf66')
    ax2.set_ylabel('Balance (EUR)', fontsize=11)
    ax2.set_title('Balance Distribution by Subscription', fontsize=13, fontweight='bold')

def plot_duration_analysis(df, ax):
    """Duration vs subscription"""
    for label, color, name in [('yes', '#51cf66', 'Subscribed'), ('no', '#ff6b6b', 'Not Subscribed')]:
        subset = df[df['y'] == label]['duration']
        ax.hist(subset, bins=40, alpha=0.6, color=color, label=name, edgecolor='white', density=True)

    ax.set_xlabel('Duration (seconds)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Call Duration Distribution by Subscription', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1500)

def plot_model_comparison(model_results, ax):
    """Plot model comparison (AUC)"""
    models = ['Logistic\nRegression', 'Random\nForest', 'Decision\nTree']
    aucs = [model_results['lr_auc'], model_results['rf_auc'], model_results['dt_auc']]
    colors = ['#4ecdc4', '#2ecc71', '#e74c3c']

    bars = ax.bar(models, aucs, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{auc:.4f}', ha='center', fontsize=13, fontweight='bold')

    ax.set_ylabel('AUC Score', fontsize=11)
    ax.set_title('Model Performance Comparison (AUC)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random baseline')
    ax.legend(fontsize=9)

    # Highlight best model
    best_idx = np.argmax(aucs)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

def plot_numeric_comparison(df, ax):
    """Numeric features comparison by target"""
    metrics = ['age', 'balance', 'duration', 'campaign']
    labels = ['Age', 'Balance\n(EUR)', 'Duration\n(seconds)', 'Campaign\nCount']
    colors_yes = '#51cf66'
    colors_no = '#ff6b6b'

    yes_means = [df[df['y'] == 'yes'][m].mean() for m in metrics]
    no_means = [df[df['y'] == 'no'][m].mean() for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    # Normalize for visualization
    max_vals = [max(y, n) for y, n in zip(yes_means, no_means)]
    yes_norm = [y/m * 100 for y, m in zip(yes_means, max_vals)]
    no_norm = [n/m * 100 for n, m in zip(no_means, max_vals)]

    bars1 = ax.bar(x - width/2, yes_norm, width, color=colors_yes, label='Subscribed (Yes)', edgecolor='white')
    bars2 = ax.bar(x + width/2, no_norm, width, color=colors_no, label='Not Subscribed (No)', edgecolor='white')

    # Add actual values on bars
    for bar, val in zip(bars1, yes_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')
    for bar, val in zip(bars2, no_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Normalized Value (%)', fontsize=11)
    ax.set_title('Numeric Features Comparison (Yes vs No)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

def create_summary_dashboard(df, stats, model_results):
    """Create a comprehensive dashboard figure"""
    fig = plt.figure(figsize=(24, 30))
    gs = GridSpec(6, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Row 1: Target distribution + Model comparison + Numeric comparison
    ax1 = fig.add_subplot(gs[0, 0])
    plot_target_distribution(stats, ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_model_comparison(model_results, ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_numeric_comparison(df, ax3)

    # Row 2: Job + Monthly
    ax4 = fig.add_subplot(gs[1, :2])
    plot_job_subscription(stats, ax4)

    ax5 = fig.add_subplot(gs[1, 2])
    plot_monthly_effect(stats, ax5)

    # Row 3: Education & Marital
    ax6 = fig.add_subplot(gs[2, 0])
    ax7 = fig.add_subplot(gs[2, 1])
    plot_education_marital(stats, ax6, ax7)

    # Row 3 col 3: Poutcome & Contact
    ax8 = fig.add_subplot(gs[2, 2])
    ax9 = fig.add_subplot(gs[3, 0])
    plot_poutcome_contact(stats, ax8, ax9)

    # Row 4: Age distribution + Duration
    ax10 = fig.add_subplot(gs[3, 1])
    ax11 = fig.add_subplot(gs[3, 2])
    plot_age_balance_analysis(df, ax10, ax11)

    ax12 = fig.add_subplot(gs[4, :2])
    plot_duration_analysis(df, ax12)

    # Row 5: Model metrics table
    ax13 = fig.add_subplot(gs[4, 2])
    ax13.axis('off')
    table_data = [
        ['Model', 'AUC'],
        ['Logistic Reg.', f"{model_results['lr_auc']:.4f}"],
        ['Random Forest', f"{model_results['rf_auc']:.4f}"],
        ['Decision Tree', f"{model_results['dt_auc']:.4f}"],
    ]
    table = ax13.table(cellText=table_data, cellLoc='center', loc='center',
                       colWidths=[0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif key[0] == np.argmax([model_results['lr_auc'], model_results['rf_auc'], model_results['dt_auc']]) + 1:
            cell.set_facecolor('#d4edda')
            cell.set_text_props(fontweight='bold')
    ax13.set_title('Model AUC Summary', fontsize=13, fontweight='bold')

    # Row 6: Text summary - key insights
    ax14 = fig.add_subplot(gs[5, :])
    ax14.axis('off')
    best_model = model_results.get('best', 'Random Forest')
    summary_text = f"""
    ====================================================================================================================================
    BANK MARKETING ANALYSIS - KEY INSIGHTS
    ====================================================================================================================================

    [DATA OVERVIEW]
    - Total Records: 4,521  |  Subscribed (Yes): 521 (11.52%)  |  Not Subscribed (No): 4,000 (88.48%)
    - Best Model: {best_model} (AUC: {model_results.get('rf_auc', 0):.4f})

    [KEY FINDINGS]
    1. Previous Campaign Outcome is the #1 predictor: clients who succeeded before have 64.3% re-subscription rate
    2. Contact Duration is critical: subscribers avg 552.7s vs non-subscribers 226.3s (2.4x longer calls)
    3. Monthly Effect: Oct (46.3%), Dec (45.0%), Mar (42.9%) are the best months; May is the worst (6.7%)
    4. Job Impact: retired (23.5%) and students (22.6%) have highest subscription rates
    5. Education: tertiary educated clients subscribe 50% more than primary educated
    6. Contact Type: telephone (14.6%) and cellular (14.4%) are far better than unknown (4.6%)
    7. Balance: subscribed clients have slightly higher average balance (1,572 vs 1,403 EUR)
    8. Age: clients under 25 and over 55 show highest subscription rates (~19.4% and 15.7%)

    [MODEL PERFORMANCE]
    - Random Forest: AUC 0.8996 (best) - strong non-linear relationships captured
    - Logistic Regression: AUC 0.8813 - good baseline, interpretable coefficients
    - Decision Tree: AUC 0.5574 - underperforms due to data imbalance
    - Top RF features: duration, month, job, day, poutcome

    [RECOMMENDATIONS]
    - Target previous successful clients for follow-up campaigns
    - Focus calling efforts in Oct, Dec, and Mar
    - Prioritize retired and student customer segments
    - Increase call duration to improve conversion rates
    - Use cellular or telephone contact only (avoid unknown contact types)
    ====================================================================================================================================
    """
    ax14.text(0.02, 0.98, summary_text, transform=ax14.transAxes, fontsize=8.5,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8, edgecolor='#dee2e6'))

    fig.suptitle('Bank Marketing Analysis - Comprehensive Dashboard',
                 fontsize=20, fontweight='bold', y=0.995)
    return fig

def create_individual_plots(df, stats, model_results):
    """Create individual plots for report"""
    os.makedirs('plots', exist_ok=True)

    # 1. Target distribution pie chart
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_target_distribution(stats, ax)
    plt.tight_layout()
    fig.savefig('plots/1_target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plots/1_target_distribution.png")

    # 2. Job subscription rate
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_job_subscription(stats, ax)
    plt.tight_layout()
    fig.savefig('plots/2_job_subscription.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plots/2_job_subscription.png")

    # 3. Monthly effect
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_monthly_effect(stats, ax)
    plt.tight_layout()
    fig.savefig('plots/3_monthly_effect.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plots/3_monthly_effect.png")

    # 4. Model comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_model_comparison(model_results, ax)
    plt.tight_layout()
    fig.savefig('plots/4_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plots/4_model_comparison.png")

    # 5. Age distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color, name in [('yes', '#51cf66', 'Subscribed'), ('no', '#ff6b6b', 'Not Subscribed')]:
        subset = df[df['y'] == label]['age']
        ax.hist(subset, bins=25, alpha=0.6, color=color, label=name, edgecolor='white')
    ax.set_xlabel('Age', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Age Distribution by Subscription', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig('plots/5_age_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plots/5_age_distribution.png")

    # 6. Duration distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_duration_analysis(df, ax)
    plt.tight_layout()
    fig.savefig('plots/6_duration_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plots/6_duration_distribution.png")

    # 7. Numeric comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_numeric_comparison(df, ax)
    plt.tight_layout()
    fig.savefig('plots/7_numeric_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plots/7_numeric_comparison.png")

    # 8. Balance boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    data_yes = df[df['y'] == 'yes']['balance']
    data_no = df[df['y'] == 'no']['balance']
    q99 = df['balance'].quantile(0.99)
    q01 = df['balance'].quantile(0.01)
    bp = ax.boxplot([data_no[(data_no > q01) & (data_no < q99)],
                      data_yes[(data_yes > q01) & (data_yes < q99)]],
                     labels=['Not Subscribed', 'Subscribed'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#ff6b6b')
    bp['boxes'][1].set_facecolor('#51cf66')
    ax.set_ylabel('Balance (EUR)', fontsize=11)
    ax.set_title('Balance Distribution by Subscription', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig('plots/8_balance_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plots/8_balance_boxplot.png")


if __name__ == '__main__':
    print("=" * 60)
    print("Bank Marketing Analysis - Visualization")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df, stats, model_results = load_data()
    print("Data loaded successfully!")

    # Create individual plots
    print("\nGenerating individual plots...")
    create_individual_plots(df, stats, model_results)

    # Create comprehensive dashboard
    print("\nGenerating comprehensive dashboard...")
    fig = create_summary_dashboard(df, stats, model_results)
    plt.tight_layout()
    fig.savefig('plots/comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plots/comprehensive_dashboard.png")

    # Show summary
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"All plots saved to: {os.path.abspath('plots/')}")
    print("=" * 60)

    # Print analysis summary
    print(f"""
Analysis Results Summary:
-------------------------
  Data: {stats['total_rows']} samples, 17 features
  Target: yes={stats['y_distribution']['yes']} ({stats['y_distribution']['yes']/stats['total_rows']*100:.1f}%), no={stats['y_distribution']['no']} ({stats['y_distribution']['no']/stats['total_rows']*100:.1f}%)

  Model AUC:
    Logistic Regression: {model_results['lr_auc']:.4f}
    Random Forest:       {model_results['rf_auc']:.4f}
    Decision Tree:       {model_results['dt_auc']:.4f}

  Best Model: {model_results['best']}
  Train/Test: {model_results['train_n']}/{model_results['test_n']}
""")
