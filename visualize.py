"""
Visualization script to generate graphs from keyword data.
Creates charts for volume distribution, CPC analysis, intent classification, etc.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import numpy as np

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_keyword_data(json_path: str) -> Dict:
    """Load keyword data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_all_keywords(data: Dict) -> pd.DataFrame:
    """Extract all keywords from ad groups into a DataFrame."""
    keywords = []
    
    for ad_group in data.get('ad_groups', []):
        for kw in ad_group.get('keywords', []):
            kw_data = kw.copy()
            kw_data['ad_group'] = ad_group.get('ad_group_name', 'Unknown')
            keywords.append(kw_data)
    
    return pd.DataFrame(keywords)


def plot_volume_distribution(df: pd.DataFrame, output_path: str):
    """Plot keyword search volume distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Volume histogram
    axes[0, 0].hist(df['volume'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='#3498db')
    axes[0, 0].set_title('Keyword Search Volume Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Monthly Search Volume')
    axes[0, 0].set_ylabel('Number of Keywords')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Volume by intent
    intent_volume = df.groupby('intent')['volume'].sum().sort_values(ascending=False)
    axes[0, 1].bar(intent_volume.index, intent_volume.values, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
    axes[0, 1].set_title('Total Volume by Intent Category', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Intent Type')
    axes[0, 1].set_ylabel('Total Monthly Volume')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Top keywords by volume
    top_keywords = df.nlargest(15, 'volume')[['keyword', 'volume']]
    axes[1, 0].barh(range(len(top_keywords)), top_keywords['volume'].values, color='#2ecc71')
    axes[1, 0].set_yticks(range(len(top_keywords)))
    axes[1, 0].set_yticklabels(top_keywords['keyword'].values, fontsize=8)
    axes[1, 0].set_title('Top 15 Keywords by Search Volume', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Monthly Search Volume')
    axes[1, 0].invert_yaxis()
    
    # Volume by ad group
    ad_group_volume = df.groupby('ad_group')['volume'].sum().sort_values(ascending=False)
    axes[1, 1].barh(range(len(ad_group_volume)), ad_group_volume.values, color='#9b59b6')
    axes[1, 1].set_yticks(range(len(ad_group_volume)))
    axes[1, 1].set_yticklabels(ad_group_volume.index.values, fontsize=9)
    axes[1, 1].set_title('Total Volume by Ad Group', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Total Monthly Volume')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved volume distribution chart: {output_path}")


def plot_cpc_analysis(df: pd.DataFrame, output_path: str):
    """Plot CPC (Cost Per Click) analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # CPC vs Competition scatter
    axes[0, 0].scatter(df['competition'], df['cpc_high'], alpha=0.6, s=100, c=df['volume'], 
                       cmap='viridis', edgecolors='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Competition Level', fontsize=12)
    axes[0, 0].set_ylabel('CPC High ($)', fontsize=12)
    axes[0, 0].set_title('CPC vs Competition (Color = Volume)', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
    cbar.set_label('Search Volume', fontsize=10)
    
    # CPC distribution
    axes[0, 1].hist(df['cpc_high'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='#e74c3c')
    axes[0, 1].set_title('CPC High Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('CPC High ($)')
    axes[0, 1].set_ylabel('Number of Keywords')
    axes[0, 1].grid(True, alpha=0.3)
    
    # CPC by intent
    cpc_by_intent = df.groupby('intent')['cpc_high'].mean().sort_values(ascending=False)
    axes[1, 0].bar(cpc_by_intent.index, cpc_by_intent.values, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
    axes[1, 0].set_title('Average CPC by Intent', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Intent Type')
    axes[1, 0].set_ylabel('Average CPC ($)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # CPC range (low vs high)
    axes[1, 1].scatter(df['cpc_low'], df['cpc_high'], alpha=0.6, s=80, c='#3498db', edgecolors='black', linewidth=0.5)
    axes[1, 1].plot([df['cpc_low'].min(), df['cpc_high'].max()], 
                    [df['cpc_low'].min(), df['cpc_high'].max()], 
                    'r--', linewidth=2, label='Equal line')
    axes[1, 1].set_xlabel('CPC Low ($)', fontsize=12)
    axes[1, 1].set_ylabel('CPC High ($)', fontsize=12)
    axes[1, 1].set_title('CPC Low vs High', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved CPC analysis chart: {output_path}")


def plot_intent_analysis(df: pd.DataFrame, output_path: str):
    """Plot intent classification analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Intent distribution pie chart
    intent_counts = df['intent'].value_counts()
    colors = {'transactional': '#e74c3c', 'commercial': '#3498db', 
              'informational': '#2ecc71', 'navigational': '#f39c12'}
    pie_colors = [colors.get(intent, '#95a5a6') for intent in intent_counts.index]
    
    axes[0, 0].pie(intent_counts.values, labels=intent_counts.index, autopct='%1.1f%%', 
                   startangle=90, colors=pie_colors, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[0, 0].set_title('Keyword Intent Distribution', fontsize=14, fontweight='bold')
    
    # Intent by volume
    intent_volume = df.groupby('intent')['volume'].sum().sort_values(ascending=False)
    axes[0, 1].bar(intent_volume.index, intent_volume.values, 
                   color=[colors.get(intent, '#95a5a6') for intent in intent_volume.index])
    axes[0, 1].set_title('Total Volume by Intent', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Intent Type')
    axes[0, 1].set_ylabel('Total Monthly Volume')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Match type distribution
    match_type_counts = df['match_type'].value_counts()
    axes[1, 0].bar(match_type_counts.index, match_type_counts.values, 
                   color=['#e74c3c', '#3498db', '#2ecc71'])
    axes[1, 0].set_title('Match Type Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Match Type')
    axes[1, 0].set_ylabel('Number of Keywords')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Intent vs Match Type heatmap
    intent_match = pd.crosstab(df['intent'], df['match_type'])
    sns.heatmap(intent_match, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': 'Count'})
    axes[1, 1].set_title('Intent vs Match Type', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Match Type')
    axes[1, 1].set_ylabel('Intent Type')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved intent analysis chart: {output_path}")


def plot_score_analysis(df: pd.DataFrame, output_path: str):
    """Plot keyword scoring analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Score distribution
    axes[0, 0].hist(df['score'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='#9b59b6')
    axes[0, 0].set_title('Keyword Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Score (0-1)')
    axes[0, 0].set_ylabel('Number of Keywords')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(df['score'].mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {df["score"].mean():.3f}')
    axes[0, 0].legend()
    
    # Score vs Volume
    axes[0, 1].scatter(df['volume'], df['score'], alpha=0.6, s=80, c=df['cpc_high'], 
                       cmap='coolwarm', edgecolors='black', linewidth=0.5)
    axes[0, 1].set_xlabel('Search Volume', fontsize=12)
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].set_title('Score vs Volume (Color = CPC)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar.set_label('CPC High ($)', fontsize=10)
    
    # Top scored keywords
    top_scored = df.nlargest(15, 'score')[['keyword', 'score']]
    axes[1, 0].barh(range(len(top_scored)), top_scored['score'].values, color='#2ecc71')
    axes[1, 0].set_yticks(range(len(top_scored)))
    axes[1, 0].set_yticklabels(top_scored['keyword'].values, fontsize=8)
    axes[1, 0].set_title('Top 15 Keywords by Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].invert_yaxis()
    
    # Score by ad group
    ad_group_score = df.groupby('ad_group')['score'].mean().sort_values(ascending=False)
    axes[1, 1].barh(range(len(ad_group_score)), ad_group_score.values, color='#3498db')
    axes[1, 1].set_yticks(range(len(ad_group_score)))
    axes[1, 1].set_yticklabels(ad_group_score.index.values, fontsize=9)
    axes[1, 1].set_title('Average Score by Ad Group', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Average Score')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved score analysis chart: {output_path}")


def plot_ad_group_performance(data: Dict, output_path: str):
    """Plot ad group performance metrics."""
    ad_groups_data = []
    
    for ad_group in data.get('ad_groups', []):
        keywords = ad_group.get('keywords', [])
        if keywords:
            df = pd.DataFrame(keywords)
            ad_groups_data.append({
                'ad_group': ad_group.get('ad_group_name', 'Unknown'),
                'keyword_count': len(keywords),
                'total_volume': ad_group.get('total_volume', 0),
                'avg_cpc': df['cpc_high'].mean(),
                'avg_score': df['score'].mean(),
                'avg_competition': df['competition'].mean()
            })
    
    if not ad_groups_data:
        print("⚠️ No ad group data to plot")
        return
    
    ag_df = pd.DataFrame(ad_groups_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Keyword count by ad group
    axes[0, 0].barh(range(len(ag_df)), ag_df['keyword_count'].values, color='#3498db')
    axes[0, 0].set_yticks(range(len(ag_df)))
    axes[0, 0].set_yticklabels(ag_df['ad_group'].values, fontsize=9)
    axes[0, 0].set_title('Keywords per Ad Group', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Keywords')
    axes[0, 0].invert_yaxis()
    
    # Total volume by ad group
    axes[0, 1].barh(range(len(ag_df)), ag_df['total_volume'].values, color='#2ecc71')
    axes[0, 1].set_yticks(range(len(ag_df)))
    axes[0, 1].set_yticklabels(ag_df['ad_group'].values, fontsize=9)
    axes[0, 1].set_title('Total Volume by Ad Group', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Total Monthly Volume')
    axes[0, 1].invert_yaxis()
    
    # Average CPC by ad group
    axes[1, 0].barh(range(len(ag_df)), ag_df['avg_cpc'].values, color='#e74c3c')
    axes[1, 0].set_yticks(range(len(ag_df)))
    axes[1, 0].set_yticklabels(ag_df['ad_group'].values, fontsize=9)
    axes[1, 0].set_title('Average CPC by Ad Group', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Average CPC ($)')
    axes[1, 0].invert_yaxis()
    
    # Volume vs Score scatter by ad group
    for i, row in ag_df.iterrows():
        axes[1, 1].scatter(row['total_volume'], row['avg_score'], s=row['keyword_count']*20, 
                         alpha=0.6, label=row['ad_group'], edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Total Volume', fontsize=12)
    axes[1, 1].set_ylabel('Average Score', fontsize=12)
    axes[1, 1].set_title('Ad Group Performance (Size = Keyword Count)', fontsize=14, fontweight='bold')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved ad group performance chart: {output_path}")


def generate_all_charts(json_path: str, output_dir: str = "charts"):
    """Generate essential visualization charts from keyword data."""
    print(f"📊 Generating essential charts from: {json_path}")
    
    # Load data
    data = load_keyword_data(json_path)
    df = extract_all_keywords(data)
    
    if df.empty:
        print("⚠️ No keyword data found!")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate only the 3 most relevant charts for SEM keyword intelligence
    print(f"\n📈 Creating visualizations...")
    
    # 1. Volume Distribution - MOST CRITICAL
    # Shows: Search volume distribution, top keywords, volume by intent & ad groups
    # Why: Search volume is the #1 metric for keyword selection
    plot_volume_distribution(df, f"{output_dir}/volume_distribution.png")
    
    # 2. CPC Analysis - CRITICAL for Budget Planning
    # Shows: CPC vs Competition, CPC distribution, cost by intent
    # Why: Directly impacts budget allocation and ROI
    plot_cpc_analysis(df, f"{output_dir}/cpc_analysis.png")
    
    # 3. Intent Analysis - CRITICAL for Targeting Strategy
    # Shows: Intent distribution, match type distribution, intent vs match type
    # Why: Essential for campaign structure and ad group organization
    plot_intent_analysis(df, f"{output_dir}/intent_analysis.png")
    
    print(f"\n✅ Essential charts generated in '{output_dir}' directory!")
    print(f"   1. Volume Distribution (Search Volume Analysis)")
    print(f"   2. CPC Analysis (Cost & Budget Planning)")
    print(f"   3. Intent Analysis (Targeting Strategy)")


if __name__ == "__main__":
    import sys
    
    # Default to latest run
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # Find latest run
        output_dirs = sorted(Path("outputs").glob("run-*"), reverse=True)
        if output_dirs:
            json_path = output_dirs[0] / "keyword_data.json"
        else:
            print("❌ No output files found!")
            sys.exit(1)
    
    if not Path(json_path).exists():
        print(f"❌ File not found: {json_path}")
        sys.exit(1)
    
    # Generate charts
    run_id = Path(json_path).parent.name
    generate_all_charts(json_path, f"charts/{run_id}")

