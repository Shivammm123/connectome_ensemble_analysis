"""
ShREC (Synaptic Hit Rate Excluding Common) Analysis

Implements functional connectivity measure from Ache et al. 2025.
ShREC = (synapses / target_total_input) × 100
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def compute_shrec_scores(connections, wing_mns, shrec_threshold=0.4):
    """
    Compute ShREC scores for all IN→MN connections.
    
    ShREC = (synapses from source / total synapses to target) × 100
    
    Parameters:
    - shrec_threshold: minimum percentage (e.g., 0.4 for 0.4%)
    """
    
    print("\n" + "="*70)
    print(f"COMPUTING ShREC SCORES (threshold ≥{shrec_threshold}%)")
    print("="*70 + "\n")
    
    # Filter for wing MN connections
    wing_conn = connections[connections['target'].isin(wing_mns)].copy()
    
    print(f"Total connections to wing MNs: {len(wing_conn):,}")
    
    # Compute total input to each MN
    print("\nComputing total input to each MN...")
    mn_total_input = connections[
        connections['target'].isin(wing_mns)
    ].groupby('target')['weight'].sum().to_dict()
    
    print(f"MNs with input data: {len(mn_total_input)}")
    
    # Compute ShREC for each connection
    print("\nComputing ShREC scores...")
    
    wing_conn['mn_total_input'] = wing_conn['target'].map(mn_total_input)
    wing_conn['shrec_score'] = (wing_conn['weight'] / wing_conn['mn_total_input']) * 100
    
    print(f"ShREC scores computed: {len(wing_conn):,}")
    print(f"Mean ShREC: {wing_conn['shrec_score'].mean():.3f}%")
    print(f"Median ShREC: {wing_conn['shrec_score'].median():.3f}%")
    print(f"Max ShREC: {wing_conn['shrec_score'].max():.3f}%")
    
    # Filter by ShREC threshold
    significant_conn = wing_conn[wing_conn['shrec_score'] >= shrec_threshold].copy()
    
    print(f"\nConnections ≥{shrec_threshold}% ShREC: {len(significant_conn):,}")
    print(f"Total synapses: {significant_conn['weight'].sum():,.0f}")
    
    # Identify premotor INs
    vnc_ins = connections[
        connections['super_class'] == 'ventral_nerve_cord_intrinsic'
    ]['source'].unique()
    
    premotor_ins = significant_conn[
        significant_conn['source'].isin(vnc_ins)
    ]['source'].unique()
    
    print(f"\nPremotor INs (ShREC ≥{shrec_threshold}%): {len(premotor_ins):,}")
    
    return significant_conn, premotor_ins


def analyze_shrec_distribution(connections, wing_mns, output_dir):
    """
    Analyze ShREC score distribution.
    """
    
    print("\n" + "="*70)
    print("ANALYZING ShREC DISTRIBUTION")
    print("="*70 + "\n")
    
    # Compute all ShREC scores
    wing_conn = connections[connections['target'].isin(wing_mns)].copy()
    
    mn_total_input = connections[
        connections['target'].isin(wing_mns)
    ].groupby('target')['weight'].sum().to_dict()
    
    wing_conn['mn_total_input'] = wing_conn['target'].map(mn_total_input)
    wing_conn['shrec_score'] = (wing_conn['weight'] / wing_conn['mn_total_input']) * 100
    
    # Create distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # Panel 1: ShREC distribution (log scale)
    ax = axes[0, 0]
    ax.hist(wing_conn['shrec_score'], bins=100, 
           color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0.4, color='red', linestyle='--', linewidth=2, 
              label='Ache et al. threshold (0.4%)')
    ax.set_xlabel('ShREC Score (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('A. ShREC Score Distribution', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 2: Cumulative distribution
    ax = axes[0, 1]
    sorted_shrec = np.sort(wing_conn['shrec_score'])
    cumulative = np.arange(1, len(sorted_shrec) + 1) / len(sorted_shrec) * 100
    
    ax.plot(sorted_shrec, cumulative, linewidth=2, color='steelblue')
    ax.axvline(0.4, color='red', linestyle='--', linewidth=2,
              label='0.4% threshold')
    ax.set_xlabel('ShREC Score (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Percentage', fontsize=12, fontweight='bold')
    ax.set_title('B. Cumulative Distribution', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 3: Number of premotor INs by threshold
    ax = axes[1, 0]
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_ins = []
    
    vnc_ins = connections[
        connections['super_class'] == 'ventral_nerve_cord_intrinsic'
    ]['source'].unique()
    
    for thresh in thresholds:
        significant = wing_conn[wing_conn['shrec_score'] >= thresh]
        premotor = significant[significant['source'].isin(vnc_ins)]['source'].nunique()
        n_ins.append(premotor)
    
    bars = ax.bar(range(len(thresholds)), n_ins,
                  color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'{t}%' for t in thresholds], fontsize=10)
    ax.set_xlabel('ShREC Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Premotor INs', fontsize=12, fontweight='bold')
    ax.set_title('C. Premotor INs by ShREC Threshold', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight 0.4%
    ax.patches[3].set_facecolor('#FF6B6B')
    
    # Panel 4: Synapse count vs ShREC score
    ax = axes[1, 1]
    
    # Sample for visibility
    sample = wing_conn.sample(min(5000, len(wing_conn)))
    
    ax.scatter(sample['weight'], sample['shrec_score'],
              alpha=0.3, s=10, color='steelblue')
    ax.axhline(0.4, color='red', linestyle='--', linewidth=2,
              label='ShREC = 0.4%')
    ax.set_xlabel('Synapse Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('ShREC Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('D. Synapse Count vs ShREC Score', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.suptitle('ShREC (Functional Connectivity) Analysis',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'shrec_analysis.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Saved: shrec_analysis.png")


def plot_named_shrec_pairs(significant_conn, loader, output_dir, top_n=25):
    """
    Plot significant ShREC pairs with cell names instead of IDs.
    Creates:
      1. Horizontal bar chart of top N pairs by ShREC score
      2. Heatmap of top INs × top MNs
    """

    df = significant_conn.copy()
    df['source_name'] = df['source'].apply(lambda x: loader.get_neuron_name(x))
    df['target_name'] = df['target'].apply(lambda x: loader.get_neuron_name(x))

    # ── Panel layout ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(22, max(8, top_n * 0.35 + 2)), dpi=300)

    # ── Panel 1: Top N pairs by ShREC score (bar chart) ──────────────────────
    ax = axes[0]
    top_pairs = df.nlargest(top_n, 'shrec_score').copy()
    top_pairs['pair_label'] = (
        top_pairs['source_name'].str[:30] + '  →  ' + top_pairs['target_name'].str[:25]
    )

    y_pos = np.arange(len(top_pairs))
    bars = ax.barh(y_pos, top_pairs['shrec_score'],
                   color='steelblue', edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_pairs['pair_label'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('ShREC Score (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Significant ShREC Pairs\n(IN → Motor Neuron)',
                 fontsize=12, fontweight='bold')
    ax.axvline(0.4, color='red', linestyle='--', linewidth=1.5,
               label='0.4% threshold (Ache et al.)')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, top_pairs['shrec_score']):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', fontsize=7, fontweight='bold')

    # ── Panel 2: Heatmap of top INs × top MNs ────────────────────────────────
    ax = axes[1]

    # Pick top INs and top MNs by total ShREC across all their connections
    top_in_names = (df.groupby('source_name')['shrec_score'].max()
                    .nlargest(20).index.tolist())
    top_mn_names = (df.groupby('target_name')['shrec_score'].max()
                    .nlargest(20).index.tolist())

    heat_df = (df[df['source_name'].isin(top_in_names) & df['target_name'].isin(top_mn_names)]
               .groupby(['source_name', 'target_name'])['shrec_score'].max()
               .unstack(fill_value=0))

    # Reindex to keep consistent ordering
    heat_df = heat_df.reindex(index=top_in_names, columns=top_mn_names, fill_value=0)

    sns.heatmap(heat_df, ax=ax, cmap='YlOrRd', linewidths=0.3,
                cbar_kws={'label': 'ShREC Score (%)', 'shrink': 0.8},
                xticklabels=True, yticklabels=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xlabel('Motor Neuron', fontsize=11, fontweight='bold')
    ax.set_ylabel('Interneuron', fontsize=11, fontweight='bold')
    ax.set_title('ShREC Score Heatmap\n(Top 20 INs × Top 20 MNs)',
                 fontsize=12, fontweight='bold')

    fig.suptitle('Significant ShREC Connections — Cell Names',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = output_dir / 'shrec_named_pairs.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {out_path}")


def main():
    print("\n" + "="*70)
    print("ShREC FUNCTIONAL CONNECTIVITY ANALYSIS")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    loader = ConnectomeDataLoader('config.yaml')
    connections, neurons, name_mapping = loader.load_all_data(verbose=False)
    
    # Add super_class to connections for filtering
    neuron_class_map = dict(zip(neurons['Root ID'], neurons['Super Class']))
    connections['super_class'] = connections['source'].map(neuron_class_map)
    
    # Load motor pools
    motor_pools = pd.read_csv('results/motor_neurons/motor_pools.csv')
    wing_mns = []
    for _, pool in motor_pools.iterrows():
        wing_mns.extend(eval(pool['motor_neuron_ids']))
    wing_mns = list(set(wing_mns))
    
    print(f"✓ Loaded {len(connections):,} connections")
    print(f"✓ Wing MNs: {len(wing_mns)}\n")
    
    # Compute ShREC scores
    significant_conn, premotor_ins = compute_shrec_scores(
        connections, wing_mns, shrec_threshold=0.4
    )
    
    # Save results
    output_dir = Path('results/shrec_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    significant_conn.to_csv(output_dir / 'significant_connections_shrec0.4.csv', index=False)
    pd.DataFrame({'neuron_id': premotor_ins}).to_csv(
        output_dir / 'premotor_neurons_shrec0.4.csv', index=False
    )
    
    print(f"\n✓ Saved: significant_connections_shrec0.4.csv")
    print(f"✓ Saved: premotor_neurons_shrec0.4.csv")

    # Named pairs plot
    plot_named_shrec_pairs(significant_conn, loader, output_dir, top_n=25)

    # Analyze distribution
    analyze_shrec_distribution(connections, wing_mns, output_dir)
    
    # Test multiple thresholds
    print("\n" + "="*70)
    print("TESTING MULTIPLE ShREC THRESHOLDS")
    print("="*70 + "\n")
    
    thresholds = [0.1, 0.2, 0.4, 0.6, 1.0]
    
    for thresh in thresholds:
        sig_conn, prem_ins = compute_shrec_scores(connections, wing_mns, thresh)
        print(f"\nShREC ≥{thresh}%:")
        print(f"  • Premotor INs: {len(prem_ins):,}")
        print(f"  • Connections: {len(sig_conn):,}")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()