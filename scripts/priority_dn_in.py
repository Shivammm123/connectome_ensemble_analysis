"""
DN to Priority Interneuron Analysis

Identifies which DNs provide input to high-priority hub INs.
Shows which brain commands control the most critical premotor neurons.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def analyze_dn_to_priority_connections(
    priority_ins,
    dn_in_conn,
    dn_class_df,
    cluster_assignments,
    loader,
    top_n_ins=50
):
    """
    Find which DNs connect to priority interneurons.
    """
    
    print("\n" + "="*70)
    print("DN → PRIORITY INTERNEURON ANALYSIS")
    print("="*70 + "\n")
    
    # Get top priority INs
    top_priority = priority_ins.head(top_n_ins)
    priority_in_ids = top_priority['interneuron_id'].tolist()
    
    print(f"Analyzing top {top_n_ins} priority interneurons...")
    print(f"Priority IN IDs: {len(priority_in_ids)}\n")
    
    # Find DN connections to these INs
    dn_to_priority = dn_in_conn[dn_in_conn['target'].isin(priority_in_ids)].copy()
    
    print(f"DN → Priority IN connections found: {len(dn_to_priority):,}")
    print(f"Unique DNs involved: {dn_to_priority['source'].nunique()}")
    print(f"Total synapses: {dn_to_priority['weight'].sum():,.0f}\n")
    
    # Add DN names
    dn_to_priority['dn_name'] = dn_to_priority['source'].apply(
        lambda x: loader.get_neuron_name(x)
    )
    
    # Add IN names and priority info
    in_info_map = dict(zip(top_priority['interneuron_id'], 
                           zip(top_priority['neuron_name'], 
                               top_priority['rank'],
                               top_priority['composite_score'],
                               top_priority['cluster'])))
    
    dn_to_priority['in_name'] = dn_to_priority['target'].apply(
        lambda x: in_info_map.get(x, ('unknown', 999, 0, -1))[0]
    )
    dn_to_priority['in_rank'] = dn_to_priority['target'].apply(
        lambda x: in_info_map.get(x, ('unknown', 999, 0, -1))[1]
    )
    dn_to_priority['in_score'] = dn_to_priority['target'].apply(
        lambda x: in_info_map.get(x, ('unknown', 999, 0, -1))[2]
    )
    dn_to_priority['in_cluster'] = dn_to_priority['target'].apply(
        lambda x: in_info_map.get(x, ('unknown', 999, 0, -1))[3]
    )
    
    # Add DN specialization
    dn_spec_map = dict(zip(dn_class_df['dn_id'], dn_class_df['specialization']))
    dn_to_priority['dn_specialization'] = dn_to_priority['source'].apply(
        lambda x: dn_spec_map.get(x, 'unknown')
    )
    
    return dn_to_priority, top_priority


def summarize_dn_priority_connections(dn_to_priority):
    """
    Create summary statistics.
    """
    
    print("-"*70)
    print("SUMMARY STATISTICS")
    print("-"*70 + "\n")
    
    # By DN
    print("Top 10 DNs by connections to priority INs:")
    print(f"{'DN Name':<40} {'Priority INs':<15} {'Total Syn':<12} {'Specialization'}")
    print("-"*70)
    
    dn_summary = dn_to_priority.groupby(['source', 'dn_name', 'dn_specialization']).agg({
        'target': 'nunique',
        'weight': 'sum'
    }).rename(columns={
        'target': 'n_priority_ins',
        'weight': 'total_synapses'
    }).reset_index()

    # Composite rank: geometric mean of normalised synapse total and breadth (n INs contacted).
    # This makes rankings stable across synapse thresholds — a DN must score well on
    # both dimensions, so one very strong but narrow connection won't dominate.
    max_syn = dn_summary['total_synapses'].max()
    max_ins = dn_summary['n_priority_ins'].max()
    dn_summary['norm_synapses'] = dn_summary['total_synapses'] / max_syn if max_syn > 0 else 0
    dn_summary['norm_ins'] = dn_summary['n_priority_ins'] / max_ins if max_ins > 0 else 0
    dn_summary['composite_rank_score'] = np.sqrt(dn_summary['norm_synapses'] * dn_summary['norm_ins'])
    dn_summary = dn_summary.sort_values('composite_rank_score', ascending=False)
    
    for i, (_, row) in enumerate(dn_summary.head(10).iterrows(), 1):
        print(f"{str(row['dn_name'])[:38]:<40} {row['n_priority_ins']:<15} {row['total_synapses']:<12.0f} {row['dn_specialization']}")
    
    # By IN
    print(f"\n\nTop 10 Priority INs by DN input:")
    print(f"{'Rank':<6} {'IN Name':<40} {'DNs':<10} {'Total Syn':<12} {'Cluster'}")
    print("-"*70)
    
    in_summary = dn_to_priority.groupby(['target', 'in_name', 'in_rank', 'in_cluster']).agg({
        'source': 'nunique',
        'weight': 'sum'
    }).rename(columns={
        'source': 'n_dns',
        'weight': 'total_synapses'
    }).reset_index().sort_values('in_rank')
    
    for _, row in in_summary.head(10).iterrows():
        print(f"{int(row['in_rank']):<6} {str(row['in_name'])[:38]:<40} {row['n_dns']:<10} {row['total_synapses']:<12.0f} {int(row['in_cluster'])}")
    
    # By DN specialization
    print(f"\n\nDN specialization breakdown:")
    spec_summary = dn_to_priority.groupby('dn_specialization').agg({
        'source': 'nunique',
        'target': 'nunique',
        'weight': 'sum'
    }).rename(columns={
        'source': 'n_dns',
        'target': 'n_priority_ins',
        'weight': 'total_synapses'
    })
    
    for spec, row in spec_summary.iterrows():
        print(f"  {spec:<20}: {row['n_dns']:3d} DNs → {row['n_priority_ins']:3d} Priority INs ({row['total_synapses']:,.0f} syn)")
    
    return dn_summary, in_summary


def find_critical_dn_in_pairs(dn_to_priority, top_n=20):
    """
    Identify strongest DN → Priority IN pairs.
    """
    
    print(f"\n\nTop {top_n} DN → Priority IN connections (strongest pairs):")
    print(f"{'DN Name':<35} → {'IN Name':<35} {'Syn':<8} {'IN Rank'}")
    print("-"*70)
    
    top_pairs = dn_to_priority.nlargest(top_n, 'weight')
    
    for _, row in top_pairs.iterrows():
        print(f"{str(row['dn_name'])[:33]:<35} → {str(row['in_name'])[:33]:<35} {row['weight']:<8.0f} #{int(row['in_rank'])}")
    
    return top_pairs


def create_dn_priority_visualizations(
    dn_to_priority,
    dn_summary,
    in_summary,
    top_priority,
    output_dir
):
    """
    Create comprehensive visualizations.
    """
    
    print("\n\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
    
    # ===== PANEL 1: Top DNs by priority IN connectivity =====
    ax = axes[0, 0]
    
    top_dns = dn_summary.head(15)
    
    colors = ['#FF6B6B' if spec == 'power_control' else '#4ECDC4' if spec == 'steering_control' else '#CCCCCC'
              for spec in top_dns['dn_specialization']]
    
    y_pos = np.arange(len(top_dns))
    bars = ax.barh(y_pos, top_dns['total_synapses'], color=colors, edgecolor='black', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(name)[:30] for name in top_dns['dn_name']], fontsize=9)
    ax.set_xlabel('Total Synapses to Priority INs', fontsize=11, fontweight='bold')
    ax.set_title('Top 15 DNs Controlling Priority Interneurons', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Add values
    for i, (bar, val) in enumerate(zip(bars, top_dns['total_synapses'])):
        ax.text(val + 50, bar.get_y() + bar.get_height()/2,
               f'{int(val):,}', va='center', fontsize=8, fontweight='bold')
    
    # ===== PANEL 2: Priority INs by DN input =====
    ax = axes[0, 1]
    
    top_ins = in_summary.head(15)
    
    # Color by cluster
    cluster_colors = {1: '#C5E1A5', 2: '#90CAF9', 3: '#FFCC80', 4: '#FF6B6B', 5: '#CE93D8'}
    colors_in = [cluster_colors.get(int(c), '#CCCCCC') for c in top_ins['in_cluster']]
    
    y_pos = np.arange(len(top_ins))
    bars = ax.barh(y_pos, top_ins['total_synapses'], color=colors_in, edgecolor='black', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{int(r)} {str(name)[:25]}" for r, name in zip(top_ins['in_rank'], top_ins['in_name'])],
                       fontsize=9)
    ax.set_xlabel('Total DN Input (Synapses)', fontsize=11, fontweight='bold')
    ax.set_title('Top 15 Priority INs by DN Input', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # ===== PANEL 3: DN specialization distribution =====
    ax = axes[0, 2]
    
    spec_counts = dn_to_priority.groupby('dn_specialization').agg({
        'source': 'nunique',
        'weight': 'sum'
    }).rename(columns={'source': 'n_dns', 'weight': 'total_synapses'})
    
    x = np.arange(len(spec_counts))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, spec_counts['n_dns'], width, 
                   label='# DNs', color='steelblue', alpha=0.8, edgecolor='black')
    
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, spec_counts['total_synapses'], width,
                   label='Total Synapses', color='coral', alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in spec_counts.index], fontsize=10)
    ax.set_ylabel('Number of DNs', fontsize=11, fontweight='bold', color='steelblue')
    ax2.set_ylabel('Total Synapses', fontsize=11, fontweight='bold', color='coral')
    ax.set_title('DN Specialization to Priority INs', fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax.grid(axis='y', alpha=0.3)
    
    # Add values
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
               f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ===== PANEL 4: Connectivity matrix (DN × Top Priority INs) =====
    ax = axes[1, 0]

    # Build matrix: Top 20 DNs × Top 20 Priority INs
    n_dns = min(20, len(dn_summary))
    n_ins = min(20, len(in_summary))
    top_20_dns = dn_summary.head(n_dns)['source'].tolist()
    top_20_ins = in_summary.head(n_ins)['target'].tolist()

    matrix = np.zeros((n_dns, n_ins))

    for i, dn in enumerate(top_20_dns):
        for j, in_id in enumerate(top_20_ins):
            conn = dn_to_priority[(dn_to_priority['source'] == dn) &
                                  (dn_to_priority['target'] == in_id)]
            if len(conn) > 0:
                matrix[i, j] = conn['weight'].values[0]

    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    ax.set_xticks(range(n_ins))
    ax.set_xticklabels([f"#{int(in_summary.iloc[i]['in_rank'])}" for i in range(n_ins)],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_dns))
    ax.set_yticklabels([str(dn_summary.iloc[i]['dn_name'])[:25] for i in range(n_dns)], fontsize=8)

    ax.set_xlabel('Priority INs (by rank)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Top DNs', fontsize=11, fontweight='bold')
    ax.set_title('DN → Priority IN Connectivity Matrix', fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Synapses')
    
    # ===== PANEL 5: Distribution of DN inputs per priority IN =====
    ax = axes[1, 1]
    
    dns_per_in = in_summary['n_dns']
    
    ax.hist(dns_per_in, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(dns_per_in.median(), color='red', linestyle='--', linewidth=2, 
              label=f'Median: {dns_per_in.median():.0f}')
    ax.set_xlabel('Number of DNs per Priority IN', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('DN Input Diversity to Priority INs', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # ===== PANEL 6: Synapse strength distribution =====
    ax = axes[1, 2]
    
    # DN→Priority vs DN→All INs comparison
    ax.hist(dn_to_priority['weight'], bins=30, alpha=0.7, 
           label='DN→Priority INs', color='#FF6B6B', edgecolor='black')
    
    ax.set_xlabel('Synapses per Connection', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Connection Strength Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add median
    median_weight = dn_to_priority['weight'].median()
    ax.axvline(median_weight, color='red', linestyle='--', linewidth=2,
              label=f'Median: {median_weight:.0f}')
    ax.legend()
    
    # Overall title
    fig.suptitle('Descending Neuron Control of Priority Interneurons\nWhich Brain Commands Drive Critical Flight Control Neurons?',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_dir / 'DN_to_Priority_IN_Analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'DN_to_Priority_IN_Analysis.pdf',
                format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("  ✓ Saved: DN_to_Priority_IN_Analysis.png/pdf")


def create_top_pathways_diagram(
    dn_to_priority,
    top_priority,
    output_dir,
    top_n=10
):
    """
    Create diagram showing top DN→IN pathways.
    """
    
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, f'Top {top_n} DN → Priority IN Pathways',
           ha='center', va='top', fontsize=18, fontweight='bold')
    ax.text(8, 9, 'Strongest Brain Command Connections to Critical Flight Control Neurons',
           ha='center', va='top', fontsize=12, style='italic', color='#555555')
    
    # Get top pairs
    top_pairs = dn_to_priority.nlargest(top_n, 'weight')
    
    y_start = 8
    y_step = 0.7
    
    for i, (_, pair) in enumerate(top_pairs.iterrows()):
        y = y_start - (i * y_step)
        
        # DN box
        dn_box = FancyBboxPatch((1, y-0.25), 5, 0.5,
                               boxstyle="round,pad=0.08",
                               facecolor='#FF6B6B' if pair['dn_specialization'] == 'power_control' else '#4ECDC4',
                               edgecolor='black',
                               linewidth=1.5)
        ax.add_patch(dn_box)
        ax.text(3.5, y, str(pair['dn_name'])[:35],
               ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Arrow
        from matplotlib.patches import FancyArrowPatch
        arrow = FancyArrowPatch((6, y), (10, y),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=3, color='#666666')
        ax.add_patch(arrow)
        
        # Synapse count on arrow
        ax.text(8, y+0.15, f'{int(pair["weight"])} syn',
               ha='center', va='bottom', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, pad=0.2))
        
        # IN box
        in_box = FancyBboxPatch((10, y-0.25), 5, 0.5,
                               boxstyle="round,pad=0.08",
                               facecolor='#95E1D3',
                               edgecolor='black',
                               linewidth=1.5)
        ax.add_patch(in_box)
        ax.text(12.5, y+0.1, f"#{int(pair['in_rank'])} {str(pair['in_name'])[:25]}",
               ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(12.5, y-0.1, f"Cluster {int(pair['in_cluster'])} | Score: {pair['in_score']:.0f}",
               ha='center', va='center', fontsize=7, style='italic')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', label='Power DN'),
        mpatches.Patch(facecolor='#4ECDC4', edgecolor='black', label='Steering DN'),
        mpatches.Patch(facecolor='#95E1D3', edgecolor='black', label='Priority IN')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'Top_DN_Priority_Pathways.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'Top_DN_Priority_Pathways.pdf',
                format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("  ✓ Saved: Top_DN_Priority_Pathways.png/pdf")


def main():
    print("\n" + "="*70)
    print("DN → PRIORITY INTERNEURON ANALYSIS")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    print("-"*70 + "\n")
    
    loader = ConnectomeDataLoader('config.yaml')
    loader.load_all_data(verbose=True)

    # Priority INs (from candidate prioritization)
    priority_ins = pd.read_csv('results/candidate_prioritization/all_interneurons_ranked.csv')
    print(f"  ✓ Priority INs: {len(priority_ins):,}")
    
    # DN→IN connections
    dn_in_conn = pd.read_csv('results/dn_pathways/dn_to_in_connections.csv')
    min_synapses = loader.config['analysis']['min_synapses']
    before = len(dn_in_conn)
    dn_in_conn = dn_in_conn[dn_in_conn['weight'] >= min_synapses].copy()
    print(f"  ✓ DN→IN connections: {len(dn_in_conn):,} (filtered from {before:,} at min={min_synapses} synapses)")
    
    # DN classifications
    dn_class = pd.read_csv('results/dn_pathways/dn_classifications.csv')
    print(f"  ✓ DN classifications: {len(dn_class)}")
    
    # Cluster assignments
    cluster_assignments = pd.read_csv('results/interneuron_clusters/premotor_interneurons_clustered.csv')
    print(f"  ✓ Cluster assignments: {len(cluster_assignments):,}\n")
    
    # Analyze
    dn_to_priority, top_priority = analyze_dn_to_priority_connections(
        priority_ins,
        dn_in_conn,
        dn_class,
        cluster_assignments,
        loader,
        top_n_ins=50
    )
    
    # Summaries
    dn_summary, in_summary = summarize_dn_priority_connections(dn_to_priority)
    
    # Top pairs
    top_pairs = find_critical_dn_in_pairs(dn_to_priority, top_n=20)
    
    # Save results
    print("\n" + "-"*70)
    print("Saving results...")
    print("-"*70 + "\n")
    
    output_dir = Path('results/dn_priority_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tables
    dn_to_priority.to_csv(output_dir / 'dn_to_priority_in_connections.csv', index=False)
    print(f"  ✓ dn_to_priority_in_connections.csv ({len(dn_to_priority):,} connections)")
    
    dn_summary.to_csv(output_dir / 'dn_summary_priority_control.csv', index=False)
    print(f"  ✓ dn_summary_priority_control.csv")
    
    in_summary.to_csv(output_dir / 'priority_in_dn_input.csv', index=False)
    print(f"  ✓ priority_in_dn_input.csv")
    
    top_pairs.to_csv(output_dir / 'top_dn_priority_pairs.csv', index=False)
    print(f"  ✓ top_dn_priority_pairs.csv\n")
    
    # Visualizations
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    create_dn_priority_visualizations(
        dn_to_priority,
        dn_summary,
        in_summary,
        top_priority,
        fig_dir
    )
    
    create_top_pathways_diagram(
        dn_to_priority,
        top_priority,
        fig_dir,
        top_n=10
    )
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")
    
    print("Key findings:")
    print(f"  • {dn_summary['source'].nunique()} DNs control top 50 priority INs")
    print(f"  • {len(dn_to_priority):,} total DN→Priority IN connections")
    print(f"  • {dn_to_priority['weight'].sum():,.0f} total synapses")
    print(f"  • Mean: {dn_to_priority['weight'].mean():.1f} synapses per connection")
    print(f"  • Median: {dn_to_priority['weight'].median():.1f} synapses per connection")
    
    print(f"\nTop DN controlling priority INs:")
    top_dn = dn_summary.iloc[0]
    print(f"  {top_dn['dn_name']}")
    print(f"  → Connects to {int(top_dn['n_priority_ins'])} priority INs")
    print(f"  → {int(top_dn['total_synapses']):,} total synapses")
    print(f"  → Specialization: {top_dn['dn_specialization']}")
    
    print(f"\nResults saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()