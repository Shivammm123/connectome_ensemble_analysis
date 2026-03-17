"""
Direct vs Indirect DN Pathways Analysis

Compares DNs that connect DIRECTLY to motor neurons vs those that
connect INDIRECTLY through interneurons.

Analyzes top 30 DNs from each category.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def load_config():
    """Load configuration."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_neuron_display_name(neuron_id, loader, prefix=""):
    """Get display name for a neuron."""
    name = loader.get_neuron_name(neuron_id)
    if name and name != str(neuron_id):
        return name
    else:
        short_id = str(neuron_id)[:12]
        return f"{prefix}{short_id}"


def analyze_direct_dn_mn_connections(connections, dns, wing_mns, min_synapses):
    """
    Identify DNs with DIRECT connections to motor neurons.
    """
    
    print("\n" + "="*70)
    print("ANALYZING DIRECT DN→MN CONNECTIONS")
    print("="*70 + "\n")
    
    # Filter for DN→MN connections (threshold already applied upstream)
    dn_mn_direct = connections[
        (connections['source'].isin(dns)) &
        (connections['target'].isin(wing_mns))
    ].copy()

    print(f"DN→MN direct connections (≥{min_synapses} syn): {len(dn_mn_direct):,}")
    print(f"Total synapses: {dn_mn_direct['weight'].sum():,.0f}")
    
    # Summarize by DN
    dn_direct_summary = dn_mn_direct.groupby('source').agg({
        'target': 'nunique',  # Number of MNs targeted
        'weight': ['sum', 'mean', 'max', 'count']
    }).reset_index()
    
    dn_direct_summary.columns = ['dn_id', 'n_target_mns', 'total_synapses', 
                                  'avg_synapses', 'max_synapses', 'n_connections']
    
    # Sort by total synaptic strength
    dn_direct_summary = dn_direct_summary.sort_values('total_synapses', ascending=False)
    
    print(f"\nDNs with direct MN connections: {len(dn_direct_summary):,}")
    print(f"Average MNs targeted per DN: {dn_direct_summary['n_target_mns'].mean():.1f}")
    print(f"Average synapses per DN: {dn_direct_summary['total_synapses'].mean():.0f}")
    
    return dn_direct_summary, dn_mn_direct


def analyze_indirect_dn_pathways(connections, dns, wing_mns, vnc_ins, min_synapses):
    """
    Identify DNs with INDIRECT connections (through INs) to motor neurons.
    """
    
    print("\n" + "="*70)
    print("ANALYZING INDIRECT DN→IN→MN CONNECTIONS")
    print("="*70 + "\n")
    
    # Step 1: Find premotor INs (IN→MN) — threshold already applied upstream
    in_mn_conn = connections[
        (connections['source'].isin(vnc_ins)) &
        (connections['target'].isin(wing_mns))
    ].copy()

    premotor_in_ids = in_mn_conn['source'].unique()
    print(f"Premotor INs: {len(premotor_in_ids):,}")

    # Step 2: Find DN→IN connections — threshold already applied upstream
    dn_in_conn = connections[
        (connections['source'].isin(dns)) &
        (connections['target'].isin(premotor_in_ids))
    ].copy()
    
    print(f"DN→IN connections (≥{min_synapses} syn): {len(dn_in_conn):,}")
    print(f"Total synapses: {dn_in_conn['weight'].sum():,.0f}")
    
    # Step 3: For each DN, count how many MNs it can reach through INs
    dn_indirect_summary = []
    
    print("\nCalculating indirect pathways...")
    
    for dn_id in dn_in_conn['source'].unique():
        # Get INs targeted by this DN
        target_ins = dn_in_conn[dn_in_conn['source'] == dn_id]['target'].unique()
        
        # Get MNs targeted by these INs
        target_mns = in_mn_conn[
            in_mn_conn['source'].isin(target_ins)
        ]['target'].unique()
        
        # DN→IN synapses
        dn_in_synapses = dn_in_conn[
            dn_in_conn['source'] == dn_id
        ]['weight'].sum()
        
        # Total IN→MN synaptic capacity of this DN's target INs.
        # NOTE: shared INs are counted once per DN that targets them — this is
        # the downstream *capacity* of the pathway, not an exclusive contribution.
        in_mn_capacity = in_mn_conn[
            in_mn_conn['source'].isin(target_ins)
        ]['weight'].sum()

        dn_indirect_summary.append({
            'dn_id': dn_id,
            'n_target_ins': len(target_ins),
            'n_reachable_mns': len(target_mns),
            'dn_in_synapses': dn_in_synapses,
            'in_mn_capacity': in_mn_capacity,   # renamed from in_mn_synapses to avoid confusion
            'pathway_strength': dn_in_synapses
        })
    
    dn_indirect_summary = pd.DataFrame(dn_indirect_summary)
    dn_indirect_summary = dn_indirect_summary.sort_values('pathway_strength', ascending=False)
    
    print(f"\nDNs with indirect pathways: {len(dn_indirect_summary):,}")
    print(f"Average INs targeted per DN: {dn_indirect_summary['n_target_ins'].mean():.1f}")
    print(f"Average MNs reachable per DN: {dn_indirect_summary['n_reachable_mns'].mean():.1f}")
    print(f"Average DN→IN synapses: {dn_indirect_summary['dn_in_synapses'].mean():.0f}")
    
    return dn_indirect_summary, dn_in_conn, in_mn_conn


def compare_direct_vs_indirect(dn_direct_summary, dn_indirect_summary, 
                               all_dns, loader):
    """
    Compare DNs with direct vs indirect pathways.
    Categorize DNs into: direct-only, indirect-only, both, neither.
    """
    
    print("\n" + "="*70)
    print("COMPARING DIRECT VS INDIRECT DN STRATEGIES")
    print("="*70 + "\n")
    
    direct_dns = set(dn_direct_summary['dn_id'])
    indirect_dns = set(dn_indirect_summary['dn_id'])
    
    # Categorize DNs
    both_dns = direct_dns & indirect_dns
    direct_only_dns = direct_dns - indirect_dns
    indirect_only_dns = indirect_dns - direct_dns
    neither_dns = set(all_dns) - (direct_dns | indirect_dns)
    
    print(f"Total DNs analyzed: {len(all_dns):,}")
    print(f"\nDN Categories:")
    print(f"  • Both direct AND indirect:  {len(both_dns):,} ({len(both_dns)/len(all_dns)*100:.1f}%)")
    print(f"  • Direct ONLY:               {len(direct_only_dns):,} ({len(direct_only_dns)/len(all_dns)*100:.1f}%)")
    print(f"  • Indirect ONLY:             {len(indirect_only_dns):,} ({len(indirect_only_dns)/len(all_dns)*100:.1f}%)")
    print(f"  • Neither (no wing MN conn): {len(neither_dns):,} ({len(neither_dns)/len(all_dns)*100:.1f}%)")
    
    # Create comprehensive DN dataframe
    dn_comparison = []
    
    for dn_id in all_dns:
        row = {'dn_id': dn_id, 'dn_name': get_neuron_display_name(dn_id, loader, "DN_")}
        
        # Direct pathway info
        if dn_id in direct_dns:
            direct_info = dn_direct_summary[dn_direct_summary['dn_id'] == dn_id].iloc[0]
            row['has_direct'] = True
            row['direct_target_mns'] = direct_info['n_target_mns']
            row['direct_synapses'] = direct_info['total_synapses']
        else:
            row['has_direct'] = False
            row['direct_target_mns'] = 0
            row['direct_synapses'] = 0
        
        # Indirect pathway info
        if dn_id in indirect_dns:
            indirect_info = dn_indirect_summary[dn_indirect_summary['dn_id'] == dn_id].iloc[0]
            row['has_indirect'] = True
            row['indirect_target_ins'] = indirect_info['n_target_ins']
            row['indirect_reachable_mns'] = indirect_info['n_reachable_mns']
            row['indirect_synapses'] = indirect_info['dn_in_synapses']
        else:
            row['has_indirect'] = False
            row['indirect_target_ins'] = 0
            row['indirect_reachable_mns'] = 0
            row['indirect_synapses'] = 0
        
        # Categorize
        if dn_id in both_dns:
            row['category'] = 'both'
        elif dn_id in direct_only_dns:
            row['category'] = 'direct_only'
        elif dn_id in indirect_only_dns:
            row['category'] = 'indirect_only'
        else:
            row['category'] = 'neither'
        
        # Total connectivity
        row['total_synapses'] = row['direct_synapses'] + row['indirect_synapses']
        
        dn_comparison.append(row)
    
    dn_comparison_df = pd.DataFrame(dn_comparison)
    dn_comparison_df = dn_comparison_df.sort_values('total_synapses', ascending=False)
    
    return dn_comparison_df, {
        'both': both_dns,
        'direct_only': direct_only_dns,
        'indirect_only': indirect_only_dns,
        'neither': neither_dns
    }


def _save(fig, path):
    """Save as PNG and PDF, handling locked PDF files on Windows."""
    fig.savefig(str(path) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    try:
        fig.savefig(str(path) + '.pdf', format='pdf', bbox_inches='tight', facecolor='white')
        print(f"  ✓ {path.name}.png/pdf")
    except PermissionError:
        print(f"  ✓ {path.name}.png  (close PDF viewer to also save PDF)")
    plt.close(fig)


def create_comparison_figures(dn_comparison_df, output_dir):
    """Create one focused figure per topic instead of one crowded subplot grid."""

    from matplotlib.patches import Patch

    print("\n" + "="*70)
    print("CREATING COMPARISON FIGURES")
    print("="*70 + "\n")

    total_dns = len(dn_comparison_df)

    top_direct = dn_comparison_df[
        dn_comparison_df['has_direct']
    ].nlargest(30, 'direct_synapses')

    top_indirect = dn_comparison_df[
        dn_comparison_df['has_indirect']
    ].nlargest(30, 'indirect_synapses')

    direct_mns   = dn_comparison_df[dn_comparison_df['has_direct']]['direct_target_mns']
    indirect_mns = dn_comparison_df[dn_comparison_df['has_indirect']]['indirect_reachable_mns']

    # ── Figure 1: Strategy overview (category bar + MN-reach boxplot) ────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300,
                                    layout='constrained')

    category_counts = (dn_comparison_df['category']
                       .value_counts()
                       .reindex(['both', 'direct_only', 'indirect_only', 'neither'],
                                fill_value=0))
    cat_colors = ['#9370DB', '#FF6B6B', '#4ECDC4', '#95A5A6']
    cat_labels = ['Both\n(Direct & Indirect)', 'Direct\nOnly',
                  'Indirect\nOnly', 'Neither']

    bars = ax1.bar(range(4), category_counts.values,
                   color=cat_colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(cat_labels, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of DNs', fontsize=12, fontweight='bold')
    ax1.set_title('DN Pathway Strategy Distribution', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, category_counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1,
                 f'{val}\n({val / total_dns * 100:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    bp = ax2.boxplot([direct_mns, indirect_mns],
                     tick_labels=['Direct\nDN→MN', 'Indirect\nDN→IN→MN'],
                     patch_artist=True, showfliers=False, widths=0.5)
    bp['boxes'][0].set_facecolor('#FF6B6B')
    bp['boxes'][1].set_facecolor('#4ECDC4')
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    ax2.set_ylabel('Number of Motor Neurons Reached', fontsize=12, fontweight='bold')
    ax2.set_title('MN Reach: Direct vs Indirect', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('DN Pathway Strategy Overview', fontsize=15, fontweight='bold')
    _save(fig, output_dir / 'fig1_strategy_overview')

    # ── Figure 2: Top 30 direct DNs ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 12), dpi=300, layout='constrained')

    colors_d = ['#9370DB' if c == 'both' else '#FF6B6B'
                for c in top_direct['category']]
    y_pos = np.arange(len(top_direct))
    ax.barh(y_pos, top_direct['direct_synapses'],
            color=colors_d, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_direct['dn_name'], fontsize=10)
    ax.set_xlabel('Total Synapses (Direct DN→MN)', fontsize=12, fontweight='bold')
    ax.set_title('Top 30 DNs — Direct MN Connectivity', fontsize=14, fontweight='bold', pad=12)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    ax.legend(handles=[
        Patch(facecolor='#9370DB', edgecolor='black', label='Also has indirect pathway'),
        Patch(facecolor='#FF6B6B', edgecolor='black', label='Direct pathway only'),
    ], fontsize=10, loc='lower right')

    _save(fig, output_dir / 'fig2_top30_direct_dns')

    # ── Figure 3: Top 30 indirect DNs ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 12), dpi=300, layout='constrained')

    colors_i = ['#9370DB' if c == 'both' else '#4ECDC4'
                for c in top_indirect['category']]
    y_pos = np.arange(len(top_indirect))
    ax.barh(y_pos, top_indirect['indirect_synapses'],
            color=colors_i, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_indirect['dn_name'], fontsize=10)
    ax.set_xlabel('Total Synapses (DN→IN)', fontsize=12, fontweight='bold')
    ax.set_title('Top 30 DNs — Indirect Pathway Connectivity', fontsize=14, fontweight='bold', pad=12)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    ax.legend(handles=[
        Patch(facecolor='#9370DB', edgecolor='black', label='Also has direct pathway'),
        Patch(facecolor='#4ECDC4', edgecolor='black', label='Indirect pathway only'),
    ], fontsize=10, loc='lower right')

    _save(fig, output_dir / 'fig3_top30_indirect_dns')

    # ── Figure 4: Direct vs Indirect scatter (DNs with both) ─────────────
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300, layout='constrained')

    both_dns_df = dn_comparison_df[dn_comparison_df['category'] == 'both']
    if len(both_dns_df) > 0:
        ax.scatter(both_dns_df['direct_synapses'], both_dns_df['indirect_synapses'],
                   s=80, c='#9370DB', alpha=0.65, edgecolors='black', linewidth=0.8)
        max_val = max(both_dns_df['direct_synapses'].max(),
                      both_dns_df['indirect_synapses'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.6,
                label='Equal direct / indirect usage')
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No DNs with both pathways',
                ha='center', va='center', transform=ax.transAxes, fontsize=13)

    ax.set_xlabel('Direct Synapses (DN→MN)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Indirect Synapses (DN→IN)', fontsize=12, fontweight='bold')
    ax.set_title('Direct vs Indirect Synapse Use\n(DNs with Both Pathways)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.grid(alpha=0.3)

    _save(fig, output_dir / 'fig4_direct_vs_indirect_scatter')

    # ── Figure 5: Side-by-side direct vs indirect top 30 ─────────────────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 12), dpi=300,
                                      layout='constrained')

    # Left: top 30 direct
    colors_d = ['#9370DB' if c == 'both' else '#FF6B6B'
                for c in top_direct['category']]
    y_pos = np.arange(len(top_direct))
    ax_l.barh(y_pos, top_direct['direct_synapses'],
              color=colors_d, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(top_direct['dn_name'], fontsize=10)
    ax_l.set_xlabel('Total Synapses (DN→MN)', fontsize=12, fontweight='bold')
    ax_l.set_title('Direct Pathway\nTop 30 DNs', fontsize=13, fontweight='bold', pad=10)
    ax_l.grid(axis='x', alpha=0.3)
    ax_l.invert_yaxis()
    ax_l.legend(handles=[
        Patch(facecolor='#9370DB', edgecolor='black', label='Also has indirect'),
        Patch(facecolor='#FF6B6B', edgecolor='black', label='Direct only'),
    ], fontsize=10, loc='lower right')

    # Right: top 30 indirect
    colors_i = ['#9370DB' if c == 'both' else '#4ECDC4'
                for c in top_indirect['category']]
    y_pos = np.arange(len(top_indirect))
    ax_r.barh(y_pos, top_indirect['indirect_synapses'],
              color=colors_i, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(top_indirect['dn_name'], fontsize=10)
    ax_r.set_xlabel('Total Synapses (DN→IN)', fontsize=12, fontweight='bold')
    ax_r.set_title('Indirect Pathway\nTop 30 DNs', fontsize=13, fontweight='bold', pad=10)
    ax_r.grid(axis='x', alpha=0.3)
    ax_r.invert_yaxis()
    ax_r.legend(handles=[
        Patch(facecolor='#9370DB', edgecolor='black', label='Also has direct'),
        Patch(facecolor='#4ECDC4', edgecolor='black', label='Indirect only'),
    ], fontsize=10, loc='lower right')

    fig.suptitle('Top 30 DNs: Direct vs Indirect Pathway Side-by-Side',
                 fontsize=15, fontweight='bold')
    _save(fig, output_dir / 'fig5_direct_vs_indirect_sidebyside')


def main():
    print("\n" + "="*80)
    print(" "*20 + "DN DIRECT vs INDIRECT PATHWAY ANALYSIS")
    print("="*80 + "\n")
    
    # Load config
    config = load_config()
    min_synapses = config['analysis']['min_synapses']   # was config.get('min_synapses', 10) — wrong key

    print(f"Configuration: min_synapses = {min_synapses}\n")

    # Load data
    print("Loading data...")
    loader = ConnectomeDataLoader('config.yaml')
    connections, neurons, _ = loader.load_all_data(verbose=False)
    # Pre-filter once — consistent with the rest of the pipeline
    connections = loader.filter_connections(min_synapses=min_synapses, verbose=False)
    
    motor_pools = pd.read_csv('results/motor_neurons/motor_pools.csv')
    wing_mns = []
    for _, pool in motor_pools.iterrows():
        wing_mns.extend(eval(pool['motor_neuron_ids']))
    wing_mns = list(set(wing_mns))
    
    vnc_ins = neurons[
        neurons['Super Class'] == 'ventral_nerve_cord_intrinsic'
    ]['Root ID'].tolist()
    
    dns = neurons[
        neurons['Super Class'] == 'descending'
    ]['Root ID'].tolist()
    
    print(f"✓ Wing motor neurons: {len(wing_mns)}")
    print(f"✓ VNC interneurons: {len(vnc_ins):,}")
    print(f"✓ Descending neurons: {len(dns):,}")
    
    # Analyze direct pathways
    dn_direct_summary, dn_mn_direct = analyze_direct_dn_mn_connections(
        connections, dns, wing_mns, min_synapses
    )
    
    # Analyze indirect pathways
    dn_indirect_summary, dn_in_conn, in_mn_conn = analyze_indirect_dn_pathways(
        connections, dns, wing_mns, vnc_ins, min_synapses
    )
    
    # Compare
    dn_comparison_df, dn_categories = compare_direct_vs_indirect(
        dn_direct_summary, dn_indirect_summary, dns, loader
    )
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70 + "\n")
    
    output_dir = Path('results/dn_direct_vs_indirect')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comprehensive comparison
    dn_comparison_df.to_csv(output_dir / 'dn_pathway_comparison.csv', index=False)
    print("✓ dn_pathway_comparison.csv")
    
    # Save top 30 lists
    top30_direct = dn_comparison_df[
        dn_comparison_df['has_direct'] == True
    ].nlargest(30, 'direct_synapses')
    top30_direct.to_csv(output_dir / 'top30_direct_dns.csv', index=False)
    print("✓ top30_direct_dns.csv")
    
    top30_indirect = dn_comparison_df[
        dn_comparison_df['has_indirect'] == True
    ].nlargest(30, 'indirect_synapses')
    top30_indirect.to_csv(output_dir / 'top30_indirect_dns.csv', index=False)
    print("✓ top30_indirect_dns.csv")
    
    # Save category lists
    for category, dn_set in dn_categories.items():
        category_df = pd.DataFrame({
            'dn_id': list(dn_set),
            'dn_name': [get_neuron_display_name(dn_id, loader, "DN_") for dn_id in dn_set]
        })
        category_df.to_csv(output_dir / f'dns_{category}.csv', index=False)
        print(f"✓ dns_{category}.csv")
    
    # Save detailed connections
    dn_mn_direct.to_csv(output_dir / 'dn_to_mn_direct_connections.csv', index=False)
    print("✓ dn_to_mn_direct_connections.csv")
    
    # Create figures
    create_comparison_figures(dn_comparison_df, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print(" "*25 + "ANALYSIS COMPLETE!")
    print("="*80 + "\n")
    
    print("KEY FINDINGS:")
    print(f"  • DNs with BOTH pathways: {len(dn_categories['both'])}")
    print(f"  • DNs with DIRECT only: {len(dn_categories['direct_only'])}")
    print(f"  • DNs with INDIRECT only: {len(dn_categories['indirect_only'])}")
    
    print(f"\nTop Direct DN: {top30_direct.iloc[0]['dn_name']}")
    print(f"  • Direct synapses: {top30_direct.iloc[0]['direct_synapses']:.0f}")
    print(f"  • MNs targeted: {top30_direct.iloc[0]['direct_target_mns']:.0f}")
    
    print(f"\nTop Indirect DN: {top30_indirect.iloc[0]['dn_name']}")
    print(f"  • Indirect synapses: {top30_indirect.iloc[0]['indirect_synapses']:.0f}")
    print(f"  • INs targeted: {top30_indirect.iloc[0]['indirect_target_ins']:.0f}")
    print(f"  • MNs reachable: {top30_indirect.iloc[0]['indirect_reachable_mns']:.0f}")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
