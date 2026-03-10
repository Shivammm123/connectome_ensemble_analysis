"""
Complete DN Pathway Analysis

Maps BOTH direct and indirect DN pathways:
1. DN → MN (direct motor control)
2. DN → IN → MN (indirect via premotor circuits)

Discovers which DNs use which strategy and why.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def find_direct_dn_mn_connections(
    connections: pd.DataFrame,
    dn_ids: list,
    wing_mn_ids: list,
    min_synapses: int = 3
):
    """
    Find DIRECT DN → MN connections (no INs).
    """
    
    print("Finding DIRECT DN → MN connections...")
    print("-"*70)
    
    # Filter: DN → MN connections
    direct_conn = connections[
        (connections['source'].isin(dn_ids)) &
        (connections['target'].isin(wing_mn_ids)) &
        (connections['weight'] >= min_synapses)
    ].copy()
    
    # Get unique DNs using direct pathway
    direct_dns = direct_conn['source'].unique()
    
    print(f"  Total DNs: {len(dn_ids):,}")
    print(f"  Wing motor neurons: {len(wing_mn_ids)}")
    print(f"  DNs with DIRECT MN connections: {len(direct_dns)}")
    print(f"  Direct DN→MN connections: {len(direct_conn):,}")
    print(f"  (min {min_synapses} synapses)")
    
    return direct_conn, direct_dns


def classify_dn_pathway_strategies(
    dn_ids: list,
    direct_dns: list,
    indirect_dns: list,
    direct_conn: pd.DataFrame,
    indirect_conn: pd.DataFrame,
    loader: ConnectomeDataLoader
):
    """
    Classify each DN by its pathway strategy.
    
    Categories:
    - Direct only: DN → MN
    - Indirect only: DN → IN → MN
    - Both: Uses both pathways
    - Neither: Doesn't connect to wing system
    """
    
    print("\nClassifying DN pathway strategies...")
    print("-"*70)
    
    dn_strategies = []
    
    for dn_id in dn_ids:
        uses_direct = dn_id in direct_dns
        uses_indirect = dn_id in indirect_dns
        
        # Get connectivity stats
        if uses_direct:
            direct_synapses = direct_conn[direct_conn['source'] == dn_id]['weight'].sum()
            direct_targets = len(direct_conn[direct_conn['source'] == dn_id])
        else:
            direct_synapses = 0
            direct_targets = 0
        
        if uses_indirect:
            indirect_synapses = indirect_conn[indirect_conn['source'] == dn_id]['weight'].sum()
            indirect_targets = len(indirect_conn[indirect_conn['source'] == dn_id])
        else:
            indirect_synapses = 0
            indirect_targets = 0
        
        # Classify strategy
        if uses_direct and uses_indirect:
            strategy = 'both'
        elif uses_direct:
            strategy = 'direct_only'
        elif uses_indirect:
            strategy = 'indirect_only'
        else:
            strategy = 'no_wing_connection'
        
        # Get DN name
        dn_name = loader.get_neuron_name(dn_id)
        
        dn_strategies.append({
            'dn_id': dn_id,
            'dn_name': dn_name,
            'strategy': strategy,
            'direct_mn_targets': direct_targets,
            'direct_synapses': direct_synapses,
            'indirect_in_targets': indirect_targets,
            'indirect_synapses': indirect_synapses,
            'total_targets': direct_targets + indirect_targets,
            'total_synapses': direct_synapses + indirect_synapses,
            'direct_fraction': direct_synapses / (direct_synapses + indirect_synapses) if (direct_synapses + indirect_synapses) > 0 else 0
        })
    
    strategies_df = pd.DataFrame(dn_strategies)
    strategies_df = strategies_df.sort_values('total_synapses', ascending=False)
    
    # Summary
    print(f"\n  DN pathway strategies:")
    strategy_counts = strategies_df['strategy'].value_counts()
    for strat, count in strategy_counts.items():
        print(f"    {strat:20s}: {count:4d} DNs")
    
    return strategies_df


def analyze_direct_pathway_by_muscle(
    direct_conn: pd.DataFrame,
    motor_pools: pd.DataFrame,
    loader: ConnectomeDataLoader
):
    """
    Analyze which muscle types receive direct DN input.
    """
    
    print("\nAnalyzing direct DN→muscle pathways...")
    print("-"*70)
    
    # Map MNs to muscles
    mn_to_muscle = {}
    for _, pool in motor_pools.iterrows():
        for mn_id in pool['motor_neuron_ids']:
            mn_to_muscle[mn_id] = {
                'muscle': pool['muscle'],
                'muscle_type': pool['muscle_type'],
                'muscle_group': pool['muscle_group']
            }
    
    # Add muscle info to connections
    direct_conn_muscles = direct_conn.copy()
    direct_conn_muscles['muscle'] = direct_conn_muscles['target'].map(
        lambda x: mn_to_muscle.get(x, {}).get('muscle', 'unknown')
    )
    direct_conn_muscles['muscle_type'] = direct_conn_muscles['target'].map(
        lambda x: mn_to_muscle.get(x, {}).get('muscle_type', 'unknown')
    )
    
    # Summary by muscle type
    print(f"\n  Direct DN connections by muscle type:")
    type_summary = direct_conn_muscles.groupby('muscle_type').agg({
        'source': 'nunique',
        'target': 'nunique',
        'weight': 'sum'
    }).rename(columns={'source': 'n_dns', 'target': 'n_mns', 'weight': 'total_synapses'})
    
    for mtype, row in type_summary.iterrows():
        print(f"    {mtype:15s}: {row['n_dns']:3d} DNs → {row['n_mns']:3d} MNs ({row['total_synapses']:5.0f} synapses)")
    
    # Summary by specific muscle
    print(f"\n  Direct DN connections by muscle:")
    muscle_summary = direct_conn_muscles.groupby('muscle').agg({
        'source': 'nunique',
        'weight': 'sum'
    }).rename(columns={'source': 'n_dns', 'weight': 'total_synapses'})
    muscle_summary = muscle_summary.sort_values('total_synapses', ascending=False)
    
    for muscle, row in muscle_summary.head(10).iterrows():
        print(f"    {muscle:10s}: {row['n_dns']:3d} DNs ({row['total_synapses']:5.0f} synapses)")
    
    return direct_conn_muscles, type_summary, muscle_summary


def create_pathway_comparison_visualizations(
    strategies_df: pd.DataFrame,
    direct_conn_muscles: pd.DataFrame,
    type_summary: pd.DataFrame,
    output_dir: Path
):
    """
    Create comprehensive pathway visualizations.
    """
    
    print("\nCreating visualizations...")
    print("-"*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)
    
    # 1. DN Strategy distribution
    ax = axes[0, 0]
    strategy_counts = strategies_df['strategy'].value_counts()
    
    colors = {
        'both': '#FFA07A',
        'direct_only': '#FF6B6B',
        'indirect_only': '#4ECDC4',
        'no_wing_connection': '#CCCCCC'
    }
    bar_colors = [colors.get(x, 'gray') for x in strategy_counts.index]
    
    strategy_counts.plot(kind='bar', ax=ax, color=bar_colors)
    ax.set_xlabel('Pathway Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of DNs', fontsize=12, fontweight='bold')
    ax.set_title('DN Pathway Strategies', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add counts
    for i, val in enumerate(strategy_counts.values):
        ax.text(i, val + 5, str(val), ha='center', va='bottom', fontweight='bold')
    
    # 2. Direct vs Indirect connectivity
    ax = axes[0, 1]
    
    connected_dns = strategies_df[strategies_df['strategy'] != 'no_wing_connection']
    
    x = np.arange(2)
    direct_total = connected_dns['direct_synapses'].sum()
    indirect_total = connected_dns['indirect_synapses'].sum()
    
    bars = ax.bar(x, [direct_total, indirect_total], 
                  color=['#FF6B6B', '#4ECDC4'])
    ax.set_xticks(x)
    ax.set_xticklabels(['Direct\n(DN→MN)', 'Indirect\n(DN→IN→MN)'], fontsize=11)
    ax.set_ylabel('Total Synapses', fontsize=12, fontweight='bold')
    ax.set_title('Direct vs Indirect Pathways', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 100,
               f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Top DNs using both pathways
    ax = axes[0, 2]
    
    both_dns = strategies_df[strategies_df['strategy'] == 'both'].head(10)
    
    if len(both_dns) > 0:
        x_pos = np.arange(len(both_dns))
        
        ax.barh(x_pos, both_dns['direct_synapses'], 
               label='Direct', color='#FF6B6B', alpha=0.7)
        ax.barh(x_pos, both_dns['indirect_synapses'], 
               left=both_dns['direct_synapses'],
               label='Indirect', color='#4ECDC4', alpha=0.7)
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels([name[:20] for name in both_dns['dn_name']], fontsize=9)
        ax.set_xlabel('Synapses', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 DNs Using Both Pathways', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    
    # 4. Direct pathway by muscle type
    ax = axes[1, 0]
    
    if len(type_summary) > 0:
        type_summary_sorted = type_summary.sort_values('total_synapses', ascending=True)
        colors_muscle = ['#FF6B6B' if 'power' in idx else '#4ECDC4' 
                        for idx in type_summary_sorted.index]
        
        type_summary_sorted['total_synapses'].plot(kind='barh', ax=ax, color=colors_muscle)
        ax.set_xlabel('Total Synapses (Direct DN→MN)', fontsize=12, fontweight='bold')
        ax.set_title('Direct Connections by Muscle Type', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    # 5. Direct fraction distribution
    ax = axes[1, 1]
    
    connected_dns_only = strategies_df[strategies_df['strategy'].isin(['both', 'direct_only', 'indirect_only'])]
    
    ax.hist(connected_dns_only['direct_fraction'], bins=20, 
           color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='50% threshold')
    ax.set_xlabel('Fraction of Direct Connectivity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of DNs', fontsize=12, fontweight='bold')
    ax.set_title('Direct vs Indirect Preference', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 6. Scatter: Direct vs Indirect connectivity
    ax = axes[1, 2]
    
    scatter_data = strategies_df[strategies_df['strategy'].isin(['both', 'direct_only', 'indirect_only'])]
    
    for strategy, color in [('direct_only', '#FF6B6B'), 
                           ('indirect_only', '#4ECDC4'),
                           ('both', '#FFA07A')]:
        subset = scatter_data[scatter_data['strategy'] == strategy]
        ax.scatter(subset['direct_synapses'], subset['indirect_synapses'],
                  c=color, label=strategy, alpha=0.6, s=50)
    
    ax.set_xlabel('Direct Synapses (DN→MN)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Indirect Synapses (DN→IN)', fontsize=12, fontweight='bold')
    ax.set_title('DN Pathway Preference', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add diagonal line (equal preference)
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dn_pathway_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: dn_pathway_comparison.png")


def main():
    print("\n" + "="*70)
    print("COMPLETE DN PATHWAY ANALYSIS")
    print("Mapping BOTH direct and indirect pathways")
    print("="*70 + "\n")
    
    # Load data
    loader = ConnectomeDataLoader('config.yaml')
    connections, neurons, name_mapping = loader.load_all_data(verbose=False)
    min_synapses = loader.config['analysis']['min_synapses']
    filtered_conn = loader.filter_connections(min_synapses=min_synapses, verbose=False)
    
    # Load previous results
    print("Loading data...")
    print("-"*70 + "\n")
    
    # DNs
    dns = pd.read_csv('results/cell_types/descending_neurons.csv')
    dn_ids = dns['Root ID'].tolist()
    print(f"  ✓ Descending neurons: {len(dn_ids):,}")
    
    # Wing MNs
    wing_mns = pd.read_csv('results/cell_types/wing_motor_neurons.csv')
    wing_mn_ids = wing_mns['Root ID'].tolist()
    print(f"  ✓ Wing motor neurons: {len(wing_mn_ids)}")
    
    # Motor pools
    motor_pools = pd.read_csv('results/motor_neurons/motor_pools.csv')
    motor_pools['motor_neuron_ids'] = motor_pools['motor_neuron_ids'].apply(eval)
    print(f"  ✓ Motor pools: {len(motor_pools)}")
    
    # Premotor INs
    premotor_ins = pd.read_csv('results/interneuron_clusters/premotor_interneurons_clustered.csv')
    premotor_in_ids = premotor_ins['Root ID'].tolist()
    print(f"  ✓ Premotor INs: {len(premotor_in_ids):,}\n")
    
    # Step 1: Find DIRECT DN→MN connections
    direct_conn, direct_dns = find_direct_dn_mn_connections(
        filtered_conn,
        dn_ids,
        wing_mn_ids,
        min_synapses=min_synapses
    )
    
    # Step 2: Get INDIRECT DN→IN connections (from previous analysis)
    indirect_file = Path('results/dn_pathways/dn_to_in_connections.csv')
    
    if indirect_file.exists():
        indirect_conn = pd.read_csv(indirect_file)
        indirect_dns = indirect_conn['source'].unique()
        print(f"\n  ✓ Loaded indirect DN→IN connections: {len(indirect_conn):,}")
        print(f"    DNs using indirect pathway: {len(indirect_dns)}")
    else:
        print("\n  ⚠ No indirect pathway data found!")
        print("    Run dn_pathway_mapping.py first to get DN→IN connections")
        indirect_conn = pd.DataFrame()
        indirect_dns = []
    
    # Step 3: Classify DN strategies
    dn_strategies = classify_dn_pathway_strategies(
        dn_ids,
        direct_dns,
        indirect_dns,
        direct_conn,
        indirect_conn,
        loader
    )
    
    # Step 4: Analyze direct pathways by muscle
    direct_muscles, type_summary, muscle_summary = analyze_direct_pathway_by_muscle(
        direct_conn,
        motor_pools,
        loader
    )
    
    # Save results
    print("\n" + "-"*70)
    print("Saving results")
    print("-"*70 + "\n")
    
    output_dir = Path('results/dn_pathways')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save strategy classifications
    dn_strategies.to_csv(output_dir / 'dn_pathway_strategies.csv', index=False)
    print(f"  ✓ dn_pathway_strategies.csv ({len(dn_strategies)} DNs)")
    
    # Save direct connections
    direct_conn.to_csv(output_dir / 'dn_to_mn_direct.csv', index=False)
    print(f"  ✓ dn_to_mn_direct.csv ({len(direct_conn):,} connections)")
    
    # Save direct connections with muscle info
    direct_muscles.to_csv(output_dir / 'dn_to_mn_by_muscle.csv', index=False)
    print(f"  ✓ dn_to_mn_by_muscle.csv")
    
    # Save summaries
    type_summary.to_csv(output_dir / 'direct_pathway_by_muscle_type.csv')
    muscle_summary.to_csv(output_dir / 'direct_pathway_by_muscle.csv')
    print(f"  ✓ Muscle summaries saved")
    
    # Visualizations
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    create_pathway_comparison_visualizations(
        dn_strategies,
        direct_muscles,
        type_summary,
        fig_dir
    )
    
    # Summary
    print("\n" + "="*70)
    print("COMPLETE PATHWAY ANALYSIS DONE")
    print("="*70 + "\n")
    
    print(f"DN Pathway Analysis:")
    print(f"  Total DNs analyzed:          {len(dn_strategies):4,}")
    
    strategy_counts = dn_strategies['strategy'].value_counts()
    for strat, count in strategy_counts.items():
        print(f"  {strat:25s}: {count:4d} DNs")
    
    print(f"\nDirect pathway statistics:")
    print(f"  Direct DN→MN connections:    {len(direct_conn):4,}")
    print(f"  Total direct synapses:       {direct_conn['weight'].sum():4,.0f}")
    
    if len(indirect_conn) > 0:
        print(f"\nIndirect pathway statistics:")
        print(f"  Indirect DN→IN connections:  {len(indirect_conn):4,}")
        print(f"  Total indirect synapses:     {indirect_conn['weight'].sum():4,.0f}")
        
        ratio = direct_conn['weight'].sum() / indirect_conn['weight'].sum()
        print(f"\n  Direct/Indirect ratio:       {ratio:.2f}")
    
    print(f"\nTop 5 DNs by total connectivity:")
    for i, (_, dn) in enumerate(dn_strategies.head(5).iterrows(), 1):
        print(f"  {i}. {dn['dn_name'][:35]:35s} ({dn['strategy']:15s})")
        print(f"     Direct: {dn['direct_synapses']:5.0f} syn, Indirect: {dn['indirect_synapses']:5.0f} syn")
    
    print(f"\nResults: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()