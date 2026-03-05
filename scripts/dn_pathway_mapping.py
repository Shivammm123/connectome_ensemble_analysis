"""
DN → IN → MN Pathway Mapping

Map complete pathways from descending neurons to motor neurons.

Discovers:
1. Which DNs connect to which IN clusters
2. DN specialization (power vs steering DNs)
3. Complete brain→muscle pathways
4. Hub pathways (critical control routes)

Based on Cheong et al. 2024 methodology.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def find_dn_to_in_connections(
    connections: pd.DataFrame,
    dn_ids: list,
    premotor_in_ids: list,
    min_synapses: int = 3
):
    """
    Find connections from DNs to premotor INs.
    """
    
    print("Finding DN → IN connections...")
    print("-"*70)
    
    # Filter: DN → IN connections
    dn_in_conn = connections[
        (connections['source'].isin(dn_ids)) &
        (connections['target'].isin(premotor_in_ids)) &
        (connections['weight'] >= min_synapses)
    ].copy()
    
    # Get unique DNs that connect
    connected_dns = dn_in_conn['source'].unique()
    
    print(f"  Total DNs: {len(dn_ids):,}")
    print(f"  Premotor INs: {len(premotor_in_ids):,}")
    print(f"  DNs connecting to premotor INs: {len(connected_dns)}")
    print(f"  DN→IN connections: {len(dn_in_conn):,}")
    print(f"  (min {min_synapses} synapses)")
    
    return dn_in_conn, connected_dns


def classify_dns_by_target_clusters(
    dn_in_conn: pd.DataFrame,
    in_cluster_assignments: pd.DataFrame,
    cluster_info: pd.DataFrame,
    loader: ConnectomeDataLoader
):
    """
    Classify DNs by which IN clusters they primarily target.
    """
    
    print("\nClassifying DNs by target clusters...")
    print("-"*70)
    
    # Merge to get cluster info for each IN
    dn_in_with_clusters = dn_in_conn.merge(
        in_cluster_assignments[['interneuron_id', 'cluster']],
        left_on='target',
        right_on='interneuron_id',
        how='left'
    )
    
    # For each DN, compute connectivity to each cluster
    dn_cluster_conn = dn_in_with_clusters.groupby(['source', 'cluster'])['weight'].sum().reset_index()
    
    # Classify each DN
    dn_classifications = []
    
    for dn_id in dn_in_with_clusters['source'].unique():
        dn_data = dn_cluster_conn[dn_cluster_conn['source'] == dn_id]
        
        if len(dn_data) == 0:
            continue
        
        # Find primary target cluster
        primary_cluster = dn_data.loc[dn_data['weight'].idxmax(), 'cluster']
        
        # Get cluster function
        cluster_func = cluster_info[cluster_info['cluster_id'] == primary_cluster]['functional_type'].values
        if len(cluster_func) > 0:
            dn_specialization = cluster_func[0]
        else:
            dn_specialization = 'unknown'
        
        # Compute cluster connectivity profile
        cluster_weights = {}
        for _, row in dn_data.iterrows():
            cluster_weights[f'cluster_{int(row["cluster"])}'] = row['weight']
        
        # Total connectivity
        total_conn = dn_data['weight'].sum()
        
        # Number of clusters targeted
        n_clusters = len(dn_data)
        
        # Get DN name
        dn_name = loader.get_neuron_name(dn_id)
        
        dn_classifications.append({
            'dn_id': dn_id,
            'dn_name': dn_name,
            'primary_cluster': int(primary_cluster),
            'specialization': dn_specialization,
            'n_clusters_targeted': n_clusters,
            'total_connectivity': total_conn,
            'n_in_targets': len(dn_in_with_clusters[dn_in_with_clusters['source'] == dn_id]),
            **cluster_weights
        })
    
    dn_class_df = pd.DataFrame(dn_classifications)
    dn_class_df = dn_class_df.sort_values('total_connectivity', ascending=False)
    
    # Summary
    print(f"\n  DNs classified: {len(dn_class_df)}")
    
    spec_counts = dn_class_df['specialization'].value_counts()
    print(f"\n  DN specializations:")
    for spec, count in spec_counts.items():
        print(f"    {spec:25s}: {count:3d} DNs")
    
    return dn_class_df


def find_complete_pathways(
    dn_in_conn: pd.DataFrame,
    in_mn_conn: pd.DataFrame,
    dn_class_df: pd.DataFrame,
    motor_pools: pd.DataFrame,
    loader: ConnectomeDataLoader,
    top_n_dns: int = 20
):
    """
    Trace complete DN → IN → MN → Muscle pathways.
    """
    
    print("\nTracing complete pathways...")
    print("-"*70)
    
    # Get top DNs
    top_dns = dn_class_df.head(top_n_dns)
    
    # For each top DN, find complete pathways
    pathways = []
    
    for _, dn_row in top_dns.iterrows():
        dn_id = dn_row['dn_id']
        dn_name = dn_row['dn_name']
        
        # Get INs this DN connects to
        dn_targets = dn_in_conn[dn_in_conn['source'] == dn_id]
        
        for _, dn_in_edge in dn_targets.iterrows():
            in_id = dn_in_edge['target']
            dn_in_weight = dn_in_edge['weight']
            
            # Get MNs this IN connects to
            in_targets = in_mn_conn[in_mn_conn['source'] == in_id]
            
            for _, in_mn_edge in in_targets.iterrows():
                mn_id = in_mn_edge['target']
                in_mn_weight = in_mn_edge['weight']
                
                # Find which motor pool this MN belongs to
                muscle = 'unknown'
                muscle_type = 'unknown'
                
                for _, pool in motor_pools.iterrows():
                    if mn_id in pool['motor_neuron_ids']:
                        muscle = pool['muscle']
                        muscle_type = pool['muscle_type']
                        break
                
                pathways.append({
                    'dn_id': dn_id,
                    'dn_name': dn_name,
                    'in_id': in_id,
                    'mn_id': mn_id,
                    'muscle': muscle,
                    'muscle_type': muscle_type,
                    'dn_in_synapses': dn_in_weight,
                    'in_mn_synapses': in_mn_weight,
                    'pathway_strength': dn_in_weight * in_mn_weight
                })
    
    pathways_df = pd.DataFrame(pathways)
    
    print(f"  Complete pathways found: {len(pathways_df):,}")
    print(f"  Unique DN→IN→MN chains: {len(pathways_df[['dn_id', 'in_id', 'mn_id']].drop_duplicates()):,}")
    
    return pathways_df


def create_dn_pathway_visualizations(
    dn_class_df: pd.DataFrame,
    dn_in_conn: pd.DataFrame,
    cluster_info: pd.DataFrame,
    output_dir: Path
):
    """
    Create visualizations of DN → IN pathways.
    """
    
    print("\nCreating visualizations...")
    print("-"*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # 1. DN specialization distribution
    ax = axes[0, 0]
    spec_counts = dn_class_df['specialization'].value_counts()
    
    colors = {
        'power_control': '#FF6B6B',
        'steering_control': '#4ECDC4',
        'unknown': '#CCCCCC'
    }
    bar_colors = [colors.get(x, 'gray') for x in spec_counts.index]
    
    spec_counts.plot(kind='bar', ax=ax, color=bar_colors)
    ax.set_xlabel('DN Specialization', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of DNs', fontsize=12, fontweight='bold')
    ax.set_title('DN Functional Specialization', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add counts
    for i, val in enumerate(spec_counts.values):
        ax.text(i, val + 1, str(val), ha='center', va='bottom', fontweight='bold')
    
    # 2. Top DNs by connectivity
    ax = axes[0, 1]
    top_dns = dn_class_df.head(15)
    
    colors_top = [colors.get(spec, 'gray') for spec in top_dns['specialization']]
    
    y_pos = np.arange(len(top_dns))
    ax.barh(y_pos, top_dns['total_connectivity'], color=colors_top)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name[:25] for name in top_dns['dn_name']], fontsize=9)
    ax.set_xlabel('Total IN Connectivity (synapses)', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 DNs by Connectivity', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # 3. DN → Cluster connectivity matrix
    ax = axes[1, 0]
    
    # Build matrix: Top DNs × Clusters
    top_20_dns = dn_class_df.head(20)
    cluster_cols = [f'cluster_{i}' for i in range(1, 6)]
    
    # Extract cluster connectivity
    dn_cluster_matrix = []
    for _, dn in top_20_dns.iterrows():
        row = []
        for col in cluster_cols:
            row.append(dn.get(col, 0))
        dn_cluster_matrix.append(row)
    
    dn_cluster_matrix = np.array(dn_cluster_matrix)
    
    # Plot heatmap
    im = ax.imshow(dn_cluster_matrix, aspect='auto', cmap='hot', interpolation='nearest')
    
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'C{i+1}' for i in range(5)], fontsize=10)
    ax.set_yticks(range(len(top_20_dns)))
    ax.set_yticklabels([name[:20] for name in top_20_dns['dn_name']], fontsize=8)
    
    ax.set_xlabel('IN Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Descending Neuron', fontsize=12, fontweight='bold')
    ax.set_title('DN → Cluster Connectivity', fontsize=14, fontweight='bold')
    
    # Color cluster labels
    cluster_functions = cluster_info['functional_type'].tolist()
    for i, (label, func) in enumerate(zip(ax.get_xticklabels(), cluster_functions)):
        if func == 'power_control':
            label.set_color('#FF6B6B')
            label.set_fontweight('bold')
        else:
            label.set_color('#4ECDC4')
    
    plt.colorbar(im, ax=ax, label='Synapses')
    
    # 4. Number of clusters per DN
    ax = axes[1, 1]
    
    cluster_dist = dn_class_df['n_clusters_targeted'].value_counts().sort_index()
    cluster_dist.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_xlabel('Number of Clusters Targeted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of DNs', fontsize=12, fontweight='bold')
    ax.set_title('DN Cluster Targeting Breadth', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dn_pathway_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: dn_pathway_analysis.png")


def main():
    print("\n" + "="*70)
    print("DN → IN → MN PATHWAY MAPPING")
    print("Mapping brain commands to motor circuits")
    print("="*70 + "\n")
    
    # Load data
    loader = ConnectomeDataLoader('config.yaml')
    connections, neurons, name_mapping = loader.load_all_data(verbose=False)
    
    # Filter connections
    filtered_conn = loader.filter_connections(min_synapses=3, verbose=False)
    
    # Load previous results
    print("Loading previous results...")
    print("-"*70 + "\n")
    
    # DNs
    dns = pd.read_csv('results/cell_types/descending_neurons.csv')
    dn_ids = dns['Root ID'].tolist()
    print(f"  ✓ Descending neurons: {len(dn_ids):,}")
    
    # Premotor INs with clusters
    premotor_ins = pd.read_csv('results/interneuron_clusters/premotor_interneurons_clustered.csv')
    in_cluster_assignments = premotor_ins[['Root ID', 'cluster']].copy()
    in_cluster_assignments.columns = ['interneuron_id', 'cluster']
    premotor_in_ids = premotor_ins['Root ID'].tolist()
    print(f"  ✓ Premotor INs: {len(premotor_in_ids):,}")
    
    # Cluster info
    cluster_info = pd.read_csv('results/interneuron_clusters/cluster_characteristics.csv')
    print(f"  ✓ IN clusters: {len(cluster_info)}")
    
    # IN→MN connections
    in_mn_conn = pd.read_csv('results/interneuron_clusters/in_to_mn_connections.csv')
    print(f"  ✓ IN→MN connections: {len(in_mn_conn):,}")
    
    # Motor pools
    motor_pools = pd.read_csv('results/motor_neurons/motor_pools.csv')
    motor_pools['motor_neuron_ids'] = motor_pools['motor_neuron_ids'].apply(eval)
    print(f"  ✓ Motor pools: {len(motor_pools)}\n")
    
    # Step 1: Find DN→IN connections
    dn_in_conn, connected_dns = find_dn_to_in_connections(
        filtered_conn,
        dn_ids,
        premotor_in_ids,
        min_synapses=3
    )
    
    if len(dn_in_conn) == 0:
        print("\n❌ No DN→IN connections found!")
        print("DNs might not directly connect to these premotor INs.")
        print("This suggests multi-step pathways (DN→intermediate IN→premotor IN)")
        sys.exit(1)
    
    # Step 2: Classify DNs
    dn_class_df = classify_dns_by_target_clusters(
        dn_in_conn,
        in_cluster_assignments,
        cluster_info,
        loader
    )
    
    # Step 3: Find complete pathways
    complete_pathways = find_complete_pathways(
        dn_in_conn,
        in_mn_conn,
        dn_class_df,
        motor_pools,
        loader,
        top_n_dns=20
    )
    
    # Save results
    print("\n" + "-"*70)
    print("Saving results")
    print("-"*70 + "\n")
    
    output_dir = Path('results/dn_pathways')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save DN classifications
    dn_class_df.to_csv(output_dir / 'dn_classifications.csv', index=False)
    print(f"  ✓ dn_classifications.csv ({len(dn_class_df)} DNs)")
    
    # Save DN→IN connections
    dn_in_conn.to_csv(output_dir / 'dn_to_in_connections.csv', index=False)
    print(f"  ✓ dn_to_in_connections.csv")
    
    # Save complete pathways
    complete_pathways.to_csv(output_dir / 'complete_pathways.csv', index=False)
    print(f"  ✓ complete_pathways.csv ({len(complete_pathways):,} pathways)")
    
    # Visualizations
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    create_dn_pathway_visualizations(
        dn_class_df,
        dn_in_conn,
        cluster_info,
        fig_dir
    )
    
    # Summary
    print("\n" + "="*70)
    print("DN PATHWAY MAPPING COMPLETE")
    print("="*70 + "\n")
    
    print(f"Pathways discovered:")
    print(f"  DNs connecting to premotor INs:  {len(dn_class_df):4}")
    print(f"  DN→IN connections:               {len(dn_in_conn):4,}")
    print(f"  Complete DN→IN→MN→Muscle paths:  {len(complete_pathways):4,}")
    
    print(f"\nDN specializations:")
    for spec, count in dn_class_df['specialization'].value_counts().items():
        print(f"  {spec:25s}: {count:3} DNs")
    
    print(f"\nTop 5 most connected DNs:")
    for i, (_, dn) in enumerate(dn_class_df.head(5).iterrows(), 1):
        print(f"  {i}. {dn['dn_name'][:40]:40s} ({dn['specialization']:15s}) - {dn['n_in_targets']:3} INs")
    
    print(f"\nResults: {output_dir}")
    print("\nNext: Visualization & summary scripts!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()