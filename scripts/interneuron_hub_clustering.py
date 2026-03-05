"""
Interneuron Hub Clustering

Discover premotor interneuron networks that control wing motor neurons.

Analysis approach (based on Azevedo et al. 2023, Cheong et al. 2024):
1. Find INs connecting to wing MNs
2. Cluster INs by connectivity patterns (which MN pools they target)
3. Identify functional modules (power vs steering networks)
4. Find hub neurons (high centrality, critical nodes)

This reveals the circuit organization between DNs and MNs.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import networkx as nx

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader
from utils.similarity import build_connectivity_vectors, cosine_similarity_matrix


def find_premotor_interneurons(
    connections: pd.DataFrame,
    motor_neuron_ids: list,
    vnc_interneurons: pd.DataFrame,
    min_synapses: int = 3
):
    """
    Find interneurons that connect to wing motor neurons.
    
    These are the premotor circuits we'll analyze.
    """
    
    print("Finding premotor interneurons...")
    print("-"*70)
    
    # Get VNC IN IDs
    vnc_in_ids = set(vnc_interneurons['Root ID'].unique())
    
    # Find connections: IN → MN
    premotor_conn = connections[
        (connections['source'].isin(vnc_in_ids)) &
        (connections['target'].isin(motor_neuron_ids)) &
        (connections['weight'] >= min_synapses)
    ].copy()
    
    # Get unique premotor INs
    premotor_in_ids = premotor_conn['source'].unique()
    
    print(f"  Total VNC interneurons: {len(vnc_in_ids):,}")
    print(f"  Wing motor neurons: {len(motor_neuron_ids)}")
    print(f"  Premotor INs found: {len(premotor_in_ids):,}")
    print(f"  IN→MN connections: {len(premotor_conn):,}")
    print(f"  (min {min_synapses} synapses)")
    
    # Get full IN data
    premotor_ins = vnc_interneurons[
        vnc_interneurons['Root ID'].isin(premotor_in_ids)
    ].copy()
    
    return premotor_ins, premotor_conn


def cluster_ins_by_mn_connectivity(
    premotor_conn: pd.DataFrame,
    premotor_in_ids: list,
    motor_pools: pd.DataFrame,
    method: str = 'ward'
):
    """
    Cluster INs by their connectivity to motor neuron pools.
    
    Uses hierarchical clustering on connectivity similarity.
    """
    
    print("\nClustering INs by MN pool connectivity...")
    print("-"*70)
    
    # Create IN → MN pool connectivity matrix
    # Rows = INs, Columns = Motor pools
    
    # Get MN pool mapping
    mn_to_pool = {}
    for _, pool in motor_pools.iterrows():
        for mn_id in pool['motor_neuron_ids']:
            mn_to_pool[mn_id] = pool['muscle']
    
    # Build connectivity matrix
    in_to_pool_conn = {}
    
    for _, row in premotor_conn.iterrows():
        in_id = row['source']
        mn_id = row['target']
        weight = row['weight']
        
        pool = mn_to_pool.get(mn_id, 'unknown')
        
        if in_id not in in_to_pool_conn:
            in_to_pool_conn[in_id] = {}
        
        if pool not in in_to_pool_conn[in_id]:
            in_to_pool_conn[in_id][pool] = 0
        
        in_to_pool_conn[in_id][pool] += weight
    
    # Convert to matrix
    pool_names = sorted(motor_pools['muscle'].unique())
    in_ids_ordered = sorted(premotor_in_ids)
    
    conn_matrix = np.zeros((len(in_ids_ordered), len(pool_names)))
    
    for i, in_id in enumerate(in_ids_ordered):
        for j, pool in enumerate(pool_names):
            conn_matrix[i, j] = in_to_pool_conn.get(in_id, {}).get(pool, 0)
    
    print(f"  Connectivity matrix: {conn_matrix.shape[0]} INs × {conn_matrix.shape[1]} motor pools")
    print(f"  Total connections: {np.sum(conn_matrix > 0)}")
    
    # Compute similarity between INs
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(conn_matrix)
    
    # Hierarchical clustering
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.maximum(distance_matrix, 0)
    
    condensed_dist = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method=method)
    
    print(f"  Clustering method: {method}")
    
    return {
        'connectivity_matrix': conn_matrix,
        'in_ids': in_ids_ordered,
        'pool_names': pool_names,
        'similarity_matrix': similarity_matrix,
        'linkage_matrix': linkage_matrix,
        'in_to_pool_conn': in_to_pool_conn
    }


def identify_in_clusters(
    clustering_results: dict,
    n_clusters: int = 5
):
    """
    Cut dendrogram to identify IN clusters.
    """
    
    print(f"\nIdentifying {n_clusters} IN clusters...")
    print("-"*70)
    
    clusters = fcluster(
        clustering_results['linkage_matrix'],
        n_clusters,
        criterion='maxclust'
    )
    
    # Assign to INs
    in_ids = clustering_results['in_ids']
    
    cluster_assignments = pd.DataFrame({
        'interneuron_id': in_ids,
        'cluster': clusters
    })
    
    # Characterize each cluster
    cluster_info = []
    
    conn_matrix = clustering_results['connectivity_matrix']
    pool_names = clustering_results['pool_names']
    
    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = clusters == cluster_id
        cluster_ins = np.array(in_ids)[cluster_mask]
        
        # Get connectivity profile for this cluster
        cluster_conn = conn_matrix[cluster_mask, :]
        avg_conn = np.mean(cluster_conn, axis=0)
        
        # Find primary target pools
        top_pools_idx = np.argsort(avg_conn)[::-1][:3]
        top_pools = [pool_names[i] for i in top_pools_idx if avg_conn[i] > 0]
        
        cluster_info.append({
            'cluster_id': cluster_id,
            'n_interneurons': len(cluster_ins),
            'primary_targets': ', '.join(top_pools) if top_pools else 'none',
            'total_connectivity': np.sum(cluster_conn)
        })
    
    cluster_info_df = pd.DataFrame(cluster_info)
    
    print("\nCluster characteristics:")
    for _, cluster in cluster_info_df.iterrows():
        print(f"  Cluster {cluster['cluster_id']}: {cluster['n_interneurons']:4d} INs → {cluster['primary_targets']}")
    
    return cluster_assignments, cluster_info_df


def classify_clusters_by_function(
    cluster_info: pd.DataFrame,
    motor_pools: pd.DataFrame
):
    """
    Classify clusters as power, steering, or mixed control.
    """
    
    print("\nClassifying clusters by function...")
    print("-"*70)
    
    # Get power and steering muscle lists
    power_muscles = set(motor_pools[motor_pools['muscle_type'] == 'power']['muscle'])
    steering_muscles = set(motor_pools[motor_pools['muscle_type'] == 'steering']['muscle'])
    
    cluster_functions = []
    
    for _, cluster in cluster_info.iterrows():
        targets = cluster['primary_targets'].split(', ')
        
        power_count = sum(1 for t in targets if t in power_muscles)
        steering_count = sum(1 for t in targets if t in steering_muscles)
        
        if power_count > steering_count:
            function = 'power_control'
        elif steering_count > power_count:
            function = 'steering_control'
        elif power_count > 0 and steering_count > 0:
            function = 'integrated_control'
        else:
            function = 'unknown'
        
        cluster_functions.append(function)
    
    cluster_info['functional_type'] = cluster_functions
    
    print("\nFunctional classification:")
    for _, cluster in cluster_info.iterrows():
        print(f"  Cluster {cluster['cluster_id']}: {cluster['functional_type']:20s} ({cluster['n_interneurons']} INs)")
    
    return cluster_info


def identify_hub_interneurons(
    premotor_conn: pd.DataFrame,
    premotor_in_ids: list,
    top_n: int = 20
):
    """
    Identify hub interneurons based on network centrality.
    """
    
    print("\nIdentifying hub interneurons...")
    print("-"*70)
    
    # Create network graph
    G = nx.DiGraph()
    
    for _, row in premotor_conn.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    
    # Compute centrality measures
    degree_centrality = nx.degree_centrality(G)
    
    try:
        betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    except:
        betweenness_centrality = {}
    
    # Get stats for INs only
    hub_stats = []
    
    for in_id in premotor_in_ids:
        if in_id in G:
            out_degree = G.out_degree(in_id, weight='weight')
            
            hub_stats.append({
                'interneuron_id': in_id,
                'out_degree': out_degree,
                'degree_centrality': degree_centrality.get(in_id, 0),
                'betweenness_centrality': betweenness_centrality.get(in_id, 0),
                'n_mn_targets': len(list(G.successors(in_id)))
            })
    
    hub_df = pd.DataFrame(hub_stats)
    
    # Rank by degree centrality
    hub_df = hub_df.sort_values('degree_centrality', ascending=False)
    
    top_hubs = hub_df.head(top_n)
    
    print(f"  Total premotor INs: {len(hub_df)}")
    print(f"\n  Top {top_n} hub interneurons:")
    print(f"  {'Rank':<6} {'IN ID':<20} {'Targets':<10} {'Degree Cent.':<15}")
    print("-"*70)
    
    for i, (_, hub) in enumerate(top_hubs.iterrows(), 1):
        print(f"  {i:<6} {hub['interneuron_id']:<20} {hub['n_mn_targets']:<10} {hub['degree_centrality']:<15.4f}")
    
    return hub_df


def create_clustering_visualizations(
    clustering_results: dict,
    cluster_assignments: pd.DataFrame,
    cluster_info: pd.DataFrame,
    motor_pools: pd.DataFrame,
    output_dir: Path
):
    """
    Create comprehensive visualizations.
    """
    
    print("\nCreating visualizations...")
    print("-"*70)
    
    # Figure 1: Dendrogram + Connectivity Matrix
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 1], hspace=0.3, wspace=0.3)
    
    # Dendrogram
    ax1 = fig.add_subplot(gs[0, :])
    dendrogram(clustering_results['linkage_matrix'], ax=ax1, no_labels=True)
    ax1.set_title('Hierarchical Clustering of Premotor Interneurons', 
                 fontweight='bold', fontsize=14)
    ax1.set_xlabel('Interneuron Index', fontweight='bold')
    ax1.set_ylabel('Distance', fontweight='bold')
    
    # Connectivity matrix heatmap
    ax2 = fig.add_subplot(gs[1, :])
    
    conn_matrix = clustering_results['connectivity_matrix']
    pool_names = clustering_results['pool_names']
    
    # Reorder by clusters
    clusters = cluster_assignments['cluster'].values
    cluster_order = np.argsort(clusters)
    
    conn_matrix_ordered = conn_matrix[cluster_order, :]
    
    im = ax2.imshow(conn_matrix_ordered, aspect='auto', cmap='viridis', 
                   interpolation='nearest')
    
    ax2.set_xlabel('Motor Pool', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Interneuron (ordered by cluster)', fontweight='bold', fontsize=12)
    ax2.set_title('IN → Motor Pool Connectivity Matrix', fontweight='bold', fontsize=14)
    
    # Set motor pool labels
    ax2.set_xticks(range(len(pool_names)))
    ax2.set_xticklabels(pool_names, rotation=90, fontsize=8)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Synapse Count', fontweight='bold', rotation=270, labelpad=20)
    
    # Cluster distribution
    ax3 = fig.add_subplot(gs[2, 0])
    cluster_counts = cluster_assignments['cluster'].value_counts().sort_index()
    cluster_counts.plot(kind='bar', ax=ax3, color='steelblue')
    ax3.set_xlabel('Cluster ID', fontweight='bold')
    ax3.set_ylabel('Number of INs', fontweight='bold')
    ax3.set_title('INs per Cluster', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Functional type distribution
    ax4 = fig.add_subplot(gs[2, 1])
    if 'functional_type' in cluster_info.columns:
        func_counts = cluster_info['functional_type'].value_counts()
        colors = {'power_control': '#FF6B6B', 
                 'steering_control': '#4ECDC4',
                 'integrated_control': '#FFA07A',
                 'unknown': '#CCCCCC'}
        bar_colors = [colors.get(x, 'gray') for x in func_counts.index]
        func_counts.plot(kind='bar', ax=ax4, color=bar_colors)
        ax4.set_xlabel('Functional Type', fontweight='bold')
        ax4.set_ylabel('Number of Clusters', fontweight='bold')
        ax4.set_title('Cluster Functional Classification', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.savefig(output_dir / 'in_clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: in_clustering_analysis.png")


def main():
    print("\n" + "="*70)
    print("INTERNEURON HUB CLUSTERING")
    print("Discovering premotor circuit organization")
    print("="*70 + "\n")
    
    # Load data
    loader = ConnectomeDataLoader('config.yaml')
    connections, neurons, name_mapping = loader.load_all_data(verbose=False)
    
    # Load previous results
    print("Loading previous results...")
    print("-"*70 + "\n")
    
    # VNC interneurons
    vnc_ins = pd.read_csv('results/cell_types/vnc_interneurons.csv')
    print(f"  ✓ VNC interneurons: {len(vnc_ins):,}")
    
    # Motor pools
    motor_pools = pd.read_csv('results/motor_neurons/motor_pools.csv')
    motor_pools['motor_neuron_ids'] = motor_pools['motor_neuron_ids'].apply(eval)
    print(f"  ✓ Motor pools: {len(motor_pools)}")
    
    # Get all MN IDs
    all_mn_ids = []
    for ids in motor_pools['motor_neuron_ids']:
        all_mn_ids.extend(ids)
    print(f"  ✓ Total wing MNs: {len(all_mn_ids)}\n")
    
    # Filter connections
    filtered_conn = loader.filter_connections(min_synapses=3, verbose=False)
    
    # Step 1: Find premotor INs
    premotor_ins, premotor_conn = find_premotor_interneurons(
        filtered_conn,
        all_mn_ids,
        vnc_ins,
        min_synapses=3
    )
    
    if len(premotor_ins) == 0:
        print("\n❌ No premotor interneurons found!")
        sys.exit(1)
    
    # Step 2: Cluster INs
    clustering_results = cluster_ins_by_mn_connectivity(
        premotor_conn,
        premotor_ins['Root ID'].tolist(),
        motor_pools,
        method='ward'
    )
    
    # Step 3: Identify clusters
    cluster_assignments, cluster_info = identify_in_clusters(
        clustering_results,
        n_clusters=5
    )
    
    # Step 4: Classify clusters functionally
    cluster_info = classify_clusters_by_function(cluster_info, motor_pools)
    
    # Step 5: Find hub neurons
    hub_neurons = identify_hub_interneurons(
        premotor_conn,
        premotor_ins['Root ID'].tolist(),
        top_n=20
    )
    
    # Save results
    print("\n" + "-"*70)
    print("Saving results")
    print("-"*70 + "\n")
    
    output_dir = Path('results/interneuron_clusters')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge cluster assignments with IN data
    premotor_ins_clustered = premotor_ins.merge(
        cluster_assignments,
        left_on='Root ID',
        right_on='interneuron_id',
        how='left'
    )
    
    premotor_ins_clustered.to_csv(output_dir / 'premotor_interneurons_clustered.csv', index=False)
    print(f"  ✓ premotor_interneurons_clustered.csv ({len(premotor_ins_clustered)} INs)")
    
    cluster_info.to_csv(output_dir / 'cluster_characteristics.csv', index=False)
    print(f"  ✓ cluster_characteristics.csv")
    
    hub_neurons.to_csv(output_dir / 'hub_interneurons.csv', index=False)
    print(f"  ✓ hub_interneurons.csv")
    
    premotor_conn.to_csv(output_dir / 'in_to_mn_connections.csv', index=False)
    print(f"  ✓ in_to_mn_connections.csv")
    
    # Visualizations
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    create_clustering_visualizations(
        clustering_results,
        cluster_assignments,
        cluster_info,
        motor_pools,
        fig_dir
    )
    
    # Summary
    print("\n" + "="*70)
    print("INTERNEURON CLUSTERING COMPLETE")
    print("="*70 + "\n")
    
    print(f"Premotor circuit discovered:")
    print(f"  Premotor interneurons:  {len(premotor_ins):4,}")
    print(f"  IN→MN connections:      {len(premotor_conn):4,}")
    print(f"  IN clusters identified: {len(cluster_info):4}")
    
    print(f"\nCluster functions:")
    for _, cluster in cluster_info.iterrows():
        print(f"  Cluster {cluster['cluster_id']}: {cluster['functional_type']:20s} ({cluster['n_interneurons']:4} INs)")
    
    print(f"\nHub interneurons: {len(hub_neurons[hub_neurons['degree_centrality'] > hub_neurons['degree_centrality'].median()]):3} high-centrality nodes")
    
    print(f"\nResults: {output_dir}")
    print("\nNext step: Run dn_pathway_mapping.py")
    print("  Map DN → IN → MN pathways!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()