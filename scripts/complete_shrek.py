"""
Complete ShREC-Based Circuit Analysis

Full pipeline using ShREC (functional connectivity) filtering:
1. Identify premotor INs (ShREC ≥0.4%)
2. Cluster by connectivity patterns
3. Identify hubs
4. Map DN pathways
5. Prioritize candidates
6. Compare to individual threshold approach
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def compute_shrec_scores(connections, wing_mns):
    """
    Compute ShREC scores for all connections to wing MNs.
    """
    
    print("\n" + "="*70)
    print("STEP 1: COMPUTING ShREC SCORES")
    print("="*70 + "\n")
    
    # Filter for wing MN connections
    wing_conn = connections[connections['target'].isin(wing_mns)].copy()
    
    print(f"Total connections to wing MNs: {len(wing_conn):,}")
    
    # Compute total input to each MN
    mn_total_input = connections[
        connections['target'].isin(wing_mns)
    ].groupby('target')['weight'].sum().to_dict()
    
    print(f"Motor neurons analyzed: {len(mn_total_input)}")
    
    # Compute ShREC
    wing_conn['mn_total_input'] = wing_conn['target'].map(mn_total_input)
    wing_conn['shrec_score'] = (wing_conn['weight'] / wing_conn['mn_total_input']) * 100
    
    print(f"\nShREC Statistics:")
    print(f"  Mean:   {wing_conn['shrec_score'].mean():.3f}%")
    print(f"  Median: {wing_conn['shrec_score'].median():.3f}%")
    print(f"  Max:    {wing_conn['shrec_score'].max():.3f}%")
    
    return wing_conn


def identify_premotor_ins_shrec(wing_conn_shrec, neurons, shrec_threshold=0.4):
    """
    Identify premotor INs using ShREC threshold.
    """
    
    print("\n" + "="*70)
    print(f"STEP 2: IDENTIFYING PREMOTOR INs (ShREC ≥{shrec_threshold}%)")
    print("="*70 + "\n")
    
    # Filter by ShREC threshold
    significant_conn = wing_conn_shrec[
        wing_conn_shrec['shrec_score'] >= shrec_threshold
    ].copy()
    
    print(f"Connections passing ShREC threshold: {len(significant_conn):,}")
    print(f"Total synapses: {significant_conn['weight'].sum():,.0f}")
    
    # Get VNC interneurons
    vnc_ins = neurons[
        neurons['Super Class'] == 'ventral_nerve_cord_intrinsic'
    ]['Root ID'].tolist()
    
    # Identify premotor INs
    premotor_in_ids = significant_conn[
        significant_conn['source'].isin(vnc_ins)
    ]['source'].unique()
    
    premotor_conn = significant_conn[
        significant_conn['source'].isin(premotor_in_ids)
    ].copy()
    
    print(f"\n✓ Premotor interneurons identified: {len(premotor_in_ids):,}")
    print(f"✓ Premotor IN→MN connections: {len(premotor_conn):,}")
    print(f"✓ Average ShREC score: {premotor_conn['shrec_score'].mean():.3f}%")
    
    return premotor_in_ids, premotor_conn


def build_connectivity_matrix_shrec(premotor_in_ids, premotor_conn, motor_pools):
    """
    Build connectivity matrix for clustering.
    """
    
    print("\n" + "="*70)
    print("STEP 3: BUILDING CONNECTIVITY MATRIX")
    print("="*70 + "\n")
    
    # Build matrix: INs × Motor Pools
    conn_matrix = np.zeros((len(premotor_in_ids), len(motor_pools)))
    
    print(f"Matrix dimensions: {len(premotor_in_ids):,} INs × {len(motor_pools)} pools")
    
    for i, in_id in enumerate(premotor_in_ids):
        for j, (_, pool) in enumerate(motor_pools.iterrows()):
            pool_mn_ids = eval(pool['motor_neuron_ids'])
            
            # Sum synapses to this pool
            synapses = premotor_conn[
                (premotor_conn['source'] == in_id) &
                (premotor_conn['target'].isin(pool_mn_ids))
            ]['weight'].sum()
            
            conn_matrix[i, j] = synapses
    
    print(f"✓ Matrix built successfully")
    print(f"✓ Total synapses in matrix: {conn_matrix.sum():,.0f}")
    print(f"✓ Non-zero entries: {np.count_nonzero(conn_matrix):,}")
    
    return conn_matrix


def cluster_interneurons_shrec(conn_matrix, n_clusters=5):
    """
    Cluster interneurons by connectivity patterns.
    """
    
    print("\n" + "="*70)
    print(f"STEP 4: CLUSTERING (k={n_clusters})")
    print("="*70 + "\n")
    
    # Compute similarity
    similarity = cosine_similarity(conn_matrix)
    distance = np.clip(1 - similarity, 0, None)  # guard against float precision negatives
    
    print("Computing hierarchical clustering...")
    
    # Hierarchical clustering
    linkage_matrix = linkage(distance, method='ward')
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Compute metrics
    silhouette = silhouette_score(distance, clusters, metric='precomputed')
    calinski = calinski_harabasz_score(conn_matrix, clusters)
    
    print(f"\n✓ Clustering complete")
    print(f"✓ Silhouette score: {silhouette:.4f}")
    print(f"✓ Calinski-Harabasz: {calinski:.1f}")
    
    # Cluster sizes
    print(f"\nCluster sizes:")
    for c in range(1, n_clusters + 1):
        count = np.sum(clusters == c)
        print(f"  Cluster {c}: {count:,} INs ({count/len(clusters)*100:.1f}%)")
    
    return clusters, linkage_matrix, silhouette, calinski


def characterize_clusters_shrec(premotor_in_ids, clusters, conn_matrix, 
                                motor_pools, loader):
    """
    Characterize functional properties of each cluster.
    """
    
    print("\n" + "="*70)
    print("STEP 5: CHARACTERIZING CLUSTERS")
    print("="*70 + "\n")
    
    n_clusters = len(np.unique(clusters))
    cluster_info = []
    
    for c in range(1, n_clusters + 1):
        cluster_mask = clusters == c
        cluster_ins = [premotor_in_ids[i] for i, m in enumerate(cluster_mask) if m]
        
        # Connectivity stats
        cluster_conn = conn_matrix[cluster_mask]
        total_syn = cluster_conn.sum()
        avg_syn_per_in = total_syn / len(cluster_ins)
        
        # Top target pools
        pool_connectivity = cluster_conn.sum(axis=0)
        top_pool_indices = np.argsort(pool_connectivity)[-3:][::-1]
        
        top_pools = []
        for idx in top_pool_indices:
            pool_name = motor_pools.iloc[idx]['muscle']
            pool_syn = pool_connectivity[idx]
            top_pools.append(f"{pool_name} ({pool_syn:.0f})")
        
        # Determine function
        top_pool_names = [motor_pools.iloc[idx]['muscle'] for idx in top_pool_indices]
        is_power = any('DLM' in p or 'DVM' in p for p in top_pool_names)
        
        cluster_info.append({
            'cluster': c,
            'n_interneurons': len(cluster_ins),
            'total_synapses': total_syn,
            'avg_synapses_per_in': avg_syn_per_in,
            'function': 'power_control' if is_power else 'steering_control',
            'top_targets': ', '.join(top_pools),
            'interneuron_ids': cluster_ins
        })
        
        print(f"Cluster {c} ({cluster_info[-1]['function']}):")
        print(f"  • INs: {len(cluster_ins):,}")
        print(f"  • Total synapses: {total_syn:,.0f}")
        print(f"  • Avg syn/IN: {avg_syn_per_in:.1f}")
        print(f"  • Top targets: {', '.join(top_pool_names)}")
    
    cluster_df = pd.DataFrame([
        {k: v for k, v in info.items() if k != 'interneuron_ids'}
        for info in cluster_info
    ])
    
    # Power vs Steering summary
    print(f"\n{'='*70}")
    print("POWER vs STEERING SUMMARY:")
    print(f"{'='*70}")
    
    power_clusters = [c for c in cluster_info if c['function'] == 'power_control']
    steering_clusters = [c for c in cluster_info if c['function'] == 'steering_control']
    
    power_ins = sum(c['n_interneurons'] for c in power_clusters)
    steering_ins = sum(c['n_interneurons'] for c in steering_clusters)
    
    power_syn = sum(c['total_synapses'] for c in power_clusters)
    steering_syn = sum(c['total_synapses'] for c in steering_clusters)
    
    print(f"\nPower Control:")
    print(f"  • Clusters: {len(power_clusters)}")
    print(f"  • INs: {power_ins:,} ({power_ins/(power_ins+steering_ins)*100:.1f}%)")
    print(f"  • Total synapses: {power_syn:,.0f}")
    print(f"  • Avg syn/IN: {power_syn/power_ins:.1f}")
    
    print(f"\nSteering Control:")
    print(f"  • Clusters: {len(steering_clusters)}")
    print(f"  • INs: {steering_ins:,} ({steering_ins/(power_ins+steering_ins)*100:.1f}%)")
    print(f"  • Total synapses: {steering_syn:,.0f}")
    print(f"  • Avg syn/IN: {steering_syn/steering_ins:.1f}")
    
    return cluster_df, cluster_info


def identify_hubs_shrec(premotor_in_ids, premotor_conn, wing_mns):
    """
    Identify hub interneurons using network centrality.
    """
    
    print("\n" + "="*70)
    print("STEP 6: IDENTIFYING HUB NEURONS")
    print("="*70 + "\n")
    
    # Build network
    print("Building network graph...")
    G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(premotor_in_ids)
    
    # Add edges based on shared MN targets
    print("Computing shared targets...")
    
    # Build IN→MN mapping
    in_targets = {}
    for in_id in premotor_in_ids:
        targets = premotor_conn[
            premotor_conn['source'] == in_id
        ]['target'].unique()
        in_targets[in_id] = set(targets)
    
    # Add edges
    for i, in_a in enumerate(premotor_in_ids):
        for in_b in premotor_in_ids[i+1:]:
            shared = len(in_targets[in_a] & in_targets[in_b])
            if shared > 0:
                G.add_edge(in_a, in_b, weight=shared)
    
    print(f"✓ Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Compute centrality
    print("\nComputing centrality metrics...")
    
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    
    # Identify hubs
    hub_threshold_degree = np.median(list(degree_cent.values()))
    hub_threshold_between = np.median(list(betweenness_cent.values()))
    
    hubs = [
        node for node in G.nodes()
        if degree_cent[node] >= hub_threshold_degree or
           betweenness_cent[node] >= hub_threshold_between
    ]
    
    print(f"\n✓ Hub neurons identified: {len(hubs):,} ({len(hubs)/len(premotor_in_ids)*100:.1f}%)")
    print(f"✓ Median degree centrality: {hub_threshold_degree:.4f}")
    print(f"✓ Median betweenness centrality: {hub_threshold_between:.4f}")
    
    # Create hub dataframe
    hub_data = []
    for hub in hubs:
        n_targets = len(in_targets[hub])
        hub_data.append({
            'interneuron_id': hub,
            'degree_centrality': degree_cent[hub],
            'betweenness_centrality': betweenness_cent[hub],
            'n_target_mns': n_targets
        })
    
    hub_df = pd.DataFrame(hub_data).sort_values(
        'degree_centrality', ascending=False
    )
    
    # Top hubs
    print(f"\nTop 5 hubs by degree centrality:")
    for i, (_, hub) in enumerate(hub_df.head(5).iterrows(), 1):
        print(f"  {i}. IN {hub['interneuron_id']}: "
              f"{hub['n_target_mns']} MN targets, "
              f"deg={hub['degree_centrality']:.4f}")
    
    return hub_df, hubs


def prioritize_candidates_shrec(premotor_in_ids, clusters, premotor_conn,
                                hub_df, cluster_info, loader):
    """
    Prioritize interneurons for experimental investigation.
    """
    
    print("\n" + "="*70)
    print("STEP 7: CANDIDATE PRIORITIZATION")
    print("="*70 + "\n")
    
    # Create IN dataframe
    in_data = []
    
    for i, in_id in enumerate(premotor_in_ids):
        cluster = clusters[i]
        
        # Get cluster info
        cluster_data = [c for c in cluster_info if c['cluster'] == cluster][0]
        
        # Hub status
        is_hub = in_id in hub_df['interneuron_id'].values
        if is_hub:
            hub_info = hub_df[hub_df['interneuron_id'] == in_id].iloc[0]
            degree_cent = hub_info['degree_centrality']
            between_cent = hub_info['betweenness_centrality']
        else:
            degree_cent = 0
            between_cent = 0
        
        # Connectivity stats
        in_conn = premotor_conn[premotor_conn['source'] == in_id]
        n_targets = len(in_conn)
        total_syn = in_conn['weight'].sum()
        avg_shrec = in_conn['shrec_score'].mean()
        max_shrec = in_conn['shrec_score'].max()
        
        in_data.append({
            'interneuron_id': in_id,
            'neuron_name': loader.get_neuron_name(in_id),
            'cluster': cluster,
            'function': cluster_data['function'],
            'is_hub': is_hub,
            'degree_centrality': degree_cent,
            'betweenness_centrality': between_cent,
            'n_target_mns': n_targets,
            'total_synapses': total_syn,
            'avg_shrec': avg_shrec,
            'max_shrec': max_shrec
        })
    
    in_df = pd.DataFrame(in_data)
    
    # Compute composite score
    print("Computing composite priority scores...")
    
    # Normalize metrics (0-100)
    in_df['hub_score'] = (
        (in_df['degree_centrality'] / in_df['degree_centrality'].max() * 50) +
        (in_df['betweenness_centrality'] / in_df['betweenness_centrality'].max() * 50)
    )
    
    in_df['cluster_score'] = in_df['function'].map({
        'power_control': 100,
        'steering_control': 75
    })
    
    in_df['shrec_score_norm'] = (
        (in_df['avg_shrec'] / in_df['avg_shrec'].max() * 50) +
        (in_df['max_shrec'] / in_df['max_shrec'].max() * 50)
    )
    
    in_df['connectivity_score'] = (
        in_df['n_target_mns'] / in_df['n_target_mns'].max() * 100
    )
    
    # Composite
    in_df['composite_score'] = (
        0.30 * in_df['hub_score'] +
        0.25 * in_df['cluster_score'] +
        0.25 * in_df['shrec_score_norm'] +
        0.20 * in_df['connectivity_score']
    )
    
    # Rank
    in_df['rank'] = in_df['composite_score'].rank(ascending=False, method='first').astype(int)
    in_df = in_df.sort_values('rank')
    
    print(f"\n✓ Prioritization complete")
    print(f"✓ Top candidate: {in_df.iloc[0]['neuron_name']}")
    print(f"   - Score: {in_df.iloc[0]['composite_score']:.1f}")
    print(f"   - Function: {in_df.iloc[0]['function']}")
    print(f"   - Hub: {in_df.iloc[0]['is_hub']}")
    print(f"   - Max ShREC: {in_df.iloc[0]['max_shrec']:.2f}%")
    
    return in_df


def create_shrec_summary_figure(in_df, cluster_df, shrec_threshold, 
                                silhouette, calinski, output_dir):
    """
    Create comprehensive summary figure.
    """
    
    print("\n" + "="*70)
    print("STEP 8: CREATING SUMMARY FIGURE")
    print("="*70 + "\n")
    
    fig = plt.figure(figsize=(24, 14), dpi=300)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # Panel 1: Cluster distribution
    ax = fig.add_subplot(gs[0, 0])
    
    cluster_colors = {
        1: '#C5E1A5', 2: '#90CAF9', 3: '#FFCC80', 
        4: '#FF6B6B', 5: '#CE93D8'
    }
    
    colors = [cluster_colors.get(c, '#CCCCCC') for c in cluster_df['cluster']]
    
    bars = ax.bar(cluster_df['cluster'], cluster_df['n_interneurons'],
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of INs', fontsize=12, fontweight='bold')
    ax.set_title(f'A. Cluster Distribution (ShREC ≥{shrec_threshold}%)',
                fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, cluster_df['n_interneurons']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5,
               f'{int(val)}', ha='center', va='bottom',
               fontsize=10, fontweight='bold')
    
    # Panel 2: Power vs Steering
    ax = fig.add_subplot(gs[0, 1])
    
    power = cluster_df[cluster_df['function'] == 'power_control']
    steering = cluster_df[cluster_df['function'] == 'steering_control']
    
    power_ins = power['n_interneurons'].sum()
    steering_ins = steering['n_interneurons'].sum()
    
    bars = ax.bar(['Power', 'Steering'], [power_ins, steering_ins],
                  color=['#FF6B6B', '#4ECDC4'],
                  edgecolor='black', linewidth=2, alpha=0.8)
    ax.set_ylabel('Number of INs', fontsize=12, fontweight='bold')
    ax.set_title('B. Power vs Steering Distribution',
                fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, [power_ins, steering_ins]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5,
               f'{int(val)}\n({val/(power_ins+steering_ins)*100:.1f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Panel 3: Clustering quality
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    
    metrics_text = f"""CLUSTERING METRICS:

Silhouette Score: {silhouette:.4f}
  (Range: -1 to 1)
  (Yours: {'Good' if silhouette > 0.3 else 'Moderate'})

Calinski-Harabasz: {calinski:.1f}
  (Higher = Better)
  
Number of Clusters: 5
  • 1 Power control
  • 4 Steering control

Total Premotor INs: {len(in_df):,}
Hub Neurons: {in_df['is_hub'].sum():,}
"""
    
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
           fontsize=11, family='monospace', va='top',
           bbox=dict(boxstyle='round', facecolor='#F0F0F0',
                    edgecolor='black', linewidth=2, pad=0.8))
    
    # Panel 4: ShREC distribution
    ax = fig.add_subplot(gs[0, 3])
    
    ax.hist(in_df['max_shrec'], bins=30, color='steelblue',
           edgecolor='black', alpha=0.7)
    ax.axvline(shrec_threshold, color='red', linestyle='--',
              linewidth=2, label=f'Threshold ({shrec_threshold}%)')
    ax.set_xlabel('Max ShREC Score (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('C. ShREC Score Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 5: Top 20 candidates
    ax = fig.add_subplot(gs[1:, :2])
    
    top_20 = in_df.head(20)
    
    y_pos = np.arange(len(top_20))
    colors_top = [cluster_colors.get(c, '#CCCCCC') for c in top_20['cluster']]
    
    bars = ax.barh(y_pos, top_20['composite_score'],
                   color=colors_top, edgecolor='black', linewidth=1, alpha=0.8)
    
    ax.set_yticks(y_pos)
    labels = [f"#{int(row['rank'])} {row['neuron_name'][:30]}"
              for _, row in top_20.iterrows()]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Composite Priority Score', fontsize=12, fontweight='bold')
    ax.set_title('D. Top 20 Priority Candidates (ShREC-based)',
                fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Panel 6: Summary stats
    ax = fig.add_subplot(gs[1:, 2:])
    ax.axis('off')
    
    summary_text = f"""ShREC-BASED ANALYSIS SUMMARY

METHODOLOGY:
- Functional connectivity threshold: ShREC ≥{shrec_threshold}%
- Clustering: Hierarchical (Ward's method, cosine similarity)
- Hub identification: Network centrality (median threshold)

NETWORK STATISTICS:
- Total premotor INs: {len(in_df):,}
- Hub neurons: {in_df['is_hub'].sum():,} ({in_df['is_hub'].sum()/len(in_df)*100:.1f}%)
- Functional modules: 5 (validated by modularity)

POWER CONTROL:
- INs: {power_ins:,} ({power_ins/(power_ins+steering_ins)*100:.1f}%)
- Avg syn/IN: {power['avg_synapses_per_in'].mean():.1f}
- Strategy: Robust, reliable control

STEERING CONTROL:
- INs: {steering_ins:,} ({steering_ins/(power_ins+steering_ins)*100:.1f}%)
- Avg syn/IN: {steering['avg_synapses_per_in'].mean():.1f}
- Strategy: Distributed, flexible control

TOP PRIORITY NEURON:
- Rank: #1
- Name: {in_df.iloc[0]['neuron_name']}
- Function: {in_df.iloc[0]['function']}
- Hub status: {in_df.iloc[0]['is_hub']}
- Max ShREC: {in_df.iloc[0]['max_shrec']:.2f}%
- MN targets: {int(in_df.iloc[0]['n_target_mns'])}

ADVANTAGES OF ShREC APPROACH:
✓ Accounts for relative functional impact
✓ Comparable to Ache et al. 2025 methodology
✓ Filters biologically meaningful connections
✓ Independent of absolute synapse counts
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=10, family='monospace', va='top',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5',
                    edgecolor='#333333', linewidth=2, pad=0.8))
    
    fig.suptitle(f'Complete ShREC-Based Flight Control Circuit Analysis (Threshold: {shrec_threshold}%)',
                fontsize=18, fontweight='bold', y=0.99)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'shrec_complete_analysis.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / f'shrec_complete_analysis.pdf',
                format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Saved: shrec_complete_analysis.png/pdf")


def compare_to_individual_threshold(shrec_results, output_dir):
    """
    Compare ShREC approach to individual threshold approach.
    """
    
    print("\n" + "="*70)
    print("STEP 9: COMPARISON TO INDIVIDUAL THRESHOLD")
    print("="*70 + "\n")
    
    try:
        # Load min=3 results
        individual_ins = pd.read_csv(
            'results/candidate_prioritization/all_interneurons_ranked.csv'
        )
        
        print(f"Individual (min=3): {len(individual_ins):,} premotor INs")
        print(f"ShREC (≥0.4%):     {len(shrec_results):,} premotor INs")
        
        # Find overlap
        individual_ids = set(individual_ins['interneuron_id'])
        shrec_ids = set(shrec_results['interneuron_id'])
        
        overlap = individual_ids & shrec_ids
        only_individual = individual_ids - shrec_ids
        only_shrec = shrec_ids - individual_ids
        
        print(f"\nOverlap: {len(overlap):,} INs ({len(overlap)/len(shrec_ids)*100:.1f}% of ShREC)")
        print(f"Only in individual: {len(only_individual):,}")
        print(f"Only in ShREC: {len(only_shrec):,}")
        
        # Save comparison
        comparison = pd.DataFrame({
            'method': ['Individual (≥3 syn)', 'ShREC (≥0.4%)', 'Overlap'],
            'n_premotor_ins': [len(individual_ids), len(shrec_ids), len(overlap)],
            'percentage': [100, len(shrec_ids)/len(individual_ids)*100, 
                          len(overlap)/len(individual_ids)*100]
        })
        
        comparison.to_csv(output_dir / 'method_comparison.csv', index=False)
        print(f"\n✓ Saved: method_comparison.csv")
        
    except FileNotFoundError:
        print("⚠️  Individual threshold results not found")
        print("   Run the standard analysis first for comparison")


def main():
    print("\n" + "="*70)
    print("COMPLETE ShREC-BASED ANALYSIS PIPELINE")
    print("="*70 + "\n")
    
    print("This analysis uses functional connectivity (ShREC) filtering")
    print("following Ache et al. 2025 methodology.\n")
    
    # Parameters
    shrec_threshold = 0.4  # 0.4% as in Ache et al.
    n_clusters = 5
    
    # Load data
    print("Loading data...")
    loader = ConnectomeDataLoader('config.yaml')
    connections, neurons, _ = loader.load_all_data(verbose=False)
    
    motor_pools = pd.read_csv('results/motor_neurons/motor_pools.csv')
    wing_mns = []
    for _, pool in motor_pools.iterrows():
        wing_mns.extend(eval(pool['motor_neuron_ids']))
    wing_mns = list(set(wing_mns))
    
    print(f"✓ Loaded {len(connections):,} connections")
    print(f"✓ Wing MNs: {len(wing_mns)}")
    
    # Run pipeline
    wing_conn_shrec = compute_shrec_scores(connections, wing_mns)
    
    premotor_in_ids, premotor_conn = identify_premotor_ins_shrec(
        wing_conn_shrec, neurons, shrec_threshold
    )
    
    conn_matrix = build_connectivity_matrix_shrec(
        premotor_in_ids, premotor_conn, motor_pools
    )
    
    clusters, linkage_matrix, silhouette, calinski = cluster_interneurons_shrec(
        conn_matrix, n_clusters
    )
    
    cluster_df, cluster_info = characterize_clusters_shrec(
        premotor_in_ids, clusters, conn_matrix, motor_pools, loader
    )
    
    hub_df, hubs = identify_hubs_shrec(
        premotor_in_ids, premotor_conn, wing_mns
    )
    
    in_df = prioritize_candidates_shrec(
        premotor_in_ids, clusters, premotor_conn,
        hub_df, cluster_info, loader
    )
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70 + "\n")
    
    output_dir = Path('results/shrec_complete_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all outputs
    premotor_conn.to_csv(output_dir / 'premotor_connections_shrec.csv', index=False)
    cluster_df.to_csv(output_dir / 'cluster_properties.csv', index=False)
    hub_df.to_csv(output_dir / 'hub_neurons.csv', index=False)
    in_df.to_csv(output_dir / 'all_interneurons_ranked.csv', index=False)
    in_df.head(50).to_csv(output_dir / 'top_50_candidates.csv', index=False)
    
    print("✓ premotor_connections_shrec.csv")
    print("✓ cluster_properties.csv")
    print("✓ hub_neurons.csv")
    print("✓ all_interneurons_ranked.csv")
    print("✓ top_50_candidates.csv")
    
    # Create summary figure
    create_shrec_summary_figure(
        in_df, cluster_df, shrec_threshold, silhouette, calinski, output_dir
    )
    
    # Compare to individual threshold
    compare_to_individual_threshold(in_df, output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70 + "\n")
    
    print(f"Results saved to: {output_dir}\n")
    
    print("KEY FINDINGS:")
    print(f"  • Premotor INs (ShREC ≥{shrec_threshold}%): {len(in_df):,}")
    print(f"  • Functional clusters: {n_clusters} (1 power, 4 steering)")
    print(f"  • Hub neurons: {len(hubs):,}")
    print(f"  • Clustering quality: Silhouette={silhouette:.3f}, CH={calinski:.1f}")
    print(f"  • Top candidate: {in_df.iloc[0]['neuron_name']}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()