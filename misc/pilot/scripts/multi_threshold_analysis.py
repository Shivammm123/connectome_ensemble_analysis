"""
Multi-Threshold Analysis Pipeline

Runs COMPLETE analysis at multiple min_synapses thresholds:
- 15, 20, 25, 30, 40, 50 synapses

For each threshold:
1. Identify premotor INs
2. Build connectivity matrix
3. Cluster (k=5)
4. Identify hubs
5. Map DN pathways
6. Prioritize candidates

Then creates comprehensive comparison figures and tables.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def identify_premotor_ins(connections, wing_mns, vnc_ins, min_synapses):
    """Identify premotor INs using synapse threshold."""
    
    # Filter for wing MN connections
    wing_conn = connections[connections['target'].isin(wing_mns)].copy()
    
    # Filter by synapse threshold
    wing_conn = wing_conn[wing_conn['weight'] >= min_synapses]
    
    # Get premotor INs
    premotor_in_ids = wing_conn[
        wing_conn['source'].isin(vnc_ins)
    ]['source'].unique()
    
    premotor_conn = wing_conn[
        wing_conn['source'].isin(premotor_in_ids)
    ].copy()
    
    return premotor_in_ids, premotor_conn


def cluster_interneurons(premotor_in_ids, premotor_conn, motor_pools, n_clusters=5):
    """Cluster interneurons by connectivity patterns."""
    
    # Build connectivity matrix
    conn_matrix = np.zeros((len(premotor_in_ids), len(motor_pools)))
    
    for i, in_id in enumerate(premotor_in_ids):
        for j, (_, pool) in enumerate(motor_pools.iterrows()):
            pool_mn_ids = eval(pool['motor_neuron_ids'])
            synapses = premotor_conn[
                (premotor_conn['source'] == in_id) &
                (premotor_conn['target'].isin(pool_mn_ids))
            ]['weight'].sum()
            conn_matrix[i, j] = synapses
    
    # Clustering
    similarity = cosine_similarity(conn_matrix)
    distance = np.clip(1 - similarity, 0, None)  # avoid tiny negative floats from fp error

    linkage_matrix = linkage(distance, method='ward')
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Metrics
    silhouette = silhouette_score(distance, clusters, metric='precomputed')
    calinski = calinski_harabasz_score(conn_matrix, clusters)
    
    # Characterize clusters
    cluster_info = []
    
    for c in range(1, n_clusters + 1):
        cluster_mask = clusters == c
        cluster_ins = [premotor_in_ids[i] for i, m in enumerate(cluster_mask) if m]
        
        cluster_conn = conn_matrix[cluster_mask]
        total_syn = cluster_conn.sum()
        
        # Top targets
        pool_connectivity = cluster_conn.sum(axis=0)
        top_pool_idx = np.argsort(pool_connectivity)[-3:][::-1]
        top_pool_names = [motor_pools.iloc[idx]['muscle'] for idx in top_pool_idx]
        
        is_power = any('DLM' in p or 'DVM' in p for p in top_pool_names)
        
        cluster_info.append({
            'cluster': c,
            'n_interneurons': len(cluster_ins),
            'total_synapses': total_syn,
            'function': 'power_control' if is_power else 'steering_control',
            'top_targets': ', '.join(top_pool_names),
            'interneuron_ids': cluster_ins
        })
    
    return clusters, cluster_info, silhouette, calinski, conn_matrix


def identify_hubs(premotor_in_ids, premotor_conn):
    """Identify hub neurons using network centrality."""
    
    # Build network
    G = nx.Graph()
    G.add_nodes_from(premotor_in_ids)
    
    # Build IN→MN mapping
    in_targets = {}
    for in_id in premotor_in_ids:
        targets = premotor_conn[
            premotor_conn['source'] == in_id
        ]['target'].unique()
        in_targets[in_id] = set(targets)
    
    # Add edges based on shared targets
    for i, in_a in enumerate(premotor_in_ids):
        for in_b in premotor_in_ids[i+1:]:
            shared = len(in_targets[in_a] & in_targets[in_b])
            if shared > 0:
                G.add_edge(in_a, in_b, weight=shared)
    
    # Compute centrality
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
    
    return hubs, len(hubs)


def analyze_dn_pathways(connections, premotor_in_ids, dns, min_synapses):
    """Analyze DN→IN connections."""
    
    # Filter for DN→IN connections
    dn_in_conn = connections[
        (connections['source'].isin(dns)) &
        (connections['target'].isin(premotor_in_ids)) &
        (connections['weight'] >= min_synapses)
    ].copy()
    
    n_dns = len(dn_in_conn['source'].unique())
    n_connections = len(dn_in_conn)
    total_synapses = dn_in_conn['weight'].sum()
    
    return n_dns, n_connections, total_synapses


def run_single_threshold(connections, neurons, motor_pools, wing_mns, 
                         min_synapses, output_base_dir):
    """Run complete analysis for a single threshold."""
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: min_synapses = {min_synapses}")
    print(f"{'='*70}\n")
    
    # Get neuron types
    vnc_ins = neurons[
        neurons['Super Class'] == 'ventral_nerve_cord_intrinsic'
    ]['Root ID'].tolist()
    
    dns = neurons[
        neurons['Super Class'] == 'descending'
    ]['Root ID'].tolist()
    
    # Step 1: Identify premotor INs
    print(f"Step 1: Identifying premotor INs (≥{min_synapses} syn)...")
    premotor_in_ids, premotor_conn = identify_premotor_ins(
        connections, wing_mns, vnc_ins, min_synapses
    )
    print(f"  ✓ Premotor INs: {len(premotor_in_ids):,}")
    print(f"  ✓ Connections: {len(premotor_conn):,}")
    print(f"  ✓ Total synapses: {premotor_conn['weight'].sum():,.0f}")
    
    # Step 2: Cluster
    print(f"\nStep 2: Clustering (k=5)...")
    clusters, cluster_info, silhouette, calinski, conn_matrix = cluster_interneurons(
        premotor_in_ids, premotor_conn, motor_pools, n_clusters=5
    )
    print(f"  ✓ Silhouette: {silhouette:.4f}")
    print(f"  ✓ Calinski-Harabasz: {calinski:.1f}")
    
    # Step 3: Identify hubs
    print(f"\nStep 3: Identifying hubs...")
    hubs, n_hubs = identify_hubs(premotor_in_ids, premotor_conn)
    print(f"  ✓ Hub neurons: {n_hubs} ({n_hubs/len(premotor_in_ids)*100:.1f}%)")
    
    # Step 4: DN pathways
    print(f"\nStep 4: Analyzing DN pathways (≥{min_synapses} syn)...")
    n_dns, dn_connections, dn_synapses = analyze_dn_pathways(
        connections, premotor_in_ids, dns, min_synapses
    )
    print(f"  ✓ DNs: {n_dns}")
    print(f"  ✓ DN→IN connections: {dn_connections:,}")
    print(f"  ✓ Total synapses: {dn_synapses:,.0f}")
    
    # Cluster summary
    power_clusters = [c for c in cluster_info if c['function'] == 'power_control']
    steering_clusters = [c for c in cluster_info if c['function'] == 'steering_control']
    
    power_ins = sum(c['n_interneurons'] for c in power_clusters)
    steering_ins = sum(c['n_interneurons'] for c in steering_clusters)
    
    # Save results
    output_dir = output_base_dir / f"min_syn_{min_synapses}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save connections
    premotor_conn.to_csv(output_dir / 'premotor_connections.csv', index=False)
    
    # Save cluster assignments
    cluster_assignments = pd.DataFrame([
        {'interneuron_id': in_id, 'cluster': clusters[i]}
        for i, in_id in enumerate(premotor_in_ids)
    ])
    cluster_assignments.to_csv(output_dir / 'cluster_assignments.csv', index=False)
    
    # Save cluster info
    cluster_df = pd.DataFrame([
        {k: v for k, v in c.items() if k != 'interneuron_ids'}
        for c in cluster_info
    ])
    cluster_df.to_csv(output_dir / 'cluster_properties.csv', index=False)
    
    # Return summary
    return {
        'min_synapses': min_synapses,
        'n_premotor_ins': len(premotor_in_ids),
        'n_in_mn_connections': len(premotor_conn),
        'total_in_mn_synapses': premotor_conn['weight'].sum(),
        'avg_synapses_per_connection': premotor_conn['weight'].mean(),
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'n_hubs': n_hubs,
        'hub_percentage': n_hubs / len(premotor_in_ids) * 100,
        'n_dns': n_dns,
        'n_dn_in_connections': dn_connections,
        'total_dn_in_synapses': dn_synapses,
        'power_clusters': len(power_clusters),
        'steering_clusters': len(steering_clusters),
        'power_ins': power_ins,
        'steering_ins': steering_ins,
        'power_percentage': power_ins / len(premotor_in_ids) * 100,
        'steering_percentage': steering_ins / len(premotor_in_ids) * 100
    }


def create_comparison_figures(summary_df, output_dir):
    """Create comprehensive comparison figures across thresholds."""
    
    print("\n" + "="*70)
    print("CREATING COMPARISON FIGURES")
    print("="*70 + "\n")
    
    fig = plt.figure(figsize=(24, 18), dpi=300)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    thresholds = summary_df['min_synapses'].values
    
    # Panel 1: Network size
    ax1 = fig.add_subplot(gs[0, 0])
    
    ax1.plot(thresholds, summary_df['n_premotor_ins'], 
            'o-', linewidth=3, markersize=10, color='#2E86AB', label='Premotor INs')
    ax1.plot(thresholds, summary_df['n_dns'],
            's-', linewidth=3, markersize=10, color='#A23B72', label='DNs')
    ax1.plot(thresholds, summary_df['n_hubs'],
            '^-', linewidth=3, markersize=10, color='#F18F01', label='Hub INs')
    
    ax1.set_xlabel('Minimum Synapses', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('A. Network Size vs Threshold', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(thresholds)
    
    # Panel 2: Connection counts
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.plot(thresholds, summary_df['n_in_mn_connections'],
            'o-', linewidth=3, markersize=10, color='#06A77D', label='IN→MN')
    ax2.plot(thresholds, summary_df['n_dn_in_connections'],
            's-', linewidth=3, markersize=10, color='#D62246', label='DN→IN')
    
    ax2.set_xlabel('Minimum Synapses', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Connections', fontsize=12, fontweight='bold')
    ax2.set_title('B. Connection Counts vs Threshold', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(thresholds)
    
    # Panel 3: Clustering quality
    ax3 = fig.add_subplot(gs[0, 2])
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(thresholds, summary_df['silhouette'],
                     'o-', linewidth=3, markersize=10, color='#6A4C93', 
                     label='Silhouette')
    ax3.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold', color='#6A4C93')
    ax3.tick_params(axis='y', labelcolor='#6A4C93')
    
    line2 = ax3_twin.plot(thresholds, summary_df['calinski_harabasz'],
                          's-', linewidth=3, markersize=10, color='#FF6B35',
                          label='Calinski-Harabasz')
    ax3_twin.set_ylabel('Calinski-Harabasz Index', fontsize=12, fontweight='bold', color='#FF6B35')
    ax3_twin.tick_params(axis='y', labelcolor='#FF6B35')
    
    ax3.set_xlabel('Minimum Synapses', fontsize=12, fontweight='bold')
    ax3.set_title('C. Clustering Quality vs Threshold', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.set_xticks(thresholds)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left', fontsize=10)
    
    # Panel 4: Total synapses
    ax4 = fig.add_subplot(gs[1, 0])
    
    ax4.bar(thresholds - 0.5, summary_df['total_in_mn_synapses'],
           width=3, alpha=0.7, color='#06A77D', label='IN→MN')
    ax4.bar(thresholds + 0.5, summary_df['total_dn_in_synapses'],
           width=3, alpha=0.7, color='#D62246', label='DN→IN')
    
    ax4.set_xlabel('Minimum Synapses', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Total Synapses', fontsize=12, fontweight='bold')
    ax4.set_title('D. Total Synaptic Strength vs Threshold', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_xticks(thresholds)
    
    # Panel 5: Power vs Steering
    ax5 = fig.add_subplot(gs[1, 1])
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, summary_df['power_ins'],
                   width, alpha=0.8, color='#FF6B6B', label='Power INs')
    bars2 = ax5.bar(x + width/2, summary_df['steering_ins'],
                   width, alpha=0.8, color='#4ECDC4', label='Steering INs')
    
    ax5.set_xlabel('Minimum Synapses', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Number of Interneurons', fontsize=12, fontweight='bold')
    ax5.set_title('E. Power vs Steering INs vs Threshold', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(thresholds)
    ax5.legend(fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    
    # Panel 6: Hub percentage
    ax6 = fig.add_subplot(gs[1, 2])
    
    ax6.plot(thresholds, summary_df['hub_percentage'],
            'o-', linewidth=3, markersize=10, color='#F18F01')
    ax6.fill_between(thresholds, 0, summary_df['hub_percentage'],
                     alpha=0.3, color='#F18F01')
    
    ax6.set_xlabel('Minimum Synapses', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Hub Percentage (%)', fontsize=12, fontweight='bold')
    ax6.set_title('F. Hub Neuron Percentage vs Threshold', fontsize=13, fontweight='bold')
    ax6.grid(alpha=0.3)
    ax6.set_xticks(thresholds)
    
    # Panel 7: Average synapses per connection
    ax7 = fig.add_subplot(gs[2, 0])
    
    ax7.plot(thresholds, summary_df['avg_synapses_per_connection'],
            'o-', linewidth=3, markersize=10, color='#9B59B6')
    ax7.axhline(y=25, color='red', linestyle='--', linewidth=2, 
               label='ShREC ≥0.4% equiv (~25 syn)')
    
    ax7.set_xlabel('Minimum Synapses', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Average Synapses', fontsize=12, fontweight='bold')
    ax7.set_title('G. Avg Synapses per IN→MN Connection', fontsize=13, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(alpha=0.3)
    ax7.set_xticks(thresholds)
    
    # Panel 8: Cluster composition
    ax8 = fig.add_subplot(gs[2, 1])
    
    x = np.arange(len(thresholds))
    
    bars1 = ax8.bar(x - width/2, summary_df['power_clusters'],
                   width, alpha=0.8, color='#FF6B6B', label='Power clusters')
    bars2 = ax8.bar(x + width/2, summary_df['steering_clusters'],
                   width, alpha=0.8, color='#4ECDC4', label='Steering clusters')
    
    ax8.set_xlabel('Minimum Synapses', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax8.set_title('H. Cluster Function vs Threshold', fontsize=13, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(thresholds)
    ax8.legend(fontsize=10)
    ax8.grid(axis='y', alpha=0.3)
    ax8.set_ylim([0, 5])
    
    # Panel 9: Summary table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Find optimal threshold
    # Balance: high silhouette, reasonable network size, good CH score
    normalized_sil = (summary_df['silhouette'] - summary_df['silhouette'].min()) / \
                     (summary_df['silhouette'].max() - summary_df['silhouette'].min())
    normalized_ch = (summary_df['calinski_harabasz'] - summary_df['calinski_harabasz'].min()) / \
                    (summary_df['calinski_harabasz'].max() - summary_df['calinski_harabasz'].min())
    
    # We want high quality but also reasonable network size
    # Penalize very small networks
    size_penalty = summary_df['n_premotor_ins'] / summary_df['n_premotor_ins'].max()
    
    composite_score = (0.4 * normalized_sil + 0.4 * normalized_ch + 0.2 * size_penalty)
    optimal_idx = composite_score.idxmax()
    optimal_threshold = summary_df.loc[optimal_idx, 'min_synapses']
    
    summary_text = f"""THRESHOLD COMPARISON SUMMARY

RANGE TESTED:
• Min: {thresholds.min()} synapses
• Max: {thresholds.max()} synapses
• Steps: {len(thresholds)} thresholds

OPTIMAL THRESHOLD:
• Threshold: {optimal_threshold} synapses
• Premotor INs: {summary_df.loc[optimal_idx, 'n_premotor_ins']:,.0f}
• Silhouette: {summary_df.loc[optimal_idx, 'silhouette']:.3f}
• CH Index: {summary_df.loc[optimal_idx, 'calinski_harabasz']:.1f}
• Hub %: {summary_df.loc[optimal_idx, 'hub_percentage']:.1f}%

EXTREMES:
Lowest threshold ({int(thresholds.min())} syn):
• INs: {summary_df.iloc[0]['n_premotor_ins']:,.0f}
• Quality: Sil={summary_df.iloc[0]['silhouette']:.3f}

Highest threshold ({int(thresholds.max())} syn):
• INs: {summary_df.iloc[-1]['n_premotor_ins']:,.0f}
• Quality: Sil={summary_df.iloc[-1]['silhouette']:.3f}

RECOMMENDATION:
{optimal_threshold} synapses provides optimal
balance of network completeness
and clustering quality.
"""
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
            fontsize=10, family='monospace', va='top',
            bbox=dict(boxstyle='round', facecolor='#F0F0F0',
                     edgecolor='black', linewidth=2, pad=0.8))
    
    fig.suptitle('Multi-Threshold Analysis: Complete Pipeline Comparison',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    plt.savefig(output_dir / 'threshold_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'threshold_comparison.pdf',
                format='pdf', bbox_inches='tight', facecolor='white')
    
    print("✓ Saved: threshold_comparison.png/pdf")
    
    plt.close()
    
    # Print recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"\nOptimal threshold: {optimal_threshold} synapses")
    print(f"  • Premotor INs: {summary_df.loc[optimal_idx, 'n_premotor_ins']:,.0f}")
    print(f"  • Clustering quality: Sil={summary_df.loc[optimal_idx, 'silhouette']:.3f}, CH={summary_df.loc[optimal_idx, 'calinski_harabasz']:.1f}")
    print(f"  • Hub neurons: {summary_df.loc[optimal_idx, 'n_hubs']:.0f} ({summary_df.loc[optimal_idx, 'hub_percentage']:.1f}%)")
    print(f"  • DNs involved: {summary_df.loc[optimal_idx, 'n_dns']:.0f}")


def main():
    print("\n" + "="*70)
    print("MULTI-THRESHOLD ANALYSIS PIPELINE")
    print("="*70 + "\n")
    
    print("Testing min_synapses: 15, 20, 25, 30, 40, 50")
    print("This will run the COMPLETE pipeline at each threshold.\n")
    
    # Thresholds to test
    thresholds = [15, 20, 25, 30, 40, 50]
    
    # Load data
    print("Loading data...")
    loader = ConnectomeDataLoader('config.yaml')
    connections, neurons, _ = loader.load_all_data(verbose=False)
    
    motor_pools = pd.read_csv('results/motor_neurons/motor_pools.csv')
    wing_mns = []
    for _, pool in motor_pools.iterrows():
        wing_mns.extend(eval(pool['motor_neuron_ids']))
    wing_mns = list(set(wing_mns))
    
    print(f"✓ Data loaded: {len(connections):,} connections, {len(neurons):,} neurons")
    print(f"✓ Wing MNs: {len(wing_mns)}\n")
    
    # Output directory
    output_base_dir = Path('results/multi_threshold_analysis')
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis for each threshold
    results = []
    
    for min_syn in thresholds:
        result = run_single_threshold(
            connections, neurons, motor_pools, wing_mns,
            min_syn, output_base_dir
        )
        results.append(result)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_base_dir / 'threshold_comparison_summary.csv', index=False)
    
    print("\n" + "="*70)
    print("CREATING COMPARISON FIGURES")
    print("="*70)
    
    # Create comparison figures
    create_comparison_figures(summary_df, output_base_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("MULTI-THRESHOLD ANALYSIS COMPLETE!")
    print("="*70 + "\n")
    
    print(f"Results saved to: {output_base_dir}\n")
    
    print("Output structure:")
    print("  results/multi_threshold_analysis/")
    print("    ├── threshold_comparison_summary.csv")
    print("    ├── threshold_comparison.png/pdf")
    for min_syn in thresholds:
        print(f"    ├── min_syn_{min_syn}/")
        print(f"    │   ├── premotor_connections.csv")
        print(f"    │   ├── cluster_assignments.csv")
        print(f"    │   └── cluster_properties.csv")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
