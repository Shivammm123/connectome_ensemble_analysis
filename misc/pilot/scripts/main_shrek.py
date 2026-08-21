"""
Complete ShREC-Based Analysis Including DN Pathways

Full pipeline:
1. ShREC-filtered IN→MN connections (premotor INs)
2. ShREC-filtered DN→IN connections (DN pathways)
3. DN→IN→MN complete pathways
4. DN specialization (power vs steering)
5. Complete prioritization
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

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def compute_shrec_scores(connections, target_neurons, target_type="MN"):
    """
    Compute ShREC scores for connections to target neurons.
    
    Parameters:
    - connections: full connection dataframe
    - target_neurons: list of target neuron IDs
    - target_type: "MN" or "IN" (for labeling)
    """
    
    print(f"\nComputing ShREC scores for connections to {target_type}s...")
    
    # Filter for target connections
    target_conn = connections[connections['target'].isin(target_neurons)].copy()
    
    print(f"  Total connections to {target_type}s: {len(target_conn):,}")
    
    # Compute total input to each target
    target_total_input = connections[
        connections['target'].isin(target_neurons)
    ].groupby('target')['weight'].sum().to_dict()
    
    # Compute ShREC
    target_conn['target_total_input'] = target_conn['target'].map(target_total_input)
    target_conn['shrec_score'] = (target_conn['weight'] / target_conn['target_total_input']) * 100
    
    print(f"  ShREC statistics:")
    print(f"    Mean:   {target_conn['shrec_score'].mean():.3f}%")
    print(f"    Median: {target_conn['shrec_score'].median():.3f}%")
    print(f"    Max:    {target_conn['shrec_score'].max():.3f}%")
    
    return target_conn


def identify_premotor_ins_shrec(wing_conn_shrec, neurons, shrec_threshold=0.4):
    """Identify premotor INs using ShREC threshold."""
    
    print("\n" + "="*70)
    print(f"STEP 1: IDENTIFYING PREMOTOR INs (ShREC ≥{shrec_threshold}%)")
    print("="*70)
    
    # Filter by ShREC threshold
    significant_conn = wing_conn_shrec[
        wing_conn_shrec['shrec_score'] >= shrec_threshold
    ].copy()
    
    print(f"\nConnections passing ShREC threshold: {len(significant_conn):,}")
    
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
    print(f"✓ Premotor IN→MN connections (ShREC ≥{shrec_threshold}%): {len(premotor_conn):,}")
    print(f"✓ Average ShREC score: {premotor_conn['shrec_score'].mean():.3f}%")
    
    return premotor_in_ids, premotor_conn


def identify_dn_to_in_pathways_shrec(connections, premotor_in_ids, neurons, 
                                     shrec_threshold=0.4):
    """
    Identify DN→IN connections using ShREC threshold.
    
    This finds which DNs provide functionally significant input to our 
    ShREC-filtered premotor INs.
    """
    
    print("\n" + "="*70)
    print(f"STEP 2: IDENTIFYING DN→IN PATHWAYS (ShREC ≥{shrec_threshold}%)")
    print("="*70)
    
    # Compute ShREC for connections TO premotor INs
    in_conn_shrec = compute_shrec_scores(
        connections, premotor_in_ids, target_type="IN"
    )
    
    # Filter by ShREC threshold
    significant_dn_in = in_conn_shrec[
        in_conn_shrec['shrec_score'] >= shrec_threshold
    ].copy()
    
    print(f"\nConnections passing ShREC threshold: {len(significant_dn_in):,}")
    
    # Get descending neurons
    dns = neurons[
        neurons['Super Class'] == 'descending'
    ]['Root ID'].tolist()
    
    # Filter for DN→IN connections
    dn_in_conn = significant_dn_in[
        significant_dn_in['source'].isin(dns)
    ].copy()
    
    dn_ids = dn_in_conn['source'].unique()
    
    print(f"\n✓ DNs connecting to premotor INs: {len(dn_ids):,}")
    print(f"✓ DN→IN connections (ShREC ≥{shrec_threshold}%): {len(dn_in_conn):,}")
    print(f"✓ Average ShREC score: {dn_in_conn['shrec_score'].mean():.3f}%")
    print(f"✓ Total synapses: {dn_in_conn['weight'].sum():,.0f}")
    
    return dn_ids, dn_in_conn


def classify_dns_by_target_shrec(dn_ids, dn_in_conn, premotor_in_ids, 
                                 clusters, cluster_info):
    """
    Classify DNs as power_control or steering_control based on which 
    premotor IN clusters they target.
    """
    
    print("\n" + "="*70)
    print("STEP 3: CLASSIFYING DN SPECIALIZATION")
    print("="*70)
    
    # Create IN→cluster mapping
    in_cluster_map = dict(zip(premotor_in_ids, clusters))
    
    # Add cluster info to DN→IN connections
    dn_in_conn['target_cluster'] = dn_in_conn['target'].map(in_cluster_map)
    
    # Get cluster functions
    cluster_function_map = {
        c['cluster']: c['function'] 
        for c in cluster_info
    }
    
    dn_in_conn['target_function'] = dn_in_conn['target_cluster'].map(
        cluster_function_map
    )
    
    # Classify each DN
    dn_classifications = []
    
    for dn_id in dn_ids:
        dn_connections = dn_in_conn[dn_in_conn['source'] == dn_id]
        
        # Sum synapses by function
        power_synapses = dn_connections[
            dn_connections['target_function'] == 'power_control'
        ]['weight'].sum()
        
        steering_synapses = dn_connections[
            dn_connections['target_function'] == 'steering_control'
        ]['weight'].sum()
        
        total_synapses = power_synapses + steering_synapses
        
        # Classify based on majority
        if power_synapses > steering_synapses:
            specialization = 'power_control'
            primary_synapses = power_synapses
            secondary_synapses = steering_synapses
        else:
            specialization = 'steering_control'
            primary_synapses = steering_synapses
            secondary_synapses = power_synapses
        
        # Count target INs
        n_target_ins = dn_connections['target'].nunique()
        
        # Count target clusters
        target_clusters = dn_connections['target_cluster'].value_counts().to_dict()
        
        dn_classifications.append({
            'dn_id': dn_id,
            'specialization': specialization,
            'total_synapses': total_synapses,
            'power_synapses': power_synapses,
            'steering_synapses': steering_synapses,
            'primary_synapses': primary_synapses,
            'secondary_synapses': secondary_synapses,
            'specificity': primary_synapses / total_synapses if total_synapses > 0 else 0,
            'n_target_ins': n_target_ins,
            'target_clusters': target_clusters
        })
    
    dn_class_df = pd.DataFrame(dn_classifications)
    
    # Summary
    power_dns = dn_class_df[dn_class_df['specialization'] == 'power_control']
    steering_dns = dn_class_df[dn_class_df['specialization'] == 'steering_control']
    
    print(f"\nDN Specialization:")
    print(f"  Power DNs:    {len(power_dns):,} ({len(power_dns)/len(dn_class_df)*100:.1f}%)")
    print(f"  Steering DNs: {len(steering_dns):,} ({len(steering_dns)/len(dn_class_df)*100:.1f}%)")
    
    print(f"\nAverage specificity:")
    print(f"  Power DNs:    {power_dns['specificity'].mean():.2f}")
    print(f"  Steering DNs: {steering_dns['specificity'].mean():.2f}")
    
    return dn_class_df


def build_complete_pathways_shrec(dn_in_conn, premotor_conn, motor_pools, loader):
    """
    Build complete DN→IN→MN→Muscle pathways.
    """
    
    print("\n" + "="*70)
    print("STEP 4: BUILDING COMPLETE PATHWAYS")
    print("="*70)
    
    # Create muscle mapping
    mn_muscle_map = {}
    for _, pool in motor_pools.iterrows():
        muscle = pool['muscle']
        mn_ids = eval(pool['motor_neuron_ids'])
        for mn_id in mn_ids:
            mn_muscle_map[mn_id] = muscle
    
    # Build pathways
    pathways = []
    
    print("\nBuilding DN→IN→MN pathways...")
    
    for _, dn_in in dn_in_conn.iterrows():
        dn_id = dn_in['source']
        in_id = dn_in['target']
        dn_in_synapses = dn_in['weight']
        dn_in_shrec = dn_in['shrec_score']
        
        # Find IN→MN connections for this IN
        in_mn = premotor_conn[premotor_conn['source'] == in_id]
        
        for _, in_mn_row in in_mn.iterrows():
            mn_id = in_mn_row['target']
            in_mn_synapses = in_mn_row['weight']
            in_mn_shrec = in_mn_row['shrec_score']
            
            muscle = mn_muscle_map.get(mn_id, 'unknown')
            
            pathways.append({
                'dn_id': dn_id,
                'dn_name': loader.get_neuron_name(dn_id),
                'in_id': in_id,
                'in_name': loader.get_neuron_name(in_id),
                'mn_id': mn_id,
                'mn_name': loader.get_neuron_name(mn_id),
                'muscle': muscle,
                'dn_in_synapses': dn_in_synapses,
                'in_mn_synapses': in_mn_synapses,
                'dn_in_shrec': dn_in_shrec,
                'in_mn_shrec': in_mn_shrec,
                'pathway_strength': dn_in_synapses * in_mn_synapses
            })
    
    pathways_df = pd.DataFrame(pathways)
    
    print(f"\n✓ Complete pathways: {len(pathways_df):,}")
    print(f"✓ Unique DN→IN→MN chains: {len(pathways_df):,}")
    print(f"✓ DNs involved: {pathways_df['dn_id'].nunique():,}")
    print(f"✓ INs involved: {pathways_df['in_id'].nunique():,}")
    print(f"✓ MNs involved: {pathways_df['mn_id'].nunique():,}")
    
    return pathways_df


def cluster_interneurons_shrec(premotor_in_ids, premotor_conn, motor_pools, 
                               n_clusters=5):
    """Cluster interneurons by connectivity patterns."""
    
    print("\n" + "="*70)
    print(f"STEP 5: CLUSTERING INTERNEURONS (k={n_clusters})")
    print("="*70)
    
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
    
    print(f"\nConnectivity matrix: {conn_matrix.shape}")
    
    # Clustering
    similarity = cosine_similarity(conn_matrix)
    distance = np.clip(1 - similarity, 0, None)  # guard against float precision negatives

    linkage_matrix = linkage(distance, method='ward')
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Metrics
    silhouette = silhouette_score(distance, clusters, metric='precomputed')
    calinski = calinski_harabasz_score(conn_matrix, clusters)
    
    print(f"\n✓ Silhouette: {silhouette:.4f}")
    print(f"✓ Calinski-Harabasz: {calinski:.1f}")
    
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
            'avg_synapses_per_in': total_syn / len(cluster_ins) if len(cluster_ins) > 0 else 0,
            'function': 'power_control' if is_power else 'steering_control',
            'top_targets': ', '.join(top_pool_names)
        })
        
        print(f"  Cluster {c} ({cluster_info[-1]['function']}): "
              f"{len(cluster_ins)} INs")
    
    return clusters, cluster_info, silhouette, calinski


def create_shrec_dn_figure(dn_class_df, dn_in_conn, pathways_df, 
                           premotor_in_ids, output_dir):
    """Create comprehensive DN analysis figure."""
    
    print("\n" + "="*70)
    print("CREATING DN ANALYSIS FIGURE")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
    
    # Panel 1: DN specialization
    ax = axes[0, 0]
    
    spec_counts = dn_class_df['specialization'].value_counts()
    colors = ['#FF6B6B' if s == 'power_control' else '#4ECDC4' 
              for s in spec_counts.index]
    
    bars = ax.bar(range(len(spec_counts)), spec_counts.values,
                  color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    ax.set_xticks(range(len(spec_counts)))
    ax.set_xticklabels([s.replace('_', '\n') for s in spec_counts.index], 
                       fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of DNs', fontsize=12, fontweight='bold')
    ax.set_title('A. DN Specialization (ShREC-based)',
                fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, spec_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2,
               f'{val}', ha='center', va='bottom',
               fontsize=12, fontweight='bold')
    
    # Panel 2: Top DNs by connectivity
    ax = axes[0, 1]
    
    top_dns = dn_class_df.nlargest(15, 'total_synapses')
    colors_top = ['#FF6B6B' if s == 'power_control' else '#4ECDC4'
                  for s in top_dns['specialization']]
    
    y_pos = np.arange(len(top_dns))
    bars = ax.barh(y_pos, top_dns['total_synapses'],
                   color=colors_top, edgecolor='black', linewidth=1, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"DN {dn_id}" for dn_id in top_dns['dn_id']], fontsize=9)
    ax.set_xlabel('Total Synapses to Premotor INs', fontsize=11, fontweight='bold')
    ax.set_title('B. Top 15 DNs (ShREC ≥0.4%)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Panel 3: DN specificity distribution
    ax = axes[0, 2]
    
    ax.hist(dn_class_df['specificity'], bins=20,
           color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(dn_class_df['specificity'].median(), color='red',
              linestyle='--', linewidth=2,
              label=f"Median: {dn_class_df['specificity'].median():.2f}")
    ax.set_xlabel('Specificity (primary/total)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('C. DN Specialization Specificity', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 4: Number of target INs per DN
    ax = axes[1, 0]
    
    ax.hist(dn_class_df['n_target_ins'], bins=20,
           color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(dn_class_df['n_target_ins'].median(), color='red',
              linestyle='--', linewidth=2,
              label=f"Median: {dn_class_df['n_target_ins'].median():.0f}")
    ax.set_xlabel('Number of Target Premotor INs', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('D. DN Connectivity Breadth', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 5: Pathway strength distribution
    ax = axes[1, 1]
    
    ax.hist(np.log10(pathways_df['pathway_strength'] + 1), bins=30,
           color='#9370DB', edgecolor='black', alpha=0.7)
    ax.set_xlabel('log10(Pathway Strength)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('E. DN→IN→MN Pathway Strength', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Panel 6: Summary stats
    ax = axes[1, 2]
    ax.axis('off')
    
    power_dns = dn_class_df[dn_class_df['specialization'] == 'power_control']
    steering_dns = dn_class_df[dn_class_df['specialization'] == 'steering_control']
    
    summary_text = f"""ShREC-BASED DN ANALYSIS

DN STATISTICS:
- Total DNs: {len(dn_class_df)}
- Power DNs: {len(power_dns)} ({len(power_dns)/len(dn_class_df)*100:.1f}%)
- Steering DNs: {len(steering_dns)} ({len(steering_dns)/len(dn_class_df)*100:.1f}%)

CONNECTIVITY:
- DN→IN connections: {len(dn_in_conn):,}
- Complete pathways: {len(pathways_df):,}
- Avg INs per DN: {dn_class_df['n_target_ins'].mean():.1f}

PATHWAY ANALYSIS:
- DNs in pathways: {pathways_df['dn_id'].nunique()}
- INs in pathways: {pathways_df['in_id'].nunique()}
- MNs in pathways: {pathways_df['mn_id'].nunique()}

TOP DN:
- ID: {dn_class_df.iloc[0]['dn_id']}
- Type: {dn_class_df.iloc[0]['specialization']}
- Target INs: {int(dn_class_df.iloc[0]['n_target_ins'])}
- Total syn: {int(dn_class_df.iloc[0]['total_synapses']):,}
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=10, family='monospace', va='top',
           bbox=dict(boxstyle='round', facecolor='#F0F0F0',
                    edgecolor='black', linewidth=2, pad=0.8))
    
    fig.suptitle('Descending Neuron Analysis (ShREC-Filtered Network)',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dn_analysis_shrec.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'dn_analysis_shrec.pdf',
                format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Saved: dn_analysis_shrec.png/pdf")


def main():
    print("\n" + "="*70)
    print("COMPLETE ShREC ANALYSIS WITH DN PATHWAYS")
    print("="*70 + "\n")
    
    shrec_threshold = 0.4
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
    
    print(f"✓ Data loaded: {len(connections):,} connections, {len(neurons):,} neurons")
    
    # Step 1: Identify premotor INs (ShREC on IN→MN)
    wing_conn_shrec = compute_shrec_scores(connections, wing_mns, "MN")
    premotor_in_ids, premotor_conn = identify_premotor_ins_shrec(
        wing_conn_shrec, neurons, shrec_threshold
    )
    
    # Step 2: Identify DN→IN pathways (ShREC on DN→IN)
    dn_ids, dn_in_conn = identify_dn_to_in_pathways_shrec(
        connections, premotor_in_ids, neurons, shrec_threshold
    )
    
    # Step 3: Cluster INs
    clusters, cluster_info, silhouette, calinski = cluster_interneurons_shrec(
        premotor_in_ids, premotor_conn, motor_pools, n_clusters
    )
    
    # Step 4: Classify DNs
    dn_class_df = classify_dns_by_target_shrec(
        dn_ids, dn_in_conn, premotor_in_ids, clusters, cluster_info
    )
    
    # Step 5: Build complete pathways
    pathways_df = build_complete_pathways_shrec(
        dn_in_conn, premotor_conn, motor_pools, loader
    )
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    output_dir = Path('results/shrec_complete_with_dns')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all outputs
    premotor_conn.to_csv(output_dir / 'in_to_mn_connections_shrec.csv', index=False)
    dn_in_conn.to_csv(output_dir / 'dn_to_in_connections_shrec.csv', index=False)
    dn_class_df.to_csv(output_dir / 'dn_classifications.csv', index=False)
    pathways_df.to_csv(output_dir / 'complete_pathways_dn_in_mn.csv', index=False)
    
    cluster_df = pd.DataFrame(cluster_info)
    cluster_df.to_csv(output_dir / 'cluster_properties.csv', index=False)
    
    print(f"\n✓ in_to_mn_connections_shrec.csv")
    print(f"✓ dn_to_in_connections_shrec.csv")
    print(f"✓ dn_classifications.csv")
    print(f"✓ complete_pathways_dn_in_mn.csv")
    print(f"✓ cluster_properties.csv")
    
    # Create figures
    create_shrec_dn_figure(dn_class_df, dn_in_conn, pathways_df,
                          premotor_in_ids, output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("COMPLETE ShREC ANALYSIS FINISHED!")
    print("="*70 + "\n")
    
    print(f"Results saved to: {output_dir}\n")
    
    print("NETWORK SUMMARY:")
    print(f"  • Premotor INs (ShREC ≥{shrec_threshold}%): {len(premotor_in_ids):,}")
    print(f"  • DNs (ShREC ≥{shrec_threshold}%): {len(dn_ids):,}")
    print(f"  • IN→MN connections: {len(premotor_conn):,}")
    print(f"  • DN→IN connections: {len(dn_in_conn):,}")
    print(f"  • Complete DN→IN→MN pathways: {len(pathways_df):,}")
    print(f"  • Clustering: {n_clusters} modules (Sil={silhouette:.3f})")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()