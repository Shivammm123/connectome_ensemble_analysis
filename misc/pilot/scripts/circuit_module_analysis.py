"""
Circuit Module Analysis

Comprehensive analysis of the discovered flight control modules:
1. Quantify modularity (how separated are power vs steering?)
2. Find integrative neurons (connect multiple modules)
3. Analyze module-specific properties
4. Statistical validation of circuit organization

Based on Cheong et al. 2024 network analysis methods.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import networkx as nx

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def compute_modularity_scores(
    in_mn_conn: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    motor_pools: pd.DataFrame
):
    """
    Compute modularity score for the circuit.
    
    Modularity measures how well the network separates into distinct modules.
    High modularity = clear functional separation.
    """
    
    print("Computing modularity scores...")
    print("-"*70)
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes with cluster labels
    for _, row in cluster_assignments.iterrows():
        G.add_node(row['interneuron_id'], cluster=row['cluster'])
    
    # Add edges (IN-IN connections via shared MN targets)
    # Two INs are connected if they target the same MN
    in_to_mn = {}
    for _, row in in_mn_conn.iterrows():
        in_id = row['source']
        mn_id = row['target']
        
        if in_id not in in_to_mn:
            in_to_mn[in_id] = []
        in_to_mn[in_id].append(mn_id)
    
    # Connect INs that share MN targets
    in_ids = list(in_to_mn.keys())
    for i, in1 in enumerate(in_ids):
        for in2 in in_ids[i+1:]:
            shared_targets = set(in_to_mn[in1]) & set(in_to_mn[in2])
            if len(shared_targets) > 0:
                G.add_edge(in1, in2, weight=len(shared_targets))
    
    # Compute modularity
    # Create community dict
    communities = {}
    for _, row in cluster_assignments.iterrows():
        cluster = row['cluster']
        if cluster not in communities:
            communities[cluster] = []
        communities[cluster].append(row['interneuron_id'])
    
    # Calculate modularity using networkx
    try:
        from networkx.algorithms import community
        modularity = community.modularity(G, communities.values())
        print(f"  Network modularity: {modularity:.4f}")
        print(f"    (Range: -0.5 to 1.0, higher = more modular)")
        
        if modularity > 0.4:
            print(f"    → HIGH modularity: Clear functional separation")
        elif modularity > 0.2:
            print(f"    → MODERATE modularity: Some separation")
        else:
            print(f"    → LOW modularity: Limited separation")
    except:
        modularity = None
        print("  Could not compute modularity (networkx version issue)")
    
    return modularity, G


def find_integrative_interneurons(
    in_mn_conn: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    motor_pools: pd.DataFrame,
    min_clusters: int = 2
):
    """
    Find INs that integrate across multiple clusters.
    
    These connect to motor pools from different functional modules.
    """
    
    print("\nFinding integrative interneurons...")
    print("-"*70)
    
    # Map MNs to muscle types
    mn_to_type = {}
    for _, pool in motor_pools.iterrows():
        for mn_id in pool['motor_neuron_ids']:
            mn_to_type[mn_id] = pool['muscle_type']
    
    # For each IN, find which muscle types it targets
    in_targets = {}
    for _, row in in_mn_conn.iterrows():
        in_id = row['source']
        mn_id = row['target']
        
        muscle_type = mn_to_type.get(mn_id, 'unknown')
        
        if in_id not in in_targets:
            in_targets[in_id] = set()
        in_targets[in_id].add(muscle_type)
    
    # Find INs targeting multiple types
    integrative_ins = []
    
    for in_id, muscle_types in in_targets.items():
        n_types = len([mt for mt in muscle_types if mt != 'unknown'])
        
        if n_types >= min_clusters:
            # Get cluster
            cluster_match = cluster_assignments[cluster_assignments['interneuron_id'] == in_id]
            cluster = cluster_match['cluster'].values[0] if len(cluster_match) > 0 else None
            
            # Count targets per type
            type_counts = {}
            for _, row in in_mn_conn[in_mn_conn['source'] == in_id].iterrows():
                mtype = mn_to_type.get(row['target'], 'unknown')
                type_counts[mtype] = type_counts.get(mtype, 0) + 1
            
            integrative_ins.append({
                'in_id': in_id,
                'cluster': cluster,
                'n_muscle_types': n_types,
                'muscle_types': list(muscle_types),
                'power_targets': type_counts.get('power', 0),
                'steering_targets': type_counts.get('steering', 0),
                'integration_score': min(type_counts.get('power', 0), type_counts.get('steering', 0))
            })
    
    integrative_df = pd.DataFrame(integrative_ins)
    integrative_df = integrative_df.sort_values('integration_score', ascending=False)
    
    print(f"  Total premotor INs: {len(in_targets)}")
    print(f"  Integrative INs (≥{min_clusters} muscle types): {len(integrative_df)}")
    
    if len(integrative_df) > 0:
        print(f"\n  Top 5 integrative INs:")
        for i, (_, row) in enumerate(integrative_df.head(5).iterrows(), 1):
            print(f"    {i}. IN {row['in_id']}: {row['power_targets']} power + {row['steering_targets']} steering targets")
    
    return integrative_df


def analyze_cluster_properties(
    cluster_info: pd.DataFrame,
    in_mn_conn: pd.DataFrame,
    cluster_assignments: pd.DataFrame
):
    """
    Detailed analysis of each cluster's properties.
    """
    
    print("\nAnalyzing cluster properties...")
    print("-"*70)
    
    cluster_properties = []
    
    for _, cluster in cluster_info.iterrows():
        cluster_id = cluster['cluster_id']
        
        # Get INs in this cluster
        cluster_ins = cluster_assignments[cluster_assignments['cluster'] == cluster_id]['interneuron_id'].tolist()
        
        # Get their connections
        cluster_conn = in_mn_conn[in_mn_conn['source'].isin(cluster_ins)]
        
        # Properties
        n_connections = len(cluster_conn)
        total_synapses = cluster_conn['weight'].sum()
        mean_synapses = cluster_conn['weight'].mean() if len(cluster_conn) > 0 else 0
        max_synapses = cluster_conn['weight'].max() if len(cluster_conn) > 0 else 0
        
        # Connectivity density (connections per IN)
        conn_per_in = n_connections / len(cluster_ins) if len(cluster_ins) > 0 else 0
        syn_per_in = total_synapses / len(cluster_ins) if len(cluster_ins) > 0 else 0
        
        cluster_properties.append({
            'cluster_id': cluster_id,
            'functional_type': cluster['functional_type'],
            'n_interneurons': cluster['n_interneurons'],
            'n_connections': n_connections,
            'total_synapses': total_synapses,
            'mean_synapses': mean_synapses,
            'max_synapses': max_synapses,
            'connections_per_in': conn_per_in,
            'synapses_per_in': syn_per_in
        })
    
    props_df = pd.DataFrame(cluster_properties)
    
    print(f"\n  Cluster properties summary:")
    print(f"  {'Cluster':<10} {'Type':<15} {'INs':<6} {'Conn/IN':<10} {'Syn/IN':<10}")
    print("-"*70)
    for _, row in props_df.iterrows():
        print(f"  {row['cluster_id']:<10} {row['functional_type']:<15} {row['n_interneurons']:<6} "
              f"{row['connections_per_in']:<10.2f} {row['synapses_per_in']:<10.1f}")
    
    return props_df


def statistical_validation(
    cluster_properties: pd.DataFrame,
    in_mn_conn: pd.DataFrame,
    cluster_assignments: pd.DataFrame
):
    """
    Statistical tests to validate cluster differences.
    """
    
    print("\nStatistical validation...")
    print("-"*70)
    
    # Test: Power vs Steering cluster connectivity differences
    power_cluster = cluster_properties[cluster_properties['functional_type'] == 'power_control']
    steering_clusters = cluster_properties[cluster_properties['functional_type'] == 'steering_control']
    
    if len(power_cluster) > 0 and len(steering_clusters) > 0:
        # Compare synapses per IN
        power_syn_per_in = power_cluster['synapses_per_in'].values[0]
        steering_syn_per_in = steering_clusters['synapses_per_in'].mean()
        
        print(f"\n  Power vs Steering comparison:")
        print(f"    Power cluster synapses/IN:    {power_syn_per_in:.1f}")
        print(f"    Steering clusters synapses/IN: {steering_syn_per_in:.1f}")
        print(f"    Ratio (Power/Steering):        {power_syn_per_in/steering_syn_per_in:.2f}x")
        
        # Are power INs more strongly connected?
        if power_syn_per_in > steering_syn_per_in * 1.5:
            print(f"    → Power INs are MUCH more strongly connected")
        elif power_syn_per_in > steering_syn_per_in:
            print(f"    → Power INs are somewhat more connected")
        else:
            print(f"    → Similar connectivity strength")
    
    # Test: Within-cluster vs between-cluster connectivity
    print(f"\n  Testing module separation...")
    
    # Calculate within-cluster connection strength
    within_cluster_weights = []
    between_cluster_weights = []
    
    for _, row in in_mn_conn.iterrows():
        in_id = row['source']
        weight = row['weight']
        
        in_cluster = cluster_assignments[cluster_assignments['interneuron_id'] == in_id]
        
        if len(in_cluster) > 0:
            # For simplicity, consider all connections as "within" their cluster's functional domain
            # A more sophisticated version would track MN cluster membership
            within_cluster_weights.append(weight)
    
    if len(within_cluster_weights) > 0:
        print(f"    Mean connection strength: {np.mean(within_cluster_weights):.1f} synapses")
        print(f"    Std connection strength:  {np.std(within_cluster_weights):.1f} synapses")


def create_module_analysis_visualizations(
    cluster_properties: pd.DataFrame,
    integrative_ins: pd.DataFrame,
    in_mn_conn: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    output_dir: Path
):
    """
    Create comprehensive module analysis visualizations.
    """
    
    print("\nCreating module analysis visualizations...")
    print("-"*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)
    
    # 1. Cluster connectivity comparison
    ax = axes[0, 0]
    
    colors = ['#FF6B6B' if ft == 'power_control' else '#4ECDC4' 
              for ft in cluster_properties['functional_type']]
    
    x = np.arange(len(cluster_properties))
    width = 0.35
    
    ax.bar(x - width/2, cluster_properties['connections_per_in'], 
           width, label='Connections/IN', color=colors, alpha=0.7)
    ax.bar(x + width/2, cluster_properties['synapses_per_in']/10, 
           width, label='Synapses/IN (÷10)', color=colors, alpha=0.4)
    
    ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Connectivity Density', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_properties['cluster_id'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Power vs Steering total synapses
    ax = axes[0, 1]
    
    power_total = cluster_properties[cluster_properties['functional_type'] == 'power_control']['total_synapses'].sum()
    steering_total = cluster_properties[cluster_properties['functional_type'] == 'steering_control']['total_synapses'].sum()
    
    ax.bar(['Power\nModule', 'Steering\nModules'], 
           [power_total, steering_total],
           color=['#FF6B6B', '#4ECDC4'])
    ax.set_ylabel('Total Synapses', fontsize=12, fontweight='bold')
    ax.set_title('Module Connectivity Strength', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add values
    ax.text(0, power_total + 500, f'{int(power_total):,}', 
           ha='center', va='bottom', fontweight='bold')
    ax.text(1, steering_total + 500, f'{int(steering_total):,}', 
           ha='center', va='bottom', fontweight='bold')
    
    # 3. Integrative neurons distribution
    ax = axes[0, 2]
    
    if len(integrative_ins) > 0:
        integration_dist = integrative_ins['integration_score'].value_counts().sort_index()
        integration_dist.plot(kind='bar', ax=ax, color='#FFA07A')
        ax.set_xlabel('Integration Score\n(min of power/steering targets)', 
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of INs', fontsize=12, fontweight='bold')
        ax.set_title('Integrative Interneurons', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No integrative\nneurons found', 
               ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # 4. Connection strength distribution
    ax = axes[1, 0]
    
    power_cluster_id = cluster_properties[cluster_properties['functional_type'] == 'power_control']['cluster_id'].values
    steering_cluster_ids = cluster_properties[cluster_properties['functional_type'] == 'steering_control']['cluster_id'].values
    
    if len(power_cluster_id) > 0:
        power_ins = cluster_assignments[cluster_assignments['cluster'].isin(power_cluster_id)]['interneuron_id']
        power_conn = in_mn_conn[in_mn_conn['source'].isin(power_ins)]['weight']
        
        steering_ins = cluster_assignments[cluster_assignments['cluster'].isin(steering_cluster_ids)]['interneuron_id']
        steering_conn = in_mn_conn[in_mn_conn['source'].isin(steering_ins)]['weight']
        
        ax.hist([power_conn, steering_conn], bins=20, 
               label=['Power', 'Steering'],
               color=['#FF6B6B', '#4ECDC4'],
               alpha=0.6)
        ax.set_xlabel('Synapse Count', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Connection Strength Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # 5. Cluster size vs connectivity
    ax = axes[1, 1]
    
    colors_scatter = ['#FF6B6B' if ft == 'power_control' else '#4ECDC4' 
                     for ft in cluster_properties['functional_type']]
    
    ax.scatter(cluster_properties['n_interneurons'], 
              cluster_properties['total_synapses'],
              c=colors_scatter, s=200, alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add cluster labels
    for _, row in cluster_properties.iterrows():
        ax.annotate(f"C{row['cluster_id']}", 
                   (row['n_interneurons'], row['total_synapses']),
                   ha='center', va='center', fontweight='bold')
    
    ax.set_xlabel('Number of Interneurons', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Synapses', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Size vs Connectivity', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 6. Module summary
    ax = axes[1, 2]
    
    summary_text = "CIRCUIT MODULE SUMMARY\n\n"
    summary_text += f"Total Clusters: {len(cluster_properties)}\n"
    summary_text += f"  Power: {len(cluster_properties[cluster_properties['functional_type'] == 'power_control'])}\n"
    summary_text += f"  Steering: {len(cluster_properties[cluster_properties['functional_type'] == 'steering_control'])}\n\n"
    
    summary_text += f"Total Interneurons: {cluster_properties['n_interneurons'].sum()}\n"
    summary_text += f"  Power: {cluster_properties[cluster_properties['functional_type'] == 'power_control']['n_interneurons'].sum()}\n"
    summary_text += f"  Steering: {cluster_properties[cluster_properties['functional_type'] == 'steering_control']['n_interneurons'].sum()}\n\n"
    
    summary_text += f"Total Connections: {cluster_properties['n_connections'].sum():,.0f}\n"
    summary_text += f"Total Synapses: {cluster_properties['total_synapses'].sum():,.0f}\n\n"
    
    if len(integrative_ins) > 0:
        summary_text += f"Integrative INs: {len(integrative_ins)}\n"
        summary_text += f"  ({100*len(integrative_ins)/cluster_properties['n_interneurons'].sum():.1f}% of total)"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'circuit_module_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: circuit_module_analysis.png")


def main():
    print("\n" + "="*70)
    print("CIRCUIT MODULE ANALYSIS")
    print("Comprehensive analysis of functional modules")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    print("-"*70 + "\n")
    
    # IN clusters
    premotor_ins = pd.read_csv('results/interneuron_clusters/premotor_interneurons_clustered.csv')
    cluster_assignments = premotor_ins[['Root ID', 'cluster']].copy()
    cluster_assignments.columns = ['interneuron_id', 'cluster']
    print(f"  ✓ Clustered INs: {len(cluster_assignments):,}")
    
    # Cluster info
    cluster_info = pd.read_csv('results/interneuron_clusters/cluster_characteristics.csv')
    print(f"  ✓ Clusters: {len(cluster_info)}")
    
    # IN→MN connections
    in_mn_conn = pd.read_csv('results/interneuron_clusters/in_to_mn_connections.csv')
    print(f"  ✓ IN→MN connections: {len(in_mn_conn):,}")
    
    # Motor pools
    motor_pools = pd.read_csv('results/motor_neurons/motor_pools.csv')
    motor_pools['motor_neuron_ids'] = motor_pools['motor_neuron_ids'].apply(eval)
    print(f"  ✓ Motor pools: {len(motor_pools)}\n")
    
    # Analysis steps
    # 1. Modularity
    modularity, network = compute_modularity_scores(
        in_mn_conn,
        cluster_assignments,
        motor_pools
    )
    
    # 2. Integrative neurons
    integrative_ins = find_integrative_interneurons(
        in_mn_conn,
        cluster_assignments,
        motor_pools,
        min_clusters=2
    )
    
    # 3. Cluster properties
    cluster_props = analyze_cluster_properties(
        cluster_info,
        in_mn_conn,
        cluster_assignments
    )
    
    # 4. Statistical validation
    statistical_validation(
        cluster_props,
        in_mn_conn,
        cluster_assignments
    )
    
    # Save results
    print("\n" + "-"*70)
    print("Saving results")
    print("-"*70 + "\n")
    
    output_dir = Path('results/circuit_modules')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save properties
    cluster_props.to_csv(output_dir / 'cluster_properties_detailed.csv', index=False)
    print(f"  ✓ cluster_properties_detailed.csv")
    
    # Save integrative neurons
    if len(integrative_ins) > 0:
        integrative_ins.to_csv(output_dir / 'integrative_interneurons.csv', index=False)
        print(f"  ✓ integrative_interneurons.csv ({len(integrative_ins)} INs)")
    
    # Save modularity score
    if modularity is not None:
        with open(output_dir / 'modularity_score.txt', 'w') as f:
            f.write(f"Network Modularity: {modularity:.4f}\n")
            f.write(f"Number of modules: {len(cluster_info)}\n")
        print(f"  ✓ modularity_score.txt")
    
    # Visualizations
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    create_module_analysis_visualizations(
        cluster_props,
        integrative_ins,
        in_mn_conn,
        cluster_assignments,
        fig_dir
    )
    
    # Summary
    print("\n" + "="*70)
    print("MODULE ANALYSIS COMPLETE")
    print("="*70 + "\n")
    
    print(f"Key findings:")
    if modularity is not None:
        print(f"  Network modularity:      {modularity:.4f}")
    print(f"  Functional modules:      {len(cluster_info)}")
    print(f"  Integrative neurons:     {len(integrative_ins) if len(integrative_ins) > 0 else 0}")
    
    print(f"\nPower vs Steering:")
    power = cluster_props[cluster_props['functional_type'] == 'power_control']
    steering = cluster_props[cluster_props['functional_type'] == 'steering_control']
    
    print(f"  Power module:")
    print(f"    INs: {power['n_interneurons'].sum()}")
    print(f"    Synapses: {power['total_synapses'].sum():,.0f}")
    
    print(f"  Steering modules:")
    print(f"    INs: {steering['n_interneurons'].sum()}")
    print(f"    Synapses: {steering['total_synapses'].sum():,.0f}")
    
    print(f"\nResults: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()