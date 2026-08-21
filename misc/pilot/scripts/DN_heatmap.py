"""
DN→IN Connectivity Heatmap (ShREC-filtered)

Creates publication-quality heatmap showing:
- Top DNs (rows) × Top premotor INs (columns)
- Connection strength via ShREC scores
- Neuron names and IDs
- Cluster annotations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def create_dn_in_heatmap(dn_in_conn, cluster_assignments, loader,
                         top_n_dns=15, top_n_ins=20, output_dir=None):
    """
    Create DN→IN connectivity heatmap with neuron names.
    
    Parameters:
    - dn_in_conn: DN→IN connections (ShREC-filtered)
    - cluster_assignments: dict mapping IN ID → cluster number
    - loader: ConnectomeDataLoader for neuron names
    - top_n_dns: Number of top DNs to show
    - top_n_ins: Number of top INs to show
    """
    
    print("\n" + "="*70)
    print("CREATING DN→IN CONNECTIVITY HEATMAP (ShREC)")
    print("="*70 + "\n")
    
    # Select top DNs by total connectivity
    dn_totals = dn_in_conn.groupby('source')['weight'].sum().sort_values(ascending=False)
    top_dns = dn_totals.head(top_n_dns).index.tolist()
    
    print(f"Top {top_n_dns} DNs selected:")
    print(f"  Range: {dn_totals.iloc[0]:.0f} - {dn_totals.iloc[top_n_dns-1]:.0f} total synapses")
    
    # Select top INs by total DN input
    in_totals = dn_in_conn.groupby('target')['weight'].sum().sort_values(ascending=False)
    top_ins = in_totals.head(top_n_ins).index.tolist()
    
    print(f"\nTop {top_n_ins} INs selected:")
    print(f"  Range: {in_totals.iloc[0]:.0f} - {in_totals.iloc[top_n_ins-1]:.0f} total synapses")
    
    # Build connectivity matrix
    conn_matrix = np.zeros((len(top_dns), len(top_ins)))
    shrec_matrix = np.zeros((len(top_dns), len(top_ins)))
    
    print("\nBuilding connectivity matrices...")
    
    for i, dn_id in enumerate(top_dns):
        for j, in_id in enumerate(top_ins):
            connections = dn_in_conn[
                (dn_in_conn['source'] == dn_id) &
                (dn_in_conn['target'] == in_id)
            ]
            
            if len(connections) > 0:
                conn_matrix[i, j] = connections['weight'].values[0]
                shrec_matrix[i, j] = connections['shrec_score'].values[0]
    
    print(f"  Connections in matrix: {np.count_nonzero(conn_matrix)}")
    print(f"  Total synapses: {conn_matrix.sum():.0f}")
    print(f"  Average ShREC: {shrec_matrix[shrec_matrix > 0].mean():.3f}%")
    
    # Get neuron names
    print("\nFetching neuron names...")
    
    dn_labels = []
    for dn_id in top_dns:
        name = loader.get_neuron_name(dn_id)
        label = str(name)[:35] if name != str(dn_id) else f"DN_{dn_id}"
        dn_labels.append(label)

    in_labels = []
    in_clusters = []
    for in_id in top_ins:
        name = loader.get_neuron_name(in_id)
        cluster = cluster_assignments.get(in_id, 0)
        in_clusters.append(cluster)
        name_str = str(name)[:28] if name != str(in_id) else f"IN_{in_id}"
        label = f"{name_str}\nC{cluster}"
        in_labels.append(label)
    
    # Sort INs by cluster
    cluster_order = np.argsort(in_clusters)
    top_ins_sorted = [top_ins[i] for i in cluster_order]
    in_labels_sorted = [in_labels[i] for i in cluster_order]
    in_clusters_sorted = [in_clusters[i] for i in cluster_order]
    conn_matrix_sorted = conn_matrix[:, cluster_order]
    shrec_matrix_sorted = shrec_matrix[:, cluster_order]
    
    print(f"  DNs labeled: {len(dn_labels)}")
    print(f"  INs labeled: {len(in_labels)}")
    
    # Create figure
    print("\nCreating heatmap figure...")
    
    fig, axes = plt.subplots(2, 1, figsize=(24, 18), 
                            gridspec_kw={'height_ratios': [1, 1]},
                            dpi=300)
    
    # =================================================================
    # PANEL 1: Synapse count heatmap
    # =================================================================
    
    ax = axes[0]
    
    # Plot heatmap
    # Use log scale for better visualization
    plot_data = np.log10(conn_matrix_sorted + 1)
    
    im = ax.imshow(plot_data, aspect='auto', cmap='YlOrRd', 
                   interpolation='nearest')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('log10(Synapses + 1)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add cluster dividers
    cluster_changes = []
    for i in range(1, len(in_clusters_sorted)):
        if in_clusters_sorted[i] != in_clusters_sorted[i-1]:
            cluster_changes.append(i - 0.5)
    
    for change_pos in cluster_changes:
        ax.axvline(change_pos, color='black', linewidth=2, linestyle='--', alpha=0.7)
    
    # Labels
    ax.set_xticks(range(len(in_labels_sorted)))
    ax.set_xticklabels(in_labels_sorted, rotation=90, fontsize=8, ha='center')
    
    ax.set_yticks(range(len(dn_labels)))
    ax.set_yticklabels(dn_labels, fontsize=8)
    
    ax.set_xlabel('Premotor Interneurons (sorted by cluster)', 
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Descending Neurons', fontsize=13, fontweight='bold')
    ax.set_title('A. DN→IN Connectivity Matrix (Synapse Count)\nShREC ≥0.4%',
                fontsize=14, fontweight='bold', pad=15)
    
    # Add grid
    ax.set_xticks(np.arange(len(in_labels_sorted)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(dn_labels)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
    
    # =================================================================
    # PANEL 2: ShREC score heatmap
    # =================================================================
    
    ax = axes[1]
    
    # Plot ShREC heatmap
    im2 = ax.imshow(shrec_matrix_sorted, aspect='auto', cmap='viridis',
                    interpolation='nearest', vmin=0, vmax=5)
    
    # Colorbar
    cbar2 = plt.colorbar(im2, ax=ax, fraction=0.02, pad=0.02)
    cbar2.set_label('ShREC Score (%)', fontsize=12, fontweight='bold')
    cbar2.ax.tick_params(labelsize=10)
    
    # Add cluster dividers
    for change_pos in cluster_changes:
        ax.axvline(change_pos, color='white', linewidth=2, linestyle='--', alpha=0.7)
    
    # Labels
    ax.set_xticks(range(len(in_labels_sorted)))
    ax.set_xticklabels(in_labels_sorted, rotation=90, fontsize=8, ha='center')
    
    ax.set_yticks(range(len(dn_labels)))
    ax.set_yticklabels(dn_labels, fontsize=8)
    
    ax.set_xlabel('Premotor Interneurons (sorted by cluster)', 
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Descending Neurons', fontsize=13, fontweight='bold')
    ax.set_title('B. DN→IN Functional Connectivity (ShREC Score)\n% of Target IN Total Input',
                fontsize=14, fontweight='bold', pad=15)
    
    # Add grid
    ax.set_xticks(np.arange(len(in_labels_sorted)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(dn_labels)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'DN→IN Connectivity Matrix (Top {top_n_dns} DNs × Top {top_n_ins} INs)',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save
    if output_dir is None:
        output_dir = Path('results/shrec_complete_with_dns')
    
    plt.savefig(output_dir / 'dn_in_heatmap_shrec.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    try:
        plt.savefig(output_dir / 'dn_in_heatmap_shrec.pdf',
                    format='pdf', bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved: dn_in_heatmap_shrec.png/pdf")
    except PermissionError:
        print(f"\n✓ Saved: dn_in_heatmap_shrec.png (PDF skipped — close the file first)")
    
    plt.close()
    
    # Create summary statistics
    print("\nHeatmap Statistics:")
    print(f"  Matrix dimensions: {len(top_dns)} DNs × {len(top_ins)} INs")
    print(f"  Non-zero connections: {np.count_nonzero(conn_matrix_sorted)}")
    print(f"  Connection density: {np.count_nonzero(conn_matrix_sorted) / conn_matrix_sorted.size * 100:.1f}%")
    print(f"  Average synapses (non-zero): {conn_matrix_sorted[conn_matrix_sorted > 0].mean():.1f}")
    print(f"  Max synapses: {conn_matrix_sorted.max():.0f}")
    print(f"  Average ShREC (non-zero): {shrec_matrix_sorted[shrec_matrix_sorted > 0].mean():.3f}%")
    print(f"  Max ShREC: {shrec_matrix_sorted.max():.3f}%")
    
    return conn_matrix_sorted, shrec_matrix_sorted, top_dns, top_ins_sorted


def create_focused_heatmap(dn_in_conn, cluster_assignments, dn_classifications,
                           loader, output_dir=None):
    """
    Create a more focused heatmap showing power vs steering specialization.
    
    Rows: Top power DNs and top steering DNs (separate sections)
    Cols: INs from each cluster
    """
    
    print("\n" + "="*70)
    print("CREATING SPECIALIZED DN→IN HEATMAP")
    print("="*70 + "\n")
    
    # Separate power and steering DNs
    power_dns = dn_classifications[
        dn_classifications['specialization'] == 'power_control'
    ].nlargest(15, 'total_synapses')['dn_id'].tolist()
    
    steering_dns = dn_classifications[
        dn_classifications['specialization'] == 'steering_control'
    ].nlargest(15, 'total_synapses')['dn_id'].tolist()
    
    print(f"Power DNs: {len(power_dns)}")
    print(f"Steering DNs: {len(steering_dns)}")
    
    # Select top INs from each cluster
    ins_per_cluster = 10
    top_ins_by_cluster = []
    
    # Group INs by cluster
    cluster_ins = {}
    for in_id, cluster in cluster_assignments.items():
        if cluster not in cluster_ins:
            cluster_ins[cluster] = []
        cluster_ins[cluster].append(in_id)
    
    # For each cluster, get top INs by DN input
    for cluster in sorted(cluster_ins.keys()):
        cluster_in_ids = cluster_ins[cluster]
        
        # Get total DN input for each IN
        in_totals = dn_in_conn[
            dn_in_conn['target'].isin(cluster_in_ids)
        ].groupby('target')['weight'].sum().sort_values(ascending=False)
        
        # Take top INs from this cluster
        top_cluster_ins = in_totals.head(ins_per_cluster).index.tolist()
        top_ins_by_cluster.extend(top_cluster_ins)
    
    print(f"\nSelected {len(top_ins_by_cluster)} INs across {len(cluster_ins)} clusters")
    
    # Combine DNs: power first, then steering
    all_dns = power_dns + steering_dns
    
    # Build matrix
    conn_matrix = np.zeros((len(all_dns), len(top_ins_by_cluster)))
    
    for i, dn_id in enumerate(all_dns):
        for j, in_id in enumerate(top_ins_by_cluster):
            connections = dn_in_conn[
                (dn_in_conn['source'] == dn_id) &
                (dn_in_conn['target'] == in_id)
            ]
            if len(connections) > 0:
                conn_matrix[i, j] = connections['weight'].values[0]
    
    # Create labels
    dn_labels = []
    dn_types = []
    
    for dn_id in all_dns:
        name = loader.get_neuron_name(dn_id)
        dn_type = 'Power' if dn_id in power_dns else 'Steering'
        dn_types.append(dn_type)
        name_str = str(name)[:32] if name != str(dn_id) else f"DN_{dn_id}"
        label = f"{name_str}\n[{dn_type}]"
        dn_labels.append(label)

    in_labels = []
    in_cluster_list = []
    for in_id in top_ins_by_cluster:
        name = loader.get_neuron_name(in_id)
        cluster = cluster_assignments.get(in_id, 0)
        in_cluster_list.append(cluster)
        name_str = str(name)[:28] if name != str(in_id) else f"IN_{in_id}"
        label = f"{name_str}\nC{cluster}"
        in_labels.append(label)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(28, 16), dpi=300)
    
    # Plot heatmap (log scale)
    plot_data = np.log10(conn_matrix + 1)
    im = ax.imshow(plot_data, aspect='auto', cmap='RdYlBu_r',
                   interpolation='nearest')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
    cbar.set_label('log10(Synapses + 1)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Add divider between power and steering DNs
    power_steering_divider = len(power_dns) - 0.5
    ax.axhline(power_steering_divider, color='black', linewidth=3)
    
    # Add cluster dividers
    cluster_changes = []
    for i in range(1, len(in_cluster_list)):
        if in_cluster_list[i] != in_cluster_list[i-1]:
            cluster_changes.append(i - 0.5)
    
    for change_pos in cluster_changes:
        ax.axvline(change_pos, color='black', linewidth=2, linestyle='--', alpha=0.7)
    
    # Labels
    ax.set_xticks(range(len(in_labels)))
    ax.set_xticklabels(in_labels, rotation=90, fontsize=7, ha='center')
    
    ax.set_yticks(range(len(dn_labels)))
    ax.set_yticklabels(dn_labels, fontsize=8)
    
    # Color-code DN labels
    for i, (label_obj, dn_type) in enumerate(zip(ax.get_yticklabels(), dn_types)):
        if dn_type == 'Power':
            label_obj.set_color('#FF6B6B')
            label_obj.set_fontweight('bold')
        else:
            label_obj.set_color('#4ECDC4')
            label_obj.set_fontweight('bold')
    
    ax.set_xlabel('Premotor Interneurons (grouped by cluster)', 
                  fontsize=14, fontweight='bold')
    ax.set_ylabel('Descending Neurons (Power | Steering)', 
                  fontsize=14, fontweight='bold')
    ax.set_title('DN→IN Connectivity: Power vs Steering Specialization\n' +
                f'Top {len(power_dns)} Power DNs & Top {len(steering_dns)} Steering DNs × ' +
                f'Top {ins_per_cluster} INs per Cluster (ShREC ≥0.4%)',
                fontsize=15, fontweight='bold', pad=20)
    
    # Add text annotations for DN sections
    ax.text(-1, len(power_dns)/2, 'POWER\nDNs', 
           fontsize=12, fontweight='bold', color='#FF6B6B',
           ha='right', va='center', rotation=90)
    
    ax.text(-1, len(power_dns) + len(steering_dns)/2, 'STEERING\nDNs',
           fontsize=12, fontweight='bold', color='#4ECDC4',
           ha='right', va='center', rotation=90)
    
    # Grid
    ax.set_xticks(np.arange(len(in_labels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(dn_labels)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if output_dir is None:
        output_dir = Path('results/shrec_complete_with_dns')
    
    plt.savefig(output_dir / 'dn_in_specialized_heatmap.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    try:
        plt.savefig(output_dir / 'dn_in_specialized_heatmap.pdf',
                    format='pdf', bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved: dn_in_specialized_heatmap.png/pdf")
    except PermissionError:
        print(f"\n✓ Saved: dn_in_specialized_heatmap.png (PDF skipped — close the file first)")
    
    plt.close()


def main():
    print("\n" + "="*70)
    print("DN→IN HEATMAP VISUALIZATION (ShREC)")
    print("="*70 + "\n")
    
    # Setup
    output_dir = Path('results/shrec_complete_with_dns')
    
    if not output_dir.exists():
        print("ERROR: ShREC analysis results not found!")
        print(f"Please run 'complete_shrec_with_dns.py' first.")
        return
    
    # Load data
    print("Loading ShREC analysis results...")
    loader = ConnectomeDataLoader('config.yaml')
    loader.load_all_data(verbose=False)
    
    dn_in_conn = pd.read_csv(output_dir / 'dn_to_in_connections_shrec.csv')
    dn_classifications = pd.read_csv(output_dir / 'dn_classifications.csv')
    
    # Load cluster assignments
    # We need to reconstruct this from the premotor connections
    premotor_conn = pd.read_csv(output_dir / 'in_to_mn_connections_shrec.csv')
    premotor_in_ids = premotor_conn['source'].unique()
    
    print(f"✓ Loaded {len(dn_in_conn):,} DN→IN connections")
    print(f"✓ Loaded {len(dn_classifications)} DN classifications")
    print(f"✓ Found {len(premotor_in_ids):,} premotor INs")
    
    # We need to re-run clustering to get cluster assignments
    # Or load from saved results if available
    print("\nNote: Cluster assignments needed for heatmap.")
    print("If cluster_assignments.csv exists, loading...")
    
    cluster_file = output_dir / 'cluster_assignments.csv'
    if cluster_file.exists():
        cluster_df = pd.read_csv(cluster_file)
        cluster_assignments = dict(zip(
            cluster_df['interneuron_id'], 
            cluster_df['cluster']
        ))
        print(f"✓ Loaded cluster assignments for {len(cluster_assignments)} INs")
    else:
        print("\nWARNING: cluster_assignments.csv not found.")
        print("Creating dummy cluster assignments based on connectivity patterns...")
        
        # Quick clustering based on DN input patterns
        from sklearn.cluster import KMeans
        
        # Build simple feature matrix
        in_dn_features = {}
        for in_id in premotor_in_ids:
            in_connections = dn_in_conn[dn_in_conn['target'] == in_id]
            total_syn = in_connections['weight'].sum()
            n_dns = len(in_connections)
            avg_shrec = in_connections['shrec_score'].mean() if len(in_connections) > 0 else 0
            in_dn_features[in_id] = [total_syn, n_dns, avg_shrec]
        
        # Cluster
        feature_matrix = np.array([in_dn_features[in_id] for in_id in premotor_in_ids])
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(feature_matrix)
        
        cluster_assignments = dict(zip(premotor_in_ids, clusters + 1))
        
        # Save for future use
        cluster_df = pd.DataFrame([
            {'interneuron_id': in_id, 'cluster': cluster}
            for in_id, cluster in cluster_assignments.items()
        ])
        cluster_df.to_csv(cluster_file, index=False)
        print(f"✓ Created and saved cluster assignments")
    
    # Create heatmaps
    print("\n" + "="*70)
    print("Creating heatmap visualizations...")
    print("="*70)
    
    # Heatmap 1: Top DNs × Top INs
    create_dn_in_heatmap(
        dn_in_conn,
        cluster_assignments,
        loader,
        top_n_dns=15,
        top_n_ins=25,
        output_dir=output_dir
    )
    
    # Heatmap 2: Power vs Steering specialization
    create_focused_heatmap(
        dn_in_conn,
        cluster_assignments,
        dn_classifications,
        loader,
        output_dir=output_dir
    )
    
    print("\n" + "="*70)
    print("HEATMAP CREATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nFiles created:")
    print("  • dn_in_heatmap_shrec.png/pdf")
    print("  • dn_in_specialized_heatmap.png/pdf")
    print("  • cluster_assignments.csv (if created)")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()