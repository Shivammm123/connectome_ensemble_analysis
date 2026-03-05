"""
Network Visualization - Improved Clean Version

Creates clear, publication-quality network visualizations.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def plot_hierarchical_network(
    query_id: int,
    similar_neurons: pd.DataFrame,
    connections: pd.DataFrame,
    loader: ConnectomeDataLoader,
    output_file: Path,
    top_n: int = 5
):
    """
    Create hierarchical network: Query → Similar → Top Targets
    
    Much cleaner layout with 3 tiers.
    """
    fig, ax = plt.subplots(figsize=(18, 12), dpi=300)
    
    # Get data
    query_name = loader.get_neuron_name(query_id)
    top_similar = similar_neurons.head(top_n)
    similar_ids = top_similar['neuron_id'].tolist()
    
    # Get connections from query and similar neurons
    all_sources = [query_id] + similar_ids
    source_conn = connections[connections['source'].isin(all_sources)]
    
    # Find top shared targets (targets that multiple neurons connect to)
    target_counts = source_conn.groupby('target')['source'].nunique()
    shared_targets = target_counts[target_counts >= 2].nlargest(10).index.tolist()
    
    # Manual hierarchical layout positions
    positions = {}
    
    # Layer 1: Query (top center)
    positions[query_id] = (0.5, 0.9)
    
    # Layer 2: Similar neurons (middle, spread out)
    similar_y = 0.5
    for i, nid in enumerate(similar_ids):
        x = 0.2 + (i / max(len(similar_ids)-1, 1)) * 0.6
        positions[nid] = (x, similar_y)
    
    # Layer 3: Shared targets (bottom, spread out)
    target_y = 0.1
    for i, tid in enumerate(shared_targets):
        x = 0.15 + (i / max(len(shared_targets)-1, 1)) * 0.7
        positions[tid] = (x, target_y)
    
    # Draw connections (edges)
    for _, row in source_conn.iterrows():
        source = row['source']
        target = row['target']
        weight = row['weight']
        
        if source in positions and target in positions:
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            
            # Line width based on synapse count
            line_width = 0.5 + (weight / 50) * 4
            
            # Color based on source
            if source == query_id:
                color = '#FF6B6B'  # Red for query
                alpha = 0.6
            else:
                color = '#4ECDC4'  # Teal for similar
                alpha = 0.3
            
            ax.plot([x1, x2], [y1, y2], 
                   color=color, linewidth=line_width, alpha=alpha, zorder=1)
            
            # Add arrowhead
            dx = x2 - x1
            dy = y2 - y1
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=line_width*0.5,
                                     color=color, alpha=alpha))
    
    # Draw nodes
    # Query node (large, red square)
    qx, qy = positions[query_id]
    query_patch = mpatches.FancyBboxPatch((qx-0.04, qy-0.025), 0.08, 0.05,
                                         boxstyle="round,pad=0.01", 
                                         facecolor='#FF6B6B', edgecolor='black',
                                         linewidth=3, zorder=10)
    ax.add_patch(query_patch)
    ax.text(qx, qy, query_name[:15], ha='center', va='center',
           fontsize=11, fontweight='bold', zorder=11)
    ax.text(qx, qy+0.06, 'QUERY', ha='center', va='bottom',
           fontsize=9, style='italic', color='#FF6B6B', fontweight='bold')
    
    # Similar neurons (medium, colored by similarity)
    for _, row in top_similar.iterrows():
        nid = row['neuron_id']
        name = loader.get_neuron_name(nid)
        sim = row['similarity']
        
        x, y = positions[nid]
        
        # Color gradient based on similarity
        color = plt.cm.viridis(sim)
        
        circle = plt.Circle((x, y), 0.025, facecolor=color, edgecolor='black',
                          linewidth=2, zorder=10)
        ax.add_patch(circle)
        
        # Label
        ax.text(x, y, name[:12], ha='center', va='center',
               fontsize=9, fontweight='bold', zorder=11)
        ax.text(x, y-0.045, f'sim: {sim:.2f}', ha='center', va='top',
               fontsize=7, style='italic', color='gray')
    
    # Target neurons (small, coral)
    for tid in shared_targets:
        if tid in positions:
            name = loader.get_neuron_name(tid)
            x, y = positions[tid]
            
            circle = plt.Circle((x, y), 0.018, facecolor='#FFA07A', 
                              edgecolor='black', linewidth=1, zorder=10, alpha=0.8)
            ax.add_patch(circle)
            
            ax.text(x, y, name[:10], ha='center', va='center',
                   fontsize=7, zorder=11)
    
    # Title and labels
    ax.text(0.5, 0.98, 'Hierarchical Connectivity Network',
           ha='center', va='top', fontsize=18, fontweight='bold',
           transform=ax.transAxes)
    
    ax.text(0.02, 0.9, 'Query\nNeuron', ha='left', va='center',
           fontsize=10, style='italic', color='#FF6B6B', fontweight='bold')
    ax.text(0.02, 0.5, 'Similar\nNeurons', ha='left', va='center',
           fontsize=10, style='italic', color='#4ECDC4', fontweight='bold')
    ax.text(0.02, 0.1, 'Shared\nTargets', ha='left', va='center',
           fontsize=10, style='italic', color='#FFA07A', fontweight='bold')
    
    # Add legend for edge thickness
    legend_x = 0.85
    legend_y = 0.95
    ax.text(legend_x, legend_y, 'Edge thickness =\nsynapse count',
           ha='left', va='top', fontsize=9, style='italic',
           transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Colorbar for similarity
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, aspect=30)
    cbar.set_label('Similarity Score', fontsize=11, fontweight='bold')
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file.name}")


def plot_top3_focused(
    query_id: int,
    similar_neurons: pd.DataFrame,
    connections: pd.DataFrame,
    loader: ConnectomeDataLoader,
    output_file: Path
):
    """
    Focus on just top 3 similar neurons - very clean.
    """
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    
    query_name = loader.get_neuron_name(query_id)
    top3 = similar_neurons.head(3)
    
    # Positions - radial layout
    positions = {query_id: (0.5, 0.5)}  # Center
    
    for i, (_, row) in enumerate(top3.iterrows()):
        angle = (i / 3) * 2 * np.pi - np.pi/2  # Start from top
        radius = 0.3
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        positions[row['neuron_id']] = (x, y)
    
    # Get connections between these neurons only
    all_neurons = [query_id] + top3['neuron_id'].tolist()
    subset_conn = connections[
        connections['source'].isin(all_neurons) &
        connections['target'].isin(all_neurons)
    ]
    
    # Draw edges
    for _, row in subset_conn.iterrows():
        if row['source'] in positions and row['target'] in positions:
            x1, y1 = positions[row['source']]
            x2, y2 = positions[row['target']]
            weight = row['weight']
            
            line_width = 1 + (weight / 20) * 5
            
            ax.plot([x1, x2], [y1, y2], 
                   color='#888', linewidth=line_width, alpha=0.5, zorder=1)
            
            # Add synapse count label
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            ax.text(mid_x, mid_y, f'{int(weight)}', 
                   ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.2),
                   zorder=5)
    
    # Draw query node (large)
    qx, qy = positions[query_id]
    query_circle = plt.Circle((qx, qy), 0.08, facecolor='#FF6B6B', 
                             edgecolor='black', linewidth=4, zorder=10)
    ax.add_patch(query_circle)
    ax.text(qx, qy, query_name[:15], ha='center', va='center',
           fontsize=13, fontweight='bold', zorder=11, color='white')
    ax.text(qx, qy-0.12, 'QUERY NEURON', ha='center', va='top',
           fontsize=10, style='italic', fontweight='bold', color='#FF6B6B')
    
    # Draw similar neurons
    for idx, (_, row) in enumerate(top3.iterrows()):
        nid = row['neuron_id']
        name = loader.get_neuron_name(nid)
        sim = row['similarity']
        
        x, y = positions[nid]
        
        color = plt.cm.viridis(sim)
        circle = plt.Circle((x, y), 0.06, facecolor=color, 
                          edgecolor='black', linewidth=3, zorder=10)
        ax.add_patch(circle)
        
        ax.text(x, y, f'{name[:12]}', ha='center', va='center',
               fontsize=11, fontweight='bold', zorder=11)
        ax.text(x, y-0.095, f'Similarity: {sim:.3f}', ha='center', va='top',
               fontsize=9, style='italic', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.3))
    
    ax.set_title('Top 3 Most Similar Neurons\n(Numbers on edges = synapse count)',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file.name}")


def main():
    print("\n" + "="*70)
    print("IMPROVED NETWORK VISUALIZATION")
    print("="*70 + "\n")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize loader
    loader = ConnectomeDataLoader('config.yaml')
    connections, neurons, name_mapping = loader.load_all_data(verbose=False)
    print("✓ Data loaded\n")
    
    # Get query neuron
    query_id = config['single_query'].get('neuron_id')
    query_name = loader.get_neuron_name(query_id)
    print(f"Query neuron: {query_name} ({query_id})\n")
    
    # Load results
    results_file = Path('results/single_neuron/similarity_results.csv')
    if not results_file.exists():
        print("❌ Run single_neuron_analysis.py first!")
        sys.exit(1)
    
    similar_neurons = pd.read_csv(results_file)
    
    # Filter connections
    min_syn = config['analysis']['min_synapses']
    filtered_conn = loader.filter_connections(min_synapses=min_syn, verbose=False)
    
    # Create output directory
    output_dir = Path('results/networks')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("-"*70)
    print("Creating visualizations...")
    print("-"*70 + "\n")
    
    # Version 1: Hierarchical (cleaner!)
    plot_hierarchical_network(
        query_id,
        similar_neurons,
        filtered_conn,
        loader,
        output_dir / 'network_hierarchical.png',
        top_n=5
    )
    
    # Version 2: Top 3 focus (simplest!)
    plot_top3_focused(
        query_id,
        similar_neurons,
        filtered_conn,
        loader,
        output_dir / 'network_top3_focus.png'
    )
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nCheck {output_dir} for:")
    print("  • network_hierarchical.png - Clean 3-layer view")
    print("  • network_top3_focus.png - Just top 3 neurons")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()