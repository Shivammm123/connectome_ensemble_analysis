"""
Sankey Flow Diagram

Creates beautiful flow diagram showing connectivity:
Query Neuron → Similar Neurons → Shared Targets

Flow thickness represents synapse count.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("ERROR: plotly required for Sankey diagrams")
    print("Install with: pip install plotly")
    sys.exit(1)

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def create_sankey_diagram(
    query_id: int,
    similar_neurons: pd.DataFrame,
    connections: pd.DataFrame,
    loader: ConnectomeDataLoader,
    output_file: Path,
    top_similar: int = 5,
    top_targets: int = 15,
    min_synapses: int = 5
):
    """
    Create Sankey flow diagram.
    
    Parameters
    ----------
    query_id : int
        Query neuron
    similar_neurons : pd.DataFrame
        Similar neurons with scores
    connections : pd.DataFrame
        Connectivity data
    loader : ConnectomeDataLoader
        Data loader
    output_file : Path
        Output HTML file
    top_similar : int
        Number of similar neurons to show
    top_targets : int
        Number of target neurons to show
    min_synapses : int
        Minimum synapses for flow
    """
    
    # Get top similar neurons
    top_sim = similar_neurons.head(top_similar)
    similar_ids = top_sim['neuron_id'].tolist()
    
    # Get connections
    query_conn = connections[connections['source'] == query_id]
    similar_conn = connections[connections['source'].isin(similar_ids)]
    
    # Find shared targets (targets that multiple neurons connect to)
    all_targets = pd.concat([query_conn['target'], similar_conn['target']])
    target_counts = all_targets.value_counts()
    shared_targets = target_counts[target_counts >= 2].index.tolist()
    
    # Limit to top targets by total synapse count
    if len(shared_targets) > top_targets:
        target_synapses = pd.concat([query_conn, similar_conn])
        target_synapses = target_synapses[target_synapses['target'].isin(shared_targets)]
        target_totals = target_synapses.groupby('target')['weight'].sum()
        top_target_ids = target_totals.nlargest(top_targets).index.tolist()
        shared_targets = top_target_ids
    
    # Build node lists
    # Layer 0: Query neuron
    # Layer 1: Similar neurons
    # Layer 2: Target neurons
    
    nodes = []
    node_colors = []
    
    # Query neuron (red)
    query_name = loader.get_neuron_name(query_id)
    nodes.append(f"QUERY: {query_name[:20]}")
    node_colors.append('#FF6B6B')
    query_idx = 0
    
    # Similar neurons (gradient by similarity)
    similar_start_idx = len(nodes)
    for _, row in top_sim.iterrows():
        name = loader.get_neuron_name(row['neuron_id'])
        sim = row['similarity']
        nodes.append(f"{name[:20]} ({sim:.2f})")
        # Color gradient from teal to yellow
        color = plt.cm.viridis(sim)
        node_colors.append(f"rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},0.8)")
    
    # Target neurons (coral)
    target_start_idx = len(nodes)
    target_id_to_idx = {}
    for tid in shared_targets:
        name = loader.get_neuron_name(tid)
        nodes.append(f"{name[:20]}")
        node_colors.append('#FFA07A')
        target_id_to_idx[tid] = len(nodes) - 1
    
    # Build flows
    sources = []
    targets = []
    values = []
    link_colors = []
    
    # Query → Similar neurons (show similarity as flow)
    for i, (_, row) in enumerate(top_sim.iterrows()):
        sources.append(query_idx)
        targets.append(similar_start_idx + i)
        # Use similarity * 100 as flow width
        values.append(row['similarity'] * 100)
        link_colors.append('rgba(255, 107, 107, 0.3)')  # Red transparent
    
    # Similar neurons → Targets (synapse count)
    for _, row in similar_conn.iterrows():
        source_neuron = row['source']
        target_neuron = row['target']
        weight = row['weight']
        
        if target_neuron in target_id_to_idx and weight >= min_synapses:
            # Find index of source neuron
            source_idx = similar_start_idx + similar_ids.index(source_neuron)
            target_idx = target_id_to_idx[target_neuron]
            
            sources.append(source_idx)
            targets.append(target_idx)
            values.append(weight)
            link_colors.append('rgba(78, 205, 196, 0.4)')  # Teal transparent
    
    # Query → Targets (direct connections)
    for _, row in query_conn.iterrows():
        target_neuron = row['target']
        weight = row['weight']
        
        if target_neuron in target_id_to_idx and weight >= min_synapses:
            target_idx = target_id_to_idx[target_neuron]
            
            sources.append(query_idx)
            targets.append(target_idx)
            values.append(weight)
            link_colors.append('rgba(255, 107, 107, 0.2)')  # Red transparent
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors,
            customdata=[f"Node: {n}" for n in nodes],
            hovertemplate='%{customdata}<br>Connections: %{value}<extra></extra>'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate='%{source.label} → %{target.label}<br>Synapses: %{value}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title=dict(
            text=f"Connectivity Flow: {query_name}<br><sub>Flow thickness = synapse count</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        font=dict(size=12),
        height=800,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.write_html(output_file)
    print(f"  ✓ Saved: {output_file}")
    print(f"    Open in browser to interact!")


def main():
    print("\n" + "="*70)
    print("SANKEY FLOW DIAGRAM")
    print("="*70 + "\n")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize loader
    loader = ConnectomeDataLoader('config.yaml')
    connections, neurons, name_mapping = loader.load_all_data(verbose=False)
    print("✓ Data loaded\n")
    
    # Get query neuron (neuron_id can be a name string or numeric root ID)
    raw = config['single_query'].get('neuron_id')
    try:
        query_id = int(raw)
    except (ValueError, TypeError):
        query_id = loader.get_neuron_id(str(raw))
        if query_id is None:
            print(f"❌ Could not find neuron: {raw}")
            sys.exit(1)
    query_name = loader.get_neuron_name(query_id)
    print(f"Query neuron: {query_name} ({query_id})\n")
    
    # Load similarity results
    results_file = Path('results/single_neuron/similarity_results.csv')
    if not results_file.exists():
        print("❌ Run single_neuron_analysis.py first!")
        sys.exit(1)
    
    similar_neurons = pd.read_csv(results_file)
    print(f"Loaded {len(similar_neurons)} similar neurons\n")
    
    # Filter connections
    min_syn = config['analysis']['min_synapses']
    filtered_conn = loader.filter_connections(min_synapses=min_syn, verbose=False)
    
    # Create output directory
    output_dir = Path('results/networks')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Sankey diagram
    print("-"*70)
    print("Creating Sankey flow diagram...")
    print("-"*70 + "\n")
    
    create_sankey_diagram(
        query_id,
        similar_neurons,
        filtered_conn,
        loader,
        output_dir / 'connectivity_flow_sankey.html',
        top_similar=5,
        top_targets=15,
        min_synapses=min_syn
    )
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nOpen in browser:")
    print(f"  {output_dir / 'connectivity_flow_sankey.html'}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()