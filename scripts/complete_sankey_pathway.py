"""
Complete Pathway Sankey Diagrams

Creates beautiful flow diagrams showing complete pathways:
Brain → DN → IN Cluster → MN Pool → Muscle

Multiple visualizations for different perspectives.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("ERROR: plotly required")
    print("Install: pip install plotly")
    sys.exit(1)

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def create_full_pathway_sankey(
    dn_class_df: pd.DataFrame,
    cluster_info: pd.DataFrame,
    motor_pools: pd.DataFrame,
    loader: ConnectomeDataLoader,
    output_file: Path,
    top_n_dns: int = 15
):
    """
    Create comprehensive Sankey: DN → Cluster → Motor Pool → Muscle Type
    """
    
    print("Creating complete pathway Sankey diagram...")
    
    # Build node list
    nodes = []
    node_colors = []
    
    # Layer 1: Top DNs
    top_dns = dn_class_df.head(top_n_dns)
    dn_start = 0
    
    for _, dn in top_dns.iterrows():
        nodes.append(f"DN: {dn['dn_name'][:20]}")
        # Color by specialization
        if dn['specialization'] == 'power_control':
            node_colors.append('#FF6B6B')
        elif dn['specialization'] == 'steering_control':
            node_colors.append('#4ECDC4')
        else:
            node_colors.append('#CCCCCC')
    
    # Layer 2: IN Clusters
    cluster_start = len(nodes)
    
    for _, cluster in cluster_info.iterrows():
        nodes.append(f"Cluster {cluster['cluster_id']}\n({cluster['functional_type'].split('_')[0]})")
        if cluster['functional_type'] == 'power_control':
            node_colors.append('#FF6B6B')
        else:
            node_colors.append('#4ECDC4')
    
    # Layer 3: Motor Pools
    pool_start = len(nodes)
    
    for _, pool in motor_pools.iterrows():
        nodes.append(f"{pool['muscle']}")
        if pool['muscle_type'] == 'power':
            node_colors.append('#FFA07A')
        else:
            node_colors.append('#95E1D3')
    
    # Layer 4: Muscle Types
    muscle_type_start = len(nodes)
    nodes.extend(['Power Muscles', 'Steering Muscles'])
    node_colors.extend(['#FF6B6B', '#4ECDC4'])
    
    # Build links
    sources = []
    targets = []
    values = []
    link_colors = []
    
    # DN → Cluster links
    for i, (_, dn) in enumerate(top_dns.iterrows()):
        primary_cluster = dn['primary_cluster']
        cluster_idx = cluster_info[cluster_info['cluster_id'] == primary_cluster].index[0]
        
        sources.append(dn_start + i)
        targets.append(cluster_start + cluster_idx)
        values.append(dn['total_connectivity'])
        link_colors.append('rgba(78, 205, 196, 0.3)')
    
    # Cluster → Motor Pool links (simplified - based on primary targets)
    for i, (_, cluster) in enumerate(cluster_info.iterrows()):
        primary_targets = cluster['primary_targets'].split(', ')
        
        for target in primary_targets[:3]:  # Top 3 targets
            pool_match = motor_pools[motor_pools['muscle'] == target]
            if len(pool_match) > 0:
                pool_idx = pool_match.index[0]
                
                sources.append(cluster_start + i)
                targets.append(pool_start + pool_idx)
                values.append(cluster['total_connectivity'] / len(primary_targets))
                link_colors.append('rgba(149, 225, 211, 0.3)')
    
    # Motor Pool → Muscle Type links
    for i, (_, pool) in enumerate(motor_pools.iterrows()):
        if pool['muscle_type'] == 'power':
            target_idx = muscle_type_start
        else:
            target_idx = muscle_type_start + 1
        
        sources.append(pool_start + i)
        targets.append(target_idx)
        values.append(pool['n_motor_neurons'] * 10)  # Scale for visibility
        link_colors.append('rgba(255, 160, 122, 0.3)')
    
    # Create Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors,
            customdata=[f"Node: {n}" for n in nodes],
            hovertemplate='%{customdata}<br>Flow: %{value}<extra></extra>'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate='%{source.label} → %{target.label}<br>Strength: %{value:.0f}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title=dict(
            text="Complete Flight Control Pathway<br><sub>Brain → DN → IN Cluster → Motor Pool → Muscle</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=20, family='Arial Black')
        ),
        font=dict(size=12),
        height=900,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.write_html(output_file)
    print(f"  ✓ Saved: {output_file}")


def create_power_vs_steering_sankey(
    cluster_info: pd.DataFrame,
    motor_pools: pd.DataFrame,
    output_file: Path
):
    """
    Simplified Sankey showing power vs steering pathways.
    """
    
    print("Creating power vs steering comparison Sankey...")
    
    # Nodes
    nodes = [
        "Brain Commands",
        "Power IN Cluster",
        "Steering IN Clusters",
        "Power Motor Pools",
        "Steering Motor Pools",
        "Power Muscles",
        "Steering Muscles"
    ]
    
    node_colors = [
        '#FFFFFF',  # Brain
        '#FF6B6B',  # Power cluster
        '#4ECDC4',  # Steering clusters
        '#FFA07A',  # Power pools
        '#95E1D3',  # Steering pools
        '#FF6B6B',  # Power muscles
        '#4ECDC4'   # Steering muscles
    ]
    
    # Get totals
    power_cluster = cluster_info[cluster_info['functional_type'] == 'power_control']
    steering_clusters = cluster_info[cluster_info['functional_type'] == 'steering_control']
    
    power_pools = motor_pools[motor_pools['muscle_type'] == 'power']
    steering_pools = motor_pools[motor_pools['muscle_type'] == 'steering']
    
    power_total = power_cluster['total_connectivity'].sum()
    steering_total = steering_clusters['total_connectivity'].sum()
    
    # Links
    sources = [0, 0, 1, 2, 3, 4]
    targets = [1, 2, 3, 4, 5, 6]
    values = [
        power_total,
        steering_total,
        power_total,
        steering_total,
        len(power_pools) * 100,
        len(steering_pools) * 100
    ]
    
    link_colors = [
        'rgba(255, 107, 107, 0.4)',
        'rgba(78, 205, 196, 0.4)',
        'rgba(255, 160, 122, 0.4)',
        'rgba(149, 225, 211, 0.4)',
        'rgba(255, 107, 107, 0.4)',
        'rgba(78, 205, 196, 0.4)'
    ]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=1),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])
    
    fig.update_layout(
        title=dict(
            text="Power vs Steering Control Pathways<br><sub>Comparative Flow</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        font=dict(size=14),
        height=700,
        plot_bgcolor='white'
    )
    
    fig.write_html(output_file)
    print(f"  ✓ Saved: {output_file}")


def main():
    print("\n" + "="*70)
    print("COMPLETE PATHWAY SANKEY DIAGRAMS")
    print("="*70 + "\n")
    
    # Load results
    print("Loading data...")
    print("-"*70 + "\n")
    
    loader = ConnectomeDataLoader('config.yaml')
    connections, neurons, name_mapping = loader.load_all_data(verbose=False)
    
    # DN classifications
    dn_class_df = pd.read_csv('results/dn_pathways/dn_classifications.csv')
    print(f"  ✓ DN classifications: {len(dn_class_df)}")
    
    # Cluster info
    cluster_info = pd.read_csv('results/interneuron_clusters/cluster_characteristics.csv')
    print(f"  ✓ Clusters: {len(cluster_info)}")
    
    # Motor pools
    motor_pools = pd.read_csv('results/motor_neurons/motor_pools.csv')
    motor_pools['motor_neuron_ids'] = motor_pools['motor_neuron_ids'].apply(eval)
    print(f"  ✓ Motor pools: {len(motor_pools)}\n")
    
    # Create output directory
    output_dir = Path('results/pathway_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Sankeydiagrams
    print("-"*70)
    print("Creating Sankey diagrams...")
    print("-"*70 + "\n")
    
    # Full pathway
    create_full_pathway_sankey(
        dn_class_df,
        cluster_info,
        motor_pools,
        loader,
        output_dir / 'complete_pathway_sankey.html',
        top_n_dns=15
    )
    
    # Power vs Steering comparison
    create_power_vs_steering_sankey(
        cluster_info,
        motor_pools,
        output_dir / 'power_vs_steering_sankey.html'
    )
    
    print("\n" + "="*70)
    print("SANKEY DIAGRAMS COMPLETE")
    print("="*70)
    print(f"\nInteractive visualizations:")
    print(f"  • complete_pathway_sankey.html")
    print(f"  • power_vs_steering_sankey.html")
    print(f"\nLocation: {output_dir}")
    print(f"\nOpen in browser to explore pathways!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()