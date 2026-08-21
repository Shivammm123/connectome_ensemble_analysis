"""
Core Ensemble Network Visualization

Focuses on neurons similar to multiple query neurons (core ensemble).
Creates detailed network showing connectivity patterns.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch, Circle, Patch

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader
from utils.similarity import build_connectivity_vectors, cosine_similarity_matrix


def extract_core_ensemble(
    query_neurons: list,
    loader: ConnectomeDataLoader,
    min_queries_shared: int = 2,
    min_similarity: float = 0.3,
    top_n_per_query: int = 20,
    min_synapses: int = 3
):
    """
    Extract core ensemble members.

    Returns neurons similar to multiple query neurons.
    """

    print("\nExtracting core ensemble members...")
    print("-"*70)

    filtered_conn = loader.filter_connections(min_synapses=min_synapses, verbose=False)

    # Find similar neurons for each query
    all_similar = {}

    for query_id in query_neurons:
        print(f"  Analyzing {loader.get_neuron_name(query_id)[:25]}...")

        query_conn = loader.get_neuron_connectivity(query_id, filtered_conn)

        if len(query_conn) == 0:
            continue

        query_targets = set(query_conn['target'].unique())
        shared_conn = filtered_conn[filtered_conn['target'].isin(query_targets)]
        candidates = shared_conn['source'].unique()
        candidates = [n for n in candidates if n != query_id][:200]

        all_neurons = [query_id] + candidates
        analysis_conn = filtered_conn[filtered_conn['source'].isin(all_neurons)]

        conn_matrix, _ = build_connectivity_vectors(analysis_conn, neuron_ids=all_neurons)
        sim_matrix = cosine_similarity_matrix(conn_matrix)

        similarities = sim_matrix[0, :]

        for idx, sim in enumerate(similarities[1:], 1):
            if sim >= min_similarity:
                nid = all_neurons[idx]

                if nid not in all_similar:
                    all_similar[nid] = []

                all_similar[nid].append({
                    'query': query_id,
                    'similarity': sim
                })

    # Filter to core members
    core_members = []
    for nid, matches in all_similar.items():
        if len(matches) >= min_queries_shared:
            core_members.append({
                'neuron_id': nid,
                'neuron_name': loader.get_neuron_name(nid),
                'n_queries': len(matches),
                'mean_similarity': np.mean([m['similarity'] for m in matches]),
                'similarities': matches
            })

    core_members.sort(key=lambda x: (x['n_queries'], x['mean_similarity']), reverse=True)

    print(f"\n✓ Found {len(core_members)} core ensemble members")
    print(f"  Criteria: Similar to >={min_queries_shared} queries at >={min_similarity}")

    return core_members


def create_core_ensemble_network(
    query_neurons: list,
    core_members: list,
    loader: ConnectomeDataLoader,
    output_file: Path,
    min_synapses: int = 3
):
    """
    Create focused network of core ensemble.
    """

    print("\nCreating core ensemble network...")
    print("-"*70)

    filtered_conn = loader.filter_connections(min_synapses=min_synapses, verbose=False)

    # Create graph
    G = nx.DiGraph()

    # Add query neurons
    for qid in query_neurons:
        G.add_node(qid,
                   name=loader.get_neuron_name(qid),
                   node_type='query',
                   n_queries=len(query_neurons))

    # Add core ensemble members
    for member in core_members:
        nid = member['neuron_id']
        G.add_node(nid,
                   name=member['neuron_name'],
                   node_type='core',
                   n_queries=member['n_queries'],
                   mean_similarity=member['mean_similarity'])

    # Get connections between these neurons
    all_neurons = query_neurons + [m['neuron_id'] for m in core_members]

    for _, row in filtered_conn.iterrows():
        if row['source'] in all_neurons and row['target'] in all_neurons:
            if row['weight'] >= min_synapses:
                G.add_edge(row['source'], row['target'], weight=row['weight'])

    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 16), dpi=300)

    # Layout - circular with queries at center
    pos = {}

    # Query neurons in center
    n_queries = len(query_neurons)
    query_radius = 0.2
    for i, qid in enumerate(query_neurons):
        angle = 2 * np.pi * i / n_queries
        pos[qid] = (query_radius * np.cos(angle), query_radius * np.sin(angle))

    # Core members in outer circle, grouped by n_queries
    groups = {}
    for member in core_members:
        nq = member['n_queries']
        if nq not in groups:
            groups[nq] = []
        groups[nq].append(member)

    angle_offset = 0
    for nq in sorted(groups.keys(), reverse=True):
        members = groups[nq]
        n_members = len(members)
        radius = 0.5 + (len(query_neurons) - nq) * 0.15

        for i, member in enumerate(members):
            angle = angle_offset + 2 * np.pi * i / max(n_members, 1)
            nid = member['neuron_id']
            pos[nid] = (radius * np.cos(angle), radius * np.sin(angle))

        angle_offset += 2 * np.pi / max(n_members, 1) * 0.5

    # Draw edges
    edges = list(G.edges())
    if edges:
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1

        for u, v in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            weight = G[u][v]['weight']

            line_width = 0.5 + (weight / max_weight) * 4

            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=line_width,
                                     color='gray', alpha=0.4))

    # Draw query nodes
    for qid in query_neurons:
        if qid in pos:
            x, y = pos[qid]
            name = G.nodes[qid]['name']

            # Large red square
            square = FancyBboxPatch((x-0.05, y-0.03), 0.1, 0.06,
                                   boxstyle="round,pad=0.01",
                                   facecolor='#FF6B6B',
                                   edgecolor='black',
                                   linewidth=3)
            ax.add_patch(square)

            ax.text(x, y, name[:15], ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white')

    # Draw core member nodes
    for member in core_members:
        nid = member['neuron_id']
        if nid in pos:
            x, y = pos[nid]
            name = member['neuron_name']
            nq = member['n_queries']
            sim = member['mean_similarity']

            # Size by n_queries, color by similarity
            size = 0.02 + (nq / len(query_neurons)) * 0.03
            color = plt.cm.viridis(sim)

            circle = Circle((x, y), size, facecolor=color,
                          edgecolor='black', linewidth=2, zorder=10)
            ax.add_patch(circle)

            # Label
            ax.text(x, y, name[:12], ha='center', va='center',
                   fontsize=8, fontweight='bold', zorder=11)

            # Add metadata below
            ax.text(x, y - size - 0.03, f'[{nq}] {sim:.2f}',
                   ha='center', va='top', fontsize=6,
                   style='italic', color='gray')

    # Title
    ax.set_title(f'Core Ensemble Network\n({len(query_neurons)} query neurons, {len(core_members)} core members)',
                fontsize=16, fontweight='bold', pad=20)

    # Legend
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Query neurons'),
        Patch(facecolor='yellow', label='High similarity (>0.6)'),
        Patch(facecolor='green', label='Medium similarity (0.4-0.6)'),
        Patch(facecolor='blue', label='Lower similarity (0.3-0.4)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

    # Add text box with info
    info_text = "Core Ensemble Members:\n"
    for i, member in enumerate(core_members[:5], 1):
        info_text += f"{i}. {member['neuron_name'][:20]}\n"
        info_text += f"   Similar to {member['n_queries']}/{len(query_neurons)} queries\n"
        info_text += f"   Mean sim: {member['mean_similarity']:.3f}\n"

    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {output_file}")


def main():
    print("\n" + "="*70)
    print("CORE ENSEMBLE NETWORK ANALYSIS")
    print("="*70)

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    loader = ConnectomeDataLoader('config.yaml')
    loader.load_all_data(verbose=False)
    min_synapses = loader.config['analysis']['min_synapses']
    print("✓ Data loaded")

    # Get query neurons from paper_style results
    core_file = Path('results/paper_style/core_ensemble_candidates.csv')

    if not core_file.exists():
        print("\n❌ Run paper_style_plots.py first to identify core candidates!")
        sys.exit(1)

    # Load core candidates from previous analysis
    core_df = pd.read_csv(core_file)

    # Get query neurons from config
    query_ids = []
    for item in config.get('multi_query', {}).get('neuron_names', []):
        try:
            query_ids.append(int(item))
        except (ValueError, TypeError):
            nid = loader.get_neuron_id(str(item))
            if nid:
                query_ids.append(nid)

    if not query_ids:
        print("Using queries from single neuron analysis...")
        raw = config['single_query']['neuron_id']
        try:
            single_id = int(raw)
        except (ValueError, TypeError):
            single_id = loader.get_neuron_id(str(raw))
            if single_id is None:
                print(f"❌ Could not resolve fallback neuron: {raw}")
                sys.exit(1)

        results_file = Path('results/single_neuron/similarity_results.csv')
        if results_file.exists():
            similar = pd.read_csv(results_file)
            query_ids = [single_id] + similar.head(2)['neuron_id'].tolist()
        else:
            query_ids = [single_id]

    print(f"\nQuery neurons: {len(query_ids)}")
    for qid in query_ids:
        print(f"  * {loader.get_neuron_name(qid)}")

    # Extract core ensemble
    core_members = extract_core_ensemble(
        query_ids,
        loader,
        min_queries_shared=2,
        min_similarity=0.15,
        top_n_per_query=50,
        min_synapses=min_synapses
    )

    if not core_members:
        print("\n❌ No core ensemble members found!")
        print("Try lowering min_similarity or min_queries_shared")
        sys.exit(1)

    # Create output directory
    output_dir = Path('results/core_ensemble')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save core members list (drop similarities list before CSV)
    save_df = pd.DataFrame([
        {k: v for k, v in m.items() if k != 'similarities'}
        for m in core_members
    ])
    save_df.to_csv(output_dir / 'core_ensemble_members.csv', index=False)
    print(f"\n✓ Saved: core_ensemble_members.csv")

    # Create network visualization
    create_core_ensemble_network(
        query_ids,
        core_members,
        loader,
        output_dir / 'core_ensemble_network.png',
        min_synapses=min_synapses
    )

    # Summary
    print("\n" + "="*70)
    print("CORE ENSEMBLE ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nCore ensemble members: {len(core_members)}")
    print(f"\nTop 5 core members:")
    for i, member in enumerate(core_members[:5], 1):
        print(f"  {i}. {member['neuron_name'][:30]}")
        print(f"     Similar to {member['n_queries']}/{len(query_ids)} queries")
        print(f"     Mean similarity: {member['mean_similarity']:.3f}")

    print(f"\nResults: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
