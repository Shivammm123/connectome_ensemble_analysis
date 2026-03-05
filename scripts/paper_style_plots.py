"""
Paper-Style Ensemble Visualization

Creates publication-quality plots matching Liessem et al. (2025) style.

Main figure: Similarity heatmap with hierarchical clustering showing
query neurons and their ensemble members.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader
from utils.similarity import build_connectivity_vectors, cosine_similarity_matrix


def create_ensemble_similarity_matrix(
    query_neurons: list,
    loader: ConnectomeDataLoader,
    output_file: Path,
    top_n_per_query: int = 15,
    min_synapses: int = 3,
    cluster_method: str = 'average'
):
    """
    Create Figure 1D-style similarity matrix.

    Shows query neurons and their top similar neurons with hierarchical clustering.

    Parameters
    ----------
    query_neurons : list
        List of query neuron IDs
    loader : ConnectomeDataLoader
        Data loader
    output_file : Path
        Output file path
    top_n_per_query : int
        Number of similar neurons per query
    min_synapses : int
        Minimum synapses threshold
    cluster_method : str
        Clustering method (average, ward, complete)
    """
    print("\n" + "="*70)
    print("Creating paper-style ensemble similarity matrix...")
    print("="*70 + "\n")

    filtered_conn = loader.filter_connections(min_synapses=min_synapses, verbose=False)

    # -------------------------------------------------------------------------
    # Step 1: Find similar neurons for each query
    # -------------------------------------------------------------------------

    print("Step 1: Finding similar neurons for each query")
    print("-"*70 + "\n")

    all_similar_neurons = {}
    query_to_similar = {}

    for query_id in query_neurons:
        print(f"  Analyzing {loader.get_neuron_name(query_id)[:25]}...")

        query_conn = loader.get_neuron_connectivity(query_id, filtered_conn)

        if len(query_conn) == 0:
            print(f"    ⚠ No connections found")
            continue

        query_targets = set(query_conn['target'].unique())
        shared_conn = filtered_conn[filtered_conn['target'].isin(query_targets)]
        candidates = [n for n in shared_conn['source'].unique() if n != query_id]
        candidates = candidates[:200]

        all_neurons = [query_id] + candidates
        analysis_conn = filtered_conn[filtered_conn['source'].isin(all_neurons)]

        conn_matrix, _ = build_connectivity_vectors(analysis_conn, neuron_ids=all_neurons)
        sim_matrix = cosine_similarity_matrix(conn_matrix)

        similarities = sim_matrix[0, :]
        similar_indices = np.argsort(similarities)[::-1][1:top_n_per_query + 1]

        similar_neurons = []
        for idx in similar_indices:
            nid = all_neurons[idx]
            sim = similarities[idx]
            similar_neurons.append({'neuron_id': nid, 'similarity': sim})

            if nid not in all_similar_neurons:
                all_similar_neurons[nid] = []
            all_similar_neurons[nid].append({'query': query_id, 'similarity': sim})

        query_to_similar[query_id] = similar_neurons
        print(f"    ✓ Found {len(similar_neurons)} similar neurons")

    # -------------------------------------------------------------------------
    # Step 2: Build complete neuron set
    # -------------------------------------------------------------------------

    print("\nStep 2: Building complete neuron set")
    print("-"*70 + "\n")

    all_neurons_set = set(query_neurons)
    all_neurons_set.update(all_similar_neurons.keys())
    all_neurons_list = list(all_neurons_set)
    n_total = len(all_neurons_list)

    print(f"  Total neurons to analyze: {n_total}")
    print(f"    Query neurons: {len(query_neurons)}")
    print(f"    Similar neurons: {n_total - len(query_neurons)}")

    # -------------------------------------------------------------------------
    # Step 3: Compute full pairwise similarity matrix
    # -------------------------------------------------------------------------

    print("\nStep 3: Computing pairwise similarity")
    print("-"*70 + "\n")

    analysis_conn = filtered_conn[filtered_conn['source'].isin(all_neurons_list)]

    print(f"  Building connectivity matrix...")
    conn_matrix, _ = build_connectivity_vectors(analysis_conn, neuron_ids=all_neurons_list)

    print(f"  Computing cosine similarity...")
    sim_matrix = cosine_similarity_matrix(conn_matrix)
    print(f"  ✓ Similarity matrix shape: {sim_matrix.shape}")

    # -------------------------------------------------------------------------
    # Step 4: Hierarchical clustering
    # -------------------------------------------------------------------------

    print("\nStep 4: Hierarchical clustering")
    print("-"*70 + "\n")

    distance_matrix = np.maximum(1 - sim_matrix, 0)
    condensed_dist = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method=cluster_method)
    ordered_indices = leaves_list(linkage_matrix)

    sim_matrix_ordered = sim_matrix[ordered_indices, :][:, ordered_indices]
    ordered_neurons = [all_neurons_list[i] for i in ordered_indices]

    print(f"  ✓ Clustering complete using '{cluster_method}' method")

    # -------------------------------------------------------------------------
    # Step 5: Create visualization
    # -------------------------------------------------------------------------

    print("\nStep 5: Creating visualization")
    print("-"*70 + "\n")

    labels = []
    label_colors = []

    for nid in ordered_neurons:
        name = loader.get_neuron_name(nid)
        if len(name) > 20:
            name = name[:17] + "..."

        if nid in query_neurons:
            labels.append(f"★ {name}")
            label_colors.append('#FF6B6B')
        else:
            n_queries_similar = len(all_similar_neurons.get(nid, []))
            if n_queries_similar > 1:
                labels.append(f"{name} [{n_queries_similar}]")
                label_colors.append('#4ECDC4')
            else:
                labels.append(name)
                label_colors.append('#2C3E50')

    fig_height = max(12, n_total * 0.3)
    fig_width = max(14, n_total * 0.3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

    im = ax.imshow(sim_matrix_ordered, cmap='viridis', aspect='auto',
                   vmin=0, vmax=1, interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', fontsize=14, fontweight='bold',
                   rotation=270, labelpad=25)
    cbar.ax.tick_params(labelsize=11)

    tick_positions = range(n_total)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for tick_label, color in zip(ax.get_xticklabels(), label_colors):
        tick_label.set_color(color)
        if color == '#FF6B6B':
            tick_label.set_fontweight('bold')

    for tick_label, color in zip(ax.get_yticklabels(), label_colors):
        tick_label.set_color(color)
        if color == '#FF6B6B':
            tick_label.set_fontweight('bold')

    ax.set_xticks(np.arange(n_total) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_total) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    query_indices_in_plot = [ordered_neurons.index(qid)
                             for qid in query_neurons if qid in ordered_neurons]
    for idx in query_indices_in_plot:
        rect = plt.Rectangle((idx - 0.5, idx - 0.5), 1, 1,
                              fill=False, edgecolor='red', linewidth=2.5, alpha=0.8)
        ax.add_patch(rect)

    title = (f'Ensemble Connectivity Similarity Matrix\n'
             f'({len(query_neurons)} query neurons, '
             f'{n_total - len(query_neurons)} similar neurons)')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Query neurons (★)'),
        Patch(facecolor='#4ECDC4', label='Similar to multiple queries [N]'),
        Patch(facecolor='#2C3E50', label='Similar to one query'),
    ]
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {output_file}")
    print(f"    Size: {fig_width:.1f} x {fig_height:.1f} inches")

    # -------------------------------------------------------------------------
    # Step 6: Summary statistics
    # -------------------------------------------------------------------------

    print("\nStep 6: Summary statistics")
    print("-"*70 + "\n")

    core_candidates = []
    for nid, matches in all_similar_neurons.items():
        if len(matches) >= 2:
            core_candidates.append({
                'neuron_id': nid,
                'neuron_name': loader.get_neuron_name(nid),
                'n_queries': len(matches),
                'mean_similarity': np.mean([m['similarity'] for m in matches]),
            })

    core_candidates.sort(key=lambda x: x['mean_similarity'], reverse=True)
    print(f"  Neurons similar to multiple queries: {len(core_candidates)}")

    if core_candidates:
        print(f"\n  Top 5 core ensemble candidates:")
        for i, candidate in enumerate(core_candidates[:5], 1):
            print(f"    {i}. {candidate['neuron_name'][:30]}")
            print(f"       Similar to {candidate['n_queries']}/{len(query_neurons)} queries")
            print(f"       Mean similarity: {candidate['mean_similarity']:.3f}")

        summary_df = pd.DataFrame(core_candidates)
        summary_file = output_file.parent / 'core_ensemble_candidates.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n  ✓ Saved core candidates: {summary_file}")

    return {
        'similarity_matrix': sim_matrix_ordered,
        'neurons': ordered_neurons,
        'core_candidates': core_candidates,
    }


def main():
    print("\n" + "="*70)
    print("PAPER-STYLE ENSEMBLE VISUALIZATION")
    print("Replicating Liessem et al. (2025) Figure 1D")
    print("="*70 + "\n")

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    loader = ConnectomeDataLoader('config.yaml')
    loader.load_all_data(verbose=False)
    print("✓ Data loaded\n")

    print("-"*70)
    print("Selecting query neurons")
    print("-"*70 + "\n")

    query_ids = []
    for item in config.get('multi_query', {}).get('neuron_names', []):
        try:
            query_ids.append(int(item))
        except (ValueError, TypeError):
            nid = loader.get_neuron_id(str(item))
            if nid:
                query_ids.append(nid)

    # Fallback to single query + top similar
    if not query_ids:
        print("  Using single query + top 2 similar neurons\n")
        raw = config['single_query']['neuron_id']
        try:
            single_id = int(raw)
        except (ValueError, TypeError):
            single_id = loader.get_neuron_id(str(raw))
            if single_id is None:
                print(f"  ❌ Could not resolve fallback neuron: {raw}")
                sys.exit(1)

        results_file = Path('results/single_neuron/similarity_results.csv')
        if results_file.exists():
            similar = pd.read_csv(results_file)
            query_ids = [single_id] + similar.head(2)['neuron_id'].tolist()
        else:
            query_ids = [single_id]

    print(f"Query neurons ({len(query_ids)}):")
    for qid in query_ids:
        print(f"  • {loader.get_neuron_name(qid)} ({qid})")

    output_dir = Path('results/paper_style')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = create_ensemble_similarity_matrix(
        query_ids,
        loader,
        output_dir / 'ensemble_similarity_matrix.png',
        top_n_per_query=15,
        min_synapses=config['analysis']['min_synapses'],
        cluster_method='average'
    )

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70 + "\n")

    print(f"Output: {output_dir}")
    print(f"\nMain figure: ensemble_similarity_matrix.png")

    if results['core_candidates']:
        print(f"\nCore ensemble candidates: {len(results['core_candidates'])}")
        print(f"(See: core_ensemble_candidates.csv)")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
