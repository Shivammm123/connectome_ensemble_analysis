"""
Comparative Analysis

Compare similarity analysis across multiple query neurons.

Shows:
- Which neurons are similar to multiple queries
- Pairwise similarity between query neurons
- Shared vs unique similar neurons
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

try:
    from matplotlib_venn import venn2, venn3
    VENN_AVAILABLE = True
except ImportError:
    VENN_AVAILABLE = False
    print("Note: Install matplotlib-venn for Venn diagrams: pip install matplotlib-venn")

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader
from utils.similarity import build_connectivity_vectors, cosine_similarity_matrix, find_top_similar


def analyze_multiple_neurons(
    neuron_ids: list,
    loader: ConnectomeDataLoader,
    min_synapses: int = 3,
    top_n: int = 20
):
    """
    Run similarity analysis for multiple neurons.

    Returns dict with results for each neuron.
    """
    results = {}
    filtered_conn = loader.filter_connections(min_synapses=min_synapses, verbose=False)

    for neuron_id in neuron_ids:
        print(f"  Analyzing {loader.get_neuron_name(neuron_id)[:20]}...")

        query_conn = loader.get_neuron_connectivity(neuron_id, filtered_conn)

        if len(query_conn) == 0:
            print(f"    ⚠ No connections found")
            continue

        query_targets = set(query_conn['target'].unique())
        shared_conn = filtered_conn[filtered_conn['target'].isin(query_targets)]
        candidates = shared_conn['source'].unique()
        candidates = [n for n in candidates if n != neuron_id]
        candidates = candidates[:200]

        all_neurons = [neuron_id] + list(candidates)
        analysis_conn = filtered_conn[filtered_conn['source'].isin(all_neurons)]

        conn_matrix, _ = build_connectivity_vectors(analysis_conn, neuron_ids=all_neurons)
        sim_matrix = cosine_similarity_matrix(conn_matrix)

        query_idx = 0
        similar = find_top_similar(query_idx, sim_matrix, all_neurons, top_n=top_n)
        similar = loader.add_names_to_dataframe(similar, ['neuron_id'])

        results[neuron_id] = {
            'similar_neurons': similar,
            'connectivity': query_conn,
            'n_connections': len(query_conn),
            'n_targets': query_conn['target'].nunique(),
            'total_synapses': query_conn['weight'].sum()
        }

    return results


def find_shared_similar_neurons(results: dict, min_shared: int = 2):
    """Find neurons similar to multiple query neurons."""
    all_similar = {}

    for query_id, data in results.items():
        for _, row in data['similar_neurons'].iterrows():
            nid = row['neuron_id']
            sim = row['similarity']

            if nid not in all_similar:
                all_similar[nid] = []

            all_similar[nid].append({'query_id': query_id, 'similarity': sim})

    shared_neurons = []
    for nid, matches in all_similar.items():
        if len(matches) >= min_shared:
            shared_neurons.append({
                'neuron_id': nid,
                'n_queries': len(matches),
                'mean_similarity': np.mean([m['similarity'] for m in matches]),
                'max_similarity': np.max([m['similarity'] for m in matches]),
                'min_similarity': np.min([m['similarity'] for m in matches]),
            })

    if not shared_neurons:
        return pd.DataFrame()

    df = pd.DataFrame(shared_neurons)
    df = df.sort_values('mean_similarity', ascending=False)
    return df


def plot_comparison_heatmap(
    neuron_ids: list,
    results: dict,
    loader: ConnectomeDataLoader,
    output_file: Path
):
    """Create table comparing query neurons."""
    comparison_data = []
    for nid in neuron_ids:
        if nid in results:
            comparison_data.append({
                'Neuron': loader.get_neuron_name(nid)[:20],
                'Connections': results[nid]['n_connections'],
                'Targets': results[nid]['n_targets'],
                'Total Synapses': results[nid]['total_synapses'],
                'Top Similarity': results[nid]['similar_neurons'].iloc[0]['similarity']
            })

    df = pd.DataFrame(comparison_data)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Query Neuron Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file.name}")


def plot_venn_diagram(
    results: dict,
    neuron_ids: list,
    loader: ConnectomeDataLoader,
    output_file: Path,
    top_n: int = 20
):
    """Create Venn diagram of similar neurons overlap."""
    if not VENN_AVAILABLE:
        print("  ⚠ matplotlib-venn not available, skipping Venn diagram")
        return

    if len(neuron_ids) not in [2, 3]:
        print("  ⚠ Venn diagrams only for 2-3 neurons")
        return

    sets = []
    labels = []
    for nid in neuron_ids[:3]:
        if nid in results:
            similar_set = set(results[nid]['similar_neurons'].head(top_n)['neuron_id'])
            sets.append(similar_set)
            labels.append(loader.get_neuron_name(nid)[:15])

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    if len(sets) == 2:
        venn2(sets, set_labels=labels, ax=ax)
    elif len(sets) == 3:
        venn3(sets, set_labels=labels, ax=ax)

    plt.title(f'Overlap of Top {top_n} Similar Neurons',
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file.name}")


def main():
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70 + "\n")

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    loader = ConnectomeDataLoader('config.yaml')
    loader.load_all_data(verbose=False)
    print("✓ Data loaded\n")

    print("-"*70)
    print("Select neurons to compare")
    print("-"*70 + "\n")

    # Resolve each entry — accepts names or numeric root IDs
    final_ids = []
    for item in config.get('multi_query', {}).get('neuron_names', []):
        try:
            final_ids.append(int(item))
        except (ValueError, TypeError):
            nid = loader.get_neuron_id(str(item))
            if nid:
                final_ids.append(nid)

    # Fallback: single query + top 2 similar
    if not final_ids:
        print("  No neurons in multi_query config")
        print("  Using: single query + top 2 similar\n")

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
            final_ids = [single_id] + similar.head(2)['neuron_id'].tolist()
        else:
            final_ids = [single_id]

    print(f"Comparing {len(final_ids)} neurons:")
    for nid in final_ids:
        print(f"  • {loader.get_neuron_name(nid)[:30]} ({nid})")

    print("\n" + "-"*70)
    print("Running similarity analysis for each neuron")
    print("-"*70 + "\n")

    results = analyze_multiple_neurons(
        final_ids,
        loader,
        min_synapses=config['analysis']['min_synapses'],
        top_n=config['analysis']['top_n_similar']
    )
    print(f"\n✓ Analyzed {len(results)} neurons")

    print("\n" + "-"*70)
    print("Finding shared similar neurons")
    print("-"*70 + "\n")

    shared = find_shared_similar_neurons(results, min_shared=2)

    if len(shared) > 0:
        shared = loader.add_names_to_dataframe(shared, ['neuron_id'])
        print(f"✓ Found {len(shared)} neurons similar to multiple queries")
        print(f"\nTop 5 shared similar neurons:")
        for idx in range(min(5, len(shared))):
            row = shared.iloc[idx]
            print(f"  {idx+1}. {row['neuron_id_name'][:30]}")
            print(f"      Similar to {row['n_queries']}/{len(final_ids)} queries")
            print(f"      Mean similarity: {row['mean_similarity']:.3f}")
    else:
        print("  No neurons similar to multiple queries found")

    print("\n" + "-"*70)
    print("Saving results")
    print("-"*70 + "\n")

    output_dir = Path('results/comparative')
    output_dir.mkdir(parents=True, exist_ok=True)

    for nid, data in results.items():
        filename = f"similar_to_{nid}.csv"
        data['similar_neurons'].to_csv(output_dir / filename, index=False)
        print(f"  ✓ {filename}")

    if len(shared) > 0:
        shared_file = output_dir / 'shared_similar_neurons.csv'
        shared.to_csv(shared_file, index=False)
        print(f"  ✓ shared_similar_neurons.csv")

    print("\n" + "-"*70)
    print("Creating visualizations")
    print("-"*70 + "\n")

    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    plot_comparison_heatmap(final_ids, results, loader, fig_dir / 'comparison_table.png')

    if len(final_ids) in [2, 3]:
        plot_venn_diagram(results, final_ids, loader, fig_dir / 'venn_diagram.png', top_n=20)

    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS COMPLETE")
    print("="*70 + "\n")

    print(f"Compared {len(results)} neurons")
    if len(shared) > 0:
        print(f"Shared similar neurons: {len(shared)}")
        row = shared.iloc[0]
        print(f"\nTop shared neuron:")
        print(f"  {row['neuron_id_name']}")
        print(f"  Similar to {row['n_queries']}/{len(final_ids)} queries")
        print(f"  Mean similarity: {row['mean_similarity']:.3f}")

    print(f"\nResults: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
