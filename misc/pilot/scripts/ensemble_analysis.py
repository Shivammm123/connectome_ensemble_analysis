"""
Multi-Query Ensemble Analysis

Replicates Liessem et al. (2025) ensemble discovery method.

Finds neurons similar to MULTIPLE query neurons to discover
complete functional ensembles.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader
from utils.similarity import build_connectivity_vectors, cosine_similarity_matrix, find_top_similar


def find_ensemble_members(
    similarity_matrix: np.ndarray,
    neuron_list: list,
    query_neuron_ids: list,
    core_threshold: float = 0.6,
    peripheral_threshold: float = 0.4,
    min_queries_shared: int = 2
):
    """
    Find ensemble members from multiple query neurons.
    
    Parameters
    ----------
    similarity_matrix : np.ndarray
        Pairwise similarity matrix
    neuron_list : list
        All neuron IDs
    query_neuron_ids : list
        Query neuron IDs
    core_threshold : float
        Min similarity for core ensemble
    peripheral_threshold : float  
        Min similarity for peripheral
    min_queries_shared : int
        Min queries to match
    
    Returns
    -------
    dict with 'core', 'peripheral', 'all' DataFrames
    """
    # Get indices for query neurons
    query_indices = [neuron_list.index(qid) for qid in query_neuron_ids]
    
    results = []
    
    for i, nid in enumerate(neuron_list):
        # Skip if this is a query neuron
        if nid in query_neuron_ids:
            continue
        
        # Get similarities to all queries
        similarities = [similarity_matrix[query_indices[qi], i] for qi in range(len(query_indices))]
        
        # Count how many queries it's similar to
        n_core = sum(s >= core_threshold for s in similarities)
        n_peripheral = sum(s >= peripheral_threshold for s in similarities)
        
        if n_peripheral >= min_queries_shared:
            results.append({
                'neuron_id': nid,
                'mean_similarity': np.mean(similarities),
                'max_similarity': np.max(similarities),
                'min_similarity': np.min(similarities),
                'n_queries_core': n_core,
                'n_queries_peripheral': n_peripheral,
                'is_core': n_core == len(query_neuron_ids),
                'similarities': similarities
            })
    
    if not results:
        return {
            'core': pd.DataFrame(),
            'peripheral': pd.DataFrame(),
            'all': pd.DataFrame()
        }
    
    # Create DataFrame
    all_df = pd.DataFrame(results)
    all_df = all_df.sort_values('mean_similarity', ascending=False)
    
    # Split into core and peripheral
    core_df = all_df[all_df['is_core']].copy()
    peripheral_df = all_df[~all_df['is_core']].copy()
    
    return {
        'core': core_df,
        'peripheral': peripheral_df,
        'all': all_df
    }


def plot_ensemble_heatmap(
    similarity_matrix: np.ndarray,
    neuron_list: list,
    query_ids: list,
    ensemble_members: list,
    loader: ConnectomeDataLoader,
    output_file: Path
):
    """Create heatmap of query neurons and ensemble members."""
    
    # Get indices
    all_neurons = query_ids + ensemble_members[:20]  # Limit to top 20
    indices = [neuron_list.index(nid) for nid in all_neurons]
    
    # Subset similarity matrix
    subset_sim = similarity_matrix[np.ix_(indices, indices)]
    
    # Get names
    names = [loader.get_neuron_name(nid)[:20] for nid in all_neurons]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # Heatmap
    sns.heatmap(subset_sim, 
                xticklabels=names,
                yticklabels=names,
                cmap='viridis',
                vmin=0, vmax=1,
                square=True,
                cbar_kws={'label': 'Cosine Similarity'},
                ax=ax)
    
    # Highlight query neurons
    n_queries = len(query_ids)
    ax.add_patch(plt.Rectangle((0, 0), n_queries, n_queries, 
                               fill=False, edgecolor='red', lw=3))
    
    ax.set_title('Ensemble Similarity Heatmap\n(Red box = query neurons)',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_file.name}")


def main():
    print("\n" + "="*70)
    print("MULTI-QUERY ENSEMBLE ANALYSIS")
    print("Replicating Liessem et al. (2025) ensemble discovery")
    print("="*70 + "\n")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize loader
    loader = ConnectomeDataLoader('config.yaml')
    
    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    
    print("-"*70)
    print("1. Loading data")
    print("-"*70 + "\n")
    
    connections, neurons, name_mapping = loader.load_all_data(verbose=True)
    
    # -------------------------------------------------------------------------
    # 2. Get query neurons
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("2. Identifying query neurons")
    print("-"*70 + "\n")
    
    # Each entry can be a numeric root ID or a cell name string
    query_ids = []
    for entry in config.get('multi_query', {}).get('neuron_names', []):
        try:
            query_ids.append(int(entry))
            print(f"  {entry} (root ID)")
        except (ValueError, TypeError):
            nid = loader.get_neuron_id(str(entry))
            if nid:
                query_ids.append(nid)
                print(f"  {entry} → {nid}")
            else:
                print(f"  ⚠ Could not find: {entry}")
    
    # Fallback: use single query neuron + top similar as queries
    if not query_ids:
        print("  No multi-query specified in config.")
        print("  Using single query + top 2 similar neurons as ensemble queries\n")
        
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
    
    print(f"  Query neurons ({len(query_ids)}):")
    for qid in query_ids:
        qname = loader.get_neuron_name(qid)
        print(f"    • {qname} ({qid})")
    
    # -------------------------------------------------------------------------
    # 3. Filter connections
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("3. Filtering connectivity")
    print("-"*70 + "\n")
    
    min_syn = config['analysis']['min_synapses']
    filtered_conn = loader.filter_connections(min_synapses=min_syn, verbose=True)
    
    # -------------------------------------------------------------------------
    # 4. Find comparison neurons
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("4. Finding comparison neurons")
    print("-"*70 + "\n")
    
    # Get targets of all query neurons
    all_query_targets = set()
    for qid in query_ids:
        query_conn = loader.get_neuron_connectivity(qid, filtered_conn)
        all_query_targets.update(query_conn['target'].unique())
    
    print(f"  Query neurons connect to {len(all_query_targets)} total targets")
    
    # Find neurons sharing these targets
    shared_conn = filtered_conn[filtered_conn['target'].isin(all_query_targets)]
    candidates = shared_conn['source'].unique()
    candidates = [n for n in candidates if n not in query_ids]
    
    print(f"  Found {len(candidates)} candidate neurons")
    
    # Limit to max
    max_comp = config['analysis']['max_comparison_neurons']
    if len(candidates) > max_comp:
        overlap = shared_conn.groupby('source').size()
        top = overlap.nlargest(max_comp).index.tolist()
        candidates = [n for n in top if n not in query_ids]
        print(f"  Limited to top {len(candidates)}")
    
    all_neurons = query_ids + candidates
    
    # -------------------------------------------------------------------------
    # 5. Build connectivity matrix and compute similarity
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("5. Computing pairwise similarity (paper method)")
    print("-"*70 + "\n")
    
    analysis_conn = filtered_conn[filtered_conn['source'].isin(all_neurons)]
    
    print(f"  Building connectivity vectors...")
    conn_matrix, _ = build_connectivity_vectors(analysis_conn, neuron_ids=all_neurons)
    
    print(f"  Computing cosine similarity...")
    sim_matrix = cosine_similarity_matrix(conn_matrix)
    
    print(f"  ✓ Similarity matrix: {sim_matrix.shape}")
    
    # -------------------------------------------------------------------------
    # 6. Find ensemble members
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("6. Identifying ensemble members")
    print("-"*70 + "\n")
    
    ensemble = find_ensemble_members(
        sim_matrix,
        all_neurons,
        query_ids,
        core_threshold=0.6,
        peripheral_threshold=0.4,
        min_queries_shared=min(2, len(query_ids))
    )
    
    print(f"  ✓ Core ensemble members: {len(ensemble['core'])}")
    print(f"  ✓ Peripheral members: {len(ensemble['peripheral'])}")
    print(f"  ✓ Total candidates: {len(ensemble['all'])}")
    
    # -------------------------------------------------------------------------
    # 7. Add names
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("7. Adding cell names")
    print("-"*70 + "\n")
    
    for key in ['core', 'peripheral', 'all']:
        if len(ensemble[key]) > 0:
            ensemble[key] = loader.add_names_to_dataframe(ensemble[key], ['neuron_id'])
    
    print("  ✓ Names added")
    
    # -------------------------------------------------------------------------
    # 8. Save results
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("8. Saving results")
    print("-"*70 + "\n")
    
    output_dir = Path('results/ensembles')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each category (drop the raw similarities list — not CSV-friendly)
    save_cols = [c for c in ensemble['all'].columns if c != 'similarities']

    if len(ensemble['core']) > 0:
        core_file = output_dir / 'core_ensemble.csv'
        ensemble['core'][save_cols].to_csv(core_file, index=False)
        print(f"  ✓ {core_file.name}")

    if len(ensemble['peripheral']) > 0:
        periph_file = output_dir / 'peripheral_ensemble.csv'
        ensemble['peripheral'][save_cols].to_csv(periph_file, index=False)
        print(f"  ✓ {periph_file.name}")

    if len(ensemble['all']) > 0:
        all_file = output_dir / 'all_ensemble_candidates.csv'
        ensemble['all'][save_cols].to_csv(all_file, index=False)
        print(f"  ✓ {all_file.name}")
    
    # -------------------------------------------------------------------------
    # 9. Create visualizations
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("9. Creating visualizations")
    print("-"*70 + "\n")
    
    if len(ensemble['all']) > 0:
        fig_dir = output_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)
        
        plot_ensemble_heatmap(
            sim_matrix,
            all_neurons,
            query_ids,
            ensemble['all']['neuron_id'].tolist(),
            loader,
            fig_dir / 'ensemble_heatmap.png'
        )
    
    # -------------------------------------------------------------------------
    # 10. Generate report
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("10. Generating report")
    print("-"*70 + "\n")
    
    report = f"""# Ensemble Analysis Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Method

Replication of **Liessem et al. (2025)** ensemble discovery:
- Multiple query neurons analyzed together
- Core ensemble: similar to ALL queries (≥0.6)
- Peripheral ensemble: similar to SOME queries (≥0.4)

## Query Neurons ({len(query_ids)})

"""
    
    for qid in query_ids:
        qname = loader.get_neuron_name(qid)
        report += f"- **{qname}** (ID: {qid})\n"
    
    report += f"\n## Results\n\n"
    report += f"**Core ensemble members:** {len(ensemble['core'])}\n"
    report += f"**Peripheral members:** {len(ensemble['peripheral'])}\n"
    report += f"**Total candidates:** {len(ensemble['all'])}\n\n"
    
    if len(ensemble['core']) > 0:
        report += f"### Core Ensemble Members\n\n"
        report += f"Neurons similar to ALL {len(query_ids)} query neurons:\n\n"
        
        for i, (_, row) in enumerate(ensemble['core'].head(10).iterrows(), start=1):
            report += f"{i}. **{row['neuron_id_name']}** (ID: {row['neuron_id']})\n"
            report += f"   - Mean similarity: {row['mean_similarity']:.4f}\n"
            report += f"   - Similar to {row['n_queries_core']}/{len(query_ids)} queries (core)\n\n"
    
    if len(ensemble['peripheral']) > 0:
        report += f"\n### Peripheral Ensemble Members\n\n"
        report += f"Neurons similar to SOME query neurons:\n\n"
        
        for i, (_, row) in enumerate(ensemble['peripheral'].head(10).iterrows(), start=1):
            report += f"{i}. **{row['neuron_id_name']}** (ID: {row['neuron_id']})\n"
            report += f"   - Mean similarity: {row['mean_similarity']:.4f}\n"
            report += f"   - Similar to {row['n_queries_peripheral']}/{len(query_ids)} queries\n\n"
    
    report_file = output_dir / 'ensemble_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✓ {report_file.name}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    
    print("\n" + "="*70)
    print("ENSEMBLE ANALYSIS COMPLETE")
    print("="*70 + "\n")
    
    print(f"Query neurons: {len(query_ids)}")
    print(f"Core ensemble: {len(ensemble['core'])}")
    print(f"Peripheral ensemble: {len(ensemble['peripheral'])}")
    
    if len(ensemble['core']) > 0:
        print(f"\nTop 3 core members:")
        for idx in range(min(3, len(ensemble['core']))):
            row = ensemble['core'].iloc[idx]
            print(f"  {idx+1}. {row['neuron_id_name']}: {row['mean_similarity']:.4f}")
    
    print(f"\nResults: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()