"""
Single Neuron Similarity Analysis

Exact replication of Liessem et al. (2025) cosine similarity method.

Paper method:
"We treated each DN's typewise synapse count onto all other neurons 
as a vector, and computed the cosine similarity of this vector"
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.data_loader import ConnectomeDataLoader
from utils.similarity import build_connectivity_vectors, cosine_similarity_matrix, find_top_similar

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("\n" + "="*70)
    print("SINGLE NEURON SIMILARITY ANALYSIS")
    print("Replicating Liessem et al. (2025) methodology")
    print("="*70 + "\n")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data loader
    loader = ConnectomeDataLoader('config.yaml')
    
    # -------------------------------------------------------------------------
    # 1. Load all data
    # -------------------------------------------------------------------------
    
    print("-"*70)
    print("1. Loading data")
    print("-"*70 + "\n")
    
    connections, neurons, name_mapping = loader.load_all_data(verbose=True)
    
    # -------------------------------------------------------------------------
    # 2. Get query neuron
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("2. Identifying query neuron")
    print("-"*70 + "\n")
    
    query_id = config['single_query'].get('neuron_id')
    query_name = config['single_query'].get('neuron_name')

    # neuron_id can be either a numeric root ID or a cell name string
    if query_id is not None:
        try:
            query_id = int(query_id)
        except (ValueError, TypeError):
            # Treat it as a name and resolve to root ID
            query_name = str(query_id)
            query_id = None

    if query_name:
        query_id = loader.get_neuron_id(query_name)
        if query_id is None:
            print(f"  ❌ Could not find neuron: {query_name}")
            sys.exit(1)
        print(f"  Neuron name: {query_name}")
        print(f"  Root ID: {query_id}")
    elif query_id:
        query_name = loader.get_neuron_name(query_id)
        print(f"  Root ID: {query_id}")
        print(f"  Neuron name: {query_name}")
    else:
        print("  ❌ No query neuron specified in config.yaml")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # 3. Filter connections (paper used 3+ synapses)
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("3. Filtering connectivity (paper method: 3+ synapses)")
    print("-"*70 + "\n")
    
    min_syn = config['analysis']['min_synapses']
    filtered_conn = loader.filter_connections(min_synapses=min_syn, verbose=True)
    
    # -------------------------------------------------------------------------
    # 4. Get query neuron connectivity
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("4. Extracting query neuron connectivity")
    print("-"*70 + "\n")
    
    query_conn = loader.get_neuron_connectivity(query_id, filtered_conn)
    
    if len(query_conn) == 0:
        print(f"  ❌ No connections found for {query_name}")
        sys.exit(1)
    
    print(f"  Query neuron: {query_name}")
    print(f"  Downstream connections: {len(query_conn)}")
    print(f"  Total synapses: {query_conn['weight'].sum()}")
    print(f"  Unique targets: {query_conn['target'].nunique()}")
    
    # -------------------------------------------------------------------------
    # 5. Find comparison neurons (neurons sharing targets)
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("5. Finding comparison neurons")
    print("-"*70 + "\n")
    
    query_targets = set(query_conn['target'].unique())
    print(f"  Query targets: {len(query_targets)}")
    
    # Find neurons connecting to same targets
    shared_conn = filtered_conn[filtered_conn['target'].isin(query_targets)]
    candidates = shared_conn['source'].unique()
    candidates = [n for n in candidates if n != query_id]
    
    print(f"  Found {len(candidates)} neurons with shared targets")
    
    # Limit to max
    max_comp = config['analysis']['max_comparison_neurons']
    if len(candidates) > max_comp:
        # Prioritize by overlap
        overlap = shared_conn.groupby('source').size()
        top = overlap.nlargest(max_comp).index.tolist()
        candidates = [n for n in top if n != query_id]
        print(f"  Selected top {len(candidates)} by overlap")
    
    comparison_neurons = candidates[:max_comp]
    all_neurons = [query_id] + comparison_neurons
    
    # -------------------------------------------------------------------------
    # 6. Compute similarity (EXACT PAPER METHOD)
    # -------------------------------------------------------------------------
    
    # Compute similarity using paper's method
    conn_matrix, _ = build_connectivity_vectors(
        filtered_conn,
        neuron_ids=all_neurons
    )

    sim_matrix = cosine_similarity_matrix(conn_matrix)

    query_idx = all_neurons.index(query_id)
    results = find_top_similar(
        query_idx,
        sim_matrix,
        all_neurons,
        top_n=config['analysis']['top_n_similar'],
        threshold=config['analysis']['similarity_threshold']
    )
    
    # -------------------------------------------------------------------------
    # 7. Add names to results
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("7. Adding cell names to results")
    print("-"*70 + "\n")
    
    results = loader.add_names_to_dataframe(results, ['neuron_id'])
    
    print("  ✓ Names added")
    print(f"\nTop 5 most similar neurons:")
    for idx in range(min(5, len(results))):
        row = results.iloc[idx]
        print(f"  {idx+1}. {row['neuron_id_name']}: {row['similarity']:.4f}")
    
    # -------------------------------------------------------------------------
    # 8. Save results
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("8. Saving results")
    print("-"*70 + "\n")
    
    output_dir = Path('results/single_neuron')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    results_file = output_dir / 'similarity_results.csv'
    results.to_csv(results_file, index=False)
    print(f"  ✓ {results_file}")
    
    # Create summary report
    report = f"""# Similarity Analysis Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Query Neuron

- **Name:** {query_name}
- **Root ID:** {query_id}
- **Connections:** {len(query_conn)}
- **Total synapses:** {query_conn['weight'].sum()}
- **Unique targets:** {query_conn['target'].nunique()}

## Method

Exact replication of **Liessem et al. (2025)**:
- Created output vector (synapse counts to all targets)
- Computed cosine similarity
- Min synapses threshold: {min_syn}
- Normalization: {config['analysis']['normalization']}

## Results

**Neurons analyzed:** {len(all_neurons)}
**Similar neurons found:** {len(results)}

### Top 10 Most Similar

"""
    
    for idx in range(min(10, len(results))):
        row = results.iloc[idx]
        report += f"{idx+1}. **{row['neuron_id_name']}** (ID: {row['neuron_id']})\n"
        report += f"   - Similarity: {row['similarity']:.4f}\n\n"
    
    report_file = output_dir / 'report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✓ {report_file}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")
    
    print(f"Query: {query_name} ({query_id})")
    print(f"Similar neurons: {len(results)}")
    
    if len(results) > 0:
        print(f"\nTop 3:")
        for idx in range(min(3, len(results))):
            row = results.iloc[idx]
            print(f"  {idx+1}. {row['neuron_id_name']}: {row['similarity']:.4f}")
    
    print(f"\nResults: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()