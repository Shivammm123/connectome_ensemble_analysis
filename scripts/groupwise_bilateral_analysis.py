"""
Groupwise Analysis: Bilateral Pair Grouping

Groups neurons into bilateral pairs and applies groupwise thresholds
similar to Ache et al. 2025 and Cheong et al. 2024.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def identify_bilateral_pairs(neurons, connections):
    """
    Identify potential bilateral pairs based on connectivity similarity.
    
    Strategy:
    1. Group neurons by neuropil location
    2. Find pairs with mirror-symmetric connectivity patterns
    3. Create bilateral groups
    """
    
    print("\n" + "="*70)
    print("IDENTIFYING BILATERAL PAIRS")
    print("="*70 + "\n")
    
    # Get VNC interneurons
    vnc_ins = neurons[neurons['Super Class'] == 'ventral_nerve_cord_intrinsic'].copy()
    print(f"Total VNC interneurons: {len(vnc_ins):,}")

    # Build connectivity fingerprints for each IN
    print("\nBuilding connectivity fingerprints...")

    in_ids = vnc_ins['Root ID'].tolist()
    
    # Get unique targets
    targets = connections[connections['source'].isin(in_ids)]['target'].unique()
    print(f"Unique downstream targets: {len(targets):,}")
    
    # Build connectivity matrix (INs × Targets)
    conn_matrix = pd.crosstab(
        connections[connections['source'].isin(in_ids)]['source'],
        connections[connections['source'].isin(in_ids)]['target'],
        values=connections[connections['source'].isin(in_ids)]['weight'],
        aggfunc='sum',
        dropna=False
    ).fillna(0)
    
    print(f"Connectivity matrix: {conn_matrix.shape}")
    
    # Find bilateral pairs using connectivity similarity
    print("\nFinding bilateral pairs...")
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(conn_matrix.values)
    
    # Find high-similarity pairs (but not self)
    # Use only neurons present in the connectivity matrix
    conn_in_ids = conn_matrix.index.tolist()
    pairs = []
    paired_neurons = set()

    for i in range(len(conn_in_ids)):
        if conn_in_ids[i] in paired_neurons:
            continue

        # Find most similar neuron (excluding self)
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1  # Exclude self

        most_similar_idx = np.argmax(similarities)
        max_similarity = similarities[most_similar_idx]

        # If very similar (>0.95), consider as bilateral pair
        if max_similarity > 0.95:
            neuron_a = conn_in_ids[i]
            neuron_b = conn_in_ids[most_similar_idx]
            
            pairs.append({
                'group_id': len(pairs),
                'neuron_1': neuron_a,
                'neuron_2': neuron_b,
                'similarity': max_similarity,
                'group_size': 2
            })
            
            paired_neurons.add(neuron_a)
            paired_neurons.add(neuron_b)
    
    # Create singleton groups for unpaired neurons (including those with no connections)
    unpaired = [n for n in in_ids if n not in paired_neurons]
    
    for neuron_id in unpaired:
        pairs.append({
            'group_id': len(pairs),
            'neuron_1': neuron_id,
            'neuron_2': None,
            'similarity': 1.0,
            'group_size': 1
        })
    
    bilateral_groups = pd.DataFrame(pairs)
    
    print(f"\nBilateral pairs identified: {len([p for p in pairs if p['group_size']==2]):,}")
    print(f"Singleton neurons: {len([p for p in pairs if p['group_size']==1]):,}")
    print(f"Total groups: {len(bilateral_groups):,}")
    
    return bilateral_groups


def compute_groupwise_connectivity(bilateral_groups, connections, wing_mns, 
                                   groupwise_threshold=50):
    """
    Compute groupwise IN→MN connectivity.
    """
    
    print("\n" + "="*70)
    print(f"COMPUTING GROUPWISE CONNECTIVITY (threshold ≥{groupwise_threshold})")
    print("="*70 + "\n")
    
    # Filter for wing MN connections
    wing_conn = connections[connections['target'].isin(wing_mns)].copy()

    # Build neuron → group_id map (vectorized, avoids nested loop)
    rows = [{'neuron_id': g['neuron_1'], 'group_id': g['group_id'], 'group_size': g['group_size']}
            for _, g in bilateral_groups.iterrows()]
    rows += [{'neuron_id': g['neuron_2'], 'group_id': g['group_id'], 'group_size': g['group_size']}
             for _, g in bilateral_groups.iterrows() if g['neuron_2'] is not None]
    neuron_group_map = pd.DataFrame(rows)

    # Join group info onto wing connections and aggregate
    merged = wing_conn.merge(neuron_group_map, left_on='source', right_on='neuron_id', how='inner')
    agg = merged.groupby(['group_id', 'group_size', 'target'])['weight'].sum().reset_index()
    agg.columns = ['group_id', 'group_size', 'target_mn', 'groupwise_synapses']

    # Apply threshold
    agg = agg[agg['groupwise_synapses'] >= groupwise_threshold]

    # Add group_neurons column to match original schema
    group_neuron_map = (neuron_group_map.groupby('group_id')['neuron_id']
                        .apply(list).reset_index()
                        .rename(columns={'neuron_id': 'group_neurons'}))
    groupwise_df = agg.merge(group_neuron_map, on='group_id', how='left')
    
    if len(groupwise_df) > 0:
        print(f"Groupwise connections found: {len(groupwise_df):,}")
        print(f"Unique groups with connections: {groupwise_df['group_id'].nunique():,}")
        print(f"Total groupwise synapses: {groupwise_df['groupwise_synapses'].sum():,.0f}")
    else:
        print("⚠️  No connections found at this threshold!")
    
    return groupwise_df


def identify_premotor_groups(groupwise_df, bilateral_groups):
    """
    Identify premotor interneuron groups.
    """
    
    print("\n" + "="*70)
    print("IDENTIFYING PREMOTOR GROUPS")
    print("="*70 + "\n")
    
    if len(groupwise_df) == 0:
        print("⚠️  No premotor groups identified!")
        return pd.DataFrame()
    
    # Get unique premotor groups
    premotor_group_ids = groupwise_df['group_id'].unique()
    
    premotor_groups = bilateral_groups[
        bilateral_groups['group_id'].isin(premotor_group_ids)
    ].copy()
    
    # Add connectivity stats
    group_stats = groupwise_df.groupby('group_id').agg({
        'target_mn': 'count',
        'groupwise_synapses': 'sum'
    }).rename(columns={
        'target_mn': 'n_target_mns',
        'groupwise_synapses': 'total_synapses'
    })
    
    premotor_groups = premotor_groups.merge(group_stats, on='group_id', how='left')
    
    # Expand to individual neurons
    premotor_neurons = []
    for _, group in premotor_groups.iterrows():
        premotor_neurons.append(group['neuron_1'])
        if group['neuron_2'] is not None:
            premotor_neurons.append(group['neuron_2'])
    
    print(f"Premotor groups identified: {len(premotor_groups):,}")
    print(f"Total premotor neurons: {len(premotor_neurons):,}")
    print(f"  - From pairs: {len([g for _, g in premotor_groups.iterrows() if g['group_size']==2]) * 2}")
    print(f"  - From singletons: {len([g for _, g in premotor_groups.iterrows() if g['group_size']==1])}")
    
    return premotor_groups, premotor_neurons


def main():
    print("\n" + "="*70)
    print("GROUPWISE BILATERAL PAIR ANALYSIS")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    loader = ConnectomeDataLoader('config.yaml')
    connections, neurons, name_mapping = loader.load_all_data(verbose=False)
    
    # Load motor pools
    motor_pools = pd.read_csv('results/motor_neurons/motor_pools.csv')
    wing_mns = []
    for _, pool in motor_pools.iterrows():
        wing_mns.extend(eval(pool['motor_neuron_ids']))
    wing_mns = list(set(wing_mns))
    
    print(f"✓ Loaded {len(connections):,} connections")
    print(f"✓ Loaded {len(neurons):,} neurons")
    print(f"✓ Wing MNs: {len(wing_mns)}\n")
    
    # Identify bilateral pairs
    bilateral_groups = identify_bilateral_pairs(neurons, connections)
    
    # Save bilateral groups
    output_dir = Path('results/groupwise_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bilateral_groups.to_csv(output_dir / 'bilateral_groups.csv', index=False)
    print(f"\n✓ Saved: bilateral_groups.csv")
    
    # Test different groupwise thresholds
    thresholds = [50, 100]
    results = {}
    
    for thresh in thresholds:
        print(f"\n{'='*70}")
        print(f"TESTING GROUPWISE THRESHOLD: ≥{thresh} synapses")
        print(f"{'='*70}")
        
        groupwise_df = compute_groupwise_connectivity(
            bilateral_groups, connections, wing_mns,
            groupwise_threshold=thresh
        )
        
        if len(groupwise_df) > 0:
            premotor_groups, premotor_neurons = identify_premotor_groups(
                groupwise_df, bilateral_groups
            )
            
            results[thresh] = {
                'groupwise_connections': groupwise_df,
                'premotor_groups': premotor_groups,
                'premotor_neurons': premotor_neurons,
                'n_groups': len(premotor_groups),
                'n_neurons': len(premotor_neurons)
            }
            
            # Save
            groupwise_df.to_csv(
                output_dir / f'groupwise_connections_thresh{thresh}.csv',
                index=False
            )
            premotor_groups.to_csv(
                output_dir / f'premotor_groups_thresh{thresh}.csv',
                index=False
            )
            
            # Save neuron list
            pd.DataFrame({'neuron_id': premotor_neurons}).to_csv(
                output_dir / f'premotor_neurons_thresh{thresh}.csv',
                index=False
            )
        else:
            results[thresh] = None
    
    # Create comparison
    print("\n" + "="*70)
    print("GROUPWISE THRESHOLD COMPARISON")
    print("="*70 + "\n")
    
    comparison_data = []
    for thresh, result in results.items():
        if result is not None:
            comparison_data.append({
                'threshold': f'≥{thresh} (groupwise)',
                'n_groups': result['n_groups'],
                'n_neurons': result['n_neurons'],
                'n_connections': len(result['groupwise_connections'])
            })
            
            print(f"Threshold ≥{thresh} (groupwise):")
            print(f"  • Premotor groups: {result['n_groups']:,}")
            print(f"  • Premotor neurons: {result['n_neurons']:,}")
            print(f"  • Connections: {len(result['groupwise_connections']):,}\n")
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(output_dir / 'groupwise_threshold_comparison.csv', index=False)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()