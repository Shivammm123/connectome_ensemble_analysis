"""
Complete Pathway Database

Creates comprehensive database files documenting all discovered pathways.
Includes CSV exports and summary reports for further analysis.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def build_complete_pathway_database(
    dn_class_df: pd.DataFrame,
    dn_in_conn: pd.DataFrame,
    in_mn_conn: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    motor_pools: pd.DataFrame,
    loader: ConnectomeDataLoader
):
    """
    Build complete pathway database with all DN→IN→MN→Muscle chains.
    """
    
    print("Building complete pathway database...")
    print("-"*70)
    
    pathways = []
    
    # Map MNs to muscles
    mn_to_muscle = {}
    for _, pool in motor_pools.iterrows():
        for mn_id in pool['motor_neuron_ids']:
            mn_to_muscle[mn_id] = {
                'muscle': pool['muscle'],
                'muscle_type': pool['muscle_type'],
                'muscle_group': pool['muscle_group']
            }
    
    # Map INs to clusters
    in_to_cluster = dict(zip(cluster_assignments['interneuron_id'], 
                            cluster_assignments['cluster']))
    
    # Build pathways
    pathway_id = 1
    
    for _, dn_row in dn_class_df.iterrows():
        dn_id = dn_row['dn_id']
        dn_name = dn_row['dn_name']
        
        # Get INs this DN connects to
        dn_targets = dn_in_conn[dn_in_conn['source'] == dn_id]
        
        for _, dn_in_edge in dn_targets.iterrows():
            in_id = dn_in_edge['target']
            dn_in_weight = dn_in_edge['weight']
            
            # Get cluster
            cluster = in_to_cluster.get(in_id, None)
            
            # Get MNs this IN connects to
            in_targets = in_mn_conn[in_mn_conn['source'] == in_id]
            
            for _, in_mn_edge in in_targets.iterrows():
                mn_id = in_mn_edge['target']
                in_mn_weight = in_mn_edge['weight']
                
                # Get muscle info
                muscle_info = mn_to_muscle.get(mn_id, {})
                
                pathways.append({
                    'pathway_id': pathway_id,
                    'dn_id': dn_id,
                    'dn_name': dn_name,
                    'dn_specialization': dn_row['specialization'],
                    'in_id': in_id,
                    'in_cluster': cluster,
                    'mn_id': mn_id,
                    'muscle': muscle_info.get('muscle', 'unknown'),
                    'muscle_type': muscle_info.get('muscle_type', 'unknown'),
                    'muscle_group': muscle_info.get('muscle_group', 'unknown'),
                    'dn_in_synapses': dn_in_weight,
                    'in_mn_synapses': in_mn_weight,
                    'pathway_strength': dn_in_weight * in_mn_weight,
                    'pathway_length': 3  # DN→IN→MN
                })
                
                pathway_id += 1
    
    pathways_df = pd.DataFrame(pathways)
    
    print(f"  Total pathways: {len(pathways_df):,}")
    print(f"  Unique DN→IN→MN chains: {len(pathways_df[['dn_id','in_id','mn_id']].drop_duplicates()):,}")
    print(f"  DNs involved: {pathways_df['dn_id'].nunique()}")
    print(f"  INs involved: {pathways_df['in_id'].nunique()}")
    print(f"  MNs involved: {pathways_df['mn_id'].nunique()}")
    
    return pathways_df


def create_pathway_summaries(pathways_df: pd.DataFrame):
    """
    Create summary statistics for pathways.
    """
    
    print("\nCreating pathway summaries...")
    print("-"*70)
    
    summaries = {}
    
    # By DN
    dn_summary = pathways_df.groupby('dn_name').agg({
        'pathway_id': 'count',
        'in_id': 'nunique',
        'mn_id': 'nunique',
        'muscle': lambda x: x.nunique(),
        'pathway_strength': 'sum'
    }).rename(columns={
        'pathway_id': 'n_pathways',
        'in_id': 'n_ins',
        'mn_id': 'n_mns',
        'muscle': 'n_muscles'
    }).sort_values('pathway_strength', ascending=False)
    
    summaries['by_dn'] = dn_summary
    print(f"  ✓ DN summary ({len(dn_summary)} DNs)")
    
    # By cluster
    cluster_summary = pathways_df[pathways_df['in_cluster'].notna()].groupby('in_cluster').agg({
        'pathway_id': 'count',
        'dn_id': 'nunique',
        'mn_id': 'nunique',
        'muscle_type': lambda x: x.mode()[0] if len(x) > 0 else 'unknown',
        'pathway_strength': 'sum'
    }).rename(columns={
        'pathway_id': 'n_pathways',
        'dn_id': 'n_dns',
        'mn_id': 'n_mns'
    })
    
    summaries['by_cluster'] = cluster_summary
    print(f"  ✓ Cluster summary ({len(cluster_summary)} clusters)")
    
    # By muscle
    muscle_summary = pathways_df[pathways_df['muscle'] != 'unknown'].groupby('muscle').agg({
        'pathway_id': 'count',
        'dn_id': 'nunique',
        'in_id': 'nunique',
        'muscle_type': 'first',
        'pathway_strength': 'sum'
    }).rename(columns={
        'pathway_id': 'n_pathways',
        'dn_id': 'n_dns',
        'in_id': 'n_ins'
    }).sort_values('pathway_strength', ascending=False)
    
    summaries['by_muscle'] = muscle_summary
    print(f"  ✓ Muscle summary ({len(muscle_summary)} muscles)")
    
    # By muscle type
    type_summary = pathways_df.groupby('muscle_type').agg({
        'pathway_id': 'count',
        'dn_id': 'nunique',
        'in_id': 'nunique',
        'mn_id': 'nunique',
        'pathway_strength': 'sum'
    }).rename(columns={
        'pathway_id': 'n_pathways',
        'dn_id': 'n_dns',
        'in_id': 'n_ins',
        'mn_id': 'n_mns'
    })
    
    summaries['by_type'] = type_summary
    print(f"  ✓ Type summary ({len(type_summary)} types)")
    
    return summaries


def create_connectivity_matrices(pathways_df: pd.DataFrame):
    """
    Create connectivity matrices for different levels.
    """
    
    print("\nCreating connectivity matrices...")
    print("-"*70)
    
    matrices = {}
    
    # DN × Cluster matrix
    dn_cluster = pathways_df[pathways_df['in_cluster'].notna()].groupby(
        ['dn_name', 'in_cluster']
    )['pathway_strength'].sum().unstack(fill_value=0)
    
    matrices['dn_cluster'] = dn_cluster
    print(f"  ✓ DN×Cluster matrix ({dn_cluster.shape[0]}×{dn_cluster.shape[1]})")
    
    # Cluster × Muscle matrix
    cluster_muscle = pathways_df[
        (pathways_df['in_cluster'].notna()) & 
        (pathways_df['muscle'] != 'unknown')
    ].groupby(['in_cluster', 'muscle'])['pathway_strength'].sum().unstack(fill_value=0)
    
    matrices['cluster_muscle'] = cluster_muscle
    print(f"  ✓ Cluster×Muscle matrix ({cluster_muscle.shape[0]}×{cluster_muscle.shape[1]})")
    
    # DN × Muscle type matrix
    dn_type = pathways_df.groupby(
        ['dn_name', 'muscle_type']
    )['pathway_strength'].sum().unstack(fill_value=0)
    
    matrices['dn_type'] = dn_type
    print(f"  ✓ DN×Type matrix ({dn_type.shape[0]}×{dn_type.shape[1]})")
    
    return matrices


def export_database(
    pathways_df: pd.DataFrame,
    summaries: dict,
    matrices: dict,
    output_dir: Path
):
    """
    Export all database files.
    """
    
    print("\nExporting database files...")
    print("-"*70)
    
    # Main pathway database
    pathways_df.to_csv(output_dir / 'complete_pathways_database.csv', index=False)
    print(f"  ✓ complete_pathways_database.csv ({len(pathways_df):,} pathways)")
    
    # Summaries
    for name, summary in summaries.items():
        filename = f'pathway_summary_{name}.csv'
        summary.to_csv(output_dir / filename)
        print(f"  ✓ {filename}")
    
    # Matrices
    for name, matrix in matrices.items():
        filename = f'connectivity_matrix_{name}.csv'
        matrix.to_csv(output_dir / filename)
        print(f"  ✓ {filename}")
    
    # JSON metadata
    metadata = {
        'total_pathways': len(pathways_df),
        'unique_dns': pathways_df['dn_id'].nunique(),
        'unique_ins': pathways_df['in_id'].nunique(),
        'unique_mns': pathways_df['mn_id'].nunique(),
        'unique_muscles': pathways_df[pathways_df['muscle']!='unknown']['muscle'].nunique(),
        'total_pathway_strength': float(pathways_df['pathway_strength'].sum()),
        'mean_pathway_strength': float(pathways_df['pathway_strength'].mean()),
        'clusters_involved': int(pathways_df['in_cluster'].nunique())
    }
    
    with open(output_dir / 'database_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ database_metadata.json")


def main():
    print("\n" + "="*70)
    print("COMPLETE PATHWAY DATABASE")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    print("-"*70 + "\n")
    
    loader = ConnectomeDataLoader('config.yaml')
    
    dn_class = pd.read_csv('results/dn_pathways/dn_classifications.csv')
    print(f"  ✓ DN classifications: {len(dn_class)}")
    
    dn_in_conn = pd.read_csv('results/dn_pathways/dn_to_in_connections.csv')
    print(f"  ✓ DN→IN connections: {len(dn_in_conn):,}")
    
    in_mn_conn = pd.read_csv('results/interneuron_clusters/in_to_mn_connections.csv')
    print(f"  ✓ IN→MN connections: {len(in_mn_conn):,}")
    
    cluster_assignments = pd.read_csv('results/interneuron_clusters/premotor_interneurons_clustered.csv')
    cluster_assignments = cluster_assignments[['Root ID', 'cluster']].copy()
    cluster_assignments.columns = ['interneuron_id', 'cluster']
    print(f"  ✓ Cluster assignments: {len(cluster_assignments):,}")
    
    motor_pools = pd.read_csv('results/motor_neurons/motor_pools.csv')
    motor_pools['motor_neuron_ids'] = motor_pools['motor_neuron_ids'].apply(eval)
    print(f"  ✓ Motor pools: {len(motor_pools)}\n")
    
    # Build database
    pathways = build_complete_pathway_database(
        dn_class,
        dn_in_conn,
        in_mn_conn,
        cluster_assignments,
        motor_pools,
        loader
    )
    
    # Create summaries
    summaries = create_pathway_summaries(pathways)
    
    # Create matrices
    matrices = create_connectivity_matrices(pathways)
    
    # Export
    output_dir = Path('results/pathway_database')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    export_database(pathways, summaries, matrices, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("PATHWAY DATABASE COMPLETE")
    print("="*70 + "\n")
    
    print(f"Database statistics:")
    print(f"  Total pathways documented: {len(pathways):,}")
    print(f"  Unique DNs: {pathways['dn_id'].nunique()}")
    print(f"  Unique INs: {pathways['in_id'].nunique()}")
    print(f"  Unique MNs: {pathways['mn_id'].nunique()}")
    print(f"  Muscles targeted: {pathways[pathways['muscle']!='unknown']['muscle'].nunique()}")
    
    print(f"\nFiles created:")
    print(f"  • complete_pathways_database.csv - Full pathway table")
    print(f"  • pathway_summary_*.csv - Summary statistics")
    print(f"  • connectivity_matrix_*.csv - Connectivity matrices")
    print(f"  • database_metadata.json - Metadata")
    
    print(f"\nLocation: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()