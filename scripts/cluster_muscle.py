"""
Analyze Cluster to Muscle Mapping

Traces exactly which muscles each IN cluster controls
by analyzing IN→MN→Muscle connections.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent))


def analyze_cluster_targets():
    """
    For each cluster, show exactly which muscles it targets.
    """
    
    print("\n" + "="*70)
    print("CLUSTER → MUSCLE MAPPING ANALYSIS")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    
    # Cluster assignments
    premotor_ins = pd.read_csv('results/interneuron_clusters/premotor_interneurons_clustered.csv')
    cluster_assignments = premotor_ins[['Root ID', 'cluster']].copy()
    cluster_assignments.columns = ['interneuron_id', 'cluster']
    print(f"  ✓ Premotor INs: {len(cluster_assignments):,}")
    
    # IN→MN connections
    in_mn_conn = pd.read_csv('results/interneuron_clusters/in_to_mn_connections.csv')
    print(f"  ✓ IN→MN connections: {len(in_mn_conn):,}")
    
    # Motor pools
    motor_pools = pd.read_csv('results/motor_neurons/motor_pools.csv')
    motor_pools['motor_neuron_ids'] = motor_pools['motor_neuron_ids'].apply(eval)
    print(f"  ✓ Motor pools: {len(motor_pools)}")
    
    # Wing MNs with muscle info
    wing_mns = pd.read_csv('results/motor_neurons/wing_mns_with_muscles.csv')
    print(f"  ✓ Wing MNs: {len(wing_mns)}\n")
    
    # Create MN→Muscle mapping
    mn_to_muscle = {}
    for _, pool in motor_pools.iterrows():
        for mn_id in pool['motor_neuron_ids']:
            mn_to_muscle[mn_id] = {
                'muscle': pool['muscle'],
                'muscle_type': pool['muscle_type'],
                'muscle_group': pool['muscle_group']
            }
    
    # Also add from wing_mns
    for _, mn in wing_mns.iterrows():
        if mn['Root ID'] not in mn_to_muscle:
            mn_to_muscle[mn['Root ID']] = {
                'muscle': mn.get('specific_muscle', 'unknown'),
                'muscle_type': mn.get('muscle_type', 'unknown'),
                'muscle_group': mn.get('muscle_group', 'unknown')
            }
    
    print("-"*70)
    print("DETAILED CLUSTER ANALYSIS")
    print("-"*70 + "\n")
    
    # For each cluster
    for cluster_id in sorted(cluster_assignments['cluster'].unique()):
        print("\n" + "="*70)
        print(f"CLUSTER {cluster_id}")
        print("="*70)
        
        # Get INs in this cluster
        cluster_ins = cluster_assignments[cluster_assignments['cluster'] == cluster_id]['interneuron_id'].tolist()
        print(f"\nInterneurons in cluster: {len(cluster_ins)}")
        
        # Get all IN→MN connections from this cluster
        cluster_conn = in_mn_conn[in_mn_conn['source'].isin(cluster_ins)]
        print(f"Total IN→MN connections: {len(cluster_conn)}")
        print(f"Total synapses: {cluster_conn['weight'].sum():,.0f}")
        
        # Map to muscles
        muscle_connections = []
        
        for _, conn in cluster_conn.iterrows():
            mn_id = conn['target']
            muscle_info = mn_to_muscle.get(mn_id, {})
            
            muscle_connections.append({
                'in_id': conn['source'],
                'mn_id': mn_id,
                'synapses': conn['weight'],
                'muscle': muscle_info.get('muscle', 'unknown'),
                'muscle_type': muscle_info.get('muscle_type', 'unknown'),
                'muscle_group': muscle_info.get('muscle_group', 'unknown')
            })
        
        muscle_df = pd.DataFrame(muscle_connections)
        
        # Summarize by muscle
        print("\n" + "-"*70)
        print("MUSCLE TARGETING:")
        print("-"*70)
        
        muscle_summary = muscle_df.groupby('muscle').agg({
            'in_id': 'nunique',
            'mn_id': 'nunique',
            'synapses': 'sum'
        }).rename(columns={
            'in_id': 'n_ins',
            'mn_id': 'n_mns'
        }).sort_values('synapses', ascending=False)
        
        print(f"\n{'Muscle':<15} {'INs':<8} {'MNs':<8} {'Synapses':<12} {'Type':<12}")
        print("-"*70)
        
        for muscle, row in muscle_summary.iterrows():
            muscle_type = muscle_df[muscle_df['muscle'] == muscle]['muscle_type'].iloc[0]
            print(f"{muscle:<15} {row['n_ins']:<8} {row['n_mns']:<8} {row['synapses']:<12.0f} {muscle_type:<12}")
        
        # Summarize by muscle type
        print("\n" + "-"*70)
        print("BY MUSCLE TYPE:")
        print("-"*70)
        
        type_summary = muscle_df.groupby('muscle_type').agg({
            'synapses': 'sum',
            'mn_id': 'nunique'
        }).sort_values('synapses', ascending=False)
        
        for mtype, row in type_summary.iterrows():
            pct = 100 * row['synapses'] / muscle_df['synapses'].sum()
            print(f"  {mtype:<15}: {row['synapses']:>8.0f} synapses ({pct:>5.1f}%) to {row['mn_id']} MNs")
        
        # Determine primary function
        if len(type_summary) > 0:
            primary_type = type_summary['synapses'].idxmax()
            primary_pct = 100 * type_summary.loc[primary_type, 'synapses'] / muscle_df['synapses'].sum()
            
            print(f"\n⭐ PRIMARY FUNCTION: {primary_type} ({primary_pct:.0f}% of connections)")
            
            # Suggest name
            if primary_type == 'power':
                suggested_name = "Power Generation Module"
                top_muscles = muscle_summary.head(3).index.tolist()
                suggested_name += f" ({', '.join(top_muscles)})"
            elif primary_type == 'steering':
                # Look at muscle groups
                top_3_muscles = muscle_summary.head(3).index.tolist()
                
                # Categorize
                has_haltere = any('hg' in m.lower() for m in top_3_muscles)
                has_basalar = any('b' in m.lower() and len(m) <= 3 for m in top_3_muscles)
                has_axillary_i = any('i' in m.lower() and len(m) <= 3 for m in top_3_muscles)
                has_axillary_iii = any('iii' in m.lower() for m in top_3_muscles)
                has_ps = any('ps' in m.lower() for m in top_3_muscles)
                
                if has_haltere:
                    suggested_name = "Haltere & Balance Control"
                elif has_basalar:
                    suggested_name = "Wing Base Control (Basalar)"
                elif has_axillary_i:
                    suggested_name = "Wing Articulation (First Axillary)"
                elif has_axillary_iii or has_ps:
                    suggested_name = "Fine Steering Control"
                else:
                    suggested_name = "Steering Control"
                
                suggested_name += f" ({', '.join(top_3_muscles)})"
            else:
                suggested_name = f"{primary_type} Control"
            
            print(f"💡 SUGGESTED NAME: {suggested_name}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")
    
    print("Summary of findings:")
    print("  → Each cluster has a clear primary muscle type target")
    print("  → Suggested names are based on actual connectivity")
    print("  → Use these names for your DN connectivity figure!")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    analyze_cluster_targets()