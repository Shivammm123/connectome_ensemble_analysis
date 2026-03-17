"""
Candidate Prioritization System

Ranks all interneurons by experimental priority based on:
- Hub connectivity
- Cluster importance
- Integration properties
- DN input strength

Provides top candidates for functional studies.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def score_hub_importance(hub_neurons: pd.DataFrame):
    """
    Score INs based on hub properties.
    """
    
    # Normalize metrics to 0-100
    scores = hub_neurons.copy()
    
    # Degree centrality score (0-100)
    if 'degree_centrality' in scores.columns:
        max_deg = scores['degree_centrality'].max()
        scores['hub_score'] = 100 * scores['degree_centrality'] / max_deg if max_deg > 0 else 0
    else:
        scores['hub_score'] = 0
    
    # MN target count score
    if 'n_mn_targets' in scores.columns:
        max_targets = scores['n_mn_targets'].max()
        scores['target_score'] = 100 * scores['n_mn_targets'] / max_targets if max_targets > 0 else 0
    else:
        scores['target_score'] = 0
    
    # Betweenness centrality score
    if 'betweenness_centrality' in scores.columns:
        max_between = scores['betweenness_centrality'].max()
        scores['betweenness_score'] = 100 * scores['betweenness_centrality'] / max_between if max_between > 0 else 0
    else:
        scores['betweenness_score'] = 0
    
    return scores


def score_integration(integrative_ins: pd.DataFrame):
    """
    Score INs based on integration properties.
    """
    
    scores = integrative_ins.copy()
    
    # Integration score (already computed)
    if 'integration_score' in scores.columns:
        max_int = scores['integration_score'].max()
        scores['integration_score_norm'] = 100 * scores['integration_score'] / max_int if max_int > 0 else 0
    else:
        scores['integration_score_norm'] = 0
    
    # Number of muscle types
    if 'n_muscle_types' in scores.columns:
        scores['diversity_score'] = 100 * (scores['n_muscle_types'] - 1) / 2  # Max 3 types
    else:
        scores['diversity_score'] = 0
    
    return scores


def score_cluster_membership(
    in_ids: list,
    cluster_assignments: pd.DataFrame,
    cluster_info: pd.DataFrame
):
    """
    Score INs based on their cluster importance.
    """
    
    # Map cluster importance (power cluster = highest)
    cluster_importance = {}
    for _, cluster in cluster_info.iterrows():
        cid = cluster['cluster_id']
        
        # Power cluster gets highest score
        if cluster['functional_type'] == 'power_control':
            importance = 100
        else:
            # Steering clusters by size
            importance = 50 + (cluster['n_interneurons'] / cluster_info['n_interneurons'].max()) * 50
        
        cluster_importance[cid] = importance
    
    # Score each IN
    scores = []
    for in_id in in_ids:
        cluster_match = cluster_assignments[cluster_assignments['interneuron_id'] == in_id]
        
        if len(cluster_match) > 0:
            cluster = cluster_match['cluster'].values[0]
            score = cluster_importance.get(cluster, 0)
        else:
            score = 0
        
        scores.append({
            'interneuron_id': in_id,
            'cluster_score': score
        })
    
    return pd.DataFrame(scores)


def score_dn_input(
    in_ids: list,
    dn_in_conn: pd.DataFrame
):
    """
    Score INs based on DN input strength.
    """
    
    # Count DN inputs for each IN
    dn_inputs = dn_in_conn.groupby('target').agg({
        'source': 'count',
        'weight': 'sum'
    }).rename(columns={'source': 'n_dns', 'weight': 'total_dn_input'})
    
    scores = []
    for in_id in in_ids:
        if in_id in dn_inputs.index:
            n_dns = dn_inputs.loc[in_id, 'n_dns']
            total_input = dn_inputs.loc[in_id, 'total_dn_input']
        else:
            n_dns = 0
            total_input = 0
        
        scores.append({
            'interneuron_id': in_id,
            'n_dn_inputs': n_dns,
            'dn_input_strength': total_input
        })
    
    scores_df = pd.DataFrame(scores)
    
    # Normalize
    if len(scores_df) > 0:
        max_dns = scores_df['n_dn_inputs'].max()
        max_strength = scores_df['dn_input_strength'].max()
        
        norm_dns = scores_df['n_dn_inputs'] / max_dns if max_dns > 0 else 0
        norm_strength = scores_df['dn_input_strength'] / max_strength if max_strength > 0 else 0
        scores_df['dn_score'] = 50 * norm_dns + 50 * norm_strength
    else:
        scores_df['dn_score'] = 0
    
    return scores_df


def combine_scores_and_rank(
    hub_scores: pd.DataFrame,
    integration_scores: pd.DataFrame,
    cluster_scores: pd.DataFrame,
    dn_scores: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    loader: ConnectomeDataLoader
):
    """
    Combine all scores and create final ranking.
    """
    
    print("Combining scores and ranking...")
    print("-"*70)
    
    # Start with all premotor INs
    all_ins = cluster_assignments[['interneuron_id', 'cluster']].copy()
    
    # Merge hub scores
    if 'interneuron_id' in hub_scores.columns:
        all_ins = all_ins.merge(
            hub_scores[['interneuron_id', 'hub_score', 'target_score', 'n_mn_targets']],
            on='interneuron_id',
            how='left'
        )
    else:
        all_ins = all_ins.merge(
            hub_scores.rename(columns={hub_scores.columns[0]: 'interneuron_id'})[['interneuron_id', 'hub_score', 'target_score', 'n_mn_targets']],
            on='interneuron_id',
            how='left'
        )
    
    all_ins['hub_score'] = all_ins['hub_score'].fillna(0)
    all_ins['target_score'] = all_ins['target_score'].fillna(0)
    all_ins['n_mn_targets'] = all_ins['n_mn_targets'].fillna(0)
    
    # Merge integration scores
    if len(integration_scores) > 0:
        all_ins = all_ins.merge(
            integration_scores[['in_id', 'integration_score_norm', 'n_muscle_types']].rename(columns={'in_id': 'interneuron_id'}),
            on='interneuron_id',
            how='left'
        )
        all_ins['integration_score_norm'] = all_ins['integration_score_norm'].fillna(0)
        all_ins['n_muscle_types'] = all_ins['n_muscle_types'].fillna(1)
    else:
        all_ins['integration_score_norm'] = 0
        all_ins['n_muscle_types'] = 1
    
    # Merge cluster scores
    all_ins = all_ins.merge(cluster_scores, on='interneuron_id', how='left')
    all_ins['cluster_score'] = all_ins['cluster_score'].fillna(0)
    
    # Merge DN scores
    all_ins = all_ins.merge(dn_scores[['interneuron_id', 'dn_score', 'n_dn_inputs']], 
                           on='interneuron_id', how='left')
    all_ins['dn_score'] = all_ins['dn_score'].fillna(0)
    all_ins['n_dn_inputs'] = all_ins['n_dn_inputs'].fillna(0)
    
    # Calculate composite score
    # Weights: Hub 30%, Cluster 25%, DN input 25%, Integration 20%
    all_ins['composite_score'] = (
        0.30 * all_ins['hub_score'] +
        0.25 * all_ins['cluster_score'] +
        0.25 * all_ins['dn_score'] +
        0.20 * all_ins['integration_score_norm']
    )
    
    # Add neuron names
    all_ins['neuron_name'] = all_ins['interneuron_id'].apply(
        lambda x: loader.get_neuron_name(x)
    )
    
    # Rank
    all_ins = all_ins.sort_values('composite_score', ascending=False)
    all_ins['rank'] = range(1, len(all_ins) + 1)
    
    # Assign priority tier
    n_ins = len(all_ins)
    all_ins['priority_tier'] = pd.cut(
        all_ins['rank'],
        bins=[0, n_ins*0.05, n_ins*0.20, n_ins],
        labels=['Tier 1: High Priority', 'Tier 2: Medium Priority', 'Tier 3: Lower Priority']
    )
    
    print(f"  Ranked {len(all_ins):,} interneurons")
    print(f"  Tier 1 (top 5%): {len(all_ins[all_ins['priority_tier']=='Tier 1: High Priority'])}")
    print(f"  Tier 2 (top 20%): {len(all_ins[all_ins['priority_tier']=='Tier 2: Medium Priority'])}")
    
    return all_ins


def suggest_experiments(candidates: pd.DataFrame, top_n: int = 20):
    """
    Suggest experiments for top candidates based on their properties.
    """
    
    print(f"\nSuggesting experiments for top {top_n} candidates...")
    print("-"*70)
    
    top_candidates = candidates.head(top_n)
    
    experiments = []
    
    for _, candidate in top_candidates.iterrows():
        rank = candidate['rank']
        in_id = candidate['interneuron_id']
        name = candidate['neuron_name']
        cluster = candidate['cluster']
        hub_score = candidate['hub_score']
        integration = candidate['integration_score_norm']
        dn_inputs = candidate['n_dn_inputs']
        
        # Determine experiment type
        exp_types = []
        
        # Silencing/activation
        if hub_score > 70:
            exp_types.append("Optogenetic silencing (expect strong flight deficit)")
            exp_types.append("Calcium imaging during flight")
        elif hub_score > 40:
            exp_types.append("Optogenetic activation (characterize motor output)")
        
        # Integration experiments
        if integration > 50:
            exp_types.append("Test during maneuvers (integrates power+steering)")
            exp_types.append("Compare straight flight vs turns")
        
        # DN pathway
        if dn_inputs > 2:
            exp_types.append("Trace upstream DNs (finds behavior triggers)")
        
        # Cluster-specific
        if cluster == 4:
            exp_types.append("Test wing beat frequency modulation")
            exp_types.append("Power control characterization")
        else:
            exp_types.append("Test steering precision")
            exp_types.append("Measure wing angle control")
        
        experiments.append({
            'rank': rank,
            'interneuron_id': in_id,
            'neuron_name': name,
            'cluster': cluster,
            'suggested_experiments': '; '.join(exp_types) if exp_types else 'General characterization'
        })
    
    return pd.DataFrame(experiments)


def main():
    print("\n" + "="*70)
    print("CANDIDATE PRIORITIZATION SYSTEM")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    print("-"*70 + "\n")
    
    loader = ConnectomeDataLoader('config.yaml')
    
    hub_neurons = pd.read_csv('results/interneuron_clusters/hub_interneurons.csv')
    print(f"  ✓ Hub neurons: {len(hub_neurons):,}")
    
    try:
        integrative_ins = pd.read_csv('results/circuit_modules/integrative_interneurons.csv')
        print(f"  ✓ Integrative INs: {len(integrative_ins)}")
    except:
        integrative_ins = pd.DataFrame()
        print(f"  ⚠ No integrative INs file")
    
    cluster_assignments = pd.read_csv('results/interneuron_clusters/premotor_interneurons_clustered.csv')
    cluster_assignments = cluster_assignments[['Root ID', 'cluster']].copy()
    cluster_assignments.columns = ['interneuron_id', 'cluster']
    print(f"  ✓ Cluster assignments: {len(cluster_assignments):,}")
    
    cluster_info = pd.read_csv('results/interneuron_clusters/cluster_characteristics.csv')
    print(f"  ✓ Cluster info: {len(cluster_info)}")
    
    dn_in_conn = pd.read_csv('results/dn_pathways/dn_to_in_connections.csv')
    print(f"  ✓ DN→IN connections: {len(dn_in_conn):,}\n")
    
    # Score components
    print("-"*70)
    print("Computing scores...")
    print("-"*70 + "\n")
    
    hub_scores = score_hub_importance(hub_neurons)
    print(f"  ✓ Hub scores")
    
    integration_scores = score_integration(integrative_ins) if len(integrative_ins) > 0 else pd.DataFrame()
    print(f"  ✓ Integration scores")
    
    all_in_ids = cluster_assignments['interneuron_id'].tolist()
    cluster_scores = score_cluster_membership(all_in_ids, cluster_assignments, cluster_info)
    print(f"  ✓ Cluster scores")
    
    dn_scores = score_dn_input(all_in_ids, dn_in_conn)
    print(f"  ✓ DN input scores\n")
    
    # Combine and rank
    print("-"*70)
    ranked_candidates = combine_scores_and_rank(
        hub_scores,
        integration_scores,
        cluster_scores,
        dn_scores,
        cluster_assignments,
        loader
    )
    
    # Suggest experiments
    experiment_suggestions = suggest_experiments(ranked_candidates, top_n=50)
    
    # Save results
    print("\n" + "-"*70)
    print("Saving results...")
    print("-"*70 + "\n")
    
    output_dir = Path('results/candidate_prioritization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full ranked list
    ranked_candidates.to_csv(output_dir / 'all_interneurons_ranked.csv', index=False)
    print(f"  ✓ all_interneurons_ranked.csv ({len(ranked_candidates):,} INs)")
    
    # Top candidates
    top_50 = ranked_candidates.head(50)
    top_50.to_csv(output_dir / 'top_50_candidates.csv', index=False)
    print(f"  ✓ top_50_candidates.csv")
    
    # Experiment suggestions
    experiment_suggestions.to_csv(output_dir / 'experimental_suggestions.csv', index=False)
    print(f"  ✓ experimental_suggestions.csv")
    
    # Tier summaries
    for tier in ranked_candidates['priority_tier'].unique():
        tier_data = ranked_candidates[ranked_candidates['priority_tier'] == tier]
        tier_name = tier.split(':')[0].replace(' ', '_').lower()
        tier_data.to_csv(output_dir / f'{tier_name}_candidates.csv', index=False)
        print(f"  ✓ {tier_name}_candidates.csv ({len(tier_data)} INs)")
    
    # Summary
    print("\n" + "="*70)
    print("CANDIDATE PRIORITIZATION COMPLETE")
    print("="*70 + "\n")
    
    print(f"Priority tiers:")
    for tier in ['Tier 1: High Priority', 'Tier 2: Medium Priority', 'Tier 3: Lower Priority']:
        count = len(ranked_candidates[ranked_candidates['priority_tier'] == tier])
        print(f"  {tier}: {count:,} INs")
    
    print(f"\nTop 10 experimental candidates:")
    for i, (_, cand) in enumerate(ranked_candidates.head(10).iterrows(), 1):
        print(f"  {i}. {cand['neuron_name'][:40]:40s} (Score: {cand['composite_score']:.1f})")
        print(f"     Cluster {int(cand['cluster'])}, {int(cand['n_mn_targets'])} MN targets, {int(cand['n_dn_inputs'])} DN inputs")
    
    print(f"\nResults: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()