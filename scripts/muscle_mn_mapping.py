"""
Motor Neuron to Muscle Mapping

Organize wing motor neurons into functional pools by muscle target.

Based on Azevedo et al. 2023 wing muscle classification:
- Power muscles: DLM (4 types), DVM (3 types)
- Steering muscles: basalar (b1-3), axillary (III1-4, i1-2), haltere (hg1-4)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def classify_wing_mns_by_muscle(wing_mns: pd.DataFrame):
    """
    Classify wing motor neurons by muscle target.
    
    Muscle categories (from Azevedo et al. 2023):
    - Power muscles: DLM1-4, DVM1-3
    - Steering muscles: b1-3, III1-4, i1-2, hg1-4, ps1, ttm
    """
    
    print("Classifying wing MNs by muscle target...")
    print("-"*70)
    
    df = wing_mns.copy()
    df['muscle_type'] = 'unknown'
    df['muscle_group'] = 'unknown'
    df['specific_muscle'] = 'unknown'
    
    # Power muscle patterns
    power_patterns = {
        'DLM': r'DLM|dorsal.longitudinal',
        'DVM': r'DVM|dorsal.?ventral|dorsoventral',
    }
    
    # Steering muscle patterns  
    steering_patterns = {
        'basalar': r'\bb[123]\b|basalar',
        'axillary_III': r'III[1-4]|third.axillary',
        'axillary_i': r'\bi[12]\b|first.axillary',
        'haltere': r'hg[1-4]|haltere',
        'pleurosternal': r'ps1|pleurosternal',
        'tergotrochanter': r'ttm|tergotrochanter',
    }
    
    # Check multiple columns for muscle info
    search_columns = ['Primary Cell Type', 'Alternative Cell Type(s)', 'Class', 'Function']
    
    for idx, row in df.iterrows():
        # Combine text from all relevant columns
        search_text = ' '.join([
            str(row.get(col, '')) for col in search_columns if col in df.columns
        ])
        search_text = search_text.lower()
        
        # Check power muscles
        for muscle, pattern in power_patterns.items():
            if re.search(pattern, search_text, re.IGNORECASE):
                df.at[idx, 'muscle_type'] = 'power'
                df.at[idx, 'muscle_group'] = muscle
                
                # Extract specific muscle number (e.g., DLM1, DVM2a)
                match = re.search(r'(DLM|DVM)\d+[a-z]*', search_text, re.IGNORECASE)
                if match:
                    df.at[idx, 'specific_muscle'] = match.group(0).upper()
                else:
                    df.at[idx, 'specific_muscle'] = muscle
                break
        
        # Check steering muscles
        if df.at[idx, 'muscle_type'] == 'unknown':
            for muscle, pattern in steering_patterns.items():
                if re.search(pattern, search_text, re.IGNORECASE):
                    df.at[idx, 'muscle_type'] = 'steering'
                    df.at[idx, 'muscle_group'] = muscle
                    
                    # Extract specific muscle
                    match = re.search(pattern, search_text, re.IGNORECASE)
                    if match:
                        df.at[idx, 'specific_muscle'] = match.group(0)
                    else:
                        df.at[idx, 'specific_muscle'] = muscle
                    break
        
        # If still unknown, check if it's haltere control
        if df.at[idx, 'muscle_type'] == 'unknown':
            if 'haltere' in search_text:
                df.at[idx, 'muscle_type'] = 'haltere_control'
                df.at[idx, 'muscle_group'] = 'haltere'
        
        # Check for wing general
        if df.at[idx, 'muscle_type'] == 'unknown':
            if 'wing' in search_text and 'motor' in search_text:
                df.at[idx, 'muscle_type'] = 'wing_general'
                df.at[idx, 'muscle_group'] = 'wing'
    
    # Summary
    print("\nMuscle type distribution:")
    type_counts = df['muscle_type'].value_counts()
    for mtype, count in type_counts.items():
        print(f"  {mtype:20s}: {count:3d} MNs")
    
    print("\nMuscle group distribution:")
    group_counts = df['muscle_group'].value_counts()
    for group, count in group_counts.items():
        print(f"  {group:20s}: {count:3d} MNs")
    
    return df


def create_motor_pools(df: pd.DataFrame):
    """
    Create motor neuron pools (MNs targeting same muscle).
    """
    
    print("\n" + "-"*70)
    print("Creating motor neuron pools")
    print("-"*70 + "\n")
    
    pools = []
    
    # Group by specific muscle
    for muscle in df['specific_muscle'].unique():
        if muscle == 'unknown':
            continue
        
        pool_mns = df[df['specific_muscle'] == muscle]
        
        if len(pool_mns) > 0:
            pools.append({
                'muscle': muscle,
                'muscle_type': pool_mns['muscle_type'].iloc[0],
                'muscle_group': pool_mns['muscle_group'].iloc[0],
                'n_motor_neurons': len(pool_mns),
                'motor_neuron_ids': pool_mns['Root ID'].tolist()
            })
    
    pools_df = pd.DataFrame(pools)
    pools_df = pools_df.sort_values(['muscle_type', 'muscle'], ascending=[True, True])
    
    print(f"Total motor pools: {len(pools_df)}")
    print("\nMotor pools:")
    for _, pool in pools_df.iterrows():
        print(f"  {pool['muscle']:15s} ({pool['muscle_type']:12s}): {pool['n_motor_neurons']:2d} MNs")
    
    return pools_df


def create_mn_visualizations(df: pd.DataFrame, pools_df: pd.DataFrame, output_dir: Path):
    """
    Create visualization of motor neuron organization.
    """
    
    print("\n" + "-"*70)
    print("Creating visualizations")
    print("-"*70 + "\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    
    # 1. Muscle type distribution
    ax = axes[0, 0]
    type_counts = df['muscle_type'].value_counts()
    colors_type = {'power': '#FF6B6B', 'steering': '#4ECDC4', 
                   'wing_general': '#95E1D3', 'haltere_control': '#FFA07A',
                   'unknown': '#CCCCCC'}
    bar_colors = [colors_type.get(x, 'gray') for x in type_counts.index]
    
    type_counts.plot(kind='bar', ax=ax, color=bar_colors)
    ax.set_xlabel('Muscle Type', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Motor Neurons', fontweight='bold', fontsize=12)
    ax.set_title('Motor Neurons by Muscle Type', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add counts
    for i, (idx, val) in enumerate(type_counts.items()):
        ax.text(i, val + 0.5, str(val), ha='center', va='bottom', fontweight='bold')
    
    # 2. Muscle group distribution
    ax = axes[0, 1]
    group_counts = df['muscle_group'].value_counts()
    group_counts.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Number of Motor Neurons', fontweight='bold', fontsize=12)
    ax.set_title('Motor Neurons by Muscle Group', fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # 3. Motor pool sizes
    ax = axes[1, 0]
    if len(pools_df) > 0:
        pool_data = pools_df.sort_values('n_motor_neurons', ascending=True)
        
        colors_pool = []
        for mtype in pool_data['muscle_type']:
            colors_pool.append(colors_type.get(mtype, 'gray'))
        
        pool_data.plot(x='muscle', y='n_motor_neurons', kind='barh', 
                      ax=ax, color=colors_pool, legend=False)
        ax.set_xlabel('Number of Motor Neurons', fontweight='bold', fontsize=12)
        ax.set_ylabel('Muscle', fontweight='bold', fontsize=12)
        ax.set_title('Motor Pool Sizes', fontweight='bold', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
    
    # 4. Power vs Steering comparison
    ax = axes[1, 1]
    power_steering = df[df['muscle_type'].isin(['power', 'steering'])]['muscle_type'].value_counts()
    
    if len(power_steering) > 0:
        colors_ps = [colors_type.get(x, 'gray') for x in power_steering.index]
        power_steering.plot(kind='pie', ax=ax, colors=colors_ps, 
                           autopct='%1.1f%%', startangle=90)
        ax.set_ylabel('')
        ax.set_title('Power vs Steering Motor Neurons', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'motor_neuron_organization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: motor_neuron_organization.png")


def main():
    print("\n" + "="*70)
    print("MOTOR NEURON TO MUSCLE MAPPING")
    print("Organizing wing MNs by muscle targets")
    print("="*70 + "\n")
    
    # Load wing motor neurons from Step 1
    cell_types_dir = Path('results/cell_types')
    wing_mns_file = cell_types_dir / 'wing_motor_neurons.csv'
    
    if not wing_mns_file.exists():
        print("❌ Wing motor neurons file not found!")
        print("Run cell_type_classification.py first!")
        sys.exit(1)
    
    wing_mns = pd.read_csv(wing_mns_file)
    print(f"✓ Loaded {len(wing_mns)} wing motor neurons\n")
    
    # Classify by muscle target
    wing_mns_classified = classify_wing_mns_by_muscle(wing_mns)
    
    # Create motor pools
    motor_pools = create_motor_pools(wing_mns_classified)
    
    # Create output directory
    output_dir = Path('results/motor_neurons')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print("\n" + "-"*70)
    print("Saving results")
    print("-"*70 + "\n")
    
    wing_mns_classified.to_csv(output_dir / 'wing_mns_with_muscles.csv', index=False)
    print(f"  ✓ wing_mns_with_muscles.csv")
    
    motor_pools.to_csv(output_dir / 'motor_pools.csv', index=False)
    print(f"  ✓ motor_pools.csv")
    
    # Create visualizations
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    create_mn_visualizations(wing_mns_classified, motor_pools, fig_dir)
    
    # Summary
    print("\n" + "="*70)
    print("MUSCLE MAPPING COMPLETE")
    print("="*70 + "\n")
    
    print("Motor neuron summary:")
    print(f"  Total wing MNs:      {len(wing_mns_classified):3d}")
    print(f"  Power muscle MNs:    {len(wing_mns_classified[wing_mns_classified['muscle_type'] == 'power']):3d}")
    print(f"  Steering muscle MNs: {len(wing_mns_classified[wing_mns_classified['muscle_type'] == 'steering']):3d}")
    print(f"  Motor pools:         {len(motor_pools):3d}")
    
    if len(motor_pools) > 0:
        print(f"\nLargest motor pools:")
        top_pools = motor_pools.nlargest(5, 'n_motor_neurons')
        for _, pool in top_pools.iterrows():
            print(f"  {pool['muscle']:15s}: {pool['n_motor_neurons']:2d} MNs")
    
    print(f"\nResults saved to: {output_dir}")
    print("\nNext step: Run interneuron_hub_clustering.py")
    print("  This will find INs connecting to these motor pools!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()