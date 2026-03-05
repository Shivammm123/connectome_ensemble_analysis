"""
Cell Type Classification

Classify neurons from BANC connectome into functional categories:
- Descending Neurons (DNs) - Brain → VNC
- Motor Neurons (MNs) - VNC → Muscles  
- Interneurons (INs) - VNC local circuits
- Ascending Neurons (ANs) - VNC → Brain

Based on Azevedo et al. 2023 and Cheong et al. 2024 methodology.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def classify_neurons(neurons_df: pd.DataFrame):
    """
    Classify neurons by type using Super Class and Class columns.
    
    Parameters
    ----------
    neurons_df : pd.DataFrame
        Neuron attributes dataframe
    
    Returns
    -------
    pd.DataFrame
        Neurons with added 'neuron_type' column
    """
    
    print("Classifying neurons by type...")
    print("-"*70)
    
    # Create copy
    df = neurons_df.copy()
    
    # Initialize neuron_type column
    df['neuron_type'] = 'unknown'
    
    # Classify by Super Class
    if 'Super Class' in df.columns:
        df.loc[df['Super Class'] == 'descending', 'neuron_type'] = 'DN'
        df.loc[df['Super Class'] == 'motor', 'neuron_type'] = 'MN'
        df.loc[df['Super Class'] == 'ascending', 'neuron_type'] = 'AN'
        df.loc[df['Super Class'] == 'ventral_nerve_cord_intrinsic', 'neuron_type'] = 'IN'
        df.loc[df['Super Class'] == 'central_brain_intrinsic', 'neuron_type'] = 'brain_IN'
        df.loc[df['Super Class'] == 'sensory', 'neuron_type'] = 'sensory'
        df.loc[df['Super Class'] == 'glia', 'neuron_type'] = 'glia'
    
    # Count classifications
    counts = df['neuron_type'].value_counts()
    
    print("\nNeuron type distribution:")
    for ntype, count in counts.items():
        print(f"  {ntype:15s}: {count:6,} neurons")
    
    total_classified = len(df[df['neuron_type'] != 'unknown'])
    print(f"\nTotal classified: {total_classified:,} / {len(df):,} ({100*total_classified/len(df):.1f}%)")
    
    return df


def extract_wing_motor_neurons(df: pd.DataFrame):
    """
    Extract wing motor neurons specifically.
    
    Based on Function and Class columns.
    """
    
    print("\nExtracting wing motor neurons...")
    print("-"*70)
    
    # Filter to motor neurons
    mns = df[df['neuron_type'] == 'MN'].copy()
    
    if len(mns) == 0:
        print("  No motor neurons found!")
        return pd.DataFrame()
    
    # Identify wing MNs by function or class
    wing_keywords = ['wing', 'flight', 'DLM', 'DVM', 'basalar', 'axillary', 
                     'pleural', 'pleurosternal', 'tergotrochanter']
    
    is_wing_mn = pd.Series(False, index=mns.index)
    
    if 'Function' in mns.columns:
        for keyword in wing_keywords:
            is_wing_mn |= mns['Function'].str.contains(keyword, case=False, na=False)
    
    if 'Class' in mns.columns:
        for keyword in wing_keywords:
            is_wing_mn |= mns['Class'].str.contains(keyword, case=False, na=False)
    
    wing_mns = mns[is_wing_mn].copy()
    
    print(f"  Total motor neurons: {len(mns):,}")
    print(f"  Wing motor neurons: {len(wing_mns):,}")
    
    if len(wing_mns) > 0:
        print(f"\nSample wing MNs:")
        sample = wing_mns[['Root ID', 'Class', 'Function']].head(10)
        for idx, row in sample.iterrows():
            print(f"  {row['Root ID']}: {row.get('Class', 'N/A')[:40]}")
    
    return wing_mns


def extract_vnc_interneurons(df: pd.DataFrame):
    """
    Extract VNC interneurons.
    
    These are the premotor circuits we'll cluster.
    """
    
    print("\nExtracting VNC interneurons...")
    print("-"*70)
    
    ins = df[df['neuron_type'] == 'IN'].copy()
    
    print(f"  VNC interneurons: {len(ins):,}")
    
    # Further classify by location if available
    if 'Body Part' in ins.columns:
        body_parts = ins['Body Part'].value_counts().head(10)
        if len(body_parts) > 0:
            print(f"\n  Top body parts:")
            for part, count in body_parts.items():
                print(f"    {part}: {count:,}")
    
    return ins


def extract_descending_neurons(df: pd.DataFrame):
    """
    Extract descending neurons.
    
    These send commands from brain to VNC.
    """
    
    print("\nExtracting descending neurons...")
    print("-"*70)
    
    dns = df[df['neuron_type'] == 'DN'].copy()
    
    print(f"  Descending neurons: {len(dns):,}")
    
    # Check if we have the DNa/DNb neurons from earlier
    if 'Class' in dns.columns:
        dn_classes = dns['Class'].value_counts().head(10)
        if len(dn_classes) > 0:
            print(f"\n  Top DN classes:")
            for dn_class, count in dn_classes.items():
                print(f"    {dn_class}: {count:,}")
    
    return dns


def create_summary_plots(df: pd.DataFrame, output_dir: Path):
    """
    Create summary visualizations.
    """
    
    print("\nCreating summary plots...")
    print("-"*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    
    # 1. Neuron type distribution
    type_counts = df['neuron_type'].value_counts()
    
    ax = axes[0, 0]
    type_counts.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Number of Neurons', fontweight='bold')
    ax.set_title('Neuron Type Distribution', fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Add counts on bars
    for i, (idx, val) in enumerate(type_counts.items()):
        ax.text(val + 100, i, f'{val:,}', va='center')
    
    # 2. Super Class distribution (top 10)
    ax = axes[0, 1]
    if 'Super Class' in df.columns:
        super_counts = df['Super Class'].value_counts().head(10)
        super_counts.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Number of Neurons', fontweight='bold')
        ax.set_title('Top 10 Super Classes', fontweight='bold', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
    
    # 3. Function distribution (top 15)
    ax = axes[1, 0]
    if 'Function' in df.columns:
        func_counts = df[df['Function'].notna()]['Function'].value_counts().head(15)
        if len(func_counts) > 0:
            func_counts.plot(kind='barh', ax=ax, color='lightgreen')
            ax.set_xlabel('Number of Neurons', fontweight='bold')
            ax.set_title('Top 15 Functions', fontweight='bold', fontsize=14)
            ax.set_ylabel('')
            ax.grid(axis='x', alpha=0.3)
    
    # 4. Circuit-relevant neurons
    ax = axes[1, 1]
    circuit_neurons = df[df['neuron_type'].isin(['DN', 'IN', 'MN', 'AN'])]
    circuit_counts = circuit_neurons['neuron_type'].value_counts()
    
    colors = {'DN': '#FF6B6B', 'IN': '#4ECDC4', 'MN': '#95E1D3', 'AN': '#FFA07A'}
    bar_colors = [colors.get(x, 'gray') for x in circuit_counts.index]
    
    circuit_counts.plot(kind='bar', ax=ax, color=bar_colors)
    ax.set_xlabel('Neuron Type', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Circuit-Relevant Neurons', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    # Add counts on bars
    for i, (idx, val) in enumerate(circuit_counts.items()):
        ax.text(i, val + 50, f'{val:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cell_type_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: cell_type_summary.png")


def main():
    print("\n" + "="*70)
    print("CELL TYPE CLASSIFICATION")
    print("Preparing neurons for circuit analysis")
    print("="*70 + "\n")
    
    # Load data
    loader = ConnectomeDataLoader('config.yaml')
    connections, neurons, name_mapping = loader.load_all_data(verbose=False)
    print("✓ Data loaded")
    print(f"  Total neurons: {len(neurons):,}\n")
    
    # Classify neurons
    neurons_classified = classify_neurons(neurons)
    
    # Extract specific types
    wing_mns = extract_wing_motor_neurons(neurons_classified)
    vnc_ins = extract_vnc_interneurons(neurons_classified)
    dns = extract_descending_neurons(neurons_classified)
    
    # Create output directory
    output_dir = Path('results/cell_types')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save classified neurons
    print("\n" + "-"*70)
    print("Saving results")
    print("-"*70 + "\n")
    
    # Save full classification
    neurons_classified.to_csv(output_dir / 'all_neurons_classified.csv', index=False)
    print(f"  ✓ all_neurons_classified.csv ({len(neurons_classified):,} neurons)")
    
    # Save specific types
    if len(dns) > 0:
        dns.to_csv(output_dir / 'descending_neurons.csv', index=False)
        print(f"  ✓ descending_neurons.csv ({len(dns):,} neurons)")
    
    if len(vnc_ins) > 0:
        vnc_ins.to_csv(output_dir / 'vnc_interneurons.csv', index=False)
        print(f"  ✓ vnc_interneurons.csv ({len(vnc_ins):,} neurons)")
    
    if len(wing_mns) > 0:
        wing_mns.to_csv(output_dir / 'wing_motor_neurons.csv', index=False)
        print(f"  ✓ wing_motor_neurons.csv ({len(wing_mns):,} neurons)")
    
    # Save summary statistics
    summary = {
        'total_neurons': len(neurons_classified),
        'descending_neurons': len(dns),
        'vnc_interneurons': len(vnc_ins),
        'motor_neurons': len(neurons_classified[neurons_classified['neuron_type'] == 'MN']),
        'wing_motor_neurons': len(wing_mns),
        'ascending_neurons': len(neurons_classified[neurons_classified['neuron_type'] == 'AN']),
        'brain_interneurons': len(neurons_classified[neurons_classified['neuron_type'] == 'brain_IN']),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'classification_summary.csv', index=False)
    print(f"  ✓ classification_summary.csv")
    
    # Create plots
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    create_summary_plots(neurons_classified, fig_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("CLASSIFICATION COMPLETE")
    print("="*70 + "\n")
    
    print("Circuit-relevant neurons:")
    print(f"  Descending (DNs):     {len(dns):6,}")
    print(f"  VNC Interneurons:     {len(vnc_ins):6,}")
    print(f"  Motor Neurons (all):  {len(neurons_classified[neurons_classified['neuron_type'] == 'MN']):6,}")
    print(f"  Wing Motor Neurons:   {len(wing_mns):6,}")
    print(f"  Ascending (ANs):      {len(neurons_classified[neurons_classified['neuron_type'] == 'AN']):6,}")
    
    print(f"\nResults saved to: {output_dir}")
    print("\nNext step: Run muscle_mn_mapping.py to map MNs to muscles")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()