"""
Clean DN Connectivity Figure

Simple, accurate figure showing DN → IN Module → MN pathways.
- Cluster 4: Power Control
- Clusters 1,2,3,5: Steering Control (A, B, C, D)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def create_dn_connectivity_figure(
    dn_class_df,
    cluster_info,
    motor_pools,
    loader,
    top_n_dns=20
):
    """
    Create comprehensive DN connectivity figure.
    """
    
    # Assign simple, accurate names
    module_names = {}
    module_colors = {}
    steering_labels = ['A', 'B', 'C', 'D']
    steering_idx = 0
    
    for _, cluster in cluster_info.iterrows():
        cid = cluster['cluster_id']
        
        if cluster['functional_type'] == 'power_control':
            module_names[cid] = 'Power Control'
            module_colors[cid] = '#FF6B6B'
        else:
            module_names[cid] = f'Steering {steering_labels[steering_idx]}'
            module_colors[cid] = '#4ECDC4'
            steering_idx += 1
    
    # Get top DNs
    top_dns = dn_class_df.head(top_n_dns)
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(24, 10), dpi=300)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, figure=fig, wspace=0.3, 
                  left=0.05, right=0.95, top=0.90, bottom=0.08)
    
    # Title
    fig.suptitle('Descending Neuron Connectivity to Flight Control Modules\nDN → Premotor IN Module → Motor Neuron Pathways',
                fontsize=20, fontweight='bold', y=0.97)
    
    # ===== PANEL A: DN → Module Connectivity Heatmap =====
    ax_a = fig.add_subplot(gs[0, 0:2])
    
    # Build connectivity matrix
    dn_names = [name[:35] for name in top_dns['dn_name']]
    cluster_ids = sorted(cluster_info['cluster_id'].unique())
    module_labels = [module_names[cid] for cid in cluster_ids]
    
    # Get muscle targets for each module
    muscle_targets = []
    for cid in cluster_ids:
        targets = cluster_info[cluster_info['cluster_id'] == cid]['primary_targets'].iloc[0]
        # Simplify targets
        target_list = targets.split(', ')[:3]  # Top 3
        muscle_targets.append('\n→ ' + ', '.join(target_list))
    
    # Add targets to labels
    module_labels_full = [f"{name}\n{targets}" for name, targets in zip(module_labels, muscle_targets)]
    
    # Extract connectivity
    conn_matrix = np.zeros((len(top_dns), len(cluster_ids)))
    
    for i, (_, dn) in enumerate(top_dns.iterrows()):
        for j, cid in enumerate(cluster_ids):
            col_name = f'cluster_{cid}'
            if col_name in dn.index:
                conn_matrix[i, j] = dn[col_name]
    
    # Plot heatmap
    im = ax_a.imshow(conn_matrix, aspect='auto', cmap='YlOrRd', 
                     interpolation='nearest', vmin=0)
    
    # Axes
    ax_a.set_xticks(range(len(cluster_ids)))
    ax_a.set_xticklabels(module_labels_full, rotation=0, ha='center', 
                         fontsize=10, fontweight='bold')
    
    ax_a.set_yticks(range(len(top_dns)))
    ax_a.set_yticklabels(dn_names, fontsize=9)
    
    ax_a.set_xlabel('Premotor IN Modules', fontsize=13, fontweight='bold', labelpad=10)
    ax_a.set_ylabel(f'Descending Neurons (Top {top_n_dns})', fontsize=13, fontweight='bold')
    ax_a.set_title('A. DN Specialization Matrix', 
                   fontsize=15, fontweight='bold', loc='left', pad=15)
    
    # Color module labels
    for tick, cid in zip(ax_a.get_xticklabels(), cluster_ids):
        tick.set_color(module_colors[cid])
    
    # Color DN labels by specialization
    for tick, (_, dn) in zip(ax_a.get_yticklabels(), top_dns.iterrows()):
        if dn['specialization'] == 'power_control':
            tick.set_color('#D32F2F')
        elif dn['specialization'] == 'steering_control':
            tick.set_color('#0277BD')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_a, fraction=0.03, pad=0.02)
    cbar.set_label('Synaptic Strength', fontsize=11, fontweight='bold', 
                  rotation=270, labelpad=20)
    
    # Grid lines
    for i in range(len(cluster_ids) + 1):
        ax_a.axvline(i - 0.5, color='white', linewidth=2)
    for i in range(len(top_dns) + 1):
        ax_a.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
    
    # Highlight power module
    power_idx = [i for i, cid in enumerate(cluster_ids) 
                 if cluster_info[cluster_info['cluster_id']==cid]['functional_type'].iloc[0] == 'power_control']
    if power_idx:
        for idx in power_idx:
            ax_a.axvline(idx - 0.5, color='red', linewidth=3, alpha=0.7)
            ax_a.axvline(idx + 0.5, color='red', linewidth=3, alpha=0.7)
    
    # ===== PANEL B: Pathway Schematic =====
    ax_b = fig.add_subplot(gs[0, 2])
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.axis('off')
    
    ax_b.set_title('B. Pathway Architecture', fontsize=15, fontweight='bold', 
                   loc='left', pad=15, x=0)
    
    # Layer positions
    y_dn = 8.5
    y_module = 5.5
    y_mn = 2.5
    
    # DNs
    ax_b.text(5, 9.5, 'Brain Commands', ha='center', va='center',
             fontsize=11, fontweight='bold', style='italic', color='#666666')
    
    power_dns = len(dn_class_df[dn_class_df['specialization'] == 'power_control'])
    steer_dns = len(dn_class_df[dn_class_df['specialization'] == 'steering_control'])
    
    # Power DN box
    dn_power = FancyBboxPatch((1, y_dn-0.35), 3.5, 0.7,
                             boxstyle="round,pad=0.08",
                             facecolor='#FF6B6B', edgecolor='black', linewidth=2)
    ax_b.add_patch(dn_power)
    ax_b.text(2.75, y_dn, f'Power DNs (n={power_dns})',
             ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Steering DN box
    dn_steer = FancyBboxPatch((5.5, y_dn-0.35), 3.5, 0.7,
                             boxstyle="round,pad=0.08",
                             facecolor='#4ECDC4', edgecolor='black', linewidth=2)
    ax_b.add_patch(dn_steer)
    ax_b.text(7.25, y_dn, f'Steering DNs (n={steer_dns})',
             ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Modules
    ax_b.text(5, 6.5, 'Premotor IN Modules', ha='center', va='center',
             fontsize=11, fontweight='bold', style='italic', color='#666666')
    
    # Power module
    power_cluster = cluster_info[cluster_info['functional_type'] == 'power_control'].iloc[0]
    power_cid = power_cluster['cluster_id']
    
    mod_power = FancyBboxPatch((1, y_module-0.4), 3.5, 0.8,
                              boxstyle="round,pad=0.08",
                              facecolor='#FF6B6B', edgecolor='black', 
                              linewidth=2.5, alpha=0.85)
    ax_b.add_patch(mod_power)
    ax_b.text(2.75, y_module+0.15, 'Power Control',
             ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax_b.text(2.75, y_module-0.15, f'({power_cluster["n_interneurons"]} INs)',
             ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # Steering modules (compact)
    steer_clusters = cluster_info[cluster_info['functional_type'] == 'steering_control']
    total_steer_ins = steer_clusters['n_interneurons'].sum()
    
    mod_steer = FancyBboxPatch((5.5, y_module-0.4), 3.5, 0.8,
                              boxstyle="round,pad=0.08",
                              facecolor='#4ECDC4', edgecolor='black',
                              linewidth=2.5, alpha=0.85)
    ax_b.add_patch(mod_steer)
    ax_b.text(7.25, y_module+0.15, 'Steering Control (A-D)',
             ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax_b.text(7.25, y_module-0.15, f'({total_steer_ins} INs in 4 modules)',
             ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # Motor output
    ax_b.text(5, 3.2, 'Motor Output', ha='center', va='center',
             fontsize=11, fontweight='bold', style='italic', color='#666666')
    
    power_pools = motor_pools[motor_pools['muscle_type'] == 'power']
    steer_pools = motor_pools[motor_pools['muscle_type'] == 'steering']
    
    # Power MNs
    mn_power = FancyBboxPatch((1, y_mn-0.3), 3.5, 0.6,
                             boxstyle="round,pad=0.08",
                             facecolor='#FFA07A', edgecolor='black', linewidth=2)
    ax_b.add_patch(mn_power)
    ax_b.text(2.75, y_mn, f'Power MNs\n({len(power_pools)} pools)',
             ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Steering MNs
    mn_steer = FancyBboxPatch((5.5, y_mn-0.3), 3.5, 0.6,
                             boxstyle="round,pad=0.08",
                             facecolor='#95E1D3', edgecolor='black', linewidth=2)
    ax_b.add_patch(mn_steer)
    ax_b.text(7.25, y_mn, f'Steering MNs\n({len(steer_pools)} pools)',
             ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    # DN → Module
    arr1 = FancyArrowPatch((2.75, y_dn-0.35), (2.75, y_module+0.4),
                          arrowstyle='->', mutation_scale=25, linewidth=3, color='#D32F2F')
    ax_b.add_patch(arr1)
    
    arr2 = FancyArrowPatch((7.25, y_dn-0.35), (7.25, y_module+0.4),
                          arrowstyle='->', mutation_scale=25, linewidth=3, color='#0277BD')
    ax_b.add_patch(arr2)
    
    # Module → MN
    arr3 = FancyArrowPatch((2.75, y_module-0.4), (2.75, y_mn+0.3),
                          arrowstyle='->', mutation_scale=25, linewidth=3, color='#D32F2F')
    ax_b.add_patch(arr3)
    
    arr4 = FancyArrowPatch((7.25, y_module-0.4), (7.25, y_mn+0.3),
                          arrowstyle='->', mutation_scale=25, linewidth=3, color='#0277BD')
    ax_b.add_patch(arr4)
    
    # Muscles (bottom)
    ax_b.text(2.75, 1.2, 'DLM, DVM', ha='center', va='center',
             fontsize=10, fontweight='bold', color='#D32F2F',
             bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8, pad=0.3))
    
    ax_b.text(7.25, 1.2, 'Basalar, Axillary,\nHaltere, etc.', ha='center', va='center',
             fontsize=9, fontweight='bold', color='#0277BD',
             bbox=dict(boxstyle='round', facecolor='#E1F5FE', alpha=0.8, pad=0.3))
    
    # MN → Muscle arrows
    arr5 = FancyArrowPatch((2.75, y_mn-0.3), (2.75, 1.6),
                          arrowstyle='->', mutation_scale=20, linewidth=2.5, color='black')
    ax_b.add_patch(arr5)
    
    arr6 = FancyArrowPatch((7.25, y_mn-0.3), (7.25, 1.6),
                          arrowstyle='->', mutation_scale=20, linewidth=2.5, color='black')
    ax_b.add_patch(arr6)
    
    # Add summary box
    summary_box = FancyBboxPatch((0.5, 0.1), 9, 0.7,
                                boxstyle="round,pad=0.08",
                                facecolor='#F5F5F5',
                                edgecolor='#666666',
                                linewidth=2)
    ax_b.add_patch(summary_box)
    
    ax_b.text(5, 0.6, 'Flight Control Organization:', ha='center', va='center',
             fontsize=9, fontweight='bold')
    ax_b.text(5, 0.3, f'1 Power Module (370 INs) + 4 Steering Modules (1,710 INs) → {len(motor_pools)} Motor Pools',
             ha='center', va='center', fontsize=8)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', label='Power System'),
        mpatches.Patch(facecolor='#4ECDC4', edgecolor='black', label='Steering System'),
        mpatches.Rectangle((0,0), 1, 1, facecolor='none', edgecolor='red', 
                          linewidth=3, label='Power Module Boundary')
    ]
    ax_a.legend(handles=legend_elements, loc='lower right', fontsize=10, 
               framealpha=0.95, edgecolor='black', fancybox=True)
    
    return fig


def main():
    print("\n" + "="*70)
    print("CREATING CLEAN DN CONNECTIVITY FIGURE")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    
    loader = ConnectomeDataLoader('config.yaml')
    
    dn_class = pd.read_csv('results/dn_pathways/dn_classifications.csv')
    print(f"  ✓ DN classifications: {len(dn_class)}")
    
    cluster_info = pd.read_csv('results/interneuron_clusters/cluster_characteristics.csv')
    print(f"  ✓ Cluster info: {len(cluster_info)}")
    
    motor_pools = pd.read_csv('results/motor_neurons/motor_pools.csv')
    motor_pools['motor_neuron_ids'] = motor_pools['motor_neuron_ids'].apply(eval)
    print(f"  ✓ Motor pools: {len(motor_pools)}\n")
    
    print("Creating figure...")
    
    fig = create_dn_connectivity_figure(
        dn_class,
        cluster_info,
        motor_pools,
        loader,
        top_n_dns=20
    )
    
    # Save
    output_dir = Path('results/final_figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # PNG
    output_png = output_dir / 'DN_to_Module_Connectivity.png'
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n  ✓ Saved PNG: {output_png}")
    
    # PDF
    output_pdf = output_dir / 'DN_to_Module_Connectivity.pdf'
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved PDF: {output_pdf}")
    
    plt.close()
    
    print("\n" + "="*70)
    print("CLEAN DN CONNECTIVITY FIGURE COMPLETE")
    print("="*70)
    print(f"\nSaved to: {output_dir}")
    print("\nFigure shows:")
    print("  • Panel A: DN → Module connectivity heatmap (top 20 DNs)")
    print("  • Panel B: Simplified pathway architecture schematic")
    print("\nModule names:")
    print("  • Power Control (1 module)")
    print("  • Steering A, B, C, D (4 modules)")
    print("\nClean, simple, and accurate! ✨")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()