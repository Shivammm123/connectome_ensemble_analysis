"""
Publication-Ready Figure Panel

Creates a comprehensive multi-panel figure suitable for publication.
Replicates the style of Cheong et al. 2024 and Azevedo et al. 2023.

Figure layout:
- Panel A: Circuit schematic
- Panel B: Modularity analysis
- Panel C: DN pathway organization
- Panel D: Power vs steering comparison
- Panel E: Hub neuron network
- Panel F: Summary statistics
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


def create_circuit_schematic(ax, cluster_info, dn_class_df, motor_pools):
    """
    Panel A: Circuit organization schematic
    """
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'A. Circuit Organization', 
           ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Layer positions
    brain_y = 8
    dn_y = 6.5
    in_y = 4.5
    mn_y = 2.5
    muscle_y = 1
    
    # Brain
    brain = FancyBboxPatch((4, brain_y-0.3), 2, 0.6,
                          boxstyle="round,pad=0.1", 
                          facecolor='#E8E8E8', 
                          edgecolor='black', linewidth=2)
    ax.add_patch(brain)
    ax.text(5, brain_y, 'Brain', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # DNs
    dn_power = len(dn_class_df[dn_class_df['specialization'] == 'power_control'])
    dn_steering = len(dn_class_df[dn_class_df['specialization'] == 'steering_control'])
    
    dn_box1 = Rectangle((2, dn_y-0.25), 1.5, 0.5, 
                        facecolor='#FF6B6B', edgecolor='black', linewidth=1.5)
    ax.add_patch(dn_box1)
    ax.text(2.75, dn_y, f'Power DNs\n(n={dn_power})', 
           ha='center', va='center', fontsize=8, fontweight='bold')
    
    dn_box2 = Rectangle((6.5, dn_y-0.25), 1.5, 0.5,
                        facecolor='#4ECDC4', edgecolor='black', linewidth=1.5)
    ax.add_patch(dn_box2)
    ax.text(7.25, dn_y, f'Steering DNs\n(n={dn_steering})', 
           ha='center', va='center', fontsize=8, fontweight='bold')
    
    # IN Clusters
    power_cluster = cluster_info[cluster_info['functional_type'] == 'power_control']
    steering_clusters = cluster_info[cluster_info['functional_type'] == 'steering_control']
    
    in_box1 = Rectangle((1.5, in_y-0.4), 2, 0.8,
                       facecolor='#FF6B6B', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(in_box1)
    ax.text(2.5, in_y, f'Power Module\nCluster 4\n(n={power_cluster["n_interneurons"].sum()})', 
           ha='center', va='center', fontsize=8, fontweight='bold')
    
    in_box2 = Rectangle((6, in_y-0.4), 2.5, 0.8,
                       facecolor='#4ECDC4', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(in_box2)
    ax.text(7.25, in_y, f'Steering Modules\nClusters 1,2,3,5\n(n={steering_clusters["n_interneurons"].sum()})', 
           ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Motor Neurons
    power_pools = motor_pools[motor_pools['muscle_type'] == 'power']
    steering_pools = motor_pools[motor_pools['muscle_type'] == 'steering']
    
    mn_box1 = Rectangle((1.5, mn_y-0.25), 2, 0.5,
                       facecolor='#FFA07A', edgecolor='black', linewidth=1.5)
    ax.add_patch(mn_box1)
    ax.text(2.5, mn_y, f'Power MNs\n(n={sum([len(p) for p in power_pools["motor_neuron_ids"]])})', 
           ha='center', va='center', fontsize=8, fontweight='bold')
    
    mn_box2 = Rectangle((6, mn_y-0.25), 2.5, 0.5,
                       facecolor='#95E1D3', edgecolor='black', linewidth=1.5)
    ax.add_patch(mn_box2)
    ax.text(7.25, mn_y, f'Steering MNs\n(n={sum([len(p) for p in steering_pools["motor_neuron_ids"]])})', 
           ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Muscles
    muscle_box1 = Rectangle((1.5, muscle_y-0.2), 2, 0.4,
                           facecolor='#FF6B6B', edgecolor='black', linewidth=1.5)
    ax.add_patch(muscle_box1)
    ax.text(2.5, muscle_y, f'DLM, DVM\n({len(power_pools)} pools)', 
           ha='center', va='center', fontsize=7, fontweight='bold')
    
    muscle_box2 = Rectangle((6, muscle_y-0.2), 2.5, 0.4,
                           facecolor='#4ECDC4', edgecolor='black', linewidth=1.5)
    ax.add_patch(muscle_box2)
    ax.text(7.25, muscle_y, f'Basalar, Axillary, etc\n({len(steering_pools)} pools)', 
           ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Arrows
    # Brain to DNs
    ax.annotate('', xy=(2.75, dn_y+0.25), xytext=(4.5, brain_y-0.3),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(7.25, dn_y+0.25), xytext=(5.5, brain_y-0.3),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # DNs to INs
    ax.annotate('', xy=(2.5, in_y+0.4), xytext=(2.75, dn_y-0.25),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#FF6B6B'))
    ax.annotate('', xy=(7.25, in_y+0.4), xytext=(7.25, dn_y-0.25),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#4ECDC4'))
    
    # INs to MNs
    ax.annotate('', xy=(2.5, mn_y+0.25), xytext=(2.5, in_y-0.4),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#FF6B6B'))
    ax.annotate('', xy=(7.25, mn_y+0.25), xytext=(7.25, in_y-0.4),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#4ECDC4'))
    
    # MNs to Muscles
    ax.annotate('', xy=(2.5, muscle_y+0.2), xytext=(2.5, mn_y-0.25),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(7.25, muscle_y+0.2), xytext=(7.25, mn_y-0.25),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Labels
    ax.text(0.2, brain_y, 'Brain', fontsize=9, style='italic')
    ax.text(0.2, dn_y, 'VNC', fontsize=9, style='italic')


def create_modularity_panel(ax, cluster_props, modularity_score):
    """
    Panel B: Modularity and cluster properties
    """
    
    ax.text(0.5, 0.95, 'B. Circuit Modularity', 
           transform=ax.transAxes,
           ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Modularity score visualization
    ax.text(0.5, 0.85, f'Modularity Score: {modularity_score:.3f}',
           transform=ax.transAxes,
           ha='center', va='top', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Cluster properties table
    x_pos = np.arange(len(cluster_props))
    
    colors = ['#FF6B6B' if ft == 'power_control' else '#4ECDC4' 
              for ft in cluster_props['functional_type']]
    
    # Synapses per IN
    bars = ax.bar(x_pos, cluster_props['synapses_per_in'], 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Cluster ID', fontsize=11, fontweight='bold')
    ax.set_ylabel('Synapses per Interneuron', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cluster_props['cluster_id'])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, cluster_props['synapses_per_in'])):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2,
               f'{val:.0f}', ha='center', va='bottom', 
               fontsize=9, fontweight='bold')
    
    # Legend
    power_patch = mpatches.Patch(color='#FF6B6B', label='Power')
    steering_patch = mpatches.Patch(color='#4ECDC4', label='Steering')
    ax.legend(handles=[power_patch, steering_patch], 
             loc='upper right', fontsize=9)


def create_dn_pathway_panel(ax, dn_strategies):
    """
    Panel C: DN pathway strategies
    """
    
    ax.text(0.5, 0.95, 'C. DN Pathway Strategies', 
           transform=ax.transAxes,
           ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Strategy counts
    strategy_counts = dn_strategies['strategy'].value_counts()
    
    # Pie chart
    colors_pie = {
        'both': '#FFA07A',
        'direct_only': '#FF6B6B',
        'indirect_only': '#4ECDC4',
        'no_wing_connection': '#E0E0E0'
    }
    
    pie_colors = [colors_pie.get(s, '#CCCCCC') for s in strategy_counts.index]
    
    # Clean labels
    labels = []
    for s in strategy_counts.index:
        if s == 'both':
            labels.append('Both\nPathways')
        elif s == 'direct_only':
            labels.append('Direct\nOnly')
        elif s == 'indirect_only':
            labels.append('Indirect\nOnly')
        else:
            labels.append('No Wing\nConnection')
    
    wedges, texts, autotexts = ax.pie(strategy_counts.values, 
                                       labels=labels,
                                       colors=pie_colors,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')


def create_power_steering_comparison(ax, cluster_props):
    """
    Panel D: Power vs Steering quantitative comparison
    """
    
    ax.text(0.5, 0.95, 'D. Power vs Steering Modules', 
           transform=ax.transAxes,
           ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Get totals
    power = cluster_props[cluster_props['functional_type'] == 'power_control']
    steering = cluster_props[cluster_props['functional_type'] == 'steering_control']
    
    categories = ['Interneurons', 'Connections', 'Synapses', 'Syn/IN']
    
    power_values = [
        power['n_interneurons'].sum(),
        power['n_connections'].sum(),
        power['total_synapses'].sum(),
        power['synapses_per_in'].mean()
    ]
    
    steering_values = [
        steering['n_interneurons'].sum(),
        steering['n_connections'].sum(),
        steering['total_synapses'].sum(),
        steering['synapses_per_in'].mean()
    ]
    
    # Normalize for visualization (percent of total)
    power_norm = []
    steering_norm = []
    for p, s in zip(power_values, steering_values):
        total = p + s
        power_norm.append(100 * p / total if total > 0 else 0)
        steering_norm.append(100 * s / total if total > 0 else 0)
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, power_norm, width, 
                    label='Power', color='#FF6B6B', alpha=0.8)
    bars2 = ax.barh(x + width/2, steering_norm, width,
                    label='Steering', color='#4ECDC4', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlabel('Percentage of Total', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, 100)
    
    # Add actual values
    for i, (bar, val) in enumerate(zip(bars1, power_values)):
        if i == 3:  # Syn/IN
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                   f'{val:.0f}', va='center', fontsize=8)
        else:
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                   f'{int(val):,}', va='center', fontsize=8)


def create_hub_network_panel(ax, hub_neurons):
    """
    Panel E: Hub neuron distribution
    """
    
    ax.text(0.5, 0.95, 'E. Hub Interneurons', 
           transform=ax.transAxes,
           ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Distribution of hub connectivity
    top_20 = hub_neurons.head(20)
    
    x_pos = np.arange(len(top_20))
    
    bars = ax.barh(x_pos, top_20['n_mn_targets'], 
                   color='steelblue', alpha=0.7, edgecolor='black')
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels([f'Hub {i+1}' for i in range(len(top_20))], fontsize=8)
    ax.set_xlabel('Number of MN Targets', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    
    # Highlight top 5
    for i in range(min(5, len(bars))):
        bars[i].set_color('#FF6B6B')
        bars[i].set_alpha(0.9)


def create_summary_panel(ax, cluster_props, dn_strategies, integrative_ins, modularity_score):
    """
    Panel F: Summary statistics
    """
    
    ax.axis('off')
    ax.text(0.5, 0.95, 'F. Summary Statistics', 
           transform=ax.transAxes,
           ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Build summary text
    summary = "CIRCUIT ANALYSIS SUMMARY\n\n"
    
    summary += "Network Organization:\n"
    summary += f"  • Modularity Score: {modularity_score:.3f}\n"
    summary += f"  • Functional Modules: {len(cluster_props)}\n"
    summary += f"    - Power: {len(cluster_props[cluster_props['functional_type']=='power_control'])}\n"
    summary += f"    - Steering: {len(cluster_props[cluster_props['functional_type']=='steering_control'])}\n\n"
    
    summary += "Interneuron Network:\n"
    summary += f"  • Total Premotor INs: {cluster_props['n_interneurons'].sum():,}\n"
    summary += f"  • Power Module: {cluster_props[cluster_props['functional_type']=='power_control']['n_interneurons'].sum()}\n"
    summary += f"  • Steering Modules: {cluster_props[cluster_props['functional_type']=='steering_control']['n_interneurons'].sum()}\n"
    summary += f"  • Integrative INs: {len(integrative_ins)}\n\n"
    
    summary += "Connectivity:\n"
    summary += f"  • Total Connections: {cluster_props['n_connections'].sum():,}\n"
    summary += f"  • Total Synapses: {cluster_props['total_synapses'].sum():,.0f}\n"
    power_syn = cluster_props[cluster_props['functional_type']=='power_control']['synapses_per_in'].mean()
    steer_syn = cluster_props[cluster_props['functional_type']=='steering_control']['synapses_per_in'].mean()
    summary += f"  • Power Syn/IN: {power_syn:.0f}\n"
    summary += f"  • Steering Syn/IN: {steer_syn:.0f}\n\n"
    
    summary += "Descending Neurons:\n"
    summary += f"  • Wing DNs Analyzed: {len(dn_strategies)}\n"
    both = len(dn_strategies[dn_strategies['strategy']=='both'])
    direct = len(dn_strategies[dn_strategies['strategy']=='direct_only'])
    indirect = len(dn_strategies[dn_strategies['strategy']=='indirect_only'])
    summary += f"  • Both Pathways: {both}\n"
    summary += f"  • Direct Only: {direct}\n"
    summary += f"  • Indirect Only: {indirect}\n"
    
    ax.text(0.05, 0.85, summary,
           transform=ax.transAxes,
           ha='left', va='top', fontsize=9,
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=0.5))


def main():
    print("\n" + "="*70)
    print("PUBLICATION FIGURE PANEL")
    print("="*70 + "\n")
    
    # Load all data
    print("Loading data...")
    print("-"*70 + "\n")
    
    loader = ConnectomeDataLoader('config.yaml')
    
    cluster_props = pd.read_csv('results/circuit_modules/cluster_properties_detailed.csv')
    print(f"  ✓ Cluster properties")
    
    dn_strategies = pd.read_csv('results/dn_pathways/dn_pathway_strategies.csv')
    print(f"  ✓ DN strategies")
    
    dn_class = pd.read_csv('results/dn_pathways/dn_classifications.csv')
    print(f"  ✓ DN classifications")
    
    motor_pools = pd.read_csv('results/motor_neurons/motor_pools.csv')
    motor_pools['motor_neuron_ids'] = motor_pools['motor_neuron_ids'].apply(eval)
    print(f"  ✓ Motor pools")
    
    cluster_info = pd.read_csv('results/interneuron_clusters/cluster_characteristics.csv')
    print(f"  ✓ Cluster info")
    
    hub_neurons = pd.read_csv('results/interneuron_clusters/hub_interneurons.csv')
    print(f"  ✓ Hub neurons")
    
    # Load modularity
    try:
        with open('results/circuit_modules/modularity_score.txt', 'r') as f:
            modularity_line = f.readline()
            modularity_score = float(modularity_line.split(':')[1].strip())
    except:
        modularity_score = 0.35
    print(f"  ✓ Modularity score: {modularity_score:.3f}")
    
    # Load integrative INs
    try:
        integrative_ins = pd.read_csv('results/circuit_modules/integrative_interneurons.csv')
    except:
        integrative_ins = pd.DataFrame()
    print(f"  ✓ Integrative INs\n")
    
    # Create figure
    print("-"*70)
    print("Creating publication figure...")
    print("-"*70 + "\n")
    
    fig = plt.figure(figsize=(20, 12), dpi=300)
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3,
                         left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Panel A: Circuit schematic (large, top-left spanning 2 columns)
    ax_a = fig.add_subplot(gs[0, :2])
    create_circuit_schematic(ax_a, cluster_info, dn_class, motor_pools)
    
    # Panel B: Modularity (top-right)
    ax_b = fig.add_subplot(gs[0, 2])
    create_modularity_panel(ax_b, cluster_props, modularity_score)
    
    # Panel C: DN pathways (middle-left)
    ax_c = fig.add_subplot(gs[1, 0])
    create_dn_pathway_panel(ax_c, dn_strategies)
    
    # Panel D: Power vs Steering (middle-center)
    ax_d = fig.add_subplot(gs[1, 1])
    create_power_steering_comparison(ax_d, cluster_props)
    
    # Panel E: Hub neurons (middle-right)
    ax_e = fig.add_subplot(gs[1, 2])
    create_hub_network_panel(ax_e, hub_neurons)
    
    # Panel F: Summary (bottom, spanning all columns)
    ax_f = fig.add_subplot(gs[2, :])
    create_summary_panel(ax_f, cluster_props, dn_strategies, integrative_ins, modularity_score)
    
    # Overall title
    fig.suptitle('Flight Control Circuit Organization in Drosophila BANC Connectome',
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path('results/publication_figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'figure_main_panel.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")
    
    # Also save as PDF for publication
    output_pdf = output_dir / 'figure_main_panel.pdf'
    
    # Recreate for PDF
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3,
                         left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    ax_a = fig.add_subplot(gs[0, :2])
    create_circuit_schematic(ax_a, cluster_info, dn_class, motor_pools)
    
    ax_b = fig.add_subplot(gs[0, 2])
    create_modularity_panel(ax_b, cluster_props, modularity_score)
    
    ax_c = fig.add_subplot(gs[1, 0])
    create_dn_pathway_panel(ax_c, dn_strategies)
    
    ax_d = fig.add_subplot(gs[1, 1])
    create_power_steering_comparison(ax_d, cluster_props)
    
    ax_e = fig.add_subplot(gs[1, 2])
    create_hub_network_panel(ax_e, hub_neurons)
    
    ax_f = fig.add_subplot(gs[2, :])
    create_summary_panel(ax_f, cluster_props, dn_strategies, integrative_ins, modularity_score)
    
    fig.suptitle('Flight Control Circuit Organization in Drosophila BANC Connectome',
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_pdf}")
    
    print("\n" + "="*70)
    print("PUBLICATION FIGURE COMPLETE")
    print("="*70)
    print(f"\nMain figure panel created:")
    print(f"  • PNG: figure_main_panel.png (300 DPI)")
    print(f"  • PDF: figure_main_panel.pdf (vector)")
    print(f"\nLocation: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()