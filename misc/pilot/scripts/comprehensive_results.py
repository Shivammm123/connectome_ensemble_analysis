"""
Results Figure - Comprehensive Findings

Single comprehensive figure presenting all major results.
Clean, presentation-worthy visualization of discoveries.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge
import seaborn as sns

sys.path.append(str(Path(__file__).parent))


def load_results_data():
    """Load all necessary results."""
    
    data = {}
    
    try:
        data['cluster_info'] = pd.read_csv('results/interneuron_clusters/cluster_characteristics.csv')
        data['cluster_props'] = pd.read_csv('results/circuit_modules/cluster_properties_detailed.csv')
        data['motor_pools'] = pd.read_csv('results/motor_neurons/motor_pools.csv')
        data['dn_strategies'] = pd.read_csv('results/dn_pathways/dn_pathway_strategies.csv')
        data['hub_neurons'] = pd.read_csv('results/interneuron_clusters/hub_interneurons.csv')
        
        with open('results/circuit_modules/modularity_score.txt', 'r') as f:
            line = f.readline()
            data['modularity'] = float(line.split(':')[1].strip())
        
        try:
            data['integrative_ins'] = pd.read_csv('results/circuit_modules/integrative_interneurons.csv')
        except:
            data['integrative_ins'] = pd.DataFrame()
        
        print("✓ All results loaded")
        return data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_results_figure(data):
    """
    Create comprehensive results figure with 6 panels.
    """
    
    fig = plt.figure(figsize=(24, 14), dpi=300)
    
    # Create main title
    fig.suptitle('Flight Control Circuit Organization in Drosophila BANC Connectome\nMajor Findings',
                fontsize=22, fontweight='bold', y=0.98)
    
    # Define grid - 2 rows x 3 columns
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    # Panel A: Circuit Architecture (top-left, large)
    ax_a = fig.add_subplot(gs[0, 0])
    create_circuit_architecture_panel(ax_a, data)
    
    # Panel B: Modularity (top-center)
    ax_b = fig.add_subplot(gs[0, 1])
    create_modularity_panel(ax_b, data)
    
    # Panel C: Power vs Steering (top-right)
    ax_c = fig.add_subplot(gs[0, 2])
    create_power_steering_panel(ax_c, data)
    
    # Panel D: Hub Neurons (bottom-left)
    ax_d = fig.add_subplot(gs[1, 0])
    create_hub_panel(ax_d, data)
    
    # Panel E: DN Pathways (bottom-center)
    ax_e = fig.add_subplot(gs[1, 1])
    create_dn_pathways_panel(ax_e, data)
    
    # Panel F: Summary Statistics (bottom-right)
    ax_f = fig.add_subplot(gs[1, 2])
    create_summary_statistics_panel(ax_f, data)
    
    return fig


def create_circuit_architecture_panel(ax, data):
    """
    Panel A: Overall circuit architecture discovered.
    """
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.8, 'A. Circuit Architecture', 
           ha='center', va='top', fontsize=16, fontweight='bold')
    
    cluster_info = data['cluster_info']
    
    # Main finding box
    finding_box = FancyBboxPatch((0.5, 7), 9, 2.3,
                                boxstyle="round,pad=0.2",
                                facecolor='#E8F5E9',
                                edgecolor='#2E7D32',
                                linewidth=3)
    ax.add_patch(finding_box)
    
    ax.text(5, 9, '5 Functional Modules Discovered',
           ha='center', va='top', fontsize=14, fontweight='bold', color='#2E7D32')
    ax.text(5, 8.5, '2,080 Premotor Interneurons Organized into Power and Steering Networks',
           ha='center', va='top', fontsize=11)
    
    # Power module
    power_box = Rectangle((1, 5.5), 3.5, 1.2,
                         facecolor='#FF6B6B', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(power_box)
    
    power_cluster = cluster_info[cluster_info['functional_type'] == 'power_control']
    ax.text(2.75, 6.5, 'POWER MODULE',
           ha='center', va='top', fontsize=12, fontweight='bold', color='white')
    ax.text(2.75, 6.1, f'Cluster 4',
           ha='center', va='top', fontsize=10, color='white')
    ax.text(2.75, 5.8, f'{power_cluster["n_interneurons"].sum()} interneurons',
           ha='center', va='top', fontsize=9, color='white', fontweight='bold')
    
    # Steering modules
    steering_box = Rectangle((5.5, 5.5), 3.5, 1.2,
                            facecolor='#4ECDC4', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(steering_box)
    
    steering_clusters = cluster_info[cluster_info['functional_type'] == 'steering_control']
    ax.text(7.25, 6.5, 'STEERING MODULES',
           ha='center', va='top', fontsize=12, fontweight='bold', color='white')
    ax.text(7.25, 6.1, f'Clusters 1, 2, 3, 5',
           ha='center', va='top', fontsize=10, color='white')
    ax.text(7.25, 5.8, f'{steering_clusters["n_interneurons"].sum()} interneurons',
           ha='center', va='top', fontsize=9, color='white', fontweight='bold')
    
    # Cluster breakdown
    y_pos = 4.8
    ax.text(5, y_pos, 'Individual Clusters:', ha='center', va='top', 
           fontsize=10, fontweight='bold')
    
    for i, (_, cluster) in enumerate(cluster_info.iterrows()):
        y = y_pos - 0.5 - (i * 0.5)
        
        if cluster['functional_type'] == 'power_control':
            color = '#FF6B6B'
        else:
            color = '#4ECDC4'
        
        # Cluster circle
        circle = Circle((1.5, y), 0.15, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        
        ax.text(1.8, y, f"C{cluster['cluster_id']}: {cluster['n_interneurons']} INs → {cluster['primary_targets']}",
               ha='left', va='center', fontsize=8)
    
    # Key insight
    ax.text(5, 0.5, '⭐ Power: Strong connections | Steering: Distributed control',
           ha='center', va='top', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))


def create_modularity_panel(ax, data):
    """
    Panel B: Modularity score and cluster separation.
    """
    
    ax.text(0.5, 0.98, 'B. Network Modularity', 
           transform=ax.transAxes,
           ha='center', va='top', fontsize=16, fontweight='bold')
    
    modularity = data['modularity']
    cluster_props = data['cluster_props']
    
    # Modularity gauge
    from matplotlib.patches import Wedge
    
    # Draw semi-circle gauge
    theta = 180 * modularity  # 0 to 1 maps to 0 to 180 degrees
    
    # Background arc
    bg_arc = Wedge((0.5, 0.3), 0.25, 0, 180, width=0.05,
                   facecolor='#E0E0E0', edgecolor='none',
                   transform=ax.transAxes)
    ax.add_patch(bg_arc)
    
    # Score arc
    score_arc = Wedge((0.5, 0.3), 0.25, 0, theta, width=0.05,
                     facecolor='#4CAF50', edgecolor='none',
                     transform=ax.transAxes)
    ax.add_patch(score_arc)
    
    # Score text
    ax.text(0.5, 0.35, f'{modularity:.3f}',
           transform=ax.transAxes,
           ha='center', va='center', fontsize=32, fontweight='bold',
           color='#2E7D32')
    
    ax.text(0.5, 0.2, 'Modularity Score',
           transform=ax.transAxes,
           ha='center', va='top', fontsize=11, fontweight='bold')
    
    # Interpretation
    ax.text(0.5, 0.12, 'MODERATE Functional Separation',
           transform=ax.transAxes,
           ha='center', va='top', fontsize=10, color='#2E7D32', fontweight='bold')
    
    # Cluster connectivity comparison
    ax.text(0.5, 0.02, 'Cluster Connectivity Strength:',
           transform=ax.transAxes,
           ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Remove default axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def create_power_steering_panel(ax, data):
    """
    Panel C: Power vs Steering quantitative comparison.
    """
    
    ax.text(0.5, 0.98, 'C. Power vs Steering Networks', 
           transform=ax.transAxes,
           ha='center', va='top', fontsize=16, fontweight='bold')
    
    cluster_props = data['cluster_props']
    
    power = cluster_props[cluster_props['functional_type'] == 'power_control']
    steering = cluster_props[cluster_props['functional_type'] == 'steering_control']
    
    # Data
    categories = ['Interneurons', 'Total\nSynapses', 'Synapses\nper IN']
    
    power_vals = [
        power['n_interneurons'].sum(),
        power['total_synapses'].sum(),
        power['synapses_per_in'].mean()
    ]
    
    steering_vals = [
        steering['n_interneurons'].sum(),
        steering['total_synapses'].sum(),
        steering['synapses_per_in'].mean()
    ]
    
    # Normalize for side-by-side comparison
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, power_vals, width, 
                   label='Power', color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, steering_vals, width,
                   label='Steering', color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 1000:
                label = f'{int(height):,}'
            else:
                label = f'{height:.0f}'
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Key finding
    ratio = steering_vals[1] / power_vals[1]
    ax.text(0.5, -0.15, f'⚡ Steering has {ratio:.1f}× more total synapses',
           transform=ax.transAxes,
           ha='center', va='top', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))


def create_hub_panel(ax, data):
    """
    Panel D: Hub neurons distribution.
    """
    
    ax.text(0.5, 0.98, 'D. Hub Interneurons', 
           transform=ax.transAxes,
           ha='center', va='top', fontsize=16, fontweight='bold')
    
    hub_neurons = data['hub_neurons']
    
    # Top 15 hubs
    top_hubs = hub_neurons.head(15)
    
    y_pos = np.arange(len(top_hubs))
    
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_hubs)))
    
    bars = ax.barh(y_pos, top_hubs['n_mn_targets'], 
                   color=colors, edgecolor='black', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'Hub {i+1}' for i in range(len(top_hubs))], fontsize=9)
    ax.set_xlabel('Number of MN Targets', fontsize=11, fontweight='bold')
    ax.set_title(f'Top 15 of {len(hub_neurons):,} Hub Neurons', 
                fontsize=10, pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    
    # Highlight top 3
    for i in range(min(3, len(bars))):
        bars[i].set_edgecolor('#D32F2F')
        bars[i].set_linewidth(2.5)
    
    # Add values
    for i, (bar, val) in enumerate(zip(bars, top_hubs['n_mn_targets'])):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
               f'{int(val)}', va='center', fontsize=8, fontweight='bold')
    
    # Key insight
    max_targets = top_hubs['n_mn_targets'].iloc[0]
    ax.text(0.98, 0.02, f'⭐ Top hub: {int(max_targets)} MN targets\n({int(100*max_targets/50)}% of wing MNs)',
           transform=ax.transAxes,
           ha='right', va='bottom', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8))


def create_dn_pathways_panel(ax, data):
    """
    Panel E: DN pathway strategies.
    """
    
    ax.text(0.5, 0.98, 'E. DN Pathway Strategies', 
           transform=ax.transAxes,
           ha='center', va='top', fontsize=16, fontweight='bold')
    
    dn_strategies = data['dn_strategies']
    
    # Count strategies
    strategy_counts = dn_strategies['strategy'].value_counts()
    
    # Clean labels
    labels_map = {
        'both': 'Both\nPathways',
        'direct_only': 'Direct\nOnly',
        'indirect_only': 'Indirect\nOnly',
        'no_wing_connection': 'No Wing\nConnection'
    }
    
    labels = [labels_map.get(s, s) for s in strategy_counts.index]
    sizes = strategy_counts.values
    
    colors = ['#FFA07A', '#FF6B6B', '#4ECDC4', '#E0E0E0']
    
    # Pie chart
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 10, 'fontweight': 'bold'},
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    # Legend with counts
    legend_labels = [f'{label.replace(chr(10), " ")}: {count}' 
                    for label, count in zip(labels, sizes)]
    ax.legend(legend_labels, loc='lower center', 
             bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9, framealpha=0.9)


def create_summary_statistics_panel(ax, data):
    """
    Panel F: Summary statistics table.
    """
    
    ax.axis('off')
    ax.text(0.5, 0.98, 'F. Key Statistics', 
           transform=ax.transAxes,
           ha='center', va='top', fontsize=16, fontweight='bold')
    
    cluster_props = data['cluster_props']
    hub_neurons = data['hub_neurons']
    dn_strategies = data['dn_strategies']
    integrative_ins = data['integrative_ins']
    
    # Build summary text
    summary = "CIRCUIT SUMMARY\n\n"
    
    summary += "Premotor Network:\n"
    summary += f"  • Total INs: {cluster_props['n_interneurons'].sum():,}\n"
    summary += f"  • Connections: {cluster_props['n_connections'].sum():,}\n"
    summary += f"  • Synapses: {int(cluster_props['total_synapses'].sum()):,}\n\n"
    
    summary += "Functional Modules:\n"
    summary += f"  • Total: {len(data['cluster_info'])}\n"
    summary += f"  • Power: 1 cluster (370 INs)\n"
    summary += f"  • Steering: 4 clusters (1,710 INs)\n\n"
    
    summary += "Hub Neurons:\n"
    median_deg = hub_neurons['degree_centrality'].median()
    high_cent = len(hub_neurons[hub_neurons['degree_centrality'] > median_deg])
    summary += f"  • Total identified: {len(hub_neurons):,}\n"
    summary += f"  • High centrality: {high_cent}\n"
    summary += f"  • Max targets: {int(hub_neurons['n_mn_targets'].max())}\n\n"
    
    if len(integrative_ins) > 0:
        pct = 100 * len(integrative_ins) / cluster_props['n_interneurons'].sum()
        summary += "Integration:\n"
        summary += f"  • Integrative INs: {len(integrative_ins)}\n"
        summary += f"  • Percentage: {pct:.1f}%\n\n"
    
    summary += "Descending Neurons:\n"
    summary += f"  • Analyzed: {len(dn_strategies)}\n"
    summary += f"  • Both pathways: {len(dn_strategies[dn_strategies['strategy']=='both'])}\n"
    summary += f"  • Direct only: {len(dn_strategies[dn_strategies['strategy']=='direct_only'])}\n"
    summary += f"  • Indirect only: {len(dn_strategies[dn_strategies['strategy']=='indirect_only'])}\n"
    
    ax.text(0.05, 0.85, summary,
           transform=ax.transAxes,
           ha='left', va='top', fontsize=10,
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', 
                    edgecolor='#666666', linewidth=2, alpha=0.9))


def main():
    print("\n" + "="*70)
    print("CREATING RESULTS FIGURE")
    print("="*70 + "\n")
    
    print("Loading results data...")
    data = load_results_data()
    
    if data is None:
        print("❌ Could not load results data!")
        print("Make sure all analysis scripts have been run.")
        return
    
    print("\nGenerating comprehensive results figure...")
    
    fig = create_results_figure(data)
    
    # Save
    output_dir = Path('results/presentation_figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # PNG
    output_png = output_dir / 'Figure_2_Results_Summary.png'
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n  ✓ Saved PNG: {output_png}")
    
    # PDF
    output_pdf = output_dir / 'Figure_2_Results_Summary.pdf'
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved PDF: {output_pdf}")
    
    plt.close()
    
    print("\n" + "="*70)
    print("RESULTS FIGURE COMPLETE")
    print("="*70)
    print(f"\nFigure 2 saved to: {output_dir}")
    print("  • Figure_2_Results_Summary.png (300 DPI)")
    print("  • Figure_2_Results_Summary.pdf (vector)")
    print("\nBoth presentation figures ready! 📊✨")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()