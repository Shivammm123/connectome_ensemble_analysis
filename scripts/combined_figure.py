"""
Combined DN Analysis Comparison Figure

Shows both analyses side-by-side:
- Panel A: DN → All Premotor INs (Network Breadth)
- Panel B: DN → Priority Hub INs (Experimental Focus)
- Panel C: Overlap analysis (DNs in both lists)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches
from matplotlib_venn import venn2

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader


def create_combined_comparison_figure():
    """
    Create comprehensive comparison figure.
    """
    
    print("\n" + "="*70)
    print("CREATING DN COMPARISON FIGURE")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    
    # Overall DN analysis
    dn_class = pd.read_csv('results/dn_pathways/dn_classifications.csv')
    print(f"  ✓ Overall DN classifications: {len(dn_class)}")
    
    # Priority IN analysis
    dn_priority_summary = pd.read_csv('results/dn_priority_analysis/dn_summary_priority_control.csv')
    print(f"  ✓ DN → Priority IN summary: {len(dn_priority_summary)}")
    
    dn_to_priority_conn = pd.read_csv('results/dn_priority_analysis/dn_to_priority_in_connections.csv')
    print(f"  ✓ DN → Priority connections: {len(dn_to_priority_conn):,}\n")
    
    # Create figure
    fig = plt.figure(figsize=(24, 14), dpi=300)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.05, right=0.95, top=0.93, bottom=0.05)
    
    # Overall title
    fig.suptitle('Descending Neuron Analysis: Network Breadth vs Experimental Focus\nTwo Complementary Views of DN Function',
                fontsize=20, fontweight='bold', y=0.97)
    
    # ===== ROW 1: TOP DNs COMPARISON =====
    
    # Panel A: Top DNs by TOTAL connectivity (all INs)
    ax_a = fig.add_subplot(gs[0, 0])
    
    top_overall = dn_class.nlargest(15, 'total_connectivity')
    
    colors_overall = ['#FF6B6B' if spec == 'power_control' else '#4ECDC4' if spec == 'steering_control' else '#CCCCCC'
                     for spec in top_overall['specialization']]
    
    y_pos = np.arange(len(top_overall))
    bars = ax_a.barh(y_pos, top_overall['total_connectivity'], 
                     color=colors_overall, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([str(name)[:30] for name in top_overall['dn_name']], fontsize=10)
    ax_a.set_xlabel('Total Synapses to All Premotor INs', fontsize=12, fontweight='bold')
    ax_a.set_title('A. Network Breadth\nDNs by Total IN Connectivity (n=2,080 INs)', 
                   fontsize=13, fontweight='bold', loc='left')
    ax_a.grid(axis='x', alpha=0.3)
    ax_a.invert_yaxis()
    
    # Add values
    for i, (bar, val) in enumerate(zip(bars, top_overall['total_connectivity'])):
        ax_a.text(val + 20, bar.get_y() + bar.get_height()/2,
                 f'{int(val):,}', va='center', fontsize=9, fontweight='bold')
    
    # Add annotation
    ax_a.text(0.02, 0.98, 'QUANTITY:\nBroad network influence',
             transform=ax_a.transAxes, ha='left', va='top',
             fontsize=10, fontweight='bold', style='italic',
             bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8, pad=0.5))
    
    # Panel B: Top DNs by PRIORITY connectivity
    ax_b = fig.add_subplot(gs[0, 1])
    
    top_priority = dn_priority_summary.head(15)
    
    colors_priority = ['#FF6B6B' if spec == 'power_control' else '#4ECDC4' if spec == 'steering_control' else '#CCCCCC'
                      for spec in top_priority['dn_specialization']]
    
    y_pos = np.arange(len(top_priority))
    bars = ax_b.barh(y_pos, top_priority['total_synapses'],
                     color=colors_priority, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([str(name)[:30] for name in top_priority['dn_name']], fontsize=10)
    ax_b.set_xlabel('Total Synapses to Priority Hub INs', fontsize=12, fontweight='bold')
    ax_b.set_title('B. Experimental Focus\nDNs by Priority IN Control (Top 50 Hubs)', 
                   fontsize=13, fontweight='bold', loc='left')
    ax_b.grid(axis='x', alpha=0.3)
    ax_b.invert_yaxis()
    
    # Add values
    for i, (bar, val) in enumerate(zip(bars, top_priority['total_synapses'])):
        ax_b.text(val + 5, bar.get_y() + bar.get_height()/2,
                 f'{int(val)}', va='center', fontsize=9, fontweight='bold')
    
    # Add annotation
    ax_b.text(0.02, 0.98, 'QUALITY:\nHigh-value targets',
             transform=ax_b.transAxes, ha='left', va='top',
             fontsize=10, fontweight='bold', style='italic',
             bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.8, pad=0.5))
    
    # Panel C: Overlap analysis
    ax_c = fig.add_subplot(gs[0, 2])
    
    # Get top 15 from each
    top15_overall = set(top_overall['dn_name'].tolist())
    top15_priority = set(top_priority['dn_name'].tolist())
    
    # Venn diagram
    try:
        from matplotlib_venn import venn2
        venn = venn2([top15_overall, top15_priority], 
                     set_labels=('Network\nBreadth', 'Experimental\nFocus'),
                     ax=ax_c)
        
        # Color the circles
        venn.get_patch_by_id('10').set_color('#4ECDC4')
        venn.get_patch_by_id('10').set_alpha(0.5)
        venn.get_patch_by_id('01').set_color('#FFA07A')
        venn.get_patch_by_id('01').set_alpha(0.5)
        venn.get_patch_by_id('11').set_color('#FFD700')
        venn.get_patch_by_id('11').set_alpha(0.7)
        
        for text in venn.set_labels:
            text.set_fontsize(11)
            text.set_fontweight('bold')
        
        for text in venn.subset_labels:
            text.set_fontsize(14)
            text.set_fontweight('bold')
        
        ax_c.set_title('C. Overlap Analysis\nTop 15 DNs from Each View',
                      fontsize=13, fontweight='bold', loc='left')
        
    except:
        # Fallback if matplotlib_venn not available
        overlap = top15_overall & top15_priority
        only_overall = top15_overall - top15_priority
        only_priority = top15_priority - top15_overall
        
        ax_c.text(0.5, 0.7, f'Overlap: {len(overlap)} DNs',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 transform=ax_c.transAxes)
        ax_c.text(0.5, 0.5, f'Only Breadth: {len(only_overall)} DNs',
                 ha='center', va='center', fontsize=12,
                 transform=ax_c.transAxes)
        ax_c.text(0.5, 0.3, f'Only Priority: {len(only_priority)} DNs',
                 ha='center', va='center', fontsize=12,
                 transform=ax_c.transAxes)
        ax_c.set_xlim(0, 1)
        ax_c.set_ylim(0, 1)
        ax_c.axis('off')
        ax_c.set_title('C. Overlap Analysis\nTop 15 DNs from Each View',
                      fontsize=13, fontweight='bold', loc='left')
    
    # ===== ROW 2: SPECIALIZATION & TARGETING =====
    
    # Panel D: Overall specialization
    ax_d = fig.add_subplot(gs[1, 0])
    
    spec_overall = dn_class['specialization'].value_counts()
    
    colors_spec = ['#4ECDC4' if 'steering' in s else '#FF6B6B' if 'power' in s else '#CCCCCC'
                   for s in spec_overall.index]
    
    bars = ax_d.bar(range(len(spec_overall)), spec_overall.values,
                    color=colors_spec, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax_d.set_xticks(range(len(spec_overall)))
    ax_d.set_xticklabels([s.replace('_', '\n') for s in spec_overall.index], 
                         fontsize=10, fontweight='bold')
    ax_d.set_ylabel('Number of DNs', fontsize=12, fontweight='bold')
    ax_d.set_title('D. Overall DN Specialization\nAll DNs Classified', 
                   fontsize=13, fontweight='bold', loc='left')
    ax_d.grid(axis='y', alpha=0.3)
    
    # Add values
    for bar, val in zip(bars, spec_overall.values):
        ax_d.text(bar.get_x() + bar.get_width()/2, val + 10,
                 str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Panel E: Priority specialization
    ax_e = fig.add_subplot(gs[1, 1])
    
    spec_priority = dn_to_priority_conn.groupby('dn_specialization').agg({
        'source': 'nunique',
        'weight': 'sum'
    }).rename(columns={'source': 'n_dns', 'weight': 'total_synapses'})
    
    x = np.arange(len(spec_priority))
    width = 0.35
    
    colors_bar1 = ['#4ECDC4' if 'steering' in s else '#FF6B6B' if 'power' in s else '#CCCCCC'
                   for s in spec_priority.index]
    
    bars1 = ax_e.bar(x - width/2, spec_priority['n_dns'], width,
                     label='# DNs', color=colors_bar1, alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
    
    ax_e2 = ax_e.twinx()
    bars2 = ax_e2.bar(x + width/2, spec_priority['total_synapses'], width,
                      label='Total Synapses', color='coral', alpha=0.7,
                      edgecolor='black', linewidth=1.5)
    
    ax_e.set_xticks(x)
    ax_e.set_xticklabels([s.replace('_', '\n') for s in spec_priority.index],
                         fontsize=10, fontweight='bold')
    ax_e.set_ylabel('Number of DNs', fontsize=12, fontweight='bold', color='steelblue')
    ax_e2.set_ylabel('Total Synapses', fontsize=12, fontweight='bold', color='coral')
    ax_e.set_title('E. Priority DN Specialization\nDNs Controlling Top 50 Hubs',
                   fontsize=13, fontweight='bold', loc='left')
    ax_e.tick_params(axis='y', labelcolor='steelblue')
    ax_e2.tick_params(axis='y', labelcolor='coral')
    ax_e.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        ax_e.text(bar.get_x() + bar.get_width()/2, height,
                 f'{int(height)}', ha='center', va='bottom', 
                 fontsize=10, fontweight='bold', color='steelblue')
    
    # Panel F: Scatter comparison
    ax_f = fig.add_subplot(gs[1, 2])
    
    # Merge both datasets
    overall_dict = dict(zip(dn_class['dn_name'], dn_class['total_connectivity']))
    priority_dict = dict(zip(dn_priority_summary['dn_name'], 
                            dn_priority_summary['total_synapses']))
    
    # Get all DNs that appear in priority analysis
    scatter_data = []
    for dn_name in dn_priority_summary['dn_name']:
        overall_conn = overall_dict.get(dn_name, 0)
        priority_conn = priority_dict.get(dn_name, 0)
        
        # Get specialization
        spec = dn_priority_summary[dn_priority_summary['dn_name']==dn_name]['dn_specialization'].values[0]
        
        scatter_data.append({
            'dn_name': dn_name,
            'overall': overall_conn,
            'priority': priority_conn,
            'spec': spec
        })
    
    scatter_df = pd.DataFrame(scatter_data)
    
    # Plot
    for spec, color in [('power_control', '#FF6B6B'), 
                       ('steering_control', '#4ECDC4'),
                       ('unknown', '#CCCCCC')]:
        subset = scatter_df[scatter_df['spec'] == spec]
        ax_f.scatter(subset['overall'], subset['priority'],
                    c=color, s=80, alpha=0.6, edgecolors='black',
                    linewidth=1, label=spec.replace('_', ' ').title())
    
    ax_f.set_xlabel('Total Connectivity (All INs)', fontsize=12, fontweight='bold')
    ax_f.set_ylabel('Priority Connectivity (Top 50 Hubs)', fontsize=12, fontweight='bold')
    ax_f.set_title('F. Breadth vs Focus\nEach Point = 1 DN',
                   fontsize=13, fontweight='bold', loc='left')
    ax_f.legend(loc='upper left', fontsize=9)
    ax_f.grid(alpha=0.3)
    
    # Add diagonal reference line
    max_val = max(ax_f.get_xlim()[1], ax_f.get_ylim()[1])
    ax_f.plot([0, max_val], [0, max_val], 'k--', alpha=0.2, linewidth=1)
    
    # ===== ROW 3: KEY INSIGHTS =====
    
    # Panel G: Overlap DNs (appear in both top 15)
    ax_g = fig.add_subplot(gs[2, 0])
    
    overlap_dns = list(top15_overall & top15_priority)
    
    if len(overlap_dns) > 0:
        # Get their stats
        overlap_stats = []
        for dn in overlap_dns:
            overall_val = top_overall[top_overall['dn_name']==dn]['total_connectivity'].values[0]
            priority_val = top_priority[top_priority['dn_name']==dn]['total_synapses'].values[0]
            spec = top_overall[top_overall['dn_name']==dn]['specialization'].values[0]
            
            overlap_stats.append({
                'dn': dn,
                'overall': overall_val,
                'priority': priority_val,
                'spec': spec
            })
        
        overlap_df = pd.DataFrame(overlap_stats).sort_values('priority', ascending=False)
        
        y_pos = np.arange(len(overlap_df))
        
        colors_overlap = ['#FF6B6B' if spec == 'power_control' else '#4ECDC4'
                         for spec in overlap_df['spec']]
        
        ax_g.barh(y_pos, overlap_df['priority'], color=colors_overlap,
                 edgecolor='black', linewidth=1.5, alpha=0.8)
        ax_g.set_yticks(y_pos)
        ax_g.set_yticklabels([str(dn)[:30] for dn in overlap_df['dn']], fontsize=9)
        ax_g.set_xlabel('Priority Synapses', fontsize=11, fontweight='bold')
        ax_g.set_title(f'G. "Super DNs" (n={len(overlap_dns)})\nHigh in BOTH Breadth & Focus',
                      fontsize=13, fontweight='bold', loc='left')
        ax_g.grid(axis='x', alpha=0.3)
        ax_g.invert_yaxis()
        
        ax_g.text(0.98, 0.02, '⭐ Best experimental\ntargets!',
                 transform=ax_g.transAxes, ha='right', va='bottom',
                 fontsize=11, fontweight='bold', color='#D32F2F',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    else:
        ax_g.text(0.5, 0.5, 'No overlap in top 15',
                 ha='center', va='center', fontsize=14,
                 transform=ax_g.transAxes)
        ax_g.axis('off')
    
    # Panel H: Summary table
    ax_h = fig.add_subplot(gs[2, 1:])
    ax_h.axis('off')
    
    summary_text = "KEY INSIGHTS:\n\n"
    
    summary_text += "NETWORK BREADTH (All 2,080 Premotor INs):\n"
    summary_text += f"  • Top DN: {top_overall.iloc[0]['dn_name']}\n"
    summary_text += f"  • Connectivity: {int(top_overall.iloc[0]['total_connectivity']):,} total synapses\n"
    summary_text += f"  • Specialization: {top_overall.iloc[0]['specialization'].replace('_', ' ')}\n"
    summary_text += f"  • Interpretation: Widespread network influence\n\n"
    
    summary_text += "EXPERIMENTAL FOCUS (Top 50 Priority Hub INs):\n"
    summary_text += f"  • Top DN: {top_priority.iloc[0]['dn_name']}\n"
    summary_text += f"  • Priority Connectivity: {int(top_priority.iloc[0]['total_synapses'])} synapses\n"
    summary_text += f"  • Priority Targets: {int(top_priority.iloc[0]['n_priority_ins'])} hub INs\n"
    summary_text += f"  • Interpretation: Controls critical experimental candidates\n\n"
    
    summary_text += "OVERLAP ANALYSIS:\n"
    if len(overlap_dns) > 0:
        summary_text += f"  • {len(overlap_dns)} DNs appear in BOTH top 15 lists\n"
        summary_text += f"  • These are 'Super DNs' - high breadth AND high focus\n"
        summary_text += f"  • Best targets for functional experiments\n\n"
    else:
        summary_text += f"  • No DNs in both top 15 lists\n"
        summary_text += f"  • Breadth and focus represent different DN populations\n\n"
    
    summary_text += "RECOMMENDATION:\n"
    summary_text += "  → For NETWORK studies: Use 'Network Breadth' DNs\n"
    summary_text += "  → For FUNCTIONAL experiments: Use 'Priority Focus' DNs\n"
    summary_text += "  → For HIGH-IMPACT studies: Use 'Super DNs' (overlap)\n"
    
    ax_h.text(0.05, 0.95, summary_text,
             transform=ax_h.transAxes, ha='left', va='top',
             fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='#F5F5F5', 
                      edgecolor='#666666', linewidth=2, alpha=0.9, pad=0.8))
    
    # Legend for whole figure
    legend_elements = [
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', label='Power Control DN'),
        mpatches.Patch(facecolor='#4ECDC4', edgecolor='black', label='Steering Control DN'),
        mpatches.Patch(facecolor='#FFD700', edgecolor='black', label='Super DN (Both Lists)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=3, fontsize=12, frameon=True, fancybox=True,
              bbox_to_anchor=(0.5, 0.01))
    
    return fig


def main():
    print("\n" + "="*70)
    print("CREATING COMBINED DN COMPARISON FIGURE")
    print("="*70 + "\n")
    
    fig = create_combined_comparison_figure()
    
    # Save
    output_dir = Path('results/dn_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_png = output_dir / 'DN_Analysis_Comparison.png'
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n  ✓ Saved PNG: {output_png}")
    
    output_pdf = output_dir / 'DN_Analysis_Comparison.pdf'
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved PDF: {output_pdf}")
    
    plt.close()
    
    print("\n" + "="*70)
    print("COMPARISON FIGURE COMPLETE")
    print("="*70)
    print(f"\nComprehensive comparison saved to: {output_dir}")
    print("\nThis figure shows:")
    print("  • Network Breadth: DNs with widespread influence")
    print("  • Experimental Focus: DNs controlling priority hubs")
    print("  • Overlap: 'Super DNs' appearing in both analyses")
    print("  • Complete comparison with recommendations")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()