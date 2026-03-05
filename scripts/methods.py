"""
Methods Figure - Complete Pipeline Overview

Single comprehensive figure explaining the entire analysis methodology.
Clean, presentation-worthy schematic.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

sys.path.append(str(Path(__file__).parent))


def create_methods_figure():
    """
    Create comprehensive methods overview figure.
    """
    
    fig = plt.figure(figsize=(20, 14), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(10, 13.5, 'Flight Control Circuit Analysis Pipeline',
           ha='center', va='top', fontsize=24, fontweight='bold')
    ax.text(10, 13, 'Systematic Reconstruction of DN → IN → MN Pathways in Drosophila BANC Connectome',
           ha='center', va='top', fontsize=14, style='italic', color='#555555')
    
    # ===== STEP 1: DATA =====
    y_start = 11.5
    
    # Step 1 box
    step1_box = FancyBboxPatch((0.5, y_start-1.2), 5, 1.5,
                              boxstyle="round,pad=0.15",
                              facecolor='#E8F4F8', 
                              edgecolor='#2E86AB', 
                              linewidth=3)
    ax.add_patch(step1_box)
    
    ax.text(3, y_start-0.2, 'STEP 1: Data Loading',
           ha='center', va='top', fontsize=14, fontweight='bold', color='#2E86AB')
    ax.text(3, y_start-0.5, 'FlyWire BANC Connectome',
           ha='center', va='top', fontsize=11)
    ax.text(3, y_start-0.75, '• 115,151 neurons',
           ha='center', va='top', fontsize=9)
    ax.text(3, y_start-0.95, '• 3.7M connections',
           ha='center', va='top', fontsize=9)
    
    # ===== STEP 2: CELL CLASSIFICATION =====
    step2_box = FancyBboxPatch((6.5, y_start-1.2), 5, 1.5,
                              boxstyle="round,pad=0.15",
                              facecolor='#FFF3E0',
                              edgecolor='#F57C00',
                              linewidth=3)
    ax.add_patch(step2_box)
    
    ax.text(9, y_start-0.2, 'STEP 2: Cell Type Classification',
           ha='center', va='top', fontsize=14, fontweight='bold', color='#F57C00')
    ax.text(9, y_start-0.55, 'DN: 1,313 | IN: 12,759 | MN: 831',
           ha='center', va='top', fontsize=10)
    ax.text(9, y_start-0.85, 'Wing MNs: 60 → 18 motor pools',
           ha='center', va='top', fontsize=9, style='italic')
    
    # ===== STEP 3: MOTOR ORGANIZATION =====
    step3_box = FancyBboxPatch((12.5, y_start-1.2), 5, 1.5,
                              boxstyle="round,pad=0.15",
                              facecolor='#F3E5F5',
                              edgecolor='#7B1FA2',
                              linewidth=3)
    ax.add_patch(step3_box)
    
    ax.text(15, y_start-0.2, 'STEP 3: Motor Pool Mapping',
           ha='center', va='top', fontsize=14, fontweight='bold', color='#7B1FA2')
    ax.text(15, y_start-0.55, 'Power: DLM, DVM (7 pools)',
           ha='center', va='top', fontsize=10)
    ax.text(15, y_start-0.85, 'Steering: b, III, i, hg (11 pools)',
           ha='center', va='top', fontsize=9, style='italic')
    
    # Arrows between steps 1-3
    arrow1 = FancyArrowPatch((5.5, y_start-0.5), (6.5, y_start-0.5),
                            arrowstyle='->', mutation_scale=30, 
                            linewidth=3, color='#333333')
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((11.5, y_start-0.5), (12.5, y_start-0.5),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='#333333')
    ax.add_patch(arrow2)
    
    # ===== STEP 4: IN CLUSTERING (CENTRAL/MAIN) =====
    y_mid = 8
    
    step4_box = FancyBboxPatch((2, y_mid-1.8), 16, 2.5,
                              boxstyle="round,pad=0.2",
                              facecolor='#E8F5E9',
                              edgecolor='#2E7D32',
                              linewidth=4)
    ax.add_patch(step4_box)
    
    ax.text(10, y_mid+0.5, 'STEP 4: Interneuron Hub Clustering (CORE ANALYSIS)',
           ha='center', va='top', fontsize=16, fontweight='bold', color='#2E7D32')
    
    # Sub-steps
    ax.text(3.5, y_mid-0.1, '4A. Find Premotor INs',
           ha='left', va='top', fontsize=11, fontweight='bold')
    ax.text(3.5, y_mid-0.4, 'IN → MN connections',
           ha='left', va='top', fontsize=9)
    ax.text(3.5, y_mid-0.6, '→ 2,080 premotor INs',
           ha='left', va='top', fontsize=9, color='#2E7D32', fontweight='bold')
    
    ax.text(7.5, y_mid-0.1, '4B. Hierarchical Clustering',
           ha='left', va='top', fontsize=11, fontweight='bold')
    ax.text(7.5, y_mid-0.4, 'Connectivity similarity',
           ha='left', va='top', fontsize=9)
    ax.text(7.5, y_mid-0.6, '→ 5 functional clusters',
           ha='left', va='top', fontsize=9, color='#2E7D32', fontweight='bold')
    
    ax.text(11.5, y_mid-0.1, '4C. Module Classification',
           ha='left', va='top', fontsize=11, fontweight='bold')
    ax.text(11.5, y_mid-0.4, 'Power vs Steering',
           ha='left', va='top', fontsize=9)
    ax.text(11.5, y_mid-0.6, '→ 1 Power + 4 Steering',
           ha='left', va='top', fontsize=9, color='#2E7D32', fontweight='bold')
    
    ax.text(15.5, y_mid-0.1, '4D. Hub Identification',
           ha='left', va='top', fontsize=11, fontweight='bold')
    ax.text(15.5, y_mid-0.4, 'Centrality measures',
           ha='left', va='top', fontsize=9)
    ax.text(15.5, y_mid-0.6, '→ 829 hub neurons',
           ha='left', va='top', fontsize=9, color='#2E7D32', fontweight='bold')
    
    # Vertical arrow from step 3 to step 4
    arrow3 = FancyArrowPatch((15, y_start-1.2), (15, y_mid+0.7),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='#333333')
    ax.add_patch(arrow3)
    
    # ===== STEP 5 & 6: DN PATHWAYS =====
    y_lower = 5
    
    # Step 5
    step5_box = FancyBboxPatch((0.5, y_lower-1.2), 5.5, 1.5,
                              boxstyle="round,pad=0.15",
                              facecolor='#FCE4EC',
                              edgecolor='#C2185B',
                              linewidth=3)
    ax.add_patch(step5_box)
    
    ax.text(3.25, y_lower-0.2, 'STEP 5: DN Pathway Mapping',
           ha='center', va='top', fontsize=14, fontweight='bold', color='#C2185B')
    ax.text(3.25, y_lower-0.55, 'DN → IN connections',
           ha='center', va='top', fontsize=10)
    ax.text(3.25, y_lower-0.85, 'Classify DN specializations',
           ha='center', va='top', fontsize=9, style='italic')
    
    # Step 6
    step6_box = FancyBboxPatch((7, y_lower-1.2), 5.5, 1.5,
                              boxstyle="round,pad=0.15",
                              facecolor='#E1F5FE',
                              edgecolor='#0277BD',
                              linewidth=3)
    ax.add_patch(step6_box)
    
    ax.text(9.75, y_lower-0.2, 'STEP 6: Direct Pathways',
           ha='center', va='top', fontsize=14, fontweight='bold', color='#0277BD')
    ax.text(9.75, y_lower-0.55, 'DN → MN (direct)',
           ha='center', va='top', fontsize=10)
    ax.text(9.75, y_lower-0.85, 'Compare direct vs indirect',
           ha='center', va='top', fontsize=9, style='italic')
    
    # Step 7
    step7_box = FancyBboxPatch((13.5, y_lower-1.2), 5.5, 1.5,
                              boxstyle="round,pad=0.15",
                              facecolor='#FFF9C4',
                              edgecolor='#F9A825',
                              linewidth=3)
    ax.add_patch(step7_box)
    
    ax.text(16.25, y_lower-0.2, 'STEP 7: Module Analysis',
           ha='center', va='top', fontsize=14, fontweight='bold', color='#F9A825')
    ax.text(16.25, y_lower-0.55, 'Modularity & integration',
           ha='center', va='top', fontsize=10)
    ax.text(16.25, y_lower-0.85, 'Statistical validation',
           ha='center', va='top', fontsize=9, style='italic')
    
    # Arrows from step 4 to steps 5-7
    arrow4 = FancyArrowPatch((5, y_mid-1.8), (3.25, y_lower+0.3),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='#333333')
    ax.add_patch(arrow4)
    
    arrow5 = FancyArrowPatch((10, y_mid-1.8), (9.75, y_lower+0.3),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='#333333')
    ax.add_patch(arrow5)
    
    arrow6 = FancyArrowPatch((15, y_mid-1.8), (16.25, y_lower+0.3),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='#333333')
    ax.add_patch(arrow6)
    
    # ===== OUTPUTS (BOTTOM) =====
    y_output = 2
    
    output_box = FancyBboxPatch((1, y_output-0.8), 18, 1.2,
                               boxstyle="round,pad=0.15",
                               facecolor='#EEEEEE',
                               edgecolor='#424242',
                               linewidth=3)
    ax.add_patch(output_box)
    
    ax.text(10, y_output+0.2, 'OUTPUTS & DELIVERABLES',
           ha='center', va='top', fontsize=14, fontweight='bold')
    
    outputs = [
        '• Complete pathway database',
        '• Functional module maps',
        '• Hub neuron rankings',
        '• Candidate prioritization',
        '• Publication figures',
        '• Statistical reports'
    ]
    
    x_positions = [2, 5.5, 9, 12.5, 16, 19.5]
    for i, (x, out) in enumerate(zip(x_positions[:6], outputs)):
        ax.text(x, y_output-0.2, out, ha='left', va='top', fontsize=9)
    
    # ===== KEY METHODS BOX =====
    methods_box = FancyBboxPatch((0.3, 0.2), 8, 1.3,
                                boxstyle="round,pad=0.1",
                                facecolor='#FAFAFA',
                                edgecolor='#666666',
                                linewidth=2)
    ax.add_patch(methods_box)
    
    ax.text(4.3, 1.3, 'Key Methods:',
           ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(1, 0.95, '• Hierarchical clustering (Ward linkage)',
           ha='left', va='top', fontsize=8)
    ax.text(1, 0.7, '• Cosine similarity on connectivity vectors',
           ha='left', va='top', fontsize=8)
    ax.text(1, 0.45, '• Network centrality (degree, betweenness)',
           ha='left', va='top', fontsize=8)
    ax.text(5, 0.95, '• Modularity score (community detection)',
           ha='left', va='top', fontsize=8)
    ax.text(5, 0.7, '• Min synapses threshold: 3',
           ha='left', va='top', fontsize=8)
    ax.text(5, 0.45, '• Multi-level pathway reconstruction',
           ha='left', va='top', fontsize=8)
    
    # ===== DATA SOURCE BOX =====
    data_box = FancyBboxPatch((11.5, 0.2), 8, 1.3,
                             boxstyle="round,pad=0.1",
                             facecolor='#FAFAFA',
                             edgecolor='#666666',
                             linewidth=2)
    ax.add_patch(data_box)
    
    ax.text(15.5, 1.3, 'Data Source:',
           ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(12, 0.95, 'FlyWire BANC Connectome (Princeton)',
           ha='left', va='top', fontsize=9, fontweight='bold')
    ax.text(12, 0.7, '• Complete VNC connectivity at synaptic resolution',
           ha='left', va='top', fontsize=8)
    ax.text(12, 0.45, '• Cell type annotations (Super Class, Function)',
           ha='left', va='top', fontsize=8)
    
    plt.tight_layout()
    
    return fig


def main():
    print("\n" + "="*70)
    print("CREATING METHODS FIGURE")
    print("="*70 + "\n")
    
    print("Generating comprehensive methods overview...")
    
    fig = create_methods_figure()
    
    # Save
    output_dir = Path('results/presentation_figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # PNG
    output_png = output_dir / 'Figure_1_Methods_Pipeline.png'
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved PNG: {output_png}")
    
    # PDF
    output_pdf = output_dir / 'Figure_1_Methods_Pipeline.pdf'
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved PDF: {output_pdf}")
    
    plt.close()
    
    print("\n" + "="*70)
    print("METHODS FIGURE COMPLETE")
    print("="*70)
    print(f"\nFigure 1 saved to: {output_dir}")
    print("  • Figure_1_Methods_Pipeline.png (300 DPI)")
    print("  • Figure_1_Methods_Pipeline.pdf (vector)")
    print("\nReady for presentations! 📊")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()