"""
Cluster Characterization Figure

Comprehensive figure explaining HOW premotor interneuron clusters were created
and WHAT each cluster represents biologically.

Panels:
  A. IN × Muscle-pool connectivity matrix (sorted by cluster) — shows the raw
     data used for clustering and why neurons grouped together.
  B. Cluster sizes and functional classification.
  C. Per-cluster muscle-type preference (% power vs steering synapses).
  D. Per-cluster primary muscle target profile (which muscles each cluster drives).
  E. Dendrogram (truncated to top 40 merges for readability).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).parent))
from utils.data_loader import ConnectomeDataLoader

# ── Colour scheme ────────────────────────────────────────────────────────────
CLUSTER_COLORS = {1: '#C5E1A5', 2: '#FF6B6B', 3: '#FFCC80', 4: '#90CAF9', 5: '#CE93D8'}
FUNC_COLORS = {
    'power_control':    '#FF6B6B',
    'steering_control': '#4ECDC4',
    'integrated_control': '#FFA07A',
    'unknown': '#CCCCCC'
}
MUSCLE_COLORS = {'power': '#FF6B6B', 'steering': '#4ECDC4'}


def build_in_pool_matrix(in_mn_conn, clustered_ins, motor_pools):
    """Build IN × muscle-pool connectivity matrix sorted by cluster."""

    mn_to_pool = {}
    mn_to_type = {}
    for _, pool in motor_pools.iterrows():
        for mn_id in pool['motor_neuron_ids']:
            mn_to_pool[mn_id] = pool['muscle']
            mn_to_type[mn_id]  = pool['muscle_type']

    pool_names = sorted(motor_pools['muscle'].unique())
    in_ids     = clustered_ins['Root ID'].tolist()
    cluster_of = dict(zip(clustered_ins['Root ID'], clustered_ins['cluster']))

    # Aggregate weights
    in_to_pool = {}
    for _, row in in_mn_conn.iterrows():
        src = row['source']
        pool = mn_to_pool.get(row['target'], None)
        if pool is None:
            continue
        in_to_pool.setdefault(src, {})
        in_to_pool[src][pool] = in_to_pool[src].get(pool, 0) + row['weight']

    conn = np.zeros((len(in_ids), len(pool_names)))
    for i, in_id in enumerate(in_ids):
        for j, pool in enumerate(pool_names):
            conn[i, j] = in_to_pool.get(in_id, {}).get(pool, 0)

    # Sort by cluster
    clusters_arr = np.array([cluster_of.get(in_id, 0) for in_id in in_ids])
    order = np.argsort(clusters_arr, kind='stable')
    return conn[order], clusters_arr[order], pool_names, mn_to_type


def compute_linkage(conn_matrix):
    """Compute Ward linkage on cosine-distance between INs."""
    sim  = cosine_similarity(conn_matrix + 1e-10)
    dist = np.maximum(1 - sim, 0)
    return linkage(squareform(dist, checks=False), method='ward')


def main():
    print("\n" + "="*70)
    print("CLUSTER CHARACTERIZATION FIGURE")
    print("="*70 + "\n")

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading data...")
    cluster_info  = pd.read_csv('results/interneuron_clusters/cluster_characteristics.csv')
    clustered_ins = pd.read_csv('results/interneuron_clusters/premotor_interneurons_clustered.csv')
    in_mn_conn    = pd.read_csv('results/interneuron_clusters/in_to_mn_connections.csv')
    motor_pools   = pd.read_csv('results/motor_neurons/motor_pools.csv')
    motor_pools['motor_neuron_ids'] = motor_pools['motor_neuron_ids'].apply(eval)

    loader = ConnectomeDataLoader('config.yaml')
    loader.load_all_data(verbose=False)

    # ── Build connectivity matrix ─────────────────────────────────────────
    print("Building IN × muscle-pool matrix...")
    conn_sorted, clusters_sorted, pool_names, mn_to_type = build_in_pool_matrix(
        in_mn_conn, clustered_ins, motor_pools
    )

    # Cluster boundary positions (row indices where cluster changes)
    boundaries = [0]
    for k in range(1, len(clusters_sorted)):
        if clusters_sorted[k] != clusters_sorted[k - 1]:
            boundaries.append(k)
    boundaries.append(len(clusters_sorted))

    # Pool type lookup (for x-axis colouring)
    pool_type = {row['muscle']: row['muscle_type']
                 for _, row in motor_pools.iterrows()}

    # ── Linkage for dendrogram ────────────────────────────────────────────
    print("Computing linkage (this may take a moment)...")
    Z = compute_linkage(conn_sorted)

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 20), dpi=300)
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            height_ratios=[2.2, 1.2, 1.1],
                            hspace=0.50, wspace=0.38)

    # =================================================================
    # PANEL A: IN × Muscle pool connectivity matrix
    # =================================================================
    ax_A = fig.add_subplot(gs[0, :])

    plot_data = np.log10(conn_sorted + 1)
    im = ax_A.imshow(plot_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    # Cluster boundary horizontal lines
    for b in boundaries[1:-1]:
        ax_A.axhline(b - 0.5, color='white', linewidth=2, linestyle='--')

    # Cluster labels on left (coloured text boxes)
    for ci, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        cl_id  = int(clusters_sorted[start])
        row    = cluster_info[cluster_info['cluster_id'] == cl_id].iloc[0]
        func   = row['functional_type']
        n_ins  = row['n_interneurons']
        label  = f"C{cl_id}\n{'Power' if func == 'power_control' else 'Steering'}\nn={n_ins}"
        mid_y  = (start + end) / 2
        ax_A.text(-0.5, mid_y, label,
                  transform=ax_A.get_yaxis_transform(),
                  ha='right', va='center', fontsize=8, fontweight='bold',
                  color=FUNC_COLORS.get(func, 'black'),
                  bbox=dict(boxstyle='round,pad=0.2',
                            facecolor=CLUSTER_COLORS.get(cl_id, '#CCCCCC'),
                            alpha=0.6, edgecolor='grey', linewidth=0.5))

    # X-axis: muscle pool names, coloured by power/steering
    ax_A.set_xticks(range(len(pool_names)))
    ax_A.set_xticklabels(pool_names, rotation=90, fontsize=8)
    for tick, pool in zip(ax_A.get_xticklabels(), pool_names):
        ptype = pool_type.get(pool, 'unknown')
        tick.set_color(MUSCLE_COLORS.get(ptype, 'black'))
        tick.set_fontweight('bold')

    ax_A.set_yticks([])
    ax_A.set_xlabel('Muscle Pool  (red = power muscle · teal = steering muscle)',
                    fontsize=12, fontweight='bold')
    ax_A.set_ylabel('Premotor Interneurons\n(sorted by cluster)', fontsize=12, fontweight='bold')
    ax_A.set_title('A.  IN → Muscle Pool Connectivity Matrix\n'
                   'Each row = one premotor interneuron; colour = log₁₀(synapses+1). '
                   'Neurons with similar target profiles cluster together.',
                   fontsize=13, fontweight='bold', pad=10)
    plt.colorbar(im, ax=ax_A, label='log₁₀(synapses + 1)', fraction=0.015, pad=0.01)

    # =================================================================
    # PANEL B: Cluster sizes & functional type
    # =================================================================
    ax_B = fig.add_subplot(gs[1, 0])

    cl_ids = cluster_info['cluster_id'].tolist()
    n_each = cluster_info['n_interneurons'].tolist()
    funcs  = cluster_info['functional_type'].tolist()
    bar_colors = [FUNC_COLORS.get(f, '#CCCCCC') for f in funcs]

    bars = ax_B.bar([f'C{c}' for c in cl_ids], n_each, color=bar_colors,
                    edgecolor='black', linewidth=1)
    for bar, n in zip(bars, n_each):
        ax_B.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                  str(n), ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax_B.set_xlabel('Cluster', fontsize=11, fontweight='bold')
    ax_B.set_ylabel('Number of Interneurons', fontsize=11, fontweight='bold')
    ax_B.set_title('B.  Cluster Sizes\n& Functional Type', fontsize=12, fontweight='bold')
    ax_B.grid(axis='y', alpha=0.3)

    legend_handles = [mpatches.Patch(facecolor=FUNC_COLORS['power_control'],    label='Power control'),
                      mpatches.Patch(facecolor=FUNC_COLORS['steering_control'], label='Steering control')]
    ax_B.legend(handles=legend_handles, fontsize=9, loc='upper right')

    # =================================================================
    # PANEL C: Per-cluster muscle-type preference (% synapses)
    # =================================================================
    ax_C = fig.add_subplot(gs[1, 1])

    power_pools    = set(motor_pools[motor_pools['muscle_type'] == 'power']['muscle'])
    steering_pools = set(motor_pools[motor_pools['muscle_type'] == 'steering']['muscle'])

    pct_power    = []
    pct_steering = []

    for cl_id in cl_ids:
        mask = clusters_sorted == cl_id
        sub  = conn_sorted[mask]
        total = sub.sum()
        if total == 0:
            pct_power.append(0);    pct_steering.append(0)
            continue
        pw  = sum(sub[:, i].sum() for i, p in enumerate(pool_names) if p in power_pools)
        st  = sum(sub[:, i].sum() for i, p in enumerate(pool_names) if p in steering_pools)
        pct_power.append(100 * pw / total)
        pct_steering.append(100 * st / total)

    x = np.arange(len(cl_ids))
    ax_C.bar(x, pct_power,    label='Power muscles',    color=FUNC_COLORS['power_control'],    alpha=0.85, edgecolor='black')
    ax_C.bar(x, pct_steering, bottom=pct_power,         label='Steering muscles', color=FUNC_COLORS['steering_control'], alpha=0.85, edgecolor='black')
    ax_C.set_xticks(x)
    ax_C.set_xticklabels([f'C{c}' for c in cl_ids])
    ax_C.set_ylabel('% of Total Synapses', fontsize=11, fontweight='bold')
    ax_C.set_title('C.  Muscle-Type Preference\nper Cluster', fontsize=12, fontweight='bold')
    ax_C.set_ylim(0, 105)
    ax_C.legend(fontsize=9)
    ax_C.grid(axis='y', alpha=0.3)

    # =================================================================
    # PANEL D: Per-cluster top muscle targets heatmap
    # =================================================================
    ax_D = fig.add_subplot(gs[1, 2])

    # Build 5-cluster × N-pool matrix (mean synapses per IN per pool)
    cluster_profiles = np.zeros((len(cl_ids), len(pool_names)))
    for ci, cl_id in enumerate(cl_ids):
        mask = clusters_sorted == cl_id
        sub  = conn_sorted[mask]
        cluster_profiles[ci] = sub.mean(axis=0)

    im_D = ax_D.imshow(np.log10(cluster_profiles + 1), aspect='auto',
                       cmap='Blues', interpolation='nearest')
    ax_D.set_yticks(range(len(cl_ids)))
    ax_D.set_yticklabels([f'C{c}' for c in cl_ids], fontsize=10, fontweight='bold')
    ax_D.set_xticks(range(len(pool_names)))
    ax_D.set_xticklabels(pool_names, rotation=90, fontsize=7)
    for tick, pool in zip(ax_D.get_xticklabels(), pool_names):
        tick.set_color(MUSCLE_COLORS.get(pool_type.get(pool, 'unknown'), 'black'))
        tick.set_fontweight('bold')
    ax_D.set_title('D.  Average Target Profile\nper Cluster', fontsize=12, fontweight='bold')
    ax_D.set_xlabel('Muscle Pool', fontsize=10, fontweight='bold')
    ax_D.set_ylabel('Cluster', fontsize=10, fontweight='bold')
    plt.colorbar(im_D, ax=ax_D, label='log₁₀(mean syn+1)', fraction=0.06, pad=0.04)

    # =================================================================
    # PANEL E: Dendrogram (truncated)
    # =================================================================
    ax_E = fig.add_subplot(gs[2, :])

    dend_colors = {}
    for ci, cl_id in enumerate(cl_ids):
        func = cluster_info[cluster_info['cluster_id'] == cl_id]['functional_type'].values[0]
        dend_colors[cl_id] = FUNC_COLORS.get(func, 'black')

    dendrogram(Z, ax=ax_E,
               truncate_mode='lastp', p=40,
               no_labels=True,
               above_threshold_color='grey',
               color_threshold=0)

    ax_E.set_xlabel('Merged Interneuron Groups  (showing final 40 merges)', fontsize=11, fontweight='bold')
    ax_E.set_ylabel('Ward Distance', fontsize=11, fontweight='bold')
    ax_E.set_title('E.  Hierarchical Clustering Dendrogram\n'
                   'Ward linkage on cosine distance between IN → muscle-pool connectivity vectors.',
                   fontsize=12, fontweight='bold')
    ax_E.grid(axis='y', alpha=0.3)

    # Cluster legend on dendrogram
    legend_handles_E = [
        mpatches.Patch(facecolor=CLUSTER_COLORS[cl], label=
            f"C{cl}  {cluster_info[cluster_info['cluster_id']==cl]['functional_type'].values[0].replace('_',' ')} "
            f"(n={cluster_info[cluster_info['cluster_id']==cl]['n_interneurons'].values[0]})")
        for cl in cl_ids
    ]
    ax_E.legend(handles=legend_handles_E, loc='upper right', fontsize=9, framealpha=0.9)

    # ── Overall title ─────────────────────────────────────────────────────
    fig.suptitle('Premotor Interneuron Cluster Characterization\n'
                 'How clusters were built and what they represent',
                 fontsize=16, fontweight='bold', y=0.995)

    # ── Save ─────────────────────────────────────────────────────────────
    output_dir = Path('results/interneuron_clusters/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'cluster_characterization.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    try:
        plt.savefig(output_dir / 'cluster_characterization.pdf',
                    format='pdf', bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved: cluster_characterization.png/pdf")
    except PermissionError:
        print(f"\n✓ Saved: cluster_characterization.png  (close PDF viewer to save PDF)")

    plt.close()
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
