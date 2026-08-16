"""Rendering for Figure 3(a) of the LaGarNet T-RL manuscript.

The internal review objected that the previous version identified both the method rows and the
horizon columns by colour alone, which fails in greyscale and for colour-vision deficiency.  Rows
are now carried by marker *shape* (colour is kept but redundant) and the columns are labelled in
text under the bottom two panels.

`render_flattening_heatmap` takes the metric arrays and draws the figure; it is imported by cell 4
of `flattening_comparison_for_lagarnet_new.ipynb`, which builds those arrays from the evaluation
CSVs.  That notebook remains the source of truth.

`FIGURE_VALUES` below is a transcription of the numbers printed in the previous render
(`lagarnet_flattening_heatmap.png`, 4 Aug), made because the source CSVs live on the `/media/halid/T7`
drive, which is not mounted.  Running this module directly redraws the figure from that
transcription.  Re-run the notebook cell against the CSVs once the drive is back.
"""

import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple

METHOD_LABELS = [
    'Task-Specific SAC',
    'Task-Specific ClothFunnels',
    'Task-Specific Diffusion Policy',
    'Task-Specific LaGarNet',
    'All-Garment PlaNet-ClothPick',
    'All-Garment MEDOR',
    'Ours: All-Garment LaGarNet',
    'Human',
]

# One distinct shape per method so the rows survive greyscale printing; the star marks our method.
METHOD_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
METHOD_MARKER_SIZES = [20, 19, 21, 18, 21, 22, 28, 21]

STEP_LIMITS = [5, 10, 20, 30]

# Columns are (Max, Last) at 5, 10, 20 and 30 action steps; rows follow METHOD_LABELS.
FIGURE_VALUES = {
    'NC': np.array([
        [73.2, 64.9, 79.2, 63.7, 83.9, 64.4, 86.9, 67.1],
        [68.8, 66.7, 73.7, 70.5, 79.8, 76.1, 82.8, 75.9],
        [71.8, 67.8, 80.1, 77.6, 92.4, 91.5, 96.3, 90.8],
        [71.0, 65.2, 87.1, 79.1, 93.4, 80.9, 97.4, 87.0],
        [72.9, 71.8, 86.5, 84.5, 93.7, 88.3, 97.4, 92.9],
        [81.7, 80.9, 92.2, 91.6, 98.3, 97.9, 99.1, 98.6],
        [81.0, 80.2, 93.2, 90.3, 97.0, 94.4, 98.2, 95.5],
        [81.8, 80.1, 91.2, 89.5, 97.2, 95.4, 98.8, 98.0],
    ]),
    'NI': np.array([
        [41.4, 26.7, 53.4, 25.3, 63.9, 26.8, 70.9, 30.8],
        [34.4, 30.6, 44.8, 37.7, 57.3, 49.4, 64.0, 49.7],
        [39.7, 33.3, 56.9, 50.3, 83.4, 81.6, 91.3, 79.5],
        [37.4, 28.5, 71.8, 55.4, 85.7, 59.1, 94.6, 70.5],
        [42.6, 40.5, 71.2, 66.8, 86.2, 73.6, 94.3, 83.4],
        [61.0, 59.2, 83.0, 81.5, 96.6, 95.9, 98.3, 97.1],
        [61.6, 59.2, 86.5, 80.6, 94.4, 88.0, 96.7, 91.1],
        [63.0, 60.5, 82.4, 78.3, 94.3, 90.6, 97.8, 96.2],
    ]),
    'IoU': np.array([
        [62.2, 55.0, 66.5, 53.9, 69.6, 54.1, 71.4, 56.0],
        [58.3, 56.6, 62.9, 59.4, 66.5, 62.4, 68.5, 62.8],
        [61.8, 58.5, 69.1, 66.4, 78.3, 74.4, 83.7, 76.1],
        [60.3, 54.9, 72.5, 64.9, 77.8, 64.3, 82.3, 69.4],
        [60.5, 57.9, 70.0, 67.0, 74.1, 67.2, 78.3, 68.9],
        [71.1, 70.1, 76.9, 72.3, 82.2, 77.4, 84.1, 76.3],
        [67.3, 66.0, 77.5, 72.1, 81.2, 70.5, 82.7, 75.7],
        [68.1, 65.3, 78.3, 75.5, 85.5, 81.8, 87.7, 84.7],
    ]),
    'SR': np.array([
        [3.3, 0.0, 3.3, 0.0, 6.7, 0.0, 6.7, 0.0],
        [0.0, 0.0, 0.0, 0.0, 6.7, 3.3, 6.7, 3.3],
        [0.0, 0.0, 16.7, 13.3, 43.3, 33.3, 80.0, 46.7],
        [0.0, 0.0, 23.3, 10.0, 53.3, 16.7, 80.0, 20.0],
        [3.3, 3.3, 13.3, 10.0, 23.3, 6.7, 36.7, 3.3],
        [6.7, 0.0, 20.0, 13.3, 70.0, 40.0, 86.7, 40.0],
        [6.7, 3.3, 40.0, 23.3, 76.7, 13.3, 86.7, 36.7],
        [10.0, 10.0, 33.3, 23.3, 86.7, 73.3, 96.7, 83.3],
    ]),
}


def render_flattening_heatmap(data, labels=METHOD_LABELS, step_limits=STEP_LIMITS,
                              save_filename='lagarnet_flattening_heatmap.png'):
    """Draw the 2x2 metric heatmaps.

    `data` maps 'NC', 'NI', 'IoU' and 'SR' to (num_methods, 2 * len(step_limits)) arrays on a
    0-100 scale, with Max and Last alternating along the columns.
    """
    num_methods = len(labels)
    method_colors = sns.color_palette("tab10", num_methods)
    # Our method's row is ringed in red so it can be picked out without reading the legend.
    ours_index = next((i for i, lab in enumerate(labels) if lab.lower().startswith('ours')), None)

    # Taller than wide enough to give the 8 rows room; the width is what the paper scales to, so
    # extra height buys taller cells rather than a bigger figure on the page.
    fig, axes = plt.subplots(2, 2, figsize=(16, 16.5))
    plot_configs = [
        (axes[0, 0], data['NC'], 'Normalised Coverage (NC)'),
        (axes[0, 1], data['NI'], 'Normalised Improvement (NI)'),
        (axes[1, 0], data['IoU'], 'Max IoU'),
        (axes[1, 1], data['SR'], 'Success Rate (SR)'),
    ]

    for ax, arr, title in plot_configs:
        arr = np.asarray(arr, dtype=float)
        is_left = ax in (axes[0, 0], axes[1, 0])
        is_bottom = ax in (axes[1, 0], axes[1, 1])

        sns.heatmap(arr, annot=True, fmt='.1f', cmap='YlGnBu', vmin=0, vmax=100, square=False,
                    mask=np.isnan(arr),
                    xticklabels=False, yticklabels=False,
                    ax=ax, cbar=False, annot_kws={'size': 17, 'weight': 'bold'})

        ax.set_title(title, pad=12, fontsize=18, fontweight='bold')

        # Separate the (Max, Last) pairs so the column grouping reads without the labels.
        for boundary in range(2, arr.shape[1], 2):
            ax.axvline(boundary, color='white', linewidth=3)

        # Rows: one shape per method, drawn just left of the grid. Shape carries the identity,
        # colour only reinforces it, so the rows stay readable in greyscale.
        if is_left:
            for i in range(num_methods):
                ax.plot([-0.55], [i + 0.5], marker=METHOD_MARKERS[i], color=method_colors[i],
                        markersize=METHOD_MARKER_SIZES[i], linestyle='None', clip_on=False,
                        zorder=5)
                if i == ours_index:
                    ax.plot([-0.55], [i + 0.5], marker='o', markerfacecolor='none',
                            markeredgecolor='red', markeredgewidth=2.5,
                            markersize=METHOD_MARKER_SIZES[i] + 10, linestyle='None',
                            clip_on=False, zorder=6)

        # Columns: text labels under the bottom row of panels, replacing the coloured squares.
        if is_bottom:
            ax.set_xticks(np.arange(arr.shape[1]) + 0.5)
            ax.set_xticklabels(['Max', 'Last'] * len(step_limits), fontsize=18)
            ax.tick_params(axis='x', rotation=0, length=0, pad=8)
            for k, step in enumerate(step_limits):
                ax.text(2 * k + 1, -0.075, f'$N = {step}$', transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=22, fontweight='bold')
        ax.set_xlabel('')

    panel_right = 0.9
    # The strip under the panels holds the column labels and then the legend. Size it in inches so
    # the extra canvas height goes into the cells rather than into this margin.
    panel_bottom = 2.8 / fig.get_figheight()
    legend_top = 2.1 / fig.get_figheight()
    plt.subplots_adjust(wspace=0.03, hspace=0.19, right=panel_right, bottom=panel_bottom)

    method_handles = []
    for i in range(num_methods):
        handle = mlines.Line2D([], [], color=method_colors[i], marker=METHOD_MARKERS[i],
                               linestyle='None', markersize=METHOD_MARKER_SIZES[i] * 0.7)
        if i == ours_index:
            ring = mlines.Line2D([], [], marker='o', markerfacecolor='none', markeredgecolor='red',
                                 markeredgewidth=2.0, linestyle='None',
                                 markersize=(METHOD_MARKER_SIZES[i] + 10) * 0.7)
            handle = (handle, ring)
        method_handles.append(handle)
    # Start the legend at the left edge of the heatmap cells rather than centring it on the figure.
    legend_left = axes[1, 0].get_position().x0
    # A legend centres each marker half a handle-length inside its box, on top of the border pad,
    # so pull the box left by that much to line the markers up with the heatmap's left edge.
    legend_fontsize, handlelength = 16, 2.0
    marker_indent = (handlelength / 2) * legend_fontsize / 72 / fig.get_figwidth()
    legend = fig.legend(handles=method_handles, labels=list(labels), loc='upper left',
                        bbox_to_anchor=(legend_left - marker_indent, legend_top),
                        bbox_transform=fig.transFigure, ncol=4, fontsize=legend_fontsize,
                        title='Methods', title_fontsize=16, frameon=False, borderpad=0,
                        handlelength=handlelength,
                        handler_map={tuple: HandlerTuple(ndivide=1)})

    # Span the colourbar over exactly the panel block, so it follows any change to the geometry.
    cbar_bottom = axes[1, 0].get_position().y0
    cbar_top = axes[0, 0].get_position().y1
    cbar_ax = fig.add_axes([panel_right + 0.02, cbar_bottom, 0.015, cbar_top - cbar_bottom])
    sm = plt.cm.ScalarMappable(cmap='YlGnBu', norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Score (%)', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    # The legend is wider than the panels, so stretch the panels and carry the colourbar out with
    # them until the colourbar's right edge (bar, ticks and label) meets the legend's right edge.
    # Iterated because widening the panels nudges the colourbar's own tight bounds.
    for _ in range(3):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        delta = (legend.get_window_extent(renderer).x1
                 - cbar_ax.get_tightbbox(renderer).x1) / fig.bbox.width
        if abs(delta) < 1e-4:
            break
        panel_right += delta
        box = cbar_ax.get_position()
        cbar_ax.set_position([box.x0 + delta, box.y0, box.width, box.height])
        fig.subplots_adjust(right=panel_right)

    plt.savefig(save_filename, bbox_inches='tight', dpi=300)
    print(f'Saved {save_filename}')
    return fig


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    primary = os.path.join(here, 'lagarnet_flattening_heatmap.png')
    render_flattening_heatmap(FIGURE_VALUES, save_filename=primary)

    paper_copy = os.path.abspath(os.path.join(
        here, '..', '..', 'papers', 'lagarnet-TRL', 'plots', 'lagarnet_flattening_heatmap.png'))
    import shutil
    shutil.copyfile(primary, paper_copy)
    print(f'Copied to {paper_copy}')
