#!/usr/bin/env python3
"""Generate explanatory diagrams for the Fire Detection presentation.

All text is 18pt minimum for presentation readability.

Usage:
    python plot_presentation_diagrams.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def _box(ax, x, y, w, h, text, color='#3B82F6', text_color='white',
         fontsize=18, linewidth=2):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='#333333',
                         linewidth=linewidth)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, wrap=True)


def _arrow(ax, x1, y1, x2, y2, color='#555555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=3))


def plot_mlp_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(22, 11))
    ax.set_xlim(-1, 22)
    ax.set_ylim(-1, 11)
    ax.axis('off')
    ax.set_title('How the MLP Fire Detector Works\n'
                 '"A chain of simple math filters that learn to spot fire"',
                 fontsize=26, fontweight='bold', pad=20)

    _box(ax, 0, 2.5, 4, 5.5,
         '12 Measurements\n\nT4 max\nT4 mean\nT11 mean\n'
         'dT max\nSWIR max/mean\nRed/NIR mean\n'
         'NDVI min/mean\nNDVI drop\nobs count',
         color='#059669', fontsize=14)
    ax.text(2, 8.5, 'INPUT\n(what the sensor sees)',
            ha='center', fontsize=20, fontweight='bold', color='#059669')

    _arrow(ax, 4.2, 5.5, 5.5, 5.5)

    _box(ax, 5.5, 3.5, 3, 4,
         '64 neurons\n\n"Is it hot?"\n"Is veg gone?"',
         color='#3B82F6', fontsize=18)
    ax.text(7, 8, 'Layer 1\n(detect patterns)',
            ha='center', fontsize=18, fontweight='bold', color='#3B82F6')

    _arrow(ax, 8.7, 5.5, 10, 5.5)

    _box(ax, 10, 3.5, 3, 4,
         '32 neurons\n\n"Combine\nthe clues"',
         color='#3B82F6', fontsize=18)
    ax.text(11.5, 8, 'Layer 2\n(combine clues)',
            ha='center', fontsize=18, fontweight='bold', color='#3B82F6')

    _arrow(ax, 13.2, 5.5, 14.5, 5.5)

    _box(ax, 14.5, 3.5, 3, 4,
         'P(fire)\n\n0.0 = no fire\n1.0 = fire!',
         color='#DC2626', fontsize=18)
    ax.text(16, 8, 'OUTPUT\n(fire probability)',
            ha='center', fontsize=20, fontweight='bold', color='#DC2626')

    _arrow(ax, 17.7, 5.5, 18.8, 5.5)

    _box(ax, 18.8, 4.0, 2.5, 3,
         'P > 0.5?\n\nFIRE!',
         color='#F59E0B', text_color='black', fontsize=20)

    ax.text(11, 0.3,
            'Each "neuron" = ReLU(weight × input + bias)    |    '
            'Training = adjusting weights to get the right answer\n'
            'Best model: 12 → 64 → 32 → 1  (only 2,945 parameters!  ~12 KB)',
            ha='center', fontsize=18, color='#444444',
            bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.9))

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/diagram_mlp_architecture.png', dpi=200, bbox_inches='tight')
    print('Saved plots/diagram_mlp_architecture.png')
    plt.close()


def plot_tabpfn_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(22, 11))
    ax.set_xlim(-1, 22)
    ax.set_ylim(-1, 11)
    ax.axis('off')
    ax.set_title('How TabPFN Works\n'
                 '"A pre-trained brain that learns by looking at examples"',
                 fontsize=26, fontweight='bold', pad=20)

    _box(ax, 0, 7.5, 7, 2.5,
         'Pre-Training (done once, by researchers)\n\n'
         'Trained on millions of synthetic tabular datasets\n'
         '"Learned what patterns in tables look like"',
         color='#7C3AED', fontsize=15)
    _arrow(ax, 3.5, 7.3, 3.5, 6.5)

    _box(ax, 0.5, 3.5, 5.5, 2.8,
         'TabPFN Transformer\n\n7.2 million parameters\n~28 MB model\n'
         '(pre-loaded with\ntable knowledge)',
         color='#7C3AED', fontsize=15)

    _box(ax, 7.5, 7, 5.5, 2.5,
         'Training Examples (context)\n\n'
         '"Here are 1000 pixels I know\n'
         'are fire, and 1000 I know are not"',
         color='#059669', fontsize=15)
    _arrow(ax, 10.25, 6.8, 9, 6)

    _box(ax, 7, 3.5, 6, 2.5,
         'In-Context Learning\n\n'
         'Reads examples + new pixel together\n'
         '"This new pixel looks like the fire ones"',
         color='#3B82F6', fontsize=15)

    _box(ax, 14, 7, 4.5, 2.5,
         'New Pixel\n\n12 measurements\n(same features as MLP)',
         color='#F59E0B', text_color='black', fontsize=15)
    _arrow(ax, 16.25, 6.8, 14.5, 5.8)
    _arrow(ax, 13.2, 4.75, 15, 4.75)

    _box(ax, 15, 3.2, 3.5, 3,
         'P(fire)\n\n0.0 = no fire\n1.0 = fire!',
         color='#DC2626', fontsize=18)

    _box(ax, 19, 3.5, 2.5, 2.5,
         'n_estimators\n\nRun N times\nwith different\npreprocessing\n→ average',
         color='#6B7280', fontsize=13)
    _arrow(ax, 18.7, 4.75, 19, 4.75)

    ax.text(11, 0.3,
            'Key idea: TabPFN reads examples at inference time — like showing a smart kid examples then asking "is this fire?"\n'
            'Fine-tuning: we adjust the pre-trained weights specifically for our fire data (20 epochs)',
            ha='center', fontsize=18, color='#444444',
            bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.9))

    plt.tight_layout()
    plt.savefig('plots/diagram_tabpfn_architecture.png', dpi=200, bbox_inches='tight')
    print('Saved plots/diagram_tabpfn_architecture.png')
    plt.close()


def plot_dataset_organization():
    fig, ax = plt.subplots(1, 1, figsize=(24, 12))
    ax.set_xlim(-0.5, 24)
    ax.set_ylim(-0.5, 12)
    ax.axis('off')
    ax.set_title('How the Data is Organized\n'
                 '"From airplane scans to a table the model can read"',
                 fontsize=26, fontweight='bold', pad=20)

    _box(ax, 0, 6, 3.5, 4.5,
         '4 Flights\n\n03: Pre-burn\n04: Active fire\n05: Overnight\n06: Smoldering',
         color='#1E40AF', fontsize=15)
    ax.text(1.75, 11, 'Step 1', ha='center', fontsize=20, fontweight='bold',
            color='#1E40AF')

    _arrow(ax, 3.7, 8.25, 4.8, 8.25)

    _box(ax, 4.8, 6, 3.5, 4.5,
         '61 HDF Files\n\nEach file =\n1 scan sweep\n\n50 channels\n+ lat/lon',
         color='#1E40AF', fontsize=15)
    ax.text(6.55, 11, 'Step 2', ha='center', fontsize=20, fontweight='bold',
            color='#1E40AF')

    _arrow(ax, 8.5, 8.25, 9.6, 8.25)

    _box(ax, 9.6, 6, 3.5, 4.5,
         'Millions of\nRaw Pixels\n\nEach pixel:\nT4, T11, SWIR\nRed, NIR, NDVI\nlat, lon',
         color='#059669', fontsize=15)
    ax.text(11.35, 11, 'Step 3', ha='center', fontsize=20, fontweight='bold',
            color='#059669')

    _arrow(ax, 13.3, 8.25, 14.4, 8.25)

    _box(ax, 14.4, 6, 3.5, 4.5,
         'Regular Grid\n\n0.00025° cells\n(~28 m each)\n\nStack multiple\nsweeps per cell',
         color='#F59E0B', text_color='black', fontsize=15)
    ax.text(16.15, 11, 'Step 4: Grid', ha='center', fontsize=20,
            fontweight='bold', color='#F59E0B')

    _arrow(ax, 18.1, 8.25, 19.2, 8.25)

    _box(ax, 19.2, 6, 4, 4.5,
         '12 Features\nper cell\n\nT4_max, T4_mean\nT11_mean, dT_max\nSWIR_max/mean\n'
         'NDVI_min/mean/drop\nobs_count',
         color='#DC2626', fontsize=14)
    ax.text(21.2, 11, 'Step 5: Features', ha='center', fontsize=20,
            fontweight='bold', color='#DC2626')

    ax.text(12, 5, '\u2193  Result: a table with one row per grid cell  \u2193',
            ha='center', fontsize=22, fontweight='bold', color='#444444')

    headers = ['T4_max', 'T4_mean', 'T11_mean', 'dT_max', '...', 'obs_count', 'label']
    values1 = ['412 K', '380 K', '295 K', '85 K', '...', '5', 'FIRE']
    values2 = ['298 K', '295 K', '292 K', '3 K', '...', '3', 'no fire']

    x_start = 3.5
    col_w = 2.4
    for i, (h, v1, v2) in enumerate(zip(headers, values1, values2)):
        x = x_start + i * col_w
        _box(ax, x, 3.5, col_w - 0.15, 1.0, h, color='#374151', fontsize=15)
        c1 = '#FEE2E2' if v1 == 'FIRE' else '#FEF3C7'
        _box(ax, x, 2.2, col_w - 0.15, 1.0, v1,
             color=c1, text_color='black', fontsize=18, linewidth=1)
        _box(ax, x, 0.9, col_w - 0.15, 1.0, v2,
             color='#D1FAE5', text_color='black', fontsize=18, linewidth=1)

    ax.text(1.8, 4.0, 'Feature:', ha='center', fontsize=18,
            fontweight='bold', color='#444444')
    ax.text(1.8, 2.7, 'Fire pixel:', ha='center', fontsize=18,
            fontweight='bold', color='#DC2626')
    ax.text(1.8, 1.4, 'Normal:', ha='center', fontsize=18,
            fontweight='bold', color='#059669')

    ax.text(12, 0.0,
            'Fire pixels are HOT (high T4) and have BIG temperature differences (high dT) '
            '\u2014 the model learns these patterns!',
            ha='center', fontsize=20, color='#444444',
            bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.9))

    plt.tight_layout()
    plt.savefig('plots/diagram_dataset_organization.png', dpi=200, bbox_inches='tight')
    print('Saved plots/diagram_dataset_organization.png')
    plt.close()


def plot_model_io():
    fig, ax = plt.subplots(1, 1, figsize=(24, 12))
    ax.set_xlim(-0.5, 24)
    ax.set_ylim(-1, 12)
    ax.axis('off')
    ax.set_title('What Goes In, What Comes Out\n'
                 '"The model reads 12 numbers and answers: fire or not?"',
                 fontsize=26, fontweight='bold', pad=20)

    ax.text(3.5, 11, 'INPUT: 12 numbers per location',
            ha='center', fontsize=22, fontweight='bold', color='#059669')

    features = [
        ('T4_max', '412 K', 'Hottest temp at 3.9\u03bcm'),
        ('T4_mean', '380 K', 'Average temp at 3.9\u03bcm'),
        ('T11_mean', '295 K', 'Background temp at 11\u03bcm'),
        ('dT_max', '85 K', 'Max temp difference'),
        ('SWIR_max', '142', 'Max shortwave IR'),
        ('SWIR_mean', '95', 'Avg shortwave IR'),
        ('Red_mean', '28', 'Avg visible red'),
        ('NIR_mean', '35', 'Avg near-infrared'),
        ('NDVI_min', '0.05', 'Worst veg health'),
        ('NDVI_mean', '0.18', 'Avg veg health'),
        ('NDVI_drop', '0.32', 'Veg loss (burned!)'),
        ('obs_count', '5', 'Times we saw this spot'),
    ]
    for i, (name, val, desc) in enumerate(features):
        y = 10 - i * 0.82
        c = '#DC2626' if 'T4' in name or 'dT' in name else (
            '#F59E0B' if 'NDVI' in name else '#3B82F6')
        _box(ax, 0, y, 2.0, 0.65, name, color=c, fontsize=13, linewidth=1)
        ax.text(2.3, y + 0.32, f'= {val}', fontsize=15, va='center',
                fontweight='bold', color='#333')
        ax.text(4.2, y + 0.32, desc, fontsize=14, va='center', color='#555')

    ax.text(1, 0.2, 'Thermal', fontsize=18, fontweight='bold', color='#DC2626')
    ax.text(3.5, 0.2, 'Vegetation', fontsize=18, fontweight='bold', color='#F59E0B')
    ax.text(6.5, 0.2, 'Reflectance', fontsize=18, fontweight='bold', color='#3B82F6')

    _arrow(ax, 8.5, 5.5, 10, 5.5)
    ax.text(9.25, 6.2, 'feed into', ha='center', fontsize=18, color='#666',
            fontstyle='italic')

    _box(ax, 10, 2.5, 5, 6.5, '', color='#F3F4F6', text_color='black', linewidth=3)
    ax.text(12.5, 8.5, 'MODEL', ha='center', fontsize=22, fontweight='bold', color='#333')

    _box(ax, 10.3, 6, 4.4, 2,
         'MLP\n2,945 params | 12 KB | ~2 min', color='#3B82F6', fontsize=14)
    _box(ax, 10.3, 3, 4.4, 2,
         'TabPFN\n7.2M params | 83 MB | ~10 min', color='#7C3AED', fontsize=14)
    ax.text(12.5, 2.5, '(pick one)', ha='center', fontsize=18,
            color='#666', fontstyle='italic')

    _arrow(ax, 15.2, 5.5, 16.5, 5.5)

    ax.text(20, 11, 'OUTPUT: one number',
            ha='center', fontsize=22, fontweight='bold', color='#DC2626')

    _box(ax, 16.5, 4.5, 6.5, 5.5, '', color='white', linewidth=2)
    ax.text(19.75, 9.5, 'P(fire) = 0.93', ha='center', fontsize=36,
            fontweight='bold', color='#DC2626')
    ax.text(19.75, 8.0, '"93% chance this is fire"',
            ha='center', fontsize=20, color='#666', fontstyle='italic')
    ax.text(19.75, 6.5, 'If P(fire) > 0.5', ha='center', fontsize=20,
            fontweight='bold', color='#333')
    _box(ax, 17.5, 5.2, 4.5, 1, 'FIRE DETECTED', color='#DC2626', fontsize=20)
    ax.text(19.75, 4.0, 'If P(fire) \u2264 0.5', ha='center', fontsize=20,
            fontweight='bold', color='#333')
    _box(ax, 17.5, 2.8, 4.5, 1, 'No fire', color='#059669', fontsize=20)

    ax.text(12, -0.5,
            'The model sees NUMBERS, not pictures.  Each grid cell becomes 12 numbers \u2192 1 answer.',
            ha='center', fontsize=20, color='#444444',
            bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.9))

    plt.tight_layout()
    plt.savefig('plots/diagram_model_io.png', dpi=200, bbox_inches='tight')
    print('Saved plots/diagram_model_io.png')
    plt.close()


def plot_model_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(22, 12))
    fig.suptitle('MLP vs TabPFN  \u2014  Two Very Different Approaches',
                 fontsize=26, fontweight='bold', y=0.98)

    for side, (ax, title, color, items) in enumerate(zip(
        axes,
        ['FireMLP', 'TabPFN'],
        ['#3B82F6', '#7C3AED'],
        [
            [('Approach', 'Learn weights from scratch\nby seeing data 100 times'),
             ('Size', '2,945 parameters\n~12 KB on disk'),
             ('Memory', '<1 MB during training\nRuns on any laptop CPU'),
             ('Speed', 'Train: ~2 min\nPredict: instant'),
             ('Analogy', 'Student who studies the\ntextbook 100 times')],
            [('Approach', 'Pre-trained on millions of\nsynthetic tables, fine-tuned'),
             ('Size', '7.2 million parameters\n~83 MB  (2,400\u00d7 bigger!)'),
             ('Memory', '~1\u20132 GB during training\nBenefits from GPU'),
             ('Speed', 'Train: ~10 min\nPredict: ~1 sec'),
             ('Analogy', 'Expert who already knows\nstatistics, shown our data')],
        ],
    )):
        ax.set_xlim(-0.5, 11)
        ax.set_ylim(-0.5, 10.5)
        ax.axis('off')
        ax.set_title(title, fontsize=24, fontweight='bold', color=color, pad=15)

        for i, (label, text) in enumerate(items):
            y = 8.5 - i * 1.9
            _box(ax, 0.2, y - 0.3, 2.8, 1.5, label, color=color, fontsize=18)
            ax.text(3.3, y + 0.45, text, fontsize=18, va='center',
                    color='#333', linespacing=1.4)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.text(0.5, 0.01,
             'Both models get the SAME 12 input features and produce the SAME output: P(fire) per grid cell',
             ha='center', fontsize=20, color='#444444',
             bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.9))

    plt.savefig('plots/diagram_model_comparison.png', dpi=200, bbox_inches='tight')
    print('Saved plots/diagram_model_comparison.png')
    plt.close()


def plot_threshold_explanation():
    """Explain how the traditional T4-T11 threshold fire detection works."""
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle('Traditional Fire Detection: The T4 – T11 Method\n'
                 '"If it\'s hot AND hotter than its surroundings → probably fire"',
                 fontsize=26, fontweight='bold', y=0.98)

    # Left panel: conceptual scatter diagram
    ax = axes[0]
    ax.set_xlim(270, 450)
    ax.set_ylim(-10, 120)

    # Scatter "normal" background pixels
    rng = np.random.RandomState(42)
    n_bg = 400
    t4_bg = rng.normal(295, 8, n_bg)
    dt_bg = rng.normal(3, 3, n_bg)
    dt_bg = np.clip(dt_bg, -5, 15)
    ax.scatter(t4_bg, dt_bg, s=30, c='#3B82F6', alpha=0.5, label='Normal ground', zorder=3)

    # Scatter "warm" pixels (sunlit rock, pavement)
    n_warm = 60
    t4_warm = rng.normal(320, 8, n_warm)
    dt_warm = rng.normal(5, 3, n_warm)
    ax.scatter(t4_warm, dt_warm, s=30, c='#F59E0B', alpha=0.6, label='Warm surfaces', zorder=3)

    # Scatter "fire" pixels
    n_fire = 80
    t4_fire = rng.normal(380, 30, n_fire)
    t4_fire = np.clip(t4_fire, 330, 450)
    dt_fire = rng.normal(60, 25, n_fire)
    dt_fire = np.clip(dt_fire, 15, 120)
    ax.scatter(t4_fire, dt_fire, s=50, c='#DC2626', alpha=0.7, label='Fire pixels', zorder=4)

    # Draw threshold lines
    ax.axvline(x=325, color='#DC2626', linestyle='--', linewidth=3, zorder=5)
    ax.axhline(y=10, color='#DC2626', linestyle='--', linewidth=3, zorder=5)
    ax.text(327, 115, 'T4 > 325 K', fontsize=20, fontweight='bold',
            color='#DC2626', va='top')
    ax.text(445, 12, 'T4 – T11 > 10 K', fontsize=20, fontweight='bold',
            color='#DC2626', ha='right', va='bottom')

    # Shade the "fire detected" quadrant
    ax.fill_between([325, 450], 10, 120, alpha=0.08, color='red', zorder=1)
    ax.text(387, 100, 'FIRE\nDETECTED', fontsize=28, fontweight='bold',
            color='#DC2626', ha='center', va='center', alpha=0.4)

    ax.set_xlabel('T4 Brightness Temperature (K)  —  3.9 μm channel', fontsize=20)
    ax.set_ylabel('T4 – T11 (K)  —  temperature difference', fontsize=20)
    ax.set_title('How the Threshold Test Works', fontsize=22, fontweight='bold')
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=18, loc='upper left')

    ax.text(0.03, 0.50,
            'Rule: a pixel is fire if BOTH:\n'
            '  1. T4 > 325 K  (it\'s hot at 3.9 μm)\n'
            '  2. T4 – T11 > 10 K  (much hotter\n'
            '       than the 11 μm background)',
            transform=ax.transAxes, fontsize=18, color='#333',
            bbox=dict(boxstyle='round', facecolor='#FFF8E1', alpha=0.95),
            verticalalignment='center')

    # Right panel: what the dashed boxes mean
    ax = axes[1]
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 10.5)
    ax.axis('off')
    ax.set_title('What the Realtime Plots Show', fontsize=22, fontweight='bold')

    items = [
        ('Background', 'RdYlGn (NDVI) for daytime\ninferno (T4) for nighttime\n'
         '→ shows vegetation health or thermal scene', '#059669'),
        ('Cyan dots', 'Thermal fire pixels\n'
         'Grid cells where T4_max > threshold\nand T4 – T11 > 10 K', '#00CCCC'),
        ('Magenta dots', 'Veg-confirmed fire\n'
         'Fire + NDVI dropped ≥ 0.15\n→ vegetation was actually destroyed', '#CC00CC'),
        ('Yellow dashes', 'Fire zone bounding boxes\n'
         'Connected groups of fire pixels\nwith area labels (ha or m²)', '#DAA520'),
        ('Stats box', 'Sweep count, coverage %,\nfire pixel count, total area,\n'
         'and zone breakdown', '#555555'),
    ]

    for i, (label, desc, color) in enumerate(items):
        y = 9 - i * 1.9
        _box(ax, 0, y - 0.3, 3, 1.4, label, color=color, fontsize=18)
        ax.text(3.3, y + 0.4, desc, fontsize=18, va='center',
                color='#333', linespacing=1.3)

    ax.text(5, 0,
            'The dashed yellow boxes group fire pixels into\n'
            '"zones" — connected burn areas with estimated size.',
            ha='center', fontsize=20, color='#444',
            bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.9))

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig('plots/diagram_threshold_explanation.png', dpi=200, bbox_inches='tight')
    print('Saved plots/diagram_threshold_explanation.png')
    plt.close()


def main():
    plot_mlp_architecture()
    plot_tabpfn_architecture()
    plot_dataset_organization()
    plot_model_io()
    plot_model_comparison()
    plot_threshold_explanation()
    print('\nAll diagrams generated in plots/')


if __name__ == '__main__':
    main()
