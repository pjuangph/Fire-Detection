#!/usr/bin/env python3
"""Generate a starter PowerPoint presentation for Fire-Detection project."""

from __future__ import annotations

import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE


# ── Colour palette ──────────────────────────────────────────────────────
DARK_BG = RGBColor(0x1A, 0x1A, 0x2E)
ACCENT_RED = RGBColor(0xE0, 0x3C, 0x31)
ACCENT_ORANGE = RGBColor(0xF4, 0x8C, 0x06)
ACCENT_BLUE = RGBColor(0x3B, 0x82, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
MED_GRAY = RGBColor(0x99, 0x99, 0x99)
VERY_LIGHT = RGBColor(0xF0, 0xF0, 0xF0)
DARK_TEXT = RGBColor(0x22, 0x22, 0x22)


def _set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_textbox(slide, left, top, width, height, text, font_size=18,
                 color=WHITE, bold=False, alignment=PP_ALIGN.LEFT,
                 font_name='Calibri'):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def _add_bullet_frame(slide, left, top, width, height, items,
                      font_size=16, color=WHITE, font_name='Calibri',
                      spacing=Pt(6)):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = font_name
        p.level = 0
        p.space_after = spacing
    return txBox


def _add_title_bar(slide, title_text):
    """Dark accent bar at top with title text."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
        Inches(13.333), Inches(1.2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK_BG
    shape.line.fill.background()

    _add_textbox(slide, Inches(0.6), Inches(0.15), Inches(12), Inches(0.9),
                 title_text, font_size=32, color=WHITE, bold=True)


def _add_subtitle(slide, text, top=Inches(1.35)):
    _add_textbox(slide, Inches(0.6), top, Inches(12), Inches(0.5),
                 text, font_size=18, color=MED_GRAY, bold=False)


def _safe_add_image(slide, path, left, top, width=None, height=None):
    if os.path.isfile(path):
        kwargs = {'left': left, 'top': top}
        if width:
            kwargs['width'] = width
        if height:
            kwargs['height'] = height
        slide.shapes.add_picture(path, **kwargs)
        return True
    return False


def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ════════════════════════════════════════════════════════════════════
    # SLIDE 1 — Title
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    _set_slide_bg(slide, DARK_BG)

    _add_textbox(slide, Inches(1), Inches(1.5), Inches(11), Inches(1.2),
                 'Airborne Fire Detection with Machine Learning',
                 font_size=44, color=WHITE, bold=True,
                 alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(1), Inches(3.0), Inches(11), Inches(0.8),
                 'MASTER Hyperspectral Sensor  |  FireSense 2023 Campaign',
                 font_size=24, color=ACCENT_ORANGE, bold=False,
                 alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(1), Inches(4.2), Inches(11), Inches(0.6),
                 'Multi-Layer Perceptron & Tabular Prior-Fitted Networks (TabPFN)',
                 font_size=20, color=LIGHT_GRAY,
                 alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(1), Inches(5.8), Inches(11), Inches(0.5),
                 'Kaibab Plateau, Arizona  |  October 18-20, 2023',
                 font_size=16, color=MED_GRAY,
                 alignment=PP_ALIGN.CENTER)

    # ════════════════════════════════════════════════════════════════════
    # SLIDE 2 — Problem Statement
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'Problem Statement')

    _add_bullet_frame(slide, Inches(0.8), Inches(1.8), Inches(5.5), Inches(4.5), [
        'Detect active fire and burn scars from airborne '
        'hyperspectral imagery during prescribed burns',
        'Traditional threshold-based detectors suffer from false positives '
        '(sun glint, hot bare soil) and false negatives (smoldering edges)',
        'Need a robust, automated method that generalizes across '
        'day/night conditions and multiple flight passes',
        'Goal: reduce error rate (FP + FN) / P while maintaining '
        'near-perfect recall on true fire pixels',
    ], font_size=18, color=DARK_TEXT, spacing=Pt(12))

    # Right column — key numbers
    _add_textbox(slide, Inches(7.2), Inches(1.8), Inches(5), Inches(0.5),
                 'Key Challenges', font_size=22, color=ACCENT_RED, bold=True)
    _add_bullet_frame(slide, Inches(7.2), Inches(2.5), Inches(5), Inches(4), [
        'Extreme class imbalance: ~1.7% of pixels are fire',
        'Day/night radiometric differences',
        'SWIR saturation near flame front',
        'Pre-burn flight has zero true fire — all detections are FP',
    ], font_size=16, color=DARK_TEXT, spacing=Pt(10))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE 3 — MASTER Instrument
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'The MASTER Instrument')
    _add_subtitle(slide, 'MODIS/ASTER Airborne Simulator — 50 Spectral Channels')

    _add_bullet_frame(slide, Inches(0.8), Inches(2.0), Inches(5.8), Inches(4.5), [
        'Airborne whisk-broom scanner on NASA ER-2 / Twin Otter',
        '50 channels spanning 0.4 - 13 \u03bcm (VNIR, SWIR, MWIR, TIR)',
        'Spatial resolution: ~5-50 m depending on altitude',
        'High spectral resolution enables physics-based fire detection',
    ], font_size=18, color=DARK_TEXT, spacing=Pt(10))

    # Channel table
    _add_textbox(slide, Inches(7.0), Inches(2.0), Inches(5.5), Inches(0.5),
                 'Key Channels for Fire Detection',
                 font_size=20, color=ACCENT_BLUE, bold=True)
    rows = [
        ('Band', 'Channel', 'Wavelength', 'Use'),
        ('T4', 'Ch 31', '3.903 \u03bcm', 'Fire detection (MWIR peak)'),
        ('T11', 'Ch 48', '11.327 \u03bcm', 'Background temperature'),
        ('SWIR', 'Ch 22', '2.162 \u03bcm', 'Solar reflection / false alarm'),
        ('Red', 'Ch 5', '0.654 \u03bcm', 'NDVI (vegetation health)'),
        ('NIR', 'Ch 9', '0.866 \u03bcm', 'NDVI (vegetation health)'),
    ]
    tbl = slide.shapes.add_table(len(rows), 4,
                                 Inches(7.0), Inches(2.7),
                                 Inches(5.5), Inches(2.5)).table
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(14)
                p.font.name = 'Calibri'
                if ri == 0:
                    p.font.bold = True
                    p.font.color.rgb = WHITE
                else:
                    p.font.color.rgb = DARK_TEXT
            if ri == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = DARK_BG

    # ════════════════════════════════════════════════════════════════════
    # SLIDE 4 — Dataset & Flights
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'FireSense 2023 — Dataset & Flights')
    _add_subtitle(slide, 'Prescribed burn on the Kaibab Plateau, Arizona')

    flights = [
        ('Flight', 'Date', 'Files', 'Condition', 'Fire Present'),
        ('24-801-03', 'Oct 18', '14', 'Pre-burn (daytime)', 'No'),
        ('24-801-04', 'Oct 19', '22', 'Active fire (night)', 'Yes'),
        ('24-801-05', 'Oct 19', '12', 'Overnight (night)', 'Yes'),
        ('24-801-06', 'Oct 20', '13', 'Smoldering (day)', 'Yes'),
    ]
    tbl = slide.shapes.add_table(len(flights), 5,
                                 Inches(0.8), Inches(2.0),
                                 Inches(8.0), Inches(2.5)).table
    for ri, row in enumerate(flights):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(16)
                p.font.name = 'Calibri'
                if ri == 0:
                    p.font.bold = True
                    p.font.color.rgb = WHITE
                else:
                    p.font.color.rgb = DARK_TEXT
            if ri == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = DARK_BG
            elif ri % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = VERY_LIGHT

    _add_bullet_frame(slide, Inches(0.8), Inches(5.0), Inches(11), Inches(2), [
        '61 HDF files total across 4 flights, covering pre-burn baseline through post-burn smoldering',
        'Each HDF file = one scan sweep with 50 spectral channels + lat/lon geolocation',
    ], font_size=16, color=DARK_TEXT, spacing=Pt(8))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE 5 — Grid Resolution
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'Grid Resolution & Spatial Resampling')
    _add_subtitle(slide, 'Why we grid: from irregular scan pixels to a regular mosaic')

    _add_bullet_frame(slide, Inches(0.8), Inches(2.0), Inches(5.8), Inches(2.2), [
        'Native MASTER pixel spacing: ~8 m (whisk-broom scan geometry)',
        'Grid resolution: GRID_RES = 0.00025\u00b0 \u2248 28 m at 36\u00b0N',
        'This is a ~3\u00d7 downsampling from native resolution',
        'Each sweep has irregular pixel layout; gridding enables multi-pass stacking',
    ], font_size=18, color=DARK_TEXT, spacing=Pt(10))

    _add_textbox(slide, Inches(7.0), Inches(2.0), Inches(5.5), Inches(0.5),
                 'Why Grid Resolution Matters', font_size=20,
                 color=ACCENT_RED, bold=True)
    _add_bullet_frame(slide, Inches(7.0), Inches(2.7), Inches(5.5), Inches(2.0), [
        'Enables multi-pass consistency filter (same grid cell, multiple sweeps)',
        'Aggregates ~3 raw pixels per cell \u2192 noise reduction',
        'Running accumulators: T4_max, dT_max, obs_count per cell',
        'Trades spatial resolution for temporal depth and compute speed',
    ], font_size=16, color=DARK_TEXT, spacing=Pt(10))

    # Add grid resolution comparison image
    added = _safe_add_image(slide, 'plots/grid_resolution_comparison.png',
                            Inches(0.8), Inches(4.8), width=Inches(11.5))
    if not added:
        _add_textbox(slide, Inches(1), Inches(5.5), Inches(11), Inches(0.5),
                     '[Run plot_grid_resolution.py to generate comparison image]',
                     font_size=16, color=MED_GRAY, alignment=PP_ALIGN.CENTER)

    # ════════════════════════════════════════════════════════════════════
    # SLIDE 6 — Grid Resolution Zoom
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'Grid Resolution — Pixel-to-Cell Mapping')
    _add_subtitle(slide, 'How multiple raw pixels map into a single grid cell')

    # Add zoom image
    added = _safe_add_image(slide, 'plots/grid_resolution_zoom.png',
                            Inches(0.5), Inches(1.8), width=Inches(12))
    if not added:
        _add_textbox(slide, Inches(1), Inches(3.5), Inches(11), Inches(0.5),
                     '[Run plot_grid_resolution.py to generate zoom image]',
                     font_size=16, color=MED_GRAY, alignment=PP_ALIGN.CENTER)

    _add_bullet_frame(slide, Inches(0.8), Inches(5.8), Inches(11), Inches(1.5), [
        'Row index: (lat_max \u2212 lat) / GRID_RES  |  Col index: (lon \u2212 lon_min) / GRID_RES',
        'Multiple raw pixels landing in the same cell: last observation writes T4/T11, '
        'but running accumulators track max, sum, and count across all sweeps',
        'Single sweep: ~1 pixel/cell  |  After 40 sweeps (Flight 04): up to 40 observations/cell',
    ], font_size=16, color=DARK_TEXT, spacing=Pt(8))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE 7 — Fire Detection Physics
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'Fire Detection — Physics')
    _add_subtitle(slide, "Planck's Law & Brightness Temperature")

    _add_bullet_frame(slide, Inches(0.8), Inches(2.0), Inches(5.8), Inches(4.5), [
        'Fire at ~800 K is 23,000\u00d7 brighter than 300 K background at 3.9 \u03bcm (T4)',
        'At 11.3 \u03bcm (T11), fire is only ~6\u00d7 brighter — \u0394T = T4 - T11 is key discriminator',
        'Planck radiance: B(\u03bb,T) = (2hc\u00b2/\u03bb\u2075) / (exp(hc/\u03bbkT) - 1)',
        'Brightness temperature inverts Planck equation to get equivalent blackbody T',
    ], font_size=18, color=DARK_TEXT, spacing=Pt(12))

    _add_textbox(slide, Inches(7.2), Inches(2.0), Inches(5.5), Inches(0.5),
                 'Threshold Detection Rules', font_size=20,
                 color=ACCENT_RED, bold=True)
    _add_bullet_frame(slide, Inches(7.2), Inches(2.7), Inches(5.5), Inches(4.0), [
        'Simple: T4 > 325 K AND \u0394T > 10 K',
        'Contextual: pixel T4 > mean + 3\u03c3 of 61\u00d761 neighborhood',
        'SWIR rejection: if radiance in 2.2 \u03bcm band is high, '
        'likely solar reflection not fire',
        'Multi-pass consistency: require fire in \u22652 passes '
        '(52% FP reduction on pre-burn flight)',
    ], font_size=16, color=DARK_TEXT, spacing=Pt(10))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE — T4-T11 Threshold Explanation (diagram)
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _safe_add_image(slide, 'plots/diagram_threshold_explanation.png',
                    Inches(0), Inches(0), width=Inches(13.333))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE — Vegetation Loss Confirmation
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'Vegetation-Loss Fire Confirmation')
    _add_subtitle(slide, 'NDVI tracking for burn scar verification')

    _add_bullet_frame(slide, Inches(0.8), Inches(2.0), Inches(5.8), Inches(4.5), [
        'NDVI = (NIR - Red) / (NIR + Red) — measures vegetation health',
        'Baseline NDVI captured from first valid daytime pass (write-once per pixel)',
        'Fire confirmed when NDVI drops \u22650.15 from baseline at a thermal-fire pixel',
        'Nighttime fire with existing daytime baseline directly confirms vegetation loss',
        'Confirmed pixels override best-illuminated compositing rule '
        '— ensures burn scars remain visible',
    ], font_size=18, color=DARK_TEXT, spacing=Pt(12))

    _add_textbox(slide, Inches(7.2), Inches(2.0), Inches(5.5), Inches(0.5),
                 'Why This Matters', font_size=20, color=ACCENT_BLUE, bold=True)
    _add_bullet_frame(slide, Inches(7.2), Inches(2.7), Inches(5.5), Inches(3.5), [
        'Reduces false positives from transient heat sources',
        'Physical confirmation: fire should destroy vegetation',
        'Provides temporal evidence — links thermal anomaly to actual burn',
        'VEG_LOSS_THRESHOLD = 0.15 (configurable in lib/constants.py)',
    ], font_size=16, color=DARK_TEXT, spacing=Pt(10))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE — How the Data is Organized (diagram)
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _safe_add_image(slide, 'plots/diagram_dataset_organization.png',
                    Inches(0), Inches(0), width=Inches(13.333))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE — What Goes In, What Comes Out (diagram)
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _safe_add_image(slide, 'plots/diagram_model_io.png',
                    Inches(0), Inches(0), width=Inches(13.333))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE — MLP Architecture (diagram)
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _safe_add_image(slide, 'plots/diagram_mlp_architecture.png',
                    Inches(0), Inches(0), width=Inches(13.333))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE 9 — MLP Results
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'MLP Results — Best Model')
    _add_subtitle(slide, 'Run 37: SoftErrorRateLoss | [64, 32] | lr=0.01')

    # Results table
    metrics = [
        ('Metric', 'Value'),
        ('Error Rate', '0.0309  (= (FP + FN) / P)'),
        ('True Positives (TP)', '9,009'),
        ('False Positives (FP)', '221'),
        ('False Negatives (FN)', '59'),
        ('True Negatives (TN)', '519,558'),
        ('Precision', '0.9761'),
        ('Recall', '0.9935'),
        ('Architecture', '12 \u2192 64 \u2192 32 \u2192 1'),
    ]
    tbl = slide.shapes.add_table(len(metrics), 2,
                                 Inches(0.8), Inches(2.0),
                                 Inches(5.5), Inches(4.0)).table
    tbl.columns[0].width = Inches(2.8)
    tbl.columns[1].width = Inches(2.7)
    for ri, (k, v) in enumerate(metrics):
        for ci, val in enumerate((k, v)):
            cell = tbl.cell(ri, ci)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(16)
                p.font.name = 'Calibri'
                if ri == 0:
                    p.font.bold = True
                    p.font.color.rgb = WHITE
                else:
                    p.font.color.rgb = DARK_TEXT
                    if ci == 0:
                        p.font.bold = True
            if ri == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = DARK_BG
            elif ri % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = VERY_LIGHT

    # Add convergence plot if available
    _safe_add_image(slide, 'plots/convergence_mlp.png',
                    Inches(7.0), Inches(1.8), width=Inches(5.8))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE 10 — MLP Prediction Maps
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'MLP — Spatial Prediction Maps')
    _add_subtitle(slide, 'Flight 24-801-06 (daytime smoldering)')

    added = _safe_add_image(slide, 'plots/tune_prediction_map_2480106.png',
                            Inches(1.5), Inches(1.8), width=Inches(10))
    if not added:
        _add_textbox(slide, Inches(2), Inches(3.5), Inches(9), Inches(1),
                     '[Prediction map will be generated after training — '
                     'plots/tune_prediction_map_2480106.png]',
                     font_size=18, color=MED_GRAY,
                     alignment=PP_ALIGN.CENTER)

    # ════════════════════════════════════════════════════════════════════
    # SLIDE 11 — MLP Probability Distribution
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'MLP — Probability Calibration')
    _add_subtitle(slide, 'P(fire) distribution for fire vs non-fire pixels')

    added = _safe_add_image(slide, 'plots/tune_probability_hist.png',
                            Inches(3), Inches(1.8), width=Inches(7))
    if not added:
        _add_textbox(slide, Inches(2), Inches(3.5), Inches(9), Inches(1),
                     '[Probability histogram will be generated after training — '
                     'plots/tune_probability_hist.png]',
                     font_size=18, color=MED_GRAY,
                     alignment=PP_ALIGN.CENTER)

    # ════════════════════════════════════════════════════════════════════
    # SLIDE — TabPFN Architecture (diagram)
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _safe_add_image(slide, 'plots/diagram_tabpfn_architecture.png',
                    Inches(0), Inches(0), width=Inches(13.333))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE — TabPFN Results (placeholder with convergence if available)
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'TabPFN Results')
    _add_subtitle(slide, 'Classification & Regression — pending training completion')

    added_cls = _safe_add_image(slide, 'plots/convergence_tabpfn_classification.png',
                                Inches(0.3), Inches(2.0), width=Inches(6.2))
    added_reg = _safe_add_image(slide, 'plots/convergence_tabpfn_regression.png',
                                Inches(6.8), Inches(2.0), width=Inches(6.2))
    if not (added_cls or added_reg):
        _add_textbox(slide, Inches(1), Inches(3.5), Inches(11), Inches(1),
                     'TabPFN training has not been completed yet.\n'
                     'Run train-all-models.sh to generate results and convergence plots.',
                     font_size=18, color=MED_GRAY, alignment=PP_ALIGN.CENTER)

    # ════════════════════════════════════════════════════════════════════
    # SLIDE — Model Comparison (diagram)
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _safe_add_image(slide, 'plots/diagram_model_comparison.png',
                    Inches(0), Inches(0), width=Inches(13.333))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE 15 — Pipeline Architecture
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'Processing Pipeline')
    _add_subtitle(slide, 'From raw HDF to fire detection')

    steps = [
        ('1. Ingest', 'Read MASTER L1B\nHDF files\n(50 channels)'),
        ('2. Calibrate', 'Planck inversion\n\u2192 Brightness\nTemperature'),
        ('3. Grid', 'Mosaic sweeps\nonto lat/lon\ngrid'),
        ('4. Features', 'Compute 12\naggregate\nfeatures'),
        ('5. Detect', 'ML inference\nor threshold\nrules'),
        ('6. Validate', 'Multi-pass\n+ Veg-loss\nconfirmation'),
    ]
    box_w = Inches(1.8)
    box_h = Inches(2.0)
    start_x = Inches(0.5)
    y = Inches(2.5)
    gap = Inches(0.3)
    for i, (title, body) in enumerate(steps):
        x = start_x + i * (box_w + gap)
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, x, y, box_w, box_h)
        shape.fill.solid()
        if i == 4:  # Detect step highlighted
            shape.fill.fore_color.rgb = ACCENT_RED
            text_color = WHITE
        else:
            shape.fill.fore_color.rgb = DARK_BG
            text_color = WHITE
        shape.line.fill.background()

        tf = shape.text_frame
        tf.word_wrap = True
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = text_color
        p.font.name = 'Calibri'

        p2 = tf.add_paragraph()
        p2.text = body
        p2.font.size = Pt(12)
        p2.font.color.rgb = LIGHT_GRAY
        p2.font.name = 'Calibri'
        p2.alignment = PP_ALIGN.CENTER

        # Arrow between boxes
        if i < len(steps) - 1:
            arrow_x = x + box_w
            arrow_y = y + box_h / 2
            _add_textbox(slide, arrow_x, arrow_y - Inches(0.15),
                         gap, Inches(0.3), '\u2192',
                         font_size=24, color=ACCENT_ORANGE, bold=True,
                         alignment=PP_ALIGN.CENTER)

    _add_bullet_frame(slide, Inches(0.8), Inches(5.2), Inches(11), Inches(1.5), [
        'Real-time simulation mode: process sweeps sequentially with '
        'live matplotlib animation (realtime_fire.py)',
        'All shared logic in lib/ package \u2014 scripts are thin orchestration wrappers',
    ], font_size=16, color=DARK_TEXT, spacing=Pt(8))

    # ════════════════════════════════════════════════════════════════════
    # Real-Time Simulation — Pre-Burn (Daytime)
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'Real-Time Simulation — Pre-Burn Baseline')
    _add_subtitle(slide, 'Flight 24-801-03: Daytime, no fire (9 sweeps)')

    # Show first and last frame side by side
    added_first = _safe_add_image(slide, 'plots/realtime/ml-2480103-001.png',
                                  Inches(0.3), Inches(2.0), width=Inches(6.2))
    added_last = _safe_add_image(slide, 'plots/realtime/ml-2480103-009.png',
                                 Inches(6.8), Inches(2.0), width=Inches(6.2))
    if not (added_first or added_last):
        _add_textbox(slide, Inches(1), Inches(3.5), Inches(11), Inches(0.5),
                     '[Run realtime_mlp.py to generate frames]',
                     font_size=16, color=MED_GRAY, alignment=PP_ALIGN.CENTER)
    else:
        _add_textbox(slide, Inches(0.3), Inches(5.8), Inches(6.2), Inches(0.4),
                     'Sweep 1/9 — First sweep (NDVI baseline)',
                     font_size=14, color=MED_GRAY, alignment=PP_ALIGN.CENTER)
        _add_textbox(slide, Inches(6.8), Inches(5.8), Inches(6.2), Inches(0.4),
                     'Sweep 9/9 — Complete coverage, 0 fire pixels',
                     font_size=14, color=MED_GRAY, alignment=PP_ALIGN.CENTER)

    _add_bullet_frame(slide, Inches(0.8), Inches(6.3), Inches(11), Inches(1), [
        'Pre-burn flight establishes NDVI vegetation baseline and confirms zero fire detections',
        'GIF animation: plots/gifs/ml_2480103_pre-burn.gif',
    ], font_size=14, color=DARK_TEXT, spacing=Pt(4))

    # ════════════════════════════════════════════════════════════════════
    # Real-Time Simulation — Active Fire (Night)
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'Real-Time Simulation — Active Fire (Night)')
    _add_subtitle(slide, 'Flight 24-801-04: 40 sweeps, fire grows over time')

    added_first = _safe_add_image(slide, 'plots/realtime/ml-2480104-005.png',
                                  Inches(0.3), Inches(2.0), width=Inches(6.2))
    added_last = _safe_add_image(slide, 'plots/realtime/ml-2480104-040.png',
                                 Inches(6.8), Inches(2.0), width=Inches(6.2))
    if not (added_first or added_last):
        _add_textbox(slide, Inches(1), Inches(3.5), Inches(11), Inches(0.5),
                     '[Run realtime_mlp.py to generate frames]',
                     font_size=16, color=MED_GRAY, alignment=PP_ALIGN.CENTER)
    else:
        _add_textbox(slide, Inches(0.3), Inches(5.8), Inches(6.2), Inches(0.4),
                     'Sweep 5/40 — Early fire detection',
                     font_size=14, color=MED_GRAY, alignment=PP_ALIGN.CENTER)
        _add_textbox(slide, Inches(6.8), Inches(5.8), Inches(6.2), Inches(0.4),
                     'Sweep 40/40 — Full fire extent with zone labels',
                     font_size=14, color=MED_GRAY, alignment=PP_ALIGN.CENTER)

    _add_bullet_frame(slide, Inches(0.8), Inches(6.3), Inches(11), Inches(1), [
        'Night flight with modified 3.9\u03bcm preamp — fire zones detected and labelled in real time',
        'GIF animation: plots/gifs/ml_2480104_active.gif',
    ], font_size=14, color=DARK_TEXT, spacing=Pt(4))

    # ════════════════════════════════════════════════════════════════════
    # Real-Time Simulation — Smoldering (Daytime)
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'Real-Time Simulation — Daytime Smoldering')
    _add_subtitle(slide, 'Flight 24-801-06: Vegetation-confirmed burn scars (14 sweeps)')

    added_first = _safe_add_image(slide, 'plots/realtime/ml-2480106-001.png',
                                  Inches(0.3), Inches(2.0), width=Inches(6.2))
    added_last = _safe_add_image(slide, 'plots/realtime/ml-2480106-014.png',
                                 Inches(6.8), Inches(2.0), width=Inches(6.2))
    if not (added_first or added_last):
        _add_textbox(slide, Inches(1), Inches(3.5), Inches(11), Inches(0.5),
                     '[Run realtime_mlp.py to generate frames]',
                     font_size=16, color=MED_GRAY, alignment=PP_ALIGN.CENTER)
    else:
        _add_textbox(slide, Inches(0.3), Inches(5.8), Inches(6.2), Inches(0.4),
                     'Sweep 1/14 — Initial scan',
                     font_size=14, color=MED_GRAY, alignment=PP_ALIGN.CENTER)
        _add_textbox(slide, Inches(6.8), Inches(5.8), Inches(6.2), Inches(0.4),
                     'Sweep 14/14 — 7,483 fire px, 5,329 veg-confirmed, 463 ha',
                     font_size=14, color=MED_GRAY, alignment=PP_ALIGN.CENTER)

    _add_bullet_frame(slide, Inches(0.8), Inches(6.3), Inches(11), Inches(1), [
        'Daytime smoldering with NDVI vegetation-loss confirmation (magenta markers)',
        'GIF animation: plots/gifs/ml_2480106_smoldering.gif',
    ], font_size=14, color=DARK_TEXT, spacing=Pt(4))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE — Simple vs ML Comparison: Active Fire
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'Threshold vs MLP — Active Fire (Flight 04)')
    _add_subtitle(slide, 'Left: Simple threshold (T4>325K, dT>10K)  |  Right: MLP detector')

    added = _safe_add_image(slide, 'plots/compare_final_2480104.png',
                            Inches(0.2), Inches(1.8), width=Inches(12.8))
    if added:
        _add_bullet_frame(slide, Inches(0.8), Inches(6.0), Inches(11), Inches(1.2), [
            'Simple: 3,603 fire px (223 ha)  |  MLP: 4,057 fire px (251 ha)',
            'MLP detects more fire at edges and in smoldering regions — higher sensitivity',
            'Side-by-side GIF: plots/gifs/compare_2480104_active.gif',
        ], font_size=14, color=DARK_TEXT, spacing=Pt(4))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE — Simple vs ML Comparison: Smoldering
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'Threshold vs MLP — Smoldering (Flight 06)')
    _add_subtitle(slide, 'Left: Simple threshold  |  Right: MLP detector')

    added = _safe_add_image(slide, 'plots/compare_final_2480106.png',
                            Inches(0.2), Inches(1.8), width=Inches(12.8))
    if added:
        _add_bullet_frame(slide, Inches(0.8), Inches(6.0), Inches(11), Inches(1.2), [
            'Simple: 6,903 fire px (428 ha)  |  MLP: 7,483 fire px (464 ha)',
            'MLP captures additional smoldering pixels and low-intensity burn edges',
            'Side-by-side GIF: plots/gifs/compare_2480106_smoldering.gif',
        ], font_size=14, color=DARK_TEXT, spacing=Pt(4))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE — Simple vs ML Comparison: Pre-burn
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'Threshold vs MLP — Pre-burn Baseline (Flight 03)')
    _add_subtitle(slide, 'Left: Simple threshold  |  Right: MLP detector  |  No real fire present')

    added = _safe_add_image(slide, 'plots/compare_final_2480103.png',
                            Inches(0.2), Inches(1.8), width=Inches(12.8))
    if added:
        _add_bullet_frame(slide, Inches(0.8), Inches(6.0), Inches(11), Inches(1.2), [
            'Simple: 70 false fire px (4.3 ha)  |  MLP: 128 false fire px (7.9 ha)',
            'Both detectors show some false positives on pre-burn — hot bare soil and sun glint',
            'Multi-pass consistency filter reduces these by 52%',
        ], font_size=14, color=DARK_TEXT, spacing=Pt(4))

    # ════════════════════════════════════════════════════════════════════
    # Future Work
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, 'Future Work & Next Steps')

    _add_bullet_frame(slide, Inches(0.8), Inches(2.0), Inches(5.8), Inches(4.5), [
        'Complete TabPFN grid search and compare against MLP baseline',
        'Ensemble methods: combine MLP + TabPFN predictions',
        'Transfer learning: apply trained models to new fire campaigns',
        'Temporal features: leverage sequential pass information',
        'Operational deployment: integrate with real-time mosaic pipeline',
    ], font_size=18, color=DARK_TEXT, spacing=Pt(12))

    _add_textbox(slide, Inches(7.2), Inches(2.0), Inches(5.5), Inches(0.5),
                 'Technical Improvements', font_size=20,
                 color=ACCENT_BLUE, bold=True)
    _add_bullet_frame(slide, Inches(7.2), Inches(2.7), Inches(5.5), Inches(3.5), [
        'Explore deeper MLP architectures with dropout/batch norm',
        'Investigate TabPFN with larger n_estimators',
        'Add spatial context features (neighbor statistics)',
        'Cross-validation across flights for robustness',
    ], font_size=16, color=DARK_TEXT, spacing=Pt(10))

    # ════════════════════════════════════════════════════════════════════
    # SLIDE 17 — Summary / Thank You
    # ════════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, DARK_BG)

    _add_textbox(slide, Inches(1), Inches(1.5), Inches(11), Inches(1),
                 'Summary', font_size=40, color=WHITE, bold=True,
                 alignment=PP_ALIGN.CENTER)

    _add_bullet_frame(slide, Inches(2), Inches(3.0), Inches(9), Inches(3.5), [
        'Physics-based fire detection from 50-channel MASTER hyperspectral data',
        'Multi-pass consistency + vegetation-loss confirmation reduce false positives',
        'FireMLP achieves 0.031 error rate with 99.4% recall on test set',
        'TabPFN offers meta-learned alternative with minimal hyperparameter tuning',
        'Modular Python pipeline supports real-time simulation and batch processing',
    ], font_size=20, color=WHITE, spacing=Pt(14))

    _add_textbox(slide, Inches(1), Inches(6.5), Inches(11), Inches(0.5),
                 'github.com  |  FireSense 2023  |  MASTER L1B',
                 font_size=14, color=MED_GRAY,
                 alignment=PP_ALIGN.CENTER)

    # ── Save ──
    out_path = 'Fire_Detection_Presentation.pptx'
    prs.save(out_path)
    print(f'Saved presentation to {out_path}')
    print(f'  {len(prs.slides)} slides')


if __name__ == '__main__':
    create_presentation()
