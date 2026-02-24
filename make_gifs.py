#!/usr/bin/env python3
"""Create animated GIFs from realtime fire detection plot frames.

Auto-discovers all detector prefixes (simple, ml, tabpfn_classification,
tabpfn_regression, etc.) from frame filenames.  Produces one GIF per
(detector, flight) combination and pairwise side-by-side comparison GIFs
and static PNGs for every detector pair that shares a flight.

Output goes to plots/gifs/.

Usage:
    python make_gifs.py                    # all detectors/flights
    python make_gifs.py --fps 2            # slower animation
    python make_gifs.py --flights 03 04    # only specific flights
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys

import imageio.v3 as iio
import numpy as np


# Flight metadata for labelling
FLIGHT_INFO = {
    '2480103': ('24-801-03', 'Pre-burn (daytime)'),
    '2480104': ('24-801-04', 'Active fire (night)'),
    '2480105': ('24-801-05', 'Overnight fire (night)'),
    '2480106': ('24-801-06', 'Smoldering (daytime)'),
}

# Display labels for each detector prefix
DETECTOR_LABELS = {
    'simple': 'Threshold',
    'ml': 'MLP',
    'tabpfn_classification': 'TabPFN-Cls',
    'tabpfn_regression': 'TabPFN-Reg',
}

FRAME_DIR = 'plots/realtime'
OUT_DIR = 'plots/gifs'


def discover_frames(frame_dir: str) -> dict[str, dict[str, list[str]]]:
    """Discover all detector prefixes and their flights from frame PNGs.

    Frame naming convention: {prefix}-{flight_id}-{frame_num:03d}.png
    Prefix may contain underscores but not hyphens.

    Returns:
        Dict mapping prefix -> {flight_id -> sorted list of PNG paths}.
    """
    all_pngs = sorted(glob.glob(os.path.join(frame_dir, '*.png')))
    detectors: dict[str, dict[str, list[str]]] = {}
    for path in all_pngs:
        basename = os.path.basename(path).replace('.png', '')
        parts = basename.split('-')
        if len(parts) >= 3:
            prefix = parts[0]
            flight_id = parts[1]
            detectors.setdefault(prefix, {}).setdefault(flight_id, []).append(path)
    for prefix in detectors:
        for fid in detectors[prefix]:
            detectors[prefix][fid].sort()
    return detectors


def make_gif(frames: list[str], out_path: str, fps: int = 3,
             loop: int = 0) -> None:
    """Create an animated GIF from a list of PNG frame paths."""
    raw_images = []
    for f in frames:
        img = iio.imread(f)
        raw_images.append(img)

    # Pad all frames to the size of the largest frame (white padding)
    max_h = max(img.shape[0] for img in raw_images)
    max_w = max(img.shape[1] for img in raw_images)
    channels = raw_images[0].shape[2] if raw_images[0].ndim == 3 else 1

    images = []
    for img in raw_images:
        h, w = img.shape[:2]
        if h == max_h and w == max_w:
            images.append(img)
        else:
            padded = np.full((max_h, max_w, channels), 255, dtype=np.uint8)
            padded[:h, :w] = img
            images.append(padded)

    duration_ms = 1000 // fps
    durations = [duration_ms] * len(images)
    if len(durations) > 1:
        durations[-1] = duration_ms * 4  # 4x hold on last frame

    iio.imwrite(out_path, images, duration=durations, loop=loop)


def _pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Pad image to target height with white rows at the bottom."""
    if img.shape[0] == target_h:
        return img
    channels = img.shape[2] if img.ndim == 3 else 1
    padded = np.full((target_h, img.shape[1], channels), 255, dtype=np.uint8)
    padded[:img.shape[0], :img.shape[1]] = img
    return padded


def make_side_by_side(left_frames: list[str], right_frames: list[str],
                      out_path: str, fps: int = 3) -> None:
    """Create a side-by-side comparison GIF for matching frames.

    Uses the minimum frame count between both detectors.
    """
    n = min(len(left_frames), len(right_frames))
    if n == 0:
        return

    images = []
    max_h = 0
    max_w = 0

    for i in range(n):
        left = iio.imread(left_frames[i])
        right = iio.imread(right_frames[i])

        h = max(left.shape[0], right.shape[0])
        channels = left.shape[2] if left.ndim == 3 else 1

        left = _pad_to_height(left, h)
        right = _pad_to_height(right, h)

        sep = np.full((h, 4, channels), 200, dtype=np.uint8)
        combined = np.concatenate([left, sep, right], axis=1)
        images.append(combined)
        max_h = max(max_h, combined.shape[0])
        max_w = max(max_w, combined.shape[1])

    # Pad all to consistent size
    final = []
    channels = images[0].shape[2] if images[0].ndim == 3 else 1
    for img in images:
        if img.shape[0] == max_h and img.shape[1] == max_w:
            final.append(img)
        else:
            padded = np.full((max_h, max_w, channels), 255, dtype=np.uint8)
            padded[:img.shape[0], :img.shape[1]] = img
            final.append(padded)

    duration_ms = 1000 // fps
    durations = [duration_ms] * len(final)
    if len(durations) > 1:
        durations[-1] = duration_ms * 4

    iio.imwrite(out_path, final, duration=durations, loop=0)


def make_compare_png(left_frames: list[str], right_frames: list[str],
                     out_path: str) -> None:
    """Create a static side-by-side comparison PNG from the last frames."""
    if not left_frames or not right_frames:
        return
    left = iio.imread(left_frames[-1])
    right = iio.imread(right_frames[-1])

    h = max(left.shape[0], right.shape[0])
    channels = left.shape[2] if left.ndim == 3 else 1

    left = _pad_to_height(left, h)
    right = _pad_to_height(right, h)
    sep = np.full((h, 4, channels), 200, dtype=np.uint8)
    combined = np.concatenate([left, sep, right], axis=1)
    iio.imwrite(out_path, combined)


def main():
    parser = argparse.ArgumentParser(
        description='Create animated GIFs from realtime fire detection frames.')
    parser.add_argument('--fps', type=int, default=3,
                        help='Frames per second (default: 3)')
    parser.add_argument('--flights', nargs='*', default=None,
                        help='Flight suffixes to include (e.g., 03 04). '
                             'Default: all available flights.')
    parser.add_argument('--frame-dir', default=FRAME_DIR,
                        help='Directory containing frame PNGs')
    parser.add_argument('--out-dir', default=OUT_DIR,
                        help='Output directory for GIFs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Auto-discover all detector prefixes and flights
    detectors = discover_frames(args.frame_dir)
    if not detectors:
        print(f'No frames found in {args.frame_dir}/')
        sys.exit(1)

    prefixes = sorted(detectors.keys())
    all_flight_ids = sorted(set(
        fid for det in detectors.values() for fid in det.keys()))

    # Filter flights if specified
    if args.flights:
        suffixes = [s.zfill(2) for s in args.flights]
        all_flight_ids = [fid for fid in all_flight_ids
                          if any(fid.endswith(s) for s in suffixes)]

    if not all_flight_ids:
        print('No matching flights found.')
        sys.exit(1)

    print(f'Creating GIFs at {args.fps} fps')
    print(f'Detectors found: {", ".join(prefixes)}')
    print(f'Output directory: {args.out_dir}/')
    print()

    for flight_id in all_flight_ids:
        info = FLIGHT_INFO.get(flight_id, (flight_id, 'Unknown'))
        label, condition = info
        cond_slug = condition.split()[0].lower()

        # --- Individual GIFs per detector ---
        for prefix in prefixes:
            if flight_id not in detectors.get(prefix, {}):
                continue
            frames = detectors[prefix][flight_id]
            det_label = DETECTOR_LABELS.get(prefix, prefix)
            out_path = os.path.join(
                args.out_dir, f'{prefix}_{flight_id}_{cond_slug}.gif')
            print(f'[{det_label}] Flight {label} ({condition})')
            print(f'  Frames: {len(frames)} \u2192 {out_path}')
            make_gif(frames, out_path, fps=args.fps)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f'  Size:   {size_mb:.1f} MB')

        # --- Pairwise comparison GIFs and static PNGs ---
        available = [p for p in prefixes
                     if flight_id in detectors.get(p, {})]
        for left_pfx, right_pfx in itertools.combinations(available, 2):
            left_label = DETECTOR_LABELS.get(left_pfx, left_pfx)
            right_label = DETECTOR_LABELS.get(right_pfx, right_pfx)
            left_frames = detectors[left_pfx][flight_id]
            right_frames = detectors[right_pfx][flight_id]

            # Comparison GIF
            gif_name = f'compare_{left_pfx}_vs_{right_pfx}_{flight_id}_{cond_slug}.gif'
            out_path = os.path.join(args.out_dir, gif_name)
            n = min(len(left_frames), len(right_frames))
            print(f'[{left_label} vs {right_label}] Flight {label} ({condition})')
            print(f'  Frames: {n} (side-by-side) \u2192 {out_path}')
            make_side_by_side(left_frames, right_frames, out_path,
                              fps=args.fps)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f'  Size:   {size_mb:.1f} MB')

            # Static comparison PNG (last frame)
            png_name = f'compare_{left_pfx}_vs_{right_pfx}_{flight_id}.png'
            png_path = os.path.join('plots', png_name)
            print(f'[Static] {png_path}')
            make_compare_png(left_frames, right_frames, png_path)

        print()

    print('Done.')


if __name__ == '__main__':
    main()
