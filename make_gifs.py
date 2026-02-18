#!/usr/bin/env python3
"""Create animated GIFs from realtime fire detection plot frames.

Groups frames by detector and flight, produces one GIF per combination.
Also creates side-by-side comparison GIFs (simple vs ML) per flight.
Output goes to plots/gifs/.

Usage:
    python make_gifs.py                    # all detectors/flights
    python make_gifs.py --fps 2            # slower animation
    python make_gifs.py --flights 03 04    # only specific flights
"""

from __future__ import annotations

import argparse
import glob
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

FRAME_DIR = 'plots/realtime'
OUT_DIR = 'plots/gifs'


def find_flights(frame_dir: str, prefix: str) -> dict[str, list[str]]:
    """Discover available flights for a given detector prefix.

    Args:
        frame_dir: Directory containing frame PNGs.
        prefix: Detector prefix, e.g. 'ml' or 'simple'.

    Returns:
        Dict mapping flight ID (e.g. '2480103') to sorted list of PNG paths.
    """
    patterns = glob.glob(os.path.join(frame_dir, f'{prefix}-*.png'))
    flights: dict[str, list[str]] = {}
    for path in sorted(patterns):
        basename = os.path.basename(path)
        # ml-2480103-001.png -> 2480103
        parts = basename.replace('.png', '').split('-')
        if len(parts) >= 3:
            flight_id = parts[1]
            flights.setdefault(flight_id, []).append(path)
    for fid in flights:
        flights[fid].sort()
    return flights


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


def make_side_by_side(simple_frames: list[str], ml_frames: list[str],
                      out_path: str, fps: int = 3) -> None:
    """Create a side-by-side comparison GIF (simple | ML) for matching frames.

    Uses the minimum frame count between both detectors.
    """
    n = min(len(simple_frames), len(ml_frames))
    if n == 0:
        return

    images = []
    max_h = 0
    max_w = 0

    for i in range(n):
        left = iio.imread(simple_frames[i])
        right = iio.imread(ml_frames[i])

        # Ensure same height
        h = max(left.shape[0], right.shape[0])
        channels = left.shape[2] if left.ndim == 3 else 1

        def pad_to_h(img, target_h):
            if img.shape[0] == target_h:
                return img
            padded = np.full((target_h, img.shape[1], channels), 255, dtype=np.uint8)
            padded[:img.shape[0], :img.shape[1]] = img
            return padded

        left = pad_to_h(left, h)
        right = pad_to_h(right, h)

        # Add a thin separator
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

    # Find both detector types
    ml_flights = find_flights(args.frame_dir, 'ml')
    simple_flights = find_flights(args.frame_dir, 'simple')

    all_flight_ids = sorted(set(ml_flights.keys()) | set(simple_flights.keys()))
    if not all_flight_ids:
        print(f'No frames found in {args.frame_dir}/')
        sys.exit(1)

    # Filter flights if specified
    if args.flights:
        suffixes = [s.zfill(2) for s in args.flights]
        all_flight_ids = [fid for fid in all_flight_ids
                          if any(fid.endswith(s) for s in suffixes)]

    if not all_flight_ids:
        print('No matching flights found.')
        sys.exit(1)

    print(f'Creating GIFs at {args.fps} fps')
    print(f'Output directory: {args.out_dir}/')
    print()

    for flight_id in all_flight_ids:
        info = FLIGHT_INFO.get(flight_id, (flight_id, 'Unknown'))
        label, condition = info
        cond_slug = condition.split()[0].lower()

        # ML GIF
        if flight_id in ml_flights:
            frames = ml_flights[flight_id]
            out_path = os.path.join(args.out_dir, f'ml_{flight_id}_{cond_slug}.gif')
            print(f'[ML] Flight {label} ({condition})')
            print(f'  Frames: {len(frames)} → {out_path}')
            make_gif(frames, out_path, fps=args.fps)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f'  Size:   {size_mb:.1f} MB')

        # Simple threshold GIF
        if flight_id in simple_flights:
            frames = simple_flights[flight_id]
            out_path = os.path.join(args.out_dir, f'simple_{flight_id}_{cond_slug}.gif')
            print(f'[Simple] Flight {label} ({condition})')
            print(f'  Frames: {len(frames)} → {out_path}')
            make_gif(frames, out_path, fps=args.fps)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f'  Size:   {size_mb:.1f} MB')

        # Side-by-side comparison GIF
        if flight_id in ml_flights and flight_id in simple_flights:
            out_path = os.path.join(args.out_dir, f'compare_{flight_id}_{cond_slug}.gif')
            print(f'[Compare] Flight {label} ({condition})')
            n = min(len(simple_flights[flight_id]), len(ml_flights[flight_id]))
            print(f'  Frames: {n} (side-by-side) → {out_path}')
            make_side_by_side(
                simple_flights[flight_id], ml_flights[flight_id],
                out_path, fps=args.fps)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f'  Size:   {size_mb:.1f} MB')

        print()

    print('Done.')


if __name__ == '__main__':
    main()
