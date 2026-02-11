"""realtime_mlp.py - Real-time fire detection simulation with MLP model.

Thin wrapper around lib.realtime that loads an MLP model from a YAML config.

Usage:
    python realtime_mlp.py                                   # threshold detector
    python realtime_mlp.py --config configs/best_model.yaml  # MLP from config
"""

from __future__ import annotations

import argparse
import sys

import yaml

from lib import group_files_by_flight, init_grid_state, load_fire_model
from lib.realtime import simulate_flight


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Real-time fire detection simulation (MLP)')
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to model config YAML (e.g. configs/best_model.yaml)')
    args = parser.parse_args()

    ml_model = None
    detector = 'simple'
    if args.config:
        with open(args.config) as f:
            model_cfg = yaml.safe_load(f)
        model_path = model_cfg['checkpoint']
        threshold = model_cfg.get('threshold')
        model_type = model_cfg.get('model_type', 'firemlp')
        ml_model = load_fire_model(model_path, threshold=threshold,
                                   model_type=model_type)
        if ml_model is None:
            print(f'ERROR: checkpoint not found: {model_path}',
                  file=sys.stderr)
            sys.exit(1)
        detector = 'ml'
        print(f'Using MLP fire detector ({args.config})')
    else:
        print('Using threshold fire detector (simple)')

    flights = group_files_by_flight()
    gs = init_grid_state()

    print(f'\nScanned {len(flights)} flights:')
    for fnum, info in sorted(flights.items()):
        print(f'  {fnum}: {len(info["files"])} lines \u2014 {info["comment"]}')
    print()
    for fnum, info in sorted(flights.items()):
        simulate_flight(fnum, info['files'], info['comment'], gs,
                        ml_model=ml_model, detector_name=detector)

    print('All simulations complete.')


if __name__ == '__main__':
    main()
