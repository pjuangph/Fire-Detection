"""realtime_tabpfn_classification.py - Real-time fire detection with TabPFN classifier.

Thin wrapper around lib.realtime that loads a TabPFN classification model from
a YAML config.

Usage:
    python realtime_tabpfn_classification.py --config configs/best_model.yaml
"""

from __future__ import annotations

import argparse
import sys

import yaml

from lib import group_files_by_flight, init_grid_state, load_fire_model
from lib.realtime import simulate_flight


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Real-time fire detection simulation (TabPFN classification)')
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to model config YAML (e.g. configs/best_model.yaml)')
    args = parser.parse_args()

    with open(args.config) as f:
        model_cfg = yaml.safe_load(f)
    model_path = model_cfg['checkpoint']
    threshold = model_cfg.get('threshold')
    model_type = model_cfg.get('model_type', 'tabpfn')
    ml_model = load_fire_model(model_path, threshold=threshold,
                               model_type=model_type)
    if ml_model is None:
        print(f'ERROR: checkpoint not found: {model_path}',
              file=sys.stderr)
        sys.exit(1)
    print(f'Using TabPFN fire detector ({args.config})')

    flights = group_files_by_flight()
    gs = init_grid_state()

    print(f'\nScanned {len(flights)} flights:')
    for fnum, info in sorted(flights.items()):
        print(f'  {fnum}: {len(info["files"])} lines \u2014 {info["comment"]}')
    print()
    for fnum, info in sorted(flights.items()):
        simulate_flight(fnum, info['files'], info['comment'], gs,
                        ml_model=ml_model, detector_name='ml')

    print('All simulations complete.')


if __name__ == '__main__':
    main()
