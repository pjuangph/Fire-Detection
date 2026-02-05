"""download_data.py - Download MASTER FireSense 2023 data from NASA Earthdata.

Downloads the MASTER Level 1B HDF4 files used in this project from the
ORNL DAAC archive. Requires a free NASA Earthdata account.

Dataset: MASTER: FireSense, western US, October 2023
DOI:     https://doi.org/10.3334/ORNLDAAC/2330

First-time setup:
    1. Create a free account at https://urs.earthdata.nasa.gov/
    2. pip install earthaccess
    3. python download_data.py

The script will prompt for your Earthdata credentials on first run
and cache them in ~/.netrc for future use.

Usage:
    python download_data.py                # download all 4 flights (83 files, ~9 GB)
    python download_data.py --flight 04    # download only flight 24-801-04
    python download_data.py --list         # list available files without downloading
"""

import os
import sys
import argparse
import earthaccess

# ORNL DAAC dataset DOI for MASTER FireSense 2023
DATASET_DOI = "10.3334/ORNLDAAC/2330"

# Flights used in this project (Kaibab Plateau, Arizona)
FLIGHTS = {
    "03": {"id": "2480103", "description": "Pre-burn (day), 9 lines",  "files": 9},
    "04": {"id": "2480104", "description": "Burn flight 1 (day), 40 lines", "files": 40},
    "05": {"id": "2480105", "description": "Burn flight 2 (night), 16 lines", "files": 16},
    "06": {"id": "2480106", "description": "Burn flight 3 (day), 18 lines", "files": 18},
}

OUTPUT_DIR = "ignite_fire_data"


def search_flight_files(flight_id):
    """Search for MASTER L1B HDF files for a specific flight."""
    results = earthaccess.search_data(
        doi=DATASET_DOI,
        granule_name=f"MASTERL1B_{flight_id}_*",
    )
    return results


def list_files(flights_to_list):
    """List available files for the specified flights."""
    total = 0
    for fnum, info in sorted(flights_to_list.items()):
        print(f"\nFlight 24-801-{fnum} ({info['description']}):")
        results = search_flight_files(info["id"])
        for r in results:
            name = r["meta"]["native-id"] if "meta" in r else str(r)
            print(f"  {name}")
        print(f"  Found: {len(results)} files")
        total += len(results)
    print(f"\nTotal: {total} files")


def download_files(flights_to_download):
    """Download HDF files for the specified flights."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_files = 0
    for fnum, info in sorted(flights_to_download.items()):
        print(f"\n{'=' * 60}")
        print(f"Flight 24-801-{fnum}: {info['description']}")
        print(f"{'=' * 60}")

        results = search_flight_files(info["id"])
        if not results:
            print(f"  No files found for flight {fnum}. Skipping.")
            continue

        print(f"  Found {len(results)} files. Downloading...")
        downloaded = earthaccess.download(results, OUTPUT_DIR)
        print(f"  Downloaded {len(downloaded)} files to {OUTPUT_DIR}/")
        total_files += len(downloaded)

    print(f"\n{'=' * 60}")
    print(f"Done. {total_files} files downloaded to {OUTPUT_DIR}/")
    print(f"\nNext steps:")
    print(f"  python mosaic_flight.py          # build mosaics")
    print(f"  python realtime_fire.py          # real-time simulation")
    print(f"  python plot_burn_locations.py     # burn analysis plots")


def main():
    parser = argparse.ArgumentParser(
        description="Download MASTER FireSense 2023 data from NASA Earthdata.",
        epilog="Requires a free NASA Earthdata account: https://urs.earthdata.nasa.gov/",
    )
    parser.add_argument(
        "--flight",
        choices=["03", "04", "05", "06"],
        help="Download only this flight (default: all 4 flights)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available files without downloading",
    )
    args = parser.parse_args()

    # Authenticate (prompts on first run, caches in ~/.netrc)
    print("Authenticating with NASA Earthdata...")
    earthaccess.login(strategy="interactive", persist=True)
    print("Authenticated.\n")

    # Select flights
    if args.flight:
        flights = {args.flight: FLIGHTS[args.flight]}
    else:
        flights = FLIGHTS

    print("MASTER FireSense 2023 - Kaibab Plateau Prescribed Burns")
    print(f"Dataset DOI: https://doi.org/{DATASET_DOI}")
    print(f"Flights: {', '.join(f'24-801-{f}' for f in sorted(flights))}")

    if args.list:
        list_files(flights)
    else:
        download_files(flights)


if __name__ == "__main__":
    main()
