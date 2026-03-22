from __future__ import annotations

import argparse

try:
    from .datasets import save_dataset_bundle_from_config as save_daily_dataset_bundle
    from .datasets_weekly import save_dataset_bundle_from_config as save_weekly_dataset_bundle
    from .fetch import fetch_and_save_data
    from .preprocess import preprocess_and_save_data
except ImportError:
    from datasets import save_dataset_bundle_from_config as save_daily_dataset_bundle
    from datasets_weekly import save_dataset_bundle_from_config as save_weekly_dataset_bundle
    from fetch import fetch_and_save_data
    from preprocess import preprocess_and_save_data


def parse_args():
    parser = argparse.ArgumentParser(description="Run the market data pipeline.")
    parser.add_argument(
        "--mode",
        choices=["daily", "weekly", "both"],
        default="both",
        help="Select which dataset artifacts to generate after fetch and preprocess.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n=== Fetching raw market data ===")
    fetch_and_save_data()

    print("\n=== Preprocessing raw market data ===")
    preprocess_and_save_data()

    if args.mode in {"daily", "both"}:
        print("\n=== Building daily datasets ===")
        save_daily_dataset_bundle()

    if args.mode in {"weekly", "both"}:
        print("\n=== Building weekly datasets ===")
        save_weekly_dataset_bundle()

    print("\nFULL DATA PIPELINE COMPLETE\n")


if __name__ == "__main__":
    main()
