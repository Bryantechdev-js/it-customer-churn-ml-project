"""Measure local batch inference throughput for deployment sizing."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churn_prediction.inference import MAX_BATCH_ROWS, predict_batch
from churn_prediction.preprocessing import FEATURE_COLUMNS, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=int,
        default=MAX_BATCH_ROWS,
        help=f"Rows to score (maximum {MAX_BATCH_ROWS:,}).",
    )
    args = parser.parse_args()

    if not 1 <= args.rows <= MAX_BATCH_ROWS:
        parser.error(f"--rows must be between 1 and {MAX_BATCH_ROWS:,}.")

    source = load_dataset()[FEATURE_COLUMNS]
    sample = source.sample(
        n=args.rows,
        replace=args.rows > len(source),
        random_state=42,
    ).reset_index(drop=True)

    started = perf_counter()
    scored = predict_batch(sample)
    elapsed = perf_counter() - started

    if len(scored) != args.rows:
        raise RuntimeError("Inference returned an unexpected row count.")

    if not scored["churn_probability"].between(0, 1).all():
        raise RuntimeError("Inference returned an invalid probability.")

    print(f"Rows scored: {len(scored):,}")
    print(f"Elapsed: {elapsed:.3f} seconds")
    print(f"Throughput: {len(scored) / elapsed:,.0f} rows/second")
    print("This measures local inference only, not Streamlit Cloud latency.")


if __name__ == "__main__":
    main()
