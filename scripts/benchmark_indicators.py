"""Opt-in RSI/EMA benchmark; run against an installed local wheel.

Example: python scripts/benchmark_indicators.py --repeats 5 --output benchmark.json
"""

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

import polars as pl

from composer_alloc_polars import ema_with_validity, rsi_with_validity

try:
    import resource
except ImportError:  # Windows
    resource = None


def frame_for(size: int, validity_case: str) -> pl.DataFrame:
    rng = random.Random(1977)
    values = [100.0]
    values.extend(100.0 + rng.uniform(-2.0, 2.0) for _ in range(size - 1))
    validity = [True] * size
    if validity_case == "prefix":
        validity[: max(1, size // 10)] = [False] * max(1, size // 10)
    elif validity_case == "poison":
        validity[max(1, size // 10)] = False
    return pl.DataFrame({"price": values, "source_valid": validity})


def collect(frame: pl.DataFrame, kind: str, mode: str, count: str) -> None:
    function = rsi_with_validity if kind == "rsi" else ema_with_validity
    amount = 1 if count == "one" else 4
    expressions = [
        function(
            pl.col("price") if index == 0 else pl.col("price") + index * 1e-9,
            pl.col("source_valid"),
            14,
        ).alias(f"{kind}_{index}")
        for index in range(amount)
    ]
    if mode == "eager":
        frame.select(expressions)
    else:
        frame.lazy().select(expressions).collect()


def peak_rss_bytes():
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1_000, 10_000, 100_000, 1_000_000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmark_indicators.json"))
    args = parser.parse_args()
    if args.repeats < 2 or any(size < 15 for size in args.sizes):
        parser.error("use at least two repeats and sizes of at least 15")

    results = []
    for size in args.sizes:
        for validity_case in ["all_valid", "prefix", "poison"]:
            frame = frame_for(size, validity_case)
            cases = [(kind, mode, count) for kind in ["rsi", "ema"] for mode in ["eager", "lazy"] for count in ["one", "many"]]
            timings = {case: [] for case in cases}
            for case in cases:
                collect(frame, *case)  # warm run
            for repeat in range(args.repeats):
                for case in cases if repeat % 2 == 0 else reversed(cases):
                    start = time.perf_counter()
                    collect(frame, *case)
                    timings[case].append((time.perf_counter() - start) * 1_000)
            for (kind, mode, count), samples in timings.items():
                results.append({
                    "rows": size,
                    "validity": validity_case,
                    "indicator": kind,
                    "execution": mode,
                    "expressions": count,
                    "median_ms": statistics.median(samples),
                    "min_ms": min(samples),
                    "max_ms": max(samples),
                    "samples_ms": samples,
                })
            print(f"completed {size:,} rows, {validity_case}", flush=True)

    lock_path = Path(__file__).resolve().parents[1] / "Cargo.lock"
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "polars": pl.__version__,
        "composer_alloc_polars": version("composer-alloc-polars"),
        "cargo_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "peak_process_rss_bytes": peak_rss_bytes(),
        "note": "Peak RSS is for the full benchmark process, including its input frames.",
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
