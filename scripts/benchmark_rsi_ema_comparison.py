"""Compare composer-alloc-polars RSI/EMA with polars-talib.

This is an opt-in benchmark. Install ``polars-talib`` in an isolated environment,
install the composer-alloc-polars wheel being measured, then run this script from
outside the repository so imports resolve to those installed packages.
"""

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

import polars as pl
import polars_talib  # noqa: F401 -- registers the ``Expr.ta`` namespace

from composer_alloc_polars import ema_with_validity, rsi_with_validity

try:
    import resource
except ImportError:  # Windows
    resource = None


PERIOD = 14
INVALID_CASES = (
    "prefix_provenance",
    "poison_provenance",
    "poison_null",
    "poison_nan",
    "poison_inf",
)


def frame_for(size: int, case: str, implementation: str) -> pl.DataFrame:
    """Build deterministic inputs before timing expression collection."""
    rng = random.Random(1977)
    values = [100.0]
    values.extend(100.0 + rng.uniform(-2.0, 2.0) for _ in range(size - 1))
    valid = [True] * size
    prefix_end = max(1, size // 10)
    poison_at = max(PERIOD + 1, size // 10)

    if case == "prefix_provenance":
        if implementation == "composer_alloc_polars":
            valid[:prefix_end] = [False] * prefix_end
        else:
            values[:prefix_end] = [None] * prefix_end
    elif case == "poison_provenance":
        if implementation == "composer_alloc_polars":
            valid[poison_at] = False
        else:
            values[poison_at] = None
    elif case.startswith("poison_"):
        if implementation == "composer_alloc_polars":
            if case == "poison_null":
                values[poison_at] = None
            elif case == "poison_nan":
                values[poison_at] = float("nan")
            elif case == "poison_inf":
                values[poison_at] = float("inf")
            else:
                valid[poison_at] = False
        elif case == "poison_null":
            values[poison_at] = None
        elif case == "poison_nan":
            values[poison_at] = float("nan")
        elif case == "poison_inf":
            values[poison_at] = float("inf")

        if implementation == "composer_alloc_polars" and case in (
            "poison_null",
            "poison_nan",
            "poison_inf",
        ):
            # For null/NaN/Inf cases, provenance remains true; numeric validity
            # alone must trigger the absorbing invalid state.
            valid[poison_at] = True

    data = {"price": values}
    if implementation == "composer_alloc_polars":
        data["source_valid"] = valid
    return pl.DataFrame(data)


def expression(kind: str, implementation: str) -> pl.Expr:
    """Create an equivalent numeric output expression for each library."""
    if implementation == "composer_alloc_polars":
        function = rsi_with_validity if kind == "rsi" else ema_with_validity
        return function(
            pl.col("price"), pl.col("source_valid"), PERIOD
        ).struct.field("value")
    if kind == "rsi":
        return pl.col("price").ta.rsi(timeperiod=PERIOD)
    return pl.col("price").ta.ema(timeperiod=PERIOD)


def collect(frame: pl.DataFrame, kind: str, implementation: str, execution: str) -> pl.DataFrame:
    expr = expression(kind, implementation).alias(kind)
    if execution == "eager":
        return frame.select(expr)
    return frame.lazy().select(expr).collect()


def verify_behavior() -> None:
    """Check finite outputs before invalidity agree with the TALib formulas."""
    size = 512
    for case in ("all_valid", *INVALID_CASES):
        outputs = {}
        for implementation in ("composer_alloc_polars", "polars-talib"):
            frame = frame_for(size, case, implementation)
            outputs[implementation] = {
                kind: collect(frame, kind, implementation, "eager")[kind].to_list()
                for kind in ("ema", "rsi")
            }

        poison_at = max(PERIOD + 1, size // 10) if case.startswith("poison_") else size
        for kind in ("ema", "rsi"):
            new = outputs["composer_alloc_polars"][kind]
            old = outputs["polars-talib"][kind]
            for index in range(poison_at):
                left, right = new[index], old[index]
                if left is None or right is None:
                    assert left is None and (right is None or math.isnan(right)), (case, kind, index, left, right)
                elif math.isnan(right):
                    assert math.isnan(left), (case, kind, index, left, right)
                else:
                    assert math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10), (case, kind, index, left, right)


def peak_rss_bytes():
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1_000, 10_000, 100_000, 1_000_000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("rsi_ema_comparison.json"))
    args = parser.parse_args()
    minimum_size = PERIOD + 2
    if args.repeats < 2 or any(size < minimum_size for size in args.sizes):
        parser.error(f"use at least two repeats and sizes of at least {minimum_size}")

    verify_behavior()
    cases = ("all_valid", *INVALID_CASES)
    results = []
    for size in args.sizes:
        for case in cases:
            frames = {
                implementation: frame_for(size, case, implementation)
                for implementation in ("composer_alloc_polars", "polars-talib")
            }
            benchmarks = [
                (implementation, kind, execution)
                for implementation in ("composer_alloc_polars", "polars-talib")
                for kind in ("ema", "rsi")
                for execution in ("eager", "lazy")
            ]
            samples = {benchmark: [] for benchmark in benchmarks}
            for implementation, kind, execution in benchmarks:
                collect(frames[implementation], kind, implementation, execution)
            for repeat in range(args.repeats):
                order = benchmarks if repeat % 2 == 0 else list(reversed(benchmarks))
                for implementation, kind, execution in order:
                    start = time.perf_counter()
                    collect(frames[implementation], kind, implementation, execution)
                    samples[(implementation, kind, execution)].append(
                        (time.perf_counter() - start) * 1_000
                    )
            for (implementation, kind, execution), times in samples.items():
                results.append(
                    {
                        "rows": size,
                        "input_case": case,
                        "implementation": implementation,
                        "indicator": kind,
                        "execution": execution,
                        "median_ms": statistics.median(times),
                        "min_ms": min(times),
                        "max_ms": max(times),
                        "samples_ms": times,
                    }
                )
            print(f"completed {size:,} rows, {case}", flush=True)

    project_root = Path(__file__).resolve().parents[1]
    lock_path = project_root / "Cargo.lock"
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "polars": pl.__version__,
        "composer_alloc_polars": version("composer-alloc-polars"),
        "polars_talib": version("polars-talib"),
        "period": PERIOD,
        "cargo_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "peak_process_rss_bytes": peak_rss_bytes(),
        "note": (
            "Collection timings use one numeric output per expression. The new plugin's "
            "Struct is projected to its value field; inputs are built before timing. "
            "For prefix and provenance-invalid cases, the new plugin receives finite "
            "values with a false mask and polars-talib receives nulls. Numeric-invalid "
            "cases pass null/NaN/Inf to both APIs; the new plugin's provenance mask "
            "stays true for those rows."
        ),
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
