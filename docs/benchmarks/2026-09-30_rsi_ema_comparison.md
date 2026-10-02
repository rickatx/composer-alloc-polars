# RSI/EMA comparison with polars-talib

This local benchmark compares the `composer-alloc-polars` 0.2.0 wheel with
`polars-talib` 0.2.0 at period 14. Both libraries produce numeric outputs from
the same deterministic price series. The new plugin's `Struct` result is
projected to its `value` field so the timed results are numeric outputs on both
sides.

## Results

All values below are median milliseconds from five timed collections, eager
execution, one expression, after one warm run. Input frames are built outside
the timed section.

All-valid input scaling:

| Rows | EMA new | EMA polars-talib | RSI new | RSI polars-talib |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.476 | 0.154 | 0.485 | 0.114 |
| 1,000 | 0.453 | 0.148 | 0.435 | 0.110 |
| 10,000 | 0.634 | 0.168 | 0.573 | 0.140 |
| 100,000 | 2.469 | 0.483 | 2.805 | 0.501 |
| 1,000,000 | 20.075 | 11.794 | 24.361 | 12.989 |

At one million rows, medians by input validity profile:

| Profile | EMA new | EMA polars-talib | RSI new | RSI polars-talib |
| --- | ---: | ---: | ---: | ---: |
| All valid | 20.075 | 11.794 | 24.361 | 12.989 |
| 10% invalid prefix | 20.405 | 22.126 | 23.393 | 21.911 |
| False provenance at 10% | 17.395 | 23.476 | 17.676 | 23.804 |
| Null at 10% | 18.229 | 22.220 | 18.551 | 22.298 |
| NaN at 10% | 17.173 | 11.897 | 17.670 | 12.643 |
| Infinity at 10% | 17.622 | 12.355 | 17.822 | 12.506 |

These results suggest approximately linear scaling for both implementations at
large sizes. For fully valid inputs, polars-talib is faster in this run. Null
inputs notably increase its runtime; NaN and infinity do not show the same
increase. The new plugin stays in a narrower range across these input profiles.
Small-input timings are dominated by plugin and collection overhead. Results
are specific to this machine and environment; they are not performance
thresholds.

## Method

- Environment: Linux 6.6.87.2 WSL2 x86_64, CPython 3.13.9, Polars 1.44.2.
- Packages: `composer-alloc-polars` 0.2.0 wheel and
  [`polars-talib`](https://pypi.org/project/polars-talib/) 0.2.0.
- Period: 14. Sizes: 100, 1,000, 10,000, 100,000, and 1,000,000 rows.
- For every size and profile, measured EMA and RSI in eager and lazy execution;
  the table reports eager timings. Each expression was warmed once, then timed
  five times with implementation order alternated. Frames were constructed
  before timing.
- Data profiles: all valid; 10% leading false provenance (mapped to leading
  nulls for polars-talib); one interior false provenance row near 10% (mapped
  to null for polars-talib); or a null, NaN, or infinity near 10% with
  provenance true for the new plugin. In the 100-row case, the poison row is
  moved to zero-based index 15 (the 16th row) so it follows the period-14
  warm-up.
- A 512-row preflight compared the finite outputs to within `1e-10` before any
  interior invalid row. This confirms the formulas on the tested valid portions,
  not equivalence of invalid-output representations.

The APIs do not express identical validity semantics: polars-talib has no
provenance-valid argument and returns numeric values (including NaN/Inf), while
the new plugin returns a value plus a validity field and marks every row after
an invalid interior row invalid. The benchmark maps false provenance to a null
for polars-talib as a practical timing analogue. For infinity, the libraries
can also differ at the invalid row itself. Interpret invalid-profile timings as
input sensitivity measurements, not a claim that the APIs are interchangeable.

## Reproduction

The benchmark script is [benchmark_rsi_ema_comparison.py](../../scripts/benchmark_rsi_ema_comparison.py).
It requires `polars-talib==0.2.0`, Polars 1.44.2, and an installed
`composer-alloc-polars` 0.2.0 wheel. Run it from outside the repository so the
installed wheel is used instead of the source checkout:

```sh
python /path/to/composer-alloc-polars/scripts/benchmark_rsi_ema_comparison.py \
  --repeats 5 \
  --output /path/to/composer-alloc-polars/docs/benchmarks/2026-09-30_rsi_ema_comparison.json
```

The [raw JSON results](2026-09-30_rsi_ema_comparison.json) include all sizes,
input profiles, eager/lazy timings, individual samples, package versions,
platform details, and Cargo lockfile hash. Peak RSS covers the entire benchmark
process, including the prebuilt input frames.
