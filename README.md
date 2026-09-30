# composer-alloc-polars

Polars-accelerated helpers used by `composer-alloc`.

## Install

From PyPI (when published):

```bash
pip install composer-alloc-polars
```

## Development

Building from source requires a Rust toolchain (rustup + cargo) and `maturin`.

Build and install in-place:

```bash
maturin develop
```

Build a wheel:

```bash
maturin build --release
```

Run the opt-in RSI/EMA benchmark against an installed wheel:

```bash
python scripts/benchmark_indicators.py --repeats 5 --output benchmark_indicators.json
```

### Local cibuildwheel check

For a CI-like wheel build, use the helper script. It creates a local `.venv`,
installs `cibuildwheel` + `maturin`, and builds wheels into `dist/`.

```bash
./scripts/local_cibw.sh
# Optional pin:
CIBW_VERSION=2.20.0 ./scripts/local_cibw.sh
```

## Usage

This package is imported by `composer-alloc`. Most users do not need to call it directly.

### Validity-aware indicators

```python
import polars as pl
from composer_alloc_polars import ema_with_validity, rsi_with_validity

prices = pl.DataFrame({"price": [100.0, 101.0, 102.0], "source_valid": [True] * 3})
rsi = rsi_with_validity(pl.col("price"), pl.col("source_valid"), period=2)
result = prices.with_columns(rsi.alias("indicator")).select(
    pl.col("indicator").struct.field("value").alias("rsi"),
    pl.col("indicator").struct.field("valid").alias("rsi_valid"),
)
```

`rsi_with_validity(values: pl.Expr, validity: pl.Expr, period: int)` and
`ema_with_validity(values: pl.Expr, validity: pl.Expr, period: int)` each return
`Struct(value: Float64, valid: Boolean)`. The caller supplies provenance validity:
`true` means the numeric input is an actual available value, rather than a finite
synthetic placeholder. Null, NaN, and infinite numeric inputs are also invalid.
An invalid output is always `(value=null, valid=false)`; a valid output is finite.

The functions ignore a contiguous invalid prefix, so calculation can begin at
the first valid input. Once a valid input has been accepted, any later invalid
row makes that row and all subsequent rows invalid within the evaluation frame.

EMA seeds with the mean of its first `period` valid values, then uses
`alpha = 2 / (period + 1)` and `EMA = alpha * value + (1 - alpha) * prior EMA`.
EMA period 1 is the input identity. Wilder RSI seeds from the mean gain and mean
loss of the first `period` price changes, so its first score follows
`period + 1` valid prices. Later averages use
`(prior_average * (period - 1) + current_gain_or_loss) / period`; RSI is `0`
when both averages are zero, otherwise `100 * avg_gain / (avg_gain + avg_loss)`.
RSI period 1 first emits after two valid prices: `100` for an increase and `0`
for a decrease or flat move.
