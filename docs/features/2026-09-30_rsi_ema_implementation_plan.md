# `composer-alloc-polars` RSI/EMA implementation plan

Date: 2026-09-29  
Target project: `/home/rick/dev/myproj/composer-alloc-polars`  
Purpose: implementation handoff for owned, validity-aware RSI and EMA Polars plugin functions.

## Objective

Add single-pass Rust implementations of Wilder RSI and SMA-seeded EMA to
`composer-alloc-polars`. Each function must consume both numeric values and caller-supplied
provenance validity, and return both indicator values and indicator validity.

These functions will replace `polars-talib` in `composer-alloc`, but this project must not
implement `composer-alloc`'s allocation/NAV validity sidecar. Its contract is narrower:

```text
(numeric series, provenance-valid series, period)
    -> struct(value: Float64, valid: Boolean)
```

The implementation must never turn invalid required input into a usable numeric indicator.

## Why this is needed

Two independent failures were confirmed in `composer-alloc`:

1. Fast evaluation can construct a finite but synthetic aggregate-NAV prefix while an inner
   allocation is unavailable. The caller will mark those NAV rows provenance-invalid. RSI must
   ignore them when establishing its seed.
2. `polars-talib` 0.2.0 RSI can turn an interior null, NaN, `+inf`, or `-inf` into finite `0.0`.
   An owned implementation must preserve invalidity instead.

A legitimate flat input may also produce RSI `0.0`, so validity cannot be inferred from the
numeric score.

## Existing project structure

- Rust expressions: `src/expressions.rs`
- PyO3 module: `src/lib.rs`
- Python wrappers: `composer_alloc_polars/__init__.py`
- Build: maturin, `pyo3-polars` plugin expressions
- Existing functions: `filter_select_weights`, `rolling_max_drawdown`
- Existing release builds: CPython 3.13/3.14 wheels on Linux, macOS, and Windows
- No current automated test suite was found.
- `pyproject.toml` reports version `0.1.3`; `Cargo.toml` reports `0.1.0`. Synchronize versions as
  part of the release work.

Do not change existing max-drawdown or selector behavior in this task.

## Public Python API

Add these functions:

```python
def rsi_with_validity(
    values: pl.Expr,
    validity: pl.Expr,
    period: int,
) -> pl.Expr:
    """Return Struct(value=Float64, valid=Boolean) Wilder RSI results."""


def ema_with_validity(
    values: pl.Expr,
    validity: pl.Expr,
    period: int,
) -> pl.Expr:
    """Return Struct(value=Float64, valid=Boolean) SMA-seeded EMA results."""
```

The returned expression must have this stable schema:

```text
Struct({"value": Float64, "valid": Boolean})
```

Add and export capability flags consistent with the existing symbol check:

```python
HAS_RSI_WITH_VALIDITY
HAS_EMA_WITH_VALIDITY
```

Wrapper requirements:

- Reject booleans, non-integers, zero, and negative periods with `ValueError`.
- Cast `values` to `Float64` and `validity` to `Boolean` before registration.
- Pass the period as an `Int64` literal.
- Register both functions with `is_elementwise=False`; each recurrence depends on prior rows.
- Export functions and flags through `__all__`.

The Rust functions must repeat essential validation because plugin expressions can be invoked
without the Python wrapper.

## Input validity contract

For row `t`:

```text
effective_valid[t] =
    validity[t] is exactly true
    AND values[t] is non-null
    AND values[t] is not NaN
    AND values[t] is finite
```

Null provenance validity is false. A finite numeric value with false provenance is invalid; it
may be a synthetic aggregate NAV and must not affect the seed or recurrence.

Do not add a positive-value requirement. Source prices are positive under the caller's contract,
but aggregate sequences and the mathematical functions should be defined for any finite values.

Output invariant for every row:

```text
valid == true  => value is present and finite
valid == false => value is null
```

The Boolean `valid` field itself must never be null.

## Shared recursive state semantics

Both functions use the following state rule:

1. Ignore a contiguous invalid prefix before the first effective-valid input.
2. Once any valid input has been accepted, any later invalid input permanently poisons the
   remaining output, even if the indicator has not yet accumulated enough rows to emit its first
   value.
3. A poisoned sequence never implicitly restarts. A new evaluation frame is the reset boundary.

Examples:

```text
validity: F F T T T T ...  -> normal leading warm-up, later output may become valid
validity: F T T F T T ...  -> invalid from the interior F through the end
validity: T T F T T T ...  -> invalid from the interior F through the end
```

This distinction lets aggregate warm-up recover while preserving genuine interior failure.

## EMA algorithm

For period `n >= 1`:

1. Accumulate the first `n` effective-valid values.
2. On the `n`th value, seed EMA with their arithmetic mean and emit it as valid.
3. For every later valid value `x_t`, calculate:

```text
alpha = 2 / (n + 1)
EMA_t = alpha * x_t + (1 - alpha) * EMA_(t-1)
```

4. Follow the shared poisoning rule.

EMA period 1 is the identity for valid input.

Use running state only; do not allocate or scan a trailing window per row. Accumulate the seed in
`f64`, then retain only the preceding EMA.

## RSI algorithm

For period `n >= 1`:

1. Accept the first effective-valid value as the prior price; emit invalid output.
2. For each of the next `n` valid values, calculate `delta = current - prior`:

```text
gain = max(delta, 0)
loss = max(-delta, 0)
```

3. On the `(n + 1)`th valid price, seed average gain and loss with the arithmetic mean of those
   first `n` gains/losses and emit the first RSI.
4. For every later valid price, update with Wilder's recurrence:

```text
avg_gain = (previous_avg_gain * (n - 1) + gain) / n
avg_loss = (previous_avg_loss * (n - 1) + loss) / n
```

5. Calculate RSI as:

```text
if avg_gain + avg_loss == 0: RSI = 0
else: RSI = 100 * avg_gain / (avg_gain + avg_loss)
```

This also gives `100` when loss is zero and gain is positive, and `0` when gain is zero.

6. Follow the shared poisoning rule.

Support RSI period 1. Its first score occurs after two valid prices: an increase gives `100`, a
decrease gives `0`, and no change gives `0`. This aligns with `composer-alloc` discovery's existing
`n + 1` eligibility contract; it intentionally improves on `polars-talib` 0.2.0, which rejects
RSI-1.

## Rust implementation shape

Use one reusable internal representation for result rows and, where helpful, shared input-state
logic. Keep RSI and EMA recurrence state separate and explicit; avoid an overly abstract framework.

Recommended organization in `src/expressions.rs` or small focused modules:

- output schema helper for `Struct(value, valid)`;
- input length/type/period validation;
- effective-valid predicate;
- one O(rows) EMA kernel;
- one O(rows) RSI kernel;
- plugin expression wrappers that build a struct series.

Requirements:

- Equal lengths for value and validity inputs; otherwise return `PolarsError::ComputeError`.
- Period literal must be present, positive, and convertible to `usize` without overflow.
- Empty input returns an empty struct series with the declared schema.
- Period greater than available valid history returns all invalid/null rows, not an error.
- Work correctly with chunked Polars series.
- Complexity: O(rows) time and O(rows) output memory, with O(1) recurrence state. Do not allocate a
  per-row window or intermediate vector beyond the output fields.
- No runtime fallback to Python or another indicator library.

## Correctness tests

Add both Rust unit tests for the recurrence kernels and Python integration tests for the actual
Polars plugin API.

### Deterministic numeric cases

For both indicators, cover periods 1, 2, 3, 14, and 200 where input length permits:

- strictly rising;
- strictly falling;
- flat;
- mixed gains/losses;
- deterministic random walk;
- period equal to available seed length;
- period longer than the series.

Use a small independent reference implementation in tests and frozen expected vectors. Do not add
`polars-talib` as a runtime dependency or make test success depend solely on matching it.

Compatibility evidence from the research prototype, for fully valid random input:

- SMA-seeded EMA matched `polars-talib` 0.2.0 within approximately `4.3e-14` maximum absolute
  error for periods 2, 3, 14, and 200.
- Wilder RSI matched within approximately `7.8e-14`.

Use an explicit tight tolerance rather than requiring cross-platform bitwise identity.

### Validity cases

Test each of these separately for RSI and EMA:

- finite values with a false leading provenance prefix;
- leading numeric null, NaN, `+inf`, and `-inf` while provenance is true;
- finite synthetic leading values while provenance is false;
- interior false provenance after output has begun;
- interior false provenance after the first valid row but before the seed is complete;
- interior numeric null, NaN, `+inf`, and `-inf`;
- null validity treated as false;
- valid flat RSI producing `(0.0, true)`;
- invalid input producing `(null, false)`, never `(0.0, true)`;
- later valid rows remaining invalid after poison;
- output invariant `valid == value.is_not_null()`, plus finiteness of every valid value.

For leading false provenance, assert that the valid output tail equals evaluating the contiguous
valid suffix by itself.

### API and Polars behavior

- Eager and lazy collection.
- Field extraction with `.struct.field("value")` and `.struct.field("valid")`.
- Aliasing, multiple indicator expressions in one frame, and chunked inputs.
- Stable output schema for empty and nonempty frames.
- Invalid Python periods and direct Rust-side invalid periods.
- Mismatched value/validity lengths at the Rust boundary.
- Capability flags and missing-symbol behavior.
- Existing max-drawdown and selector regression tests remain green.

## Performance work

Add a reproducible opt-in benchmark, not timing assertions in ordinary tests.

Benchmark RSI-14 and EMA-14 at 100, 1,000, 10,000, 100,000, and 1,000,000 rows, including:

- fully valid input;
- a leading invalid prefix;
- an early interior poison;
- eager and lazy collection;
- one and many indicator expressions in the same plan.

Report median and spread from alternating warm runs, peak memory, platform, and dependency lock.
The kernels should remain linear and should be close to the prior native implementation, while
materially outperforming the pure-Polars research prototype. Reference prototype medians at
10,000 rows were approximately 12.2 ms for RSI and 2.4 ms for EMA, versus approximately 0.15 ms
and 0.16 ms for `polars-talib`; these are comparison data, not hard CI thresholds.

## CI and release hardening

The project currently builds release wheels but has no committed test suite. Add:

1. A normal CI workflow for pull requests/pushes that runs Rust unit tests, builds/installs the
   extension with maturin, and runs Python tests.
2. At minimum, Linux Python 3.13 for fast PR validation; exercise the supported Python/platform
   matrix before release.
3. A pre-publish test gate in both PyPI workflows.
4. Post-build wheel smoke tests that execute small RSI and EMA cases and verify struct fields—not
   merely import the package.
5. Synchronized Cargo and Python package versions for the release.

Keep existing CPython 3.13/3.14 and Linux/macOS/Windows wheel coverage. Include Linux aarch64 as
currently configured.

## Documentation

Update the package README with:

- the two new Python signatures;
- the struct result and field extraction example;
- the definition of provenance validity;
- leading-prefix versus interior-poison behavior;
- RSI/EMA seed formulas and period-1 behavior; and
- a statement that invalid output has `value=null, valid=false`.

Do not describe validity as protection from “native code.” It is part of the owned financial-data
contract.

## `composer-alloc` integration contract

The downstream agent will handle integration separately. The plugin implementation must provide
these guarantees:

1. `composer-alloc` can pass finite synthetic aggregate NAV values with `validity=false`; those
   values do not influence the seed or recurrence.
2. The struct value and validity fields are produced by a single plugin evaluation.
3. `composer-alloc` can cache the struct expression once and extract both fields without invoking
   the recurrence twice.
4. A false output validity can be propagated through comparisons, branches, filters, allocations,
   and requested-output checks.
5. Direct valid price inputs reproduce the specified Wilder RSI and SMA-seeded EMA formulas.

`composer-alloc` remains responsible for:

- constructing provenance validity for conditions, allocations, portfolio returns, and NAV;
- passing that validity to the plugin;
- requiring all filter scores before ranking;
- preserving selected-branch validity; and
- rejecting invalid requested allocations.

Do not add `composer-alloc` as a dependency of this plugin project.

## Delivery sequence

1. Add failing Rust kernel tests and Python API/schema tests.
2. Implement shared input validation and struct-output construction.
3. Implement/test EMA.
4. Implement/test RSI.
5. Add capability flags and Python exports.
6. Add benchmark and record local results.
7. Add CI and release smoke tests.
8. Synchronize versions, update README, build wheels, and publish a TestPyPI candidate.
9. Hand the released version and benchmark/test evidence back to the `composer-alloc` agent.

## Completion criteria

- All numeric, validity, edge-case, Rust, and Python integration tests pass.
- Every invalid result is `(null, false)` and every valid result is finite.
- Synthetic finite rows with false provenance never affect RSI/EMA state.
- Leading invalidity recovers; interior invalidity is absorbing.
- RSI/EMA match the specified seeds and recurrences within the approved tolerance.
- Kernels are O(rows), use O(1) recurrence state, and have recorded benchmark evidence.
- Existing plugin functions are unchanged and tested.
- CI tests wheels on supported environments before publication.
- Cargo/Python versions agree, README is updated, and a TestPyPI wheel passes functional smoke
  tests.

# Implementation Notes

## Implementation summary

- Added validity-aware `ema_with_validity` and `rsi_with_validity` Rust plugin expressions in
  `src/expressions.rs`. Both kernels are single-pass, use constant recurrence state, and emit one
  `Struct(value: Float64, valid: Boolean)` series assembled from a nullable value field and a
  non-null validity field.
- Implemented the effective-valid predicate as provenance validity exactly equal to true plus a
  present, finite numeric value. A leading invalid prefix is ignored. After the first accepted
  input, invalid input or non-finite arithmetic permanently poisons the remaining frame.
- Implemented SMA-seeded EMA and Wilder-seeded RSI, including period-1 behavior and the specified
  flat-RSI value of `0.0` with `valid=true`.
- Added Python wrappers, capability flags, exports, period validation, input casts, and
  non-elementwise plugin registration in `composer_alloc_polars/__init__.py`. Public wrappers and
  private Rust/Python helpers are documented.
- Added Rust kernel/boundary tests and Python integration tests covering numerical formulas,
  validity behavior, warm-up boundaries, periods 1/2/3/14/200, eager and lazy execution, actual
  multi-chunk inputs, empty and nonempty schema, cached Struct field extraction, capability
  checks, malformed direct plugin calls, and regressions for the preexisting selector and
  max-drawdown expressions.
- Added normal CI, pre-publish test gates, functional wheel smoke tests, an opt-in benchmark and
  recorded results, and README API/contract documentation. Cargo and Python versions are
  synchronized at `0.2.0`.

## Tests and verification run

- The tests-first baseline behaved as expected: Rust tests initially failed to compile because
  the new result type, kernels, and plugin symbols were absent; the initial Python suite reported
  111 failures because the wrappers and compiled symbols did not yet exist.
- `cargo test --locked --offline`: 6 passed, 0 failed. This includes recurrence, validity,
  overflow, arity, input type, input length, and period validation at the Rust boundary.
- CPython 3.13 installed/extracted wheel integration suite: 130 passed, 0 failed. This includes
  the expanded direct-plugin boundary probes and assertions that chunked test inputs really have
  multiple chunks.
- CPython 3.14 Linux wheel: built successfully. The integration suite passed 120 tests and the
  functional smoke test before the final additions to documentation and boundary-test coverage;
  those later changes do not alter compiled indicator behavior.
- Functional wheel smoke tests passed for the locally built CPython 3.13 and CPython 3.14 Linux
  wheels, checking RSI/EMA values, Struct fields, eager/lazy collection, and absorbing poison.
- `cargo fmt --check`, Python bytecode compilation of the wrapper module, and
  `git diff --check` passed.
- Source comparison confirmed that the preexisting Rust selector/max-drawdown implementation is
  identical to `HEAD` after formatting and that the executable AST of the existing Python helper
  and wrapper functions is unchanged. Their regression tests pass.
- The opt-in benchmark completed at 100, 1,000, 10,000, 100,000, and 1,000,000 rows for fully
  valid, leading-prefix, and early-poison inputs; eager/lazy execution; and one/four indicator
  expressions. Full results are recorded in `docs/benchmarks/2026-09-30_rsi_ema.json` and
  summarized in `docs/benchmarks/2026-09-30_rsi_ema.md`.

## Deviations from the approved plan

- Finite inputs can overflow intermediate `f64` arithmetic. The implementation treats any such
  non-finite computed state as invalid and permanently poisons the remaining frame so the output
  invariant remains true. The original plan did not explicitly define this case.
- Polars' `dtype-struct` Cargo feature was enabled because constructing and declaring Struct
  output is unavailable without it.
- The PyO3 `extension-module` feature was moved from the direct Cargo dependency to maturin's
  build features. This preserves extension-wheel builds while allowing the Rust unit-test binary
  to link against Python normally.
- The synchronized release version was changed to `0.2.0` to follow the requested semantic
  versioning policy for this backward-compatible feature addition. The benchmark record below
  retains `0.1.4` because that was the version of the wheel used for the measured run.
- The local benchmark used three timed alternating runs per case. The benchmark script defaults
  to five runs for future measurements; the recorded run reports its median, range, platform,
  peak process RSS, dependency-lock hash, and raw samples.
- No TestPyPI candidate was published and no downstream `composer-alloc` integration or handoff
  was performed. Publishing requires repository credentials and an explicit release workflow
  run. The macOS, Windows, and Linux aarch64 wheel matrix was configured but not executed in this
  local environment.
