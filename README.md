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
