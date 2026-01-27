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

## Usage

This package is imported by `composer-alloc`. Most users do not need to call it directly.
