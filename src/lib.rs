use pyo3::prelude::*;

mod expressions;

#[pymodule]
fn _lib(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
