use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn filter_weights_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let _ = input_fields;
    Ok(Field::new(
        "weights".into(),
        DataType::List(Box::new(DataType::Float64)),
    ))
}

#[polars_expr(output_type_func=filter_weights_output)]
pub fn filter_select_weights(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 3 {
        return Err(PolarsError::ComputeError(
            "filter_select_weights requires >= 1 score column plus N and reverse".into(),
        ));
    }

    let asset_count = inputs.len() - 2;
    let n_series = inputs[asset_count].i64()?;
    let reverse_series = inputs[asset_count + 1].bool()?;
    let n = n_series.get(0).unwrap_or(0).max(0) as usize;
    let reverse = reverse_series.get(0).unwrap_or(false);

    let mut asset_cols = Vec::with_capacity(asset_count);
    for s in &inputs[..asset_count] {
        asset_cols.push(s.f64()?);
    }
    let len = asset_cols
        .first()
        .map(|col| col.len())
        .unwrap_or(0);
    for col in &asset_cols[1..] {
        if col.len() != len {
            return Err(PolarsError::ComputeError(
                "filter_select_weights input columns must have the same length".into(),
            ));
        }
    }

    let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        "weights".into(),
        len,
        asset_count * len,
        DataType::Float64,
    );

    for row in 0..len {
        let mut items: Vec<(usize, f64)> = Vec::with_capacity(asset_count);
        for (idx, col) in asset_cols.iter().enumerate() {
            if let Some(value) = col.get(row) {
                items.push((idx, value));
            }
        }

        items.sort_by(|(idx_a, val_a), (idx_b, val_b)| {
            let ord = match val_a.partial_cmp(val_b) {
                Some(ordering) => ordering,
                None => std::cmp::Ordering::Equal,
            };
            let ord = if reverse { ord.reverse() } else { ord };
            if ord == std::cmp::Ordering::Equal {
                idx_a.cmp(idx_b)
            } else {
                ord
            }
        });

        let take = n.min(items.len());
        let mut weights = vec![0.0; asset_count];
        if take > 0 {
            let weight = 1.0 / take as f64;
            for (idx, _value) in items.iter().take(take) {
                weights[*idx] = weight;
            }
        }
        builder.append_slice(&weights);
    }

    Ok(builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn rolling_max_drawdown(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 2 {
        return Err(PolarsError::ComputeError(
            "rolling_max_drawdown requires a value series and a window size".into(),
        ));
    }

    let values = inputs[0].f64()?;
    let window_series = inputs[1].i64()?;
    let window = window_series.get(0).unwrap_or(0).max(0) as usize;
    if window == 0 {
        return Err(PolarsError::ComputeError(
            "rolling_max_drawdown requires a positive window size".into(),
        ));
    }

    let len = values.len();
    let mut out: Vec<Option<f64>> = Vec::with_capacity(len);

    for i in 0..len {
        if i + 1 < window {
            out.push(None);
            continue;
        }

        let start = i + 1 - window;
        let mut peak: Option<f64> = None;
        let mut mdd = 0.0;
        let mut ok = true;

        for idx in start..=i {
            match values.get(idx) {
                Some(value) => {
                    if peak.map_or(true, |p| value > p) {
                        peak = Some(value);
                    }
                    let running_peak = peak.unwrap();
                    let drawdown = value / running_peak - 1.0;
                    if drawdown < mdd {
                        mdd = drawdown;
                    }
                }
                None => {
                    ok = false;
                    break;
                }
            }
        }

        if ok {
            out.push(Some(mdd.abs() * 100.0));
        } else {
            out.push(None);
        }
    }

    Ok(Float64Chunked::from_iter_options(
        "max_drawdown".into(),
        out.into_iter(),
    )
    .into_series())
}
