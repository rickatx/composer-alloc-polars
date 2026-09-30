use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

/// One indicator result; invalid rows always carry a null value.
#[derive(Clone, Copy, Debug, PartialEq)]
struct IndicatorRow {
    value: Option<f64>,
    valid: bool,
}

impl IndicatorRow {
    /// Construct the sole invalid output representation.
    fn invalid() -> Self {
        Self {
            value: None,
            valid: false,
        }
    }

    /// Admit only finite computed values as valid results.
    fn finite(value: f64) -> Option<Self> {
        value.is_finite().then_some(Self {
            value: Some(value),
            valid: true,
        })
    }
}

/// Declare the stable two-field Struct returned by both plugin expressions.
fn indicator_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let _ = input_fields;
    Ok(Field::new(
        "indicator".into(),
        DataType::Struct(vec![
            Field::new("value".into(), DataType::Float64),
            Field::new("valid".into(), DataType::Boolean),
        ]),
    ))
}

/// Validate the three typed inputs, equal data lengths, and positive period.
fn indicator_inputs(inputs: &[Series]) -> PolarsResult<(&Float64Chunked, &BooleanChunked, usize)> {
    if inputs.len() != 3 {
        return Err(PolarsError::ComputeError(
            "indicator requires values, validity, and period".into(),
        ));
    }
    let values = inputs[0].f64()?;
    let validity = inputs[1].bool()?;
    if values.len() != validity.len() {
        return Err(PolarsError::ComputeError(
            "indicator values and validity must have equal lengths".into(),
        ));
    }
    let period = inputs[2]
        .i64()?
        .get(0)
        .ok_or_else(|| PolarsError::ComputeError("indicator period must be present".into()))?;
    let period = usize::try_from(period).map_err(|_| {
        PolarsError::ComputeError("indicator period must be a positive usize".into())
    })?;
    if period == 0 {
        return Err(PolarsError::ComputeError(
            "indicator period must be positive".into(),
        ));
    }
    Ok((values, validity, period))
}

/// Accept a numeric row only when provenance is exactly true and value finite.
fn effective_value(value: Option<f64>, validity: Option<bool>) -> Option<f64> {
    match (value, validity) {
        (Some(value), Some(true)) if value.is_finite() => Some(value),
        _ => None,
    }
}

/// Stream SMA-seeded EMA rows with constant recurrence state.
///
/// An invalid prefix is skipped. An invalid row after the first accepted
/// value poisons all later output, including during seed accumulation.
fn ema_kernel(
    values: impl Iterator<Item = Option<f64>>,
    validity: impl Iterator<Item = Option<bool>>,
    period: usize,
    mut emit: impl FnMut(IndicatorRow),
) {
    let mut started = false;
    let mut poisoned = false;
    let mut seed_count = 0;
    let mut seed_sum = 0.0;
    let mut previous = None;
    let alpha = 2.0 / (period as f64 + 1.0);

    for (value, validity) in values.zip(validity) {
        if poisoned {
            emit(IndicatorRow::invalid());
            continue;
        }
        let Some(value) = effective_value(value, validity) else {
            poisoned = started;
            emit(IndicatorRow::invalid());
            continue;
        };
        started = true;
        let next = if let Some(previous) = previous {
            alpha * value + (1.0 - alpha) * previous
        } else {
            seed_count += 1;
            seed_sum += value;
            if seed_count < period {
                if !seed_sum.is_finite() {
                    poisoned = true;
                }
                emit(IndicatorRow::invalid());
                continue;
            }
            seed_sum / period as f64
        };
        if let Some(row) = IndicatorRow::finite(next) {
            previous = Some(next);
            emit(row);
        } else {
            poisoned = true;
            emit(IndicatorRow::invalid());
        }
    }
}

/// Apply the specified RSI zero-total convention, rejecting non-finite scores.
fn rsi_score(avg_gain: f64, avg_loss: f64) -> Option<f64> {
    let total = avg_gain + avg_loss;
    if !total.is_finite() {
        return None;
    }
    let score = if total == 0.0 {
        0.0
    } else {
        100.0 * avg_gain / total
    };
    score.is_finite().then_some(score)
}

/// Stream Wilder RSI rows with constant recurrence state.
///
/// The first accepted price starts the seed. Interior invalidity and
/// non-finite arithmetic permanently poison later rows in this frame.
fn rsi_kernel(
    values: impl Iterator<Item = Option<f64>>,
    validity: impl Iterator<Item = Option<bool>>,
    period: usize,
    mut emit: impl FnMut(IndicatorRow),
) {
    let mut prior = None;
    let mut poisoned = false;
    let mut delta_count = 0;
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    let period_f64 = period as f64;

    for (value, validity) in values.zip(validity) {
        if poisoned {
            emit(IndicatorRow::invalid());
            continue;
        }
        let Some(value) = effective_value(value, validity) else {
            poisoned = prior.is_some();
            emit(IndicatorRow::invalid());
            continue;
        };
        let Some(previous) = prior.replace(value) else {
            emit(IndicatorRow::invalid());
            continue;
        };
        let delta = value - previous;
        if !delta.is_finite() {
            poisoned = true;
            emit(IndicatorRow::invalid());
            continue;
        }
        let gain = delta.max(0.0);
        let loss = (-delta).max(0.0);
        if delta_count < period {
            delta_count += 1;
            avg_gain += gain;
            avg_loss += loss;
            if !avg_gain.is_finite() || !avg_loss.is_finite() {
                poisoned = true;
                emit(IndicatorRow::invalid());
                continue;
            }
            if delta_count < period {
                emit(IndicatorRow::invalid());
                continue;
            }
            avg_gain /= period_f64;
            avg_loss /= period_f64;
        } else {
            avg_gain = (avg_gain * (period_f64 - 1.0) + gain) / period_f64;
            avg_loss = (avg_loss * (period_f64 - 1.0) + loss) / period_f64;
        }
        let row = rsi_score(avg_gain, avg_loss).and_then(IndicatorRow::finite);
        if let Some(row) = row {
            emit(row);
        } else {
            poisoned = true;
            emit(IndicatorRow::invalid());
        }
    }
}

/// Build both output fields in one kernel evaluation, then pack one Struct.
fn indicator_series(
    len: usize,
    calculate: impl FnOnce(&mut dyn FnMut(IndicatorRow)),
) -> PolarsResult<Series> {
    let mut output_values = Vec::with_capacity(len);
    let mut output_validity = Vec::with_capacity(len);
    calculate(&mut |row| {
        output_values.push(row.value);
        output_validity.push(row.valid);
    });
    let value_series =
        Float64Chunked::from_iter_options("value".into(), output_values.into_iter()).into_series();
    let valid_series = Series::new("valid".into(), output_validity);
    Ok(StructChunked::from_series(
        "indicator".into(),
        len,
        [&value_series, &valid_series].into_iter(),
    )?
    .into_series())
}

/// Polars EMA entry point using the shared validity-aware series builder.
#[polars_expr(output_type_func=indicator_output)]
pub fn ema_with_validity(inputs: &[Series]) -> PolarsResult<Series> {
    ema_series(inputs)
}

/// Validate EMA inputs and stream its rows into the Struct field vectors.
fn ema_series(inputs: &[Series]) -> PolarsResult<Series> {
    let (values, validity, period) = indicator_inputs(inputs)?;
    indicator_series(values.len(), |emit| {
        ema_kernel(values.into_iter(), validity.into_iter(), period, emit);
    })
}

/// Polars RSI entry point using the shared validity-aware series builder.
#[polars_expr(output_type_func=indicator_output)]
pub fn rsi_with_validity(inputs: &[Series]) -> PolarsResult<Series> {
    rsi_series(inputs)
}

/// Validate RSI inputs and stream its rows into the Struct field vectors.
fn rsi_series(inputs: &[Series]) -> PolarsResult<Series> {
    let (values, validity, period) = indicator_inputs(inputs)?;
    indicator_series(values.len(), |emit| {
        rsi_kernel(values.into_iter(), validity.into_iter(), period, emit);
    })
}

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
    let len = asset_cols.first().map(|col| col.len()).unwrap_or(0);
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

    Ok(Float64Chunked::from_iter_options("max_drawdown".into(), out.into_iter()).into_series())
}

#[cfg(test)]
mod indicator_tests {
    use super::*;

    fn rows(
        values: &[Option<f64>],
        validity: &[Option<bool>],
        period: usize,
        rsi: bool,
    ) -> Vec<IndicatorRow> {
        let mut output = Vec::with_capacity(values.len());
        if rsi {
            rsi_kernel(
                values.iter().copied(),
                validity.iter().copied(),
                period,
                |row| output.push(row),
            );
        } else {
            ema_kernel(
                values.iter().copied(),
                validity.iter().copied(),
                period,
                |row| output.push(row),
            );
        }
        output
    }

    fn assert_values(actual: &[IndicatorRow], expected: &[Option<f64>]) {
        assert_eq!(actual.len(), expected.len());
        for (row, expected) in actual.iter().zip(expected) {
            assert_eq!(row.valid, expected.is_some());
            match (row.value, expected) {
                (Some(value), Some(want)) => {
                    assert!(value.is_finite());
                    assert!((value - want).abs() <= 1e-12, "{value} != {want}");
                }
                (None, None) => {}
                other => panic!("unexpected row: {other:?}"),
            }
        }
    }

    #[test]
    fn frozen_ema_seed_and_recurrence() {
        let values = [1., 2., 3., 2., 1.].map(Some);
        let valid = [Some(true); 5];
        assert_values(
            &rows(&values, &valid, 2, false),
            &[None, Some(1.5), Some(2.5), Some(13. / 6.), Some(25. / 18.)],
        );
        assert_values(&rows(&values, &valid, 1, false), &values);
        assert_values(
            &rows(&values, &valid, 5, false),
            &[None, None, None, None, Some(1.8)],
        );
        assert_values(&rows(&values, &valid, 6, false), &[None; 5]);
    }

    #[test]
    fn frozen_rsi_wilder_seed_and_recurrence() {
        let values = [1., 2., 3., 2., 1., 2.].map(Some);
        let valid = [Some(true); 6];
        assert_values(
            &rows(&values, &valid, 2, true),
            &[None, None, Some(100.), Some(50.), Some(25.), Some(62.5)],
        );
        assert_values(
            &rows(&values, &valid, 1, true),
            &[None, Some(100.), Some(100.), Some(0.), Some(0.), Some(100.)],
        );
        assert_values(
            &rows(&values, &valid, 5, true),
            &[None, None, None, None, None, Some(60.)],
        );
        assert_values(&rows(&values, &valid, 6, true), &[None; 6]);
        let flat = [3., 3., 3.].map(Some);
        assert_values(
            &rows(&flat, &[Some(true); 3], 2, true),
            &[None, None, Some(0.)],
        );
    }

    #[test]
    fn prefix_recovery_and_absorbing_interior_failure() {
        for rsi in [false, true] {
            let values = [
                Some(999.),
                Some(888.),
                Some(1.),
                Some(2.),
                Some(3.),
                Some(4.),
            ];
            let validity = [
                Some(false),
                None,
                Some(true),
                Some(true),
                Some(true),
                Some(true),
            ];
            let prefixed = rows(&values, &validity, 2, rsi);
            let suffix = rows(&values[2..], &validity[2..], 2, rsi);
            assert_values(&prefixed[..2], &[None, None]);
            assert_eq!(&prefixed[2..], suffix);

            let poisoned = rows(
                &[Some(1.), Some(2.), Some(3.), Some(4.), Some(5.)],
                &[Some(true), Some(true), Some(false), Some(true), Some(true)],
                3,
                rsi,
            );
            assert_values(&poisoned, &[None; 5]);
        }
    }

    #[test]
    fn every_non_finite_or_null_input_poisons_after_first_valid() {
        for rsi in [false, true] {
            for bad in [
                None,
                Some(f64::NAN),
                Some(f64::INFINITY),
                Some(f64::NEG_INFINITY),
            ] {
                let values = [Some(1.), Some(2.), Some(3.), bad, Some(4.)];
                let out = rows(&values, &[Some(true); 5], 2, rsi);
                assert_values(&out[3..], &[None, None]);
                let leading = rows(
                    &[bad, Some(1.), Some(2.), Some(3.)],
                    &[Some(true); 4],
                    2,
                    rsi,
                );
                let suffix = rows(&[Some(1.), Some(2.), Some(3.)], &[Some(true); 3], 2, rsi);
                assert_values(&leading[..1], &[None]);
                assert_eq!(&leading[1..], suffix);
            }
        }
    }

    #[test]
    fn finite_inputs_cannot_emit_non_finite_results() {
        let rsi = rows(
            &[Some(-f64::MAX), Some(f64::MAX), Some(1.)],
            &[Some(true); 3],
            1,
            true,
        );
        assert_values(&rsi[1..], &[None, None]);
        let ema = rows(
            &[Some(f64::MAX), Some(f64::MAX), Some(1.)],
            &[Some(true); 3],
            2,
            false,
        );
        assert_values(&ema[1..], &[None, None]);
    }

    #[test]
    fn plugin_rejects_mismatched_lengths_and_invalid_periods() {
        let values = Series::new("value".into(), &[1.0, 2.0]);
        let validity = Series::new("validity".into(), &[true]);
        let two = Series::new("period".into(), &[2_i64]);
        for plugin in [ema_series, rsi_series] {
            assert!(plugin(&[]).is_err());
            assert!(plugin(&[values.clone(), validity.clone()]).is_err());
            assert!(plugin(&[values.clone(), validity.clone(), two.clone(), two.clone()]).is_err());
            assert!(plugin(&[values.clone(), validity.clone(), two.clone()]).is_err());
            for period in [None, Some(0_i64), Some(-1_i64)] {
                let period = Series::new("period".into(), &[period]);
                assert!(plugin(&[values.clone(), validity.clone(), period]).is_err());
            }
            let wrong_type = Series::new("period".into(), &[2.0_f64]);
            assert!(plugin(&[values.clone(), validity.clone(), wrong_type]).is_err());
            let wrong_values = Series::new("value".into(), &[true, false]);
            assert!(plugin(&[
                wrong_values,
                Series::new("validity".into(), &[true, true]),
                two.clone()
            ])
            .is_err());
            let wrong_validity = Series::new("validity".into(), &[1.0_f64, 1.0]);
            assert!(plugin(&[values.clone(), wrong_validity, two.clone()]).is_err());
        }
    }
}
