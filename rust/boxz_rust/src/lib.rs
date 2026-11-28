//! Boxz Rust performance kernels (optional backend).
//!
//! Scope
//! -----
//! Only operations that are *measured* to be hot in the NumPy reference
//! implementation live here.  Large matmuls stay in NumPy/OpenBLAS, which is
//! already near the FMA ceiling on laptop CPUs; the wins come from
//! memory-bound elementwise passes and from sampling, which is a long tail of
//! tiny Python calls during generation.
//!
//! Correctness contract
//! --------------------
//! Every kernel is numerically comparable with its Python counterpart, and
//! `tests/test_backend_parity.py` asserts agreement on random inputs when the
//! extension is importable.  The Python implementation remains the reference;
//! this crate is never required to train or run Boxz.
//!
//! Build
//! -----
//!     cd rust/boxz_rust && maturin develop --release
//! or, without maturin:
//!     cargo build --release            # target/release/libboxz_rust.so
//!
//! The Python loader (`boxz/backends/rust_backend.py`) looks for the built
//! library and silently falls back to NumPy when it is absent.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[inline]
fn gelu_scalar(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

/// Tanh-approximation GELU: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))).
#[pyfunction]
fn gelu<'py>(py: Python<'py>, input: PyReadonlyArrayDyn<'py, f32>) -> Bound<'py, PyArrayDyn<f32>> {
    input.as_array().mapv(gelu_scalar).into_pyarray(py)
}

/// GELU applied in place on a contiguous 1-D buffer.
#[pyfunction]
fn gelu_inplace(buffer: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
    let mut view = buffer.readwrite();
    let slice = view
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("array must be contiguous"))?;
    for value in slice.iter_mut() {
        *value = gelu_scalar(*value);
    }
    Ok(())
}

/// Numerically stable softmax over the last axis of a 2-D array.
#[pyfunction]
fn softmax<'py>(py: Python<'py>, logits: PyReadonlyArray2<'py, f32>) -> Bound<'py, PyArray2<f32>> {
    let a = logits.as_array();
    let (rows, cols) = (a.shape()[0], a.shape()[1]);
    let mut out = a.to_owned();
    for r in 0..rows {
        let mut row = out.row_mut(r);
        let mut max = f32::NEG_INFINITY;
        for c in 0..cols {
            if row[c] > max {
                max = row[c];
            }
        }
        let mut sum = 0.0f32;
        for c in 0..cols {
            let e = (row[c] - max).exp();
            row[c] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for c in 0..cols {
            row[c] *= inv;
        }
    }
    out.into_pyarray(py)
}

/// LayerNorm over the last axis.  Returns (y, xhat) so a backward pass can
/// reuse the normalised activations instead of recomputing them.
#[pyfunction]
fn layer_norm<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    gamma: PyReadonlyArray1<'py, f32>,
    beta: PyReadonlyArray1<'py, f32>,
    eps: f32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let a = x.as_array();
    let g = gamma.as_array();
    let b = beta.as_array();
    let (rows, cols) = (a.shape()[0], a.shape()[1]);
    if g.len() != cols || b.len() != cols {
        return Err(PyValueError::new_err(
            "gamma/beta length must equal the normalised dimension",
        ));
    }
    let mut y = a.to_owned();
    let mut xhat = a.to_owned();
    for r in 0..rows {
        let mut mean = 0.0f32;
        for c in 0..cols {
            mean += a[[r, c]];
        }
        mean /= cols as f32;
        let mut var = 0.0f32;
        for c in 0..cols {
            let d = a[[r, c]] - mean;
            var += d * d;
        }
        var /= cols as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            let h = (a[[r, c]] - mean) * rstd;
            xhat[[r, c]] = h;
            y[[r, c]] = h * g[c] + b[c];
        }
    }
    Ok((y.into_pyarray(py), xhat.into_pyarray(py)))
}

/// Top-k / top-p / temperature sampling in log space.
///
/// Mirrors `boxz.inference.sampler.sample_token`, including the multiplicative
/// repetition penalty.  Returns the sampled token id.
#[pyfunction]
#[pyo3(signature = (logits, temperature, top_k, top_p, repetition_penalty, recent, seed))]
fn sample(
    logits: PyReadonlyArray1<f32>,
    temperature: f32,
    top_k: i64,
    top_p: f64,
    repetition_penalty: f32,
    recent: PyReadonlyArray1<i64>,
    seed: u64,
) -> PyResult<i64> {
    let l = logits.as_array();
    let n = l.len();
    if n == 0 {
        return Err(PyValueError::new_err("empty logits"));
    }
    let mut work: Vec<f32> = l.to_vec();

    if repetition_penalty != 1.0 {
        for &t in recent.as_array().iter() {
            if t >= 0 && (t as usize) < n {
                let v = work[t as usize];
                work[t as usize] = if v > 0.0 {
                    v / repetition_penalty
                } else {
                    v * repetition_penalty
                };
            }
        }
    }

    if temperature <= 0.0 {
        let mut best = 0usize;
        for i in 1..n {
            if work[i] > work[best] {
                best = i;
            }
        }
        return Ok(best as i64);
    }

    let inv_temp = 1.0 / temperature;
    for v in work.iter_mut() {
        *v *= inv_temp;
    }

    if top_k > 0 && (top_k as usize) < n {
        let k = top_k as usize;
        let mut sorted: Vec<f32> = work.clone();
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[k - 1];
        for v in work.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    let mut max = f32::NEG_INFINITY;
    for &v in work.iter() {
        if v > max {
            max = v;
        }
    }
    let mut probs: Vec<f64> = Vec::with_capacity(n);
    let mut sum = 0.0f64;
    for &v in work.iter() {
        let e = ((v - max) as f64).exp();
        probs.push(e);
        sum += e;
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    if top_p > 0.0 && top_p < 1.0 {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cum = 0.0f64;
        let mut keep = 0usize;
        for (i, &idx) in order.iter().enumerate() {
            cum += probs[idx];
            keep = i + 1;
            if cum >= top_p {
                break;
            }
        }
        for &idx in order.iter().skip(keep) {
            probs[idx] = 0.0;
        }
        let s: f64 = probs.iter().sum();
        if s > 0.0 {
            for p in probs.iter_mut() {
                *p /= s;
            }
        }
    }

    // xorshift64*: deterministic, platform independent
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    if state == 0 {
        state = 0x9E37_79B9_7F4A_7C15;
    }
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    let bits = state.wrapping_mul(2685821657736338717);
    let u = ((bits >> 11) as f64) / ((1u64 << 53) as f64);

    let mut acc = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return Ok(i as i64);
        }
    }
    let mut best = 0usize;
    for i in 1..n {
        if probs[i] > probs[best] {
            best = i;
        }
    }
    Ok(best as i64)
}

#[pymodule]
fn boxz_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    m.add_function(wrap_pyfunction!(gelu_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm, m)?)?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
