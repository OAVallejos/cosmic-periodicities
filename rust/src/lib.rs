use pyo3::prelude::*;
use std::f64::consts::PI;               // --- FUNCIÓN PARA EL RANGO BAJO (< 0.5) ---
#[pyfunction]              
fn log_prior_fundamental(omega: f64, a_amp: f64) -> f64 {
    let armonicos = [0.191, 0.382]; // Fundamental y 2do armónico
    if omega < 0.05 || omega > 0.505 || a_amp < 0.001 || a_amp > 0.950 {
        return f64::NEG_INFINITY;
    }
    calcular_p_max(omega, &armonicos)
}

// --- FUNCIÓN PARA EL RANGO ALTO (> 0.5) ---
#[pyfunction]
fn log_prior_armonicos(omega: f64, a_amp: f64) -> f64 {
    let armonicos = [0.573, 0.764, 0.955, 1.146]; // Armónicos superiores
    if omega < 0.505 || omega > 1.35 || a_amp < 0.001 || a_amp > 0.950 {
        return f64::NEG_INFINITY;
    }
    calcular_p_max(omega, &armonicos)
}

// Función auxiliar para evitar repetir lógica
fn calcular_p_max(omega: f64, lista: &[f64]) -> f64 {
    let mut max_p = f64::NEG_INFINITY;
    for &h in lista {
        let diff = omega - h;
        let p = -0.5 * (diff / 0.03).powi(2);
        if p > max_p { max_p = p; }
    }
    max_p
}

#[pyfunction]
fn log_likelihood_fast(theta: Vec<f64>, t: Vec<f64>, y: Vec<f64>, yerr: Vec<f64>) -> f64 {
    let m0 = theta[0]; let alpha = theta[1]; let a = theta[2];
    let omega = theta[3]; let phi = theta[4]; let log_sigma = theta[5];
    let sigma2_extra = (2.0 * log_sigma).exp();
    let mut total_ll = 0.0;

    for i in 0..t.len() {
        let y_pred = m0 - alpha * t[i] + a * (omega * t[i] + phi).cos();
        let sigma2 = sigma2_extra + yerr[i].powi(2);
        total_ll += -0.5 * ((y[i] - y_pred).powi(2) / sigma2 + (2.0 * PI * sigma2).ln());
    }
    total_ll
}

#[pymodule]
fn vpm_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(log_prior_fundamental, m)?)?; // Para el script "Bajo"
    m.add_function(wrap_pyfunction!(log_prior_armonicos, m)?)?;   // Para el script "Alto"
    m.add_function(wrap_pyfunction!(log_likelihood_fast, m)?)?;
    Ok(())
}
