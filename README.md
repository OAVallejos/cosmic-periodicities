# Evidence for Cosmic Harmonic Periodicity in Galaxy Evolution

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18406995.svg)](https://doi.org/10.5281/zenodo.18406995)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by Rust](https://img.shields.io/badge/Powered%20by-Rust-orange.svg)](https://www.rust-lang.org/)

Repository for the Bayesian analysis of 59,569 JWST and 1,734 SDSS/eROSITA galaxies.

## 🌌 Scientific Discovery
This code reproduces the decisive Bayesian evidence ($\log\mathrm{BF} > 2500$) for harmonic periodicities in cosmic evolution:
Fundamental Frequency: $\omega_0 = 0.191 \pm 0.012$ (detected in SDSS, $z < 0.15$).
Harmonic Series: Detection of $n \times \omega_0$ (2nd, 4th, 6th harmonics) in JWST deep fields.
Significance: Results challenge standard stochastic evolution models.

## ⚡ Technical Implementation
To handle the computational load of high-dimensional MCMC sampling on ~60k galaxies, the likelihood kernel is written in Rust using `PyO3`, achieving a 40x speedup over pure Python.

Python: Data processing, cosmology (`astropy`), MCMC driver (`emcee`).
Rust: SIMD-optimized Log-Likelihood & Prior calculations.

## 🚀 Usage

### Prerequisites
Python 3.9+ (`numpy`, `pandas`, `emcee`, `astropy`, `scipy`)
Rust (Cargo)

### Installation & Execution

1.  Compile the Rust Kernel:
    bash
    maturin develop --release
    
    pip install .

2.  Run the Analysis:


## 📂 Data
SDSS/eROSITA: Cross-matched catalogs (DR19 + eRASS1).
JWST: Bouwens et al. (2023) photometry from 7 fields (CEERS, PRIMER, JADES, etc.).


# preprint (v1.0)
JWST_A.py            
JWST_B.py
eROSITA_fits_csv.py
eROSITA_OMEGA.py


# preprint (v1.2)
FUNCTION_JWST.py 
COEFICCIENT_JWST.py   
DERIVATIVE_PLOT.py 

# preprint (v1.3)
lib_rs.txt
STRUCTURE_JWST.py
STRUCTURE_JWST_v2.py
OSC_JWST.py
OSC_JWST_v3.py



## 📜 Citation
If you use this code or data, please cite:

> Vallejos, O. A. (2026). Evidence for Cosmic Harmonic Periodicity in Galaxy. (v1.0). Zenodo. https://doi.org/10.5281/zenodo.18406995

Evidence for Cosmic Harmonic Periodicity in Galaxy. (v1.2). Zenodo. https://doi.org/10.5281/zenodo.18445297

Discovery of Cosmic Crystal Structure at High Redshift. (v1.3). Zenodo. https://doi.org/10.5281/zenodo.18475085

Analysis of Cosmic Crystalline Structure. (v1.4). Zenodo. https://doi.org/10.5281/zenodo.18528720
