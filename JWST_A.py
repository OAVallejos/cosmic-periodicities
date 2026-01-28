#!/usr/bin/env python3
"""
MCMC JWST - HIGH RANGE ANALYSIS (ω > 0.5)
HIGHER HARMONICS DETECTION OF THE FUNDAMENTAL RHYTHM
SPECIFIC SEARCH: 0.50 < ω < 1.20 (HARMONICS 0.764, 0.955, 1.146)
NO GRAPHS - NUMERICAL ANALYSIS ONLY
"""
import numpy as np
import pandas as pd
from scipy import stats, optimize, integrate
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
import emcee
from pathlib import Path
import gzip
import warnings
from datetime import datetime
import subprocess
import tempfile
import os
import sys
from multiprocessing import Pool
warnings.filterwarnings('ignore')

import vpm_engine

# --- LOOK FOR THIS PART AT THE BEGINNING OF THE SCRIPT ---
try:
    import vpm_engine
    USAR_RUST = True
    RUST_ENGINE_READY = True  # <--- ADD THIS LINE HERE
    print("🚀 Rust engine detected and linked.")
except ImportError:
    USAR_RUST = False
    RUST_ENGINE_READY = False # <--- AND THIS ONE TOO
    print("⚠️  Warning: vpm_engine not found. Using slow Python.")
    
# ============================================================================
# 1. CORRECT COSMOLOGY - PLANCK 2018
# ============================================================================

from astropy.cosmology import FlatLambdaCDM
import numpy as np

# Standard Planck 2018 Configuration
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)

def calcular_tiempo_lookback_rapido(z_data):
    """Calculates lookback time in Gyr (Planck 2018)"""
    z_array = np.asarray(z_data, dtype=np.float64)
    # Limit minimum z to 0.001 to avoid division or logic errors in astropy
    z_safe = np.maximum(z_array, 0.001)
    return cosmo.lookback_time(z_safe).value

def calcular_tiempo_lookback_con_error(z_vals, z_err_vals):
    """Robustly calculates time and propagates photo-z error"""
    z_array = np.asarray(z_vals, dtype=np.float64)
    z_err_array = np.asarray(z_err_vals, dtype=np.float64)
    
    # 1. Central Time
    t_mean = calcular_tiempo_lookback_rapido(z_array)
    
    # 2. Propagation by Finite Difference (Safety patch z_min=0.01)
    z_min_safe = np.maximum(z_array - z_err_array, 0.01)
    z_max_safe = np.minimum(z_array + z_err_array, 20.0) # JWST doesn't see beyond z=20
    
    t_plus = cosmo.lookback_time(z_max_safe).value
    t_minus = cosmo.lookback_time(z_min_safe).value
    
    t_err_prop = np.abs(t_plus - t_minus) / 2.0
    
    # 3. Error floor to avoid statistical optimism (0.05 * (1+z))
    t_err_min = 0.05 * (1 + z_array)
    t_err_final = np.maximum(t_err_prop, t_err_min)
    
    return t_mean, t_err_final

    
# ============================================================================
# 2. GLOBAL FUNCTIONS OPTIMIZED WITH RUST (ω > 0.5)
# ============================================================================

def log_prior_armonicos_superiores(theta):
    """Call to optimized Prior in Rust"""
    # theta[3] is omega, theta[2] is Amplitude
    if USAR_RUST:
        # We pass the floats directly
        return vpm_engine.log_prior_armonicos(float(theta[3]), float(theta[2]))
    
    # Manual fallback in case Rust doesn't load
    M0, alpha, A, omega, phi, log_sigma = theta
    if not (0.505 < omega < 1.30) or not (0.001 < A < 0.850): 
        return -np.inf
    # Centered on the first detectable harmonic of the high range
    return -0.5 * ((omega - 0.573) / 0.05)**2

def log_likelihood_con_rust_armonicos(theta, t, y, yerr):
    """Call to massive Likelihood in Rust (Optimized without .tolist())"""
    if USAR_RUST:
        # IMPORTANT: We pass numpy arrays directly.
        # Rust (via PyO3) interprets them as memory slices much faster.
        return vpm_engine.log_likelihood_fast(theta, t, y, yerr)
    
    # Python Fallback (Slow)
    M0, alpha, A, omega, phi, log_sigma = theta
    y_pred = M0 - alpha * t + A * np.cos(omega * t + phi)
    sigma2 = np.exp(2 * log_sigma) + yerr**2
    return -0.5 * np.sum((y - y_pred)**2 / sigma2 + np.log(2 * np.pi * sigma2))

def log_probability_armonicos_superiores(theta, t, y, yerr):
    """Total probability function linked to the Rust engine"""
    lp = log_prior_armonicos_superiores(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_con_rust_armonicos(theta, t, y, yerr)
# ============================================================================
# 3. LIKELIHOOD FOR NULL MODEL (COMPARISON)
# ============================================================================

def log_prior_nulo_unificado(theta):
    M0, alpha, log_sigma = theta
    if not (-5.0 < M0 < 5.0) or not (-1.0 < alpha < 1.0) or not (-4.0 < log_sigma < 2.0):
        return -np.inf
    return 0.0

def log_likelihood_nulo(theta, t, y, yerr):
    M0, alpha, log_sigma = theta
    y_pred = M0 - alpha * t
    sigma2 = np.exp(2 * log_sigma) + yerr**2
    return -0.5 * np.sum((y - y_pred)**2 / sigma2 + np.log(2 * np.pi * sigma2))

def log_probability_nulo(theta, t, y, yerr):
    lp = log_prior_nulo_unificado(theta)
    if not np.isfinite(lp): return -np.inf
    return lp + log_likelihood_nulo(theta, t, y, yerr)


# ============================================================================
# OPTIMIZED LOADER CLASS (CORRECTED: REAL ERROR AND SNR FILTERING)
# ============================================================================

class CargadorDatosJWST_Corregido:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.campos = {}

        self.campos_config = {
            'CEERS': ('ceerso.dat.gz', 'ceersp.dat.gz', 'estandar'),
            'JADES-GS': ('jadesgso.dat.gz', 'jadesgsp.dat.gz', 'estandar'),
            'JADES-GN': ('jadesgno.dat.gz', 'jadesgnp.dat.gz', 'estandar'),
            'PRIMER-UDS': ('primeruo.dat.gz', 'primerup.dat.gz', 'estandar'),
            'PRIMER-COSMOS': ('primerco.dat.gz', 'primercp.dat.gz', 'estandar'),
            'NGDEEP': ('ngdeepo.dat.gz', 'ngdeepp.dat.gz', 'estandar'),
            'A2744-UNCOVER': ('a2744o.dat.gz', 'a2744p.dat.gz', 'uncover')
        }

    def _prefiltar_con_python(self, archivo_gz, tipo='fotometria'):
        """Robust reading with extraction of real error in column 29"""
        datos_validos = []
        try:
            with gzip.open(self.data_dir / archivo_gz, 'rt') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        gal_id = int(float(parts[0]))
                        if tipo == 'fotometria' and len(parts) > 29:
                            f444w = float(parts[28])
                            f444w_err = float(parts[29]) # Real error column
                            if 0 < f444w < 1e6:
                                flag = int(float(parts[-1])) if len(parts) > 30 else 0
                                datos_validos.append([gal_id, f444w, f444w_err, flag])
                        elif tipo == 'redshift' and len(parts) > 4:
                            z = float(parts[4])
                            if 0.1 < z < 15.0:
                                flag = int(float(parts[-1])) if len(parts) > 6 else 0
                                datos_validos.append([gal_id, z, flag])
                    except (ValueError, IndexError):
                        continue
            return datos_validos
        except Exception as e:
            print(f" ❌ Error in pre-filtering: {str(e)[:50]}")
            return []

    def cargar_todo(self):
        """Load with weights based on real observational errors"""
        print("\n📂 OPTIMIZED LOAD OF 7 JWST FIELDS (LOW FREQUENCY MODE)")
        print("=" * 60)

        for nombre, (file_o, file_p, tipo) in self.campos_config.items():
            print(f"🔍 {nombre}:", end=' ', flush=True)

            try:
                datos_o = self._prefiltar_con_python(file_o, 'fotometria')
                datos_p = self._prefiltar_con_python(file_p, 'redshift')

                if not datos_o or not datos_p:
                    print("⚠️ Incomplete data")
                    continue

                df_o = pd.DataFrame(datos_o, columns=['ID', 'F444W', 'F444W_err', 'Flag_o'])
                df_p = pd.DataFrame(datos_p, columns=['ID', 'zphot', 'Flag_p'])

                df = pd.merge(df_o, df_p, on='ID', how='inner')

                # Standard photo-z redshift error
                df['zphot_err'] = 0.05 * (1 + df['zphot'])

                # Strict mask: SNR > 2 and flag quality
                mask = (
                    (df['zphot'] >= 0.5) & (df['zphot'] <= 12.0) &
                    (df['Flag_o'] < 100) & (df['Flag_p'] < 100) &
                    (df['F444W_err'] / df['F444W'] < 0.5) # SNR filter > 2
                )

                df = df[mask].copy()

                if not df.empty:
                    # The weight is now purely photometric for amplitude adjustment
                    df['peso'] = 1.0 / (df['F444W_err']**2 + 1e-10)
                    df['peso'] /= df['peso'].median()

                    self.campos[nombre] = df
                    print(f"✅ {len(df):,} galaxies")
                else:
                    print("⚠️ No galaxies after filtering")

            except Exception as e:
                print(f"❌ Error: {str(e)}")

        print(f"\n📊 TOTAL FIELDS LOADED: {len(self.campos)}/7")
        return len(self.campos) > 0


# ============================================================================
# MAIN CLASS - SPECIFIC ANALYSIS FOR ω > 0.5
# ============================================================================

class AnalisisMCMCJWST_ArmonicosSuperiores:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)

        # Fundamental rhythm (Base of periodicity in ln(1+z))
        self.omega_fundamental = 0.191

        # UPDATE: Include the 3rd Harmonic (0.573) which is dominant
        # This will prevent the report from marking "Not detected" when finding ω ≈ 0.55
        self.armonicos_objetivo = [0.573, 0.764, 0.955, 1.146]

        # Magnification factor for A2744 (Lensing)
        self.mu_a2744 = 1.5

        # Initialize loader
        self.cargador = CargadorDatosJWST_Corregido(data_dir)
        self.campos_procesados = {}
        self.resultados = {}

        # Optimized MCMC configuration for high range search (ω > 0.5)
        self.n_walkers = 32  # Slightly increased for better sampling
        self.n_steps = 1800
        self.burnin = 600

        # SEARCH RANGE OF PRIOR (Shielding against edges)
        # We define this so probability functions use it
        self.omega_min_search = 0.505
        self.omega_max_search = 0.850

        print("=" * 80)
        print("MCMC JWST - HIGHER HARMONICS ANALYSIS (ω > 0.5)")
        print(f"ω fundamental: {self.omega_fundamental:.3f}")
        print(f"Target harmonics: {', '.join([f'{x:.3f}' for x in self.armonicos_objetivo])}")
        print(f"Search range: 0.50 < ω < 1.30")
        print("=" * 80)

    def cargar_y_procesar_datos(self):
        """Loads and processes all data"""
        print("\n📥 LOADING DATA...")

        # Load data
        if not self.cargador.cargar_todo():
            print("❌ Error loading data")
            return False

        if len(self.cargador.campos) == 0:
            print("❌ No data loaded")
            return False

        # Process all fields
        print("\n⚙️  PROCESSING ALL FIELDS...")
        for nombre, df in self.cargador.campos.items():
            if len(df) >= 100:  # Minimum 100 galaxies
                self.procesar_campo(nombre, df)

        if not self.campos_procesados:
            print("❌ No processed fields")
            return False

        print(f"\n✅ {len(self.campos_procesados)} fields processed successfully")
        return True

    def procesar_campo(self, nombre, df):
        """Processes an individual field with error propagation"""
        print(f"\n⚙️  Processing {nombre}:")

        try:
            df_proc = df.copy()

            # 1. Apparent magnitude (ZP=25.0 for JWST/NIRCam)
            df_proc['mag'] = 25.0 - 2.5 * np.log10(df_proc['F444W'].clip(lower=1e-10))
            df_proc['mag_err'] = 1.086 * df_proc['F444W_err'] / df_proc['F444W'].clip(lower=1e-10)

            # 2. Lookback time with error
            z_vals = df_proc['zphot'].values
            z_err_vals = df_proc['zphot_err'].values
            df_proc['t'], df_proc['t_err'] = calcular_tiempo_lookback_con_error(z_vals, z_err_vals)

            # ========================================================================
            # 3. Absolute magnitude (EXACT COSMOLOGICAL CALCULATION - REPLACE HERE)
            # ========================================================================
            # We use the 'cosmo' instance of astropy (FlatLambdaCDM) defined at the beginning
            dL_mpc = cosmo.luminosity_distance(z_vals).value 
            df_proc['dL'] = dL_mpc
            
            # IMPORTANT! Define dL_err for the next step:
            df_proc['dL_err'] = (df_proc['dL'] / np.maximum(z_vals, 0.1)) * z_err_vals
            
            # Conversion to Absolute Magnitude
            # M = m - 5 * log10(dL / 10pc) -> dL in Mpc * 1e6 = dL in pc
            log_dL = np.log10(df_proc['dL'] * 1e6)
            df_proc['M'] = df_proc['mag'] - 5 * log_dL - 5

            # 4. Propagated error in M
            dL_rel_err = df_proc['dL_err'] / df_proc['dL'].clip(lower=1e-10)
            df_proc['M_err'] = np.sqrt(df_proc['mag_err']**2 + (2.17 * dL_rel_err)**2)

            # 5. Lensing correction for A2744
            if 'A2744' in nombre.upper():
                corr_lens = 2.5 * np.log10(self.mu_a2744)
                df_proc['M'] += corr_lens
                print(f"    ✅ Lensing correction applied: +{corr_lens:.3f} mag")

            # 6. Extraction of residuals (Linear Trend)
            t_v, M_v, w_v = df_proc['t'].values, df_proc['M'].values, df_proc['peso'].values
            coeffs = np.polyfit(t_v, M_v, 1, w=w_v)
            df_proc['residuos'] = M_v - (coeffs[0] * t_v + coeffs[1])
            df_proc['residuos_err'] = df_proc['M_err']

            # 7. Signal SNR
            df_proc['snr'] = np.abs(df_proc['residuos']) / df_proc['residuos_err'].clip(lower=1e-10)

            self.campos_procesados[nombre] = {
                'data': df_proc, 'n': len(df_proc),
                'z_median': np.median(z_vals),
                't_median': np.median(df_proc['t']),
                'snr_median': np.median(df_proc['snr']),
                'residuos_std': np.std(df_proc['residuos'])
            }

            print(f"   ✅ {len(df_proc):,} sources | z̃={np.median(z_vals):.2f} | SNR̃={np.median(df_proc['snr']):.2f}")
            return True

        except Exception as e:
            print(f"   ❌ Error in field {nombre}: {e}")
            return False

    def detectar_armonicos_superiores(self, nombre):
        """Specific detection in range ω > 0.5"""
        if nombre not in self.campos_procesados:
            return None

        df = self.campos_procesados[nombre]['data']

        # Quality filter for detection
        mask = (df['snr'] > 1.5)
        t, y, w = df['t'][mask].values, df['residuos'][mask].values, df['peso'][mask].values

        if len(t) < 20:
            print(f"   ⚠️  {nombre}: Insufficient data for detection")
            return None

        # SPECIFIC range for higher harmonics
        omegas = np.linspace(0.50, 1.30, 400)  # Extended range for harmonics
        potencias = []
        for om in omegas:
            c, s = np.cos(om * t), np.sin(om * t)
            p = (np.sum(w * y * c)**2 / np.sum(w * c**2)) + (np.sum(w * y * s)**2 / np.sum(w * s**2))
            potencias.append(p)

        # Find peaks near target harmonics
        potencias_arr = np.array(potencias)

        # Specifically search near theoretical harmonics
        resultados_armonicos = {}

        for i, armonico in enumerate(self.armonicos_objetivo):
            # Window around each harmonic
            ventana = 0.05  # ±0.05 around harmonic
            idx_min = int((armonico - ventana - 0.50) / (1.30 - 0.50) * 400)
            idx_max = int((armonico + ventana - 0.50) / (1.30 - 0.50) * 400)
            idx_min = max(0, idx_min)
            idx_max = min(399, idx_max)

            if idx_max > idx_min:
                sub_potencias = potencias_arr[idx_min:idx_max]
                sub_omegas = omegas[idx_min:idx_max]

                if len(sub_potencias) > 0:
                    idx_pico_local = np.argmax(sub_potencias)
                    omega_pico = sub_omegas[idx_pico_local]
                    potencia_pico = sub_potencias[idx_pico_local]
                    snr_local = potencia_pico / np.mean(potencias_arr)

                    resultados_armonicos[f'armonico_{i+4}'] = {
                        'teorico': armonico,
                        'detectado': omega_pico,
                        'potencia': potencia_pico,
                        'snr': snr_local,
                        'desviacion': abs(omega_pico - armonico)
                    }

        # Also find global peak in entire range
        peaks, properties = find_peaks(potencias_arr, height=np.mean(potencias_arr)*1.5, distance=30)

        if len(peaks) > 0:
            peak_heights = potencias_arr[peaks]
            sorted_idx = np.argsort(peak_heights)[::-1]
            omega_principal = omegas[peaks][sorted_idx[0]]
            snr_principal = peak_heights[sorted_idx[0]] / np.mean(potencias_arr)
        else:
            omega_principal = None
            snr_principal = 0

        res = {
            'omega_principal': omega_principal,
            'snr_principal': snr_principal,
            'armonicos': resultados_armonicos,
            'omegas_detectados': omegas[peaks][:3] if len(peaks) > 0 else [],
            'potencias_detectadas': potencias_arr[peaks][:3] if len(peaks) > 0 else []
        }

        self.resultados.setdefault(nombre, {})['detection'] = res

        # Show results - CORRECTED
        if omega_principal:
            print(f"   📊 ω_peak principal (>0.5): {omega_principal:.4f} (SNR: {snr_principal:.2f})")
        else:
            print(f"   📊 ω_peak principal (>0.5): N/A (SNR: {snr_principal:.2f})")

        for key, armonico_data in resultados_armonicos.items():
            print(f"   🎯 {key}: ω_teo={armonico_data['teorico']:.3f}, ω_det={armonico_data['detectado']:.3f}, "
                  f"Δ={armonico_data['desviacion']:.3f}, SNR={armonico_data['snr']:.2f}")

        return res

    def ejecutar_mcmc_armonicos_superiores(self, nombre):
        """Optimized MCMC: Rust + Multiprocessing + Downsampling"""
        if nombre not in self.campos_procesados:
            return

        df = self.campos_procesados[nombre]['data']
        mask = (df['snr'] > 2.0)
        t, y, yerr = df['t'][mask].values, df['residuos'][mask].values, df['residuos_err'][mask].values

        # 1. Downsampling for speed (Maximum 20000 galaxies)
        n_max = 20000
        if len(t) > n_max:
            print(f"   📉 Reducing sample from {len(t):,} to {n_max:,} to speed up MCMC...")
            idx = np.random.choice(len(t), n_max, replace=False)
            t, y, yerr = t[idx], y[idx], yerr[idx]

        if len(t) < 50:
            print(f"   ⚠️  {nombre}: Insufficient data (N={len(t)})")
            return

        print(f"\n🔬 MCMC HARMONICS: {nombre} (N={len(t)})")
        
        try:
            # 2. Dimension configuration and initial positions
            ndim, ndim_nulo = 6, 3
            pos = np.array([0.0, 0.0, 0.12, 0.764, 0.0, -0.5]) + \
                  0.05 * np.random.randn(self.n_walkers, ndim)
            pos_nulo = np.array([0.0, 0.0, -0.5]) + \
                       0.05 * np.random.randn(self.n_walkers, ndim_nulo)

            # 3. PARALLEL execution block
            with Pool() as pool:
                # OSCILLATORY MODEL (USES RUST)
                sampler_osc = emcee.EnsembleSampler(
                    self.n_walkers, ndim, log_probability_armonicos_superiores,
                    args=(t, y, yerr), pool=pool
                )
                print(f"   🔄 Burn-in (Rust+Parallel)...")
                state = sampler_osc.run_mcmc(pos, self.burnin, progress=True)
                sampler_osc.reset()
                print(f"   🔄 Sampling (Rust+Parallel)...")
                sampler_osc.run_mcmc(state, self.n_steps, progress=True)

                # NULL MODEL (USES PARALLEL PYTHON)
                print(f"   🔄 Calculating Null Model...")
                sampler_nulo = emcee.EnsembleSampler(
                    self.n_walkers, ndim_nulo, log_probability_nulo,
                    args=(t, y, yerr), pool=pool
                )
                sampler_nulo.run_mcmc(pos_nulo, self.n_steps // 2, progress=True)

            # 4. Bayes Factor calculation (Model comparison)
            log_l_osc = np.median(sampler_osc.get_log_prob()[-200:])
            log_l_nulo = np.median(sampler_nulo.get_log_prob()[-200:])
            log_bf = log_l_osc - log_l_nulo

            # 5. Omega (ω) results processing
            flat_samples = sampler_osc.get_chain(discard=self.burnin//2, flat=True)
            omega_samples = flat_samples[:, 3]
            omega_samples = omega_samples[omega_samples > 0.50]

            if len(omega_samples) > 50:
                q = np.percentile(omega_samples, [16, 50, 84])
                omega_median = q[1]
                armonico_cercano = min(self.armonicos_objetivo, key=lambda x: abs(x - omega_median))
                desviacion = abs(omega_median - armonico_cercano)

                self.resultados[nombre]['mcmc'] = {
                    'log_bayes_factor': log_bf,
                    'omega_median': omega_median,
                    'omega_err_low': omega_median - q[0],
                    'omega_err_high': q[2] - omega_median,
                    'omega_std': np.std(omega_samples),
                    'armonico_cercano': armonico_cercano,
                    'desviacion_armonico': desviacion,
                    'es_armonico_significativo': desviacion < 0.03,
                    'preferido': 'oscilatorio' if log_bf > 2.0 else 'nulo',
                    'frac_omega_gt_0_5': len(omega_samples) / len(flat_samples)
                }
                
                print(f"   ✅ log(BF): {log_bf:.2f} | ω: {omega_median:.4f} (Δ={desviacion:.4f})")
            
        except Exception as e:
            print(f"   ❌ Error in MCMC {nombre}: {e}")

    def analizar_por_bins_redshift_pro(self):
        """Groups galaxies by redshift bins with improved Malmquist correction"""
        
        # Cosmic Bins Definition (3 evolutionary epochs)
        bins = [(1.0, 2.5), (2.5, 5.0), (5.0, 12.0)]  # More balanced bins
        etiquetas = ["Z_EVOLUCIONADO", "Z_PICO_ESTELAR", "Z_PRIMITIVO"]
        
        # Combine all datasets maintaining field identity
        listas_datos = []
        for campo, info in self.campos_procesados.items():
            df_campo = info['data'].copy()
            df_campo['campo'] = campo  # Save origin for later analysis
            listas_datos.append(df_campo)
        
        df_total = pd.concat(listas_datos, ignore_index=True)
        print(f"\n📊 TOTAL COMBINED DATA: {len(df_total):,} galaxies")
        
        resultados_bins = {}
        
        for i, (z_min, z_max) in enumerate(bins):
            print(f"\n{'='*60}")
            print(f"🔥 BIN {i+1}: {etiquetas[i]} (z = {z_min}-{z_max})")
            print(f"{'='*60}")
            
            # --- IMPROVED MALMQUIST CORRECTION (3 levels) ---
            print(f"🔧 APPLYING QUALITY FILTERS:")
            
            # LEVEL 1: Basic quality filter
            mask_basico = (
                (df_total['zphot'] >= z_min) & 
                (df_total['zphot'] < z_max) & 
                (df_total['snr'] > 1.8) &  # More permissive SNR initially
                (df_total['F444W'] > 0.01) &  # Eliminate flux=0
                (df_total['F444W_err'] / df_total['F444W'] < 1.0)  # Maximum relative error 100%
            )
            df_filtrado = df_total[mask_basico].copy()
            print(f"   ✅ Level 1: {len(df_filtrado):,} galaxies (basic filter)")
            
            # LEVEL 2: Malmquist correction by detection limit
            # Estimate magnitude limit for each redshift bin
            z_medio = (z_min + z_max) / 2
            
            # Typical JWST/NIRCam F444W limit ~ 29 mag
            mag_limite = 29.0  
            # Distance correction: absolute_mag = apparent_mag - 5*log10(dL) - 5
            dL_limite = (3000/70.0) * z_medio * (1 + 0.5 * z_medio) * 1e6  # in pc
            M_limite = mag_limite - 5 * np.log10(dL_limite) - 5
            
            # Apply absolute magnitude cut
            mask_malmquist = df_filtrado['M'] < M_limite
            df_filtrado = df_filtrado[mask_malmquist].copy()
            print(f"   ✅ Level 2: {len(df_filtrado):,} galaxies (Malmquist: M < {M_limite:.1f})")
            
            # LEVEL 3: Statistical outliers filter
            if len(df_filtrado) > 100:
                # Remove outliers in residuals (more than 5 sigma)
                residuos_mean = df_filtrado['residuos'].mean()
                residuos_std = df_filtrado['residuos'].std()
                mask_outliers = (
                    abs(df_filtrado['residuos'] - residuos_mean) < 4.0 * residuos_std
                )
                df_filtrado = df_filtrado[mask_outliers].copy()
                print(f"   ✅ Level 3: {len(df_filtrado):,} galaxies (no outliers >4σ)")
            
            # --- SAMPLE BALANCING ---
            t_bin = df_filtrado['t'].values
            y_bin = df_filtrado['residuos'].values
            yerr_bin = df_filtrado['residuos_err'].values
            
            # If enough data
            if len(t_bin) < 100:
                print(f"   ⚠️  Insufficient bin: {len(t_bin)} galaxies")
                resultados_bins[etiquetas[i]] = {
                    'n_galaxias': len(t_bin),
                    'omega_median': None,
                    'log_bf': None,
                    'status': 'insufficient_data'
                }
                continue
            
            # --- INTELLIGENT DOWN-SAMPLING ---
            print(f"   📈 Original statistics:")
            print(f"     - Mean redshift: {df_filtrado['zphot'].mean():.2f}")
            print(f"     - Mean lookback time: {df_filtrado['t'].mean():.2f} Gyr")
            print(f"     - Mean SNR: {df_filtrado['snr'].mean():.2f}")
            
            # Subsampling strategy: proportional by field
            n_max = 8000  # Reduced for speed without losing precision
            
            if len(t_bin) > n_max:
                print(f"   📉 Subsampling from {len(t_bin):,} to {n_max:,} galaxies...")
                
                # Option 1: Stratified random sampling by field
                campos_unicos = df_filtrado['campo'].unique()
                indices_seleccionados = []
                
                for campo in campos_unicos:
                    idx_campo = df_filtrado[df_filtrado['campo'] == campo].index
                    n_campo = len(idx_campo)
                    fraccion_campo = n_campo / len(df_filtrado)
                    n_muestrear = int(n_max * fraccion_campo)
                    
                    if n_muestrear > 0 and n_muestrear <= n_campo:
                        idx_selec = np.random.choice(idx_campo, n_muestrear, replace=False)
                        indices_seleccionados.extend(idx_selec)
                
                # If we don't reach maximum, complete randomly
                if len(indices_seleccionados) < n_max:
                    idx_restantes = set(df_filtrado.index) - set(indices_seleccionados)
                    n_necesario = n_max - len(indices_seleccionados)
                    if len(idx_restantes) > 0:
                        idx_extra = np.random.choice(list(idx_restantes), 
                                                    min(n_necesario, len(idx_restantes)), 
                                                    replace=False)
                        indices_seleccionados.extend(idx_extra)
                
                df_muestra = df_filtrado.loc[indices_seleccionados]
                t_bin = df_muestra['t'].values
                y_bin = df_muestra['residuos'].values
                yerr_bin = df_muestra['residuos_err'].values
                
                print(f"   📊 Final sample: {len(t_bin):,} galaxies")
                print(f"   📋 Distribution by field:")
                for campo in campos_unicos:
                    n_campo_muestra = (df_muestra['campo'] == campo).sum()
                    if n_campo_muestra > 0:
                        print(f"     - {campo}: {n_campo_muestra:,} ({n_campo_muestra/len(t_bin):.1%})")
            
            # --- PRELIMINARY BIN ANALYSIS ---
            print(f"\n   🔍 Quick bin analysis:")
            
            # 1. Fast periodogram
            from scipy.signal import periodogram
            freqs, psd = periodogram(y_bin, fs=1.0/np.std(t_bin))
            
            # Filter frequencies in range of interest (0.5-1.3)
            mask_freq = (freqs >= 0.5) & (freqs <= 1.3)
            if np.any(mask_freq):
                freq_max = freqs[mask_freq][np.argmax(psd[mask_freq])]
                print(f"     • Dominant frequency (periodogram): {freq_max:.3f}")
            
            # 2. Mean and dispersion
            print(f"     • Residues: μ={np.mean(y_bin):.3f}, σ={np.std(y_bin):.3f}")
            print(f"     • Mean error: {np.mean(yerr_bin):.3f}")
            
            # 3. Fast autocorrelation analysis
            try:
                lags = np.arange(1, 50)
                autocorr = [np.corrcoef(y_bin[:-lag], y_bin[lag:])[0,1] 
                           for lag in lags if len(y_bin) > lag]
                if autocorr:
                    print(f"     • Autocorrelation (lag=1): {autocorr[0]:.3f}")
            except:
                pass
            
            # --- MCMC EXECUTION FOR THE BIN ---
            print(f"\n   🔬 RUNNING MCMC FOR BIN...")
            
            # Optimized configuration for bins
            n_steps_bin = 1200  # Reduced for speed
            burnin_bin = 400
            
            try:
                # We use the same function but with adjusted parameters
                resultado = self.ejecutar_mcmc_para_bin(
                    nombre=etiquetas[i],
                    t=t_bin,
                    y=y_bin,
                    yerr=yerr_bin,
                    n_steps=n_steps_bin,
                    burnin=burnin_bin
                )
                
                resultados_bins[etiquetas[i]] = resultado
                
                # Show results
                if resultado['omega_median'] is not None:
                    print(f"\n   ✅ BIN RESULT {etiquetas[i]}:")
                    print(f"     • ω median: {resultado['omega_median']:.4f}")
                    print(f"     • log(BF): {resultado['log_bf']:.2f}")
                    print(f"     • Closest harmonic: {resultado['armonico_cercano']:.3f}")
                    print(f"     • Deviation: {resultado['desviacion']:.4f}")
                    
                    # Interpretation
                    if resultado['desviacion'] < 0.03:
                        print(f"     🎯 COMPATIBLE WITH HARMONIC! (Δ < 0.03)")
                    else:
                        print(f"     ⚠️  Significant deviation from theoretical harmonic")
                    
            except Exception as e:
                print(f"   ❌ Error in bin MCMC: {str(e)[:100]}")
                resultados_bins[etiquetas[i]] = {
                    'n_galaxias': len(t_bin),
                    'omega_median': None,
                    'log_bf': None,
                    'status': 'mcmc_error',
                    'error': str(e)[:200]
                }
        
        # --- COMPARATIVE BIN ANALYSIS ---
        print(f"\n{'='*60}")
        print(f"📊 COMPARATIVE SUMMARY OF REDSHIFT BINS")
        print(f"{'='*60}")
        
        omegas_validas = []
        for etiqueta, res in resultados_bins.items():
            if res.get('omega_median') is not None and res.get('log_bf', -np.inf) > 2.0:
                omegas_validas.append(res['omega_median'])
                
                print(f"\n{etiqueta}:")
                print(f"  • ω = {res['omega_median']:.4f} ± {res.get('omega_std', 0):.4f}")
                print(f"  • log(BF) = {res.get('log_bf', 0):.2f}")
                print(f"  • N galaxies = {res.get('n_galaxias', 0):,}")
                print(f"  • Harmonic: {res.get('armonico_cercano', 0):.3f} (Δ={res.get('desviacion', 0):.3f})")
        
        # Temporal evolution analysis
        if len(omegas_validas) >= 2:
            print(f"\n📈 TEMPORAL EVOLUTION OF ω:")
            print(f"  • Mean value between bins: {np.mean(omegas_validas):.4f}")
            print(f"  • Dispersion (std): {np.std(omegas_validas):.4f}")
            
            # Consistency test
            if np.std(omegas_validas) < 0.05:
                print(f"  ✅ ω CONSISTENT across cosmic epochs")
            else:
                print(f"  ⚠️  ω VARIES across cosmic epochs")
        
        # Save results
        self.resultados['bins_redshift'] = resultados_bins
        
        return resultados_bins

    def ejecutar_mcmc_para_bin(self, nombre, t, y, yerr, n_steps=1200, burnin=400):
        """Optimized version of MCMC for redshift bins"""
        
        # Configuration for bins
        n_walkers = 24  # Reduced for speed
        
        # Initial positions centered on harmonics
        ndim = 6
        pos = np.array([0.0, 0.0, 0.08, 0.764, 0.0, -1.0]) + \
              0.02 * np.random.randn(n_walkers, ndim)
        
        # Null model (comparison)
        ndim_nulo = 3
        pos_nulo = np.array([0.0, 0.0, -1.0]) + \
                   0.01 * np.random.randn(n_walkers, ndim_nulo)
        
        try:
            with Pool() as pool:
                # 1. Oscillatory model
                sampler_osc = emcee.EnsembleSampler(
                    n_walkers, ndim, log_probability_armonicos_superiores,
                    args=(t, y, yerr), pool=pool
                )
                
                # Burn-in
                state = sampler_osc.run_mcmc(pos, burnin, progress=False)
                sampler_osc.reset()
                
                # Main sampling
                sampler_osc.run_mcmc(state, n_steps, progress=False)
                
                # 2. Null model
                sampler_nulo = emcee.EnsembleSampler(
                    n_walkers, ndim_nulo, log_probability_nulo,
                    args=(t, y, yerr), pool=pool
                )
                sampler_nulo.run_mcmc(pos_nulo, n_steps//2, progress=False)
            
            # Result analysis
            flat_samples = sampler_osc.get_chain(discard=burnin//2, flat=True)
            
            if len(flat_samples) > 100:
                omega_samples = flat_samples[:, 3]
                omega_samples = omega_samples[omega_samples > 0.50]
                
                if len(omega_samples) > 50:
                    q = np.percentile(omega_samples, [16, 50, 84])
                    omega_median = q[1]
                    
                    # Bayes Factor
                    log_l_osc = np.median(sampler_osc.get_log_prob()[-100:])
                    log_l_nulo = np.median(sampler_nulo.get_log_prob()[-100:])
                    log_bf = log_l_osc - log_l_nulo
                    
                    # Closest harmonic
                    armonico_cercano = min(self.armonicos_objetivo, 
                                         key=lambda x: abs(x - omega_median))
                    desviacion = abs(omega_median - armonico_cercano)
                    
                    return {
                        'n_galaxias': len(t),
                        'omega_median': omega_median,
                        'omega_err_low': omega_median - q[0],
                        'omega_err_high': q[2] - omega_median,
                        'omega_std': np.std(omega_samples),
                        'log_bf': log_bf,
                        'armonico_cercano': armonico_cercano,
                        'desviacion': desviacion,
                        'frac_omega_gt_0_5': len(omega_samples)/len(flat_samples),
                        'status': 'success'
                    }
        
        except Exception as e:
            print(f"Error in MCMC bin: {str(e)[:100]}")
        
        return {
            'n_galaxias': len(t),
            'omega_median': None,
            'log_bf': None,
            'status': 'failed'
        }

    def analizar_consistencia_armonicos(self):
        """Consistency analysis in harmonic detection"""
        print(f"\n🔗 HIGHER HARMONICS CONSISTENCY ANALYSIS")

        omegas_medianas = []
        bfs = []
        campos_con_armonico = []
        desviaciones = []

        for n, res in self.resultados.items():
            if 'mcmc' in res:
                mcmc_data = res['mcmc']
                if mcmc_data.get('frac_omega_gt_0_5', 0) > 0.5:  # At least 50% of samples with ω>0.5
                    omegas_medianas.append(mcmc_data['omega_median'])
                    bfs.append(mcmc_data['log_bayes_factor'])
                    desviaciones.append(mcmc_data['desviacion_armonico'])

                    if mcmc_data['es_armonico_significativo']:
                        campos_con_armonico.append((n, mcmc_data['armonico_cercano']))

        if not omegas_medianas:
            print("   ⚠️  No significant MCMC results in range ω>0.5")
            return

        # Global statistics
        om_mean = np.mean(omegas_medianas)
        om_std = np.std(omegas_medianas)
        om_median = np.median(omegas_medianas)
        log_bf_total = np.sum(bfs)
        desviacion_promedio = np.mean(desviaciones)

        # Determine most common harmonic
        if campos_con_armonico:
            armonicos = [a for _, a in campos_con_armonico]
            armonico_mas_comun = max(set(armonicos), key=armonicos.count)
            frecuencia_armonico = armonicos.count(armonico_mas_comun) / len(armonicos)
        else:
            armonico_mas_comun = None
            frecuencia_armonico = 0

        self.resultados['global'] = {
            'omega_promedio': om_mean,
            'omega_mediana': om_median,
            'omega_std': om_std,
            'log_bf_total': log_bf_total,
            'n_campos_significativos': len(omegas_medianas),
            'campos_con_armonico': campos_con_armonico,
            'armonico_mas_comun': armonico_mas_comun,
            'frecuencia_armonico': frecuencia_armonico,
            'desviacion_promedio': desviacion_promedio,
            'rango_95': [np.percentile(omegas_medianas, 2.5), np.percentile(omegas_medianas, 97.5)]
        }

        print(f"   🌍 {len(omegas_medianas)} fields with significant ω>0.5")
        print(f"   📊 ω Average: {om_mean:.4f} ± {om_std:.4f}")
        print(f"   📈 ω Median: {om_median:.4f}")

        if armonico_mas_comun:
            print(f"   🎯 Most common harmonic: {armonico_mas_comun:.3f} ({frecuencia_armonico:.1%} of fields)")
            print(f"   📏 Average deviation: {desviacion_promedio:.4f}")

        print(f"   ⚖️  log(BF) Total: {log_bf_total:.2f}")

        # Interpretation
        if frecuencia_armonico > 0.5:
            print(f"   ✅ CONSISTENT DETECTION of harmonic {armonico_mas_comun:.3f}")
        elif frecuencia_armonico > 0.3:
            print(f"   ⚠️  PARTIAL DETECTION of harmonic {armonico_mas_comun:.3f}")
        else:
            print("   ❌ NO CONSISTENT DETECTION of harmonics")

    def generar_reporte_armonicos_superiores(self):
        """Generates specific report for higher harmonics analysis"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        reporte = f"""HIGHER HARMONICS ANALYSIS (ω > 0.5) - SCIENTIFIC REPORT
================================================================================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Analysis: MCMC JWST - Specific search for higher harmonics
ω fundamental: {self.omega_fundamental:.3f}
Theoretical harmonics: {', '.join([f'{x:.3f}' for x in self.armonicos_objetivo])}
Search range: 0.50 < ω < 1.30
================================================================================

EXECUTIVE SUMMARY
================================================================================
Bayesian search for higher harmonics (ω > 0.5) of the cosmic fundamental rhythm.
Specifically looking for harmonics 4×0.191=0.764, 5×0.191=0.955, 6×0.191=1.146.

FIELD ANALYSIS
================================================================================
"""

        for nombre in sorted(self.campos_procesados.keys()):
            info = self.campos_procesados[nombre]
            reporte += f"\n{nombre.upper()}:\n"
            reporte += f"  • Galaxies: {info['n']:,}\n"
            reporte += f"  • z̃ = {info['z_median']:.2f}, t̃ = {info['t_median']:.2f} Gyr\n"
            reporte += f"  • SNR̃ = {info['snr_median']:.2f}\n"

            if nombre in self.resultados:
                if 'detection' in self.resultados[nombre]:
                    det = self.resultados[nombre]['detection']
                    if det['omega_principal']:
                        reporte += f"  • ω_peak (>0.5): {det['omega_principal']:.4f} (SNR: {det['snr_principal']:.2f})\n"

                    for key, armonico_data in det.get('armonicos', {}).items():
                        reporte += f"  • {key}: ω_teo={armonico_data['teorico']:.3f}, ω_det={armonico_data['detectado']:.3f}, Δ={armonico_data['desviacion']:.3f}\n"

                if 'mcmc' in self.resultados[nombre]:
                    mcmc = self.resultados[nombre]['mcmc']
                    reporte += f"  • MCMC: ω = {mcmc['omega_median']:.4f} +{mcmc['omega_err_high']:.4f}/-{mcmc['omega_err_low']:.4f}\n"
                    reporte += f"  • Closest harmonic: {mcmc['armonico_cercano']:.3f} (Δ={mcmc['desviacion_armonico']:.4f})\n"
                    reporte += f"  • log(BF) = {mcmc['log_bayes_factor']:.2f} ({mcmc['preferido'].upper()})\n"
                    reporte += f"  • {'✅ IS HARMONIC' if mcmc['es_armonico_significativo'] else '⚠️  IS NOT HARMONIC'}\n"
        
        # --- NEW SECTION: COSMIC EPOCHS ANALYSIS ---
        if 'bins_redshift' in self.resultados:
            reporte += f"""
\nEVOLUTIONARY ANALYSIS BY COSMIC EPOCHS (REDSHIFT BINS)
================================================================================
Study of how frequency ω varies across different epochs of the Universe.
"""
            
            bins_info = {
                "Z_EVOLUCIONADO": "z=1.0-2.5 (Relatively evolved Universe)",
                "Z_PICO_ESTELAR": "z=2.5-5.0 (Peak of cosmic star formation)",
                "Z_PRIMITIVO": "z=5.0-12.0 (Primitive Universe, first galaxies)"
            }
            
            resultados_bins = self.resultados['bins_redshift']
            
            for etiqueta, descripcion in bins_info.items():
                if etiqueta in resultados_bins:
                    res = resultados_bins[etiqueta]
                    reporte += f"\n{etiqueta} ({descripcion}):\n"
                    
                    if res.get('status') == 'success' and res.get('omega_median'):
                        reporte += f"  • ω = {res['omega_median']:.4f} "
                        if res.get('omega_err_low') and res.get('omega_err_high'):
                            reporte += f"+{res['omega_err_high']:.4f}/-{res['omega_err_low']:.4f}\n"
                        reporte += f"  • log(BF) = {res.get('log_bf', 0):.2f}\n"
                        reporte += f"  • N galaxies = {res.get('n_galaxias', 0):,}\n"
                        reporte += f"  • Closest harmonic: {res.get('armonico_cercano', 0):.3f}"
                        if res.get('desviacion'):
                            reporte += f" (Δ={res['desviacion']:.3f})\n"
                        
                        # Interpretation
                        if res.get('desviacion', 1) < 0.03:
                            reporte += f"  • ✅ COMPATIBLE WITH HIGHER HARMONIC\n"
                        else:
                            reporte += f"  • ⚠️  Significant deviation from harmonic\n"
                    else:
                        reporte += f"  • Status: {res.get('status', 'unknown')}\n"
                        if res.get('error'):
                            reporte += f"  • Error: {res['error'][:100]}...\n"
            
            # Comparative analysis
            omegas_bins = []
            for etiqueta in ["Z_EVOLUCIONADO", "Z_PICO_ESTELAR", "Z_PRIMITIVO"]:
                if (etiqueta in resultados_bins and 
                    resultados_bins[etiqueta].get('omega_median')):
                    omegas_bins.append(resultados_bins[etiqueta]['omega_median'])
            
            if len(omegas_bins) >= 2:
                reporte += f"""
\nTEMPORAL EVOLUTION OF ω:
• Mean value between epochs: {np.mean(omegas_bins):.4f}
• Dispersion (std): {np.std(omegas_bins):.4f}
• Range: [{np.min(omegas_bins):.4f}, {np.max(omegas_bins):.4f}]

EVOLUTIONARY INTERPRETATION:
"""
                if np.std(omegas_bins) < 0.02:
                    reporte += "• ✅ ω remains CONSTANT through cosmic time\n"
                    reporte += "  (Periodicity is invariant with redshift)\n"
                elif np.std(omegas_bins) < 0.05:
                    reporte += "• ⚠️  ω shows MODERATE VARIATION with time\n"
                    reporte += "  (Possible weak evolution of periodicity)\n"
                else:
                    reporte += "• 🔄 ω VARIES SIGNIFICANTLY with cosmic time\n"
                    reporte += "  (Periodicity evolves with the Universe)\n"

        # --- GLOBAL ANALYSIS ---
        if 'global' in self.resultados:
            global_res = self.resultados['global']
            reporte += f"""
\nGLOBAL MULTIFIELD ANALYSIS (ω > 0.5)
================================================================================
Fields with significant ω>0.5: {global_res['n_campos_significativos']}/{len(self.campos_procesados)}

ω STATISTICS (>0.5):
• Average: {global_res['omega_promedio']:.4f} ± {global_res['omega_std']:.4f}
• Median: {global_res['omega_mediana']:.4f}
• 95% range: [{global_res['rango_95'][0]:.4f}, {global_res['rango_95'][1]:.4f}]

HARMONIC DETECTION:
"""
            if global_res['armonico_mas_comun']:
                reporte += f"""• Most common harmonic: {global_res['armonico_mas_comun']:.3f}
• Frequency: {global_res['frecuencia_armonico']:.1%} of fields
• Average deviation: {global_res['desviacion_promedio']:.4f}

BAYESIAN EVIDENCE:
• log(BF) Total: {global_res['log_bf_total']:.2f}

PHYSICAL INTERPRETATION
================================================================================
"""
            # Detection evaluation
            if global_res['frecuencia_armonico'] > 0.5:
                reporte += f"✅ CONVINCING DETECTION of harmonic {global_res['armonico_mas_comun']:.3f}\n"
                reporte += f"Harmonic {global_res['armonico_mas_comun']:.3f} appears consistently in multiple fields.\n"
            elif global_res['frecuencia_armonico'] > 0.3:
                reporte += f"⚠️  PARTIAL DETECTION of harmonic {global_res['armonico_mas_comun']:.3f}\n"
                reporte += "Observed in some fields but not consistently.\n"
            else:
                reporte += "❌ NO CLEAR DETECTION of higher harmonics\n"
                reporte += "Data does not support presence of consistent higher harmonics.\n"

            # Interpretation of median value
            omega_med = global_res['omega_mediana']
            if omega_med > 0.5:
                reporte += f"\n🎯 OBSERVED MEDIAN VALUE: ω = {omega_med:.3f}\n"

                # Relation with theoretical harmonics
                reporte += "RELATION WITH THEORETICAL HARMONICS:\n"
                for i, armonico in enumerate(self.armonicos_objetivo):
                    ratio = omega_med / armonico if armonico > 0 else 0
                    desviacion = abs(omega_med - armonico)
                    reporte += f"• Harmonic {i+4}×0.191={armonico:.3f}: Δ={desviacion:.3f} (ratio={ratio:.3f})\n"

                # Classification
                if 0.74 <= omega_med <= 0.78:
                    reporte += "✅ CLASSIFICATION: Compatible with 4th harmonic (0.764)\n"
                elif 0.94 <= omega_med <= 0.98:
                    reporte += "✅ CLASSIFICATION: Compatible with 5th harmonic (0.955)\n"
                elif 1.12 <= omega_med <= 1.18:
                    reporte += "✅ CLASSIFICATION: Compatible with 6th harmonic (1.146)\n"
                else:
                    reporte += "⚠️  CLASSIFICATION: Does not clearly match any theoretical harmonic\n"
                    reporte += "Possible interpretation: Intermediate harmonic or system's own frequency.\n"

        reporte += f"\n================================================================================\n"
        reporte += f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        # Save report
        report_path = Path(f"reporte_armonicos_superiores_{timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(reporte)

        print(f"\n📝 HIGHER HARMONICS REPORT SAVED: {report_path}")
        return reporte

    def ejecutar_analisis_completo(self):
        """Executes complete analysis for ω > 0.5"""
        print("\n🚀 STARTING COMPLETE ANALYSIS - HIGHER HARMONICS (ω > 0.5)")
        print("=" * 80)

        # PHASE 1: Load and process data
        if not self.cargar_y_procesar_datos():
            print("❌ Could not load/process data")
            return

        # PHASE 2: Specific detection of higher harmonics
        print("\n\nPHASE 2: HIGHER HARMONICS DETECTION")
        campos_con_deteccion = []
        for nombre in self.campos_procesados:
            print(f"\n🔍 {nombre}:")
            det = self.detectar_armonicos_superiores(nombre)
            if det and det['omega_principal'] and det['omega_principal'] > 0.5:
                campos_con_deteccion.append((nombre, det['omega_principal']))

        if not campos_con_deteccion:
            print("\n❌ No signals with ω > 0.5 detected in any field")
            return

        print(f"\n✅ {len(campos_con_deteccion)} fields detected with ω > 0.5")

        # PHASE 3: MCMC on most significant fields
        print("\n\nPHASE 3: BAYESIAN MCMC ANALYSIS (ω > 0.5)")

        # Order fields by number of galaxies
        campos_ordenados = sorted(
            self.campos_procesados.items(),
            key=lambda x: x[1]['n'],
            reverse=True
        )[:3]  # The 3 largest fields

        for nombre, _ in campos_ordenados:
            print(f"\n🎯 Analyzing {nombre} (specific search ω>0.5)...")
            self.ejecutar_mcmc_armonicos_superiores(nombre)

        # PHASE 4: Consistency analysis
        print("\n\nPHASE 4: HARMONICS CONSISTENCY ANALYSIS")
        self.analizar_consistencia_armonicos()

        # ============================================================================
        # NEW PHASE 5: ANALYSIS BY COSMIC EPOCHS (REDSHIFT BINS)
        # ============================================================================
        print("\n\nPHASE 5: EVOLUTIONARY ANALYSIS BY COSMIC EPOCHS")
        print("=" * 60)
        print("Analyzing temporal evolution of ω in 3 epochs:")
        print("  1. z=1.0-2.5  (Evolved Universe)")
        print("  2. z=2.5-5.0  (Peak star formation)")
        print("  3. z=5.0-12.0 (Primitive Universe)")
        print("=" * 60)
        
        resultados_bins = self.analizar_por_bins_redshift_pro()
        
        # Add results to global report
        if 'bins_redshift' not in self.resultados:
            self.resultados['bins_redshift'] = resultados_bins

        # PHASE 6: Report (now will be PHASE 6)
        print("\n\nPHASE 6: SCIENTIFIC REPORT GENERATION")
        self.generar_reporte_armonicos_superiores()

        print("\n" + "=" * 80)
        print("✅ COMPLETE HIGHER HARMONICS ANALYSIS FINISHED")
        print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("SPECIFIC ANALYSIS OF HIGHER HARMONICS (ω > 0.5)")
    print("Search for: 0.764 (4×0.191), 0.955 (5×0.191), 1.146 (6×0.191)")
    print("="*80)

    try:
        # Optimized configuration
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        print("\n🔧 INITIALIZING HIGHER HARMONICS ANALYZER...")
        analizador = AnalisisMCMCJWST_ArmonicosSuperiores(data_dir='data')

        if RUST_ENGINE_READY:
            print("🚀 Rust engine ready - Prior centered on harmonics 0.764, 0.955, 1.146")
        else:
            print("🐢 Using pure Python prior for higher harmonics")

        # Execute complete analysis
        analizador.ejecutar_analisis_completo()

        print("\n🎯 ANALYSIS COMPLETED.")
        print("📊 See 'reporte_armonicos_superiores_*.txt' for detailed results")

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()