#!/usr/bin/env python3
"""
MCMC JWST - LOW RANGE ANALYSIS (0.0 < ω < 0.5)
DETECTION OF FUNDAMENTAL RHYTHM AND FIRST HARMONICS
SPECIFIC SEARCH: 0.0 < ω < 0.5 (FUNDAMENTAL 0.191, 2nd HARMONIC 0.382)
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
from astropy.cosmology import FlatLambdaCDM
from multiprocessing import Pool
warnings.filterwarnings('ignore')

import vpm_engine

try:
    import vpm_engine
    USAR_RUST = True
    RUST_ENGINE_READY = True
    print("🚀 Rust engine detected and linked.")
except ImportError:
    USAR_RUST = False
    RUST_ENGINE_READY = False
    print("⚠️  Warning: vpm_engine not found. Using slow Python.")

    
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
# 2. GLOBAL FUNCTIONS OPTIMIZED FOR ω < 0.5
# ============================================================================

def log_prior_fundamental_bajo(theta):
    """Optimized prior for fundamental rhythm (0.15 < ω < 0.45)"""
    # theta[3] is omega, theta[2] is Amplitude
    if USAR_RUST:
        return vpm_engine.log_prior_fundamental(float(theta[3]), float(theta[2]))
    
    # Manual fallback
    M0, alpha, A, omega, phi, log_sigma = theta
    if not (0.15 < omega < 0.45) or not (0.001 < A < 0.500): 
        return -np.inf
    
    # Gaussian prior centered at 0.191 with width 0.02
    return -0.5 * ((omega - 0.191) / 0.05)**2

def log_likelihood_con_rust_fundamental(theta, t, y, yerr):
    """Call to massive Likelihood in Rust for low frequencies"""
    if USAR_RUST:
        return vpm_engine.log_likelihood_fast(theta, t, y, yerr)
    
    # Python Fallback
    M0, alpha, A, omega, phi, log_sigma = theta
    y_pred = M0 - alpha * t + A * np.cos(omega * t + phi)
    sigma2 = np.exp(2 * log_sigma) + yerr**2
    return -0.5 * np.sum((y - y_pred)**2 / sigma2 + np.log(2 * np.pi * sigma2))

def log_probability_fundamental_bajo(theta, t, y, yerr):
    """Total probability function for fundamental rhythm"""
    lp = log_prior_fundamental_bajo(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_con_rust_fundamental(theta, t, y, yerr)

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
# MAIN CLASS - SPECIFIC ANALYSIS FOR ω < 0.5
# ============================================================================

class AnalisisMCMCJWST_FundamentalBajo:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)

        # Theoretical fundamental rhythm
        self.omega_fundamental = 0.191
        
        # Target frequencies in low range
        self.frecuencias_objetivo = [0.191, 0.382]  # Fundamental and 2nd harmonic
        
        # Add 0.261 based on ω>0.5 results (0.522/2)
        self.frecuencias_especiales = [0.261, 0.301]  # Possible sub-harmonics

        # Magnification factor for A2744
        self.mu_a2744 = 1.5

        # Initialize loader
        self.cargador = CargadorDatosJWST_Corregido(data_dir)
        self.campos_procesados = {}
        self.resultados = {}

        # Optimized MCMC configuration for low frequencies
        self.n_walkers = 36  # More walkers for better sampling
        self.n_steps = 2200  # More steps for precision in low frequencies
        self.burnin = 800    # More burnin

        print("=" * 80)
        print("MCMC JWST - FUNDAMENTAL RHYTHM ANALYSIS (ω < 0.5)")
        print(f"Theoretical ω fundamental: {self.omega_fundamental:.3f}")
        print(f"Theoretical 2nd harmonic: {2*self.omega_fundamental:.3f}")
        print(f"Search range: 0.15 < ω < 0.45")
        print("=" * 80)

    def cargar_y_procesar_datos(self):
        """Loads and processes all data"""
        print("\n📥 LOADING DATA...")

        if not self.cargador.cargar_todo():
            print("❌ Error loading data")
            return False

        if len(self.cargador.campos) == 0:
            print("❌ No data loaded")
            return False

        print("\n⚙️  PROCESSING ALL FIELDS...")
        for nombre, df in self.cargador.campos.items():
            if len(df) >= 100:
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

            # 1. Apparent magnitude
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

            # 6. Extraction of residuals
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

    def detectar_frecuencias_bajas(self, nombre):
        """Specific detection in range ω < 0.5"""
        if nombre not in self.campos_procesados:
            return None

        df = self.campos_procesados[nombre]['data']

        mask = (df['snr'] > 1.5)
        t, y, w = df['t'][mask].values, df['residuos'][mask].values, df['peso'][mask].values

        if len(t) < 20:
            print(f"   ⚠️  {nombre}: Insufficient data for detection")
            return None

        # SPECIFIC range for low frequencies
        omegas = np.linspace(0.15, 0.45, 600)  # More points for better resolution
        potencias = []
        for om in omegas:
            c, s = np.cos(om * t), np.sin(om * t)
            p = (np.sum(w * y * c)**2 / np.sum(w * c**2)) + (np.sum(w * y * s)**2 / np.sum(w * s**2))
            potencias.append(p)

        potencias_arr = np.array(potencias)

        resultados_frecuencias = {}

        # Specifically search near target frequencies
        todas_frecuencias = self.frecuencias_objetivo + self.frecuencias_especiales
        
        for i, freq in enumerate(todas_frecuencias):
            ventana = 0.02  # Narrower window for precision
            idx_min = int((freq - ventana - 0.15) / (0.45 - 0.15) * 600)
            idx_max = int((freq + ventana - 0.15) / (0.45 - 0.15) * 600)
            idx_min = max(0, idx_min)
            idx_max = min(599, idx_max)

            if idx_max > idx_min:
                sub_potencias = potencias_arr[idx_min:idx_max]
                sub_omegas = omegas[idx_min:idx_max]

                if len(sub_potencias) > 0:
                    idx_pico_local = np.argmax(sub_potencias)
                    omega_pico = sub_omegas[idx_pico_local]
                    potencia_pico = sub_potencias[idx_pico_local]
                    snr_local = potencia_pico / np.mean(potencias_arr)

                    etiqueta = f'freq_{i+1}'
                    if i < len(self.frecuencias_objetivo):
                        etiqueta = ['fundamental', '2do_armonico'][i]
                    else:
                        etiqueta = f'especial_{i-1}'
                    
                    resultados_frecuencias[etiqueta] = {
                        'teorico': freq,
                        'detectado': omega_pico,
                        'potencia': potencia_pico,
                        'snr': snr_local,
                        'desviacion': abs(omega_pico - freq)
                    }

        # Find global peaks
        peaks, properties = find_peaks(potencias_arr, height=np.mean(potencias_arr)*1.2, distance=40)

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
            'frecuencias': resultados_frecuencias,
            'omegas_detectados': omegas[peaks][:5] if len(peaks) > 0 else [],
            'potencias_detectadas': potencias_arr[peaks][:5] if len(peaks) > 0 else []
        }

        self.resultados.setdefault(nombre, {})['detection'] = res

        # Show results
        if omega_principal and omega_principal < 0.45:
            print(f"   📊 ω_peak principal (<0.5): {omega_principal:.4f} (SNR: {snr_principal:.2f})")
        elif omega_principal:
            print(f"   📊 ω_peak principal: {omega_principal:.4f} (out of low range)")
        else:
            print(f"   📊 ω_peak principal (<0.5): N/A")

        for key, freq_data in resultados_frecuencias.items():
            print(f"   🎯 {key}: ω_teo={freq_data['teorico']:.3f}, ω_det={freq_data['detectado']:.3f}, "
                  f"Δ={freq_data['desviacion']:.3f}, SNR={freq_data['snr']:.2f}")

        return res

    def ejecutar_mcmc_fundamental_bajo(self, nombre):
        """MCMC for low frequencies (<0.5)"""
        if nombre not in self.campos_procesados:
            return

        df = self.campos_procesados[nombre]['data']
        mask = (df['snr'] > 2.0)
        t, y, yerr = df['t'][mask].values, df['residuos'][mask].values, df['residuos_err'][mask].values

        # Downsampling for speed
        n_max = 15000  # Fewer galaxies for low frequencies (faster)
        if len(t) > n_max:
            print(f"   📉 Reducing sample from {len(t):,} to {n_max:,} to speed up MCMC...")
            idx = np.random.choice(len(t), n_max, replace=False)
            t, y, yerr = t[idx], y[idx], yerr[idx]

        if len(t) < 50:
            print(f"   ⚠️  {nombre}: Insufficient data (N={len(t)})")
            return

        print(f"\n🔬 LOW FUNDAMENTAL MCMC: {nombre} (N={len(t)})")
        
        try:
            ndim, ndim_nulo = 6, 3
            pos = np.array([0.0, 0.0, 0.08, 0.191, 0.0, -1.0]) + \
                  0.03 * np.random.randn(self.n_walkers, ndim)
            pos_nulo = np.array([0.0, 0.0, -1.0]) + \
                       0.02 * np.random.randn(self.n_walkers, ndim_nulo)

            with Pool() as pool:
                sampler_osc = emcee.EnsembleSampler(
                    self.n_walkers, ndim, log_probability_fundamental_bajo,
                    args=(t, y, yerr), pool=pool
                )
                print(f"   🔄 Burn-in (Rust+Parallel)...")
                state = sampler_osc.run_mcmc(pos, self.burnin, progress=True)
                sampler_osc.reset()
                print(f"   🔄 Sampling (Rust+Parallel)...")
                sampler_osc.run_mcmc(state, self.n_steps, progress=True)

                print(f"   🔄 Calculating Null Model...")
                sampler_nulo = emcee.EnsembleSampler(
                    self.n_walkers, ndim_nulo, log_probability_nulo,
                    args=(t, y, yerr), pool=pool
                )
                sampler_nulo.run_mcmc(pos_nulo, self.n_steps // 2, progress=True)

            # Bayes Factor
            log_l_osc = np.median(sampler_osc.get_log_prob()[-200:])
            log_l_nulo = np.median(sampler_nulo.get_log_prob()[-200:])
            log_bf = log_l_osc - log_l_nulo

            # Sample analysis
            flat_samples = sampler_osc.get_chain(discard=self.burnin//2, flat=True)
            omega_samples = flat_samples[:, 3]
            
            # Filter for ω < 0.5
            omega_samples = omega_samples[omega_samples < 0.50]

            if len(omega_samples) > 50:
                q = np.percentile(omega_samples, [16, 50, 84])
                omega_median = q[1]
                
                # Find closest target frequency
                todas_frecuencias = self.frecuencias_objetivo + self.frecuencias_especiales
                frecuencia_cercana = min(todas_frecuencias, key=lambda x: abs(x - omega_median))
                desviacion = abs(omega_median - frecuencia_cercana)
                
                # Determine what type of frequency it is
                if frecuencia_cercana in self.frecuencias_objetivo:
                    tipo = 'objetivo'
                    idx = self.frecuencias_objetivo.index(frecuencia_cercana)
                    nombre_frec = ['fundamental', '2do_armonico'][idx]
                else:
                    tipo = 'especial'
                    nombre_frec = f'especial_{frecuencia_cercana:.3f}'

                self.resultados[nombre]['mcmc'] = {
                    'log_bayes_factor': log_bf,
                    'omega_median': omega_median,
                    'omega_err_low': omega_median - q[0],
                    'omega_err_high': q[2] - omega_median,
                    'omega_std': np.std(omega_samples),
                    'frecuencia_cercana': frecuencia_cercana,
                    'nombre_frecuencia': nombre_frec,
                    'tipo_frecuencia': tipo,
                    'desviacion': desviacion,
                    'es_significativo': desviacion < 0.015,  # Stricter threshold
                    'preferido': 'oscilatorio' if log_bf > 2.0 else 'nulo',
                    'frac_omega_lt_0_5': len(omega_samples) / len(flat_samples)
                }
                
                print(f"   ✅ log(BF): {log_bf:.2f} | ω: {omega_median:.4f}")
                print(f"   🎯 Closest frequency: {nombre_frec} ({frecuencia_cercana:.3f}, Δ={desviacion:.4f})")
                
                if desviacion < 0.015:
                    print(f"   ✅ COMPATIBLE WITH {nombre_frec.upper()}!")
                else:
                    print(f"   ⚠️  Significant deviation")
            
        except Exception as e:
            print(f"   ❌ Error in MCMC {nombre}: {e}")

    def analizar_por_bins_redshift_fundamental(self):
        """Analysis by redshift bins for low frequencies"""
        
        bins = [(1.0, 2.5), (2.5, 5.0), (5.0, 12.0)]
        etiquetas = ["Z_EVOLUCIONADO", "Z_PICO_ESTELAR", "Z_PRIMITIVO"]
        
        listas_datos = []
        for campo, info in self.campos_procesados.items():
            df_campo = info['data'].copy()
            df_campo['campo'] = campo
            listas_datos.append(df_campo)
        
        df_total = pd.concat(listas_datos, ignore_index=True)
        print(f"\n📊 TOTAL COMBINED DATA: {len(df_total):,} galaxies")
        
        resultados_bins = {}
        
        for i, (z_min, z_max) in enumerate(bins):
            print(f"\n{'='*60}")
            print(f"🔥 BIN {i+1}: {etiquetas[i]} (z = {z_min}-{z_max}) - ω < 0.5")
            print(f"{'='*60}")
            
            # Quality filters
            mask_basico = (
                (df_total['zphot'] >= z_min) & 
                (df_total['zphot'] < z_max) & 
                (df_total['snr'] > 1.8) &
                (df_total['F444W'] > 0.01) &
                (df_total['F444W_err'] / df_total['F444W'] < 1.0)
            )
            df_filtrado = df_total[mask_basico].copy()
            print(f"   ✅ Basic filter: {len(df_filtrado):,} galaxies")
            
            # Malmquist correction
            z_medio = (z_min + z_max) / 2
            mag_limite = 29.0
            dL_limite = (3000/70.0) * z_medio * (1 + 0.5 * z_medio) * 1e6
            M_limite = mag_limite - 5 * np.log10(dL_limite) - 5
            
            mask_malmquist = df_filtrado['M'] < M_limite
            df_filtrado = df_filtrado[mask_malmquist].copy()
            print(f"   ✅ Malmquist (M < {M_limite:.1f}): {len(df_filtrado):,}")
            
            # Remove outliers
            if len(df_filtrado) > 100:
                residuos_mean = df_filtrado['residuos'].mean()
                residuos_std = df_filtrado['residuos'].std()
                mask_outliers = (
                    abs(df_filtrado['residuos'] - residuos_mean) < 4.0 * residuos_std
                )
                df_filtrado = df_filtrado[mask_outliers].copy()
                print(f"   ✅ No outliers >4σ: {len(df_filtrado):,}")
            
            t_bin = df_filtrado['t'].values
            y_bin = df_filtrado['residuos'].values
            yerr_bin = df_filtrado['residuos_err'].values
            
            if len(t_bin) < 100:
                print(f"   ⚠️  Insufficient bin: {len(t_bin)} galaxies")
                resultados_bins[etiquetas[i]] = {
                    'n_galaxias': len(t_bin),
                    'omega_median': None,
                    'status': 'insufficient_data'
                }
                continue
            
            # Subsampling
            n_max = 5000
            if len(t_bin) > n_max:
                print(f"   📉 Subsampling from {len(t_bin):,} to {n_max:,}...")
                idx = np.random.choice(len(t_bin), n_max, replace=False)
                t_bin, y_bin, yerr_bin = t_bin[idx], y_bin[idx], yerr_bin[idx]
            
            print(f"   📊 Final sample: {len(t_bin):,} galaxies")
            
            # Quick analysis
            print(f"\n   🔍 Quick analysis:")
            print(f"     • Mean z: {df_filtrado['zphot'].mean():.2f}")
            print(f"     • Mean t: {df_filtrado['t'].mean():.2f} Gyr")
            print(f"     • Mean SNR: {df_filtrado['snr'].mean():.2f}")
            
            # Fast periodogram for low frequencies
            from scipy.signal import periodogram
            freqs, psd = periodogram(y_bin, fs=1.0/np.std(t_bin))
            mask_freq = (freqs >= 0.15) & (freqs <= 0.45)
            if np.any(mask_freq):
                freq_max = freqs[mask_freq][np.argmax(psd[mask_freq])]
                print(f"     • Dominant frequency (0.15-0.45): {freq_max:.4f}")
            
            # MCMC for the bin
            print(f"\n   🔬 RUNNING MCMC FOR BIN...")
            
            try:
                resultado = self.ejecutar_mcmc_para_bin_fundamental(
                    nombre=etiquetas[i],
                    t=t_bin,
                    y=y_bin,
                    yerr=yerr_bin
                )
                
                resultados_bins[etiquetas[i]] = resultado
                
                if resultado['omega_median'] is not None:
                    print(f"\n   ✅ BIN RESULT {etiquetas[i]}:")
                    print(f"     • ω median: {resultado['omega_median']:.4f}")
                    print(f"     • log(BF): {resultado['log_bf']:.2f}")
                    print(f"     • Closest frequency: {resultado['nombre_frecuencia']}")
                    print(f"     • Deviation: {resultado['desviacion']:.4f}")
                    
                    if resultado['desviacion'] < 0.015:
                        print(f"     🎯 COMPATIBLE WITH {resultado['nombre_frecuencia'].upper()}!")
                    else:
                        print(f"     ⚠️  Significant deviation")
                    
            except Exception as e:
                print(f"   ❌ Error in bin MCMC: {str(e)[:100]}")
                resultados_bins[etiquetas[i]] = {
                    'n_galaxias': len(t_bin),
                    'omega_median': None,
                    'status': 'mcmc_error'
                }
        
        # Comparative analysis
        print(f"\n{'='*60}")
        print(f"📊 LOW FREQUENCIES SUMMARY BY EPOCH")
        print(f"{'='*60}")
        
        for etiqueta, res in resultados_bins.items():
            if res.get('omega_median'):
                print(f"\n{etiqueta}:")
                print(f"  • ω = {res['omega_median']:.4f} ± {res.get('omega_std', 0):.4f}")
                print(f"  • log(BF) = {res.get('log_bf', 0):.2f}")
                print(f"  • Frequency: {res.get('nombre_frecuencia', 'N/A')}")
                print(f"  • Type: {res.get('tipo_frecuencia', 'N/A')}")
                print(f"  • N galaxies = {res.get('n_galaxias', 0):,}")
        
        # Temporal evolution
        omegas_validas = []
        for etiqueta, res in resultados_bins.items():
            if res.get('omega_median') and res.get('log_bf', -np.inf) > 2.0:
                omegas_validas.append(res['omega_median'])
        
        if len(omegas_validas) >= 2:
            print(f"\n📈 TEMPORAL EVOLUTION OF ω (<0.5):")
            print(f"  • Mean value: {np.mean(omegas_validas):.4f}")
            print(f"  • Dispersion: {np.std(omegas_validas):.4f}")
            
            if np.std(omegas_validas) < 0.02:
                print(f"  ✅ LOW ω CONSTANT across epochs")
            elif np.std(omegas_validas) < 0.05:
                print(f"  ⚠️  LOW ω MODERATELY VARIES")
            else:
                print(f"  🔄 LOW ω VARIES SIGNIFICANTLY")
        
        self.resultados['bins_redshift_fundamental'] = resultados_bins
        return resultados_bins

    def ejecutar_mcmc_para_bin_fundamental(self, nombre, t, y, yerr):
        """MCMC for redshift bin (low frequencies)"""
        n_walkers_bin = 28
        n_steps_bin = 1600
        burnin_bin = 600
        
        ndim = 6
        pos = np.array([0.0, 0.0, 0.06, 0.191, 0.0, -1.0]) + \
              0.02 * np.random.randn(n_walkers_bin, ndim)
        
        ndim_nulo = 3
        pos_nulo = np.array([0.0, 0.0, -1.0]) + \
                   0.01 * np.random.randn(n_walkers_bin, ndim_nulo)
        
        try:
            with Pool() as pool:
                sampler_osc = emcee.EnsembleSampler(
                    n_walkers_bin, ndim, log_probability_fundamental_bajo,
                    args=(t, y, yerr), pool=pool
                )
                
                state = sampler_osc.run_mcmc(pos, burnin_bin, progress=False)
                sampler_osc.reset()
                sampler_osc.run_mcmc(state, n_steps_bin, progress=False)
                
                sampler_nulo = emcee.EnsembleSampler(
                    n_walkers_bin, ndim_nulo, log_probability_nulo,
                    args=(t, y, yerr), pool=pool
                )
                sampler_nulo.run_mcmc(pos_nulo, n_steps_bin//2, progress=False)
            
            flat_samples = sampler_osc.get_chain(discard=burnin_bin//2, flat=True)
            
            if len(flat_samples) > 100:
                omega_samples = flat_samples[:, 3]
                omega_samples = omega_samples[omega_samples < 0.50]
                
                if len(omega_samples) > 50:
                    q = np.percentile(omega_samples, [16, 50, 84])
                    omega_median = q[1]
                    
                    log_l_osc = np.median(sampler_osc.get_log_prob()[-100:])
                    log_l_nulo = np.median(sampler_nulo.get_log_prob()[-100:])
                    log_bf = log_l_osc - log_l_nulo
                    
                    todas_frecuencias = self.frecuencias_objetivo + self.frecuencias_especiales
                    frecuencia_cercana = min(todas_frecuencias, key=lambda x: abs(x - omega_median))
                    desviacion = abs(omega_median - frecuencia_cercana)
                    
                    if frecuencia_cercana in self.frecuencias_objetivo:
                        idx = self.frecuencias_objetivo.index(frecuencia_cercana)
                        nombre_frec = ['fundamental', '2do_armonico'][idx]
                        tipo = 'objetivo'
                    else:
                        nombre_frec = f'especial_{frecuencia_cercana:.3f}'
                        tipo = 'especial'
                    
                    return {
                        'n_galaxias': len(t),
                        'omega_median': omega_median,
                        'omega_err_low': omega_median - q[0],
                        'omega_err_high': q[2] - omega_median,
                        'omega_std': np.std(omega_samples),
                        'log_bf': log_bf,
                        'frecuencia_cercana': frecuencia_cercana,
                        'nombre_frecuencia': nombre_frec,
                        'tipo_frecuencia': tipo,
                        'desviacion': desviacion,
                        'status': 'success'
                    }
        
        except Exception as e:
            print(f"Error in MCMC bin {nombre}: {str(e)[:100]}")
        
        return {
            'n_galaxias': len(t),
            'omega_median': None,
            'status': 'failed'
        }

    def analizar_consistencia_fundamental(self):
        """Consistency analysis for low frequencies"""
        print(f"\n🔗 FUNDAMENTAL RHYTHM CONSISTENCY ANALYSIS")

        omegas_medianas = []
        bfs = []
        campos_con_fundamental = []
        campos_con_2do_armonico = []
        campos_con_especial = []

        for n, res in self.resultados.items():
            if n == 'global' or n == 'bins_redshift_fundamental':
                continue
                
            if 'mcmc' in res:
                mcmc_data = res['mcmc']
                if mcmc_data.get('frac_omega_lt_0_5', 0) > 0.3:
                    omegas_medianas.append(mcmc_data['omega_median'])
                    bfs.append(mcmc_data['log_bayes_factor'])
                    
                    tipo = mcmc_data.get('tipo_frecuencia', '')
                    if tipo == 'objetivo':
                        nombre_frec = mcmc_data.get('nombre_frecuencia', '')
                        if 'fundamental' in nombre_frec:
                            campos_con_fundamental.append((n, mcmc_data['omega_median']))
                        elif '2do_armonico' in nombre_frec:
                            campos_con_2do_armonico.append((n, mcmc_data['omega_median']))
                    elif tipo == 'especial':
                        campos_con_especial.append((n, mcmc_data['omega_median']))

        if not omegas_medianas:
            print("   ⚠️  No significant MCMC results in range ω<0.5")
            return

        om_mean = np.mean(omegas_medianas)
        om_std = np.std(omegas_medianas)
        om_median = np.median(omegas_medianas)
        log_bf_total = np.sum(bfs)

        self.resultados['global_fundamental'] = {
            'omega_promedio': om_mean,
            'omega_mediana': om_median,
            'omega_std': om_std,
            'log_bf_total': log_bf_total,
            'n_campos_significativos': len(omegas_medianas),
            'campos_con_fundamental': campos_con_fundamental,
            'campos_con_2do_armonico': campos_con_2do_armonico,
            'campos_con_especial': campos_con_especial,
            'rango_95': [np.percentile(omegas_medianas, 2.5), np.percentile(omegas_medianas, 97.5)] if omegas_medianas else [0, 0]
        }

        print(f"   🌍 {len(omegas_medianas)} fields with significant ω<0.5")
        print(f"   📊 ω Average: {om_mean:.4f} ± {om_std:.4f}")
        print(f"   📈 ω Median: {om_median:.4f}")
        print(f"   ⚖️  log(BF) Total: {log_bf_total:.2f}")
        
        if campos_con_fundamental:
            print(f"   🎯 Fields with fundamental rhythm (~0.191): {len(campos_con_fundamental)}")
            for campo, omega in campos_con_fundamental[:3]:  # Show first 3
                print(f"      • {campo}: ω={omega:.4f}")
        
        if campos_con_2do_armonico:
            print(f"   🎯 Fields with 2nd harmonic (~0.382): {len(campos_con_2do_armonico)}")
        
        if campos_con_especial:
            print(f"   🔍 Fields with special frequencies: {len(campos_con_especial)}")
            especiales_unicas = {}
            for campo, omega in campos_con_especial:
                key = f"{omega:.3f}"
                if key not in especiales_unicas:
                    especiales_unicas[key] = []
                especiales_unicas[key].append(campo)
            
            for freq, campos in list(especiales_unicas.items())[:3]:
                print(f"      • ω={freq}: {len(campos)} fields")

        # Relation with ω>0.5 results (if available)
        if hasattr(self, 'resultados_alto'):
            print(f"\n   🔗 COMPARISON WITH HIGHER HARMONICS:")
            # Here comparison with ω>0.5 results could be added

    def generar_reporte_fundamental_bajo(self):
        """Generates specific report for fundamental rhythm analysis"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        reporte = f"""FUNDAMENTAL RHYTHM ANALYSIS (ω < 0.5) - SCIENTIFIC REPORT
================================================================================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Analysis: MCMC JWST - Specific search for fundamental rhythm
Theoretical ω fundamental: {self.omega_fundamental:.3f}
Theoretical 2nd harmonic: {2*self.omega_fundamental:.3f}
Special frequencies searched: 0.261, 0.301
Search range: 0.15 < ω < 0.45
================================================================================

EXECUTIVE SUMMARY
================================================================================
Bayesian search for cosmic fundamental rhythm and its first harmonics.
Based on previous results of higher harmonics (ω=0.577, 0.769, 0.522).
================================================================================

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
                    if det['omega_principal'] and det['omega_principal'] < 0.45:
                        reporte += f"  • ω_peak (<0.5): {det['omega_principal']:.4f} (SNR: {det['snr_principal']:.2f})\n"

                    for key, freq_data in det.get('frecuencias', {}).items():
                        reporte += f"  • {key}: ω_teo={freq_data['teorico']:.3f}, ω_det={freq_data['detectado']:.3f}, Δ={freq_data['desviacion']:.3f}\n"

                if 'mcmc' in self.resultados[nombre]:
                    mcmc = self.resultados[nombre]['mcmc']
                    reporte += f"  • MCMC: ω = {mcmc['omega_median']:.4f} +{mcmc['omega_err_high']:.4f}/-{mcmc['omega_err_low']:.4f}\n"
                    reporte += f"  • Frequency: {mcmc.get('nombre_frecuencia', 'N/A')} ({mcmc.get('frecuencia_cercana', 0):.3f})\n"
                    reporte += f"  • log(BF) = {mcmc['log_bayes_factor']:.2f}\n"
                    reporte += f"  • Type: {mcmc.get('tipo_frecuencia', 'N/A').upper()}\n"
        
        # Analysis by redshift bins
        if 'bins_redshift_fundamental' in self.resultados:
            reporte += f"""
\nEVOLUTIONARY ANALYSIS BY COSMIC EPOCHS (ω < 0.5)
================================================================================
Study of how fundamental rhythm varies across different epochs.
"""
            
            bins_info = {
                "Z_EVOLUCIONADO": "z=1.0-2.5 (Relatively evolved Universe)",
                "Z_PICO_ESTELAR": "z=2.5-5.0 (Peak of cosmic star formation)",
                "Z_PRIMITIVO": "z=5.0-12.0 (Primitive Universe, first galaxies)"
            }
            
            resultados_bins = self.resultados['bins_redshift_fundamental']
            
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
                        reporte += f"  • Frequency: {res.get('nombre_frecuencia', 'N/A')}"
                        if res.get('desviacion'):
                            reporte += f" (Δ={res['desviacion']:.3f})\n"
                        
                        if res.get('desviacion', 1) < 0.015:
                            reporte += f"  • ✅ COMPATIBLE WITH EXPECTED FREQUENCY\n"
                        else:
                            reporte += f"  • ⚠️  Significant deviation\n"
                    else:
                        reporte += f"  • Status: {res.get('status', 'unknown')}\n"
            
            # Comparative analysis
            omegas_bins = []
            tipos_bins = []
            for etiqueta in ["Z_EVOLUCIONADO", "Z_PICO_ESTELAR", "Z_PRIMITIVO"]:
                if (etiqueta in resultados_bins and 
                    resultados_bins[etiqueta].get('omega_median')):
                    omegas_bins.append(resultados_bins[etiqueta]['omega_median'])
                    tipos_bins.append(resultados_bins[etiqueta].get('nombre_frecuencia', 'unknown'))
            
            if len(omegas_bins) >= 2:
                reporte += f"""
\nTEMPORAL EVOLUTION OF FUNDAMENTAL RHYTHM:
• Mean value: {np.mean(omegas_bins):.4f}
• Dispersion: {np.std(omegas_bins):.4f}
• Types found: {', '.join(tipos_bins)}

EVOLUTIONARY INTERPRETATION:
"""
                if np.std(omegas_bins) < 0.02:
                    reporte += "• ✅ FUNDAMENTAL ω CONSTANT through cosmic time\n"
                elif np.std(omegas_bins) < 0.05:
                    reporte += "• ⚠️  FUNDAMENTAL ω MODERATELY VARIES\n"
                else:
                    reporte += "• 🔄 FUNDAMENTAL ω VARIES SIGNIFICANTLY\n"
                
                # Relation with theoretical frequencies
                for i, omega in enumerate(omegas_bins):
                    relacion_191 = omega / 0.191
                    relacion_382 = omega / 0.382
                    reporte += f"• Bin {i+1} (ω={omega:.4f}): {relacion_191:.2f}×0.191, {relacion_382:.2f}×0.382\n"

        # Global analysis
        if 'global_fundamental' in self.resultados:
            global_res = self.resultados['global_fundamental']
            reporte += f"""
\nGLOBAL MULTIFIELD ANALYSIS (ω < 0.5)
================================================================================
Fields with significant ω<0.5: {global_res['n_campos_significativos']}/{len(self.campos_procesados)}

STATISTICS:
• Average: {global_res['omega_promedio']:.4f} ± {global_res['omega_std']:.4f}
• Median: {global_res['omega_mediana']:.4f}
• 95% range: [{global_res['rango_95'][0]:.4f}, {global_res['rango_95'][1]:.4f}]

FREQUENCY DISTRIBUTION:
• Fundamental rhythm (0.191): {len(global_res['campos_con_fundamental'])} fields
• 2nd harmonic (0.382): {len(global_res['campos_con_2do_armonico'])} fields
• Special frequencies: {len(global_res['campos_con_especial'])} fields

TOTAL BAYESIAN EVIDENCE:
• log(BF) Total: {global_res['log_bf_total']:.2f}

PHYSICAL INTERPRETATION
================================================================================
"""
            # Evaluation of fundamental rhythm
            if len(global_res['campos_con_fundamental']) >= 2:
                reporte += f"✅ DETECTION OF FUNDAMENTAL RHYTHM ω≈0.191\n"
                omegas_fund = [omega for _, omega in global_res['campos_con_fundamental']]
                reporte += f"  • Mean value: {np.mean(omegas_fund):.4f} ± {np.std(omegas_fund):.4f}\n"
                reporte += f"  • Accuracy: {100*abs(1-np.mean(omegas_fund)/0.191):.1f}% of theoretical value\n"
            else:
                reporte += "⚠️  WEAK DETECTION OF FUNDAMENTAL RHYTHM\n"
            
            # Relation with higher harmonics
            reporte += f"\nRELATION WITH ω>0.5 RESULTS:\n"
            reporte += f"• If ω_fund=0.191, then:\n"
            reporte += f"  - 3×0.191 = 0.573 (detected in Z_PRIMITIVO)\n"
            reporte += f"  - 4×0.191 = 0.764 (detected in Z_EVOLUCIONADO)\n"
            reporte += f"• ω=0.522 (Z_PICO_ESTELAR) ≈ 2×0.261 (special sub-harmonic)\n"

        reporte += f"\n================================================================================\n"
        reporte += f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        report_path = Path(f"reporte_fundamental_bajo_{timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(reporte)

        print(f"\n📝 FUNDAMENTAL RHYTHM REPORT SAVED: {report_path}")
        return reporte

    def ejecutar_analisis_completo(self):
        """Executes complete analysis for ω < 0.5"""
        print("\n🚀 STARTING COMPLETE ANALYSIS - FUNDAMENTAL RHYTHM (ω < 0.5)")
        print("=" * 80)

        # PHASE 1: Load and process data
        if not self.cargar_y_procesar_datos():
            print("❌ Could not load/process data")
            return

        # PHASE 2: Low frequency detection
        print("\n\nPHASE 2: LOW FREQUENCY DETECTION (<0.5)")
        campos_con_deteccion = []
        for nombre in self.campos_procesados:
            print(f"\n🔍 {nombre}:")
            det = self.detectar_frecuencias_bajas(nombre)
            if det and det['omega_principal'] and det['omega_principal'] < 0.45:
                campos_con_deteccion.append((nombre, det['omega_principal']))

        if not campos_con_deteccion:
            print("\n❌ No signals with ω < 0.45 detected in any field")
        else:
            print(f"\n✅ {len(campos_con_deteccion)} fields detected with ω < 0.45")

        # PHASE 3: MCMC on main fields
        print("\n\nPHASE 3: BAYESIAN MCMC ANALYSIS (ω < 0.5)")

        campos_ordenados = sorted(
            self.campos_procesados.items(),
            key=lambda x: x[1]['n'],
            reverse=True
        )[:3]

        for nombre, _ in campos_ordenados:
            print(f"\n🎯 Analyzing {nombre} (specific search ω<0.5)...")
            self.ejecutar_mcmc_fundamental_bajo(nombre)

        # PHASE 4: Consistency analysis
        print("\n\nPHASE 4: CONSISTENCY ANALYSIS")
        self.analizar_consistencia_fundamental()

        # PHASE 5: Analysis by cosmic epochs
        print("\n\nPHASE 5: EVOLUTIONARY ANALYSIS BY COSMIC EPOCHS")
        print("=" * 60)
        print("Analyzing evolution of fundamental rhythm in 3 epochs:")
        print("  1. z=1.0-2.5  (Evolved Universe)")
        print("  2. z=2.5-5.0  (Peak star formation)")
        print("  3. z=5.0-12.0 (Primitive Universe)")
        print("=" * 60)
        
        resultados_bins = self.analizar_por_bins_redshift_fundamental()
        if 'bins_redshift_fundamental' not in self.resultados:
            self.resultados['bins_redshift_fundamental'] = resultados_bins

        # PHASE 6: Report
        print("\n\nPHASE 6: SCIENTIFIC REPORT GENERATION")
        self.generar_reporte_fundamental_bajo()

        print("\n" + "=" * 80)
        print("✅ COMPLETE FUNDAMENTAL RHYTHM ANALYSIS FINISHED")
        print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("FUNDAMENTAL RHYTHM ANALYSIS (ω < 0.5)")
    print("Search for: ω=0.191 (fundamental), ω=0.382 (2nd harmonic)")
    print("Special frequencies: ω=0.261 (0.522/2), ω=0.301")
    print("="*80)

    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        print("\n🔧 INITIALIZING FUNDAMENTAL RHYTHM ANALYZER...")
        analizador = AnalisisMCMCJWST_FundamentalBajo(data_dir='data')

        if RUST_ENGINE_READY:
            print("🚀 Rust engine ready - Prior centered on ω=0.191, 0.382")
        else:
            print("🐢 Using pure Python prior for low frequencies")

        analizador.ejecutar_analisis_completo()

        print("\n🎯 ANALYSIS COMPLETED.")
        print("📊 See 'reporte_fundamental_bajo_*.txt' for detailed results")

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()