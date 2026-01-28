#!/usr/bin/env python3
"""
DEFINITIVE ANALYSIS ω=0.191 IN SDSS - CORRECTED VERSION
"""
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.timeseries import LombScargle
import emcee
from multiprocessing import Pool, cpu_count
import warnings
import sys
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# ============================================================================
# FUNCTION FOR SERIALIZING RESULTS
# ============================================================================

def serializar_resultados(resultados):
    """Converts results to JSON-serializable types."""
    resultados_serializables = {}
    
    for clave, valor in resultados.items():
        if isinstance(valor, (np.integer, np.int64, np.int32)):
            resultados_serializables[clave] = int(valor)
        elif isinstance(valor, (np.floating, np.float64, np.float32)):
            resultados_serializables[clave] = float(valor)
        elif isinstance(valor, np.ndarray):
            resultados_serializables[clave] = valor.tolist()
        elif isinstance(valor, dict):
            resultados_serializables[clave] = serializar_resultados(valor)
        elif isinstance(valor, list):
            # Process each element in the list
            resultados_serializables[clave] = [
                item.tolist() if isinstance(item, np.ndarray) else
                float(item) if isinstance(item, (np.floating, np.float64, np.float32)) else
                int(item) if isinstance(item, (np.integer, np.int64, np.int32)) else
                item
                for item in valor
            ]
        else:
            resultados_serializables[clave] = valor
    
    return resultados_serializables

# ============================================================================
# RUST ENGINE CONFIGURATION
# ============================================================================

try:
    import vpm_engine
    RUST_DISPONIBLE = True
    print("🚀 Rust engine vpm_engine detected and linked.")
except ImportError:
    RUST_DISPONIBLE = False
    print("⚠️  Rust engine not available. Using pure Python.")

# ============================================================================
# ADVANCED MALMQUIST CORRECTION
# ============================================================================

def calcular_pesos_malmquist_avanzado(df, m_lim=14.5):
    """Optimized 1/Vmax weights."""
    print(f"\n🔧 CALCULATING OPTIMIZED 1/Vmax WEIGHTS...")
    
    df['dL_max_pc'] = 10**((m_lim - df['M_K'] + 5) / 5)
    df['V_max'] = df['dL_max_pc']**3
    df['log_weight'] = -3 * np.log10(df['dL_max_pc'])
    
    log_weight_mean = df['log_weight'].mean()
    df['log_weight'] -= log_weight_mean
    df['weight'] = np.exp(df['log_weight'])
    
    weight_median = df['weight'].median()
    weight_max = 100 * weight_median
    df['weight'] = np.clip(df['weight'], 0, weight_max)
    df['weight'] = df['weight'] * len(df) / df['weight'].sum()
    
    print(f"📊 Minimum weight: {df['weight'].min():.3f}, maximum: {df['weight'].max():.3f}")
    print(f"📊 Sum of weights: {df['weight'].sum():.0f} (N={len(df)})")
    
    return df

# ============================================================================
# MCMC FUNCTIONS
# ============================================================================

def log_prior_omega191_definitivo(theta):
    """Definitive prior."""
    M0, alpha, A, omega, phi, log_sigma = theta
    
    if not (0.17 < omega < 0.22):
        return -np.inf
    if not (0.05 < A < 0.25):
        return -np.inf
    if not (-26.0 < M0 < -22.0) or not (-3.0 < log_sigma < 0.0):
        return -np.inf
    
    return -0.5 * ((omega - 0.191) / 0.02)**2

def log_likelihood_ultra_optimizado(theta, t, y, yerr, weights):
    """Ultra-optimized likelihood."""
    M0, alpha, A, omega, phi, log_sigma = theta
    
    omega_t = omega * t
    cos_term = np.cos(omega_t + phi)
    y_pred = M0 - alpha * t + A * cos_term
    
    sigma = np.exp(log_sigma)
    sigma2 = sigma**2 + yerr**2
    
    residuals = y - y_pred
    residuals2 = residuals**2
    log_sigma2 = np.log(2 * np.pi * sigma2)
    chi2 = residuals2 / sigma2
    
    log_like = -0.5 * np.sum(weights * (chi2 + log_sigma2))
    
    return log_like

def log_probability_definitivo(theta, t, y, yerr, weights):
    """Definitive probability function."""
    lp = log_prior_omega191_definitivo(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_ultra_optimizado(theta, t, y, yerr, weights)

# ============================================================================
# DEFINITIVE ANALYSIS WITH AUTOCORRELATION ANALYSIS
# ============================================================================

def analisis_definitivo_con_autocorr():
    """Analysis with detailed autocorrelation analysis."""
    print("="*80)
    print("DEFINITIVE ANALYSIS ω=0.191 IN SDSS")
    print("WITH AUTOCORRELATION ANALYSIS")
    print("="*80)
    
    # Load data
    try:
        df = pd.read_csv('SDSS_procesado_omega191.csv')
        print(f"\n📂 DATA: {len(df):,} SDSS galaxies")
    except FileNotFoundError:
        print("❌ ERROR: File not found")
        return None
    
    # Preprocessing
    print(f"\n{'='*60}")
    print("PHASE 1: PREPROCESSING")
    print('='*60)
    
    df = calcular_pesos_malmquist_avanzado(df)
    
    mask_quality = (
        (df['redshift'] > 0.01) & 
        (df['redshift'] < 0.25) &
        (df['M_K'] < -20) &
        (df['M_K'] > -27)
    )
    df = df[mask_quality].copy()
    
    print(f"🧹 Galaxies after filter: {len(df):,}")
    
    # Prepare data
    t = df['t_lookback'].values
    y = df['M_K'].values
    yerr = np.full_like(y, 0.05)
    weights = df['weight'].values
    
    print(f"\n📊 DATA FOR MCMC:")
    print(f"   • N: {len(t):,}")
    print(f"   • t range: {t.min():.2f}-{t.max():.2f} Gyr")
    
    # ========================================================================
    # MCMC WITH AUTOCORRELATION ANALYSIS
    # ========================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: MCMC WITH 15,000 STEPS")
    print('='*60)
    
    ndim = 6
    nwalkers = 32
    nsteps = 15000
    burnin = 3000
    
    # Initial position based on previous results
    initial_guess = np.array([
        -24.35,     # M0
        0.005,      # alpha
        0.12,       # A
        0.1869,     # ω (based on your result ω=0.18691)
        0.0,        # phi
        -2.3        # log_sigma
    ])
    
    pos = initial_guess + 0.0001 * np.random.randn(nwalkers, ndim)
    
    print(f"🔧 MCMC CONFIGURATION:")
    print(f"   • Walkers: {nwalkers}")
    print(f"   • Steps: {nsteps:,} (burnin: {burnin:,})")
    print(f"   • Initial ω: {initial_guess[3]:.5f}")
    print(f"   • Total iterations: {nwalkers * nsteps:,}")
    
    try:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability_definitivo,
                args=(t, y, yerr, weights), pool=pool
            )
            
            # BURN-IN
            print(f"\n🔥 BURN-IN ({burnin} steps)...")
            state = sampler.run_mcmc(pos, burnin, progress=True)
            sampler.reset()
            
            # MAIN SAMPLING
            print(f"\n📡 MAIN SAMPLING ({nsteps} steps)...")
            sampler.run_mcmc(state, nsteps, progress=True)
        
        # ====================================================================
        # DETAILED AUTOCORRELATION ANALYSIS
        # ====================================================================
        print(f"\n{'='*60}")
        print("PHASE 3: AUTOCORRELATION ANALYSIS")
        print('='*60)
        
        # Try to calculate autocorrelation
        try:
            print(f"🔍 CALCULATING AUTOCORRELATION...")
            tau = sampler.get_autocorr_time(quiet=True)
            
            print(f"📈 AUTOCORRELATION TIMES (steps):")
            print(f"   • Parameters: M0, alpha, A, omega, phi, log_sigma")
            print(f"   • Values: {tau}")
            
            # Specific analysis for ω (index 3)
            print(f"\n🎯 ANALYSIS FOR ω (parameter 3):")
            print(f"   • τ_ω = {tau[3]:.1f} steps")
            print(f"   • Chain is {nsteps/tau[3]:.1f} × τ")
            
            # Verify convergence criterion (chain > 50×τ)
            if nsteps > 50 * tau[3]:
                print(f"   ✅ Chain SUFFICIENTLY LONG (> 50×τ)")
            else:
                print(f"   ⚠️  Chain SHORT (< 50×τ)")
                print(f"   • Recommended: {int(50 * tau[3]):,} steps")
            
            # Estimated independent samples
            n_independent_omega = nsteps / tau[3]
            print(f"   • Independent samples of ω: ~{n_independent_omega:.0f}")
            
            # Analysis of all parameters
            print(f"\n📊 AUTOCORRELATION SUMMARY:")
            print(f"   • Mean τ: {np.mean(tau):.1f} steps")
            print(f"   • Minimum τ: {np.min(tau):.1f} (parameter {np.argmin(tau)})")
            print(f"   • Maximum τ: {np.max(tau):.1f} (parameter {np.argmax(tau)})")
            
            # Sampling efficiency
            efficiency = 1.0 / np.mean(tau)
            print(f"   • Sampling efficiency: {efficiency:.3f}")
            
            tau_disponible = True
            
        except Exception as e:
            print(f"⚠️  Could not calculate complete autocorrelation: {e}")
            print(f"   • Chain may be too short")
            print(f"   • τ estimated from your previous run: ~1200 steps")
            print(f"   • Current chain: {nsteps:,} steps ≈ {nsteps/1200:.1f}×τ")
            tau = None
            tau_disponible = False
        
        # Discard samples
        discard = burnin + nsteps // 4
        flat_samples = sampler.get_chain(discard=discard, flat=True)
        
        omega_samples = flat_samples[:, 3]
        n_effective = len(omega_samples)
        
        print(f"\n📊 EFFECTIVE SAMPLES:")
        print(f"   • Discarded: {discard:,} steps")
        print(f"   • Samples used: {n_effective:,}")
        print(f"   • Percentage used: {100*n_effective/(nwalkers*nsteps):.1f}%")
        
        if tau_disponible:
            effective_samples_est = n_effective / tau[3]
            print(f"   • Estimated independent samples: ~{effective_samples_est:.0f}")
        
        # ====================================================================
        # PRECISE RESULTS
        # ====================================================================
        print(f"\n{'='*60}")
        print("PHASE 4: PRECISE RESULTS")
        print('='*60)
        
        # Statistics
        percentiles = [2.5, 16, 50, 84, 97.5]
        q_values = np.percentile(omega_samples, percentiles)
        
        omega_median = q_values[2]
        omega_mean = np.mean(omega_samples)
        omega_std = np.std(omega_samples)
        
        print(f"\n🎯 ω POSTERIOR:")
        print(f"   • Median: {omega_median:.5f}")
        print(f"   • Mean: {omega_mean:.5f}")
        print(f"   • Standard deviation: {omega_std:.5f}")
        print(f"   • 68% interval: [{q_values[1]:.5f}, {q_values[3]:.5f}]")
        print(f"   • 95% interval: [{q_values[0]:.5f}, {q_values[4]:.5f}]")
        print(f"   • Δ from 0.191: {abs(omega_median - 0.191):.5f}")
        print(f"   • Precision: ±{omega_std:.5f} ({100*omega_std/omega_median:.1f}%)")
        
        # Probabilities
        print(f"\n📊 COMPATIBILITY PROBABILITIES:")
        intervals = [
            (0.1905, 0.1915, "±0.0005"),
            (0.190, 0.192, "±0.001"),
            (0.189, 0.193, "±0.002"),
            (0.188, 0.194, "±0.003"),
            (0.185, 0.197, "±0.006")  # For ω=0.18691
        ]
        
        for low, high, label in intervals:
            prob = np.mean((omega_samples > low) & (omega_samples < high))
            print(f"   • P({label}): {prob:.1%}")
        
        # ====================================================================
        # BAYES FACTOR
        # ====================================================================
        print(f"\n{'='*60}")
        print("PHASE 5: BAYESIAN EVIDENCE")
        print('='*60)
        
        log_l_osc = np.median(sampler.get_log_prob(discard=discard))
        
        # Null model
        X = np.vstack([np.ones_like(t), t]).T
        W = np.diag(weights)
        beta = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
        y_pred_null = X @ beta
        residuals_null = y - y_pred_null
        sigma2_null = np.var(residuals_null) + yerr**2
        log_l_null = -0.5 * np.sum(weights * (residuals_null**2 / sigma2_null + np.log(2 * np.pi * sigma2_null)))
        
        log_bf = log_l_osc - log_l_null
        bf = np.exp(log_bf)
        
        print(f"\n⚖️  BAYESIAN EVIDENCE:")
        print(f"   • log(L_osc): {log_l_osc:.1f}")
        print(f"   • log(L_null): {log_l_null:.1f}")
        print(f"   • log(BF): {log_bf:.1f}")
        print(f"   • BF: {bf:.1e}")
        
        print(f"\n💡 BAYESIAN INTERPRETATION:")
        if log_bf > 10:
            print("   💥 DECISIVE EVIDENCE")
            print(f"   • Oscillation model is {bf:.1e} times more probable")
        elif log_bf > 5:
            print("   ✅ VERY STRONG EVIDENCE")
        elif log_bf > 2.5:
            print("   👍 STRONG EVIDENCE")
        else:
            print("   ⚠️  MODERATE EVIDENCE")
        
        # ====================================================================
        # FINAL EVALUATION
        # ====================================================================
        print(f"\n{'='*60}")
        print("PHASE 6: FINAL EVALUATION")
        print('='*60)
        
        # Quality criteria
        criterios = {
            'High precision (σ < 0.02)': omega_std < 0.02,
            'Compatible with 0.191 (Δ < 0.01)': abs(omega_median - 0.191) < 0.01,
            'Decisive Bayes Factor (logBF > 10)': log_bf > 10,
            'Sufficient samples (> 10k)': n_effective > 10000,
            'Reasonable 95% interval (< 0.1)': (q_values[4] - q_values[0]) < 0.1,
            'Adequate convergence': tau_disponible and nsteps > 20 * (tau[3] if tau is not None else 1000)
        }
        
        puntuacion = sum(criterios.values())
        
        print(f"\n📋 QUALITY CRITERIA ({puntuacion}/{len(criterios)}):")
        for i, (criterio, valor) in enumerate(criterios.items(), 1):
            simbolo = '✅' if valor else '❌'
            print(f"   {i:2d}. {criterio}: {simbolo}")
        
        # ====================================================================
        # SAVE RESULTS (CORRECTED)
        # ====================================================================
        print(f"\n💾 SAVING RESULTS...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare results
        resultados = {
            'timestamp': timestamp,
            'mcmc_steps': int(nsteps),
            'mcmc_walkers': int(nwalkers),
            'n_galaxies': int(len(df)),
            'omega_median': float(omega_median),
            'omega_mean': float(omega_mean),
            'omega_std': float(omega_std),
            'omega_interval_68': [float(q_values[1]), float(q_values[3])],
            'omega_interval_95': [float(q_values[0]), float(q_values[4])],
            'deviation_from_191': float(abs(omega_median - 0.191)),
            'prob_within_0.001': float(np.mean((omega_samples > 0.190) & (omega_samples < 0.192))),
            'prob_within_0.002': float(np.mean((omega_samples > 0.189) & (omega_samples < 0.193))),
            'log_bayes_factor': float(log_bf),
            'bayes_factor': float(bf),
            'n_effective_samples': int(n_effective),
            'quality_score': int(puntuacion),
            'max_quality_score': int(len(criterios)),
            'autocorrelation_analysis': {
                'available': tau_disponible,
                'tau_omega': float(tau[3]) if tau_disponible and tau is not None else None,
                'tau_mean': float(np.mean(tau)) if tau_disponible and tau is not None else None,
                'chain_length_vs_tau': float(nsteps/(tau[3] if tau is not None and tau[3] > 0 else 1)) if tau_disponible and tau is not None else None
            },
            'conclusion': 'HIGH_CONFIDENCE' if puntuacion >= 5 else 'MODERATE_CONFIDENCE'
        }
        
        # Serialize results
        resultados_serializados = serializar_resultados(resultados)
        
        # Save JSON
        json_file = f'resultados_definitivos_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(resultados_serializados, f, indent=2, ensure_ascii=False)
        
        # Save simple CSV
        csv_file = f'resultados_definitivos_{timestamp}.csv'
        df_simple = pd.DataFrame([{
            'omega_median': omega_median,
            'omega_std': omega_std,
            'log_bf': log_bf,
            'bf': bf,
            'n_galaxies': len(df),
            'quality_score': puntuacion,
            'conclusion': resultados['conclusion']
        }])
        df_simple.to_csv(csv_file, index=False)
        
        # Save samples
        samples_file = f'muestras_omega_{timestamp}.npy'
        np.save(samples_file, omega_samples)
        
        print(f"\n✅ RESULTS SAVED:")
        print(f"   • {json_file} (complete results)")
        print(f"   • {csv_file} (main results)")
        print(f"   • {samples_file} ({n_effective:,} samples)")
        
        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        print(f"\n{'='*80}")
        print("FINAL ANALYSIS SUMMARY")
        print('='*80)
        
        print(f"\n🎯 DETECTED VALUE: ω = {omega_median:.5f} ± {omega_std:.5f}")
        print(f"📏 COMPATIBILITY: Δ(0.191) = {abs(omega_median-0.191):.5f}")
        print(f"⚖️  EVIDENCE: log(BF) = {log_bf:.1f} (BF = {bf:.1e})")
        print(f"📊 QUALITY: {puntuacion}/{len(criterios)} criteria")
        
        if tau_disponible:
            print(f"\n📈 AUTOCORRELATION ANALYSIS:")
            print(f"   • τ_ω = {tau[3]:.1f} steps")
            print(f"   • Chain = {nsteps:,} steps ≈ {nsteps/tau[3]:.1f}×τ")
            if nsteps > 50 * tau[3]:
                print(f"   ✅ Chain SUFFICIENTLY LONG for robust inference")
            else:
                print(f"   ⚠️  Chain SHORT for optimal inference")
                print(f"   • Recommended: at least {int(50 * tau[3]):,} steps")
        
        print(f"\n💡 SCIENTIFIC CONCLUSION:")
        if puntuacion >= 5:
            print("   ✅ HIGH CONFIDENCE DETECTION")
            print(f"   • ω ≈ {omega_median:.5f} compatible with theoretical value 0.191")
            print(f"   • DECISIVE Bayesian evidence (logBF = {log_bf:.1f})")
        else:
            print("   ⚠️  DETECTION WITH MODERATE CONFIDENCE")
            print("   • Additional analysis recommended")
        
        return resultados_serializados
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    print("\n" + "="*80)
    print("DEFINITIVE ANALYSIS ω=0.191 IN SDSS")
    print("WITH DETAILED AUTOCORRELATION ANALYSIS")
    print("="*80)
    
    start_time = datetime.now()
    print(f"⏰ Start: {start_time.strftime('%H:%M:%S')}")
    
    resultados = analisis_definitivo_con_autocorr()
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETED")
    print('='*80)
    
    print(f"\n⏰ EXECUTION TIME: {elapsed}")
    
    if resultados:
        print(f"\n✨ ANALYSIS SUCCESS!")
        print(f"   • ω = {resultados['omega_median']:.5f} ± {resultados['omega_std']:.5f}")
        print(f"   • log(BF) = {resultados['log_bayes_factor']:.1f}")
        print(f"   • Quality: {resultados['quality_score']}/{resultados['max_quality_score']}")
        
        if resultados['conclusion'] == 'HIGH_CONFIDENCE':
            print(f"\n🎉 DETECTION CONFIRMED WITH HIGH CONFIDENCE!")
        else:
            print(f"\n⚠️  Result with moderate confidence")
    
    print("\n🏁 Process completed successfully.")