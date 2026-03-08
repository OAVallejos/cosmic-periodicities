#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPM48_PHASE_DRIFT_FINAL.py - Mapeo definitivo de la deriva de fase
CORREGIDO: Manejo de arrays de diferente tamaño
"""

import numpy as np
from scipy import stats, special
from scipy.optimize import curve_fit
import json
from astropy.cosmology import FlatLambdaCDM

# Constantes
OMEGA0 = 0.191
OFFSET_SDSS = 0.067525           # Offset para universo local
OFFSET_JWST = 0.222125            # Offset para alto z (0.067525 + 0.1546)

# Datos SDSS (3 nodos)
sdss_data = {
    'z': np.array([0.000, 0.191, 0.382]),
    'counts': np.array([53, 535, 680]),
    'total': 6687,
    'epoch': 'Local Universe (z~0.2)'
}

# Datos JWST (46 nodos)
jwst_high_data = {
    'z': np.array([6.112, 6.303, 6.494, 6.685, 6.876, 7.067, 7.258, 7.449, 
                   7.640, 7.831, 8.022, 8.213, 8.404, 8.595, 8.786, 8.977,
                   9.168, 9.359, 9.550, 9.741, 9.932, 10.123, 10.314, 10.505,
                   10.696, 10.887, 11.078, 11.269, 11.460, 11.651, 11.842,
                   12.033, 12.224, 12.415, 12.606, 12.797, 12.988, 13.179,
                   13.370, 13.561, 13.943, 14.134, 14.325, 14.516, 14.707, 14.898]),
    'counts': np.array([475, 162, 489, 3658, 237, 223, 278, 199,
                        483, 159, 298, 385, 59, 131, 218, 123,
                        97, 154, 38, 65, 69, 36, 62, 47,
                        58, 23, 12, 15, 14, 21, 22,
                        22, 12, 47, 22, 12, 191, 2,
                        5, 24, 1, 54, 3, 2, 2, 2]),
    'total': 42254,
    'epoch': 'Cosmic Dawn (z~7-14)'
}

def calculate_phases(z_values, offset, omega=OMEGA0):
    """Calcula fase con offset específico por época"""
    return ((z_values - offset) / omega) % 1.0

def calculate_n_quantum(z, offset, omega=OMEGA0):
    """Calcula el número cuántico n"""
    return (z - offset) / omega

def map_phase_drift():
    """Mapea la deriva de fase entre épocas cósmicas"""
    
    print("="*70)
    print("VPM-48 COSMIC CRYSTAL: PHASE DRIFT MAPPING - FINAL")
    print("="*70)
    print(f"Fundamental frequency ω₀ = {OMEGA0}")
    print(f"SDSS offset = {OFFSET_SDSS:.6f}")
    print(f"JWST offset = {OFFSET_JWST:.6f}")
    print(f"Offset difference = {OFFSET_JWST - OFFSET_SDSS:.6f}")
    
    print(f"\nSample sizes:")
    print(f"  SDSS-V (z<0.5):     {sdss_data['total']:,} galaxies")
    print(f"  JWST high z (z>6):  {jwst_high_data['total']:,} galaxies")
    print(f"  TOTAL:               {sdss_data['total'] + jwst_high_data['total']:,} galaxies")
    
    print("\n" + "█"*60)
    print("██ 1. PHASE CALCULATION WITH EPOCH-SPECIFIC OFFSETS")
    print("█"*60)
    
    # Calcular fases
    phases_sdss = calculate_phases(sdss_data['z'], OFFSET_SDSS)
    phases_jwst = calculate_phases(jwst_high_data['z'], OFFSET_JWST)
    
    print(f"\nSDSS epoch (offset = {OFFSET_SDSS:.6f}):")
    for i, (z, phase, count) in enumerate(zip(sdss_data['z'], phases_sdss, sdss_data['counts'])):
        n = calculate_n_quantum(z, OFFSET_SDSS)
        print(f"  Node z={z:.3f}: n={n:+.4f} → φ={phase:.4f}, {count:4d} galaxies")
    print(f"  Mean phase SDSS: {np.mean(phases_sdss):.6f} ± {np.std(phases_sdss):.6f}")
    
    print(f"\nJWST epoch (offset = {OFFSET_JWST:.6f}):")
    unique_phases = np.unique(phases_jwst.round(6))
    print(f"  Nodes: {len(jwst_high_data['z'])}")
    print(f"  Phase values: {', '.join([f'{p:.6f}' for p in unique_phases])}")
    print(f"  Mean phase JWST: {np.mean(phases_jwst):.6f} ± {np.std(phases_jwst):.6f}")
    
    print("\n" + "█"*60)
    print("██ 2. QUANTUM NUMBER ANALYSIS")
    print("█"*60)
    
    print("\nQuantum numbers n = (z - offset)/ω₀:")
    print("\nSDSS nodes:")
    for z in sdss_data['z']:
        n_exact = calculate_n_quantum(z, OFFSET_SDSS)
        n_round = round(n_exact)
        diff_to_int = abs(n_exact - n_round)
        print(f"  z={z:.3f}: n_exact={n_exact:+.6f} → n={n_round:2d} (Δ={diff_to_int:.6f})")
    
    print("\nJWST nodes (first 10):")
    for z in jwst_high_data['z'][:10]:
        n_exact = calculate_n_quantum(z, OFFSET_JWST)
        n_round = round(n_exact)
        diff_to_int = abs(n_exact - n_round)
        print(f"  z={z:.3f}: n_exact={n_exact:+.6f} → n={n_round:3d} (Δ={diff_to_int:.6f})")
    
    # Estadísticas de cuantización
    jwst_n_exact = calculate_n_quantum(jwst_high_data['z'], OFFSET_JWST)
    jwst_n_round = np.round(jwst_n_exact)
    jwst_quant_error = np.abs(jwst_n_exact - jwst_n_round)
    
    print(f"\nQuantization statistics:")
    print(f"  JWST mean quantization error: {np.mean(jwst_quant_error):.6f}")
    print(f"  JWST std quantization error: {np.std(jwst_quant_error):.6f}")
    print(f"  JWST max quantization error: {np.max(jwst_quant_error):.6f}")
    
    print("\n" + "█"*60)
    print("██ 3. PHASE DRIFT MEASUREMENT")
    print("█"*60)
    
    # Cosmología
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    
    # Pesos por número de galaxias
    w_sdss = sdss_data['counts'] / np.sum(sdss_data['counts'])
    w_jwst = jwst_high_data['counts'] / np.sum(jwst_high_data['counts'])
    
    # Redshifts promedio ponderados
    z_sdss_mean = np.average(sdss_data['z'], weights=w_sdss)
    z_jwst_mean = np.average(jwst_high_data['z'], weights=w_jwst)
    
    # Tiempos cósmicos
    t_sdss = cosmo.age(z_sdss_mean).value
    t_jwst = cosmo.age(z_jwst_mean).value
    t_universe = cosmo.age(0).value
    
    # Fases promedio ponderadas
    phi_sdss = np.average(phases_sdss, weights=w_sdss)
    phi_jwst = np.average(phases_jwst, weights=w_jwst)
    
    # Diferencia de fase (LA DERIVA)
    delta_t = t_sdss - t_jwst
    delta_phi = phi_sdss - phi_jwst
    
    # Velocidad de deriva
    drift_velocity = delta_phi / delta_t if delta_t != 0 else 0
    
    print(f"\nCosmic timeline:")
    print(f"  JWST epoch:  z={z_jwst_mean:.2f} → t={t_jwst:.2f} Gyr")
    print(f"  SDSS epoch:  z={z_sdss_mean:.2f} → t={t_sdss:.2f} Gyr")
    print(f"  Time elapsed: Δt = {delta_t:.2f} Gyr")
    
    print(f"\nPhase evolution:")
    print(f"  φ_JWST (z~{z_jwst_mean:.1f}) = {phi_jwst:.6f}")
    print(f"  φ_SDSS (z~{z_sdss_mean:.1f}) = {phi_sdss:.6f}")
    print(f"  Δφ = {delta_phi:+.6f} cycles ({delta_phi*360:+.4f} degrees)")
    print(f"  Drift velocity = {drift_velocity:.6f} cycles/Gyr")
    print(f"                   {drift_velocity*2*np.pi:.6f} rad/Gyr")
    
    # Tiempo para un ciclo completo
    if abs(drift_velocity) > 1e-10:
        cycle_time = 1/abs(drift_velocity)
        cycles_per_hubble = t_universe / cycle_time
    else:
        cycle_time = float('inf')
        cycles_per_hubble = 0
    
    print(f"\nCrystal dynamics:")
    print(f"  Time for full phase cycle: {cycle_time:.2f} Gyr")
    print(f"  Cycles per Hubble time: {cycles_per_hubble:.4f}")
    
    if cycles_per_hubble < 0.01:
        print("  → RIGID CRYSTAL: Rotates extremely slowly")
    elif cycles_per_hubble < 1:
        print("  → SLOW DYNAMICS: Crystal rotates slower than expansion")
        print(f"     (completes {cycles_per_hubble:.2f} cycles per Hubble time)")
    else:
        print("  → FAST DYNAMICS: Crystal rotates faster than expansion")
        print(f"     (completes {cycles_per_hubble:.2f} cycles per Hubble time)")
    
    print("\n" + "█"*60)
    print("██ 4. TEST: DOES THE CRYSTAL FOLLOW THE EXPANSION?")
    print("█"*60)
    
    # Preparar datos para el test (CORREGIDO: ahora manejamos arrays separados)
    print("\nComparing two models:")
    
    # Modelo 1: Fase constante (red rígida)
    # Calculamos χ² por separado para cada dataset y sumamos
    chi2_sdss_const = np.sum(((phases_sdss - phi_sdss) / 0.01)**2)
    chi2_jwst_const = np.sum(((phases_jwst - phi_jwst) / 0.01)**2)
    chi2_const = chi2_sdss_const + chi2_jwst_const
    
    # Modelo 2: Fase ∝ ln(1+z) (sigue expansión)
    # Predicción para cada dataset
    def expansion_prediction(z, phi0, alpha):
        return phi0 + alpha * np.log(1+z)
    
    # Ajuste usando todos los puntos
    all_z = np.concatenate([sdss_data['z'], jwst_high_data['z']])
    all_phi = np.concatenate([phases_sdss, phases_jwst])
    all_weights = np.concatenate([sdss_data['counts'], jwst_high_data['counts']])
    all_weights = all_weights / np.sum(all_weights)
    
    try:
        popt, pcov = curve_fit(expansion_prediction, all_z, all_phi, 
                               sigma=1/all_weights, p0=[0.6, 0.0])
        phi0_fit, alpha_fit = popt
        
        # Calcular χ² para el modelo de expansión
        pred_all = expansion_prediction(all_z, *popt)
        chi2_exp = np.sum(((all_phi - pred_all) / 0.01)**2)
        
        print(f"\nModel 1 (rigid crystal, constant phase):")
        print(f"  χ² = {chi2_const:.2f}")
        print(f"  DoF = {len(all_phi)}")
        
        print(f"\nModel 2 (follows expansion, φ ∝ ln(1+z)):")
        print(f"  φ₀ = {phi0_fit:.6f}")
        print(f"  α = {alpha_fit:.6f}")
        print(f"  χ² = {chi2_exp:.2f}")
        print(f"  DoF = {len(all_phi)-2}")
        
        # Comparación
        delta_chi2 = chi2_const - chi2_exp
        print(f"\nΔχ² = {delta_chi2:.2f}")
        
        if delta_chi2 > 10:
            print("✅ EXPANSION MODEL IS SIGNIFICANTLY BETTER")
            print("   The crystal follows cosmic expansion")
        elif delta_chi2 < -10:
            print("✅ RIGID MODEL IS SIGNIFICANTLY BETTER")
            print("   The crystal has constant phase")
        else:
            print("⚠️ MODELS ARE STATISTICALLY EQUIVALENT")
            print("   More data needed to distinguish")
            
    except Exception as e:
        print(f"Error in model fitting: {e}")
        alpha_fit = 0
        chi2_exp = 1e10
    
    print("\n" + "█"*60)
    print("██ 5. SUMMARY FOR PAPER")
    print("█"*60)
    
    print(f"""
    PHASE DRIFT MEASUREMENT SUMMARY:
    ================================
    
    Epochs compared:
      • Cosmic Dawn:    z={z_jwst_mean:.2f} (t={t_jwst:.2f} Gyr)
      • Local Universe: z={z_sdss_mean:.2f} (t={t_sdss:.2f} Gyr)
      • Time difference: Δt = {delta_t:.2f} Gyr
    
    Phase values:
      • φ_JWST = {phi_jwst:.6f}
      • φ_SDSS = {phi_sdss:.6f}
      • Δφ = {delta_phi:+.6f} cycles ({delta_phi*360:+.3f}°)
    
    Drift velocity: {drift_velocity:.6f} cycles/Gyr
    Full cycle time: {cycle_time:.2f} Gyr
    
    INTERPRETATION:
    The crystal completes {cycles_per_hubble:.3f} full cycles per Hubble time,
    indicating a {"SLOW DYNAMIC" if cycles_per_hubble < 1 else "RAPID DYNAMIC"} evolution.
    
    The offset difference Δoffset = {OFFSET_JWST - OFFSET_SDSS:.6f}
    corresponds to a phase shift of exactly {((OFFSET_JWST - OFFSET_SDSS)/OMEGA0)*360:.2f} degrees,
    confirming the {"EXPANSION-FOLLOWING" if abs(alpha_fit - delta_phi/delta_t) < 0.001 else "INTRINSIC"} nature
    of the cosmic crystal.
    """)
    
    return {
        'delta_t': float(delta_t),
        'delta_phi': float(delta_phi),
        'drift_velocity': float(drift_velocity),
        'cycle_time': float(cycle_time),
        'cycles_per_hubble': float(cycles_per_hubble),
        'phi_sdss': float(phi_sdss),
        'phi_jwst': float(phi_jwst),
        'offset_sdss': OFFSET_SDSS,
        'offset_jwst': OFFSET_JWST,
        'offset_difference': OFFSET_JWST - OFFSET_SDSS,
        'quantization_error_jwst': float(np.mean(jwst_quant_error))
    }

if __name__ == "__main__":
    results = map_phase_drift()
    
    # Guardar resultados
    with open('phase_drift_final_results.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v 
                  for k, v in results.items()}, f, indent=2)
    print("\n✓ Results saved to phase_drift_final_results.json")