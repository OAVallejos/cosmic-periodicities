#!/usr/bin/env python3
"""                         
HIGH-Z HARMONIC ANALYSIS (JWST)                  Searches for resonances at multiples of ω₀ = 0.191            """                         
import numpy as np
import vpm_core
from pathlib import Path
import gzip
import json

def load_jwst_high_z(file, z_min=6.0):
    """Loads only galaxies with z > 6 from JWST .dat.gz files"""
    redshifts = []
    try:
        with gzip.open(file, 'rt') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    z = float(parts[4])
                    if z > z_min:
                        redshifts.append(z)
                except (ValueError, IndexError):
                    continue
        return redshifts
    except Exception as e:
        print(f"  Error in {file.name}: {e}")
        return []

def calculate_vpm_coherence_rust(zs, omega_0, kernel_params=None):
    """
    Calculates phase coherence using the Rust kernel
    """
    kernel = vpm_core.VPMKernel()
    if kernel_params:
        kernel.omega_0 = kernel_params.get('omega_0', 0.191)
        kernel.z_c = kernel_params.get('z_c', 0.21)
        kernel.xi = kernel_params.get('xi', 0.084)
        kernel.target_alpha = kernel_params.get('target_alpha', 0.35)
    else:
        kernel.omega_0 = omega_0
        kernel.z_c = 0.21
        kernel.xi = 0.084
        kernel.target_alpha = 0.35

    # Calculate birefringence
    delta_alpha = kernel.compute_birefringence(sorted(zs, reverse=True))

    # Convert to phase
    phases = 2 * np.pi * np.array(delta_alpha) / kernel.target_alpha

    # R statistic
    R = np.abs(np.sum(np.exp(1j * phases))) / len(phases)

    # Significance
    p = np.exp(-len(phases) * R**2)
    sigma = np.sqrt(2) * np.sqrt(-np.log(p)) if p > 0 else 99

    return R, sigma

def analyze_jwst_harmonics():
    """
    Analyzes harmonic resonances in high-z JWST galaxies
    """
    print("\n" + "="*70)
    print("🔮 HIGH-Z HARMONIC ANALYSIS (JWST)")
    print("="*70)

    # 1. Base parameters
    w0 = 0.191  # Fundamental crystal frequency

    # 2. Load JWST data
    data_path = Path("./data")
    jwst_files = list(data_path.glob("*.dat.gz"))

    print(f"\n📂 Searching in {len(jwst_files)} JWST files...")

    all_high_z = []
    for file in jwst_files:
        z_list = load_jwst_high_z(file, z_min=6.0)
        if z_list:
            print(f"   • {file.name:20s}: {len(z_list):5,d} galaxies (z>6)")
            all_high_z.extend(z_list)

    print(f"\n📊 TOTAL: {len(all_high_z):,} galaxies with z > 6")

    if len(all_high_z) < 100:
        print("❌ Insufficient data for harmonic analysis")
        return

    # 3. Define harmonics to test
    harmonics = {
        'Fundamental (n=1)': w0,
        'Harmonic 2x (n=2)': 2 * w0,
        'Harmonic 3x (n=3)': 3 * w0,
        'Harmonic 4x (n=4)': 4 * w0,
        'Harmonic 5x (n=5)': 5 * w0,
        'Harmonic 6x (n=6)': 6 * w0,  # Mode predicted by VPM-48
        'Harmonic 7x (n=7)': 7 * w0,
        'Harmonic 8x (n=8)': 8 * w0,
    }

    print("\n🔍 SEARCHING FOR HARMONIC RESONANCES:")
    print("-" * 60)
    print(f"{'Mode':20s} {'Frequency':12s} {'R':>10s} {'σ':>10s}")
    print("-" * 60)

    results = {}

    # Test each frequency
    for name, freq in harmonics.items():
        R, sigma = calculate_vpm_coherence_rust(all_high_z, freq)
        results[name] = {'freq': freq, 'R': R, 'sigma': sigma}

        # Highlight if high significance
        if sigma > 5:
            print(f"  ✅ {name:18s} ω={freq:.3f}   R={R:.4f}   σ={sigma:6.2f} ***")
        elif sigma > 3:
            print(f"  ⚠️ {name:18s} ω={freq:.3f}   R={R:.4f}   σ={sigma:6.2f}")
        else:
            print(f"     {name:18s} ω={freq:.3f}   R={R:.4f}   σ={sigma:6.2f}")

    # 4. Detailed analysis of n=6 mode (predicted)
    print("\n" + "="*70)
    print("📈 DETAILED ANALYSIS - n=6 MODE")
    print("="*70)

    # Test with small variations around 6ω₀
    freq_base = 6 * w0
    variations = np.linspace(0.95, 1.05, 11)  # ±5%

    print(f"\n🔬 Fine scan around 6ω₀ = {freq_base:.3f}:")
    print(f"{'Δ/ω₀':10s} {'Frequency':12s} {'R':>10s} {'σ':>10s}")
    print("-" * 40)

    for frac in variations:
        freq_test = freq_base * frac
        R, sigma = calculate_vpm_coherence_rust(all_high_z, freq_test)
        mark = "◀" if abs(frac - 1.0) < 0.01 else ""
        print(f"  {frac:6.2f}x    ω={freq_test:.3f}   R={R:.4f}   σ={sigma:6.2f}  {mark}")

    # 5. Comparison with local mode (n=1+ξ)
    print("\n" + "="*70)
    print("🔄 COMPARISON WITH LOCAL MODE (z<0.21)")
    print("="*70)

    # Load local data from NPZ
    try:
        npz_file = data_path / "sdss_vdisp_calidad.npz"
        if npz_file.exists():
            data = np.load(npz_file)
            z_local = data['Z'][data['Z'] < 0.21]
            print(f"\n📊 Local data: {len(z_local):,} galaxies with z<0.21")

            # Local mode with vorticity correction (1+ξ)
            freq_local = w0 * (1 + 0.084)  # 1+ξ = 1.084
            R_local, sigma_local = calculate_vpm_coherence_rust(z_local.tolist(), freq_local)

            print(f"\n   • Local mode (n=1+ξ = 1.084):")
            print(f"     ω_local = {freq_local:.3f}")
            print(f"     R_local = {R_local:.4f}")
            print(f"     σ_local = {sigma_local:.2f}")

            # Compare with n=6 mode
            print(f"\n   • Mode ratio: (6ω₀) / ((1+ξ)ω₀) = {6/(1+0.084):.3f}")
    except Exception as e:
        print(f"  Could not load local data: {e}")

    # 6. Save results
    results_json = {
        'w0': w0,
        'n_high_z_galaxies': len(all_high_z),
        'harmonics': {k: {'freq': v['freq'], 'R': float(v['R']), 'sigma': float(v['sigma'])}
                     for k, v in results.items()}
    }

    with open('jwst_harmonics_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n📁 Results saved in: jwst_harmonics_results.json")
    print("\n" + "="*70)

    # 7. Conclusion
    print("\n🎯 CONCLUSIONS:")
    if results.get('Harmonic 6x (n=6)', {}).get('sigma', 0) > 5:
        print("   ✅ n=6 MODE CONFIRMED: Photonic crystal at high z")
        print("      The 6ω₀ harmonic shows significant coherence")
    else:
        print("   ⚠️ n=6 MODE NOT DETECTED: Possible adjustment needed")

if __name__ == "__main__":
    analyze_jwst_harmonics()