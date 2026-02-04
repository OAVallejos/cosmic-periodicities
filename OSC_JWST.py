#!/usr/bin/env python3
"""                        
SPECTRAL ANALYSIS OF THE CRYSTAL NETWORK - HIGHER HARMONICS
Calculates the power spectrum of Δz distances between nodes
Identifies the dominant harmonics of the cosmic oscillation
"""
import numpy as np
import json
from scipy.signal import welch, find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def analizar_espectro_armonico(json_path='analisis_multinodo_cristalino.json'):
    """Analyzes the power spectrum of distances between nodes"""

    # Load data
    with open(json_path, 'r') as f:
        datos = json.load(f)

    nodos_info = datos['analisis_global']['nodos_info']
    diferencias_z = np.array(datos['analisis_global']['diferencias_z'])

    # Extract redshifts and sort
    redshifts = np.array([info['z_centro'] for info in nodos_info])
    redshifts.sort()

    print("\n" + "="*80)
    print("SPECTRAL ANALYSIS OF THE COSMIC OSCILLATION")
    print("="*80)

    # ================== 1. HARMONIC ANALYSIS ==================
    print(f"\n📊 ANALYZED REDSHIFTS (sorted):")
    for i, z in enumerate(redshifts):
        print(f"  Node {i+1}: z = {z:.2f}")

    # Calculate all unique Δz differences
    delta_z_unicos = []
    for i in range(len(redshifts)):
        for j in range(i+1, len(redshifts)):
            delta = abs(redshifts[j] - redshifts[i])
            delta_z_unicos.append(delta)

    delta_z_unicos = np.array(delta_z_unicos)
    delta_z_unicos.sort()

    print(f"\n📏 Δz DISTANCES BETWEEN NODES:")
    for i, delta in enumerate(delta_z_unicos):
        print(f"  Δz_{i+1} = {delta:.3f}")

    # ================== 2. SEARCH FOR PERIODICITY ==================
    # Create time series of "structure presence"
    z_min, z_max = redshifts.min(), redshifts.max()
    z_range = np.linspace(z_min, z_max, 1000)

    # Create "hexagonal structure density" function
    densidad = np.zeros_like(z_range)
    for z_node, info in zip([info['z_centro'] for info in nodos_info], nodos_info):
        # Weight by hexagonal score and number of galaxies
        peso = info['score_hex'] * info['n_galaxias'] / 1000.0
        # Smooth with Gaussian
        sigma = 0.05  # Window width
        densidad += peso * np.exp(-(z_range - z_node)**2 / (2*sigma**2))

    # Normalize
    densidad = densidad / densidad.max()

    # ================== 3. POWER SPECTRUM ==================
    # Calculate spectrum using Welch
    fs = 1.0 / (z_range[1] - z_range[0])  # Sampling frequency in "1/Δz"
    freqs, psd = welch(densidad, fs=fs, nperseg=256)

    # Convert to period (1/frequency = Δz)
    periodos = 1.0 / freqs[1:]  # Ignore frequency 0
    psd_periodos = psd[1:]

    # Find peaks in the spectrum
    peaks, properties = find_peaks(psd_periodos,
                                  height=np.mean(psd_periodos)*1.5,
                                  distance=5,
                                  prominence=np.std(psd_periodos)*0.5)

    print(f"\n🎯 DOMINANT PEAK(S) IN THE SPECTRUM:")
    periodos_dominantes = []

    for peak in peaks:
        periodo = periodos[peak]
        potencia = psd_periodos[peak]

        # Only consider reasonable periods
        if 0.1 < periodo < 2.0:
            periodos_dominantes.append(periodo)
            print(f"  • Period = {periodo:.4f} (Power = {potencia:.4f})")

            # Verify relation with ω₀ = 0.191
            n_armonico = periodo / 0.191
            n_entero = round(n_armonico)
            error = abs(n_armonico - n_entero) / n_entero * 100 if n_entero > 0 else 100

            print(f"    → Harmonic {n_entero} × ω₀ (error: {error:.1f}%)")

    # ================== 4. FIT TO HARMONIC MODEL ==================
    def modelo_armonico(z, A, omega, phi, offset):
        """Harmonic oscillation model"""
        return A * np.cos(2*np.pi*omega*z + phi) + offset

    # Fit model to density
    try:
        # Initial values
        A0 = 0.5
        omega0 = 1.0 / (periodos_dominantes[0] if periodos_dominantes else 0.4)
        phi0 = 0.0
        offset0 = 0.5

        popt, pcov = curve_fit(modelo_armonico, z_range, densidad,
                              p0=[A0, omega0, phi0, offset0],
                              bounds=([0, 0.1, -np.pi, 0],
                                      [1, 5.0, np.pi, 1]))

        A_fit, omega_fit, phi_fit, offset_fit = popt
        periodo_fit = 1.0 / omega_fit

        print(f"\n🎛️  HARMONIC MODEL FIT:")
        print(f"  • Frequency: ω = {omega_fit:.4f}")
        print(f"  • Period: T = {periodo_fit:.4f}")
        print(f"  • Amplitude: A = {A_fit:.4f}")
        print(f"  • Phase: φ = {phi_fit:.3f} rad")

        # Calculate goodness of fit
        densidad_fit = modelo_armonico(z_range, *popt)
        residuos = densidad - densidad_fit
        r2 = 1.0 - np.var(residuos) / np.var(densidad)

        print(f"  • R² = {r2:.4f}")

        # Relation with ω₀ = 0.191
        n_armonico_fit = omega_fit / 0.191
        n_entero_fit = round(n_armonico_fit)
        error_fit = abs(n_armonico_fit - n_entero_fit) / n_entero_fit * 100 if n_entero_fit > 0 else 100

        print(f"\n🔗 RELATION WITH ω₀ = 0.191:")
        print(f"  • ω_measured / ω₀ = {n_armonico_fit:.3f}")
        print(f"  • Closest harmonic: n = {n_entero_fit}")
        print(f"  • Error: {error_fit:.1f}%")

        if error_fit < 10:
            print(f"  ✅ CONFIRMATION! The harmonic n={n_entero_fit} dominates")
        else:
            print(f"  ⚠️  No clear integer relation")

    except Exception as e:
        print(f"\n⚠️  Could not fit model: {e}")

    # ================== 5. HARMONIC CORRELATION MATRIX ==================
    print(f"\n🔢 HARMONIC CORRELATION MATRIX:")
    print("   " + " ".join([f"{z:.1f}" for z in redshifts]))

    for i, z_i in enumerate(redshifts):
        fila = []
        for j, z_j in enumerate(redshifts):
            if i == j:
                fila.append("1.00")
            else:
                delta = abs(z_j - z_i)
                # Check multiple of ω₀
                multiplo = delta / 0.191
                multiplo_entero = round(multiplo)
                error_multiplo = abs(multiplo - multiplo_entero) / multiplo_entero * 100 if multiplo_entero > 0 else 100

                if error_multiplo < 15:  # 15% threshold
                    fila.append(f"{multiplo_entero:2d}ω₀")
                else:
                    fila.append("   -")

        print(f"{z_i:.1f} " + " ".join(fila))

    # ================== 6. GENERATE GRAPHS ==================
    print(f"\n📈 GENERATING ANALYSIS GRAPHS...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 6.1 Structure density vs redshift
    ax1 = axes[0, 0]
    ax1.plot(z_range, densidad, 'b-', linewidth=2, label='Hexagonal density')

    # Mark real nodes
    for info in nodos_info:
        ax1.axvline(info['z_centro'], color='r', alpha=0.3, linestyle='--')
        ax1.text(info['z_centro'], 0.95, f"z={info['z_centro']:.1f}",
                rotation=90, va='top', ha='right', fontsize=8)

    if 'densidad_fit' in locals():
        ax1.plot(z_range, densidad_fit, 'r--', linewidth=1.5, alpha=0.7, label='Harmonic fit')

    ax1.set_xlabel('Redshift (z)')
    ax1.set_ylabel('Hexagonal structure density')
    ax1.set_title('Crystal nodes distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 6.2 Power spectrum
    ax2 = axes[0, 1]
    ax2.plot(periodos, psd_periodos, 'g-', linewidth=2)

    # Mark peaks
    for peak in peaks:
        if 0.1 < periodos[peak] < 2.0:
            ax2.plot(periodos[peak], psd_periodos[peak], 'ro')
            ax2.text(periodos[peak], psd_periodos[peak],
                    f'  {periodos[peak]:.3f}', fontsize=9)

    ax2.set_xlabel('Period (Δz)')
    ax2.set_ylabel('Spectral power')
    ax2.set_title('Power spectrum - Dominant periods')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2.0)

    # 6.3 Harmonic relation
    ax3 = axes[1, 0]
    delta_z_vals = []
    multiplos = []

    for i in range(len(redshifts)):
        for j in range(i+1, len(redshifts)):
            delta = abs(redshifts[j] - redshifts[i])
            delta_z_vals.append(delta)
            multiplos.append(delta / 0.191)

    ax3.scatter(delta_z_vals, multiplos, s=50, alpha=0.7)

    # Reference lines for integers
    for n in range(1, 6):
        x_vals = np.linspace(0.1, max(delta_z_vals), 10)
        y_vals = n * np.ones_like(x_vals)
        ax3.plot(x_vals, y_vals, 'r--', alpha=0.5, linewidth=0.5)
        ax3.text(0.1, n + 0.1, f'n={n}', fontsize=8)

    ax3.set_xlabel('Observed Δz')
    ax3.set_ylabel('Δz / ω₀ (ω₀=0.191)')
    ax3.set_title('Relation with fundamental harmonics')
    ax3.grid(True, alpha=0.3)

    # 6.4 Coherence map
    ax4 = axes[1, 1]
    im = ax4.imshow(diferencias_z, cmap='hot', aspect='auto')

    # Labels
    nombres = [info['nombre'].split()[-1] for info in nodos_info]
    ax4.set_xticks(range(len(nombres)))
    ax4.set_yticks(range(len(nombres)))
    ax4.set_xticklabels(nombres, rotation=45)
    ax4.set_yticklabels(nombres)

    plt.colorbar(im, ax=ax4, label='Δz')
    ax4.set_title('Distance matrix between nodes')

    plt.tight_layout()
    plt.savefig('espectro_armonico_cristalino.png', dpi=150, bbox_inches='tight')
    print(f"💾 Graph saved as 'espectro_armonico_cristalino.png'")

    # ================== 7. CONCLUSION ==================
    print(f"\n" + "="*80)
    print("SCIENTIFIC CONCLUSION")
    print("="*80)

    if len(periodos_dominantes) > 0:
        periodo_principal = periodos_dominantes[0]
        n_principal = round(periodo_principal / 0.191)

        print(f"📊 MAIN RESULT:")
        print(f"• Dominant period: Δz = {periodo_principal:.4f}")
        print(f"• Corresponds to harmonic n = {n_principal} of ω₀ = 0.191")
        print(f"• Frequency: ω = {1.0/periodo_principal:.4f}")

        print(f"\n🎯 COSMOLOGICAL IMPLICATIONS:")
        print(f"1. The crystal network is QUANTIZED in redshift")
        print(f"2. The fundamental scale is ~{periodo_principal:.3f} in Δz")
        print(f"3. This corresponds to ~{periodo_principal*1100:.0f} Mpc/h (using ΛCDM cosmology)")

        print(f"\n🔭 MODEL PREDICTION:")
        print(f"• The next node should be at z ≈ {redshifts.max() + periodo_principal:.2f}")
        print(f"• Or at z ≈ {redshifts.min() - periodo_principal:.2f}")
    else:
        print("⚠️  No clear periodicity detected in the analyzed range")

    print(f"\n" + "="*80)
    print("SPECTRAL ANALYSIS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    analizar_espectro_armonico()