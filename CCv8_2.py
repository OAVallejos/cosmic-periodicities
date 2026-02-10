#!/usr/bin/env python3
                                                 import numpy as np
import json
import time
import sys
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')

try:
    import cristal_core as cc
    RUST_AVAILABLE = True
    print("✅ Rust module 'cristal_core' loaded")
except ImportError:
    RUST_AVAILABLE = False

class SafeAnalyzer:
    def __init__(self, h0: float = 70.0):
        self.h0 = h0
        self.c = 299792.458

        if RUST_AVAILABLE:
            self.stacker = cc.CristalStacker()

    def run_safe_analysis(self):
        """SAFE analysis that doesn't freeze the kernel"""
        print("\n" + "="*70)
        print("🔒 SAFE ANALYSIS - WITHOUT FREEZING TESTS")
        print("="*70)

        start_time = time.time()

        # 1. Use ONLY first 1000 structures for speed
        structures = self.get_limited_structures(1000)

        if not structures:
            print("❌ Could not obtain structures")
            return

        print(f"• Structures for analysis: {len(structures)}")
        print(f"• Maximum pairs: {len(structures) * (len(structures) - 1) // 2:,}")

        # 2. Main analysis WITHOUT permutations
        results = self.analyze_periodicity_safe(structures)

        # 3. Fast Fourier (limited)
        if len(structures) > 50:
            self.fast_fourier_analysis(structures)

        # 4. Results
        elapsed = time.time() - start_time

        print(f"\n" + "="*70)
        print("📊 MAIN RESULTS")
        print("="*70)

        print(f"• Time: {elapsed:.1f}s")
        print(f"• Structures: {len(structures)}")

        if results:
            # Show only key scales
            key_scales = ['410_Mpc', '820_Mpc', '841_Mpc', '1640_Mpc']
            for scale in key_scales:
                if scale in results:
                    res = results[scale]
                    sigma = res.get('sigma', 0)
                    ratio = res.get('ratio', 0)

                    if sigma > 3.0:
                        print(f"✅ {scale.replace('_', ' ')}: {sigma:.1f}σ (ratio={ratio:.2f}x)")
                    elif sigma > 2.0:
                        print(f"⚠️  {scale.replace('_', ' ')}: {sigma:.1f}σ (ratio={ratio:.2f}x)")

        # Save quick results
        self.save_quick_results(results, structures)

    def get_limited_structures(self, max_structures: int = 1000):
        """Get limited structures for fast analysis"""
        if not RUST_AVAILABLE:
            print("❌ Rust not available")
            return None

        print(f"\n🔄 Getting {max_structures} structures...")

        try:
            # Load small sample
            data = np.load('data/sdss_vdisp_calidad.npz')

            # Take random sample of 50k galaxies
            n_total = len(data['RA'])
            indices = np.random.choice(n_total, min(50000, n_total), replace=False)

            ra = data['RA'][indices].astype(np.float32)
            dec = data['DEC'][indices].astype(np.float32)
            z = data['Z'][indices].astype(np.float32)
            vdisp = data['VDISP'][indices].astype(np.float32)

            # Filter
            mask = (
                (z >= 0.1) & (z <= 0.6) &
                (dec >= -5) & (dec <= 55) &
                (ra >= 130) & (ra <= 230)
            )

            ra_f = ra[mask]
            dec_f = dec[mask]
            z_f = z[mask]

            print(f"• Filtered galaxies: {len(ra_f):,}")

            # Convert to Rust format
            galaxies_rust = []
            for i in range(len(ra_f)):
                galaxies_rust.append([float(ra_f[i]), float(dec_f[i]), float(z_f[i]), float(vdisp[i])])

            # Get structures
            results = self.stacker.identificar_super_estructuras(
                galaxias=galaxies_rust,
                z_min=0.1,
                z_max=0.6,
                umbral_densidad=0.9,
                grid_ra=40,  # Coarser grid
                grid_dec=40,
                grid_z=15
            )

            # Extract and limit
            ra_list = results.get('ra', [])
            dec_list = results.get('dec', [])
            z_list = results.get('z', [])

            if len(ra_list) > max_structures:
                # Take random sample
                indices = np.random.choice(len(ra_list), max_structures, replace=False)
                ra_list = [ra_list[i] for i in indices]
                dec_list = [dec_list[i] for i in indices]
                z_list = [z_list[i] for i in indices]

            structures = list(zip(ra_list, dec_list, z_list))
            print(f"✅ Structures obtained: {len(structures)}")

            return structures

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

        return None

    def analyze_periodicity_safe(self, structures):
        """Safe analysis without heavy calculations"""
        if not RUST_AVAILABLE or len(structures) < 10:
            return None

        print(f"\n📐 Analyzing periodicity...")

        try:
            structures_rust = [list(e) for e in structures]

            # Key scales
            scales = [410.0, 820.0, 841.0, 1640.0]

            rust_results = self.stacker.analizar_periodicidad_espacial(
                estructuras=structures_rust,
                escalas_analizar=scales,
                ventana_relativa=0.05,  # 5%
                h0=self.h0
            )

            # Process results
            results = {}
            for i, scale in enumerate(scales):
                key_sigma = f"sigma_{i}"
                key_ratio = f"ratio_{i}"
                key_obs = f"observado_{i}"
                key_esp = f"esperado_{i}"

                if key_sigma in rust_results:
                    results[f"{int(scale)}_Mpc"] = {
                        'scale': scale,
                        'sigma': rust_results[key_sigma],
                        'ratio': rust_results.get(key_ratio, 0),
                        'observed': rust_results.get(key_obs, 0),
                        'expected': rust_results.get(key_esp, 0)
                    }

            print(f"✅ Analysis completed")
            return results

        except Exception as e:
            print(f"⚠️  Error in analysis: {e}")

        return None

    def fast_fourier_analysis(self, structures):
        """Fast Fourier with limited sample"""
        if len(structures) < 50:
            return

        print(f"\n📈 Fast Fourier...")

        try:
            # Calculate some distances quickly
            n = min(100, len(structures))
            distances = []

            for i in range(n):
                for j in range(i + 1, n):
                    ra1, dec1, z1 = structures[i]
                    ra2, dec2, z2 = structures[j]

                    # Simple angular distance
                    dra = abs(ra1 - ra2)
                    if dra > 180:
                        dra = 360 - dra
                    ddec = abs(dec1 - dec2)
                    dist_degrees = np.sqrt(dra**2 + ddec**2)

                    # Comoving distance
                    z_mean = (z1 + z2) / 2
                    d_comov = (self.c / self.h0) * np.log(1 + z_mean)
                    dist_mpc = d_comov * np.radians(dist_degrees)

                    if dist_mpc < 5000:
                        distances.append(dist_mpc)

            if len(distances) > 100:
                print(f"• {len(distances)} distances calculated")

                # Simple histogram
                bins = np.linspace(0, 5000, 100)
                hist, _ = np.histogram(distances, bins=bins)

                # Find peaks manually
                peaks = []
                for i in range(1, len(hist) - 1):
                    if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                        if hist[i] > np.mean(hist) * 1.5:
                            scale = bins[i]
                            if 100 < scale < 4000:
                                peaks.append(scale)

                if peaks:
                    print(f"• Peaks detected: {', '.join(f'{p:.0f} Mpc' for p in peaks[:3])}")

        except Exception as e:
            print(f"⚠️  Fourier error: {e}")

    def save_quick_results(self, results, structures):
        """Save results quickly"""
        timestamp = time.strftime("%H%M%S")
        filename = f"resultados_rapidos_{timestamp}.json"

        data = {
            'timestamp': timestamp,
            'n_structures': len(structures),
            'results': results,
            'sample_structures': structures[:10]  # Only first 10
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n💾 Results saved in: {filename}")

def main():
    print("\n" + "="*70)
    print("⚡ FAST AND SAFE ANALYSIS")
    print("="*70)

    analyzer = SafeAnalyzer(h0=70.0)
    analyzer.run_safe_analysis()

    print("\n" + "="*70)
    print("✅ COMPLETED - Kernel intact")
    print("="*70)

if __name__ == "__main__":
    main()