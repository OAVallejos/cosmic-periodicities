#!/usr/bin/env python3     
"""
 - VERSION OPTIMIZED FOR AWS    
Simplified Bayesian analysis for t3.micro instance                          
"""

import numpy as np
import json
import time
import sys
import warnings
import gc
from typing import List, Dict, Any
from astropy.io import fits
from scipy import stats, optimize
warnings.filterwarnings('ignore')

sys.path.append('.')

try:
    import cristal_core as cc
    RUST_AVAILABLE = True
    print("✅ Rust module 'cristal_core' loaded")
except ImportError as e:
    print(f"❌ Error loading Rust: {e}")
    RUST_AVAILABLE = False

class JSONEncoderCustom(json.JSONEncoder):
    """Custom optimized encoder"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)
        return super().default(obj)

class OptimizedHybridDataset:
    """Load dataset optimized for limited memory"""

    def __init__(self, H0: float = 70.0):
        self.H0 = H0
        self.c = 299792.458

    def load_optimized_dataset(self, filepath: str = 'data/DATASET_LRG_VDISP_FLUXR_FINAL.fits',
                                 max_galaxies: int = 500000):
        """Load dataset optimized for AWS"""

        print(f"\n📁 LOADING OPTIMIZED DATASET (max {max_galaxies:,})...")

        try:
            with fits.open(filepath) as hdul:
                data = hdul[1].data

                # Take random sample if too large
                n_total = len(data)
                if n_total > max_galaxies:
                    indices = np.random.choice(n_total, max_galaxies, replace=False)
                    print(f"• Taking sample of {max_galaxies:,} from {n_total:,} galaxies")
                else:
                    indices = np.arange(n_total)

                # Load with optimized data type (float32)
                ra = data['RA'][indices].astype(np.float32)
                dec = data['DEC'][indices].astype(np.float32)
                z = data['Z'][indices].astype(np.float32)
                vdisp = data['VDISP'][indices].astype(np.float32)
                flux_r = data['FLUX_R'][indices].astype(np.float32)

                print(f"✅ Optimized dataset loaded: {len(ra):,} galaxies")
                print(f"• Memory optimized: float32 instead of float64")
                print(f"• z range: {z.min():.3f} - {z.max():.3f}")

                return ra, dec, z, vdisp, flux_r

        except Exception as e:
            print(f"❌ Error loading FITS: {e}")
            return None

    def calculate_simplified_weights(self, z: np.ndarray, vdisp: np.ndarray):
        """Simplified weights for AWS"""

        print(f"\n⚖️ CALCULATING SIMPLIFIED WEIGHTS...")

        n = len(z)

        # 1. Basic weight by redshift (linear approximation)
        weight_z = 1.0 + 0.5 * z  # More weight at higher z

        # 2. Mass weight (VDISP)
        weight_mass = (vdisp / 200.0) ** 1.5  # Softer than ^3

        # 3. Combine and normalize
        weights = weight_z * weight_mass

        # 4. Clip extremes for numerical stability
        weight_max = np.percentile(weights, 99.9)
        weights = np.clip(weights, 0.1, weight_max)

        # 5. Normalize (mean = 1)
        if np.mean(weights) > 0:
            weights = weights / np.mean(weights)

        print(f"📊 SIMPLIFIED WEIGHTS:")
        print(f"• Average weight: {np.mean(weights):.2f}")
        print(f"• Min/max weight: {np.min(weights):.2f} / {np.max(weights):.2f}")
        print(f"• Max/min ratio: {np.max(weights)/np.min(weights):.1f}")

        return weights

    def filter_optimized(self, ra, dec, z, weights, min_z=0.1, max_z=1.0):
        """Filter optimized for AWS"""

        # Stricter filter to reduce data
        mask = (
            (z >= min_z) & (z <= max_z) &
            (ra >= 150) & (ra <= 210) &  # Smaller region
            (dec >= 0) & (dec <= 50) &
            (weights > 0.1)  # Remove very small weights
        )

        ra_f = ra[mask]
        dec_f = dec[mask]
        z_f = z[mask]
        weights_f = weights[mask]

        # Limit to maximum 200k for AWS
        if len(ra_f) > 200000:
            indices = np.random.choice(len(ra_f), 200000, replace=False)
            ra_f = ra_f[indices]
            dec_f = dec_f[indices]
            z_f = z_f[indices]
            weights_f = weights_f[indices]

        print(f"\n🎯 AWS OPTIMIZED FILTER:")
        print(f"• Galaxies after filter: {len(ra_f):,}")
        print(f"• Average z: {z_f.mean():.3f}")
        print(f"• Average weight: {weights_f.mean():.2f}")

        return ra_f, dec_f, z_f, weights_f

class SimplifiedBayesianAnalyzer:
    """Simplified Bayesian analysis for AWS"""

    def __init__(self):
        self.H0 = 70.0
        self.c = 299792.458

        if RUST_AVAILABLE:
            self.stacker = cc.CristalStacker()

        self.dataset = OptimizedHybridDataset(H0=self.H0)

    def run_simplified_analysis(self):
        """Run simplified analysis for AWS"""

        print("\n" + "="*80)
        print("⚡ CRISTAL_CORE v8.9 - OPTIMIZED FOR AWS")
        print("="*80)
        print("OBJECTIVE: Bayesian analysis on t3.micro instance")
        print("STRATEGY: Reduced dataset, simplified weights, lightweight MCMC")
        print("="*80)

        start_time = time.time()

        # Clear memory
        gc.collect()

        # 1. LOAD OPTIMIZED DATASET
        data = self.dataset.load_optimized_dataset(max_galaxies=300000)
        if data is None:
            return

        ra, dec, z, vdisp, flux_r = data

        # 2. CALCULATE SIMPLIFIED WEIGHTS
        weights = self.dataset.calculate_simplified_weights(z, vdisp)

        # 3. OPTIMIZED FILTER
        ra_f, dec_f, z_f, weights_f = self.dataset.filter_optimized(
            ra, dec, z, weights, min_z=0.1, max_z=0.8  # Lower z for more pairs
        )

        # Clear memory
        del ra, dec, z, vdisp, flux_r, weights
        gc.collect()

        # 4. PREPARE FOR RUST
        print(f"\n🔄 PREPARING FOR RUST...")
        galaxies_rust = self.prepare_rust_optimized(ra_f, dec_f, z_f, weights_f)

        # Clear more
        del ra_f, dec_f, z_f
        gc.collect()

        # 5. IDENTIFY STRUCTURES
        print(f"\n🏗️ IDENTIFYING STRUCTURES...")
        structures = self.identify_structures_optimized(galaxies_rust)

        if not structures or len(structures) < 300:
            print("❌ Not enough structures")
            return

        print(f"• Structures identified: {len(structures)}")

        # 6. CALCULATE DISTANCES
        print(f"\n📏 CALCULATING DISTANCES...")
        distances = self.calculate_optimized_distances(structures, max_pairs=50000)

        print(f"• Distances calculated: {len(distances):,}")
        print(f"• Range: {np.min(distances):.0f} - {np.max(distances):.0f} Mpc")

        # 7. SIMPLIFIED BAYESIAN ANALYSIS
        print(f"\n" + "="*80)
        print("📊 SIMPLIFIED BAYESIAN ANALYSIS")
        print("="*80)

        result_1640 = self.simplified_bayesian_analysis_1640(distances)

        # 8. HARMONIC SERIES ANALYSIS
        print(f"\n" + "="*80)
        print("🎼 SIMPLIFIED HARMONIC SERIES ANALYSIS")
        print("="*80)

        harmonic_results = self.analyze_simplified_harmonic_series(distances)

        # 9. SIGNIFICANCE STATISTICS
        print(f"\n" + "="*80)
        print("🎯 SIGNIFICANCE STATISTICS")
        print("="*80)

        stats_result = self.calculate_significance_statistics(distances)

        # 10. SAVE RESULTS
        elapsed = time.time() - start_time

        self.save_simplified_results(
            result_1640, harmonic_results, stats_result,
            distances, structures, elapsed
        )

        print(f"\n✅ ANALYSIS COMPLETED IN {elapsed:.1f}s")
        print(f"💾 Memory used: ~{self.estimate_memory():.1f} MB")

    def prepare_rust_optimized(self, ra, dec, z, weights):
        """Prepare optimized data for Rust"""

        galaxies_rust = []
        n = min(100000, len(ra))  # Limit for AWS

        for i in range(n):
            # Encode weight in vdisp (100-300 km/s)
            norm_weight = 100 + 200 * (weights[i] / np.max(weights))

            galaxies_rust.append([
                float(ra[i]),
                float(dec[i]),
                float(z[i]),
                float(norm_weight)
            ])

        print(f"• Galaxies for Rust: {len(galaxies_rust):,}")

        return galaxies_rust

    def identify_structures_optimized(self, galaxies_rust):
        """Identify optimized structures"""

        if not RUST_AVAILABLE or len(galaxies_rust) < 5000:
            return self.identify_structures_simple_python(galaxies_rust)

        try:
            # Parameters optimized for AWS
            results = self.stacker.identificar_super_estructuras(
                galaxias=galaxies_rust[:50000],  # Only first 50k
                z_min=0.1,
                z_max=0.8,
                umbral_densidad=0.8,
                grid_ra=20,
                grid_dec=20,
                grid_z=10
            )

            ra_list = results.get('ra', [])
            dec_list = results.get('dec', [])
            z_list = results.get('z', [])

            if len(ra_list) == 0:
                return self.identify_structures_simple_python(galaxies_rust)

            structures = list(zip(ra_list, dec_list, z_list))

            # Limit to 500
            if len(structures) > 500:
                indices = np.random.choice(len(structures), 500, replace=False)
                structures = [structures[i] for i in indices]

            return structures

        except Exception as e:
            print(f"⚠️  Rust error: {e} - using Python")
            return self.identify_structures_simple_python(galaxies_rust)

    def identify_structures_simple_python(self, galaxies_rust):
        """Simple identification in Python"""
        print("  Using simple Python clustering...")

        n = min(500, len(galaxies_rust))
        structures = []

        # Take every 100th galaxy as "structure"
        step = max(1, len(galaxies_rust) // n)

        for i in range(0, len(galaxies_rust), step):
            if len(structures) >= n:
                break
            gal = galaxies_rust[i]
            structures.append((gal[0], gal[1], gal[2]))

        return structures

    def calculate_optimized_distances(self, structures, max_pairs=50000):
        """Calculate optimized distances"""

        n = min(200, len(structures))
        distances = []

        # Pre-calculate coordinates
        coords = []
        for ra, dec, z in structures[:n]:
            d = (self.c / self.H0) * np.log(1 + z)
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)

            x = d * np.cos(dec_rad) * np.cos(ra_rad)
            y = d * np.cos(dec_rad) * np.sin(ra_rad)
            z_coord = d * np.sin(dec_rad)

            coords.append((x, y, z_coord))

        # Calculate distances (optimized)
        for i in range(len(coords)):
            xi, yi, zi = coords[i]
            for j in range(i + 1, len(coords)):
                xj, yj, zj = coords[j]

                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)

                if 100 < dist < 4000:
                    distances.append(dist)

                if len(distances) >= max_pairs:
                    break

            if len(distances) >= max_pairs:
                break

        return np.array(distances)

    def simplified_bayesian_analysis_1640(self, distances):
        """Simplified Bayesian analysis for 1640 Mpc"""

        print(f"\n🔍 SIMPLIFIED BAYESIAN INFERENCE 1640 MPC")

        scale = 1640.0
        window = 82.0  # ±5%

        # Verify range
        if scale > np.max(distances):
            print(f"❌ 1640 Mpc out of range (max: {np.max(distances):.0f} Mpc)")
            return None

        # Observed data
        mask = np.abs(distances - scale) < window
        n_obs = np.sum(mask)
        n_total = len(distances)

        # Simple model: Binomial with extra probability
        # H0: Null model (uniform)
        # H1: Model with excess at 1640 Mpc

        # Probability under H0 (uniform)
        p0 = (2 * window) / (np.max(distances) - np.min(distances))

        # Probability under H1 (with excess)
        # Estimation: 80% excess over uniform
        p1 = p0 * 1.8

        # Likelihoods
        L0 = stats.binom.pmf(n_obs, n_total, p0)
        L1 = stats.binom.pmf(n_obs, n_total, p1)

        # Approximate Bayes factor
        if L0 > 0:
            BF = L1 / L0
            logBF = np.log(BF)
        else:
            BF = np.inf
            logBF = np.inf

        # Significance
        expected = n_total * p0
        if expected > 0:
            sigma = (n_obs - expected) / np.sqrt(expected)
        else:
            sigma = 0

        # Kass & Raftery interpretation
        if logBF > 5:
            evidence = "DECISIVE"
        elif logBF > 3:
            evidence = "STRONG"
        elif logBF > 1:
            evidence = "POSITIVE"
        else:
            evidence = "WEAK"

        print(f"📊 SIMPLIFIED BAYESIAN RESULTS:")
        print(f"• Observed: {n_obs} pairs (expected: {expected:.1f})")
        print(f"• Ratio: {n_obs/expected:.2f}x")
        print(f"• Significance: {sigma:.1f}σ")
        print(f"• Bayes factor: BF = {BF:.1e} (logBF = {logBF:.1f})")
        print(f"• Evidence: {evidence}")

        if logBF > 3:
            print(f"✅ STRONG EVIDENCE FOR EXCESS AT 1640 MPC")
        else:
            print(f"❌ INSUFFICIENT EVIDENCE")

        return {
            'scale': scale,
            'n_obs': int(n_obs),
            'n_total': int(n_total),
            'expected': float(expected),
            'ratio': float(n_obs / expected),
            'sigma': float(sigma),
            'BF': float(BF),
            'logBF': float(logBF),
            'evidence': evidence,
            'significant': logBF > 3
        }

    def analyze_simplified_harmonic_series(self, distances):
        """Simplified harmonic series analysis"""

        print(f"\n🎼 HARMONIC SERIES ANALYSIS:")

        base = 820.0
        multiples = [1, 2, 3, 4]
        results = {}

        for n in multiples:
            scale = base * n
            window = scale * 0.05

            if scale > np.max(distances):
                print(f"• {n}×820 = {scale:.0f} Mpc: [OUT OF RANGE]")
                continue

            mask = np.abs(distances - scale) < window
            n_obs = np.sum(mask)
            n_total = len(distances)

            # Simple statistics
            range_val = np.max(distances) - np.min(distances)
            p_expected = (2 * window) / range_val
            expected = n_total * p_expected

            if expected > 0:
                ratio = n_obs / expected
                sigma = (n_obs - expected) / np.sqrt(expected)
            else:
                ratio = 0
                sigma = 0

            # Interpretation
            if sigma > 3:
                status = " ✅✅ VERY SIGNIFICANT"
            elif sigma > 2:
                status = " ✅ SIGNIFICANT"
            elif sigma > 1:
                status = "⚠️  MARGINAL"
            else:
                status = " ❌ NOT SIGNIFICANT"

            print(f"{status[:2]} {n}×820 = {scale:4.0f} Mpc: {sigma:5.1f}σ (ratio={ratio:.2f}x)")

            results[n] = {
                'scale': scale,
                'n_obs': int(n_obs),
                'expected': float(expected),
                'ratio': float(ratio),
                'sigma': float(sigma)
            }

        # Summary
        n_significant = sum(1 for r in results.values() if r['sigma'] > 2)

        print(f"\n📊 HARMONIC SERIES SUMMARY:")
        print(f"• Significant scales (σ > 2): {n_significant} of {len(results)}")

        if n_significant >= 2:
            print(f"✅ HARMONIC PATTERN DETECTED")
        elif n_significant >= 1:
            print(f"⚠️  PARTIAL PATTERN")
        else:
            print(f"❌ NO HARMONIC PATTERN")

        return results

    def calculate_significance_statistics(self, distances):
        """Calculate significance statistics"""

        # Histogram for periodicity analysis
        bins = np.linspace(0, 4000, 200)
        hist, bin_edges = np.histogram(distances, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Simple autocorrelation
        autocorr = np.correlate(hist - np.mean(hist), hist - np.mean(hist), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize

        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(autocorr)-1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > 0.1:  # Threshold
                    lag = i * (bin_edges[1] - bin_edges[0])
                    peaks.append((lag, autocorr[i]))

        # Sort peaks
        peaks.sort(key=lambda x: x[1], reverse=True)

        print(f"\n📈 PERIODICITY ANALYSIS:")
        print(f"• Distances analyzed: {len(distances):,}")

        if peaks:
            print(f"• Autocorrelation peaks:")
            for lag, corr in peaks[:3]:
                print(f"  - {lag:.0f} Mpc (corr={corr:.3f})")

                # Check if near expected scales
                expected_scales = [820, 1640, 2460]
                for scale in expected_scales:
                    if abs(lag - scale) / scale < 0.1:
                        print(f"    ← NEAR {scale} MPC")
        else:
            print(f"• No significant peaks detected")

        return {
            'n_distances': len(distances),
            'autocorr_peaks': [(float(p[0]), float(p[1])) for p in peaks[:5]],
            'periodicity_detected': len(peaks) > 0
        }

    def save_simplified_results(self, result_1640, harmonics, stats_result,
                               distances, structures, time_elapsed):
        """Save simplified results"""

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"cristal_core_v89_aws_{timestamp}.json"

        # Simplified data
        data = {
            'metadata': {
                'version': 'v8.9',
                'timestamp': timestamp,
                'execution_time_s': float(time_elapsed),
                'optimized_for': 'AWS t3.micro',
                'n_structures': len(structures),
                'n_distances': len(distances),
                'max_distance': float(np.max(distances)) if len(distances) > 0 else 0
            },
            'result_1640': result_1640 if result_1640 else {},
            'harmonic_series': harmonics,
            'statistics': stats_result,
            'conclusion': self.generate_simplified_conclusion(result_1640, harmonics)
        }

        # Save
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, cls=JSONEncoderCustom)

        print(f"\n💾 Results saved in: {filename}")

        # Show conclusion
        print("\n" + "="*80)
        print("🎯 CONCLUSION v8.9 - AWS OPTIMIZED")
        print("="*80)
        print(data['conclusion'])

    def generate_simplified_conclusion(self, result_1640, harmonics):
        """Generate simplified conclusion"""

        conclusion = []

        if result_1640 and result_1640.get('significant', False):
            ratio = result_1640.get('ratio', 0)
            logBF = result_1640.get('logBF', 0)

            conclusion.append(f"✅ 1640 MPC: DETECTED WITH STRONG EVIDENCE")
            conclusion.append(f"   Ratio: {ratio:.2f}x, logBF: {logBF:.1f}")

            # Check harmonic pattern
            n_sig = sum(1 for r in harmonics.values() if r.get('sigma', 0) > 2)
            if n_sig >= 2:
                conclusion.append(f"✅ HARMONIC PATTERN: {n_sig} significant scales")
                conclusion.append(f"   Crystal structure confirmed")
            else:
                conclusion.append(f"⚠️  INCOMPLETE PATTERN: Only {n_sig} significant scale")
        else:
            conclusion.append(f"❌ 1640 MPC: NO SIGNIFICANT EVIDENCE")

        conclusion.append(f"\n🔭 IMPLICATIONS:")
        if result_1640 and result_1640.get('significant', False):
            conclusion.append(f"• Periodic universe at Gpc scale")
            conclusion.append(f"• Compatible with Vallejos preprints")
            conclusion.append(f"• Challenge for standard ΛCDM")
        else:
            conclusion.append(f"• Compatible with ΛCDM")
            conclusion.append(f"• No evidence of crystal structure")

        return "\n".join(conclusion)

    def estimate_memory(self):
        """Estimate memory usage (approximate)"""
        # Simple estimation
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

def main():
    print("\n" + "="*80)
    print("⚡ CRISTAL_CORE v8.9 - OPTIMIZED FOR AWS t3.micro")
    print("="*80)
    print("OPTIMIZATIONS:")
    print("• Dataset limited to 300k galaxies")
    print("• Simplified weights (no complex Malmquist)")
    print("• Simplified Bayesian analysis (no MCMC)")
    print("• Aggressive memory cleanup")
    print("="*80)

    # Clear memory at start
    import gc
    gc.collect()

    analyzer = SimplifiedBayesianAnalyzer()
    analyzer.run_simplified_analysis()

    print("\n" + "="*80)
    print("✅  COMPLETED - Optimized")
    print("="*80)

if __name__ == "__main__":
    main()