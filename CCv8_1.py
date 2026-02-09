#!/usr/bin/env python3
"""
CRISTAL_CORE - CORRECTED AND OPTIMIZED
Cosmic crystal structure analysis
"""
import numpy as np
import json
import time
import sys
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configure path for Rust module
sys.path.append('.')

try:
    import cristal_core as cc
    RUST_AVAILABLE = True
    print("✅ Rust module 'cristal_core' loaded correctly")
except ImportError as e:
    print(f"⚠️  Could not load Rust module: {e}")
    RUST_AVAILABLE = False

@dataclass
class Galaxy:
    ra: float
    dec: float
    z: float
    vdisp: float
    idx: int

class CrystalAnalyzerV9:
    def __init__(self, h0: float = 70.0):
        self.h0 = h0
        self.c = 299792.458
        self.galaxies: List[Galaxy] = []
        self.master_nodes: List[Tuple[float, float, float]] = []

        if RUST_AVAILABLE:
            self.stacker = cc.CristalStacker()
            print(f"✅ Rust stacker: omega_0={self.stacker.omega_0}, "
                  f"lambda_scale={self.stacker.lambda_scale} Mpc")

    def load_dataset(self) -> bool:
        """Load and filter optimized dataset"""
        print("\n📂 LOADING OPTIMIZED DATASET")
        print("="*50)

        try:
            data = np.load('sdss_vdisp_calidad.npz')

            # Filter DURING loading to save memory
            ra = data['RA'].astype(np.float32)
            dec = data['DEC'].astype(np.float32)
            z = data['Z'].astype(np.float32)
            vdisp = data['VDISP'].astype(np.float32)

            # Stricter filters for quality
            mask = (
                (z >= 0.05) & (z <= 0.7) &      # More reliable redshift
                (vdisp >= 100) & (vdisp <= 500) & # High quality VDISP
                (dec >= -10) & (dec <= 60) &     # Main SDSS region
                (ra >= 100) & (ra <= 250)        # Avoid edges
            )

            # Apply filter
            idxs = np.where(mask)[0]
            n_sample = min(500000, len(idxs))  # Limit for speed
            idxs = np.random.choice(idxs, n_sample, replace=False)

            # Create galaxy list
            self.galaxies = [
                Galaxy(ra=ra[i], dec=dec[i], z=z[i], vdisp=vdisp[i], idx=i)
                for i in idxs
            ]

            print(f"✅ Dataset loaded and filtered:")
            print(f"   • Random sample: {len(self.galaxies):,} galaxies")
            print(f"   • z: [{min(g.z for g in self.galaxies):.3f}, "
                  f"{max(g.z for g in self.galaxies):.3f}]")
            print(f"   • RA: [{min(g.ra for g in self.galaxies):.1f}°, "
                  f"{max(g.ra for g in self.galaxies):.1f}°]")

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def find_super_structures(self, radius_degrees: float = 2.0) -> List[Tuple[float, float, float]]:
        """Find super-structures using hierarchical clustering"""
        print(f"\n🔍 SEARCHING FOR SUPER-STRUCTURES (radius={radius_degrees}°)")
        print("="*50)

        if not self.galaxies:
            return []

        # Group by spatial cells
        ra_vals = np.array([g.ra for g in self.galaxies])
        dec_vals = np.array([g.dec for g in self.galaxies])
        z_vals = np.array([g.z for g in self.galaxies])

        # Finer grid for better resolution
        grid_size = 100
        ra_min, ra_max = ra_vals.min(), ra_vals.max()
        dec_min, dec_max = dec_vals.min(), dec_vals.max()

        ra_bins = np.linspace(ra_min, ra_max, grid_size + 1)
        dec_bins = np.linspace(dec_min, dec_max, grid_size + 1)

        # 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            ra_vals, dec_vals, bins=[ra_bins, dec_bins]
        )

        # Find density peaks (95th percentile)
        if np.sum(hist > 0) > 0:
            density_threshold = np.percentile(hist[hist > 0], 95)
        else:
            density_threshold = 0

        dense_cells = np.argwhere(hist > density_threshold)

        # Group nearby cells to avoid duplicates
        groups = []
        for i, j in dense_cells:
            ra_cell = (x_edges[i] + x_edges[i+1]) / 2
            dec_cell = (y_edges[j] + y_edges[j+1]) / 2

            # Check if near existing group
            added = False
            for group in groups:
                ra_g, dec_g = group['center']
                if abs(ra_cell - ra_g) < radius_degrees and abs(dec_cell - dec_g) < radius_degrees:
                    # Merge with existing group
                    group['cells'].append((i, j))
                    group['center'] = (
                        (group['center'][0] + ra_cell) / 2,
                        (group['center'][1] + dec_cell) / 2
                    )
                    added = True
                    break

            if not added:
                groups.append({
                    'cells': [(i, j)],
                    'center': (ra_cell, dec_cell)
                })

        # Calculate properties of each super-structure
        nodes = []
        for group in groups:
            ra_center, dec_center = group['center']

            # Find galaxies within radius
            nearby_galaxies = []
            for g in self.galaxies:
                dist_ra = abs(g.ra - ra_center)
                if dist_ra > 180:  # Handle RA wraparound
                    dist_ra = 360 - dist_ra

                if dist_ra < radius_degrees and abs(g.dec - dec_center) < radius_degrees:
                    nearby_galaxies.append(g)

            if len(nearby_galaxies) >= 30:  # Minimum to consider structure
                z_average = np.mean([g.z for g in nearby_galaxies])
                nodes.append((ra_center, dec_center, z_average))

        self.master_nodes = nodes
        print(f"✅ Super-structures identified: {len(nodes)}")

        return nodes

    def analyze_detailed_periodicity(self) -> Dict:
        """Detailed spatial periodicity analysis"""
        print(f"\n📊 DETAILED PERIODICITY ANALYSIS")
        print("="*50)

        if len(self.master_nodes) < 20:
            print("❌ Too few structures for analysis")
            return {}

        # Calculate complete distance matrix
        n = len(self.master_nodes)
        distances = []

        for i in range(n):
            ra1, dec1, z1 = self.master_nodes[i]
            for j in range(i+1, n):
                ra2, dec2, z2 = self.master_nodes[j]

                # Angular distance (simplified for small distances)
                dra = abs(ra1 - ra2)
                if dra > 180:
                    dra = 360 - dra
                ddec = abs(dec1 - dec2)
                dist_degrees = np.sqrt(dra**2 + ddec**2)

                # Convert to Mpc
                z_mean = (z1 + z2) / 2
                d_comov = (self.c / self.h0) * np.log(1 + z_mean)
                dist_mpc = d_comov * np.radians(dist_degrees)

                distances.append(dist_mpc)

        print(f"• Distances calculated: {len(distances):,}")

        # Analysis of multiple scales
        vallejo_scales = {
            'Base': 820.0,
            'Double': 1640.0,
            'Triple': 2460.0,
            'Quadruple': 3280.0,
            'Quintuple': 4100.0,
            'Vallejos_original': 841.0,
            'Half': 410.0,
            'Quarter': 205.0
        }

        results = {}
        dist_max = max(distances) if distances else 1

        for name, scale in vallejo_scales.items():
            if scale > dist_max * 1.1:  # Scale too large
                continue

            # Count in 5% window
            window = scale * 0.05
            count = sum(1 for d in distances if abs(d - scale) < window)

            # Expected statistics (Poisson)
            relative_window = window / dist_max
            expected = len(distances) * 2 * relative_window

            # Calculate significance (sigma)
            if expected > 0:
                sigma = (count - expected) / np.sqrt(expected)
            else:
                sigma = 0

            results[name] = {
                'scale_mpc': scale,
                'observed': count,
                'expected': expected,
                'sigma': sigma,
                'ratio': count / expected if expected > 0 else 0,
                'significant': abs(sigma) > 3.0
            }

        # Display results
        print(f"\n{'='*60}")
        print("PERIODICITY RESULTS:")
        print('='*60)

        for name, res in results.items():
            if res['sigma'] > 2.0 or res['ratio'] > 1.5:
                symbol = "✅" if res['sigma'] > 3.0 else "⚠️ "
                print(f"{symbol} {name:15} {res['scale_mpc']:6.0f} Mpc: "
                      f"{res['observed']:4d} pairs (σ={res['sigma']:.1f}, "
                      f"ratio={res['ratio']:.2f}x)")

        return results

    def analyze_with_optimized_rust(self) -> Dict:
        """Use optimized Rust functions"""
        if not RUST_AVAILABLE or not self.master_nodes:
            return {}

        print(f"\n⚡ OPTIMIZED RUST ANALYSIS")
        print("="*50)

        try:
            # Convert to Rust format
            rust_nodes = [list(n) for n in self.master_nodes]

            # CORRECTED: Remove omega_m parameter that doesn't exist
            print("1. Analyzing packing density...")
            density_result = self.stacker.analizar_densidad_empaquetamiento(
                nodos=rust_nodes,
                h0=self.h0
                # ¡Don't include omega_m! It's not a function parameter
            )

            print(f"   • Mean separation: {density_result.get('separacion_mpc', 0):.1f} Mpc")

            # Anisotropy analysis if enough galaxies
            if len(self.galaxies) > 1000:
                print("\n2. Analyzing anisotropy...")

                # Take galaxy sample for Rust
                galaxy_sample = []
                for g in self.galaxies[:5000]:  # Sample
                    galaxy_sample.append([g.ra, g.dec, g.z, g.vdisp, 0.0])

                # Use first node as center
                center = self.master_nodes[0]

                # Calculate radius in degrees for ~820 Mpc
                d_comov = (self.c / self.h0) * np.log(1 + center[2])
                radius_degrees = np.degrees(820.0 / d_comov)

                anisotropy_result = self.stacker.analizar_anisotropia_hexagonal(
                    galaxias=galaxy_sample,
                    centro=list(center),
                    radio_anillo=radius_degrees,
                    ancho_anillo=radius_degrees * 0.1
                )

                p_val = anisotropy_result.get('p_rayleigh', 1.0)
                print(f"   • Rayleigh p-value: {p_val:.2e}")
                print(f"   • Significant: {'YES' if p_val < 0.01 else 'NO'}")

            return density_result

        except Exception as e:
            print(f"⚠️  Rust error: {e}")
            return {}

    def run_complete_analysis(self):
        """Run optimized complete analysis"""
        print("\n" + "="*70)
        print("🚀 CRYSTALLINE ANALYSIS V9 - OPTIMIZED")
        print("="*70)

        start_time = time.time()

        # 1. Load data
        if not self.load_dataset():
            return

        # 2. Find super-structures
        self.find_super_structures(radius_degrees=1.5)

        # 3. Detailed periodicity analysis
        periodicity_results = self.analyze_detailed_periodicity()

        # 4. Analysis with Rust (if available)
        rust_results = self.analyze_with_optimized_rust()

        # 5. Advanced hierarchical analysis
        if RUST_AVAILABLE and len(self.galaxies) > 1000 and len(self.master_nodes) > 10:
            print(f"\n🏗️  ADVANCED HIERARCHICAL ANALYSIS")
            print("="*50)

            # Prepare data for Rust
            micro_nodes = []
            for g in self.galaxies[:2000]:  # Limited sample
                micro_nodes.append([g.ra, g.dec, g.z])

            macro_nodes = [list(n) for n in self.master_nodes]

            try:
                # Hierarchical inclusion analysis
                print("1. Hierarchical inclusion...")
                inclusion_result = self.stacker.analizar_inclusion_jerarquica(
                    micro_nodos=micro_nodes,
                    macro_nodos=macro_nodes,
                    radio_inclusion_mpc=30.0  # Smaller radius
                )

                if inclusion_result:
                    efficiency = inclusion_result.get('eficiencia_asignacion', [0])[0]
                    dimension = inclusion_result.get('dimension_fractal', [0])[0]
                    print(f"   • Efficiency: {efficiency:.1%}")
                    print(f"   • Fractal dimension: {dimension:.2f}")

                # Void mass analysis
                print("\n2. Super-void mass...")
                mass_result = self.stacker.calcular_masa_efectiva_vacios(
                    micro_nodos=micro_nodes,
                    macro_centros=macro_nodes,
                    radio_void_mpc=100.0
                )

                if mass_result and 'masa_promedio_mo' in mass_result:
                    avg_mass = mass_result['masa_promedio_mo'][0]
                    print(f"   • Average mass: {avg_mass:.2e} M☉")

            except Exception as e:
                print(f"⚠️  Error in hierarchical analysis: {e}")

        # Final summary
        elapsed = time.time() - start_time

        print(f"\n" + "="*70)
        print("🎯 FINAL SUMMARY")
        print("="*70)

        print(f"\n📊 STATISTICS:")
        print(f"• Total time: {elapsed:.1f}s")
        print(f"• Galaxies analyzed: {len(self.galaxies):,}")
        print(f"• Super-structures: {len(self.master_nodes)}")

        print(f"\n🔍 HIGHLIGHTED PERIODICITY:")
        for name, res in periodicity_results.items():
            if res['sigma'] > 2.5 or res['ratio'] > 1.8:
                print(f"• {name:15} → {res['sigma']:.1f}σ (ratio={res['ratio']:.2f}x)")

        # Interpretation
        print(f"\n💡 INTERPRETATION:")
        if any(r['sigma'] > 3.0 for r in periodicity_results.values()):
            print("✅ STRONG EVIDENCE of cosmic periodicity")
        elif any(r['sigma'] > 2.0 for r in periodicity_results.values()):
            print("⚠️  HINT of periodicity (needs more data)")
        else:
            print("❌ No significant periodicity detected")

        # Save results
        self.save_results(periodicity_results, rust_results)

    def save_results(self, periodicity: Dict, rust_results: Dict):
        """Save results to JSON"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"resultados_cristal_{timestamp}.json"

        results = {
            'metadata': {
                'version': 'v9',
                'timestamp': timestamp,
                'h0': self.h0,
                'n_galaxies': len(self.galaxies),
                'n_structures': len(self.master_nodes)
            },
            'periodicity': periodicity,
            'rust_analysis': rust_results,
            'structures': self.master_nodes
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {filename}")

def main():
    print("\n" + "="*70)
    print("🔭 CRYSTAL CORE - PERIODIC STRUCTURE ANALYSIS")
    print("="*70)

    analyzer = CrystalAnalyzerV9(h0=70.0)
    analyzer.run_complete_analysis()

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETED")
    print("="*70)

if __name__ == "__main__":
    main()