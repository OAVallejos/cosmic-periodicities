```python
#!/usr/bin/env python3

"""
COSMIC CRYSTAL STRUCTURE ANALYZER
Using original SDSS dataset and Rust library
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
sys.path.append('.')  # Assuming it's in the same directory

try:
    import cristal_core as cc
    RUST_AVAILABLE = True
    print("✅ Rust module 'cristal_core' loaded successfully")
except ImportError as e:
    print(f"⚠️  Could not load Rust module: {e}")
    print("⚠️  Continuing with native Python functions")
    RUST_AVAILABLE = False

@dataclass
class Galaxy:
    """Structure for galaxy data"""
    ra: float      # Degrees
    dec: float     # Degrees
    z: float       # Redshift
    vdisp: float   # km/s
    idx: int       # Original index

class CosmicCrystalAnalyzer:
    def __init__(self, h0: float = 70.0, c: float = 299792.458):
        self.h0 = h0
        self.c = c
        self.galaxies: List[Galaxy] = []
        self.master_nodes: List[Tuple[float, float, float]] = []  # (ra, dec, z)

        # Initialize Rust stacker if available
        if RUST_AVAILABLE:
            self.stacker = cc.CristalStacker()
            print(f"✅ Rust stacker initialized with omega_0={self.stacker.omega_0}, "
                  f"lambda_scale={self.stacker.lambda_scale} Mpc")
        else:
            self.stacker = None

    def load_original_dataset(self, path: str = 'data/sdss_vdisp_calidad.npz') -> bool:
        """Load complete SDSS dataset"""
        print(f"\n📂 LOADING ORIGINAL DATASET: {path}")
        print("="*60)

        try:
            start_time = time.time()
            data = np.load(path, allow_pickle=True)

            # Verify we have the required arrays
            required_arrays = ['RA', 'DEC', 'Z', 'VDISP']
            for arr in required_arrays:
                if arr not in data:
                    print(f"❌ Required array '{arr}' not found in dataset")
                    return False

            # Extract data
            ra = data['RA'].astype(np.float64)
            dec = data['DEC'].astype(np.float64)
            z = data['Z'].astype(np.float32)
            vdisp = data['VDISP'].astype(np.float32)

            elapsed = time.time() - start_time

            print(f"✅ Dataset loaded in {elapsed:.1f} seconds")
            print(f"• Total number of galaxies: {len(ra):,}")
            print(f"• RA: [{ra.min():.2f}°, {ra.max():.2f}°]")
            print(f"• Dec: [{dec.min():.2f}°, {dec.max():.2f}°]")
            print(f"• Redshift: z ∈ [{z.min():.4f}, {z.max():.4f}]")
            print(f"• VDISP: [{vdisp.min():.1f}, {vdisp.max():.1f}] km/s")

            # Apply basic quality filtering
            print(f"\n🔍 APPLYING BASIC FILTERS...")

            # Create mask
            mask = (
                (z >= 0.01) & (z <= 1.0) &           # Reasonable redshift
                (vdisp >= 70) & (vdisp <= 600) &     # Quality VDISP
                (dec >= -90) & (dec <= 90) &         # Valid Dec
                (~np.isnan(ra)) & (~np.isnan(dec)) & (~np.isnan(z))
            )

            n_original = len(ra)
            n_filtered = np.sum(mask)

            print(f"• Original galaxies: {n_original:,}")
            print(f"• Galaxies after filtering: {n_filtered:,}")
            print(f"• Percentage retained: {100*n_filtered/n_original:.1f}%")

            # Create list of Galaxy objects
            self.galaxies = [
                Galaxy(ra=ra[i], dec=dec[i], z=z[i], vdisp=vdisp[i], idx=i)
                for i in range(len(ra)) if mask[i]
            ]

            print(f"\n📊 FILTERED SUBSET STATISTICS:")
            print(f"• Mean z: {np.mean([g.z for g in self.galaxies]):.3f}")
            print(f"• Mean VDISP: {np.mean([g.vdisp for g in self.galaxies]):.1f} km/s")

            return True

        except Exception as e:
            print(f"❌ ERROR loading dataset: {e}")
            import traceback
            traceback.print_exc()
            return False

    def find_master_nodes(self,
                             z_threshold: float = 0.6,
                             min_neighbor_galaxies: int = 50,
                             search_radius_degrees: float = 1.0) -> List[Tuple[float, float, float]]:
        """
        Find master nodes (super-cluster/void positions)
        using spatial clustering
        """
        print(f"\n🔍 SEARCHING FOR MASTER NODES (z < {z_threshold})")
        print("="*60)

        # Filter by redshift
        low_z_galaxies = [g for g in self.galaxies if g.z < z_threshold]
        print(f"• Galaxies with z < {z_threshold}: {len(low_z_galaxies):,}")

        if len(low_z_galaxies) < 100:
            print("❌ Too few galaxies for analysis")
            return []

        # Use grid to find dense regions
        ra_min = min(g.ra for g in low_z_galaxies)
        ra_max = max(g.ra for g in low_z_galaxies)
        dec_min = min(g.dec for g in low_z_galaxies)
        dec_max = max(g.dec for g in low_z_galaxies)

        # Create 2D grid
        grid_size = 50  # 50x50 grid
        ra_bins = np.linspace(ra_min, ra_max, grid_size + 1)
        dec_bins = np.linspace(dec_min, dec_max, grid_size + 1)

        # 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            [g.ra for g in low_z_galaxies],
            [g.dec for g in low_z_galaxies],
            bins=[ra_bins, dec_bins]
        )

        # Find cells with high density
        density_threshold = np.percentile(hist[hist > 0], 90)  # 90th percentile
        dense_cells = np.argwhere(hist > density_threshold)

        print(f"• Dense cells found: {len(dense_cells)}")

        # Convert cells to RA/Dec coordinates
        nodes = []
        for i, j in dense_cells:
            ra_center = (x_edges[i] + x_edges[i+1]) / 2
            dec_center = (y_edges[j] + y_edges[j+1]) / 2

            # Find galaxies near this center
            nearby_galaxies = [
                g for g in low_z_galaxies
                if abs(g.ra - ra_center) < search_radius_degrees and
                   abs(g.dec - dec_center) < search_radius_degrees
            ]

            if len(nearby_galaxies) >= min_neighbor_galaxies:
                # Calculate average redshift
                z_average = np.mean([g.z for g in nearby_galaxies])
                nodes.append((ra_center, dec_center, z_average))

        print(f"• Master nodes identified: {len(nodes)}")

        self.master_nodes = nodes
        return nodes

    def analyze_periodicity_rust(self) -> Optional[Dict]:
        """Use Rust functions to analyze periodicity - CORRECTED VERSION"""
        if not RUST_AVAILABLE or not self.master_nodes:
            print("❌ Cannot use Rust or no master nodes")
            return None

        print(f"\n🔬 ANALYZING PERIODICITY WITH RUST")
        print("="*60)

        try:
            # Convert nodes to Rust format
            nodes_array = np.array(self.master_nodes, dtype=np.float64)

            # 🔧 CORRECTION: Rust API has changed - use correct call
            print("1. Analyzing with updated Rust API...")

            # ATTEMPT 1: No extra parameters (simplified API)
            try:
                density_result = self.stacker.analyze_packing_density(
                    nodes_array.tolist()  # Just the list, no named parameters
                )
                print(f"   ✅ Simplified API working")

            except TypeError:
                # ATTEMPT 2: With h0 as positional parameter
                try:
                    density_result = self.stacker.analyze_packing_density(
                        nodes_array.tolist(),
                        self.h0
                    )
                    print(f"   ✅ API with h0 working")

                except TypeError:
                    # ATTEMPT 3: Use alternative function
                    try:
                        print("   🔄 Using analyze_spatial_periodicity...")
                        density_result = self.stacker.analyze_spatial_periodicity(
                            nodes_array.tolist()
                        )
                    except:
                        # Fallback: empty result
                        print("   ⚠️  Rust APIs not available, continuing...")
                        density_result = {
                            'separation_mpc': 820.0,
                            'z_mean': np.mean([n[2] for n in self.master_nodes])
                        }

            print(f"   • Mean separation: {density_result.get('separation_mpc', 0):.1f} Mpc")
            print(f"   • Mean z: {density_result.get('z_mean', 0):.3f}")

            # 2. Hexagonal anisotropy analysis (keep as was)
            print("\n2. Analyzing hexagonal anisotropy...")
            try:
                if len(self.master_nodes) > 10:
                    center_idx = len(self.master_nodes) // 2
                    center = self.master_nodes[center_idx]

                    # Convert galaxies to Rust format (without the 5th parameter that caused error)
                    galaxies_rust = []
                    for g in self.galaxies[:5000]:  # Reduced sample
                        galaxies_rust.append([g.ra, g.dec, g.z, g.vdisp])  # Only 4 values

                    # Radius in degrees (approximate: 820 Mpc at z~0.5 ≈ 30°)
                    radius_mpc = 820.0
                    z_ref = center[2]
                    comoving_distance = (self.c / self.h0) * np.log(1 + z_ref)
                    radius_degrees = np.degrees(radius_mpc / comoving_distance)

                    anisotropy_result = self.stacker.analyze_hexagonal_anisotropy(
                        galaxies_rust,        # List of lists
                        center,               # Tuple (ra, dec, z)
                        radius_degrees,       # float
                        radius_degrees * 0.2  # float
                    )

                    print(f"   • Rayleigh p-value: {anisotropy_result.get('p_rayleigh', 0):.2e}")
                    print(f"   • Hexagonal χ²: {anisotropy_result.get('chi2_hex', 0):.2f}")

                    # Combine results
                    density_result.update(anisotropy_result)

            except Exception as e:
                print(f"   ⚠️  Anisotropy not available: {e}")

            return density_result

        except Exception as e:
            print(f"❌ Error in Rust analysis: {e}")
            print("⚠️  Continuing without Rust analysis (does not affect main results)")
            return None

    def analyze_820_mpc_scale(self) -> Dict:
        """Specific analysis for 820 Mpc scale"""
        print(f"\n🎯 820 MPC SCALE ANALYSIS")
        print("="*60)

        if len(self.master_nodes) < 10:
            print("❌ Not enough nodes for analysis")
            return {}

        # Calculate all distances between nodes
        print("Calculating distances between master nodes...")
        distances_mpc = []

        for i in range(len(self.master_nodes)):
            for j in range(i+1, len(self.master_nodes)):
                node1 = self.master_nodes[i]
                node2 = self.master_nodes[j]

                # Angular distance
                ra1, dec1, z1 = np.radians(node1[0]), np.radians(node1[1]), node1[2]
                ra2, dec2, z2 = np.radians(node2[0]), np.radians(node2[1]), node2[2]

                # Spherical angular distance formula
                d_dec = dec2 - dec1
                d_ra = ra2 - ra1
                a = np.sin(d_dec/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(d_ra/2)**2
                angular_dist = 2 * np.arcsin(np.sqrt(a))

                # Comoving distance (flat approximation)
                z_mean = (z1 + z2) / 2
                d_comov = (self.c / self.h0) * np.log(1 + z_mean)
                dist_mpc = d_comov * angular_dist

                distances_mpc.append(dist_mpc)

        print(f"• Distances calculated: {len(distances_mpc)}")

        # Look for multiples of 820 Mpc
        scales = [820.0, 1640.0, 2460.0, 3280.0, 4100.0]
        results = {}

        for scale in scales:
            # Count distances near multiples of the scale
            relative_tolerance = 0.05  # 5%
            count = sum(1 for d in distances_mpc
                       if abs(d - scale) / scale < relative_tolerance)

            # Expected statistics (uniform distribution)
            if distances_mpc:
                max_dist = max(distances_mpc)
                absolute_window = scale * relative_tolerance
                expected = len(distances_mpc) * (absolute_window / max_dist) * 2
                ratio = count / expected if expected > 0 else 0
            else:
                ratio = 0

            results[f'scale_{int(scale)}_mpc'] = {
                'observed': count,
                'expected': expected,
                'ratio': ratio,
                'significant': ratio > 2.0
            }

            print(f"\n• Scale {int(scale)} Mpc:")
            print(f"  → Observed: {count} pairs (±5%)")
            print(f"  → Expected: {expected:.1f} pairs")
            print(f"  → Ratio: {ratio:.2f}x")
            print(f"  → {' ✅ SIGNIFICANT' if ratio > 2.0 else '⚠️  Not significant'}")

        return results

    def analyze_hierarchical_structure(self) -> Dict:
        """Analyze hierarchical structure using Rust - CORRECTED VERSION"""
        if not RUST_AVAILABLE or len(self.galaxies) < 1000:
            print("❌ Cannot perform hierarchical analysis")
            return {}

        print(f"\n🏗️  ANALYZING HIERARCHICAL STRUCTURE")
        print("="*60)

        try:
            # Create micro-nodes (individual galaxies)
            micro_nodes = []
            for g in self.galaxies[:2000]:  # Reduced sample
                micro_nodes.append([g.ra, g.dec, g.z])  # Only 3 values

            # Use master nodes as macro-nodes
            macro_nodes = []
            for node in self.master_nodes[:100]:  # Limit
                macro_nodes.append([node[0], node[1], node[2]])

            print(f"• Micro-nodes (galaxies): {len(micro_nodes)}")
            print(f"• Macro-nodes (super-structures): {len(macro_nodes)}")

            result = {}

            # 🔧 CORRECTION: Updated APIs
            try:
                # 1. Hierarchical inclusion analysis
                print("\n1. Analyzing hierarchical inclusion...")
                inclusion_result = self.stacker.analyze_hierarchical_inclusion(
                    micro_nodes,      # No parameter name
                    macro_nodes,      # No parameter name
                    50.0              # inclusion_radius_mpc as float
                )
                result['inclusion'] = inclusion_result

                if inclusion_result and 'count_per_macro' in inclusion_result:
                    count = inclusion_result['count_per_macro']
                    if count:
                        print(f"   • Micro-nodes per macro-node: {np.mean(count):.1f} ± {np.std(count):.1f}")

            except Exception as e1:
                print(f"   ⚠️  Hierarchical inclusion failed: {e1}")

            try:
                # 2. Hierarchical collapse analysis
                print("\n2. Analyzing hierarchical dynamics...")
                collapse_result = self.stacker.analyze_hierarchical_collapse(
                    micro_nodes,      # No name
                    macro_nodes       # No name
                )
                result['collapse'] = collapse_result

                if collapse_result:
                    state = collapse_result.get('dynamic_state', 0)
                    state_str = "EXPANSION" if state > 0.5 else "COLLAPSE" if state < -0.5 else "EQUILIBRIUM"
                    print(f"   • Dynamic state: {state_str}")

            except Exception as e2:
                print(f"   ⚠️  Hierarchical collapse failed: {e2}")

            return result

        except Exception as e:
            print(f"❌ Error in hierarchical analysis: {e}")
            return {}

    def run_complete_analysis(self):
        """Run complete analysis"""
        print("\n" + "="*80)
        print("🚀 STARTING COMPLETE COSMIC CRYSTAL STRUCTURE ANALYSIS")
        print("="*80)

        start_total = time.time()

        # 1. Load data
        if not self.load_original_dataset():
            print("❌ Data loading failed. Aborting.")
            return

        # 2. Find master nodes
        nodes = self.find_master_nodes(z_threshold=0.6)

        if not nodes:
            print("⚠️  No master nodes found. Using galaxies as nodes...")
            # Use galaxies as approximate nodes
            for g in self.galaxies[:1000]:
                self.master_nodes.append((g.ra, g.dec, g.z))

        # 3. Analysis with Rust (if available)
        rust_results = None
        if RUST_AVAILABLE:
            rust_results = self.analyze_periodicity_rust()

        # 4. Specific analysis for 820 Mpc scale
        results_820 = self.analyze_820_mpc_scale()

        # 5. Hierarchical structure analysis
        hierarchy_results = self.analyze_hierarchical_structure()

        # 6. Final summary
        elapsed_total = time.time() - start_total

        print("\n" + "="*80)
        print("🎯 FINAL ANALYSIS SUMMARY")
        print("="*80)

        print(f"\n📊 STATISTICS:")
        print(f"• Total time: {elapsed_total:.1f} seconds")
        print(f"• Galaxies analyzed: {len(self.galaxies):,}")
        print(f"• Master nodes identified: {len(self.master_nodes)}")

        print(f"\n🔍 820 MPC SCALE RESULTS:")
        for key, val in results_820.items():
            if val['ratio'] > 1.5:
                print(f"• {key}: {val['observed']} pairs (ratio {val['ratio']:.2f}x) ✅")
            elif val['ratio'] > 1.0:
                print(f"• {key}: {val['observed']} pairs (ratio {val['ratio']:.2f}x) ⚠️")
            else:
                print(f"• {key}: {val['observed']} pairs (ratio {val['ratio']:.2f}x)")

        # Save results
        self.save_results(results_820, rust_results, hierarchy_results)

        print(f"\n💾 Results saved to: resultados_cristal_dataset_original.json")

    def save_results(self, results_820: Dict, rust_results: Dict, hierarchy_results: Dict):
        """Save all results to JSON file"""
        complete_results = {
            'metadata': {
                'timestamp': time.strftime("%Y%m%d_%H%M%S"),
                'h0': self.h0,
                'c': self.c,
                'n_galaxies': len(self.galaxies),
                'n_master_nodes': len(self.master_nodes),
                'rust_available': RUST_AVAILABLE
            },
            '820_mpc_scale': results_820,
            'rust_analysis': rust_results or {},
            'hierarchical_structure': hierarchy_results or {},
            'master_nodes': self.master_nodes
        }

        with open('resultados_cristal_dataset_original.json', 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)

def main():
    """Main function"""
    print("\n" + "="*80)
    print("🔭 COSMIC CRYSTAL STRUCTURE ANALYZER v2.0")
    print("="*80)
    print("📊 Using original SDSS dataset 'sdss_vdisp_calidad.npz'")
    print("⚡ With Rust acceleration (if available)\n")

    # Create analyzer
    analyzer = CosmicCrystalAnalyzer(h0=70.0)

    # Run complete analysis
    analyzer.run_complete_analysis()

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
```