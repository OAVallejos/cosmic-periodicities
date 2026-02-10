#!/usr/bin/env python3     
"""
- USING RUST KERNEL for evolutionary analysis
MULTI-DATASET VERSION: SDSS + DESI with known columns
"""

import numpy as np
import json
import time
import sys
import warnings
import os
warnings.filterwarnings('ignore')

from astropy.io import fits
from astropy.table import Table

sys.path.append('.')

try:
    import cristal_core as cc
    RUST_AVAILABLE = True
    print("✅ Rust module 'cristal_core' loaded")
except ImportError:
    RUST_AVAILABLE = False
    print("⚠️  Rust kernel not available, running in pure Python mode")

class EvolutionaryRustAnalyzer:
    """Evolutionary analysis USING RUST KERNEL - Multi-dataset"""

    def __init__(self, H0: float = 70.0):
        self.H0 = H0
        self.c = 299792.458

        if RUST_AVAILABLE:
            self.stacker = cc.CristalStacker()
            print("✅ Rust stacker initialized")
        else:
            print("⚠️  Running without Rust acceleration")

    def load_sdss_npz(self, file='data/sdss_vdisp_calidad.npz'):
        """Load SDSS dataset in .npz format"""
        if not os.path.exists(file):
            print(f"❌ File {file} not found")
            return None

        print(f"\n📁 LOADING SDSS: {file}")
        try:
            data = np.load(file)

            # Extract known columns
            ra = data['RA']
            dec = data['DEC']
            z = data['Z']
            vdisp = data['VDISP']

            print(f"✅ SDSS loaded: {len(ra):,} galaxies")
            print(f"  • RA: {ra.min():.2f} - {ra.max():.2f}")
            print(f"  • DEC: {dec.min():.2f} - {dec.max():.2f}")
            print(f"  • z: {z.min():.3f} - {z.max():.3f}")
            print(f"  • VDISP: {vdisp.min():.1f} - {vdisp.max():.1f} km/s")

            return {
                'RA': ra,
                'DEC': dec,
                'Z': z,
                'VDISP': vdisp,
                'dataset': 'SDSS'
            }

        except Exception as e:
            print(f"❌ Error loading SDSS: {e}")
            return None

    def load_desi_fits(self, file='data/DATASET_LRG_VDISP_FLUXR_FINAL.fits'):
        """Load DESI dataset in .fits format"""
        if not os.path.exists(file):
            print(f"❌ File {file} not found")
            return None

        print(f"\n📁 LOADING DESI: {file}")
        try:
            # Use astropy to read FITS
            table = Table.read(file, format='fits')

            # Extract known columns
            ra = table['RA'].data
            dec = table['DEC'].data
            z = table['Z'].data
            vdisp = table['VDISP'].data

            print(f"✅ DESI loaded: {len(ra):,} galaxies")
            print(f"  • RA: {ra.min():.2f} - {ra.max():.2f}")
            print(f"  • DEC: {dec.min():.2f} - {dec.max():.2f}")
            print(f"  • z: {z.min():.3f} - {z.max():.3f}")
            print(f"  • VDISP: {vdisp.min():.1f} - {vdisp.max():.1f} km/s")

            return {
                'RA': ra,
                'DEC': dec,
                'Z': z,
                'VDISP': vdisp,
                'dataset': 'DESI',
                'table': table  # Save complete table for other columns
            }

        except Exception as e:
            print(f"❌ Error loading DESI: {e}")
            return None

    def combine_datasets(self, sdss_data, desi_data, max_galaxies=300000):
        """Combine SDSS and DESI for joint analysis"""
        print("\n🔗 COMBINING SDSS + DESI DATASETS")

        all_data = []
        datasets_info = []

        if sdss_data is not None:
            n_sdss = len(sdss_data['RA'])
            sample_sdss = min(150000, n_sdss) if max_galaxies else n_sdss
            indices_sdss = np.random.choice(n_sdss, sample_sdss, replace=False)

            all_data.append({
                'RA': sdss_data['RA'][indices_sdss],
                'DEC': sdss_data['DEC'][indices_sdss],
                'Z': sdss_data['Z'][indices_sdss],
                'VDISP': sdss_data['VDISP'][indices_sdss],
                'dataset': 'SDSS'
            })
            datasets_info.append(f"SDSS: {sample_sdss:,} galaxies")

        if desi_data is not None:
            n_desi = len(desi_data['RA'])
            sample_desi = min(150000, n_desi) if max_galaxies else n_desi
            indices_desi = np.random.choice(n_desi, sample_desi, replace=False)

            all_data.append({
                'RA': desi_data['RA'][indices_desi],
                'DEC': desi_data['DEC'][indices_desi],
                'Z': desi_data['Z'][indices_desi],
                'VDISP': desi_data['VDISP'][indices_desi],
                'dataset': 'DESI'
            })
            datasets_info.append(f"DESI: {sample_desi:,} galaxies")

        if not all_data:
            print("❌ No data to combine")
            return None

        # Combine all data
        combined = {
            'RA': np.concatenate([d['RA'] for d in all_data]),
            'DEC': np.concatenate([d['DEC'] for d in all_data]),
            'Z': np.concatenate([d['Z'] for d in all_data]),
            'VDISP': np.concatenate([d['VDISP'] for d in all_data]),
            'datasets': datasets_info
        }

        print(f"✅ Total combined: {len(combined['RA']):,} galaxies")
        print(f"  • z range: {combined['Z'].min():.3f} - {combined['Z'].max():.3f}")
        print(f"  • Sources: {', '.join(datasets_info)}")

        return combined

    def run_rust_analysis(self, use_sdss=True, use_desi=True, max_galaxies=200000):
        """Run analysis USING RUST KERNEL with multiple datasets"""

        print("\n" + "="*80)
        print("⚙️  CRISTAL_CORE v8.10_RUST - KERNEL ACTIVATED")
        print("="*80)
        print("USING: Rust kernel for real clustering")
        print("OBJECTIVE: Evolutionary analysis with SDSS + DESI")
        print("="*80)

        if not RUST_AVAILABLE:
            print("❌ Rust kernel NOT available")
            return

        start_time = time.time()

        # 1. LOAD DATASETS
        sdss_data = None
        desi_data = None

        if use_sdss:
            sdss_data = self.load_sdss_npz()

        if use_desi:
            desi_data = self.load_desi_fits()

        # 2. COMBINE DATASETS
        combined_data = self.combine_datasets(sdss_data, desi_data, max_galaxies)

        if combined_data is None:
            print("❌ No data to analyze")
            return

        ra = combined_data['RA'].astype(np.float32)
        dec = combined_data['DEC'].astype(np.float32)
        z = combined_data['Z'].astype(np.float32)
        vdisp = combined_data['VDISP'].astype(np.float32)

        print(f"\n✅ Combined sample for Rust: {len(ra):,} galaxies")
        print(f"• z range: {z.min():.3f} - {z.max():.3f}")

        # 3. ANALYZE DIFFERENT EPOCHS WITH RUST
        print("\n" + "="*80)
        print("🕰️  ANALYSIS BY EPOCHS (USING RUST)")
        print("="*80)

        epochs = [
            ('High_z', 0.8, 1.2),    # z ~ 1.0
            ('Medium_z', 0.4, 0.8),   # z ~ 0.6
            ('Low_z', 0.1, 0.4)     # z ~ 0.25
        ]

        results = {}

        for name, z_min, z_max in epochs:
            print(f"\n🔬 EPOCH: {name} (z={z_min}-{z_max})")

            # Filter
            mask = (z >= z_min) & (z <= z_max)
            ra_epoch = ra[mask]
            dec_epoch = dec[mask]
            z_epoch = z[mask]
            vdisp_epoch = vdisp[mask]

            if len(ra_epoch) < 5000:
                print(f"   ❌ Too few galaxies: {len(ra_epoch):,}")
                continue

            print(f"  • Galaxies: {len(ra_epoch):,}")
            print(f"  • Mean VDISP: {vdisp_epoch.mean():.1f} km/s")

            # Prepare for Rust
            galaxies_rust = []
            n_for_rust = min(40000, len(ra_epoch))

            for i in range(n_for_rust):
                galaxies_rust.append([
                    float(ra_epoch[i]),
                    float(dec_epoch[i]),
                    float(z_epoch[i]),
                    float(vdisp_epoch[i])
                ])

            # Call Rust for clustering
            print(f"  🏗️  Running Rust clustering...")

            try:
                rust_result = self.stacker.identificar_super_estructuras(
                    galaxias=galaxies_rust,
                    z_min=z_min,
                    z_max=z_max,
                    umbral_densidad=0.7,
                    grid_ra=25,
                    grid_dec=25,
                    grid_z=15
                )

                ra_structures = rust_result.get('ra', [])
                dec_structures = rust_result.get('dec', [])
                z_structures = rust_result.get('z', [])

                if len(ra_structures) == 0:
                    print(f"  ❌ Rust didn't find structures")
                    continue

                print(f"   ✅ Rust: {len(ra_structures)} structures identified")

                # Calculate distances between structures
                structures = list(zip(ra_structures, dec_structures, z_structures))
                distances = self.calculate_structure_distances(structures)

                if len(distances) > 0:
                    # Analyze scales
                    analysis = self.analyze_rust_scales(distances, z_epoch.mean())
                    analysis['n_structures'] = len(ra_structures)
                    analysis['n_galaxies'] = len(ra_epoch)
                    results[name] = analysis

                    print(f"  📊 Scale analysis:")
                    print(f"    820 Mpc: {analysis.get('820', {}).get('sigma', 0):.1f}σ")
                    print(f"    1640 Mpc: {analysis.get('1640', {}).get('sigma', 0):.1f}σ")

            except Exception as e:
                print(f"   ❌ Rust error: {e}")
                import traceback
                traceback.print_exc()

        # 4. SHOW EVOLUTIONARY TREND
        print("\n" + "="*80)
        print("📈 EVOLUTIONARY TREND (SDSS + DESI)")
        print("="*80)

        for epoch in ['High_z', 'Medium_z', 'Low_z']:
            if epoch in results:
                res = results[epoch]
                z_val = res.get('z_mean', 0)

                print(f"\n{epoch:8} (z={z_val:.2f}):")
                print(f"  • Galaxies: {res.get('n_galaxies', 0):,}")
                print(f"  • Structures: {res.get('n_structures', 0)}")

                sig_820 = res.get('820', {}).get('sigma', 0)
                ratio_820 = res.get('820', {}).get('ratio', 0)
                count_820 = res.get('820', {}).get('count', 0)

                sig_1640 = res.get('1640', {}).get('sigma', 0)
                ratio_1640 = res.get('1640', {}).get('ratio', 0)
                count_1640 = res.get('1640', {}).get('count', 0)

                if sig_820 != 0:
                    print(f"  • 820 Mpc:  {sig_820:6.1f}σ (ratio={ratio_820:.2f}x, count={count_820})")

                if sig_1640 != 0:
                    print(f"  • 1640 Mpc: {sig_1640:6.1f}σ (ratio={ratio_1640:.2f}x, count={count_1640})")

        # 5. CONCLUSION
        elapsed = time.time() - start_time

        print(f"\n" + "="*80)
        print(f"✅ ANALYSIS COMPLETED IN {elapsed:.1f}s")
        print("="*80)

        print("\n🎯 SUMMARY:")
        print("-"*40)

        # Determine cosmic crystal state
        crystal_high_z = results.get('High_z', {}).get('1640', {}).get('sigma', 0)
        crystal_medium_z = results.get('Medium_z', {}).get('1640', {}).get('sigma', 0)
        crystal_low_z = results.get('Low_z', {}).get('1640', {}).get('sigma', 0)

        if crystal_high_z > 3:
            print("✓ COMPLETE CRYSTAL in High_z (z~1.0)")
        if crystal_medium_z > 2:
            print("✓ MELTING CRYSTAL in Medium_z (z~0.6)")
        if crystal_low_z < 2:
            print("✓ MELTED CRYSTAL in Low_z (z~0.25)")

        print(f"\n🔭 Datasets used: {', '.join(combined_data.get('datasets', []))}")

        # Save results
        self.save_rust_results(results, elapsed, combined_data['datasets'])

    def calculate_structure_distances(self, structures, max_pairs=30000):
        """Calculate distances between structures identified by Rust"""

        if len(structures) < 2:
            return np.array([])

        n = min(200, len(structures))
        distances = []

        # Pre-calculate coordinates
        coords = []
        for ra, dec, z in structures[:n]:
            # Use comoving distance
            d = (self.c / self.H0) * np.log(1 + z)  # Approximation for small-medium z

            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)

            # Cartesian coordinates
            x = d * np.cos(dec_rad) * np.cos(ra_rad)
            y = d * np.cos(dec_rad) * np.sin(ra_rad)
            z_coord = d * np.sin(dec_rad)

            coords.append((x, y, z_coord))

        # Calculate pairwise distances
        for i in range(len(coords)):
            xi, yi, zi = coords[i]
            for j in range(i + 1, len(coords)):
                xj, yj, zj = coords[j]

                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)

                if 100 < dist < 5000:  # Filter reasonable distances
                    distances.append(dist)

                if len(distances) >= max_pairs:
                    break

            if len(distances) >= max_pairs:
                break

        return np.array(distances)

    def analyze_rust_scales(self, distances, z_mean):
        """Analyze characteristic scales for Rust results"""

        # Scales of interest in Mpc
        scales = {
            '820': 820.0,
            '1640': 1640.0,
            '2460': 2460.0
        }

        results = {'z_mean': float(z_mean)}

        if len(distances) == 0:
            return results

        for name, scale in scales.items():
            # Verify if scale is within data range
            if scale > np.max(distances) * 1.1:
                continue

            # 5% window around the scale
            window = scale * 0.05
            mask = np.abs(distances - scale) < window

            count = np.sum(mask)
            total = len(distances)

            # Statistics: uniform probability
            range_val = np.max(distances) - np.min(distances)
            if range_val > 0:
                prob = (2 * window) / range_val
                expected = total * prob
            else:
                expected = 0

            if expected > 0:
                ratio = count / expected if expected > 0 else 0
                sigma = (count - expected) / np.sqrt(expected) if expected > 0 else 0
            else:
                ratio = 0
                sigma = 0

            results[name] = {
                'count': int(count),
                'expected': float(expected),
                'ratio': float(ratio),
                'sigma': float(sigma),
                'window_Mpc': float(window)
            }

        return results

    def save_rust_results(self, results, time_elapsed, datasets_used):
        """Save Rust analysis results"""

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        try:
            json_filename = f"resultados_cristal_{timestamp}.json"

            # Convert numpy types to Python natives for JSON
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj

            serializable_results = convert_to_serializable(results)
            serializable_results['metadata'] = {
                'execution_time_s': float(time_elapsed),
                'datasets': datasets_used,
                'H0': float(self.H0),
                'date': timestamp
            }

            with open(json_filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"💾 JSON saved: {json_filename}")
        except Exception as e:
            print(f"⚠️  Error saving JSON: {e}")

        # Save as readable text
        txt_filename = f"resultados_cristal_{timestamp}.txt"
        self.save_text_results(results, time_elapsed, datasets_used, txt_filename)

        return txt_filename

    def save_text_results(self, results, time_elapsed, datasets_used, filename):
        """Simple text version"""

        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CRISTAL_CORE v8.10_RUST - FINAL RESULTS\n")
            f.write("="*80 + "\n\n")

            f.write("📊 METADATA:\n")
            f.write("-"*40 + "\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Execution time: {time_elapsed:.1f} seconds\n")
            f.write(f"Datasets: {', '.join(datasets_used)}\n")
            f.write(f"H0: {self.H0} km/s/Mpc\n\n")

            f.write("📈 RESULTS BY EPOCH:\n")
            f.write("="*60 + "\n\n")

            for epoch in ['High_z', 'Medium_z', 'Low_z']:
                if epoch in results:
                    data = results[epoch]

                    f.write(f"🔬 {epoch}:\n")
                    f.write(f"   mean z: {data.get('z_mean', 0):.3f}\n")
                    f.write(f"   Galaxies: {data.get('n_galaxies', 0):,}\n")
                    f.write(f"   Structures: {data.get('n_structures', 0)}\n\n")

                    if '820' in data:
                        sig_820 = data['820'].get('sigma', 0)
                        ratio_820 = data['820'].get('ratio', 0)
                        count_820 = data['820'].get('count', 0)
                        f.write(f"   820 Mpc:  {sig_820:6.1f}σ (ratio={ratio_820:.2f}x, count={count_820})\n")

                    if '1640' in data:
                        sig_1640 = data['1640'].get('sigma', 0)
                        ratio_1640 = data['1640'].get('ratio', 0)
                        count_1640 = data['1640'].get('count', 0)
                        f.write(f"   1640 Mpc: {sig_1640:6.1f}σ (ratio={ratio_1640:.2f}x, count={count_1640})\n")

                    f.write("\n" + "-"*40 + "\n\n")

            f.write("\n🎯 CONCLUSION:\n")
            f.write("="*60 + "\n\n")

            # Determine crystal state
            if 'High_z' in results:
                sig_1640_high = results['High_z'].get('1640', {}).get('sigma', 0)
                if sig_1640_high > 3:
                    f.write("✅ COMPLETE CRYSTAL in High_z (z~1.0)\n")
                elif sig_1640_high > 2:
                    f.write("⚠️  PARTIAL CRYSTAL in High_z (z~1.0)\n")
                else:
                    f.write("❌ NO CRYSTAL in High_z (z~1.0)\n")

            if 'Medium_z' in results:
                sig_1640_medium = results['Medium_z'].get('1640', {}).get('sigma', 0)
                if sig_1640_medium > 2:
                    f.write("✅ MELTING CRYSTAL in Medium_z (z~0.6)\n")
                else:
                    f.write("⚠️  ALMOST MELTED CRYSTAL in Medium_z (z~0.6)\n")

            if 'Low_z' in results:
                sig_1640_low = results['Low_z'].get('1640', {}).get('sigma', 0)
                if sig_1640_low > 1:
                    f.write("✅ RESIDUES of crystal in Low_z (z~0.25)\n")
                else:
                    f.write("❌ COMPLETELY MELTED CRYSTAL in Low_z (z~0.25)\n")

            f.write("\n🔭 Implication: Evolution of large-scale structure.\n")
            f.write("   The 'cosmic crystal' melts with cosmological time.\n")

        print(f"💾 Results saved in: {filename}")


def main():
    """Main function"""
    print("🚀 STARTING CRISTAL_CORE v8.10_RUST")
    print("="*60)

    analyzer = EvolutionaryRustAnalyzer(H0=70.0)

    # Configure analysis
    use_sdss = True   # Use SDSS dataset?
    use_desi = True   # Use DESI dataset?
    max_galaxies = 250000  # Maximum galaxies for analysis

    analyzer.run_rust_analysis(
        use_sdss=use_sdss,
        use_desi=use_desi,
        max_galaxies=max_galaxies
    )

    print("\n" + "="*60)
    print("🎉 ANALYSIS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()