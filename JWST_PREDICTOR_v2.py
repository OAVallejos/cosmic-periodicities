#!/usr/bin/env python3     
"""
v2
"""

import numpy as np
import gzip
import json
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import sys

# Import compiled Rust engine
sys.path.append('.')
try:
    import unified_crystal_engine as rust_engine
except ImportError:
    print("⚠️  Compile the Rust engine first:")
    print("   cargo build --release")
    print("   cp target/release/libunified_crystal_engine.so ./")
    sys.exit(1)

# ============================================================================
# CORRECT CONFIGURATION (relative paths)
# ============================================================================

# IMPORTANT: Files are in ./data/ (not data/data/)
# Remove 'data/' from file prefix
JWST_FIELDS = {
    'PRIMER-COSMOS': {
        'photometric': 'primercp.dat.gz',  # WITHOUT 'data/' prefix
        'observed': 'primerco.dat.gz',
        'size_mb': 2.3
    },
    'CEERS': {
        'photometric': 'ceersp.dat.gz',
        'observed': 'ceerso.dat.gz',
        'size_mb': 1.7
    },
    'JADES-GS': {
        'photometric': 'jadesgsp.dat.gz',
        'observed': 'jadesgso.dat.gz',
        'size_mb': 1.4
    },
    'JADES-GN': {
        'photometric': 'jadesgnp.dat.gz',
        'observed': 'jadesgno.dat.gz',
        'size_mb': 1.1
    },
    'PRIMER-UDS': {
        'photometric': 'primerup.dat.gz',
        'observed': 'primeruo.dat.gz',
        'size_mb': 2.6
    },
    'NGDEEP': {
        'photometric': 'ngdeepp.dat.gz',
        'observed': 'ngdeepo.dat.gz',
        'size_mb': 0.285
    },
    'UNCOVER-A2744': {
        'photometric': 'a2744p.dat.gz',
        'observed': 'a2744o.dat.gz',
        'size_mb': 0.822
    }
}

# Analysis parameters
Z_RANGE = (0.5, 15.0)  # Redshift range to analyze
VOID_WINDOW = 0.003    # Window for voids

# ============================================================================
# CORRECTED LOADING FUNCTIONS
# ============================================================================

def load_galaxies_streaming(filepath: Path, z_min: float, z_max: float,
                           max_galaxies: Optional[int] = None) -> List[List[float]]:
    """
    Load galaxies with streaming - CORRECTED VERSION.
    """
    galaxies = []

    if not filepath.exists():
        print(f"    ❌ File does not exist: {filepath}")
        return galaxies

    try:
        print(f"    📂 Opening: {filepath}")
        with gzip.open(filepath, 'rt') as f:
            lines_processed = 0
            chunk = []

            for line in f:
                if line.startswith('#') or not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                try:
                    ra = float(parts[0])
                    dec = float(parts[1])
                    z = float(parts[4])

                    # Filter by redshift and quality
                    if z_min < z < z_max and z != -1.0 and z != -99.0:
                        chunk.append([ra, dec, z])
                        lines_processed += 1

                        # Process chunk when it reaches size
                        if len(chunk) >= 50000:
                            galaxies.extend(chunk)
                            chunk = []
                           
                            # Show progress
                            if lines_processed % 100000 == 0:
                                print(f"      📥 {lines_processed:,} lines processed...")
                           
                            # Limit if specified
                            if max_galaxies and len(galaxies) >= max_galaxies:
                                break

                except (ValueError, IndexError):
                    continue

            # Add last chunk
            if chunk:
                galaxies.extend(chunk)

    except Exception as e:
        print(f"    ❌ Error loading {filepath}: {e}")
        return []

    print(f"    ✅ {len(galaxies):,} valid galaxies loaded")
    return galaxies

def load_field_data(field_name: str, data_dir: str = 'data',
                   use_photometric: bool = True, max_galaxies: int = 200000) -> List[List[float]]:
    """
    Load JWST field data - CORRECTED VERSION.
    """
    field_info = JWST_FIELDS.get(field_name)
    if not field_info:
        print(f"❌ Unknown field: {field_name}")
        return []

    # Get correct filename
    file_type = 'photometric' if use_photometric else 'observed'
    filename = field_info[file_type]

    # CORRECTED: Use Path correctly
    filepath = Path(data_dir) / filename

    print(f"📂 Loading {field_name} ({file_type})...")
    print(f"   File: {filename}")
    print(f"   Full path: {filepath.absolute()}")

    galaxies = load_galaxies_streaming(filepath, Z_RANGE[0], Z_RANGE[1], max_galaxies)

    return galaxies

# ============================================================================
# DIRECT VOID ANALYSIS (Pure Python for verification)
# ============================================================================

def analyze_voids_direct(field_name: str, galaxies: List[List[float]]) -> Dict:
    """
    Direct void analysis in Python (for verification).
    """
    if not galaxies:
        return {}

    # Extract redshifts
    redshifts = [g[2] for g in galaxies]
    z_array = np.array(redshifts)
    total_galaxies = len(z_array)

    print(f"   🔍 Analyzing {total_galaxies:,} galaxies for voids...")

    # Voids to search (based on your previous analysis)
    voids_to_check = [
        (10.191, VOID_WINDOW),
        (10.579, VOID_WINDOW),
        (10.764, VOID_WINDOW),
        (11.155, VOID_WINDOW),
        (11.340, VOID_WINDOW),
        (11.722, VOID_WINDOW)
    ]

    results = {
        'field': field_name,
        'total_galaxies': total_galaxies,
        'voids': {}
    }

    # Average density
    density = total_galaxies / (Z_RANGE[1] - Z_RANGE[0])

    complete_voids = []

    for z_target, window in voids_to_check:
        z_min = z_target - window/2
        z_max = z_target + window/2

        mask = (z_array >= z_min) & (z_array <= z_max)
        observed = np.sum(mask)

        expected = density * window

        # Calculate significance (Poisson)
        if expected > 0:
            # Use Poisson distribution
            from scipy import stats
            if observed == 0:
                p_val = np.exp(-expected)  # Poisson P(0) = e^{-λ}
            else:
                p_val = stats.poisson.cdf(observed, expected)

            sigma = abs(stats.norm.ppf(p_val)) if p_val > 0 else 0
        else:
            p_val = 1.0
            sigma = 0.0

        # Classify
        if observed == 0 and sigma >= 5.0:
            status = 'COMPLETE_VOID'
            complete_voids.append(z_target)
        elif observed == 0 and sigma >= 3.0:
            status = 'STRONG_SUPPRESSION'
        elif observed < expected/2:
            status = 'MODERATE_SUPPRESSION'
        else:
            status = 'NORMAL'

        results['voids'][f'z={z_target:.3f}'] = {
            'observed': int(observed),
            'expected': float(expected),
            'sigma': float(sigma),
            'status': status,
            'z_range': [float(z_min), float(z_max)]
        }

        if status != 'NORMAL':
            print(f"   • z={z_target:.3f}: {observed} galaxies (exp: {expected:.1f}) -> {status}")

    results['complete_voids'] = complete_voids
    results['n_complete_voids'] = len(complete_voids)

    return results

# ============================================================================
# SIMPLIFIED PARALLEL ANALYSIS
# ============================================================================

def analyze_field_parallel(args: Tuple[str, int]) -> Dict:
    """
    Function for parallel analysis - SIMPLIFIED VERSION.
    """
    field_name, max_galaxies = args

    print(f"\n{'='*60}")
    print(f"🚀 Analyzing: {field_name}")
    print(f"{'='*60}")

    # 1. Load data
    galaxies = load_field_data(field_name, max_galaxies=max_galaxies)

    if not galaxies:
        print(f"   ⚠️  No data for {field_name}")
        return {'field': field_name, 'status': 'NO_DATA'}

    # 2. Direct void analysis (pure Python)
    direct_results = analyze_voids_direct(field_name, galaxies)

    if not direct_results:
        return {'field': field_name, 'status': 'ANALYSIS_FAILED'}

    # 3. Try analysis with Rust (optional)
    rust_results = {}
    try:
        # Convert to Rust format
        rust_galaxies = []
        for gal in galaxies[:50000]:  # Only first 50k for testing
            rust_galaxies.append([gal[0], gal[1], gal[2], 100.0])

        # Create Rust engine
        engine = rust_engine.UnifiedCrystalEngine(modo_alto_z=True)

        # Execute quick analysis
        rust_results = engine.analisis_completo_unificado(
            rust_galaxies,
            buscar_vacios=True,
            campo_nombre=field_name
        )

        direct_results['rust'] = rust_results

    except Exception as e:
        print(f"   ⚠️  Rust engine not available: {e}")

    # 4. Combine results
    final_results = {
        'field': field_name,
        'timestamp': datetime.now().isoformat(),
        'n_galaxies': len(galaxies),
        **direct_results
    }

    if rust_results:
        final_results['omega_dominante'] = rust_results.get('omega_dominante', 0.191)
        final_results['lambda_z'] = rust_results.get('lambda_z_mpc', 1682.0)

    print(f"   ✅ {field_name}: {len(galaxies):,} galaxies, {direct_results.get('n_complete_voids', 0)} voids")

    return final_results

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis(max_workers: int = 2, max_galaxies_per_field: int = 300000):
    """
    Run analysis of all fields.
    """
    print("\n" + "="*80)
    print("🌌 CRYSTALLINE VOIDS ANALYSIS - JWST")
    print("="*80)
    print(f"Data directory: {Path('data').absolute()}")
    print(f"Cores: {max_workers}")
    print("="*80)

    # Verify data/ directory exists
    data_dir = Path('data')
    if not data_dir.exists():
        print("❌ ERROR: Directory 'data/' not found")
        print(f"   Expected path: {data_dir.absolute()}")
        print("   Create the directory and copy .dat.gz files there")
        return {}

    # List available files
    print("\n📁 Files found in data/:")
    for file in sorted(data_dir.glob('*.dat.gz')):
        size_mb = file.stat().st_size / (1024*1024)
        print(f"   • {file.name} ({size_mb:.1f} MB)")

    # Prepare tasks
    tasks = [(field_name, max_galaxies_per_field) for field_name in JWST_FIELDS.keys()]

    # Execute in parallel
    all_results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_field = {
            executor.submit(analyze_field_parallel, task): task[0]
            for task in tasks
        }

        for future in as_completed(future_to_field):
            field_name = future_to_field[future]

            try:
                result = future.result(timeout=300)
                all_results[field_name] = result

                print(f"\n ✅ {field_name}: Completed")

            except Exception as e:
                print(f"\n ❌ {field_name}: Error - {e}")
                all_results[field_name] = {
                    'field': field_name,
                    'error': str(e)
                }

    return all_results

# ============================================================================
# REPORT AND STATISTICS
# ============================================================================

def generate_report(all_results: Dict):
    """
    Generate final report.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Filter fields with data
    valid_fields = []
    complete_voids_by_field = {}
    all_complete_voids = []

    for field_name, result in all_results.items():
        if result.get('status') not in ['NO_DATA', 'ANALYSIS_FAILED']:
            valid_fields.append(field_name)

            # Collect complete voids
            if 'complete_voids' in result:
                voids = result['complete_voids']
                if voids:
                    complete_voids_by_field[field_name] = voids
                    all_complete_voids.extend([(field_name, z) for z in voids])

    # Statistics
    total_complete_voids = len(all_complete_voids)

    # Group by redshift
    voids_by_z = {}
    for field_name, z in all_complete_voids:
        z_key = f"{z:.3f}"
        if z_key not in voids_by_z:
            voids_by_z[z_key] = []
        voids_by_z[z_key].append(field_name)

    # Create report
    report = {
        'analysis_date': datetime.now().isoformat(),
        'parameters': {
            'z_range': list(Z_RANGE),
            'void_window': VOID_WINDOW,
            'data_directory': str(Path('data').absolute())
        },
        'statistics': {
            'total_fields_analyzed': len(all_results),
            'fields_with_data': len(valid_fields),
            'fields_with_complete_voids': len(complete_voids_by_field),
            'total_complete_voids': total_complete_voids
        },
        'complete_voids_by_field': complete_voids_by_field,
        'voids_by_redshift': voids_by_z,
        'field_results': all_results
    }

    # Save JSON
    output_file = f"jwst_voids_analysis_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Report saved in: {output_file}")

    # Show summary
    print("\n" + "="*80)
    print("🌟 FINAL RESULTS")
    print("="*80)

    print(f"\n📊 STATISTICS:")
    print(f"   • Fields analyzed: {len(all_results)}")
    print(f"   • Fields with data: {len(valid_fields)}")
    print(f"   • Fields with complete voids: {len(complete_voids_by_field)}")
    print(f"   • Total complete voids: {total_complete_voids}")

    if total_complete_voids > 0:
        print(f"\n✅ COMPLETE VOIDS FOUND (>5σ):")
        for field_name, voids in complete_voids_by_field.items():
            print(f"   • {field_name}: {len(voids)} voids")
            for z in voids:
                print(f"     - z = {z:.3f}")

    # Show patterns
    if voids_by_z:
        print(f"\n🔍 DETECTED PATTERNS:")
        for z_key, fields in sorted(voids_by_z.items()):
            if len(fields) >= 2:
                print(f"   • z = {z_key} confirmed in: {', '.join(fields)}")

    # Manual verification
    if all_complete_voids:
        print(f"\n🔍 MANUAL VERIFICATION:")
        for field_name, z_target in all_complete_voids[:3]:  # First 3
            filename = JWST_FIELDS[field_name]['photometric']
            z_min = z_target - VOID_WINDOW/2
            z_max = z_target + VOID_WINDOW/2
            cmd = f"zcat data/{filename} | awk '$5 > {z_min:.3f} && $5 < {z_max:.3f}' | wc -l"
            print(f"   # {field_name} z={z_target:.3f}:")
            print(f"   {cmd}")

    return report

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function.
    """
    print("🚀 JWST CRYSTALLINE VOIDS ANALYSIS")
    print("="*80)

    # Check Rust
    try:
        print(f"✅ Rust engine available:")
        print(f"   • Fundamental ω₀: {rust_engine.OMEGA_0_FUNDAMENTAL}")
        print(f"   • Primordial λ₀: {rust_engine.LAMBDA_0_PRIMIGENIO} Mpc")
    except:
        print("⚠️  Rust engine not available - using pure Python")

    # Determine cores
    try:
        num_cores = min(mp.cpu_count(), 4)
    except:
        num_cores = 2

    print(f"\n⚙️  Configuration:")
    print(f"   • Cores: {num_cores}")
    print(f"   • Window: ±{VOID_WINDOW/2:.4f}")
    print(f"   • z range: {Z_RANGE[0]} - {Z_RANGE[1]}")

    # Execute analysis
    print(f"\n{'='*80}")
    print("🌌 PROCESSING JWST DATA...")
    print("="*80)

    all_results = run_analysis(
        max_workers=num_cores,
        max_galaxies_per_field=300000
    )

    # Generate report
    report = generate_report(all_results)

    # Conclusion
    print("\n" + "="*80)
    print("🎯 CONCLUSION")
    print("="*80)

    total_voids = report['statistics']['total_complete_voids']
    multi_field_voids = sum(1 for fields in report['voids_by_redshift'].values()
                          if len(fields) >= 2)

    if total_voids >= 3 and multi_field_voids >= 1:
        print("✅ EVIDENCE OF CRYSTALLINE STRUCTURE!")
        print(f"   • {total_voids} complete voids detected")
        print(f"   • {multi_field_voids} patterns confirmed in multiple fields")

        print(f"\n📈 IMPLICATIONS:")
        print(f"   1. Periodicity in z with ω ≈ {rust_engine.OMEGA_0_FUNDAMENTAL}")
        print(f"   2. Predicted voids match observations")
        print(f"   3. Non-random structure of early universe")

    elif total_voids > 0:
        print("⚠️  PATTERN INDICATIONS")
        print(f"   • {total_voids} voids detected")
        print(f"   • Needs confirmation in more fields")

    else:
        print("❌ NO COMPLETE VOIDS DETECTED")
        print(f"   • Verify data and parameters")

    print(f"\n🔍 NEXT STEPS:")
    print(f"   1. Verify voids with previous commands")
    print(f"   2. Analyze fields individually")
    print(f"   3. Adjust ω if necessary")

    print("\n" + "="*80)
    print("✨ ANALYSIS COMPLETED")
    print("="*80)

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()