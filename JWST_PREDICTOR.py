#!/usr/bin/env python3
"""                        
 - Complete analysis WITHOUT artificial limits
"""                                                   import numpy as np
from pathlib import Path
import gzip
import json
from scipy import stats
from datetime import datetime

# ============================================================================
# REAL DATA (based on what you've already seen)
# ============================================================================
FIELD_STATS = {
    'PRIMER-COSMOS': {'file': 'primercp.dat.gz', 'total': 108911},
    'CEERS': {'file': 'ceersp.dat.gz', 'total': 73856},
    'JADES-GS': {'file': 'jadesgsp.dat.gz', 'total': 0},  # Had an error
    'JADES-GN': {'file': 'jadesgnp.dat.gz', 'total': 0},  # To verify
    'PRIMER-UDS': {'file': 'primerup.dat.gz', 'total': 0}, # To verify
    'NGDEEP': {'file': 'ngdeepp.dat.gz', 'total': 12691},
    'UNCOVER-A2744': {'file': 'a2744p.dat.gz', 'total': 33785}
}

# Analysis windows based on previous results
ANALYSIS_WINDOWS = {
    'PRIMER-COSMOS': {
        'voids': [(10.191, 0.003), (10.579, 0.003)],
        'nodes': [(10.000, 0.020), (10.382, 0.020)]
    },
    'CEERS': {
        'voids': [(10.579, 0.003)],
        'nodes': [(10.388, 0.020), (10.770, 0.020)]
    }
}

W0 = 0.191

# ============================================================================
# PRECISE ANALYSIS FUNCTION
# ============================================================================

def analyze_field_precise(field_name, data_dir='data'):
    """Precise analysis WITHOUT artificial limits"""
    if field_name not in FIELD_STATS:
        return None

    config = FIELD_STATS[field_name]
    filepath = Path(data_dir) / config['file']

    if not filepath.exists():
        print(f"❌ {field_name}: File not found")
        return None

    # Determine what to analyze
    if field_name in ANALYSIS_WINDOWS:
        voids_to_check = ANALYSIS_WINDOWS[field_name]['voids']
        nodes_to_check = ANALYSIS_WINDOWS[field_name]['nodes']
    else:
        # Standard predictions
        voids_to_check = [(10.191, 0.003), (10.579, 0.003), (10.764, 0.003)]
        nodes_to_check = []

    print(f"\n🔍 {field_name}:")
    print(f"   File: {config['file']}")

    # Counters
    total_galaxies = 0
    redshifts = []

    # 1. COUNT REAL TOTAL
    try:
        with gzip.open(filepath, 'rt') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                try:
                    z = float(parts[4])
                    if 0.5 < z < 15.0 and z != -1.0 and z != -99.0:
                        redshifts.append(z)
                        total_galaxies += 1
                except:
                    continue
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return None

    if total_galaxies == 0:
        print(f"   ⚠️  No valid galaxies")
        return None

    print(f"   Total galaxies: {total_galaxies:,}")

    # Convert to numpy array
    z_array = np.array(redshifts)

    # 2. VOID ANALYSIS
    results = {'field': field_name, 'total': total_galaxies, 'voids': {}}

    for z_target, window in voids_to_check:
        mask = np.abs(z_array - z_target) < window/2
        observed = np.sum(mask)

        # REAL field density
        density = total_galaxies / 14.5  # z=0.5-15.0
        expected = density * window

        if expected > 0:
            p_val = stats.poisson.cdf(observed, expected)
            sigma = abs(stats.norm.ppf(p_val)) if p_val > 0 else 0
            log10_p = -np.log10(p_val) if p_val > 0 else 0
        else:
            p_val = 1.0
            sigma = 0.0
            log10_p = 0.0

        # CORRECT classification (based on previous analysis)
        if observed == 0 and sigma >= 5.0:
            status = 'COMPLETE_VOID'
            significance = f"{sigma:.1f}σ"
        elif observed == 0 and sigma >= 3.0:
            status = 'STRONG_SUPPRESSION'
            significance = f"{sigma:.1f}σ"
        elif observed < expected/2:
            status = 'MODERATE_SUPPRESSION'
            significance = f"{sigma:.1f}σ"
        else:
            status = 'NOT_SIGNIFICANT'
            significance = f"{sigma:.1f}σ"

        results['voids'][f'z={z_target:.3f}'] = {
            'observed': int(observed),
            'expected': float(expected),
            'p_value': float(p_val),
            'sigma': float(sigma),
            'log10_p': float(log10_p),
            'status': status,
            'significance': significance
        }

        print(f"   z={z_target:.3f}±{window/2:.3f}: {observed} galaxies")
        print(f"     Expected: {expected:.1f} | p={p_val:.2e} | {significance}")
        if status == 'COMPLETE_VOID':
            print(f"     ✅ {status}")

    # 3. NODE ANALYSIS (if applicable)
    if nodes_to_check:
        results['nodes'] = {}
        for z_target, window in nodes_to_check:
            mask = np.abs(z_array - z_target) < window/2
            observed = np.sum(mask)
            results['nodes'][f'z={z_target:.3f}'] = int(observed)
            print(f"   Node z={z_target:.3f}: {observed} galaxies")

    # 4. AVERAGE DENSITY
    results['mean_density'] = float(density)

    # 5. VERIFY PREVIOUS RESULTS
    if field_name == 'PRIMER-COSMOS':
        # Your original results: 0 galaxies at z=10.191, 22.5 expected, 6.39σ
        print(f"\n   📊 COMPARISON WITH PREVIOUS ANALYSIS:")
        print(f"   • Previous: 0/108,911 at z=10.191 (6.39σ)")
        print(f"   • Current: {results['voids'].get('z=10.191', {}).get('observed', 'N/A')}/"
              f"{total_galaxies:,} at z=10.191 ({results['voids'].get('z=10.191', {}).get('sigma', 'N/A'):.2f}σ)")

    return results

# ============================================================================
# ANALYSIS OF THE 2 CONFIRMED FIELDS
# ============================================================================

def main():
    print("="*80)
    print("DEFINITIVE ANALYSIS - JWST VOIDS")
    print("Based on previous results without artificial limits")
    print("="*80)

    # Only analyze fields with confirmed data
    fields_to_analyze = ['PRIMER-COSMOS', 'CEERS']

    all_results = {}
    confirmed_voids = []

    for field in fields_to_analyze:
        result = analyze_field_precise(field, 'data')
        if result:
            all_results[field] = result

            # Verify complete voids
            for void_name, void_data in result['voids'].items():
                if void_data['status'] == 'COMPLETE_VOID':
                    confirmed_voids.append({
                        'field': field,
                        'z': void_name,
                        'sigma': void_data['sigma'],
                        'observed': void_data['observed'],
                        'expected': void_data['expected']
                    })

    # SUMMARY
    print("\n" + "="*80)
    print("DEFINITIVE SUMMARY")
    print("="*80)

    if confirmed_voids:
        print("✅ COMPLETE VOIDS CONFIRMED (>5σ):")
        for void in confirmed_voids:
            print(f"   • {void['field']}: {void['z']}")
            print(f"     {void['observed']} galaxies (expected: {void['expected']:.1f})")
            print(f"     Significance: {void['sigma']:.2f}σ")

        print(f"\n📊 TOTAL: {len(confirmed_voids)} complete voids")

        if len(confirmed_voids) >= 2:
            print("\n🎯 CONCLUSION: PREDICTION VALIDATED")
            print("   The crystalline model correctly predicts zero-density voids")
        else:
            print("\n⚠️  CONCLUSION: PARTIAL EVIDENCE")
            print("   Confirmation needed in more fields")
    else:
        print("❌ No complete voids confirmed")
        print("   Verify analysis parameters")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"jwst_definitive_results_{timestamp}.json"

    final_report = {
        'analysis_date': datetime.now().isoformat(),
        'fields_analyzed': list(all_results.keys()),
        'confirmed_voids': confirmed_voids,
        'results': all_results,
        'parameters': {
            'w0': W0,
            'window_size': 0.003,
            'z_range': [0.5, 15.0]
        }
    }

    with open(output_file, 'w') as f:
        json.dump(final_report, f, indent=2)

    print(f"\n📁 Results saved in: {output_file}")
    print("="*80)

    # MANUAL VERIFICATION INSTRUCTIONS
    print("\n🔍 MANUAL VERIFICATION:")
    print("To confirm PRIMER-COSMOS results:")
    print(f"  zcat data/primercp.dat.gz | awk '$5 > 10.188 && $5 < 10.194' | wc -l")
    print("\nTo confirm CEERS results:")
    print(f"  zcat data/ceersp.dat.gz | awk '$5 > 10.576 && $5 < 10.582' | wc -l")

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()