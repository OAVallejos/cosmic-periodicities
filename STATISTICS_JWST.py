#!/usr/bin/env python3
"""                        
Rigorous statistical significance analysis        
"""

import numpy as np
from scipy import stats
import gzip

def calcular_significancia_completa(filename):
    """Calculate rigorous statistical significance"""
    print(f"\n📊 RIGOROUS STATISTICAL ANALYSIS: {filename}")
    print("="*60)

    # 1. Load data
    redshifts = []
    with gzip.open(f"data/{filename}", 'rt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    z = float(parts[4])
                    if 0.5 < z < 15.0 and z != -1.0 and z != -99.0:
                        redshifts.append(z)
                except:
                    continue

    redshifts = np.array(redshifts)
    total = len(redshifts)
    z_min, z_max = 0.5, 15.0
    densidad = total / (z_max - z_min)

    print(f"• Total galaxies: {total:,}")
    print(f"• z range: [{z_min}, {z_max}]")
    print(f"• Density: {densidad:.1f} galaxies/unit_z")

    # 2. Voids to test (with their measured widths)
    voids_to_test = [
        (10.191, 0.0088),  # ±0.0088 (measured)
        (10.579, 0.0040),
        (10.764, 0.0110),
        (11.155, 0.0048),
        (11.340, 0.0098),
        (11.722, 0.0028)
    ]

    # 3. Calculate significance for EACH void
    significancias = []
    p_values = []

    for z_center, half_width in voids_to_test:
        window = 2 * half_width
        z_low = z_center - half_width
        z_high = z_center + half_width

        # Count galaxies in window
        mask = (redshifts >= z_low) & (redshifts <= z_high)
        observed = np.sum(mask)

        # Expected according to Poisson
        expected = densidad * window

        # Exact significance
        if observed == 0:
            # Exact Poisson probability P(0) = e^{-λ}
            p_val = np.exp(-expected)
            sigma = abs(stats.norm.ppf(p_val))
        else:
            # Use Poisson CDF
            p_val = stats.poisson.cdf(observed, expected)
            sigma = abs(stats.norm.ppf(p_val))

        significancias.append(sigma)
        p_values.append(p_val)

        print(f"\n🎯 z = {z_center:.3f} ± {half_width:.4f}:")
        print(f"   • Observed: {observed}")
        print(f"   • Expected: {expected:.2f}")
        print(f"   • p-value: {p_val:.2e}")
        print(f"   • Significance: {sigma:.2f}σ")
        print(f"   • Random probability of seeing 0: {np.exp(-expected):.2e}")

    # 4. COMBINED probability (all 6 voids)
    print(f"\n" + "="*60)
    print("🎲 COMBINED PROBABILITY (6 voids)")
    print("="*60)

    # Method 1: Product of p-values (assuming independence)
    combined_p = np.prod(p_values)
    combined_sigma = abs(stats.norm.ppf(combined_p))

    print(f"• Product of p-values: {combined_p:.2e}")
    print(f"• Combined significance: {combined_sigma:.2f}σ")

    # Method 2: Fisher's test
    chi2_fisher = -2 * np.sum(np.log(p_values))
    p_fisher = stats.chi2.sf(chi2_fisher, df=2*len(p_values))
    sigma_fisher = abs(stats.norm.ppf(p_fisher))

    print(f"• Fisher's test: χ² = {chi2_fisher:.1f}, p = {p_fisher:.2e}")
    print(f"• Fisher significance: {sigma_fisher:.2f}σ")

    # 5. Compare with null distribution
    print(f"\n" + "="*60)
    print("📈 COMPARISON WITH NULL DISTRIBUTION")
    print("="*60)

    # Simulate random positions
    n_simulations = 10000
    max_sigmas = []

    for sim in range(n_simulations):
        # Pick 6 random positions
        random_zs = np.random.uniform(1.0, 12.0, size=6)

        sigmas_sim = []
        for z_rand in random_zs:
            # Use average window from our voids
            window_avg = np.mean([2*w for _, w in voids_to_test])
            expected_rand = densidad * window_avg

            # Simulate Poisson count
            observed_rand = np.random.poisson(expected_rand)

            # Calculate significance
            if observed_rand == 0:
                p_rand = np.exp(-expected_rand)
                sigma_rand = abs(stats.norm.ppf(p_rand))
            else:
                p_rand = stats.poisson.cdf(observed_rand, expected_rand)
                sigma_rand = abs(stats.norm.ppf(p_rand))

            sigmas_sim.append(sigma_rand)

        # Save maximum significance of this simulation
        max_sigmas.append(max(sigmas_sim))

    # Our maximum observed significance
    our_max_sigma = max(significancias)

    # How extreme is our result?
    p_extreme = np.mean(np.array(max_sigmas) >= our_max_sigma)

    print(f"• Maximum observed σ: {our_max_sigma:.2f}σ")
    print(f"• In {n_simulations:,} random simulations:")
    print(f"  - Average maximum σ: {np.mean(max_sigmas):.2f}σ")
    print(f"  - Maximum σ 99.9th percentile: {np.percentile(max_sigmas, 99.9):.2f}σ")
    print(f"• Probability of seeing σ ≥ {our_max_sigma:.2f}σ by chance: {p_extreme:.2e}")

    return {
        'filename': filename,
        'total_galaxies': total,
        'voids': voids_to_test,
        'significances': significancias,
        'p_values': p_values,
        'combined_sigma': combined_sigma,
        'fisher_sigma': sigma_fisher,
        'p_extreme': p_extreme
    }

# Execute for the 3 main fields
print("="*80)
print("🧪 RIGOROUS STATISTICAL TESTS - JWST VOIDS")
print("="*80)

resultados = []
for filename in ['primerup.dat.gz', 'primercp.dat.gz', 'ceersp.dat.gz']:
    try:
        res = calcular_significancia_completa(filename)
        resultados.append(res)
    except Exception as e:
        print(f"❌ Error with {filename}: {e}")

# Final analysis
print("\n" + "="*80)
print("🎯 STATISTICAL CONCLUSION")
print("="*80)

if resultados:
    # Average significance between fields
    avg_sigma = np.mean([max(r['significances']) for r in resultados])
    min_p = min([r['p_extreme'] for r in resultados])

    print(f"📊 COMBINED RESULTS (3 fields):")
    print(f"• Average maximum significance: {avg_sigma:.2f}σ")
    print(f"• Most extreme probability: {min_p:.2e}")

    # Interpretation
    if avg_sigma > 5.0 and min_p < 1e-7:
        print(f"\n✅ CONCLUSION: STRONG STATISTICAL EVIDENCE")
        print(f"   • Significance >5σ in multiple fields")
        print(f"   • Probability of chance: <1 in {int(1/min_p):,}")

    elif avg_sigma > 3.0:
        print(f"\n⚠️  CONCLUSION: MODERATE EVIDENCE")
        print(f"   • Significance 3-5σ")
        print(f"   • Needs additional verification")

    else:
        print(f"\n❌ CONCLUSION: INSUFFICIENT EVIDENCE")
        print(f"   • Significance <3σ")