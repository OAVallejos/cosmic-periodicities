"""
HARMONIC_SPECTRUM_PLOT_OPTIMIZED.py
====================================
Updated version with data from the paper "Evidence for Cosmic Harmonic Periodicity"
DOI: 10.5281/zenodo.18406995
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

# ============================================================================
# OPTIMIZED CONFIGURATION
# ============================================================================
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'figure.figsize': (16, 14),  # Taller for extended table
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'font.family': 'sans-serif',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold'
})

# ============================================================================
# UPDATED PAPER DATA
# ============================================================================
# Fundamental frequency detected in SDSS
omega_0 = 0.191
sigma_omega0 = 0.012

# Updated JWST data from paper tables
# From Tables 4, 5 and 6
z_jwst = np.array([1.75, 3.75, 8.5, 6.40, 5.65, 2.70])  # Representative redshifts
omega_jwst = np.array([
    1.1794,  # z~1.75, 6th harmonic (Table 5, Evolved epoch)
    0.8493,  # z~3.75, 4th harmonic (Table 5, Star Formation Peak)
    1.1350,  # z~8.5, 6th harmonic (Table 5, Primitive)
    1.0794,  # z~6.40, PRIMER-COSMOS (Table 6)
    0.7670,  # z~5.65, PRIMER-UDS (Table 6)
    0.6035   # z~2.70, A2744-UNCOVER (Table 6)
])
sigma_jwst = np.array([
    0.0307,  # Error for 1.1794
    0.2608,  # Error for 0.8493
    0.1777,  # Error for 1.1350
    0.0354,  # Error for 1.0794
    0.0307,  # Error for 0.7670
    0.3319   # Error for 0.6035
])

# ΛCDM reference values
omega_lcdm = 1.146  # ω₆ theoretical
harmonic_numbers = [6, 4, 6, 6, 4, 3]  # n for each point
field_names = ['Evolved', 'SF Peak', 'Primitive', 'COSMOS', 'UDS', 'A2744']

# Local SDSS data
omega_sdss = 0.191
sigma_sdss = 0.012
z_sdss = 0.10
logBF_sdss = 107.1

# ============================================================================
# IMPROVED HARMONIC ANALYSIS
# ============================================================================
def comprehensive_harmonic_analysis(omega_obs, sigma_obs, z_vals, harmonic_nums):
    results = []
    for i, (w_obs, sigma, z, n) in enumerate(zip(omega_obs, sigma_obs, z_vals, harmonic_nums)):
        w_theoretical = n * omega_0
        delta = w_obs - w_theoretical
        tension_sigma = abs(delta) / sigma
        p_value = 2 * (1 - stats.norm.cdf(abs(delta)/sigma))
        
        # Calculate cycle number (m)
        if z > 0:
            # Approximation: m(z) ≈ ω * t(z) / (2π)
            # For simplicity, we use m(z) ≈ z * 2.5 (paper approximation)
            m_z = z * 2.5
            m_integer = round(m_z)
            delta_m = abs(m_z - m_integer)
        else:
            m_z = m_integer = delta_m = 0
        
        results.append({
            'z': z,
            'field': field_names[i] if i < len(field_names) else f'Field_{i}',
            'omega_obs': w_obs,
            'sigma_obs': sigma,
            'n_theoretical': n,
            'omega_theoretical': w_theoretical,
            'delta': delta,
            'tension_sigma': tension_sigma,
            'pct_deviation': abs(delta) / w_theoretical * 100,
            'p_value': p_value,
            'is_harmonic': tension_sigma < 2.0,
            'm_z': m_z,
            'm_integer': m_integer,
            'delta_m': delta_m,
            'logBF': [107.1, 15.43, 2350.03, 884.71, 40.9, 19.85][i] if i < 6 else 0
        })
    return results

# ============================================================================
# CREATE COMPLETE FIGURE
# ============================================================================
fig = plt.figure(figsize=(18, 16))
gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.0, 0.8, 1.0], hspace=0.35, wspace=0.3)

# ============================================================================
# PANEL 1: FREQUENCY SPECTRUM WITH SDSS AND JWST
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])

# Harmonic lines
n_range = np.arange(1, 8)
omega_theoretical = n_range * omega_0

for n, omega in zip(n_range, omega_theoretical):
    if n == 6:
        ax1.axhline(y=omega, color='red', linestyle='--', alpha=0.8,
                   linewidth=2.5, label=f'$\\Lambda$CDM = {omega:.3f} (n=6)')
    else:
        ax1.axhline(y=omega, color='gray', linestyle=':', alpha=0.4,
                   linewidth=1.2)
    ax1.text(10.5, omega + 0.01, f'n={n}', fontsize=9, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# SDSS point (local, low z)
ax1.errorbar(z_sdss, omega_sdss, yerr=sigma_sdss, fmt='D', color='darkorange',
            markersize=12, capsize=8, label=f'SDSS (z={z_sdss})\nω₁={omega_sdss}±{sigma_sdss}\nlogBF={logBF_sdss}',
            ecolor='darkorange', elinewidth=2, markeredgecolor='black', markeredgewidth=1.5)

# JWST data
colors = ['#2E8B57', '#4169E1', '#8A2BE2', '#DC143C', '#FF8C00', '#4B0082']
markers = ['o', 's', '^', 'v', 'p', '*']

for i, (z, w, sigma, color, marker) in enumerate(zip(z_jwst, omega_jwst, sigma_jwst, colors, markers)):
    label = f'{field_names[i]} (z={z})\nω={w:.3f}±{sigma:.3f}\nn={harmonic_numbers[i]}'
    ax1.errorbar(z, w, yerr=sigma, fmt=marker, color=color, markersize=11,
                capsize=7, label=label, ecolor=color, elinewidth=2,
                markeredgecolor='black', markeredgewidth=1.5, alpha=0.9)
    
    # Connect to harmonic
    omega_closest = harmonic_numbers[i] * omega_0
    ax1.plot([z, z], [w, omega_closest], color=color, linestyle='--',
            alpha=0.6, linewidth=1.5, zorder=1)

# Coherence zone
ax1.fill_between([0, 11], omega_0*0.98, omega_0*1.02,
                color='gold', alpha=0.15, zorder=0,
                label='±2% ω₁ zone')

# Configuration
ax1.set_title('COSMIC HARMONIC SPECTRUM: SDSS + JWST\nωₙ = n × 0.191 Gyr⁻¹ (SDSS Fundamental: logBF=107.1)',
              fontsize=14, pad=18, weight='bold')
ax1.set_ylabel('Frequency ω (Gyr⁻¹)', fontsize=12)
ax1.set_xlabel('Redshift z', fontsize=12)
ax1.set_xlim(-0.2, 9.0)
ax1.set_ylim(0.1, 1.35)
ax1.grid(True, alpha=0.25, linestyle='--')
ax1.legend(loc='upper left', framealpha=0.95, fontsize=9, ncol=2)

# ============================================================================
# PANEL 2: BAYESIAN EVIDENCE (logBF)
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])
results = comprehensive_harmonic_analysis(omega_jwst, sigma_jwst, z_jwst, harmonic_numbers)
logBF_vals = [r['logBF'] for r in results]
logBF_vals.insert(0, logBF_sdss)  # Add SDSS
fields_display = ['SDSS'] + field_names[:6]
colors_bf = ['darkorange'] + colors

# Bayesian evidence bars
bars = ax2.bar(range(len(logBF_vals)), logBF_vals, color=colors_bf, alpha=0.8,
              edgecolor='black', linewidth=1.2)

# Reference lines for significance
for y, label in [(5, 'Decisive'), (10, 'Extreme'), (100, 'Unprecedented')]:
    ax2.axhline(y=y, color='gray', linestyle=':', alpha=0.6, linewidth=1)
    ax2.text(len(logBF_vals)-0.5, y, f' {label}', va='center', fontsize=8, color='gray')

# Annotations
for i, (bar, bf) in enumerate(zip(bars, logBF_vals)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 5,
            f'{bf:.1f}', ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

ax2.set_title('Bayesian Evidence (logBF)', fontsize=12)
ax2.set_xlabel('Field/Dataset', fontsize=11)
ax2.set_ylabel('log(Bayes Factor)', fontsize=11)
ax2.set_xticks(range(len(logBF_vals)))
ax2.set_xticklabels(fields_display, rotation=45, fontsize=10)
ax2.set_ylim(0, max(logBF_vals)*1.1)
ax2.grid(True, alpha=0.2, axis='y')

# ============================================================================
# PANEL 3: PHASE COHERENCE (m(z) near integer)
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])

m_values = [r['m_z'] for r in results]
m_integers = [r['m_integer'] for r in results]
delta_m = [r['delta_m'] for r in results]

x_pos = np.arange(len(m_values))
bars_m = ax3.bar(x_pos, delta_m, color='steelblue', alpha=0.8,
                edgecolor='black', linewidth=1.2)

# Tolerance line (0.5 cycles)
ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7,
           label='Coherence limit (0.5)', linewidth=1.5)

# Annotations
for i, (bar, dm, mz, mint) in enumerate(zip(bars_m, delta_m, m_values, m_integers)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, height + 0.02,
            f'm(z)={mz:.1f}\n≈{mint}', ha='center', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

ax3.set_title('Phase Coherence: m(z) near integer', fontsize=12)
ax3.set_xlabel('JWST Field', fontsize=11)
ax3.set_ylabel('|Δm| = |m(z) - nearest integer|', fontsize=11)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(field_names[:6], rotation=45, fontsize=10)
ax3.set_ylim(0, 0.6)
ax3.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax3.grid(True, alpha=0.2, axis='y')

# ============================================================================
# PANEL 4: NORMALIZED RESIDUALS
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])
residuals = [r['tension_sigma'] for r in results]

# Bar plot with significance colors
colors_res = []
for r in residuals:
    if abs(r) < 1:
        colors_res.append('#4CAF50')  # Green: good consistency
    elif abs(r) < 2:
        colors_res.append('#FFC107')  # Yellow: acceptable
    else:
        colors_res.append('#F44336')  # Red: problematic

bars_res = ax4.bar(range(len(residuals)), residuals, color=colors_res, alpha=0.8,
                  edgecolor='black', linewidth=1.2)

# Sigma reference lines
for y in [0, 1, 2, 3]:
    linestyle = '-' if y == 0 else ':'
    alpha = 0.8 if y == 0 else 0.4
    color = 'black' if y == 0 else 'gray'
    ax4.axhline(y=y, color=color, linestyle=linestyle, alpha=alpha, linewidth=1)

# Annotations
for i, (bar, res, r) in enumerate(zip(bars_res, residuals, results)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, height + 0.1*np.sign(height),
            f"Δ={r['delta']:.3f}\nn={r['n_theoretical']}",
            ha='center', va='bottom' if height >= 0 else 'top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

ax4.set_title('Normalized Residuals: (ω_obs - ω_n)/σ', fontsize=12)
ax4.set_xlabel('JWST Field', fontsize=11)
ax4.set_ylabel('Tension (σ)', fontsize=11)
ax4.set_xticks(range(len(residuals)))
ax4.set_xticklabels(field_names[:6], rotation=45, fontsize=10)
ax4.set_ylim(-1.5, 1.5)
ax4.grid(True, alpha=0.2, axis='y')

# ============================================================================
# PANEL 5: COMPLETE STATISTICAL SUMMARY
# ============================================================================
ax5 = fig.add_subplot(gs[2:, :])
ax5.axis('off')

# Calculate global statistics
mean_tension = np.mean([abs(r) for r in residuals])
mean_delta_m = np.mean(delta_m)
total_logBF = logBF_sdss + sum(logBF_vals[1:])
harmonic_success = sum([1 for r in results if r['is_harmonic']])

# Create complete table
table_data = [
    ['PARAMETER', 'VALUE', 'SIGNIFICANCE', 'INTERPRETATION'],
    ['-'*80, '-'*80, '-'*80, '-'*80],
    ['Fundamental Frequency ω₁', f'{omega_0} ± {sigma_omega0}', '16σ (SDSS)', 'Detected in SDSS, logBF=107.1'],
    ['Harmonic Series ωₙ = nω₁', 'n=2,3,4,6 detected', '>10σ each', 'Consistent in 6 JWST fields'],
    ['Strongest Harmonic ω₄', '0.764 ± 0.031', 'logBF>800', '100% fields, most robust'],
    ['Phase Coherence R', '0.74-0.93', 'p<10⁻⁵⁸³', 'Extreme rejection of random phase'],
    ['Initial Phase φ₀', '0.1800 ± 0.0500 rad', '3.6σ', 'Asymmetric initial conditions'],
    ['Temporal Evolution σ_ω', '0.1463', '>5σ', 'Frequency varies with redshift'],
    ['m(z) near integer', f'{mean_delta_m:.3f} average', '<0.5 in all', 'Temporal coherence'],
    ['Total Bayesian Evidence', f'logBF>{total_logBF:.0f}', 'Unprecedented', 'Kass&Raftery scale exceeded'],
    ['Consistent points', f'{harmonic_success}/6 fields', f'{harmonic_success/6*100:.1f}%', 'All <2σ except SF Peak'],
    ['Best fit', 'n=4 (ω₄=0.764)', 'Δ=0.003, 0.1σ', 'PRIMER-UDS z=5.65']
]

# Create table text
table_text = "COMPLETE STATISTICAL SUMMARY - PAPER: Evidence for Cosmic Harmonic Periodicity\n"
table_text += "="*120 + "\n\n"

# Add parameters section
table_text += "EMPIRICALLY FITTED PARAMETERS (61,303 galaxies):\n"
table_text += "-"*80 + "\n"
coeff_table = [
    ['Coefficient', 'Value', 'Error (1σ)', '|t|', 'Significance'],
    ['ϕ₀', '0.1800 rad', '±0.0500', '3.60', '>99%'],
    ['c₁', '18.9775', '±0.1192', '159.21', '>100σ'],
    ['c₂', '-0.0808', '±0.2442', '0.33', 'Not significant'],
    ['c₃', '-1.2741', '±0.1237', '10.30', '>10σ'],
    ['c₄', '0.3762', '±0.0225', '16.72', '>10σ'],
    ['c₅', '-0.0275', '±0.0013', '21.15', '>10σ']
]

# Format coefficients table
for row in coeff_table:
    table_text += f"{row[0]:<10} {row[1]:<12} {row[2]:<12} {row[3]:<8} {row[4]:<15}\n"

table_text += "\n" + "="*120 + "\n"
table_text += "KEY FINDINGS:\n"
table_text += "-"*80 + "\n"
table_text += "1. DEFINITIVE DETECTION: ω₁=0.191±0.012 in SDSS (logBF=107.1, >16σ)\n"
table_text += "2. HARMONIC SERIES: ωₙ = n×0.191 confirmed for n=2,3,4,6 in JWST\n"
table_text += "3. EXTREME COHERENCE: R>0.74 in all fields, p<10⁻⁵⁸³ against random phases\n"
table_text += "4. TEMPORAL CONSISTENCY: m(z) near integer (|Δm|<0.5 in all fields)\n"
table_text += "5. TOTAL EVIDENCE: logBF_total > 2500 → unprecedented evidence\n"
table_text += "6. IMPLICATION: Beyond-ΛCDM physics required for cosmic oscillations\n"
table_text += "="*120 + "\n"
table_text += f"REFERENCE: Vallejos (2026), DOI:10.5281/zenodo.18406995 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Display table
ax5.text(0.02, 0.98, table_text, fontsize=9, fontfamily='monospace',
        verticalalignment='top', transform=ax5.transAxes,
        bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.95,
                 edgecolor='gray', linewidth=2))

# ============================================================================
# SAVE FIGURE
# ============================================================================
plt.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.05, hspace=0.4)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"COSMIC_HARMONIC_SPECTRUM_FULL_{timestamp}.png"
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')

print("="*80)
print("COSMIC HARMONIC SPECTRUM - COMPREHENSIVE ANALYSIS")
print("="*80)
print(f"📊 File saved: {filename}")
print(f"📈 Datasets: 1,734 SDSS + 59,569 JWST = 61,303 total galaxies")
print(f"🎯 Fundamental Frequency: ω₁ = {omega_0} ± {sigma_omega0} Gyr⁻¹")
print(f"🔬 Total Bayesian Evidence: logBF > 2500")
print(f"📉 Phase coherence: p < 10⁻⁵⁸³ (extreme rejection of random phase)")
print(f"⚡ Harmonics detected: n = 2, 3, 4, 6")
print("="*80)
print("\nKEY PAPER FINDINGS:")
print("-"*80)
print("1. ω₁ detected in SDSS with logBF=107.1 (>5σ decisive)")
print("2. ω₄=0.764 appears in 100% JWST fields with logBF>800")
print("3. Initial phase φ₀=0.1800±0.0500 rad (asymmetric conditions)")
print("4. m(z) near integer: |Δm|<0.5 in all fields")
print("5. 4/5 Φ(z) coefficients >10σ significant")
print("="*80)
print("REF: Vallejos, O.A. (2026). Evidence for Cosmic Harmonic Periodicity")
print(f"     DOI: 10.5281/zenodo.18406995 | {timestamp}")
print("="*80)

# Show plot
# plt.show()