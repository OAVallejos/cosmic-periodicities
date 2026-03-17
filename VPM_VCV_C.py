#!/usr/bin/env python3
"""
VPM-48: PHASE DIAGRAM Ψ - Δα WITH REDSHIFT COLOR CODING
Shows the photon's path through the crystal from z=12 to z=0
The loop area represents the energy dissipated by vacuum viscosity
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.patches import Polygon
from pathlib import Path
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d

# ============================================================================
# STYLE CONFIGURATION (Nature/Science style)
# ============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'figure.figsize': (10, 8)
})

# Base colors
COLOR_TRAJECTORY = 'black'
COLOR_LOOP = '#d62728'  # Red for hysteresis loop
COLOR_FINAL_POINT = '#2ca02c'  # Green for z=0
COLOR_INITIAL_POINT = '#9467bd'  # Purple for z=12

# ============================================================================
# VPM-48 MODEL PARAMETERS
# ============================================================================
z_full = np.linspace(0, 12, 5000)  # High resolution up to z=12

omega_0 = 0.191      # Fundamental crystal frequency [18916234.pdf]
xi = 0.084           # Torsion coupling (local coupling) [VPM_VCV.pdf]
delta_max = 0.35     # CMB observed birefringence (Planck 2018)
z_c = 0.21           # Glass transition redshift [VPM_VCV.pdf]
tau = 3.5            # Damping time in Gyr [18445289.pdf]

# ============================================================================
# 1. CRYSTAL WAVE FUNCTION Ψ(z)
# ============================================================================
# Represents the probability density of finding galaxies at nodes
# Based on the periodic structure detected in 85,390 galaxies [18916234.pdf]

phase = 2 * np.pi * (z_full / omega_0)
damping = np.exp(-z_full / tau)
psi = np.cos(phase) * damping

# ============================================================================
# 2. ACCUMULATED BIREFRINGENCE Δα(z)
# ============================================================================
# CONSTITUTIVE RELATION: d(Δα)/dz ∝ ξ · Ψ²
# Birefringence accumulates the oscillation energy

d_alpha_dz = xi * (psi**2)

# Accumulated integral using trapezoid (replaces trapz)
delta_alpha = np.zeros_like(z_full)
for i in range(len(z_full)):
    mask = z_full >= z_full[i]
    delta_alpha[i] = trapezoid(d_alpha_dz[mask], z_full[mask])

# Normalize to the observed CMB value
delta_alpha = (delta_alpha / delta_alpha[0]) * delta_max

# ============================================================================
# 3. FILTER FOR REDSHIFT ≤ 12 (range of interest)
# ============================================================================
mask_z = z_full <= 12
z = z_full[mask_z]
psi_z = psi[mask_z]
delta_z = delta_alpha[mask_z]

# ============================================================================
# 4. PHASE DIAGRAM Ψ vs Δα WITH COLOR CODING
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# ----------------------------------------------------------------------------
# Trajectory with redshift color coding
# ----------------------------------------------------------------------------
# Logarithmic normalization for better visualization (more resolution at low z)
norm = PowerNorm(gamma=0.5, vmin=0.01, vmax=12)
cmap = plt.cm.plasma  # 'plasma', 'viridis', 'inferno', 'magma'

# Create points for colored line
points = np.array([delta_z, psi_z]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

from matplotlib.collections import LineCollection
lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2, alpha=0.8)
lc.set_array(z[:-1])  # Color according to redshift
ax.add_collection(lc)

# ----------------------------------------------------------------------------
# Special points
# ----------------------------------------------------------------------------
# Initial point (z=12)
ax.scatter(delta_z[0], psi_z[0], s=200, color=COLOR_INITIAL_POINT,
          edgecolor='black', linewidth=2, zorder=10, marker='o',
          label=f'Start: z=12 (early universe)')

# Final point (z=0)
ax.scatter(delta_z[-1], psi_z[-1], s=250, color=COLOR_FINAL_POINT,
          edgecolor='black', linewidth=2, zorder=10, marker='s',
          label=f'End: z=0 (today)')

# Glass transition point (z_c)
idx_c = np.argmin(np.abs(z - z_c))
ax.scatter(delta_z[idx_c], psi_z[idx_c], s=180, color='white',
          edgecolor='black', linewidth=2, zorder=10, marker='D',
          label=f'Glass transition: z_c={z_c}')

# ----------------------------------------------------------------------------
# HYSTERESIS LOOP (represents dissipated energy)
# ----------------------------------------------------------------------------
# Calculate loop area (energy dissipated by vacuum viscosity)
# The loop forms between the actual trajectory and an "ideal" dissipationless curve

# Create smooth interpolation of the trajectory
f_psi = interp1d(delta_z, psi_z, kind='cubic', fill_value='extrapolate')

# Delta range for the loop
delta_range = np.linspace(delta_z.min(), delta_z.max(), 200)

# Actual trajectory
psi_real = f_psi(delta_range)

# "Ideal" trajectory (without dissipation) - simplified as straight line
# between initial and final points
psi_ideal = np.linspace(psi_z[0], psi_z[-1], len(delta_range))

# Create loop polygon (difference between real and ideal)
loop_x = np.concatenate([delta_range, delta_range[::-1]])
loop_y = np.concatenate([psi_real, psi_ideal[::-1]])
loop_points = np.array([loop_x, loop_y]).T

# Create polygon with transparency
loop_poly = Polygon(loop_points, alpha=0.2, color=COLOR_LOOP,
                   label='Hysteresis loop (dissipated energy)', linewidth=0)
ax.add_patch(loop_poly)

# Calculate loop area (dissipated energy)
area_loop = np.abs(trapezoid(psi_real - psi_ideal, delta_range))

# ----------------------------------------------------------------------------
# ANNOTATIONS
# ----------------------------------------------------------------------------
# Time direction
mid_idx = len(delta_z) // 2
ax.annotate('', xy=(delta_z[mid_idx + 500], psi_z[mid_idx + 500]),
           xytext=(delta_z[mid_idx], psi_z[mid_idx]),
           arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(delta_z[mid_idx + 300], psi_z[mid_idx + 300] + 0.05,
        'Time flow', fontsize=10, ha='center')

# Constitutive relation
ax.text(0.05, 0.95,
        r'$\frac{d\Delta\alpha}{dz} \propto \xi \cdot \Psi^2(z)$' + '\n' +
        r'$\xi = 0.084$ (torsion coupling)',
        transform=ax.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9,
                 edgecolor=COLOR_LOOP, linewidth=2))

# Loop area
ax.text(0.75, 0.15, f'Loop area = {area_loop:.4f}\n(Energy dissipated by viscosity)',
        transform=ax.transAxes, fontsize=11, ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9,
                 edgecolor=COLOR_LOOP, linewidth=1.5))

# ----------------------------------------------------------------------------
# COLOR BAR (redshift)
# ----------------------------------------------------------------------------
cbar = plt.colorbar(lc, ax=ax, label='Redshift $z$', pad=0.02)
cbar.set_label('Redshift $z$', fontsize=12, fontweight='bold')

# ----------------------------------------------------------------------------
# AXIS CONFIGURATION
# ----------------------------------------------------------------------------
ax.set_xlabel(r'Accumulated Birefringence $\Delta\alpha$ (°)', fontsize=14, fontweight='bold')
ax.set_ylabel(r'Wave Function $\Psi(z)$ (a.u.)', fontsize=14, fontweight='bold')
ax.set_title(r'VPM-48: Photon Path through the Cosmic Crystal',
             fontsize=16, fontweight='bold', pad=20)

# Reference lines
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax.axvline(x=delta_max, color='gray', linestyle='--', linewidth=1, alpha=0.5,
           label=f'δ_CMB = {delta_max}°')

# Grid
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

# Legend
ax.legend(loc='lower left', fontsize=10, framealpha=0.95)

# Limits
ax.set_xlim(0, delta_max + 0.05)
ax.set_ylim(-1.2, 1.2)

# ----------------------------------------------------------------------------
# EMPIRICAL DATA BOX
# ----------------------------------------------------------------------------
empirical_text = (
    f"EMPIRICAL DATA:\n"
    f"• 85,390 SDSS-V + JWST galaxies [18916234]\n"
    f"• Fundamental frequency ω₀ = {omega_0}\n"
    f"• Global coherence R = 0.9217 (σ = 99.0)\n"
    f"• CMB birefringence δ = {delta_max}° (Planck 2018)\n"
    f"• Glass transition z_c = {z_c}\n"
    f"• Dissipated energy = {area_loop:.4f} a.u."
)

fig.text(0.02, 0.02, empirical_text, fontsize=8,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9,
                  edgecolor='black', linewidth=1),
         verticalalignment='bottom')

plt.tight_layout()

# ============================================================================
# SAVE FIGURE
# ============================================================================
output_dir = Path("vpm48_figures")
output_dir.mkdir(exist_ok=True)

plt.savefig(output_dir / 'vpm48_phase_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_dir / 'vpm48_phase_diagram.pdf', bbox_inches='tight', facecolor='white')
plt.savefig(output_dir / 'vpm48_phase_diagram.eps', bbox_inches='tight', facecolor='white')

print("\n" + "="*80)
print("📊 VPM-48: PHASE DIAGRAM WITH REDSHIFT COLOR CODING")
print("="*80)
print(f"\n📈 PARAMETERS:")
print(f"   • ξ (local torsion): {xi}")
print(f"   • δ (CMB birefringence): {delta_max}°")
print(f"   • z_c (glass transition): {z_c}")
print(f"   • τ (damping): {tau} Gyr")
print(f"   • z range: 0 - 12")

print(f"\n🧮 DISSIPATED ENERGY:")
print(f"   • Hysteresis loop area: {area_loop:.6f} a.u.")
print(f"   • Interpretation: Energy transferred from crystal to EM field")
print(f"   • Vacuum viscosity converts coherence into rotation")

print(f"\n✅ Figures saved in: {output_dir.absolute()}/")
print(f"   • vpm48_phase_diagram.png")
print(f"   • vpm48_phase_diagram.pdf")
print(f"   • vpm48_phase_diagram.eps")
print("="*80)