import numpy as np
# Constants
C = 299792.458  # km/s
H0 = 74.15      # km/s/Mpc

def comoving_distance(z):
    """Simplified comoving distance for high z"""
    return (C / H0) * np.log(1 + z)

# Void size at z=10.191
z_center = 10.191
delta_z = 0.016  # Total void width

# Distances
d1 = comoving_distance(z_center - delta_z/2)
d2 = comoving_distance(z_center + delta_z/2)
size_mpc = d2 - d1

print(f"📏 Physical size of void z={z_center}±{delta_z/2}:")
print(f"   • Width in z: {delta_z}")
print(f"   • Comoving size: {size_mpc:.1f} Mpc")
print(f"   • Relation to λ₀: {size_mpc / 1682.0:.3f}λ₀")

W0 = 0.191
delta_z = 0.016

# How many W0 fit in delta_z?
n_W0 = delta_z / W0
print(f"🔢 Relation with ω₀:")
print(f"   • Δz = {delta_z}")
print(f"   • ω₀ = {W0}")
print(f"   • Δz / ω₀ = {n_W0:.3f}")
print(f"   • ≈ {n_W0:.0f}/12 of ω₀")

# Search for simple fractions
for denom in [2, 3, 4, 6, 8, 12, 16]:
    approx = round(n_W0 * denom) / denom
    error = abs(approx - n_W0) / n_W0
    if error < 0.1:
        print(f"   • Possible: Δz ≈ {round(n_W0*denom)}/{denom} × ω₀")