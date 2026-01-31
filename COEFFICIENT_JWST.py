#!/usr/bin/env python3
"""
FINAL COEFFICIENT GENERATOR - CORRECTED VERSION
"""

import numpy as np
import json
from datetime import datetime

# ============================================================================
# YOUR COEFFICIENTS (from previous output)
# ============================================================================

# From your output:
# c1 = 18.9775 ± 0.1192
# c2 = -0.0808 ± 0.2442
# c3 = -1.2741 ± 0.1237
# c4 = 0.3762 ± 0.0225
# c5 = -0.0275 ± 0.0013
# φ0 = 0.1800 rad

coeffs = [18.9775, -0.0808, -1.2741, 0.3762, -0.0275]
coeffs_err = [0.1192, 0.2442, 0.1237, 0.0225, 0.0013]
phi0 = 0.1800

# ============================================================================
# 1. CREATE CORRECTED PYTHON FUNCTION
# ============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
py_file = f'function_phi_z_{timestamp}.py'

# Create corrected Python file content
py_content = '''#!/usr/bin/env python3
"""
ADJUSTED Φ(z) FUNCTION - FINAL RESULTS
φ0 = 0.180000 rad
Coefficients fitted with least squares
"""

import numpy as np

# Fitted coefficients
C1 = 18.977500
C2 = -0.080800
C3 = -1.274100
C4 = 0.376200
C5 = -0.027500
PHI0 = 0.180000

def Phi(z, phi0=PHI0):
    """
    Accumulated phase Φ(z) = φ0 + Σ (c_k/k) * z^k
    """
    return phi0 + (
        (C1/1.0) * z**1 +
        (C2/2.0) * z**2 +
        (C3/3.0) * z**3 +
        (C4/4.0) * z**4 +
        (C5/5.0) * z**5
    )

def Omega(z):
    """
    Derivative Ω(z) = dΦ/dz = Σ c_k * z^(k-1)
    """
    return (
        C1 * z**0 +
        C2 * z**1 +
        C3 * z**2 +
        C4 * z**3 +
        C5 * z**4
    )

if __name__ == "__main__":
    print("Φ(z) FUNCTION - FINAL RESULTS")
    print("="*50)

    redshifts = [0.10, 0.71, 2.70, 5.65, 6.40, 9.00]
    for redshift in redshifts:
        phi = Phi(redshift)
        omega = Omega(redshift)
        print(f"z = {redshift:.2f}: Φ = {phi:.2f} rad, Ω = {omega:.2f} rad/z")
'''

with open(py_file, 'w') as f:
    f.write(py_content)

print(f"✅ Python file created: {py_file}")

# ============================================================================
# 2. CREATE JSON FILE
# ============================================================================

json_file = f'coefficients_{timestamp}.json'

results = {
    'timestamp': timestamp,
    'phi0': phi0,
    'coefficients': {
        'c1': {'value': coeffs[0], 'error': coeffs_err[0], 'significant': abs(coeffs[0]) > 2*coeffs_err[0]},
        'c2': {'value': coeffs[1], 'error': coeffs_err[1], 'significant': abs(coeffs[1]) > 2*coeffs_err[1]},
        'c3': {'value': coeffs[2], 'error': coeffs_err[2], 'significant': abs(coeffs[2]) > 2*coeffs_err[2]},
        'c4': {'value': coeffs[3], 'error': coeffs_err[3], 'significant': abs(coeffs[3]) > 2*coeffs_err[3]},
        'c5': {'value': coeffs[4], 'error': coeffs_err[4], 'significant': abs(coeffs[4]) > 2*coeffs_err[4]}
    },
    'phi_function': f"Φ(z) = {phi0:.4f} + {coeffs[0]:.4f}·z + {coeffs[1]/2:.4f}·z² + {coeffs[2]/3:.4f}·z³ + {coeffs[3]/4:.4f}·z⁴ + {coeffs[4]/5:.4f}·z⁵",
    'omega_derivative': f"Ω(z) = {coeffs[0]:.4f} + {coeffs[1]:.4f}·z + {coeffs[2]:.4f}·z² + {coeffs[3]:.4f}·z³ + {coeffs[4]:.4f}·z⁴"
}

with open(json_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ JSON file created: {json_file}")

# ============================================================================
# 3. CREATE PREPRINT TEXT
# ============================================================================

txt_file = f'preprint_text_{timestamp}.txt'

# Calculate values for different redshifts
redshifts_calc = [0.10, 0.71, 2.70, 5.65, 6.40, 9.00]
phi_values = []
omega_values = []

for z in redshifts_calc:
    phi_z = phi0 + sum([(coeffs[k]/(k+1)) * z**(k+1) for k in range(5)])
    omega_z = sum([coeffs[k] * z**k for k in range(5)])
    phi_values.append(phi_z)
    omega_values.append(omega_z)

preprint_text = f"""
==================================================
EQUATIONS FOR YOUR PREPRINT - FINAL VERSION
==================================================

I. PHASE FUNCTION Φ(z)

Φ(z) = φ₀ + Σₖ (cₖ/k) zᵏ

With:
φ₀ = {phi0:.4f} rad (phase coherence)

Fitted coefficients:
c₁ = {coeffs[0]:.4f} ± {coeffs_err[0]:.4f} {'(significant)' if abs(coeffs[0]) > 2*coeffs_err[0] else '(not significant)'}
c₂ = {coeffs[1]:.4f} ± {coeffs_err[1]:.4f} {'(significant)' if abs(coeffs[1]) > 2*coeffs_err[1] else '(not significant)'}
c₃ = {coeffs[2]:.4f} ± {coeffs_err[2]:.4f} {'(significant)' if abs(coeffs[2]) > 2*coeffs_err[2] else '(not significant)'}
c₄ = {coeffs[3]:.4f} ± {coeffs_err[3]:.4f} {'(significant)' if abs(coeffs[3]) > 2*coeffs_err[3] else '(not significant)'}
c₅ = {coeffs[4]:.4f} ± {coeffs_err[4]:.4f} {'(significant)' if abs(coeffs[4]) > 2*coeffs_err[4] else '(not significant)'}

Explicit form:
Φ(z) = {phi0:.4f} + {coeffs[0]:.4f}·z + {coeffs[1]/2:.4f}·z² + {coeffs[2]/3:.4f}·z³ + {coeffs[3]/4:.4f}·z⁴ + {coeffs[4]/5:.4f}·z⁵

II. DERIVATIVE Ω(z) = dΦ/dz

Ω(z) = Σₖ cₖ zᵏ⁻¹

Ω(z) = {coeffs[0]:.4f} + {coeffs[1]:.4f}·z + {coeffs[2]:.4f}·z² + {coeffs[3]:.4f}·z³ + {coeffs[4]:.4f}·z⁴

III. WAVE FUNCTION Ψ(z)

Ψ(z) = A(z) cos[Φ(z)]

With amplitude:
A(z) = A₀ exp[-z/τ(z)]
τ(z) = τ₀ [1 + γ Ω(z)/Φ(z)]

IV. VALUES AT KEY POINTS

"""

# Add calculated values
for i, z in enumerate(redshifts_calc):
    preprint_text += f"z={z:.2f}: Φ = {phi_values[i]:.2f} rad, Ω = {omega_values[i]:.2f} rad/z\n"

preprint_text += """
==================================================
"""

with open(txt_file, 'w') as f:
    f.write(preprint_text)

print(f"✅ Preprint text: {txt_file}")

# ============================================================================
# 4. DISPLAY RESULTS
# ============================================================================

print("\n" + "="*60)
print("FINAL COEFFICIENTS FOR YOUR PREPRINT")
print("="*60)

print(f"\nφ₀ = {phi0:.4f} rad")

print("\nFitted coefficients:")
for i, (c, err) in enumerate(zip(coeffs, coeffs_err), 1):
    sig = "✅" if abs(c) > 2*err else "⚠️ "
    print(f"  {sig} c{i} = {c:8.4f} ± {err:7.4f}")

print(f"\n📐 Φ(z) function:")
print(f"  Φ(z) = {phi0:.4f} + {coeffs[0]:.4f}·z + {coeffs[1]/2:.4f}·z² + {coeffs[2]/3:.4f}·z³ + {coeffs[3]/4:.4f}·z⁴ + {coeffs[4]/5:.4f}·z⁵")

print(f"\n📈 Ω(z) = dΦ/dz function:")
print(f"  Ω(z) = {coeffs[0]:.4f} + {coeffs[1]:.4f}·z + {coeffs[2]:.4f}·z² + {coeffs[3]:.4f}·z³ + {coeffs[4]:.4f}·z⁴")

print(f"\n📊 Values at key redshifts:")
for i, z in enumerate(redshifts_calc):
    print(f"  z={z:.2f}: Φ = {phi_values[i]:.2f} rad, Ω = {omega_values[i]:.2f} rad/z")

print(f"\n📋 Generated files:")
print(f"  1. {py_file} → Executable Python function")
print(f"  2. {json_file} → Metadata in JSON")
print(f"  3. {txt_file} → Preprint text")

print(f"\n💡 To test the function:")
print(f"  python {py_file}")

print("\n" + "="*60)
print("READY! Copy the equations into your preprint.")
print("="*60)

# ============================================================================
# 5. TEST THE GENERATED FUNCTION
# ============================================================================

print("\n🔍 Testing the generated function...")
try:
    # Execute the generated file
    import subprocess
    result = subprocess.run(['python3', py_file], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"⚠️  Warning: {result.stderr}")
except Exception as e:
    print(f"⚠️  Could not test automatically: {e}")