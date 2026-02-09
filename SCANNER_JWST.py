#!/usr/bin/env python3
"""

Scan void boundaries with 0.0001 resolution
"""
import gzip
import numpy as np
import sys

def scan_void_boundary(filename, z_center, max_width=0.05, step=0.0005):
    """Scan void boundary with high resolution"""
    print(f"\n🔬 SCANNING BOUNDARY: {filename} z={z_center}")
    print("-" * 60)

    counts = []
    windows = []

    # Generate windows from center outward
    for i in range(int(max_width/step) + 1):
        width = i * step
        z_min = z_center - width
        z_max = z_center + width

        # Count galaxies
        count = 0
        try:
            with gzip.open(f"data/{filename}", 'rt') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            z = float(parts[4])
                            if z_min <= z <= z_max:
                                count += 1
                        except:
                            continue
        except:
            count = -1

        counts.append(count)
        windows.append(width)

        # Show when changes from 0
        if i > 0 and counts[i-1] == 0 and count > 0:
            print(f"🎯 BOUNDARY DETECTED:")
            print(f"   • z = {z_center} ± {width-step:.4f} → 0 galaxies")
            print(f"   • z = {z_center} ± {width:.4f} → {count} galaxies")
            print(f"   • Sharp change at ±{width:.4f}")

    return windows, counts

# Voids to analyze
voids_to_scan = [
    ("primerup.dat.gz", 10.191),
    ("primerup.dat.gz", 10.579),
    ("primerup.dat.gz", 10.764),
    ("primerup.dat.gz", 11.155),
    ("primerup.dat.gz", 11.340),
    ("primerup.dat.gz", 11.722),
    ("primercp.dat.gz", 10.191),
    ("ceersp.dat.gz", 10.191),
]

print("="*80)
print("🛰️  HIGH RESOLUTION SCAN - VOID BOUNDARIES")
print("="*80)

for filename, z_center in voids_to_scan:
    windows, counts = scan_void_boundary(filename, z_center, max_width=0.02, step=0.0002)

    # Find exact void width
    void_width = 0
    for i, count in enumerate(counts):
        if count > 0:
            void_width = windows[i-1] if i > 0 else 0
            break

    print(f"📐 Void z={z_center}: width = ±{void_width:.4f} (total: {void_width*2:.4f})")
    print()