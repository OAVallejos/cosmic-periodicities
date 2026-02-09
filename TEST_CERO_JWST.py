import numpy as np
import gzip
from pathlib import Path

def scan_void_border(filepath, target_z, start_width=0.001, end_width=0.050, steps=15):
    print(f"\n🔍 SCANNING BORDER AT: {filepath.name}")
    print(f"🎯 Void center: z = {target_z}")
    print("-" * 50)
    print(f"{'Window (±z)':<15} | {'Total Width':<15} | {'Galaxies'}")
    print("-" * 50)

    # Load redshifts only once for speed
    zs = []
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    z = float(parts[4])
                    if 9.0 < z < 12.0: # Quick filter for area of interest
                        zs.append(z)
                except: continue

    z_array = np.array(zs)

    # Scan by increasing window width
    for width in np.linspace(start_width, end_width, steps):
        mask = np.abs(z_array - target_z) < width
        count = np.sum(mask)

        status = "✅ ZERO" if count == 0 else f"⚠️ {count} gal."
        print(f"{width:.4f}{'':<10} | {width*2:.4f}{'':<10} | {status}")

# --- EXECUTION ---
data_path = Path("data/primerup.dat.gz") # We use the densest one
if data_path.exists():
    scan_void_border(data_path, target_z=10.191)
else:
    print("File not found.")
    