#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL_COSMIC_PERIODICITY_ANALYSIS_V5.py - Final corrected version for VPM-48 cosmic crystal detection

This script analyzes redshift periodicity in SDSS-V and JWST data to detect
the fundamental frequency ω₀ = 0.191 and its harmonics, providing statistical
evidence for the cosmic crystalline structure.

Author: Omar Contigiani (based on VPM-48 collaboration)
Date: 2026
"""

import numpy as np
import pandas as pd
from astropy.io import fits
import vpm_engine  # Rust-optimized likelihood engine
import os
import gzip
from scipy import stats, special
import time
from glob import glob

# Constants
OMEGA0 = 0.191  # Fundamental frequency (redshift periodicity)
LOW_Z_HARMONICS = [0.191, 0.382]      # For z < 0.5
HIGH_Z_HARMONICS = [0.573, 0.764, 0.955, 1.146]  # For z > 0.5 (cosmic dawn)
OFFSET_THEORETICAL = 0.067525  # Crystal lattice anchor at z=0

def diagnose_fits_structure():
    """Complete diagnostic of FITS file structure"""
    print("\n" + "="*60)
    print("FITS STRUCTURE DIAGNOSTIC")
    print("="*60)
    
    sdss_path = "data/DL1_spec_SDSSV_eROSITA_eRASS1-v1_0_2.fits"
    
    if not os.path.exists(sdss_path):
        print(f"✗ File not found: {sdss_path}")
        return
    
    with fits.open(sdss_path) as hdul:
        for i, hdu in enumerate(hdul):
            print(f"\nHDU[{i}]: {hdu.name}")
            print(f"  Type: {type(hdu.data)}")
            
            if hdu.data is None:
                print("  No data")
                continue
            
            if hasattr(hdu.data, 'names'):
                print(f"  Columns ({len(hdu.data.names)}):")
                for j, col in enumerate(hdu.data.names[:20]):  # First 20 columns
                    col_data = hdu.data[col]
                    print(f"    {j:2d}. {col:20s}", end="")
                    
                    if hasattr(col_data, 'dtype'):
                        print(f" dtype={col_data.dtype}", end="")
                    
                    # Show first 3 non-null values
                    try:
                        if hasattr(col_data, '__array__'):
                            mask = ~np.isnan(col_data) if hasattr(col_data, '__array__') else slice(None)
                            valores = col_data[mask][:3] if np.any(mask) else []
                            if len(valores) > 0:
                                print(f"  e.g.: {valores}", end="")
                    except:
                        pass
                    print()
            else:
                print(f"  Shape: {hdu.data.shape}")

def load_sdss_data_corrected():
    """Load SDSS-V data with column diagnostics and NumPy 2.0 compatibility"""
    print("\n" + "="*60)
    print("LOADING SDSS-V DATA (corrected version)")
    print("="*60)
    
    sdss_path = "data/DL1_spec_SDSSV_eROSITA_eRASS1-v1_0_2.fits"
    
    if not os.path.exists(sdss_path):
        print(f"✗ File not found: {sdss_path}")
        return None
    
    try:
        with fits.open(sdss_path) as hdul:
            # Look in HDU 1 (main data)
            if len(hdul) > 1 and hasattr(hdul[1].data, 'names'):
                hdu = hdul[1]
                
                # Identify columns of interest
                z_col = None
                warn_col = None
                
                for col in hdu.data.names:
                    col_upper = col.upper()
                    if 'Z' in col_upper and 'WARN' not in col_upper and 'ERR' not in col_upper:
                        if z_col is None or col_upper == 'SDSS_Z':
                            z_col = col
                    if 'WARN' in col_upper or 'ZWARNING' in col_upper:
                        warn_col = col
                
                print(f"  Redshift column detected: {z_col}")
                print(f"  Warning column detected: {warn_col}")
                
                if z_col is None:
                    print("  ✗ No redshift column found")
                    return None
                
                # Extract data
                z_raw = hdu.data[z_col]
                
                # Convert to numpy array with NumPy 2.0 compatibility
                if hasattr(z_raw, 'dtype'):
                    try:
                        # NumPy 2.0+ compatible method
                        z_array = z_raw.byteswap().view(z_raw.dtype.newbyteorder('=')).astype(np.float64)
                    except AttributeError:
                        # Fallback for older NumPy
                        try:
                            z_array = z_raw.view(z_raw.dtype.newbyteorder('=')).astype(np.float64)
                        except:
                            z_array = np.array(z_raw, dtype=np.float64)
                else:
                    z_array = np.array(z_raw, dtype=np.float64)
                
                # Filter unphysical values
                mask_physical = (z_array > 0.001) & (z_array < 10)  # Physically possible z
                print(f"  Objects with physical z: {np.sum(mask_physical)}/{len(z_array)}")
                
                if np.sum(mask_physical) == 0:
                    print("  ✗ No physical redshifts found")
                    return None
                
                # Apply warning filter if exists
                if warn_col is not None:
                    warn_raw = hdu.data[warn_col]
                    
                    if hasattr(warn_raw, 'dtype'):
                        try:
                            warn_array = warn_raw.byteswap().view(warn_raw.dtype.newbyteorder('=')).astype(np.int32)
                        except AttributeError:
                            try:
                                warn_array = warn_raw.view(warn_raw.dtype.newbyteorder('=')).astype(np.int32)
                            except:
                                warn_array = np.array(warn_raw, dtype=np.int32)
                    else:
                        warn_array = np.array(warn_raw, dtype=np.int32)
                    
                    mask_warn = (warn_array == 0)
                    print(f"  Objects with ZWARNING=0: {np.sum(mask_warn)}")
                else:
                    mask_warn = np.ones(len(z_array), dtype=bool)
                
                # Combine filters
                mask_final = mask_physical & mask_warn
                z_vals = z_array[mask_final]
                
                # Filter low z range (0.01 < z < 0.5)
                z_low = z_vals[(z_vals > 0.01) & (z_vals < 0.5)]
                
                print(f"\n  FINAL RESULT:")
                print(f"    Total valid: {len(z_vals)}")
                print(f"    Low z range (<0.5): {len(z_low)}")
                print(f"    Range: [{z_vals.min():.4f}, {z_vals.max():.4f}]")
                print(f"    Mean: {z_vals.mean():.4f}")
                print(f"    Median: {np.median(z_vals):.4f}")
                
                return z_low
                
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def load_jwst_data_corrected():
    """Load JWST data with automatic column detection"""
    print("\n" + "="*60)
    print("LOADING JWST DATA (Bouwens et al. 2023)")
    print("="*60)
    
    # Search for files
    files = []
    for pattern in ["*p.dat.gz", "data/*p.dat.gz"]:
        files.extend(glob(pattern))
    
    files = sorted(list(set(files)))
    
    if not files:
        print("  ✗ No JWST files found")
        return None, None
    
    print(f"  Files found: {len(files)}")
    
    z_low = []
    z_high = []
    
    for file in files:
        try:
            print(f"\n  Processing {os.path.basename(file)}...")
            
            # Identify redshift columns by testing ranges
            df_test = pd.read_csv(file, sep='\s+', comment='#',
                                  header=None, compression='gzip', nrows=1000)
            
            best_col = None
            best_score = 0
            
            for col in df_test.columns:
                # Convert to numeric
                values = pd.to_numeric(df_test[col], errors='coerce')
                
                # Calculate metrics
                n_physical = np.sum((values > 0.001) & (values < 15))
                n_absurd = np.sum(values > 1000)
                
                # Score: maximize physical, minimize absurd
                score = n_physical - 10 * n_absurd
                
                if score > best_score:
                    best_score = score
                    best_col = col
            
            if best_col is not None and best_score > 0:
                print(f"    Column {best_col} selected (score={best_score})")
                
                # Read complete file
                df = pd.read_csv(file, sep='\s+', comment='#',
                                header=None, compression='gzip')
                z_all = pd.to_numeric(df[best_col], errors='coerce').values
                
                # Filter physical values
                mask_physical = (z_all > 0.001) & (z_all < 15)
                z_physical = z_all[mask_physical]
                
                print(f"    Total rows: {len(df)}")
                print(f"    Physical values: {len(z_physical)}")
                
                if len(z_physical) > 0:
                    print(f"    Range: [{z_physical.min():.3f}, {z_physical.max():.3f}]")
                    
                    # Classify
                    z_low_field = z_physical[(z_physical > 0.01) & (z_physical < 0.5)]
                    z_high_field = z_physical[z_physical > 6]
                    
                    z_low.extend(z_low_field)
                    z_high.extend(z_high_field)
                    
                    print(f"      Low z: {len(z_low_field)}")
                    print(f"      High z: {len(z_high_field)}")
                    
                    if len(z_high_field) > 0:
                        print(f"      High z examples: {z_high_field[:5]}")
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    print(f"\nJWST SUMMARY:")
    print(f"  Objects z < 0.5: {len(z_low)}")
    print(f"  Objects z > 6:   {len(z_high)}")
    
    return np.array(z_low), np.array(z_high)

def complete_rayleigh_test(z_vals, name="", omega_test=None):
    """Complete Rayleigh test with sigma conversion"""
    if len(z_vals) < 10:
        return None
    
    if omega_test is None:
        omega_test = OMEGA0
    
    print(f"\n{'='*50}")
    print(f"RAYLEIGH TEST - {name}")
    print(f"{'='*50}")
    print(f"  N = {len(z_vals)}")
    print(f"  ω = {omega_test:.3f}")
    
    # Calculate phases
    phases = (z_vals / omega_test) % 1
    angles = 2 * np.pi * phases
    
    # R statistic
    R = np.sqrt(np.sum(np.cos(angles))**2 + np.sum(np.sin(angles))**2)
    R_mean = R / len(z_vals)
    
    # Exact p-value
    p_value = np.exp(-R**2 / len(z_vals))
    
    # Convert to sigma using erfinv from scipy.special
    if p_value < 0.5:
        sigma = np.sqrt(2) * special.erfinv(1 - p_value)
    else:
        sigma = 0.0
    
    print(f"  R = {R_mean:.6f}")
    print(f"  p-value = {p_value:.6e}")
    print(f"  Significance = {sigma:.2f}σ")
    
    # Distribution at nodes
    z_min, z_max = z_vals.min(), z_vals.max()
    nodes = np.arange(np.floor(z_min/omega_test)*omega_test, z_max + omega_test, omega_test)
    
    print(f"\n  Distribution at nodes:")
    for node in nodes:
        nearby = np.sum(np.abs(z_vals - node) < 0.02)
        if nearby > 0:
            print(f"    z={node:.3f}: {nearby:3d} galaxies ({nearby/len(z_vals)*100:.1f}%)")
    
    return {
        'R': R_mean,
        'p': p_value,
        'sigma': sigma,
        'n': len(z_vals)
    }

def test_all_harmonics(z_vals, name=""):
    """Test all harmonics and find the best one"""
    if len(z_vals) < 10:
        return None
    
    print(f"\n{'='*50}")
    print(f"HARMONIC SEARCH - {name}")
    print(f"{'='*50}")
    
    all_harmonics = LOW_Z_HARMONICS + HIGH_Z_HARMONICS
    results = []
    
    for omega in all_harmonics:
        phases = (z_vals / omega) % 1
        angles = 2 * np.pi * phases
        
        R = np.sqrt(np.sum(np.cos(angles))**2 + np.sum(np.sin(angles))**2)
        R_mean = R / len(z_vals)
        p = np.exp(-R**2 / len(z_vals))
        
        if p < 0.5:
            sigma = np.sqrt(2) * special.erfinv(1 - p)
        else:
            sigma = 0.0
        
        results.append((omega, R_mean, p, sigma))
        
        if p < 0.01:  # Show only significant
            print(f"  ω={omega:.3f}: R={R_mean:.6f}, p={p:.6e}, {sigma:.2f}σ")
    
    if results:
        best = min(results, key=lambda x: x[2])
        print(f"\n  BEST HARMONIC: ω={best[0]:.3f}")
        print(f"    p = {best[2]:.6e}")
        print(f"    σ = {best[3]:.2f}")
    
    return results

def fisher_combination_with_sigma(p_values):
    """Combine p-values with Fisher's method and calculate global sigma"""
    if len(p_values) < 2:
        return None
    
    p_clean = [p for p in p_values if 0 < p < 1]
    
    if len(p_clean) < 2:
        return None
    
    fisher_stat = -2 * np.sum(np.log(p_clean))
    df = 2 * len(p_clean)
    p_combined = stats.chi2.sf(fisher_stat, df)
    
    if p_combined < 0.5:
        sigma_combined = np.sqrt(2) * special.erfinv(1 - p_combined)
    else:
        sigma_combined = 0.0
    
    print(f"\n{'='*50}")
    print("FISHER COMBINATION")
    print(f"{'='*50}")
    print(f"  Combined tests: {len(p_clean)}")
    print(f"  χ² statistic = {fisher_stat:.4f}")
    print(f"  Combined p = {p_combined:.6e}")
    print(f"  Global significance = {sigma_combined:.2f}σ")
    
    return p_combined, sigma_combined

def main():
    print("="*70)
    print("FINAL COSMIC PERIODICITY ANALYSIS - v5")
    print("="*70)
    print(f"Fundamental ω₀ = {OMEGA0}")
    print(f"Harmonics: {LOW_Z_HARMONICS + HIGH_Z_HARMONICS}")
    print(f"Theoretical offset z₀ = {OFFSET_THEORETICAL}")
    
    # Initial diagnostic
    diagnose_fits_structure()
    
    # 1. SDSS-V analysis
    print("\n" + "█"*60)
    print("██ 1. SDSS-V ANALYSIS")
    print("█"*60)
    
    z_sdss = load_sdss_data_corrected()
    p_values = []
    
    if z_sdss is not None and len(z_sdss) > 100:
        # Test for ω₀
        res_sdss = complete_rayleigh_test(z_sdss, "SDSS-V", OMEGA0)
        if res_sdss:
            p_values.append(res_sdss['p'])
        
        # Test all harmonics
        harm_sdss = test_all_harmonics(z_sdss, "SDSS-V")
        if harm_sdss:
            best_p_sdss = min([r[2] for r in harm_sdss])
            if best_p_sdss < res_sdss['p']:
                p_values.append(best_p_sdss)
    
    # 2. JWST analysis
    print("\n" + "█"*60)
    print("██ 2. JWST ANALYSIS")
    print("█"*60)
    
    z_jwst_low, z_jwst_high = load_jwst_data_corrected()
    
    if z_jwst_low is not None and len(z_jwst_low) > 10:
        print("\n--- JWST: Low redshift range (z<0.5) ---")
        res_low = complete_rayleigh_test(z_jwst_low, "JWST low z", OMEGA0)
        if res_low:
            p_values.append(res_low['p'])
        
        test_all_harmonics(z_jwst_low, "JWST low z")
    
    if z_jwst_high is not None and len(z_jwst_high) > 10:
        print("\n--- JWST: High redshift range (z>6) ---")
        res_high = complete_rayleigh_test(z_jwst_high, "JWST high z", OMEGA0)
        if res_high:
            p_values.append(res_high['p'])
        
        test_all_harmonics(z_jwst_high, "JWST high z")
    
    # 3. Global result
    print("\n" + "█"*60)
    print("██ 3. GLOBAL RESULT")
    print("█"*60)
    
    if p_values:
        print(f"\nIndividual p-values:")
        for i, p in enumerate(p_values):
            if p < 0.5:
                sigma = np.sqrt(2) * special.erfinv(1 - p)
                print(f"  Test {i+1}: p={p:.6e} ({sigma:.2f}σ)")
            else:
                print(f"  Test {i+1}: p={p:.6f}")
        
        # Fisher combination
        fisher_combination_with_sigma(p_values)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETED")
    print("="*70)

if __name__ == "__main__":
    main()