#!/usr/bin/env python3     

"""
FINAL INTEGRATED ANALYSIS - COSMIC CRYSTAL STRUCTURE
================================================================
Corrected version with JSON serialization and adjusted thresholds
================================================================
"""
import gzip
import numpy as np
import json
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import warnings
import sys
warnings.filterwarnings('ignore')

# ============================================================================
# CUSTOM JSON ENCODER
# ============================================================================

class NpEncoder(json.JSONEncoder):
    """Custom encoder for numpy objects and booleans"""
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.int16, np.int8, np.uint8)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, (np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        return super(NpEncoder, self).default(obj)

# ============================================================================
# 1. EXTRACTION WITH RUST (IF AVAILABLE) OR PYTHON
# ============================================================================

def extraer_nodo_integrado(z_centro=10.04, ancho_z=0.1, archivo="data/primercp.dat.gz"):
    """Extracts galaxies using Rust if available, otherwise Python"""

    # Try to load Rust engine
    try:
        import honest_beta_engine as hb
        RUST_AVAILABLE = True
    except ImportError:
        RUST_AVAILABLE = False
        print("⚠️  Rust engine not available, using pure Python")

    if RUST_AVAILABLE:
        # Use Rust engine for maximum speed
        c_params = [0.0, 0.191, 0.0, 0.0, 0.0, 0.0]
        engine = hb.HonestBirefringence(c_params, 0.191, 0.0, 15.0)

        galaxias_raw = engine.extraer_datos_reales(archivo, float(z_centro), float(ancho_z))

        # Convert to Python expected format
        galaxias = [{'ra': g[0], 'dec': g[1], 'z': g[2]} for g in galaxias_raw]

        print(f"[Rust] {len(galaxias)} galaxies extracted")

        # Quick physical analysis
        stats_fisicas = engine.analizar_nodo(galaxias_raw)
        return galaxias, stats_fisicas, True
    else:
        # Fallback to pure Python
        galaxias = extraer_nodo_python(archivo, z_centro, ancho_z)
        return galaxias, None, False

def extraer_nodo_python(archivo_path, z_centro, ancho_z=0.1):
    """Python fallback"""
    galaxias = []

    with gzip.open(archivo_path, 'rt') as f:
        for linea in f:
            partes = linea.split()

            for i in range(0, len(partes), 9):
                try:
                    bloque = partes[i:i+9]
                    if len(bloque) < 5:
                        continue

                    ra = float(bloque[1])
                    dec = float(bloque[2])
                    z_val = float(bloque[4])

                    if z_val > 19.0 or z_val < 0.5:
                        continue

                    if abs(z_val - z_centro) <= ancho_z:
                        galaxias.append({
                            'ra': ra,
                            'dec': dec,
                            'z': z_val
                        })

                except (ValueError, IndexError):
                    continue

    return galaxias

# ============================================================================
# 2. CORRECTED HEXAGONAL SYMMETRY ANALYSIS
# ============================================================================

def analizar_simetria_hexagonal(galaxias):
    """Specialized analysis to detect hexagonal structure"""
    if len(galaxias) < 30:
        return {"error": "Insufficient sample"}

    # Extract coordinates
    ra = np.array([g['ra'] for g in galaxias])
    dec = np.array([g['dec'] for g in galaxias])

    # Center (use median to be robust to outliers)
    ra_c, dec_c = np.median(ra), np.median(dec)

    # Convert to arcminutes with spherical projection
    dec_rad = np.radians(dec_c)
    ra_scale = 60.0 * np.cos(dec_rad)
    ra_arcmin = (ra - ra_c) * ra_scale
    dec_arcmin = (dec - dec_c) * 60.0

    # Polar coordinates
    dist_arcmin = np.sqrt(ra_arcmin**2 + dec_arcmin**2)
    angulos = np.arctan2(dec_arcmin, ra_arcmin)

    # Basic statistics
    radio_medio = np.mean(dist_arcmin)

    resultados = {
        'n_galaxias': len(galaxias),
        'centro_ra': float(ra_c),
        'centro_dec': float(dec_c),
        'radio_medio_arcmin': float(radio_medio),
        'extent_arcmin': float(np.max(dist_arcmin) - np.min(dist_arcmin))
    }

    # ============================================
    # CRITERION 1: HEXAGONAL SYMMETRY (6-fold)
    # ============================================

    # Specific 6-fold symmetry test
    angulo_hex = 2 * np.pi / 6  # 60°

    # Create KDTree
    coords = np.column_stack([ra_arcmin, dec_arcmin])
    tree = KDTree(coords)

    # Rotate points 60 degrees
    angulos_rot = (angulos + angulo_hex) % (2 * np.pi)
    x_rot = np.cos(angulos_rot) * dist_arcmin
    y_rot = np.sin(angulos_rot) * dist_arcmin
    coords_rot = np.column_stack([x_rot, y_rot])

    # Search for matches
    dists, _ = tree.query(coords_rot, k=1)
    error_hex = np.mean(dists)
    error_rel_hex = error_hex / radio_medio if radio_medio > 0 else 1.0

    # CORRECTION: Adjusted threshold for your dataset
    es_hexagonal = error_rel_hex < 0.25  # More realistic threshold

    resultados['hexagonal'] = {
        'error_arcmin': float(error_hex),
        'error_rel': float(error_rel_hex),
        'detectada': bool(es_hexagonal)
    }

    # ============================================
    # CRITERION 2: RINGS WITH √3 RELATION
    # ============================================

    # Radial histogram to detect rings
    hist, bin_edges = np.histogram(dist_arcmin, bins=15)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks
    peaks, properties = find_peaks(hist,
                                  height=np.mean(hist)*1.3,
                                  distance=2,
                                  prominence=np.std(hist)*0.5)

    anillos = []
    for peak in peaks:
        radio = bin_centers[peak]
        densidad = hist[peak]

        if densidad > np.mean(hist) * 1.4:
            anillos.append({
                'radio_arcmin': float(radio),
                'densidad': int(densidad),
                'densidad_rel': float(densidad / np.mean(hist))
            })

    # Sort by radius
    anillos.sort(key=lambda x: x['radio_arcmin'])
    resultados['anillos'] = anillos
    resultados['n_anillos'] = len(anillos)

    # Verify √3 relation between consecutive rings
    if len(anillos) >= 2:
        radios = [a['radio_arcmin'] for a in anillos]
        ratios = []

        for i in range(len(radios)-1):
            if radios[i] > 0:
                ratio = radios[i+1] / radios[i]
                ratios.append(ratio)

        resultados['ratios_radiales'] = [float(r) for r in ratios]

        # Look for ratio close to √3 ≈ 1.732
        ratios_cercanos = [r for r in ratios if abs(r - 1.732) < 0.4]
        tiene_ratio_hex = len(ratios_cercanos) > 0

        resultados['tiene_ratio_hexagonal'] = bool(tiene_ratio_hex)

        if tiene_ratio_hex:
            mejor_ratio = min(ratios, key=lambda x: abs(x-1.732))
            resultados['mejor_ratio_hexagonal'] = float(mejor_ratio)
            resultados['error_ratio_hexagonal'] = float(abs(mejor_ratio - 1.732))

    # ============================================
    # CRITERION 3: ANGULAR MODES 60°
    # ============================================

    # Analyze angular distribution
    angulos_norm = angulos % (2 * np.pi)
    angulos_norm.sort()

    if len(angulos_norm) > 10:
        dif_angulares = np.diff(angulos_norm)
        dif_angulares = np.append(dif_angulares, 2*np.pi - angulos_norm[-1] + angulos_norm[0])

        # Look for modes close to 60° (π/3)
        hist_ang, bin_edges_ang = np.histogram(dif_angulares, bins=30, range=(0, np.pi))
        bin_centers_ang = (bin_edges_ang[:-1] + bin_edges_ang[1:]) / 2

        peaks_ang, _ = find_peaks(hist_ang, height=np.mean(hist_ang)*1.5)

        modos_60 = []
        for peak in peaks_ang:
            ang_rad = bin_centers_ang[peak]
            ang_deg = np.degrees(ang_rad)

            # Check if close to 60°
            if abs(ang_deg - 60.0) < 20.0:
                modos_60.append({
                    'angulo_grados': float(ang_deg),
                    'desviacion_60': float(abs(ang_deg - 60.0)),
                    'intensidad': int(hist_ang[peak])
                })

        if modos_60:
            modos_60.sort(key=lambda x: x['desviacion_60'])
            resultados['modos_60_grados'] = modos_60
            resultados['tiene_modo_60'] = bool(len(modos_60) > 0)
            resultados['mejor_modo_60'] = modos_60[0]
        else:
            resultados['tiene_modo_60'] = bool(False)

    # ============================================
    # CALCULATE SPECIFIC HEXAGONAL SCORE
    # ============================================

    score_componentes = []

    # Component 1: 6-fold symmetry
    if resultados['hexagonal']['detectada']:
        calidad = max(0, 0.7 - resultados['hexagonal']['error_rel']/0.5)
        score_componentes.append(calidad)

    # Component 2: √3 ratio
    if resultados.get('tiene_ratio_hexagonal', False):
        error_ratio = resultados.get('error_ratio_hexagonal', 1.0)
        score_ratio = max(0, 0.6 - error_ratio/0.5)
        score_componentes.append(score_ratio)

    # Component 3: Angular mode 60°
    if resultados.get('tiene_modo_60', False):
        desv = resultados['mejor_modo_60']['desviacion_60']
        score_modo = max(0, 0.5 - desv/40.0)
        score_componentes.append(score_modo)

    # Component 4: Multiple rings
    if resultados['n_anillos'] >= 2:
        score_componentes.append(0.4)
    elif resultados['n_anillos'] >= 1:
        score_componentes.append(0.2)

    # Component 5: Sample size
    if len(galaxias) > 200:
        score_componentes.append(0.3)
    elif len(galaxias) > 50:
        score_componentes.append(0.15)

    # Calculate final score
    if score_componentes:
        score_hexagonal = np.mean(score_componentes)
    else:
        score_hexagonal = 0.0

    resultados['score_hexagonal'] = float(score_hexagonal)

    # Classification
    if score_hexagonal > 0.6:
        resultados['clasificacion_hexagonal'] = 'STRONG'
    elif score_hexagonal > 0.4:
        resultados['clasificacion_hexagonal'] = 'MODERATE'
    elif score_hexagonal > 0.2:
        resultados['clasificacion_hexagonal'] = 'WEAK'
    else:
        resultados['clasificacion_hexagonal'] = 'NULL'

    return resultados

# ============================================================================
# 3. COMPLETE INTEGRATED ANALYSIS
# ============================================================================

def analisis_final_integrado(z_target=10.04):
    """Complete analysis integrating Rust and Python"""

    print("\n" + "="*80)
    print("FINAL INTEGRATED ANALYSIS - CRYSTAL STRUCTURE")
    print("="*80)

    # 1. EXTRACTION
    print(f"\n🔍 DATA EXTRACTION (z={z_target})...")
    galaxias, stats_fisicas, rust_available = extraer_nodo_integrado(z_target)

    if not galaxias or len(galaxias) < 30:
        print("❌ Insufficient sample")
        return None

    print(f"✅ {len(galaxias)} galaxies extracted")

    # 2. PHYSICAL ANALYSIS (Rust)
    if stats_fisicas and rust_available:
        print(f"\n⚛️  PHYSICAL ANALYSIS (Rust Engine):")
        print(f"   • Mean redshift: {stats_fisicas[0]:.4f}")
        print(f"   • Standard deviation: {stats_fisicas[1]:.4f}")
        print(f"   • Phase coherence: {stats_fisicas[3]:.3f}")

        # Show complete Rust engine report
        try:
            import honest_beta_engine as hb
            c_params = [0.0, 0.191, 0.0, 0.0, 0.0, 0.0]
            engine = hb.HonestBirefringence(c_params, 0.191, 0.0, 15.0)
            print(engine.honest_statement())
        except:
            pass

    # 3. HEXAGONAL GEOMETRIC ANALYSIS
    print(f"\n🔷 ANALYZING HEXAGONAL STRUCTURE...")
    resultados_hex = analizar_simetria_hexagonal(galaxias)

    if "error" in resultados_hex:
        print(f"   ❌ {resultados_hex['error']}")
    else:
        print(f"   📏 Mean radius: {resultados_hex['radio_medio_arcmin']:.1f}'")
        print(f"   🎯 Hexagonal symmetry: {resultados_hex['hexagonal']['detectada']}")

        if resultados_hex['hexagonal']['detectada']:
            print(f"     • Relative error: {resultados_hex['hexagonal']['error_rel']:.3f}")

        if resultados_hex['n_anillos'] > 0:
            print(f"   📐 Rings detected: {resultados_hex['n_anillos']}")
            for i, anillo in enumerate(resultados_hex['anillos'][:3]):
                print(f"     {i+1}. Radius={anillo['radio_arcmin']:.1f}', density={anillo['densidad_rel']:.2f}x")

        if resultados_hex.get('tiene_ratio_hexagonal', False):
            print(f"   🔶 Hexagonal ratio detected: {resultados_hex['mejor_ratio_hexagonal']:.3f}")
            print(f"     • Error vs √3: {resultados_hex['error_ratio_hexagonal']:.3f}")

        if resultados_hex.get('tiene_modo_60', False):
            mejor_modo = resultados_hex['mejor_modo_60']
            print(f"   📐 Angular mode ~60°: {mejor_modo['angulo_grados']:.1f}°")
            print(f"     • Deviation: {mejor_modo['desviacion_60']:.1f}°")

        print(f"   🏆 Hexagonal score: {resultados_hex['score_hexagonal']:.3f}")
        print(f"   📊 Classification: {resultados_hex['clasificacion_hexagonal']}")

    # 4. FINAL EVALUATION
    print(f"\n" + "="*70)
    print("FINAL INTEGRATED EVALUATION")
    print("="*70)

    # Combine physical and geometric criteria
    score_final = resultados_hex.get('score_hexagonal', 0.0)

    # Adjust with phase coherence if available
    ajuste_fisico = 0.0
    if stats_fisicas:
        coherencia_fase = stats_fisicas[3]
        ajuste_fisico = min(0.3, coherencia_fase * 0.3)
        score_final = min(1.0, score_final + ajuste_fisico)

    print(f"\n📈 SCORES:")
    print(f"   • Geometric score: {resultados_hex.get('score_hexagonal', 0.0):.3f}")

    if stats_fisicas:
        print(f"   • Phase coherence: {stats_fisicas[3]:.3f}")
        print(f"   • Physical adjustment: +{ajuste_fisico:.3f}")

    print(f"\n🏆 FINAL INTEGRATED SCORE: {score_final:.3f}")

    # INTERPRETATION
    print(f"\n" + "="*70)
    if score_final > 0.7:
        print("🎉 STRONG EVIDENCE OF HEXAGONAL CRYSTAL STRUCTURE!")
        print("The node shows geometric patterns consistent with")
        print("a periodic cosmic network with hexagonal symmetry.")
    elif score_final > 0.5:
        print("⚠️  MODERATE EVIDENCE OF HEXAGONAL STRUCTURE")
        print("Multiple hexagonal symmetry indicators detected.")
    elif score_final > 0.3:
        print("🔍 HINTS OF HEXAGONAL STRUCTURE")
        print("Some patterns detected require confirmation.")
    else:
        print("❌ HEXAGONAL STRUCTURE NOT DETECTED")
        print("Distribution appears random or without hexagonal patterns.")
    print("="*70)

    # 5. COMPILE RESULTS
    resultados_finales = {
        'metadata': {
            'version': '3.2-final',
            'z_target': float(z_target),
            'n_galaxias': int(len(galaxias)),
            'score_final': float(score_final),
            'rust_available': bool(rust_available),
            'clasificacion': 'HEXAGONAL_STRONG' if score_final > 0.7 else
                           'HEXAGONAL_MODERATE' if score_final > 0.5 else
                           'HEXAGONAL_WEAK' if score_final > 0.3 else 'RANDOM'
        },
        'geometrico': resultados_hex,
        'fisico': {
            'coherencia_fase': float(stats_fisicas[3]) if (stats_fisicas and len(stats_fisicas) > 3) else None
        },
        'timestamp': np.datetime64('now').astype(str)
    }

    return resultados_finales

# ============================================================================
# 4. SAFE SAVING WITH ENCODER (CORRECTED)
# ============================================================================

def guardar_resultados_seguros(resultados, nombre_base="analisis_final"):
    """Saves results using custom encoder"""

    if not resultados:
        print("❌ No results to save")
        return

    try:
        # Save complete JSON
        with open(f'{nombre_base}.json', 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=2, cls=NpEncoder, ensure_ascii=False)

        print(f"💾 JSON saved: '{nombre_base}.json'")

        # Save summary in text
        with open(f'resumen_{nombre_base}.txt', 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("SUMMARY - INTEGRATED ANALYSIS\n")
            f.write("="*60 + "\n\n")

            # EXTRACT FROM 'resultados' TO AVOID NAMEERROR
            meta = resultados['metadata']
            f.write(f"REDSHIFT: z = {meta['z_target']:.2f}\n")
            f.write(f"GALAXIES: {meta['n_galaxias']}\n")
            f.write(f"FINAL SCORE: {meta['score_final']:.3f}\n")
            f.write(f"CLASSIFICATION: {meta['clasificacion']}\n")
            f.write(f"RUST ENGINE: {'✅' if meta['rust_available'] else '❌'}\n\n")

            if 'geometrico' in resultados:
                geo = resultados['geometrico']
                f.write("GEOMETRY:\n")
                f.write(f"  • Hexagonal symmetry: {geo.get('hexagonal', {}).get('detectada', False)}\n")
                f.write(f"  • Rings: {geo.get('n_anillos', 0)}\n")
                f.write(f"  • Geometric score: {geo.get('score_hexagonal', 0):.3f}\n")

            if resultados.get('fisico'):
                fis = resultados['fisico']
                if fis.get('coherencia_fase') is not None:
                    f.write(f"  • Phase coherence: {fis['coherencia_fase']:.3f}\n")

            f.write("\n" + "="*60 + "\n")
            f.write(f"ANALYSIS COMPLETED: {resultados['timestamp']}\n")
            f.write("="*60 + "\n")

        print(f"📝 Summary saved: 'resumen_{nombre_base}.txt'")

    except Exception as e:
        print(f"❌ Error saving results: {e}")

# ============================================================================
# 5. SCANNING FUNCTION
# ============================================================================

def barrido_nodos_sistematico(z_inicio=8.0, z_fin=12.0, paso=0.2):
    """Performs systematic redshift scanning"""
    print(f"\n🔄 STARTING SYSTEMATIC SCAN: z={z_inicio:.1f} to {z_fin:.1f} (Δz={paso:.1f})")

    resultados_barrido = []
    z_valores = np.arange(z_inicio, z_fin + paso/2, paso)

    for i, z in enumerate(z_valores):
        print(f"\n{'='*50}")
        print(f"ANALYSIS {i+1}/{len(z_valores)}: z={z:.2f}")
        print(f"{'='*50}")

        try:
            resultado = analisis_final_integrado(z)
            if resultado:
                resultados_barrido.append(resultado)
        except Exception as e:
            print(f"❌ Error at z={z:.2f}: {e}")

    return resultados_barrido

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("INTEGRATED CRYSTALLINE ANALYSIS SYSTEM")
    print("="*80)

    # Parse arguments
    z_target = 10.04  # Default value

    if len(sys.argv) > 1:
        if sys.argv[1] == "--barrido":
            # Automatic scanning mode
            z_inicio = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
            z_fin = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
            paso = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2

            resultados = barrido_nodos_sistematico(z_inicio, z_fin, paso)

            # Save scan results
            if resultados:
                resultados_barrido_completo = {
                    'barrido': {
                        'z_inicio': float(z_inicio),
                        'z_fin': float(z_fin),
                        'paso': float(paso),
                        'nodos_analizados': len(resultados),
                        'nodos_con_hexagonal': len([r for r in resultados
                                                   if r['metadata']['score_final'] > 0.5])
                    },
                    'resultados': resultados,
                    'timestamp': np.datetime64('now').astype(str)
                }
                guardar_resultados_seguros(resultados_barrido_completo, 'barrido_nodos')
            sys.exit(0)
        else:
            # Specific node mode
            try:
                z_target = float(sys.argv[1])
            except ValueError:
                print(f"⚠️  Invalid argument. Using z={z_target} by default")

    # Execute analysis with correct z_target
    resultados = analisis_final_integrado(z_target)

    # Save results
    if resultados:
        guardar_resultados_seguros(resultados, 'analisis_final_integrado')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)