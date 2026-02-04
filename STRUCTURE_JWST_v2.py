#!/usr/bin/env python3
"""
MULTI-NODE CROSS-VALIDATION - 3D CRYSTAL IN MULTIPLE NODES
Compares hexagonal structure between detected abundant nodes
Optimized version for multi-node analysis
"""
import json
import numpy as np
import gzip
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
import warnings
import sys

warnings.filterwarnings('ignore')

# ============================================================================
# 1. NODE CONFIGURATION TO ANALYZE (based on your scan)
# ============================================================================
NODOS_PRIORITARIOS = [
    # (z_centro, nombre, peso)
    (10.20, "Maximum Symmetry Node", 1.0),      # Score 0.661, ratio 1.665
    (9.80,  "Maximum Density Node", 0.9),      # 739 galaxias, score 0.578
    (10.60, "Expansion Node", 0.8),            # 558 galaxias, score 0.567
    (10.00, "Transition Node", 0.7),           # 388 galaxias, score 0.556
    (9.40,  "High Density Node", 0.7),        # 499 galaxias, score 0.570
]

ANCHO_Z = 0.15  # Width to capture coherent galaxies

# ============================================================================
# 2. GALAXY EXTRACTION FUNCTION (COMPATIBLE)
# ============================================================================
def extraer_galaxias_nodo(z_centro, ancho_z=0.15, archivo="data/primercp.dat.gz"):
    """Extracts galaxies for a specific node"""
    galaxias = []

    try:
        # First try with Rust engine (if available)
        try:
            import honest_beta_engine as hb
            c_params = [0.0, 0.191, 0.0, 0.0, 0.0, 0.0]
            engine = hb.HonestBirefringence(c_params, 0.191, 0.0, 15.0)

            galaxias_raw = engine.extraer_datos_reales(archivo, float(z_centro), float(ancho_z))
            galaxias = [{'ra': g[0], 'dec': g[1], 'z': g[2]} for g in galaxias_raw]
            print(f"[Rust] {len(galaxias)} galaxies extracted")

        except ImportError:
            # Fallback to Python
            with gzip.open(archivo, 'rt') as f:
                for linea in f:
                    partes = linea.split()

                    for i in range(0, len(partes), 9):
                        bloque = partes[i:i+9]
                        if len(bloque) < 5:
                            continue

                        try:
                            ra = float(bloque[1])
                            dec = float(bloque[2])
                            z_val = float(bloque[4])
                           
                            if z_centro - ancho_z <= z_val <= z_centro + ancho_z:
                                galaxias.append({
                                    'ra': ra,
                                    'dec': dec,
                                    'z': z_val
                                })
                        except (ValueError, IndexError):
                            continue

            print(f"[Python] {len(galaxias)} galaxies extracted")

    except Exception as e:
        print(f"❌ Extraction error: {e}")

    return galaxias

# ============================================================================
# 3. STANDARDIZED HEXAGONAL ANALYSIS
# ============================================================================
def analizar_hexagonal_estandarizado(galaxias, nombre_nodo):
    """Standardized hexagonal analysis for comparison between nodes"""
    if len(galaxias) < 30:
        return None

    # Extract coordinates
    ra = np.array([g['ra'] for g in galaxias])
    dec = np.array([g['dec'] for g in galaxias])

    # Robust centroid
    ra_c, dec_c = np.median(ra), np.median(dec)

    # Convert to arcminutes
    dec_rad = np.radians(dec_c)
    ra_scale = 60.0 * np.cos(dec_rad)
    ra_arcmin = (ra - ra_c) * ra_scale
    dec_arcmin = (dec - dec_c) * 60.0

    # Coordinates for analysis
    coords_rel = np.column_stack([ra_arcmin, dec_arcmin])
    distancias = np.sqrt(ra_arcmin**2 + dec_arcmin**2)
    angulos = np.arctan2(dec_arcmin, ra_arcmin)

    # ================== HEXAGONAL ANALYSIS ==================
    resultados = {
        'nombre': nombre_nodo,
        'n_galaxias': len(galaxias),
        'centro_ra': float(ra_c),
        'centro_dec': float(dec_c),
        'radio_medio': float(np.mean(distancias))
    }

    # 1. 6-fold symmetry test
    angulo_hex = 2 * np.pi / 6
    tree = KDTree(coords_rel)

    angulos_rot = (angulos + angulo_hex) % (2 * np.pi)
    x_rot = np.cos(angulos_rot) * distancias
    y_rot = np.sin(angulos_rot) * distancias
    coords_rot = np.column_stack([x_rot, y_rot])

    dists, _ = tree.query(coords_rot, k=1)
    error_hex = np.mean(dists)
    error_rel_hex = error_hex / np.mean(distancias) if np.mean(distancias) > 0 else 1.0

    resultados['hexagonal'] = {
        'detectada': error_rel_hex < 0.25,
        'error_rel': float(error_rel_hex),
        'error_abs': float(error_hex)
    }

    # 2. Ring analysis
    hist, bin_edges = np.histogram(distancias, bins=15)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

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

    anillos.sort(key=lambda x: x['radio_arcmin'])
    resultados['anillos'] = anillos
    resultados['n_anillos'] = len(anillos)

    # Relation between rings
    if len(anillos) >= 2:
        radios = [a['radio_arcmin'] for a in anillos]
        ratios = [radios[i+1]/radios[i] for i in range(len(radios)-1) if radios[i] > 0]
        resultados['ratios_radiales'] = [float(r) for r in ratios]

        # Search for ratio close to √3
        ratios_cercanos = [r for r in ratios if abs(r - 1.732) < 0.4]
        if ratios_cercanos:
            mejor_ratio = min(ratios, key=lambda x: abs(x-1.732))
            resultados['mejor_ratio_hexagonal'] = float(mejor_ratio)
            resultados['error_ratio_hexagonal'] = float(abs(mejor_ratio - 1.732))
            resultados['tiene_ratio_hexagonal'] = True
        else:
            resultados['tiene_ratio_hexagonal'] = False

    # 3. PCA for orientation
    pca = PCA(n_components=2)
    pca.fit(coords_rel)
    angulo_pca = np.degrees(np.arctan2(pca.components_[0,1], pca.components_[0,0])) % 180

    resultados['pca'] = {
        'angulo_principal': float(angulo_pca),
        'varianza_explicada': float(pca.explained_variance_ratio_[0])
    }

    # 4. Calculate hexagonal score
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

    # Component 3: Multiple rings
    if len(anillos) >= 2:
        score_componentes.append(0.4)
    elif len(anillos) >= 1:
        score_componentes.append(0.2)

    # Component 4: Sample size
    if len(galaxias) > 200:
        score_componentes.append(0.3)
    elif len(galaxias) > 50:
        score_componentes.append(0.15)

    # Final score
    resultados['score_hexagonal'] = float(np.mean(score_componentes)) if score_componentes else 0.0

    # Classification
    if resultados['score_hexagonal'] > 0.6:
        resultados['clasificacion_hexagonal'] = 'STRONG'
    elif resultados['score_hexagonal'] > 0.4:
        resultados['clasificacion_hexagonal'] = 'MODERATE'
    elif resultados['score_hexagonal'] > 0.2:
        resultados['clasificacion_hexagonal'] = 'WEAK'
    else:
        resultados['clasificacion_hexagonal'] = 'NULL'

    return resultados

# ============================================================================
# 4. MULTI-NODE COHERENCE ANALYSIS
# ============================================================================
def analizar_coherencia_multinodo(resultados_nodos):
    """Analyzes coherence between multiple nodes"""
    print(f"\n" + "="*80)
    print("📊 COHERENCE ANALYSIS BETWEEN {len(resultados_nodos)} NODES")
    print("="*80)

    n_nodos = len(resultados_nodos)

    # Coherence matrix
    coherencias = np.zeros((n_nodos, n_nodos))
    diferencias_angulo = np.zeros((n_nodos, n_nodos))
    diferencias_z = np.zeros((n_nodos, n_nodos))

    # Node information for statistics
    nodos_info = []

    for i, nodo_i in enumerate(resultados_nodos):
        nodo_info = {
            'nombre': nodo_i['nombre'],
            'z_centro': nodo_i.get('z_centro', 0),
            'score_hex': nodo_i['score_hexagonal'],
            'angulo_pca': nodo_i['pca']['angulo_principal'],
            'n_galaxias': nodo_i['n_galaxias'],
            'radio_medio': nodo_i['radio_medio'],
            'hex_detectada': nodo_i['hexagonal']['detectada']
        }
        nodos_info.append(nodo_info)

        for j, nodo_j in enumerate(resultados_nodos):
            if i == j:
                continue

            # PCA angle difference
            ang_i = nodo_i['pca']['angulo_principal']
            ang_j = nodo_j['pca']['angulo_principal']
            diff_ang = abs(ang_i - ang_j)
            if diff_ang > 90:  # Circular correction
                diff_ang = 180 - diff_ang
            diferencias_angulo[i,j] = diff_ang

            # Redshift difference
            z_i = nodo_i.get('z_centro', 0)
            z_j = nodo_j.get('z_centro', 0)
            diferencias_z[i,j] = abs(z_i - z_j)

            # Coherence score between nodes i and j
            coherencia = 0.0

            # 1. Similar orientation (< 30°)
            if diff_ang < 15:
                coherencia += 0.3
            elif diff_ang < 30:
                coherencia += 0.2
            elif diff_ang < 45:
                coherencia += 0.1

            # 2. Both have hexagonal symmetry
            if nodo_i['hexagonal']['detectada'] and nodo_j['hexagonal']['detectada']:
                coherencia += 0.25

            # 3. Similar hexagonal scores (< 0.3 difference)
            diff_score = abs(nodo_i['score_hexagonal'] - nodo_j['score_hexagonal'])
            if diff_score < 0.2:
                coherencia += 0.2
            elif diff_score < 0.4:
                coherencia += 0.1

            # 4. Similar radii (ratio between 0.8 and 1.2)
            ratio_radio = nodo_j['radio_medio'] / nodo_i['radio_medio'] if nodo_i['radio_medio'] > 0 else 0
            if 0.8 <= ratio_radio <= 1.2:
                coherencia += 0.15

            # 5. Redshift periodicity (multiples of ~0.43)
            delta_z = abs(z_i - z_j)
            if abs(delta_z % 0.43) < 0.1:  # Close to multiple of 0.43
                coherencia += 0.1

            coherencias[i,j] = coherencia

    # Pattern analysis
    print(f"\n📐 ORIENTATION ANALYSIS (PCA):")
    for i, info in enumerate(nodos_info):
        print(f"  • {info['nombre']} (z={info['z_centro']:.2f}): {info['angulo_pca']:.1f}°")

    print(f"\n🎯 HEXAGONAL SYMMETRY ANALYSIS:")
    hex_por_nodo = [info['hex_detectada'] for info in nodos_info]
    nodos_con_hex = sum(hex_por_nodo)
    print(f"  • Nodes with hexagonal symmetry: {nodos_con_hex}/{n_nodos} ({nodos_con_hex/n_nodos*100:.1f}%)")

    print(f"\n📊 HEXAGONAL SCORE STATISTICS:")
    scores_hex = [info['score_hex'] for info in nodos_info]
    print(f"  • Mean: {np.mean(scores_hex):.3f} ± {np.std(scores_hex):.3f}")
    print(f"  • Range: {min(scores_hex):.3f} - {max(scores_hex):.3f}")

    # Global coherence
    coherencia_promedio = np.mean(coherencias[coherencias > 0])

    print(f"\n" + "="*80)
    print("🏆 GLOBAL COHERENCE BETWEEN NODES")
    print("="*80)
    print(f"Average coherence: {coherencia_promedio:.3f}")

    # Summarized coherence matrix
    print(f"\n📈 COHERENCE MATRIX (values > 0.5):")
    for i in range(n_nodos):
        for j in range(i+1, n_nodos):
            if coherencias[i,j] > 0.5:
                print(f"  • {resultados_nodos[i]['nombre']} ↔ {resultados_nodos[j]['nombre']}: {coherencias[i,j]:.3f}")

    # Analysis results
    resultados_globales = {
        'coherencia_promedio': float(coherencia_promedio),
        'matriz_coherencias': coherencias.tolist(),
        'diferencias_angulo': diferencias_angulo.tolist(),
        'diferencias_z': diferencias_z.tolist(),
        'estadisticas': {
            'media_score_hex': float(np.mean(scores_hex)),
            'std_score_hex': float(np.std(scores_hex)),
            'porcentaje_nodos_con_hex': float(nodos_con_hex/n_nodos*100),
            'media_radio': float(np.mean([info['radio_medio'] for info in nodos_info])),
            'media_galaxias': float(np.mean([info['n_galaxias'] for info in nodos_info]))
        },
        'nodos_info': nodos_info
    }

    # Interpretation
    print(f"\n" + "="*80)
    print("📈 GLOBAL INTERPRETATION")
    print("="*80)

    if coherencia_promedio > 0.7:
        print("🎉 STRONG EVIDENCE OF 3D CRYSTAL NETWORK!")
        print(f"{nodos_con_hex}/{n_nodos} nodes show coherent hexagonal symmetry")
        print("Orientation and patterns remain consistent")
        print("IMPLICATION: Large-scale crystal structure confirmed")

    elif coherencia_promedio > 0.5:
        print("✅ MODERATE EVIDENCE OF 3D COHERENCE")
        print(f"{nodos_con_hex}/{n_nodos} nodes have hexagonal symmetry")
        print("Similar patterns but with some variations")
        print("IMPLICATION: Partially ordered structure")

    elif coherencia_promedio > 0.3:
        print("⚠️  HINTS OF 3D COHERENCE")
        print("Some common patterns but not conclusive")
        print("More data required for confirmation")

    else:
        print("❌ WEAK COHERENCE BETWEEN NODES")
        print("Nodes appear as independent structures")
        print("Possibly dominant local effects")

    return resultados_globales

# ============================================================================
# 5. RESULTS VISUALIZATION
# ============================================================================
def generar_resumen_multinodo(resultados_nodos, resultados_globales):
    """Generates complete summary of multi-node analysis"""
    try:
        # Complete data
        datos_completos = {
            'nodos_analizados': len(resultados_nodos),
            'resultados_individuales': resultados_nodos,
            'analisis_global': resultados_globales,
            'metadata': {
                'fecha_analisis': np.datetime64('now').astype(str),
                'instrumento': 'JWST/PRIMER-COSMOS',
                'version_analisis': '2.0-multinodo',
                'ancho_z': ANCHO_Z
            }
        }

        # Save JSON
        with open('analisis_multinodo_cristalino.json', 'w') as f:
            json.dump(datos_completos, f, indent=2, default=str)

        print(f"💾 Results saved in 'analisis_multinodo_cristalino.json'")

        # Executive summary
        with open('resumen_multinodo_cristalino.txt', 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SUMMARY - MULTI-NODE CRYSTAL STRUCTURE ANALYSIS\n")
            f.write("="*80 + "\n\n")

            f.write("📊 NODES ANALYZED:\n")
            for nodo in resultados_nodos:
                f.write(f"• {nodo['nombre']} (z={nodo.get('z_centro', 0):.2f}):\n")
                f.write(f"  - Galaxies: {nodo['n_galaxias']}\n")
                f.write(f"  - Hexagonal score: {nodo['score_hexagonal']:.3f} ({nodo['clasificacion_hexagonal']})\n")
                f.write(f"  - Hexagonal symmetry: {'✓' if nodo['hexagonal']['detectada'] else '✗'}\n")
                f.write(f"  - Rings: {nodo['n_anillos']}\n")
                f.write(f"  - Mean radius: {nodo['radio_medio']:.1f}'\n")
                f.write(f"  - PCA angle: {nodo['pca']['angulo_principal']:.1f}°\n\n")

            f.write("🏆 GLOBAL STATISTICS:\n")
            stats = resultados_globales['estadisticas']
            f.write(f"• Average coherence: {resultados_globales['coherencia_promedio']:.3f}\n")
            f.write(f"• Mean hexagonal score: {stats['media_score_hex']:.3f} ± {stats['std_score_hex']:.3f}\n")
            f.write(f"• Nodes with hexagonal symmetry: {stats['porcentaje_nodos_con_hex']:.1f}%\n")
            f.write(f"• Mean radius: {stats['media_radio']:.1f}'\n")
            f.write(f"• Galaxies per node (average): {stats['media_galaxias']:.0f}\n\n")

            # Interpretation
            coherencia = resultados_globales['coherencia_promedio']
            if coherencia > 0.7:
                f.write("🎉 INTERPRETATION: STRONG EVIDENCE OF 3D CRYSTAL NETWORK\n")
                f.write("Nodes show coherent patterns suggesting an underlying\n")
                f.write("periodic structure in the early universe.\n")
            elif coherencia > 0.5:
                f.write("✅ INTERPRETATION: MODERATE EVIDENCE OF 3D COHERENCE\n")
                f.write("Significant similarities observed between adjacent nodes.\n")
            elif coherencia > 0.3:
                f.write("⚠️  INTERPRETATION: HINTS OF 3D COHERENCE\n")
                f.write("Some common patterns require additional confirmation.\n")
            else:
                f.write("❌ INTERPRETATION: WEAK COHERENCE BETWEEN NODES\n")
                f.write("Nodes appear as independent structures without clear patterns.\n")

            f.write("\n" + "="*80 + "\n")
            f.write("© Scientific Analysis - Cosmic Crystal Structure\n")
            f.write("="*80 + "\n")

        print("📝 Summary saved in 'resumen_multinodo_cristalino.txt'")

    except Exception as e:
        print(f"❌ Error generating summary: {e}")

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*80)
    print("MULTI-NODE CROSS-VALIDATION - COSMIC CRYSTAL STRUCTURE")
    print("="*80)
    print(f"Analyzing {len(NODOS_PRIORITARIOS)} priority nodes")
    print("="*80)

    resultados_todos = []

    # Analyze each node
    for i, (z_centro, nombre, peso) in enumerate(NODOS_PRIORITARIOS):
        print(f"\n{'='*60}")
        print(f"NODE {i+1}/{len(NODOS_PRIORITARIOS)}: {nombre} (z={z_centro:.2f})")
        print(f"{'='*60}")

        # Extract galaxies
        print(f"🔍 Extracting galaxies...")
        galaxias = extraer_galaxias_nodo(z_centro, ANCHO_Z)

        if len(galaxias) < 30:
            print(f"❌ Insufficient sample ({len(galaxias)} galaxies)")
            continue

        print(f"✅ {len(galaxias)} galaxies extracted")

        # Analyze
        print(f"🎯 Analyzing hexagonal structure...")
        resultados = analizar_hexagonal_estandarizado(galaxias, nombre)

        if resultados:
            resultados['z_centro'] = float(z_centro)
            resultados['peso_analisis'] = float(peso)
            resultados_todos.append(resultados)

            # Show quick summary
            print(f"📊 Results:")
            print(f"  • Hexagonal symmetry: {'✓' if resultados['hexagonal']['detectada'] else '✗'}")
            print(f"  • Hexagonal score: {resultados['score_hexagonal']:.3f} ({resultados['clasificacion_hexagonal']})")
            print(f"  • Rings: {resultados['n_anillos']}")
            print(f"  • Mean radius: {resultados['radio_medio']:.1f}'")
            print(f"  • PCA angle: {resultados['pca']['angulo_principal']:.1f}°")

            if resultados.get('tiene_ratio_hexagonal', False):
                print(f"  • Hexagonal ratio: {resultados['mejor_ratio_hexagonal']:.3f} (error: {resultados['error_ratio_hexagonal']:.3f})")
        else:
            print(f"❌ Error in analysis of {nombre}")

    if len(resultados_todos) < 2:
        print(f"\n❌ At least 2 nodes needed for comparative analysis")
        print(f"   Nodes analyzed: {len(resultados_todos)}")
        return

    # Global coherence analysis
    resultados_globales = analizar_coherencia_multinodo(resultados_todos)

    # Generate summary
    generar_resumen_multinodo(resultados_todos, resultados_globales)

    # Final message
    coherencia = resultados_globales['coherencia_promedio']
    print(f"\n" + "="*80)
    print("MULTI-NODE ANALYSIS COMPLETED")
    print("="*80)

    if coherencia > 0.7:
        print("✨ 3D CRYSTAL NETWORK CONFIRMED! ✨")
        print(f"Coherence between {len(resultados_todos)} nodes: {coherencia:.3f}")
    elif coherencia > 0.5:
        print("✅ Analysis successful - Solid evidence of 3D structure")
        print(f"Average coherence: {coherencia:.3f}")
    else:
        print("⚠️  Analysis completed - Inconclusive results")
        print(f"Low coherence: {coherencia:.3f}")

# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    main()