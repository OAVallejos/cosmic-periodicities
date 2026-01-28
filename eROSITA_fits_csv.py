#!/usr/bin/env python3
"""
CONVERSOR SDSS FITS → CSV
Extrae solo columnas esenciales de archivos FITS grandes
"""

from astropy.io import fits
import pandas as pd
import numpy as np
import os
import sys

def convertir_fits_a_csv(fits_file, output_csv=None, max_objects=None):
    """
    Convierte archivo FITS SDSS a CSV con columnas esenciales
    
    Parameters:
    -----------
    fits_file : str
        Ruta al archivo .fits
    output_csv : str, optional
        Nombre del archivo CSV de salida
    max_objects : int, optional
        Límite máximo de objetos a extraer (para archivos grandes)
    """
    
    print(f"📂 Abriendo: {fits_file}")
    print(f"   Tamaño: {os.path.getsize(fits_file)/1e9:.1f} GB")
    
    try:
        # 1. Abrir archivo FITS
        with fits.open(fits_file) as hdul:
            print(f"✅ Archivo abierto correctamente")
            print(f"   Número de extensiones HDU: {len(hdul)}")
            
            # 2. Encontrar la extensión con datos
            data_hdu = None
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'data') and hdu.data is not None:
                    print(f"   HDU {i}: {hdu.name} - Forma: {hdu.data.shape}")
                    if len(hdu.data.shape) > 0 and hdu.data.shape[0] > 0:
                        data_hdu = hdu
                        print(f"   ✅ Usando HDU {i} para extracción")
                        break
            
            if data_hdu is None:
                print("❌ No se encontraron datos en el archivo FITS")
                return False
            
            # 3. Ver columnas disponibles
            if hasattr(data_hdu, 'columns'):
                columns = data_hdu.columns.names
                print(f"\n📊 Columnas disponibles ({len(columns)}):")
                
                # Mostrar columnas relevantes agrupadas
                grupos = {
                    'Redshifts': [c for c in columns if 'z' in c.lower() and 'err' not in c.lower()],
                    'Posiciones': [c for c in columns if any(k in c.lower() for k in ['ra', 'dec', 'glon', 'glat'])],
                    'Magnitudes': [c for c in columns if 'mag' in c.lower()],
                    'Clasificaciones': [c for c in columns if any(k in c.lower() for k in ['class', 'type', 'objtype'])],
                    'Masa/Edad': [c for c in columns if any(k in c.lower() for k in ['mass', 'age', 'metal'])],
                }
                
                for grupo, cols in grupos.items():
                    if cols:
                        print(f"   • {grupo}: {', '.join(cols[:3])}" + 
                              ("..." if len(cols) > 3 else ""))
                
                # 4. Seleccionar columnas esenciales para ω=0.191
                columnas_esenciales = []
                
                # Buscar redshift (prioridad: sdss_z, z, redshift)
                z_candidates = ['sdss_z', 'z', 'redshift', 'zspec']
                for candidate in z_candidates:
                    if candidate in columns:
                        columnas_esenciales.append(candidate)
                        print(f"   🔍 Usando '{candidate}' como redshift")
                        break
                
                # Buscar error de redshift
                zerr_candidates = [f"{col}_err" for col in z_candidates] + ['sdss_z_err', 'zerr']
                for candidate in zerr_candidates:
                    if candidate in columns:
                        columnas_esenciales.append(candidate)
                        break
                
                # Buscar RA/DEC
                for ra in ['ra', 'sdss_fiber_ra', 'racat']:
                    if ra in columns:
                        columnas_esenciales.append(ra)
                        break
                
                for dec in ['dec', 'sdss_fiber_dec', 'deccat']:
                    if dec in columns:
                        columnas_esenciales.append(dec)
                        break
                
                # Buscar magnitud (preferir K-band)
                mag_priority = [
                    'twomass_mag',  # Array [J, H, K]
                    'wise_mag',     # Array [W1, W2, W3, W4]
                    'mag_K', 'Kmag', 'kmag',
                    'mag_r', 'rmag', 'sdss_r'
                ]
                
                for mag in mag_priority:
                    if mag in columns:
                        columnas_esenciales.append(mag)
                        print(f"   🔍 Usando '{mag}' como magnitud")
                        break
                
                # Buscar clasificación
                for cls in ['sdss_class', 'class', 'objtype']:
                    if cls in columns:
                        columnas_esenciales.append(cls)
                        break
                
                print(f"\n🎯 Columnas seleccionadas: {columnas_esenciales}")
                
                # 5. Extraer datos
                print(f"\n🔄 Extrayendo datos...")
                
                # Limitar número de objetos si se especifica
                data = data_hdu.data
                n_total = len(data)
                
                if max_objects and max_objects < n_total:
                    print(f"   Limitando a {max_objects:,} de {n_total:,} objetos totales")
                    indices = np.random.choice(n_total, max_objects, replace=False)
                    data_subset = data[indices]
                else:
                    data_subset = data
                    print(f"   Extrayendo todos los {n_total:,} objetos")
                
                # 6. Crear DataFrame
                df_data = {}
                
                for col in columnas_esenciales:
                    try:
                        # Extraer columna
                        col_data = data_subset[col]
                        
                        # Manejar arrays 2D (como twomass_mag que es [n, 3])
                        if len(col_data.shape) == 2:
                            if 'twomass' in col.lower():
                                # Para 2MASS, extraer solo K-band (índice 2)
                                df_data['mag_K'] = col_data[:, 2] if col_data.shape[1] >= 3 else col_data[:, 0]
                                print(f"   ✅ Extraída K-band de '{col}'")
                            elif 'wise' in col.lower():
                                # Para WISE, extraer W1 (índice 0)
                                df_data['wise_W1'] = col_data[:, 0] if col_data.shape[1] >= 1 else col_data[:, 0]
                                print(f"   ✅ Extraída W1-band de '{col}'")
                            else:
                                # Para otros arrays, tomar primera columna
                                df_data[col] = col_data[:, 0]
                        else:
                            df_data[col] = col_data
                            
                    except Exception as e:
                        print(f"   ⚠️  Error extrayendo columna '{col}': {e}")
                
                # 7. Crear DataFrame de pandas
                df = pd.DataFrame(df_data)
                
                # 8. Limpiar datos de texto (convertir bytes a string)
                for col in df.columns:
                    if df[col].dtype == object:
                        # Intentar convertir bytes a string
                        try:
                            df[col] = df[col].apply(
                                lambda x: x.decode('utf-8', errors='ignore').strip() 
                                if isinstance(x, bytes) else str(x).strip()
                            )
                        except:
                            pass
                
                # 9. Renombrar columnas para consistencia
                rename_map = {}
                if 'sdss_z' in df.columns:
                    rename_map['sdss_z'] = 'redshift'
                elif 'z' in df.columns:
                    rename_map['z'] = 'redshift'
                
                if 'sdss_z_err' in df.columns:
                    rename_map['sdss_z_err'] = 'z_err'
                
                if 'sdss_fiber_ra' in df.columns:
                    rename_map['sdss_fiber_ra'] = 'ra'
                
                if 'sdss_fiber_dec' in df.columns:
                    rename_map['sdss_fiber_dec'] = 'dec'
                
                if 'sdss_class' in df.columns:
                    rename_map['sdss_class'] = 'class'
                
                df = df.rename(columns=rename_map)
                
                # 10. Generar nombre de archivo de salida
                if output_csv is None:
                    base_name = os.path.splitext(os.path.basename(fits_file))[0]
                    output_csv = f"{base_name}_extracted.csv"
                
                # 11. Guardar CSV
                df.to_csv(output_csv, index=False)
                
                print(f"\n✅ CONVERSIÓN COMPLETADA")
                print(f"💾 Guardado en: {output_csv}")
                print(f"📊 {len(df):,} objetos, {len(df.columns)} columnas")
                
                # 12. Mostrar estadísticas básicas
                print(f"\n📈 ESTADÍSTICAS BÁSICAS:")
                
                if 'redshift' in df.columns:
                    z_vals = pd.to_numeric(df['redshift'], errors='coerce').dropna()
                    if len(z_vals) > 0:
                        print(f"   • Redshift: [{z_vals.min():.3f}, {z_vals.max():.3f}]")
                        print(f"   • Mediana: {z_vals.median():.3f}, Media: {z_vals.mean():.3f}")
                
                if 'mag_K' in df.columns:
                    mag_vals = pd.to_numeric(df['mag_K'], errors='coerce').dropna()
                    if len(mag_vals) > 0:
                        print(f"   • Mag_K: [{mag_vals.min():.2f}, {mag_vals.max():.2f}]")
                        print(f"   • Mediana: {mag_vals.median():.2f}")
                
                if 'class' in df.columns:
                    print(f"   • Clases únicas: {df['class'].unique()[:5]}")
                
                return True
            else:
                print("❌ El HDU no tiene estructura de columnas")
                return False
    
    except Exception as e:
        print(f"❌ Error procesando archivo FITS: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("="*60)
    print("CONVERSOR FITS → CSV - SDSS/eROSITA")
    print("="*60)
    
    # Verificar argumentos
    if len(sys.argv) < 2:
        print("Uso: python3 fits_to_csv.py <archivo.fits> [output.csv] [max_objects]")
        print("\nEjemplos:")
        print("  python3 fits_to_csv.py data/DL1_spec_SDSSV_eROSITA_eRASS1-v1_0_2.fits")
        print("  python3 fits_to_csv.py big_file.fits extracted.csv 100000")
        print("  python3 fits_to_csv.py input.fits --max 50000")
        return
    
    # Parsear argumentos
    fits_file = sys.argv[1]
    output_csv = None
    max_objects = None
    
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '--max' and i+1 < len(sys.argv):
            try:
                max_objects = int(sys.argv[i+1])
            except:
                print(f"⚠️  Error parseando número máximo: {sys.argv[i+1]}")
        elif sys.argv[i].endswith('.csv'):
            output_csv = sys.argv[i]
    
    # Si no se especificó output, deducirlo del nombre
    if output_csv is None:
        base = os.path.splitext(os.path.basename(fits_file))[0]
        output_csv = f"{base}_extracted.csv"
    
    # Verificar que el archivo existe
    if not os.path.exists(fits_file):
        print(f"❌ Archivo no encontrado: {fits_file}")
        
        # Buscar en subdirectorios comunes
        posibles_rutas = [
            fits_file,
            os.path.join('data', fits_file),
            os.path.join('..', fits_file),
            os.path.basename(fits_file)
        ]
        
        for ruta in posibles_rutas:
            if os.path.exists(ruta):
                fits_file = ruta
                print(f"✅ Encontrado en: {fits_file}")
                break
        
        if not os.path.exists(fits_file):
            print("💡 Archivos SDSS disponibles para descargar:")
            print("   wget https://data.sdss.org/sas/dr19/vac/mos/DL1_SDSS_eROSITA/v1_0_2/DL1_spec_SDSSV_eROSITA_eRASS1-v1_0_2.fits")
            print("   wget https://data.sdss.org/sas/dr17/eboss/galaxy/dr17_final/galaxy_DR17_final.fits")
            return
    
    # Ejecutar conversión
    convertir_fits_a_csv(
        fits_file=fits_file,
        output_csv=output_csv,
        max_objects=max_objects
    )

if __name__ == "__main__":
    main()