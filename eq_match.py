import pandas as pd
import numpy as np

def _read_spectrum(path):
    """Lee el txt de spectrum exportado por Audacity y devuelve df con columnas 'freq' y 'level'."""
    # Intentar con tabulación, si falla usar whitespace
    try:
        df = pd.read_csv(path, sep='\t', engine='python')
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, engine='python')

    # Normalizar nombres de columna buscando keywords
    cols = df.columns.tolist()
    freq_col = next((c for c in cols if 'freq' in c.lower() or 'frequency' in c.lower()), None)
    level_col = next((c for c in cols if 'lev' in c.lower() or 'db' in c.lower()), None)

    if freq_col is None or level_col is None:
        # fallback a las dos primeras columnas
        freq_col = cols[0]
        level_col = cols[1] if len(cols) > 1 else cols[0]

    df2 = df[[freq_col, level_col]].copy()
    df2.columns = ['freq', 'level']
    df2 = df2.dropna().sort_values('freq').reset_index(drop=True)
    return df2

def generar_filtercurve(archivo_ref, archivo_edit, archivo_salida,
                         max_gain=None, n_points=None, smooth_window=1, max_points=10000):
    """
    Genera archivo FilterCurve para Audacity.
    - archivo_ref: spectrum (referencia)
    - archivo_edit: spectrum (audio a editar)
    - archivo_salida: output .txt
    - max_gain: límite en dB (None = sin límite). Ej: 9
    - n_points: número de puntos en la curva (None = usa la resolución completa / unión de frecuencias)
    - smooth_window: ventana para suavizado (1 = sin suavizado, 5 = promedio 5 puntos)
    - max_points: si n_points is None, limita la cantidad máxima de puntos para evitar archivos gigantes
    """

    # Leer espectros
    df_ref = _read_spectrum(archivo_ref)
    df_edit = _read_spectrum(archivo_edit)

    # Rangos mínimos/máximos de frecuencia para la interpolación
    fmin = max(min(df_ref['freq'].min(), df_edit['freq'].min()), 1.0)  # evitar 0
    fmax = max(df_ref['freq'].max(), df_edit['freq'].max())

    # Determinar frecuencias objetivo (target)
    if n_points is None:
        # usar la unión ordenada de puntos de ambos archivos
        freqs = np.union1d(df_ref['freq'].values, df_edit['freq'].values)
        # si quedan demasiados puntos, muestrear logaritmicamente hasta max_points
        if len(freqs) > max_points:
            freqs = np.logspace(np.log10(fmin), np.log10(fmax), max_points)
    else:
        # distribuir logarítmicamente (mejor para audio)
        freqs = np.logspace(np.log10(fmin), np.log10(fmax), int(n_points))

    # Asegurar que los arrays para interpolación están ordenados y sin duplicados
    ref_x = df_ref['freq'].values
    ref_y = df_ref['level'].values
    edit_x = df_edit['freq'].values
    edit_y = df_edit['level'].values

    # Si hay duplicados en las frecuencias, promediar para seguridad
    def _uniq_avg(x, y):
        df = pd.DataFrame({'x': x, 'y': y})
        df = df.groupby('x', as_index=False).mean().sort_values('x')
        return df['x'].values, df['y'].values

    ref_x, ref_y = _uniq_avg(ref_x, ref_y)
    edit_x, edit_y = _uniq_avg(edit_x, edit_y)

    # Interpolar niveles en las frecuencias objetivo
    ref_interp = np.interp(freqs, ref_x, ref_y,
                           left=ref_y[0], right=ref_y[-1])
    edit_interp = np.interp(freqs, edit_x, edit_y,
                            left=edit_y[0], right=edit_y[-1])

    # Ganancia necesaria (referencia - edit)
    gains = ref_interp - edit_interp

    # Suavizado opcional (media móvil)
    if smooth_window and smooth_window > 1:
        kernel = np.ones(int(smooth_window)) / float(smooth_window)
        gains = np.convolve(gains, kernel, mode='same')

    # Recortar ganancia si se pidió
    if max_gain is not None:
        gains = np.clip(gains, -abs(max_gain), abs(max_gain))

    # Reemplazar NaN/Infinito por 0
    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0)

    # Construir texto FilterCurve (f0.., v0..)
    freqs_str = " ".join([f'f{i}="{float(freq):.6f}"' for i, freq in enumerate(freqs)])
    gains_str = " ".join([f'v{i}="{float(g):.6f}"' for i, g in enumerate(gains)])

    header = 'FilterCurve:'
    footer = ' FilterLength="8191" InterpolateLin="0" InterpolationMethod="B-spline"'
    contenido = f"{header} {freqs_str} {gains_str}{footer}"

    # Guardar archivo
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.write(contenido)

    print("Archivo generado:", archivo_salida)
    print(f"  puntos: {len(freqs)}, rango: {freqs[0]:.1f} Hz - {freqs[-1]:.1f} Hz")
    if max_gain is None:
        print("  ganancia: sin recorte (max_gain=None)")
    else:
        print(f"  ganancia: recortada a ±{max_gain} dB")

# ---------- EJEMPLOS DE USO ----------
# 1) Modo FULL (usa todas las frecuencias de tus archivos; si son MUCHAS, se limitará a max_points)
generar_filtercurve("spectrum1.txt", "spectrum2.txt", "FilterCurve_full.txt",
                     max_gain=None, n_points=None, smooth_window=1, max_points=10000)

# 2) Modo controlado: 500 puntos log-distribuidos, suavizado 5, límite ±12 dB
generar_filtercurve("spectrum1.txt", "spectrum2.txt", "FilterCurve_500pts.txt",
                     max_gain=12, n_points=500, smooth_window=5)

# 3) Modo rápido: 60 puntos (para Audacity si da problemas con demasiados puntos)
generar_filtercurve("spectrum1.txt", "spectrum2.txt", "FilterCurve_200.txt",
                     max_gain=9, n_points=200, smooth_window=3)
