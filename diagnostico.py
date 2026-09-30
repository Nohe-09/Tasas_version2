# Página de diagnóstico del despliegue. Úsala temporalmente como archivo principal de la app.
import os
import platform
import importlib.metadata as md
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Diagnóstico del despliegue", page_icon="🔧")
st.title("🔧 Diagnóstico del despliegue")
st.write("Si ves este texto, Streamlit funciona en tu servidor. Revisa abajo qué falla.")

BASE = Path(__file__).resolve().parent
st.write(f"**Carpeta de ejecución:** `{BASE}`")
st.write(f"**Python:** {platform.python_version()}")

st.subheader("1. Archivos que hay junto a este script")
st.code("\n".join(sorted(os.listdir(BASE))))
esperados = ["app.py", "recomendador.py", "modelo_recomendador.joblib", "referencia_tasas.csv",
             "mediana_segmento.csv", "presencia_entidad_municipio.csv", "municipios.csv",
             "tipos_tasa.csv", "requirements.txt"]
faltan = [f for f in esperados if not (BASE / f).exists()]
if faltan:
    st.error("Faltan estos archivos: " + ", ".join(faltan))
else:
    st.success("Están los 9 archivos esperados.")
    tam = (BASE / "modelo_recomendador.joblib").stat().st_size / 1024
    st.write(f"Tamaño de `modelo_recomendador.joblib`: {tam:,.0f} KB (debe rondar 5.300 KB)")

st.subheader("2. Versiones instaladas")
for paquete in ["streamlit", "scikit-learn", "pandas", "numpy", "joblib"]:
    try:
        st.write(f"- {paquete}: {md.version(paquete)}")
    except Exception as e:
        st.write(f"- {paquete}: no instalado ({e})")

st.subheader("3. Carga del modelo y una predicción de prueba")
try:
    import sys
    sys.path.insert(0, str(BASE))
    import joblib
    import pandas as pd
    from recomendador import recomendar
    bundle = joblib.load(BASE / "modelo_recomendador.joblib")
    st.success(f"Modelo cargado. Entrenado con scikit-learn {bundle.get('version_sklearn')} "
               f"y Python {bundle.get('version_python')}.")
    ref = pd.read_csv(BASE / "referencia_tasas.csv", dtype={"nivel_monto": str})
    med = pd.read_csv(BASE / "mediana_segmento.csv", dtype={"nivel_monto": str})
    pres = pd.read_csv(BASE / "presencia_entidad_municipio.csv", dtype={"municipio": str})
    r = recomendar(bundle, ref, med, pres, bundle["categorias"]["producto"][0],
                   bundle["categorias"]["plazo"][0], bundle["categorias"]["tipo_tasa"][0], "11001", 150_000_000)
    st.success(f"Predicción de prueba correcta: {r}")
except Exception as e:
    st.error("Falló la carga o la predicción:")
    st.exception(e)
