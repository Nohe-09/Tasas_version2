# Aplicación Streamlit: recomendador de entidad financiera para crédito de vivienda
import platform
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

# Las rutas se calculan desde la carpeta donde está app.py, sin importar desde dónde se ejecute
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from recomendador import recomendar, municipios_con_cobertura

st.set_page_config(page_title="Recomendador de entidad para crédito de vivienda",
                   page_icon="🏠", layout="centered")


@st.cache_resource
def cargar_recursos():
    bundle = joblib.load(BASE / "modelo_recomendador.joblib")
    referencia = pd.read_csv(BASE / "referencia_tasas.csv", dtype={"nivel_monto": str})
    mediana = pd.read_csv(BASE / "mediana_segmento.csv", dtype={"nivel_monto": str})
    presencia = pd.read_csv(BASE / "presencia_entidad_municipio.csv", dtype={"municipio": str})
    municipios = pd.read_csv(BASE / "municipios.csv", dtype={"municipio": str})
    tipos_tasa = pd.read_csv(BASE / "tipos_tasa.csv")
    return bundle, referencia, mediana, presencia, municipios, tipos_tasa


try:
    bundle, referencia, mediana, presencia, municipios, tipos_tasa = cargar_recursos()
except Exception as e:  # muestra la causa en pantalla en lugar de una página en blanco
    st.error("No se pudo cargar el modelo o las tablas de apoyo. Revisa que todos los archivos del "
             "repositorio estén junto a app.py y que requirements.txt use la versión de scikit-learn "
             "con la que se entrenó el modelo.")
    st.code(f"{type(e).__name__}: {e}\nscikit-learn instalado: {sklearn.__version__}")
    st.stop()

if bundle.get("version_sklearn") and bundle["version_sklearn"] != sklearn.__version__:
    st.warning(f"El modelo se entrenó con scikit-learn {bundle['version_sklearn']} y este entorno tiene "
               f"{sklearn.__version__}. Si hay resultados extraños, iguala las versiones en requirements.txt.")

st.title("🏠 Recomendador de entidad para crédito de vivienda")
st.write("Indica el perfil del crédito y la aplicación sugiere la entidad que con más frecuencia ofrece "
         "tasas entre las más bajas del mercado para ese perfil en tu ciudad. Se basa en los desembolsos de "
         "bancos comerciales y compañías de financiamiento entre julio y septiembre de 2026.")

# ---- Entradas ----
cobertura = municipios_con_cobertura(presencia)
nombres = municipios.set_index("municipio")["nombre"].to_dict()
opciones = {f"{nombres.get(c, c)} ({c})": c for c in cobertura}
etiquetas = sorted(opciones.keys())
por_defecto = next((i for i, e in enumerate(etiquetas) if e.startswith("Bogota D.C.")), 0)

with st.sidebar:
    st.header("Perfil del crédito")
    ciudad_txt = st.selectbox("Ciudad", etiquetas, index=por_defecto)
    producto = st.selectbox("Producto de crédito", bundle["categorias"]["producto"])
    plazo = st.selectbox("Plazo", bundle["categorias"]["plazo"])
    tipo_tasa = st.selectbox("Tipo de tasa", bundle["categorias"]["tipo_tasa"])
    monto = st.number_input("Monto del crédito (COP)", min_value=10_000_000,
                            max_value=2_000_000_000, value=150_000_000, step=5_000_000)
    with st.expander("¿Qué significa cada tipo de tasa?"):
        for _, fila in tipos_tasa.iterrows():
            st.markdown(f"**{fila['tipo_tasa']}**: {fila['descripcion']}")

municipio = opciones[ciudad_txt]
res = recomendar(bundle, referencia, mediana, presencia, producto, plazo, tipo_tasa, municipio, monto)

# ---- Resultado ----
def fmt(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


st.subheader("Entidad recomendada")
if not res["ok"]:
    st.warning(res["mensaje"])
else:
    st.success(f"**{res['entidad']}**")
    c1, c2 = st.columns(2)
    if res["tasa_ref"] is not None:
        delta = None
        if res["diferencia"] is not None:
            delta = f"{res['diferencia']:+.2f}".replace(".", ",") + " puntos frente a la mediana"
        c1.metric("Tasa promedio observada", f"{fmt(res['tasa_ref'])}% E.A.", delta=delta, delta_color="inverse")
    else:
        c1.metric("Tasa promedio observada", "Sin datos")
    if res["pct_comp"] is not None:
        c2.metric("Desembolsos con tasa baja", f"{res['pct_comp']:.0%}",
                  help="Porcentaje de los desembolsos de esta entidad, con el mismo producto, plazo y nivel de monto, "
                       "cuya tasa quedó en el tercio más bajo del mercado.")
        c2.caption(f"Base: {res['n_comp']} desembolsos de esta entidad.")
    else:
        c2.metric("Desembolsos con tasa baja", "Sin datos suficientes")
    if res["mediana"] is not None:
        st.caption(f"Mediana de las tasas del mercado para este producto, plazo y nivel de monto: "
                   f"{fmt(res['mediana'])}% E.A.")
    if res["empate"]:
        st.info("Hay otras entidades con un desempeño muy similar para este perfil. "
                "Conviene consultar más de una antes de decidir.")
    with st.expander("¿Cómo se calcula?"):
        st.markdown(
            "1. Se comparan los créditos de vivienda desembolsados entre julio y septiembre de 2026 con los de "
            "su mismo producto, plazo y nivel de monto. Se consideran de **tasa baja** los del tercio con las "
            "tasas más bajas.\n"
            "2. Un modelo aprendió qué entidades aparecen con más frecuencia entre esos créditos de tasa baja "
            "para cada perfil.\n"
            f"3. La aplicación recomienda la entidad que más se repite para tu perfil, entre las "
            f"{res['n_disponibles']} que tienen desembolsos en tu ciudad.\n\n"
            "La tasa, la mediana y el porcentaje que ves son **datos observados**, no predicciones.")

st.divider()
st.caption("Guía de referencia basada en tasas promedio de desembolsos ya realizados (Superintendencia "
           "Financiera de Colombia, cortes semanales del 3 de julio al 11 de septiembre de 2026). "
           "No es una oferta ni incluye tu perfil crediticio, comisiones o seguros. La tasa real se "
           "confirma directamente con la entidad.")

with st.expander("Información técnica (versiones)"):
    st.write(f"Entorno actual: Python {platform.python_version()} · scikit-learn {sklearn.__version__} · pandas {pd.__version__}")
    st.write(f"Modelo entrenado con: Python {bundle.get('version_python', 'desconocida')} · "
             f"scikit-learn {bundle.get('version_sklearn', 'desconocida')}")
