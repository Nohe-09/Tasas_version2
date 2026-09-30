# Lógica de recomendación (compartida por el notebook y la app de Streamlit)
import pandas as pd

MIN_DESEMBOLSOS = 5     # desembolsos mínimos de una entidad en el municipio para considerarla disponible
MIN_ENTIDADES = 3       # entidades mínimas disponibles para dar una recomendación en un municipio
UMBRAL_EMPATE = 0.05    # diferencia de puntuación del modelo bajo la cual se avisa que hay opciones muy similares
MIN_BASE_RESPALDO = 10  # registros mínimos para mostrar el porcentaje de respaldo histórico


def nivel_monto(monto, bordes):
    """Nivel de monto (bajo, medio, alto) con los mismos cortes usados para construir la etiqueta."""
    e1, e2 = bordes
    if monto <= e1:
        return "bajo"
    if monto <= e2:
        return "medio"
    return "alto"


def entidades_disponibles(presencia, municipio):
    p = presencia[(presencia["municipio"] == municipio) & (presencia["n"] >= MIN_DESEMBOLSOS)]
    return sorted(p["entidad"].unique())


def municipios_con_cobertura(presencia):
    p = presencia[presencia["n"] >= MIN_DESEMBOLSOS]
    conteo = p.groupby("municipio")["entidad"].nunique()
    return sorted(conteo[conteo >= MIN_ENTIDADES].index)


def _fila(tabla, filtros):
    m = tabla
    for k, v in filtros.items():
        m = m[m[k] == v]
    return None if len(m) == 0 else m.iloc[0]


def _num(fila, columna):
    if fila is None or columna not in fila.index or pd.isna(fila[columna]):
        return None
    return float(fila[columna])


def recomendar(bundle, referencia, mediana, presencia, producto, plazo, tipo_tasa, municipio, monto):
    """Recomienda la entidad que más se repite entre las ofertas de tasa baja para el perfil,
    limitada a las entidades con desembolsos en el municipio."""
    disponibles = entidades_disponibles(presencia, municipio)
    if len(disponibles) < MIN_ENTIDADES:
        return {"ok": False,
                "mensaje": "No hay datos suficientes en este municipio para hacer una recomendación."}

    pipe = bundle["pipeline"]
    ciudad_grupo = municipio if municipio in bundle["ciudades_principales"] else "otras"
    X = pd.DataFrame([{"producto": producto, "plazo": plazo, "tipo_tasa": tipo_tasa,
                       "ciudad_grupo": ciudad_grupo, "monto_promedio": float(monto)}])
    punt = pd.Series(pipe.predict_proba(X)[0], index=pipe.classes_)
    punt = punt[punt.index.isin(disponibles)]
    if len(punt) == 0:
        return {"ok": False, "mensaje": "Ninguna de las entidades del modelo opera en este municipio."}
    punt = punt / punt.sum() if punt.sum() > 0 else pd.Series(1.0 / len(punt), index=punt.index)
    punt = punt.sort_values(ascending=False)

    entidad = punt.index[0]
    p1 = float(punt.iloc[0])
    p2 = float(punt.iloc[1]) if len(punt) > 1 else 0.0
    nivel = nivel_monto(monto, bundle["bordes_monto"])

    # Datos observados de la entidad en el segmento (con alternativa sin nivel de monto)
    base = {"entidad": entidad, "producto": producto, "plazo": plazo}
    exacta = _fila(referencia, {**base, "nivel_monto": nivel})
    general = _fila(referencia, {**base, "nivel_monto": "todos"})
    tasa_ref = _num(exacta, "tasa_ref")
    if tasa_ref is None:
        tasa_ref = _num(general, "tasa_ref")
    pct, n_comp = _num(exacta, "pct_competitiva"), (_num(exacta, "n_comp") or 0)
    if pct is None:
        pct, n_comp = _num(general, "pct_competitiva"), (_num(general, "n_comp") or 0)
    if pct is not None and n_comp < MIN_BASE_RESPALDO:
        pct = None

    seg = {"producto": producto, "plazo": plazo}
    med = _num(_fila(mediana, {**seg, "nivel_monto": nivel}), "mediana")
    if med is None:
        med = _num(_fila(mediana, {**seg, "nivel_monto": "todos"}), "mediana")

    return {"ok": True, "entidad": entidad, "empate": (p1 - p2) < UMBRAL_EMPATE, "nivel_monto": nivel,
            "tasa_ref": tasa_ref, "mediana": med,
            "diferencia": (tasa_ref - med) if (tasa_ref is not None and med is not None) else None,
            "pct_comp": pct, "n_comp": int(n_comp), "n_disponibles": len(disponibles),
            "puntaje_modelo": p1}
