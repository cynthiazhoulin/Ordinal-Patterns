# Patrones ordinales en textos periodísticos

Análisis de redes de coocurrencia, patrones ordinales y modelos de clasificación aplicados a un corpus de noticias.
"""

# Si se ejecuta en un entorno nuevo:
# %pip install -q unidecode python-louvain umap-learn kaleido openpyxl

import ast
import math
import os
import re
from collections import Counter, defaultdict
from itertools import combinations

import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from nltk.stem.snowball import SnowballStemmer
from plotly.subplots import make_subplots
from scipy.spatial.distance import jensenshannon
from tqdm.auto import tqdm
from unidecode import unidecode

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

"""## Configuración"""

DATA_DIR = '/content/drive/MyDrive/CIUP/COLUMNAS/BBDD'
STOPWORDS_PATH = '/content/drive/MyDrive/CIUP/COLUMNAS/stopwords.txt'

"""## Funciones auxiliares"""

def cargar_stopwords(ruta_txt):
    stopwords = set()
    with open(ruta_txt, encoding="utf-8") as f:
        for linea in f:
            palabra = linea.strip().lower()
            if palabra:
                stopwords.add(palabra)
    return stopwords

stopwords = cargar_stopwords(STOPWORDS_PATH)

stemmer_es = SnowballStemmer("spanish")

def limpiar_texto(texto, stopwords=stopwords, stemmer=stemmer_es):

    texto = texto.lower()
    texto = unidecode(texto)
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    tokens = texto.split()
    if stopwords is not None:
        tokens = [t for t in tokens if t not in stopwords and len(t)>1]

    texto_limpio = " ".join(tokens)

    return texto_limpio

def calcular_metricas_grafo(tokens, ventana=3, dibujar=False, devolver_grafo=False):

    if not isinstance(tokens, list) or len(tokens) < 2:
        resultado = {
            'densidad': 0.0,
            'nodos': 0,
            'aristas': 0,
            'grado_promedio': 0.0,
            'coef_clustering': 0.0
        }
        if devolver_grafo:
            return resultado, nx.Graph()
        else:
            return resultado

    G = nx.Graph()

    # Construcción del grafo de co-ocurrencias
    for i in range(len(tokens)):
        for j in range(i+1, min(i+ventana, len(tokens))):
            palabra1, palabra2 = tokens[i], tokens[j]
            if G.has_edge(palabra1, palabra2):
                G[palabra1][palabra2]['weight'] += 1
            else:
                G.add_edge(palabra1, palabra2, weight=1)

    # Cálculo de métricas
    if G.number_of_nodes() > 0:
        grados = dict(G.degree())
        grado_promedio = sum(grados.values()) / len(grados)
        densidad = nx.density(G)
        coef_clustering = nx.average_clustering(G)
    else:
        grado_promedio = 0.0
        densidad = 0.0
        coef_clustering = 0.0

    resultado = {
        'densidad': densidad,
        'nodos': G.number_of_nodes(),
        'aristas': G.number_of_edges(),
        'grado_promedio': grado_promedio,
        'coef_clustering': coef_clustering
    }

    # Dibujar el grafo
    if dibujar and G.number_of_nodes() > 0:
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        weights = [G[u][v]['weight'] for u, v in G.edges()]

        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=10)

        plt.axis('off')
        plt.title('Grafo de co-ocurrencias')
        plt.show()

    if devolver_grafo:
        return resultado, G
    else:
        return resultado

def to_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if x.startswith('[') and x.endswith(']'):
            try:
                lst = ast.literal_eval(x)
                return [str(t).strip() for t in lst if str(t).strip()]
            except Exception:
                pass
        return [w for w in x.replace(',', ' ').split() if w]
    return list(x)

def adj_pairs(seq):
    a, b = tee(seq)
    next(b, None)
    for x, y in zip(a, b):
        if x == y:
            continue
        u, v = sorted((x, y))
        yield (u, v)

def construir_red_palabras(
    df,
    col_tokens="Contenido_limpio",
    unidad="Fuente",
    normalizacion="total",
    MIN_COUNT=5,
    MIN_UNITS=2,
    MIN_WEIGHT=0.15,
    TOP_EDGELBL=30,
    dibujar=True,
    layout_seed=7
):

    df_local = df.copy()
    df_local[col_tokens] = df_local[col_tokens].apply(to_list)
    n_unidades = df_local[unidad].nunique()
    print("Unidades:", unidad, "| n =", n_unidades)

    pair_count = Counter()
    pair_unidades = defaultdict(set)

    for _, row in df_local[[unidad, col_tokens]].iterrows():
        uid = row[unidad]
        words = [w for w in row[col_tokens] if w]

        for p in adj_pairs(words):
            pair_count[p] += 1
            pair_unidades[p].add(uid)


    def norm_occurrences_total(pair):
        return pair_count[pair] / n_unidades

    def norm_unique_units(pair):
        return len(pair_unidades[pair]) / n_unidades

    if normalizacion == "total":
        normalize = norm_occurrences_total
    elif normalizacion == "unidades":
        normalize = norm_unique_units
    else:
        raise ValueError("normalizacion debe ser 'total' o 'unidades'")

    G = nx.Graph()
    for (u, v), cnt in pair_count.items():
        units = len(pair_unidades[(u, v)])
        if cnt < MIN_COUNT or units < MIN_UNITS:
            continue
        w = normalize((u, v))
        if w >= MIN_WEIGHT:
            G.add_edge(u, v, weight=w, count=cnt, n_units=units)

    print(f"Grafo filtrado -> Nodos: {G.number_of_nodes()} | Aristas: {G.number_of_edges()}")

    top_edges = []

    if G.number_of_edges() == 0:
        print("No hay aristas tras los umbrales. Baja MIN_COUNT/MIN_UNITS/MIN_WEIGHT.")
    elif dibujar:
        pos = nx.spring_layout(G, seed=layout_seed, k=0.6, iterations=50)

        w_vals = np.array([d["weight"] for _, _, d in G.edges(data=True)])
        w_min, w_max = w_vals.min(), w_vals.max()
        if w_max == w_min:
            edge_widths = [2.0 for _ in w_vals]
        else:
            edge_widths = list(1 + 7*(w_vals - w_min)/(w_max - w_min))

        plt.figure(figsize=(12, 10))
        nx.draw_networkx_nodes(G, pos, node_size=300)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.35)
        nx.draw_networkx_labels(G, pos, font_size=9)

        top_edges = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)[:TOP_EDGELBL]
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in top_edges}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels,
            font_size=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        plt.title(f"Red de palabras (adyacencia) — unidad: {unidad}")
        plt.axis("off")
        plt.show()

        # Top enlaces por peso en texto
        for u, v, d in top_edges:
            print(f"{u} — {v} | peso={d['weight']:.3f} | count={d['count']} | unidades={d['n_units']}")

    return G, top_edges, pair_count, pair_unidades

def construir_y_dibujar_grafo(
    pair_count,
    pair_unidades,
    normalize,
    unidad="Fuente",
    MIN_COUNT=50,
    MIN_UNITS=5,
    MIN_WEIGHT=0.3,
    TOP_EDGELBL=30,
    seed=7,
    k=0.6,
    iterations=50,
    dibujar=True
):

    G = nx.Graph()
    for (u, v), cnt in pair_count.items():
        units = len(pair_unidades[(u, v)])
        if cnt < MIN_COUNT or units < MIN_UNITS:
            continue
        w = normalize((u, v))
        if w >= MIN_WEIGHT:
            G.add_edge(u, v, weight=w, count=cnt, n_units=units)

    print(f"Grafo filtrado -> Nodos: {G.number_of_nodes()} | Aristas: {G.number_of_edges()}")

    top_edges = []

    if G.number_of_edges() == 0:
        print("No hay aristas tras los umbrales. Baja MIN_COUNT/MIN_UNITS/MIN_WEIGHT.")
        return G, top_edges

    if dibujar:
        pos = nx.spring_layout(G, seed=seed, k=k, iterations=iterations)

        w_vals = np.array([d["weight"] for _, _, d in G.edges(data=True)])
        w_min, w_max = w_vals.min(), w_vals.max()
        if w_max == w_min:
            edge_widths = [2.0 for _ in w_vals]
        else:
            edge_widths = list(1 + 7*(w_vals - w_min)/(w_max - w_min))

        plt.figure(figsize=(12, 10))
        nx.draw_networkx_nodes(G, pos, node_size=300)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.35)
        nx.draw_networkx_labels(G, pos, font_size=9)

        top_edges = sorted(
            G.edges(data=True),
            key=lambda x: x[2]["weight"],
            reverse=True
        )[:TOP_EDGELBL]

        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in top_edges}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels,
            font_size=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        plt.title(f"Red de palabras (adyacencia) — unidad: {unidad}")
        plt.axis("off")
        plt.show()

        for u, v, d in top_edges:
            print(f"{u} — {v} | peso={d['weight']:.3f} | count={d['count']} | unidades={d['n_units']}")

    return G, top_edges

def calcular_metricas_grafo2(tokens, ventana=3, dibujar=False, devolver_grafo=False):

    if not isinstance(tokens, list) or len(tokens) < 2:
        resultado = {
            'densidad': 0.0,
            'nodos': 0,
            'aristas': 0,
            'grado_promedio': 0.0,
            'coef_clustering': 0.0,
            'segundo_momento_grado': 0.0,
            'varianza_grados_sin_max': 0.0,
            'grado_maximo': 0.0,
            'relacion_grado_max_promedio': 0.0,
            'grado_promedio_sin_max': 0.0,
            'asortatividad': 0.0,
            'modularidad': 0.0,
            'num_comunidades': 0,
            'tam_comunidades': {},
            'palabras_por_comunidad': {}
        }
        if devolver_grafo:
            return resultado, nx.Graph()
        else:
            return resultado

    # Construcción del grafo
    G = nx.Graph()
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + ventana, len(tokens))):
            palabra1, palabra2 = tokens[i], tokens[j]
            if G.has_edge(palabra1, palabra2):
                G[palabra1][palabra2]['weight'] += 1
            else:
                G.add_edge(palabra1, palabra2, weight=1)

    # Cálculo de métricas
    if G.number_of_nodes() > 0:
        grados_dict = dict(G.degree())
        grados = list(grados_dict.values())
        n = len(grados)

        grado_promedio = sum(grados) / n
        densidad = nx.density(G)
        coef_clustering = nx.average_clustering(G)
        segundo_momento_grado = sum(k**2 for k in grados) / n
        grado_maximo = max(grados)

        relacion_grado_max_promedio = (
            grado_maximo / grado_promedio if grado_promedio > 0 else 0.0
        )

        if n > 1:
            grados_sin_max = grados.copy()
            grados_sin_max.remove(grado_maximo)

            n2 = len(grados_sin_max)
            if n2 > 0:
                grado_promedio_sin_max = sum(grados_sin_max) / n2
                media_sin_max = grado_promedio_sin_max
                varianza_grados_sin_max = (
                    sum((k - media_sin_max) ** 2 for k in grados_sin_max) / n2
                )
            else:
                grado_promedio_sin_max = 0.0
                varianza_grados_sin_max = 0.0
        else:
            grado_promedio_sin_max = 0.0
            varianza_grados_sin_max = 0.0

        try:
            asortatividad = nx.degree_assortativity_coefficient(G)
            if asortatividad is None or (isinstance(asortatividad, float) and math.isnan(asortatividad)):
                asortatividad = 0.0
        except Exception:
            asortatividad = 0.0

        # Comunidades (Louvain) y modularidad
        try:
            import community.community_louvain as community_louvain

            # nodo -> comunidad
            particion = community_louvain.best_partition(G)

            modularidad = community_louvain.modularity(particion, G)
            num_comunidades = len(set(particion.values()))
            tam_comunidades = dict(Counter(particion.values()))

            # comunidad -> lista de palabras
            palabras_por_comunidad = {}
            for nodo, com in particion.items():
                palabras_por_comunidad.setdefault(com, []).append(nodo)

        except Exception:
            modularidad = 0.0
            num_comunidades = 0
            tam_comunidades = {}
            palabras_por_comunidad = {}

    else:
        densidad = 0.0
        coef_clustering = 0.0
        grado_promedio = 0.0
        segundo_momento_grado = 0.0
        grado_maximo = 0.0
        relacion_grado_max_promedio = 0.0
        grado_promedio_sin_max = 0.0
        varianza_grados_sin_max = 0.0
        asortatividad = 0.0
        modularidad = 0.0
        num_comunidades = 0
        tam_comunidades = {}
        palabras_por_comunidad = {}

    resultado = {
        'densidad': densidad,
        'nodos': G.number_of_nodes(),
        'aristas': G.number_of_edges(),
        'grado_promedio': grado_promedio,
        'coef_clustering': coef_clustering,
        'segundo_momento_grado': segundo_momento_grado,
        'varianza_grados_sin_max': varianza_grados_sin_max,
        'grado_maximo': grado_maximo,
        'relacion_grado_max_promedio': relacion_grado_max_promedio,
        'grado_promedio_sin_max': grado_promedio_sin_max,
        'asortatividad': asortatividad,
        'modularidad': modularidad,
        'num_comunidades': num_comunidades,
        'tam_comunidades': tam_comunidades,
        'palabras_por_comunidad': palabras_por_comunidad  #
    }

    # Dibujar el grafo
    if dibujar and G.number_of_nodes() > 0:
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        weights = [G[u][v]['weight'] for u, v in G.edges()]

        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=10)

        plt.axis('off')
        plt.title('Grafo de co-ocurrencias')
        plt.show()

    if devolver_grafo:
        return resultado, G
    else:
        return resultado

from wordcloud import WordCloud

def wordcloud_comunidad_doc(df_comunidades, idx_doc, comunidad=None, max_words=100):
    df_doc = df_comunidades[df_comunidades['idx_doc'] == idx_doc]

    if comunidad is None:
        print("Comunidades en el documento", idx_doc, ":", df_doc['comunidad'].unique())
        return

    df_sub = df_doc[df_doc['comunidad'] == comunidad]
    if df_sub.empty:
        print(f"No hay palabras para doc {idx_doc} en comunidad {comunidad}")
        return

    texto = " ".join(df_sub['palabra'])

    wc = WordCloud(width=800, height=400, background_color="white", max_words=max_words)
    wc.generate(texto)

    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud - doc {idx_doc}, comunidad {comunidad}")
    plt.show()

def calcular_palabras_puente(G, palabras_por_comunidad):

    # reconstruir partición nodo -> comunidad
    particion = {
        palabra: com
        for com, palabras in palabras_por_comunidad.items()
        for palabra in palabras
    }

    # betweenness centrality en documento
    bet = nx.betweenness_centrality(G, weight='weight', normalized=True)

    filas = []
    for nodo in G.nodes():
        comunidad = particion.get(nodo, None)
        vecinos = list(G.neighbors(nodo))
        grado = len(vecinos)

        # vecinos que están en otras comunidades (de mismo doc)
        vecinos_externos = sum(
            1 for v in vecinos if particion.get(v, None) != comunidad
        )

        bridging = vecinos_externos / grado if grado > 0 else 0

        filas.append({
            'palabra': nodo,
            'comunidad': comunidad,
            'grado_total': grado,
            'vecinos_externos': vecinos_externos,
            'bridging_coef': bridging,
            'betweenness': bet.get(nodo, 0.0)
        })

    df = pd.DataFrame(filas)
    return df.sort_values(['bridging_coef', 'betweenness'], ascending=False)

def permutation_entropy_complexity(conteos):

    conteos = np.array(conteos, dtype=float)
    total = conteos.sum()
    if total == 0:
        return 0, 0, 0  # H, Q, C = 0

    # Probabilidades
    p = conteos / total
    Df = len(p)  # D! patterns

    # Shannon entropy
    S = -np.sum(p[p>0] * np.log(p[p>0]))
    Smax = np.log(Df)
    H = S / Smax


    # Disequilibrio Q
    pe = np.ones(Df) / Df
    M = (p + pe) / 2

    def entropy(vec):
        return -np.sum(vec[vec>0] * np.log(vec[vec>0]))

    JS = entropy(M) - 0.5*entropy(p) - 0.5*entropy(pe)

    # Normalization constant Q0
    Q0 = -2 / (( (Df+1)/Df * np.log(Df+1) ) - 2*np.log(2*Df) + np.log(Df))

    Q = Q0 * JS

    # Complexity
    C = H * Q
    return H, Q, C

def permutation_entropy_complexity_por_fuente(conteos):
    conteos = np.array(conteos, dtype=float)
    total = conteos.sum()
    if total == 0:
        return 0, 0, 0

    p = conteos / total
    Df = len(p)

    S = -np.sum(p[p>0] * np.log(p[p>0]))
    Smax = np.log(Df)
    H = S / Smax

    pe = np.ones(Df) / Df
    M = (p + pe) / 2

    def entropy(vec):
        return -np.sum(vec[vec>0] * np.log(vec[vec>0]))

    JS = entropy(M) - 0.5*entropy(p) - 0.5*entropy(pe)

    Q0 = -2 / (((Df+1)/Df * np.log(Df+1)) - 2*np.log(2*Df) + np.log(Df))
    Q = Q0 * JS

    C = H * Q
    return H, Q, C

"""# 1. Carga y preprocesamiento de datos"""

ruta = DATA_DIR
archivos = [f for f in os.listdir(ruta) if f.endswith('.xlsx')]

dfs = []

for archivo in archivos:

    ruta_archivo = os.path.join(ruta, archivo)
    df = pd.read_excel(ruta_archivo)
    nombre_sin_ext = os.path.splitext(archivo)[0]
    df["fuente"] = nombre_sin_ext

    df.to_excel(os.path.join(ruta, f"{nombre_sin_ext}.xlsx"), index=False)

    dfs.append(df)

df_final = pd.concat(dfs, ignore_index=True)
df_final = df_final.drop_duplicates()
df_final = df_final.dropna()

df_final.head()

df_final = df_final.drop_duplicates(subset=['Contenido'], keep='first')
df_final.shape

df = df_final[['fuente', 'Autor', 'Titulo', 'Contenido']].copy()

df = df[['fuente','Autor','Titulo','Contenido']]

df['Limpio'] = df['Contenido'].apply(limpiar_texto)
df['Tokens'] = df['Limpio'].str.split()
df['Palabras'] = df['Tokens'].str.len()
df = df.loc[df['Palabras'] > 10].reset_index(drop=True)
df.head()

"""# 2. Redes de coocurrencia léxica
Exploración de la base de datos
"""

fig = px.histogram(
    df_resultado,
    x='nodos',
    nbins=30,
    title='Distribución de cantidad de palabras (nodos)',
    color_discrete_sequence=['#4C78A8'])

fig.update_layout(
    xaxis_title="Cantidad de palabras",
    yaxis_title="Frecuencia",
    template="simple_white",
    bargap=0.05,
    title_font_size=20
)

fig.show()

df_plot = df_resultado.copy()

x = df_plot["nodos"]
y = df_plot["densidad"]

valid_mask = (x > 0) & (y > 0)
x_valid = x[valid_mask]
y_valid = y[valid_mask]

x_log = np.log(x_valid)
y_log = np.log(y_valid)

# Ajuste lineal log-log
slope, intercept = np.polyfit(x_log, y_log, 1)
y_log_pred = slope * x_log + intercept

# R^2
ss_res = np.sum((y_log - y_log_pred) ** 2)
ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Curva ajustada en espacio normal
x_fit = np.linspace(x_valid.min(), x_valid.max(), 200)
y_fit = np.exp(intercept) * x_fit ** slope

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Densidad vs Número de Nodos",
        f"Ajuste lineal en escala log-log<br>"
        f"Pend: {slope:.2f} · Intc: {intercept:.2f} · R^2: {r_squared:.3f}"
    )
)

palette = px.colors.qualitative.Set2

for i, (fuente, df_f) in enumerate(df_plot.groupby("fuente")):
    color = palette[i % len(palette)]
    fig.add_trace(
        go.Scatter(
            x=df_f["nodos"],
            y=df_f["densidad"],
            mode="markers",
            name=fuente,
            marker=dict(size=7, color=color, opacity=0.7),
            legendgroup=fuente
        ),
        row=1, col=1
    )

# Línea de ajuste en escala normal
fig.add_trace(
    go.Scatter(
        x=x_fit,
        y=y_fit,
        mode="lines",
        line=dict(color="black", dash="dash"),
        name=f"Ajuste log-log (Pend: {slope:.2f})",
        legendgroup="ajuste"
    ),
    row=1, col=1
)

fig.update_xaxes(title_text="Número de Nodos", range=[0, 1000], row=1, col=1)
fig.update_yaxes(title_text="Densidad", range=[0, 0.5], row=1, col=1)

for i, (fuente, df_f) in enumerate(df_plot.groupby("fuente")):
    color = palette[i % len(palette)]
    mask_f = (df_f["nodos"] > 0) & (df_f["densidad"] > 0)
    fig.add_trace(
        go.Scatter(
            x=np.log(df_f.loc[mask_f, "nodos"]),
            y=np.log(df_f.loc[mask_f, "densidad"]),
            mode="markers",
            name=fuente,
            marker=dict(size=7, color=color, opacity=0.7),
            legendgroup=fuente,
            showlegend=False
        ),
        row=1, col=2
    )

fig.add_trace(
    go.Scatter(
        x=x_log,
        y=y_log_pred,
        mode="lines",
        line=dict(color="black", dash="dash"),
        name=f"Recta log-log",
        legendgroup="ajuste",
        showlegend=False
    ),
    row=1, col=2
)

fig.update_xaxes(
    title_text="log(Número de Nodos)",
    range=[np.log(10), np.log(1000)],
    row=1, col=2
)
fig.update_yaxes(
    title_text="log(Densidad)",
    range=[np.log(0.005), np.log(0.5)],
    row=1, col=2
)

fig.update_layout(
    title="Densidad vs Nodos por fuente (con ajuste log-log global)",
    template="simple_white",
    width=1200,
    height=500,
    legend_title="Fuente"
)

fig.show()

fig = px.box(
    df_resultado,
    x="fuente",
    y="coef_clustering",
    points="outliers",
    title="Distribución del Coeficiente de Clustering por fuente",
    color="fuente",
)

fig.update_layout(
    xaxis_title="Fuente",
    yaxis_title="Coeficiente de Clustering",
    template="simple_white",
    width=900,
    height=500,
    showlegend=False
)

fig.update_yaxes(range=[0, 1])
fig.show()

G, top_edges, pair_count, pair_unidades = construir_red_palabras(
    df_resultado,
    col_tokens="Limpio",
    unidad="fuente",
    normalizacion="total",
    MIN_COUNT=5,
    MIN_UNITS=2,
    MIN_WEIGHT=0.15,
    TOP_EDGELBL=30,
    dibujar=True
)

unidad = "fuente"
n_unidades = len(set(df[unidad]))

# Normalizaciones
def norm_occurrences_total(pair):
    return pair_count[pair] / n_unidades

def norm_unique_units(pair):
    return len(pair_unidades[pair]) / n_unidades

normalize = norm_occurrences_total

G, top_edges = construir_y_dibujar_grafo(
    pair_count=pair_count,
    pair_unidades=pair_unidades,
    normalize=normalize,
    unidad=unidad,
    MIN_COUNT=50,
    MIN_UNITS=5,
    MIN_WEIGHT=0.3,
    TOP_EDGELBL=50)

df_resultado.to_excel('df_metricas_final.xlsx', index=False)

"""# 3. Redes de patrones ordinales"""

fig = px.box(
    df_resultado,
    x="fuente",
    y="asortatividad",
    points="outliers",
    title="Distribución de Asortatividad por fuente",
    color="fuente",
)

fig.update_layout(
    xaxis_title="Fuente",
    yaxis_title="Asortatividad",
    template="simple_white",
    width=900,
    height=500,
    showlegend=False
)

fig.update_yaxes(range=[-1, 1])
fig.show()

def serie_longitudes_desde_tokens(tokens):
    return [len(t) for t in tokens]


def ordinal_patterns(x, D=3, tau=1):
    x = np.asarray(x, dtype=float)

    n_trunc = (len(x) // D) * D
    x = x[:n_trunc]

    patrones, ventanas = [], []
    for t in range(0, n_trunc, D):
        ventana = x[t:t+D]
        ranks = tuple(int(k) for k in np.argsort(ventana))
        patrones.append(ranks)
        ventanas.append(ventana)

    return patrones, ventanas


def construir_red_ordinal(patrones, quitar_autoloops=True):
    G = nx.DiGraph()

    # Añadir nodos (cada patrón es un nodo)
    for p in patrones:
        if p not in G:
            G.add_node(p)

    # Añadir aristas (transiciones entre patrones consecutivos)
    for a, b in zip(patrones[:-1], patrones[1:]):
        if quitar_autoloops and a == b:
            continue

        if G.has_edge(a, b):
            G[a][b]['weight'] += 1  # sumar frecuencia
        else:
            G.add_edge(a, b, weight=1.0)

    # Normalizar pesos salientes (para que sean probabilidades)
    for u in G.nodes():
        sucesores = list(G.successors(u))
        total = sum(G[u][v]['weight'] for v in sucesores)
        if total > 0:
            for v in sucesores:
                G[u][v]['weight'] /= total

    return G


def T_geometrico(perm):
    return sum(abs(perm[i+1] - perm[i]) for i in range(len(perm)-1))

def secuencia_T(patrones):
    return [T_geometrico(p) for p in patrones]


def metricas_red_ordinal_desde_tokens(tokens, D=3, tau=1, devolver_grafo=False):

    serie = serie_longitudes_desde_tokens(tokens)
    if len(serie) < D:
        resultado = {
            'ord_num_nodos': 0,
            'ord_num_aristas': 0,
            'ord_densidad': 0.0,
            'ord_T_media': 0.0,
            'ord_T_std': 0.0,
            'ord_T_max': 0.0
        }
        return (resultado, nx.DiGraph()) if devolver_grafo else resultado

    patrones, ventanas = ordinal_patterns(serie, D=D, tau=tau)
    G = construir_red_ordinal(patrones)


    num_nodos = G.number_of_nodes()
    num_aristas = G.number_of_edges()
    densidad = nx.density(G)

    T_vals = secuencia_T(patrones)
    T_media = float(np.mean(T_vals))
    T_std   = float(np.std(T_vals))
    T_max   = float(np.max(T_vals))

    resultado = {
        'ord_num_nodos': num_nodos,
        'ord_num_aristas': num_aristas,
        'ord_densidad': densidad,
        'ord_T_media': T_media,
        'ord_T_std': T_std,
        'ord_T_max': T_max
    }

    return (resultado, G) if devolver_grafo else resultado



def metricas_ordinales(df, col_tokens='Tokens', D=3, tau=1):

    def procesar_fila(tokens):
        met = metricas_red_ordinal_desde_tokens(tokens, D=D, tau=tau, devolver_grafo=False)
        return pd.Series(met)

    df_out = df.copy()
    nuevas_cols = df_out[col_tokens].apply(procesar_fila)
    df_out = pd.concat([df_out, nuevas_cols], axis=1)
    return df_out

df_ordinal = metricas_ordinales(df, col_tokens='Tokens', D=3, tau=1)
df_ordinal.head()

tokens_0 = df.loc[0, 'Tokens']

metricas, G0 = metricas_red_ordinal_desde_tokens(tokens_0, D=3, tau=1, devolver_grafo=True)

plt.figure(figsize=(6,6))
pos = nx.spring_layout(G0, k=1.0, iterations=100)
nx.draw(G0, pos, with_labels=True, node_color='lightyellow', node_size=1500, arrows=True)
plt.title("Red ordinal de la noticia 0")
plt.show()

metricas

fig = px.box(
    df_ordinal,
    x="fuente",
    y="ord_T_media",
    points="outliers",
    title="Distribución de ord_T_media por fuente",
    color="fuente",
)

fig.update_layout(
    xaxis_title="Fuente",
    yaxis_title="ord_T_media",
    template="simple_white",
    width=900,
    height=500,
    showlegend=False
)

fig.show()

"""# 4. Comunidades y palabras puente

## Palabras en comunidades
"""

def expandir_comunidades(df, col='palabras_por_comunidad'):

    filas = []

    for idx, comunidades in df[col].items():
        if not isinstance(comunidades, dict):
            continue
        for com, palabras in comunidades.items():
            for palabra in palabras:
                filas.append({
                    'idx_doc': idx,
                    'comunidad': com,
                    'palabra': palabra
                })

    df_plano = pd.DataFrame(filas)
    return df_plano

df_comunidades = expandir_comunidades(df_resultado, col='palabras_por_comunidad')
df_comunidades.head()

wordcloud_comunidad_doc(df_comunidades, idx_doc=0)
wordcloud_comunidad_doc(df_comunidades, idx_doc=0, comunidad=8)

filas = []

for idx, tokens in tqdm(df['Tokens'].items(),
                        total=len(df),
                        desc="Procesando documentos"):

    metricas, G = calcular_metricas_grafo2(tokens, devolver_grafo=True)

    if G.number_of_nodes() == 0:
        continue

    df_doc = calcular_palabras_puente(G, metricas['palabras_por_comunidad'])
    df_doc['idx_doc'] = idx

    filas.append(df_doc)

df_puentes_todos = pd.concat(filas, ignore_index=True)

"""## 4.1. Enfoque por longitud de palabra"""

PATRONES = [
    (0,1,2),
    (0,2,1),
    (1,0,2),
    (1,2,0),
    (2,0,1),
    (2,1,0)]

NOMBRES = ['012','021','102','120','201','210']


def ordinal_patterns_blocks(x, D=3):
    x = np.asarray(x, dtype=float)
    patrones = []

    n_trunc = (len(x) // D) * D   # truncar a múltiplo de D

    for t in range(0, n_trunc, D):  # avanzar de D en D
        ventana = x[t:t+D]
        ranks = np.argsort(ventana)
        patrones.append(tuple(int(k) for k in ranks))

    return patrones


def contar_patrones_CANTIDAD(tokens):
    serie = serie_longitudes_desde_tokens(tokens)

    if len(serie) < 3:
        return pd.Series({f'pat_{n}': 0 for n in NOMBRES})

    patrones = ordinal_patterns_blocks(serie, D=3)
    c = Counter(patrones)

    datos = {}
    for pat, nombre in zip(PATRONES, NOMBRES):
        datos[f'pat_{nombre}'] = c.get(pat, 0)
    return pd.Series(datos)

df_pats = df.copy()
df_pats[[f'pat_{n}' for n in NOMBRES]] = df_pats['Tokens'].apply(contar_patrones_CANTIDAD)
df_pats.head()

conteos_por_fuente = df_pats.groupby('fuente')[[f'pat_{n}' for n in NOMBRES]].sum()
conteos_por_fuente

import math
import numpy as np
import matplotlib.pyplot as plt

fuentes = conteos_por_fuente.index.tolist()
n = len(fuentes)

cols = 4
rows = math.ceil(n / cols)

colors = plt.cm.magma(np.linspace(0.2,0.85,len(NOMBRES)))

fig, axes = plt.subplots(rows, cols,
                         figsize=(18, 4.5*rows),
                         constrained_layout=True)

axes = np.array(axes).flatten()

for ax, fuente in zip(axes, fuentes):

    valores = conteos_por_fuente.loc[fuente].values
    probabilidades = valores / valores.sum()

    ax.bar(
        NOMBRES,
        probabilidades,
        color=colors,
        edgecolor="white",
        linewidth=0.6
    )

    ax.set_title(fuente, fontsize=13)

    ax.set_ylim(0, 0.60)

    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    ax.grid(axis='y', linestyle='--', alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for ax in axes[len(fuentes):]:
    fig.delaxes(ax)

plt.show()

probabilidades_por_fuente = conteos_por_fuente.div(conteos_por_fuente.sum(axis=1), axis=0)
probabilidades_por_fuente

import matplotlib.pyplot as plt
import seaborn as sns

probabilidades = conteos_por_fuente.div(conteos_por_fuente.sum(axis=1), axis=0)
probabilidades.columns = [c.replace("pat_", "") for c in probabilidades.columns]

plt.figure(figsize=(12, 6))

sns.heatmap(
    probabilidades,
    cmap="magma",
    annot=False,
    linewidths=0.5,
    linecolor="white",
    cbar=True,
    xticklabels=True,
    yticklabels=True
)

plt.xlabel("")
plt.ylabel("")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

NOMBRES = [f"pat_{n}" for n in ["012","021","102","120","201","210"]]

df_stats = []

for idx, row in df_pats.iterrows():
    conteos = [row[n] for n in NOMBRES]
    H, Q, C = permutation_entropy_complexity(conteos)
    df_stats.append([idx, H, Q, C])

df_stats = pd.DataFrame(df_stats, columns=["idx","entropia_H","desequilibrio_Q","complejidad_C"])

df_e1 = df.merge(df_stats, left_index=True, right_on='idx')
df_e1.to_excel('df_final_e1_FINAL.xlsx', index=False)
df_e1.head()

filas = []

for fuente, row in conteos_por_fuente.iterrows():
    conteos = row.values
    H, Q, C = permutation_entropy_complexity_por_fuente(conteos)
    filas.append([fuente, H, Q, C])

df_fuente_stats = pd.DataFrame(filas, columns=["fuente","H","Q","C"])
df_fuente_stats

probabilidades = conteos_por_fuente.div(conteos_por_fuente.sum(axis=1), axis=0)

# Estadísticos del patrón 012
print("Promedio:", probabilidades["pat_012"].mean())
print("Mínimo:", probabilidades["pat_012"].min())
print("Máximo:", probabilidades["pat_012"].max())

print("Promedio:", probabilidades["pat_012"].mean() * 100)
print("Mínimo:", probabilidades["pat_012"].min() * 100)
print("Máximo:", probabilidades["pat_012"].max() * 100)

(probabilidades.mean() * 100).round(2)

conteos_por_fuente.sum().sum()

conteos_por_fuente["pat_012"].sum() / conteos_por_fuente.sum().sum()

conteos_por_fuente["pat_210"].sum() / conteos_por_fuente.sum().sum()

"""## 4.2. Enfoque por frecuencia de palabra"""

df_tf = agregar_patrones_tf_df(df)
df_tf.head()

conteos_por_fuente = df_tf.groupby('fuente')[[n for n in NOMBRES]].sum()
conteos_por_fuente

import math
import numpy as np
import matplotlib.pyplot as plt

fuentes = conteos_por_fuente.index.tolist()
n = len(fuentes)

cols = 4
rows = math.ceil(n / cols)
labels = [x.replace("pat_", "") for x in NOMBRES]

colors = plt.cm.magma(np.linspace(0.2,0.85,len(NOMBRES)))

fig, axes = plt.subplots(rows, cols,
                         figsize=(18, 4.5*rows),
                         constrained_layout=True)

axes = np.array(axes).flatten()

for ax, fuente in zip(axes, fuentes):

    valores = conteos_por_fuente.loc[fuente].values
    probabilidades = valores / valores.sum()

    ax.bar(
        labels,
        probabilidades,
        color=colors,
        edgecolor="white",
        linewidth=0.6
    )

    ax.set_title(fuente, fontsize=13)

    ax.set_ylim(0, 0.60)

    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    ax.grid(axis='y', linestyle='--', alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for ax in axes[len(fuentes):]:
    fig.delaxes(ax)

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

probabilidades_por_fuente = conteos_por_fuente.div(
    conteos_por_fuente.sum(axis=1), axis=0
)

plt.figure(figsize=(12,4))

ax = sns.heatmap(
    probabilidades_por_fuente,
    cmap="magma",
    annot=False,
    linewidths=0,
    linecolor=None,
    vmin=0.0,
    vmax=1.0,
    cbar_kws={
        "pad":0.02,
        "fraction":0.04
    }
)

ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    ha="right",
    fontsize=9
)

ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    fontsize=10
)

ax.set_xticklabels(
    [c.replace("pat_", "") for c in probabilidades_por_fuente.columns],
    rotation=45,
    ha="right",
    fontsize=9
)

for spine in ax.spines.values():
    spine.set_visible(False)

cbar = ax.collections[0].colorbar
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()

plt.savefig(
    "heatmap_probabilidades.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

NOMBRES = [f"pat_{n}" for n in ["012","021","102","120","201","210"]]

df_stats = []

for idx, row in df_tf.iterrows():
    conteos = [row[n] for n in NOMBRES]
    H, Q, C = permutation_entropy_complexity(conteos)
    df_stats.append([idx, H, Q, C])

df_stats = pd.DataFrame(df_stats, columns=["idx","entropia_H","desequilibrio_Q","complejidad_C"])

df_e2 = df.merge(df_stats, left_index=True, right_on='idx')
df_e2.to_excel('df_final_e2_FINAL.xlsx', index=False)
df_e2.head()

filas = []

for fuente, row in conteos_por_fuente.iterrows():
    conteos = row.values
    H, Q, C = permutation_entropy_complexity_por_fuente(conteos)
    filas.append([fuente, H, Q, C])

df_fuente_stats = pd.DataFrame(filas, columns=["fuente","H","Q","C"])
df_fuente_stats

probabilidades = conteos_por_fuente.div(conteos_por_fuente.sum(axis=1), axis=0)

# Estadísticos del patrón 012
print("Promedio:", probabilidades["pat_012"].mean())
print("Mínimo:", probabilidades["pat_012"].min())
print("Máximo:", probabilidades["pat_012"].max())

print("Promedio:", probabilidades["pat_012"].mean() * 100)
print("Mínimo:", probabilidades["pat_012"].min() * 100)
print("Máximo:", probabilidades["pat_012"].max() * 100)

(probabilidades.mean() * 100).round(2)

conteos_por_fuente.sum().sum()

conteos_por_fuente["pat_012"].sum() / conteos_por_fuente.sum().sum()

conteos_por_fuente["pat_210"].sum() / conteos_por_fuente.sum().sum()

"""# 5. Relaciones entre métricas de red

## 5.1. Centralidad de intermediación y coeficiente de puente
"""

fig = px.scatter(
    df_puentes_todos,
    x='bridging_coef',
    y='betweenness',
    log_x=True,
    title='Correlación entre bridging coefficient y betweenness centrality',
    labels={
        'bridging_coef': 'Bridging coefficient',
        'betweenness': 'Betweenness centrality'
    }
)

fig.update_layout(showlegend=False)
fig.show()

df_plot = df_puentes_todos[df_puentes_todos['grado_total'] > 0]

fig = px.scatter(
    df_plot,
    x='grado_total',
    y='betweenness',
    log_x=True,
    title='Correlación entre grado y betweenness centrality',
    labels={
        'grado_total': 'Grado',
        'betweenness': 'Betweenness centrality'
    },
    color_discrete_sequence=px.colors.qualitative.Plotly
)

fig.update_layout(showlegend=False)
fig.show()

df_plot = df_puentes_todos[df_puentes_todos['grado_total'] > 0]

fig = px.scatter(
    df_plot,
    x='grado_total',
    y='bridging_coef',
    log_x=True,
    title='Correlación entre grado y bridging coefficient',
    labels={
        'grado_total': 'Grado',
        'bridging_coef': 'Bridging coefficient'
    },
    color_discrete_sequence=px.colors.qualitative.Plotly
)

fig.update_layout(showlegend=False)
fig.show()

df_mean = (df_puentes_todos
    .groupby('grado_total', as_index=False)
    .agg(betweenness_media=('betweenness', 'mean'),
        n=('betweenness', 'size')))

fig = px.scatter(
    df_mean,
    x='grado_total',
    y='betweenness_media',
    log_x=True,
    log_y=True,
    title='Betweenness centrality promedio por grado',
    labels={
        'grado_total': 'Grado',
        'betweenness_media': '⟨Betweenness centrality⟩'})

fig.show()

df_mean_br = (df_puentes_todos
    .groupby('grado_total', as_index=False)
    .agg(bridging_medio=('bridging_coef', 'mean'),
        n=('bridging_coef', 'size')))

fig = px.scatter(
    df_mean_br,
    x='grado_total',
    y='bridging_medio',
    log_x=True,
    log_y=True,
    title='Bridging coefficient promedio por grado',
    labels={
        'grado_total': 'Grado',
        'bridging_medio': '⟨Bridging coefficient⟩'})

fig.show()

"""## 5.2. Plano entropía–complejidad

### Enfoque por longitud de palabra
"""

fig = px.scatter(
    df_e1,
    x="entropia_H",
    y="complejidad_C",
    color="desequilibrio_Q",
    color_continuous_scale="Viridis",
    opacity=0.75,
    hover_data=[c for c in ["fuente", "Titulo", "idx", "Palabras"] if c in df.columns],
    title="Plano Entropía–Complejidad (H–C) por documento (color = Q)"
)

fig.update_layout(template="plotly_white", width=900, height=550)
fig.update_xaxes(range=[0, 1], title="Entropía normalizada (H)", showgrid=True)
fig.update_yaxes(range=[0, 1], title="Complejidad estadística (C)", showgrid=True)

fig.show()

"""### Enfoque por frecuencia de palabra"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

color = px.colors.sequential.Magma[5]

fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.10
)

fig.add_trace(
    go.Scatter(
        x=df_e1["entropia_H"],
        y=df_e1["complejidad_C"],
        mode="markers",
        marker=dict(
            color=color,
            size=8,
            opacity=0.8,
            line=dict(color="black", width=0.5)
        ),
        showlegend=False
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=df_e2["entropia_H"],
        y=df_e2["complejidad_C"],
        mode="markers",
        marker=dict(
            color=color,
            size=8,
            opacity=0.8,
            line=dict(color="black", width=0.5)
        ),
        showlegend=False
    ),
    row=1,
    col=2
)

fig.update_xaxes(
    title_text="Entropy (H)",
    showgrid=True,
    gridcolor="lightgray",
    zeroline=False
)

fig.update_yaxes(
    title_text="Complexity (C)",
    showgrid=True,
    gridcolor="lightgray",
    zeroline=False
)

fig.update_layout(
    template="simple_white",
    width=1100,
    height=500,
    font=dict(
        size=14
    ),
    margin=dict(l=60, r=30, t=60, b=60),
    showlegend=False
)

fig.show()

fig = px.scatter(
    df_e2,
    x="entropia_H",
    y="complejidad_C",
    color="desequilibrio_Q",
    color_continuous_scale="Viridis",
    opacity=0.75,
    hover_data=[c for c in ["fuente", "Titulo", "idx", "Palabras"] if c in df.columns],
    title="Plano Entropía–Complejidad (H–C) por documento (color = Q)"
)

fig.update_layout(template="plotly_white", width=900, height=550)
fig.update_xaxes(range=[0, 1], title="Entropía normalizada (H)", showgrid=True)
fig.update_yaxes(range=[0, 1], title="Complejidad estadística (C)", showgrid=True)

fig.show()

"""# 6. Redes de transición entre patrones ordinales

## 6.1. Enfoque por longitud de palabra
"""

def prob_transiciones_doc(conteo_transiciones):
    total = sum(conteo_transiciones.values())
    if total == 0:
        return {k: 0.0 for k in conteo_transiciones}
    return {k: v / total for k, v in conteo_transiciones.items()}

def red_ordinal_causal(patrones):
    patrones_unicos = sorted(set(patrones))
    conteo_patrones = Counter(patrones)

    G = nx.DiGraph()
    G.add_nodes_from(patrones_unicos)

    for a, b in zip(patrones[:-1], patrones[1:]):
        if G.has_edge(a, b):
            G[a][b]['peso'] += 1
        else:
            G.add_edge(a, b, peso=1)

    conteo_transiciones = {(a, b): G[a][b]['peso'] for a, b in G.edges()}

    prob_cond = {}
    for a, b in G.edges():
        total_salidas = sum(G[a][c]['peso'] for c in G.successors(a))
        prob_cond[(a, b)] = G[a][b]['peso'] / total_salidas if total_salidas > 0 else 0.0

    total_doc = sum(conteo_transiciones.values())
    prob_doc = {k: (v / total_doc if total_doc > 0 else 0.0)
                for k, v in conteo_transiciones.items()}

    in_strength = {n: sum(G[pre][n]['peso'] for pre in G.predecessors(n)) for n in G.nodes()}
    out_strength = {n: sum(G[n][suc]['peso'] for suc in G.successors(n)) for n in G.nodes()}

    return {
        "conteo_patrones": conteo_patrones,
        "conteo_transiciones": conteo_transiciones,
        "prob_transiciones_cond": prob_cond,
        "prob_transiciones_doc": prob_doc,
        "in_strength": in_strength,
        "out_strength": out_strength,
        "grafo": G
    }

def pat_to_str(p):
    return "".join(str(x) for x in p)

def dibujar_red_ordinal_por_idx(df, idx, D=3, layout="spring",
                               umbral_prob=0.0, tipo_prob="cond"):
    tokens = df.loc[idx, "Tokens"]
    longitudes = [len(t) for t in tokens]

    patrones = ordinal_patterns(longitudes, D=D)
    if len(patrones) < 2:
        print("Muy pocos patrones para construir la red.")
        return

    res = red_ordinal_causal(patrones)
    G = res["grafo"]

    if tipo_prob == "cond":
        prob_trans = res["prob_transiciones_cond"]
    elif tipo_prob == "doc":
        prob_trans = res["prob_transiciones_doc"]
    else:
        raise ValueError("tipo_prob debe ser 'cond' o 'doc'.")


    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G)

    pesos = [G[u][v]["peso"] for u, v in G.edges()]
    max_peso = max(pesos) if pesos else 1
    widths = [2 + 4*(p/max_peso) for p in pesos]

    etiquetas_nodos = {n: "".join(str(x) for x in n) for n in G.nodes()}

    etiquetas_aristas = {}
    for (u, v), p in prob_trans.items():
        if p >= umbral_prob:
            etiquetas_aristas[(u, v)] = f"{p:.2f}"

    plt.figure(figsize=(8, 6))

    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="#ffefc2")
    nx.draw_networkx_edges(G, pos, width=widths, arrows=True,
                           arrowstyle="->", alpha=0.8)
    nx.draw_networkx_labels(G, pos, labels=etiquetas_nodos, font_size=12)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=etiquetas_aristas,
                                 font_size=10, label_pos=0.5)

    titulo_prob = "P(b|a)" if tipo_prob == "cond" else "P(a→b) (doc)"
    plt.title(f"Red ordinal (idx={idx}) - etiquetas: {titulo_prob}")
    plt.axis("off")
    plt.show()

"""### Transiciones y probabilidades"""

resultados = []

for idx, tokens in tqdm(df["Tokens"].items(), total=len(df),desc="Procesando documentos"):
    lengths = [len(t) for t in tokens]
    patterns = ordinal_patterns_blocks(lengths, D=3)
    res = red_ordinal_causal(patterns)

    for pat in res["conteo_patrones"]:
        resultados.append({
            "doc_id": idx,
            "pattern": pat,
            "count": res["conteo_patrones"][pat],
            "in_strength": res["in_strength"][pat],
            "out_strength": res["out_strength"][pat]})

df_patterns = pd.DataFrame(resultados)
df_patterns

result_transitions = []

for idx, tokens in tqdm(df["Tokens"].items(), total=len(df), desc="Transiciones"):

    lengths = [len(t) for t in tokens]
    patterns = ordinal_patterns_blocks(lengths, D=3)

    res = red_ordinal_causal(patterns)

    total_doc = sum(res["conteo_transiciones"].values())  # total transiciones del doc

    for (a, b), w in res["conteo_transiciones"].items():
        result_transitions.append({
            "doc_id": idx,
            "from": a,
            "to": b,
            "count": w,
            "prob": (w / total_doc) if total_doc > 0 else 0.0
        })

df_transitions = pd.DataFrame(result_transitions)
df_transitions

matrices = (df_transitions.pivot_table(index="doc_id",columns=["from", "to"],values="prob",fill_value=0))
matrices.columns = [f"prob_{pat_to_str(a)}_{pat_to_str(b)}" for (a, b) in matrices.columns]

matrices

df_pats_clean = df_pats.drop(columns=['fuente', 'Autor', 'Titulo', 'Contenido',
                                      'Limpio', 'Tokens', 'Palabras'])

df_transicion = pd.concat([df_e1.reset_index(drop=True),
                       df_pats_clean.reset_index(drop=True)], axis=1)
df_transicion = df_transicion.drop(columns=['Unnamed: 0'])
df_transicion

df1 = df_transicion.merge( matrices, left_on="idx",right_index=True, how="left")
df1.head()

dibujar_red_ordinal_por_idx(df, idx=0, D=3, tipo_prob="doc", umbral_prob=0.0)

"""## 6.2. Enfoque por frecuencia de palabra

### Funciones
"""

from collections import Counter

def serie_tf(tokens, relativa=True):

    conteos = Counter(tokens)
    N = len(tokens)

    if relativa:
        return [conteos[t] / N for t in tokens]
    else:
        return [conteos[t] for t in tokens]

def dibujar_red_ordinal_tf_por_idx(df, idx, D=3, layout="spring",
                                  umbral_prob=None, tipo_prob="doc"):

    tokens = df.loc[idx, "Tokens"]
    if not tokens or len(tokens) < D:
        print("Documento demasiado corto.")
        return

    tf_series = serie_tf(tokens, relativa=True)  # o relativa=False

    patrones = ordinal_patterns(tf_series, D=D)
    if len(patrones) < 2:
        print("Muy pocos patrones para construir la red.")
        return

    res = red_ordinal_causal(patrones)
    G = res["grafo"]

    if tipo_prob == "cond":
        prob_trans = res["prob_transiciones_cond"]
        if umbral_prob is None:
            umbral_prob = 0.2
    elif tipo_prob == "doc":
        prob_trans = res["prob_transiciones_doc"]
        if umbral_prob is None:
            umbral_prob = 0.02
    else:
        raise ValueError("tipo_prob debe ser 'cond' o 'doc'.")

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G)

    pesos = [G[u][v]["peso"] for u, v in G.edges()]
    max_peso = max(pesos) if pesos else 1
    widths = [2 + 4*(p/max_peso) for p in pesos]

    etiquetas_nodos = {n: "".join(str(x) for x in n) for n in G.nodes()}

    etiquetas_aristas = {
        (u, v): f"{p:.2f}"
        for (u, v), p in prob_trans.items()
        if p >= umbral_prob
    }

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="#d7f0ff")
    nx.draw_networkx_edges(G, pos, width=widths, arrows=True,
                           arrowstyle="->", alpha=0.8)
    nx.draw_networkx_labels(G, pos, labels=etiquetas_nodos, font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=etiquetas_aristas,
                                 font_size=10, label_pos=0.5)

    titulo_prob = "P(b|a)" if tipo_prob == "cond" else "P(a→b) (doc)"
    plt.title(f"Red ordinal (TF palabras) - idx={idx} - etiquetas: {titulo_prob}")
    plt.axis("off")
    plt.show()

"""### Transiciones y probabilidades"""

dibujar_red_ordinal_tf_por_idx(df, idx=0, D=3, tipo_prob="cond")

dibujar_red_ordinal_tf_por_idx(df, idx=0, D=3, tipo_prob="doc")

resultados_tf = []
result_transitions_tf = []

for idx, tokens in tqdm(df["Tokens"].items(), total=len(df), desc="Docs TF"):

    tf_series = serie_tf(tokens, relativa=True)
    patrones = ordinal_patterns_blocks(tf_series, D=3)

    if len(patrones) < 2:
        continue

    res = red_ordinal_causal(patrones)

    for pat, c in res["conteo_patrones"].items():
        resultados_tf.append({
            "doc_id": idx,
            "pattern": pat,
            "count": c,
            "in_strength": res["in_strength"][pat],
            "out_strength": res["out_strength"][pat]
        })

    for (a, b), w in res["conteo_transiciones"].items():
        result_transitions_tf.append({
            "doc_id": idx,
            "from": a,
            "to": b,
            "count": w,
            "prob": res["prob_transiciones_doc"][(a, b)]
        })

df_patterns_tf = pd.DataFrame(resultados_tf)
df_patterns_tf

df_transitions_tf = pd.DataFrame(result_transitions_tf)
df_transitions_tf

matrices2 = (df_transitions_tf.pivot_table(index="doc_id",columns=["from", "to"],values="prob",fill_value=0))
matrices2.columns = [f"prob_{pat_to_str(a)}_{pat_to_str(b)}" for (a, b) in matrices2.columns]
matrices2

df_pats_clean2 = df_tf.drop(columns=['fuente', 'Autor', 'Titulo', 'Contenido',
                                      'Limpio', 'Tokens', 'Palabras'])

df_transicion2 = pd.concat([df_e2.reset_index(drop=True),
                       df_pats_clean2.reset_index(drop=True)], axis=1)
df_transicion2 = df_transicion2.drop(columns=['Unnamed: 0'])
df_transicion2

df2 = df_transicion2.merge( matrices2, left_on="idx",right_index=True, how="left")
df2.head()

"""# 7. Matrices de transición por fuente"""

PATRONES = ["012", "021", "102", "120", "201", "210"]
PROB_PREFIX = "prob_"

def get_prob_cols(df, prefix=PROB_PREFIX):
    prob_cols = [c for c in df.columns if c.startswith(prefix)]
    if len(prob_cols) == 0:
        raise ValueError(f"No encontré columnas que empiecen con '{prefix}'.")
    return prob_cols


def promedio_transiciones_por_fuente(df, prob_cols):
    df_fuente = df.groupby("fuente")[prob_cols].mean()
    return df_fuente

prob_cols = get_prob_cols(df1, prefix=PROB_PREFIX)
df_fuente = promedio_transiciones_por_fuente(df1, prob_cols)

prob_cols = get_prob_cols(df2, prefix=PROB_PREFIX)
df_fuente2 = promedio_transiciones_por_fuente(df2, prob_cols)

df_fuente.to_csv('matriz_enfoque1.csv')
df_fuente2.to_csv('matriz_enfoque2.csv')

"""# 8. Modelos de aprendizaje automático"""

df1 = df1.drop_duplicates(subset=['Contenido'], keep='first')

"""## 8.1. Preparación de características"""

cols_num = df1.select_dtypes(include=[np.number]).columns
cols_num = [c for c in cols_num if c not in ["index", "idx"]]

X = df1[cols_num].fillna(0)
X.shape

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""## 8.2. Funciones de visualización"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

def pca_facet_by_cluster_plotly(X_scaled, labels, df=None, hover_cols=None,
                                title="PCA 2D por cluster (facets)", jitter=0.0):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    Xp = X_pca.copy()
    if jitter > 0:
        rng = np.random.default_rng(42)
        Xp += rng.normal(0, jitter, size=Xp.shape)

    plot_df = pd.DataFrame({"PC1": Xp[:,0], "PC2": Xp[:,1], "cluster": labels.astype(str)})
    if df is not None and hover_cols:
        for c in hover_cols:
            if c in df.columns:
                plot_df[c] = df[c].values

    fig = px.scatter(
        plot_df, x="PC1", y="PC2",
        facet_col="cluster", facet_col_wrap=4,
        hover_data=hover_cols if (df is not None and hover_cols) else None,
        title=title
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("cluster=", "Cluster ")))
    fig.show()

"""## 8.3. Agrupamiento con K-means"""

ks = range(2, 11)
scores = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(6,4))
plt.plot(list(ks), scores, marker="o")
plt.xticks(list(ks))
plt.xlabel("K (número de clusters)")
plt.ylabel("Silhouette score")
plt.title("Silhouette vs K")
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.show()

K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto', init='k-means++')
df1['cluster_kmeans'] = kmeans.fit_predict(X_scaled)

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

def pca_scatter_plotly(X_scaled, labels, df=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": labels.astype(str)
    })

    import plotly.express as px

    magma3 = px.colors.sample_colorscale(
        "Magma",
        [0.15, 0.50, 0.85]
    )

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        color_discrete_sequence=magma3
    )

    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.85,
            line=dict(width=0.5, color="black")
        )
    )

    fig.update_layout(
    template="simple_white",
    width=750,
    height=550,
    font=dict(size=14),
    showlegend=False,
    margin=dict(l=60, r=30, t=20, b=60)
)

    fig.update_xaxes(
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False
    )
    fig.update_xaxes(
    range=[-10, 16],
    showgrid=True,
    gridcolor="lightgray",
    zeroline=False
)

    fig.update_yaxes(
        range=[-8, 8],
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False
    )

    fig.update_layout(showlegend=False)
    fig.show()

pca_scatter_plotly(X_scaled, labels=df1["cluster_kmeans"], df=df1)

df1["cluster_kmeans"] = kmeans.fit_predict(X_scaled)
import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

labels = df1["cluster_kmeans"].to_numpy()

# Métricas globales de calidad del clustering
silhouette = silhouette_score(X_scaled, labels)
davies_bouldin = davies_bouldin_score(X_scaled, labels)
calinski_harabasz = calinski_harabasz_score(X_scaled, labels)

# Inercia total: suma de cuadrados de las distancias al centroide
wcss = kmeans.inertia_

# WCSS promedio por observación
wcss_promedio = wcss / len(X_scaled)

# Distancia de cada observación a su centroide
centroides = kmeans.cluster_centers_

distancias_centroide = np.linalg.norm(
    X_scaled - centroides[labels],
    axis=1
)

df1["distancia_centroide"] = distancias_centroide

distancia_media_global = distancias_centroide.mean()
distancia_mediana_global = np.median(distancias_centroide)

# Compactación por cluster
compactacion_cluster = (
    df1.groupby("cluster_kmeans")["distancia_centroide"]
       .agg(
           n="count",
           distancia_media="mean",
           distancia_mediana="median",
           desviacion_estandar="std",
           distancia_maxima="max"
       )
       .reset_index()
)

resumen_enfoque = pd.DataFrame({
    "metrica": [
        "Silhouette score",
        "Davies-Bouldin index",
        "Calinski-Harabasz score",
        "WCSS",
        "WCSS promedio por observación",
        "Distancia media al centroide",
        "Distancia mediana al centroide"
    ],
    "valor": [
        silhouette,
        davies_bouldin,
        calinski_harabasz,
        wcss,
        wcss_promedio,
        distancia_media_global,
        distancia_mediana_global
    ]
})

print("Métricas globales:")
display(resumen_enfoque.round(4))

print("\nCompactación por cluster:")
display(compactacion_cluster.round(4))

pca_facet_by_cluster_plotly(X_scaled, labels=df1["cluster_kmeans"], df=df1, hover_cols=["Titulo"], jitter=0.01)

df1.to_csv('df1_kmeans.csv', index=False)

"""## 8.4. Clasificación supervisada"""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import loguniform

y = df1["cluster_kmeans"].astype(str)

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

"""### Clasificador base"""

from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)
pred = dummy.predict(X_test)
print("TEST accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

"""### Regresión logística"""

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000))])

param_grid_lr = {
    "clf__C": np.logspace(-4, 3, 20),
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs", "saga"]}

grid_lr = GridSearchCV(
    pipe_lr,
    param_grid=param_grid_lr,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1)

grid_lr.fit(X_train, y_train)

print("Mejor LR params:", grid_lr.best_params_)
print("CV accuracy:", grid_lr.best_score_)

best_lr = grid_lr.best_estimator_
pred = best_lr.predict(X_test)
print("TEST accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

"""### Random forest"""

rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced")

param_dist_rf = {
    "n_estimators": randint(300, 1200),
    "max_depth": [None] + list(range(5, 41, 5)),
    "min_samples_split": randint(2, 30),
    "min_samples_leaf": randint(1, 20),
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]}

rand_rf = RandomizedSearchCV(
    rf,
    param_distributions=param_dist_rf,
    n_iter=80,
    scoring="accuracy",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rand_rf.fit(X_train, y_train)

print("Mejor RF params:", rand_rf.best_params_)
print("CV accuracy:", rand_rf.best_score_)

best_rf = rand_rf.best_estimator_
pred = best_rf.predict(X_test)
print("TEST accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

from sklearn.metrics import confusion_matrix
pred = best_rf.predict(X_test)
cm = confusion_matrix(y_test, pred)

labels = le.classes_
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

cm_off = cm_df.copy()
np.fill_diagonal(cm_off.values, 0)

pairs_df = (cm_off.stack().reset_index().rename(columns={"level_0": "real", "level_1": "pred", 0: "count"}).sort_values("count", ascending=False))
pairs_df.head(20)

importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
importances.head(15).plot(kind="bar", title="Importancia de características")
plt.show()

"""### Perceptrón multicapa (MLP)"""

le = LabelEncoder()
y = le.fit_transform(df1["cluster_kmeans"].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier


mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=25,
    random_state=42
)

mlp.fit(X_train_s, y_train)

pred = mlp.predict(X_test_s)

print("MLP accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=le.classes_))

plt.figure(figsize=(6,4))
plt.plot(mlp.loss_curve_, marker="o", linewidth=1)
plt.xlabel("Iteración")
plt.ylabel("Loss")
plt.title("Curva de pérdida (MLP)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from scipy.stats import loguniform

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_dist = {
    "hidden_layer_sizes": [(32,), (64,), (64,32), (128,64), (128,64,32)],
    "alpha": loguniform(1e-6, 1e-2),
    "learning_rate_init": loguniform(1e-4, 3e-2),
    "activation": ["relu", "tanh"]
}

mlp_base = MLPClassifier(
    solver="adam",
    max_iter=3000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=25,
    random_state=42
)

search = RandomizedSearchCV(
    mlp_base,
    param_distributions=param_dist,
    n_iter=30,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X_train_s, y_train)

print("Mejores params:", search.best_params_)
print("CV accuracy:", search.best_score_)

best_mlp = search.best_estimator_

pred = best_mlp.predict(X_test_s)

print("Best MLP TEST accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=le.classes_))

plt.figure(figsize=(6,4))
plt.plot(best_mlp.loss_curve_, marker="o", linewidth=1)
plt.xlabel("Iteración")
plt.ylabel("Loss")
plt.title("Curva de pérdida (Best MLP)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

from sklearn.metrics import f1_score, balanced_accuracy_score
pred = best_mlp.predict(X_test_s)
print("Balanced acc:", balanced_accuracy_score(y_test, pred))
print("Macro F1:", f1_score(y_test, pred, average="macro"))

from sklearn.metrics import top_k_accuracy_score
proba = best_mlp.predict_proba(X_test_s)
print("Top 3 accuracy:", top_k_accuracy_score(y_test, proba, k=3))

pred = best_mlp.predict(X_test_s)
cm = confusion_matrix(y_test, pred)

labels = le.classes_
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

cm_off = cm_df.copy()
np.fill_diagonal(cm_off.values, 0)

pairs_df = (cm_off.stack().reset_index().rename(columns={"level_0": "real", "level_1": "pred", 0: "count"}).sort_values("count", ascending=False))
pairs_df.head(20)