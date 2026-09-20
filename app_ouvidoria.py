"""
OuviSearch - Triagem Semantica de Manifestacoes Cidadas
=========================================================
Desafio Pratico - NLP Aplicado - UNIPE
Disciplina: Tendências em Ciência da Computação
Professor: Me. Ricardo Roberto de Lima
Aluno: Pedro Nicollas Pereira Leon Lopes

Execucao:
    pip install -r requirements.txt
    streamlit run app_ouvidoria.py

Entrega 4 do desafio - app com 4 abas:
    Busca Semantica  - consulta livre, top-k manifestacoes mais similares.
    Base Completa    - tabela com as 40 manifestacoes e matriz de similaridade.
    Espaco Vetorial  - projecao 2D (PCA/t-SNE) colorida por categoria oficial.
    Chunking         - RecursiveCharacterTextSplitter interativo sobre um texto longo.

Todas as fases usam embeddings dedensos via sentence-transformers quando disponivel;
se a biblioteca/modelo nao puder ser carregada, o app cai automaticamente em um modo
de simulacao documentado (TF-IDF + SVD/LSA), sem travar a interface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ==========================================================================
# CARREGAMENTO DO CORPUS
# ==========================================================================

DATA_PATH = Path(__file__).parent / "data" / "manifestacoes.json"


@st.cache_data(show_spinner=False)
def carregar_corpus(caminho: str) -> list[dict]:
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)["manifestacoes"]


CORPUS: list[dict] = carregar_corpus(str(DATA_PATH))
IDS = [m["id"] for m in CORPUS]
TEXTOS = [m["texto"] for m in CORPUS]
CATEGORIAS = [m["categoria_oficial"] for m in CORPUS]
N_DOCS = len(CORPUS)

# Pares de duplicatas reais conhecidos, definidos na criacao do corpus sintetico
DUPLICATAS_REAIS = [("M003", "M017"), ("M008", "M022"), ("M014", "M029")]

MODELOS_DISPONIVEIS = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-pt-v1.5",
]

# ==========================================================================
# EMBEDDINGS (modelo real com fallback documentado TF-IDF + SVD)
# ==========================================================================


@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def carregar_modelo(nome_modelo: str):
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(nome_modelo)
    except Exception:
        return None


@st.cache_data(show_spinner="Gerando embeddings...")
def gerar_embeddings(textos: tuple[str, ...], nome_modelo: str) -> tuple[np.ndarray, bool]:
    modelo = carregar_modelo(nome_modelo)
    if modelo is not None:
        vetores = modelo.encode(list(textos), normalize_embeddings=True)
        return np.asarray(vetores), True

    # Fallback: TF-IDF + SVD (LSA), normalizado por linha (L2) - simula uma
    # representacao densa sem depender de download de modelo pre-treinado.
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode")
    tfidf_matrix = vectorizer.fit_transform(textos)
    n_components = min(30, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=max(n_components, 2), random_state=42)
    denso = svd.fit_transform(tfidf_matrix)
    normas = np.linalg.norm(denso, axis=1, keepdims=True)
    normas[normas == 0] = 1
    return denso / normas, False


def cor_score(score: float) -> str:
    if score > 0.7:
        return "🟢"
    if score > 0.5:
        return "🟡"
    return "🔴"


# ==========================================================================
# INTERFACE STREAMLIT
# ==========================================================================

st.set_page_config(page_title="OuviSearch - Triagem Semantica", page_icon="📮", layout="wide")

st.title("📮 OuviSearch - Triagem Semântica de Manifestações Cidadãs")
st.caption(
    "Busca semântica + detecção de duplicatas + chunking - Desafio Prático NLP Aplicado (UNIPÊ)"
)

# ------------------------------ Sidebar --------------------------------
st.sidebar.header("Configurações")
modelo_nome = st.sidebar.selectbox("Modelo de Embedding", MODELOS_DISPONIVEIS)
top_k = st.sidebar.slider("Top-K resultados", min_value=1, max_value=10, value=5)

embeddings_corpus, usou_embeddings_reais = gerar_embeddings(tuple(TEXTOS), modelo_nome)

if usou_embeddings_reais:
    st.sidebar.success(f"Embeddings reais: `{modelo_nome}`", icon="✅")
else:
    st.sidebar.info(
        "Modo de simulação vetorial (TF-IDF + SVD) - `sentence-transformers` "
        "indisponível neste ambiente. A interface continua funcional.",
        icon="ℹ️",
    )

st.sidebar.markdown("---")
st.sidebar.caption(f"{N_DOCS} manifestações carregadas de `data/manifestacoes.json`.")

tab_busca, tab_base, tab_espaco, tab_chunking = st.tabs(
    ["🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking"]
)

# --------------------------- ABA: BUSCA SEMÂNTICA -------------------------
with tab_busca:
    st.subheader("Busca Semântica")
    st.caption("Digite uma descrição livre do problema e veja as manifestações mais parecidas.")

    exemplos = {
        "(nenhum)": "",
        "Buraco na via": "tem um buraco grande atrapalhando o trânsito na minha rua",
        "Falta de médico": "não tem médico no posto de saúde perto de casa",
        "Insegurança": "muitos assaltos e falta de policiamento no bairro",
    }
    exemplo = st.selectbox("Consulta de exemplo:", list(exemplos.keys()))
    query = st.text_input("Descreva o problema:", value=exemplos[exemplo])

    if query.strip():
        query_emb, _ = gerar_embeddings((query,) + tuple(TEXTOS), modelo_nome)
        # a primeira linha corresponde à query; o restante, ao corpus, na mesma ordem
        sims = cosine_similarity(query_emb[:1], query_emb[1:])[0]
        ranking = np.argsort(sims)[::-1][:top_k]

        st.markdown(f"### Top-{top_k} manifestações mais similares")
        for pos, idx in enumerate(ranking, start=1):
            score = float(sims[idx])
            doc = CORPUS[idx]
            with st.expander(
                f"{cor_score(score)} #{pos} — {doc['id']} ({doc['categoria_oficial']}) — score {score:.3f}"
            ):
                st.write(doc["texto"])
                st.caption(f"Data: {doc['data']}")
    else:
        st.info("Digite uma consulta ou escolha um exemplo para ver os resultados.")

# --------------------------- ABA: BASE COMPLETA ---------------------------
with tab_base:
    st.subheader("Base Completa de Manifestações")
    df_corpus = pd.DataFrame(CORPUS)
    st.dataframe(df_corpus, use_container_width=True, hide_index=True)

    st.markdown("---")
    if st.button("🔢 Gerar Matriz de Similaridade"):
        with st.spinner("Calculando similaridade entre todas as manifestações..."):
            matriz = cosine_similarity(embeddings_corpus)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            matriz, xticklabels=IDS, yticklabels=IDS, cmap="YlOrRd",
            vmin=0, vmax=1, square=True, cbar_kws={"label": "Similaridade de cosseno"}, ax=ax,
        )
        ax.set_title("Matriz de Similaridade de Cosseno - 40 Manifestações")
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(fontsize=6)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("#### Pares de duplicatas reais conhecidos (ground truth do dataset)")
        linhas = []
        for a, b in DUPLICATAS_REAIS:
            i, j = IDS.index(a), IDS.index(b)
            linhas.append({"Par": f"{a} × {b}", "Similaridade": round(float(matriz[i, j]), 4)})
        st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)

# --------------------------- ABA: ESPAÇO VETORIAL -------------------------
with tab_espaco:
    st.subheader("Espaço Vetorial das Manifestações")
    reducao = st.radio("Redução de dimensionalidade:", ["PCA", "t-SNE"], horizontal=True)

    if reducao == "PCA":
        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings_corpus)
    else:
        perplexidade = min(10, N_DOCS - 1)
        coords = TSNE(
            n_components=2, perplexity=perplexidade, random_state=42, init="pca"
        ).fit_transform(embeddings_corpus)

    df_coords = pd.DataFrame(coords, columns=["x", "y"])
    df_coords["id"] = IDS
    df_coords["categoria"] = CATEGORIAS

    fig, ax = plt.subplots(figsize=(10, 7))
    categorias_unicas = sorted(set(CATEGORIAS))
    palette = dict(zip(categorias_unicas, sns.color_palette("tab10", len(categorias_unicas))))
    for cat in categorias_unicas:
        sub = df_coords[df_coords["categoria"] == cat]
        ax.scatter(sub["x"], sub["y"], label=cat, s=90, color=palette[cat], edgecolor="black")
    for _, row in df_coords.iterrows():
        ax.annotate(row["id"], (row["x"], row["y"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"Manifestações no Espaço Semântico ({reducao}), coloridas por categoria oficial")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    try:
        codigos_categoria = pd.Categorical(CATEGORIAS).codes
        score = silhouette_score(embeddings_corpus, codigos_categoria)
        st.metric("Silhouette Score (categorias oficiais)", f"{score:.3f}")
        if score > 0.15:
            leitura = "os clusters semânticos coincidem razoavelmente com as categorias oficiais."
        elif score > 0.0:
            leitura = "há alguma coincidência entre clusters semânticos e categorias, mas fraca."
        else:
            leitura = "os clusters semânticos não coincidem claramente com as categorias oficiais."
        st.write(
            f"**Interpretação:** com um Silhouette Score de {score:.3f}, {leitura} "
            "Um score próximo de 1 indicaria separação perfeita entre categorias; próximo de 0 "
            "ou negativo indica sobreposição - esperado aqui, já que várias manifestações de "
            "categorias diferentes compartilham vocabulário (ex.: infraestrutura e meio ambiente "
            "ambas mencionam 'esgoto')."
        )
    except Exception:
        st.info("Não foi possível calcular o Silhouette Score para esta configuração.")

# --------------------------- ABA: CHUNKING --------------------------------
with tab_chunking:
    st.subheader("Chunking de Manifestações Longas")

    manifestacao_mais_longa = max(CORPUS, key=lambda d: len(d["texto"]))
    texto_input = st.text_area(
        "Cole uma manifestação longa (ou use o exemplo padrão):",
        value=manifestacao_mais_longa["texto"],
        height=180,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        estrategia = st.selectbox("Estratégia:", ["RecursiveCharacter", "Fixed-Size (Character)"])
    with col2:
        chunk_size = st.slider("Chunk Size (caracteres)", 50, 500, 150, 10)
    with col3:
        chunk_overlap = st.slider("Overlap (caracteres)", 0, 200, 50, 10)

    if st.button("🧩 Gerar Chunks"):
        if not texto_input.strip():
            st.warning("Cole um texto para gerar os chunks.")
        else:
            if estrategia == "RecursiveCharacter":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
            else:
                splitter = CharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" "
                )
            chunks = splitter.split_text(texto_input)
            st.success(f"{len(chunks)} chunks gerados com a estratégia **{estrategia}**.")

            for i, c in enumerate(chunks, start=1):
                with st.expander(f"Chunk {i} ({len(c)} caracteres)"):
                    st.write(c)

            if len(chunks) > 1:
                chunk_embeddings, _ = gerar_embeddings(tuple(chunks), modelo_nome)
                st.markdown(f"**Dimensão dos embeddings:** {chunk_embeddings.shape}")

                sim_chunks = cosine_similarity(chunk_embeddings)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    sim_chunks, xticklabels=[f"C{i+1}" for i in range(len(chunks))],
                    yticklabels=[f"C{i+1}" for i in range(len(chunks))],
                    cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax,
                )
                ax.set_title("Similaridade entre Chunks")
                st.pyplot(fig)

                coords_chunks = PCA(n_components=2, random_state=42).fit_transform(chunk_embeddings)
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                ax2.scatter(coords_chunks[:, 0], coords_chunks[:, 1], s=100, c=range(len(chunks)), cmap="viridis")
                for i, (x, y) in enumerate(coords_chunks):
                    ax2.annotate(f"C{i+1}", (x, y), xytext=(5, 5), textcoords="offset points")
                ax2.set_title("Chunks no Espaço 2D (PCA)")
                ax2.grid(alpha=0.3)
                st.pyplot(fig2)

st.markdown("---")
st.caption(
    "OuviSearch - Desafio Prático NLP Aplicado - UNIPÊ Centro Universitário de João Pessoa · "
    "Prof. Me. Ricardo Roberto de Lima - Aluno: Pedro Nícollas Pereira Leon Lopes"
)
