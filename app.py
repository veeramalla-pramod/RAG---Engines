"""Dynamic RAG Explorer & Telemetry Engine (Advanced RAG Studio)
Interactive Streamlit Laboratory with Real-Time Indexing Hyperparameter Tuning & Pipeline Telemetry.
"""

import os
import streamlit as st
import numpy as np
import pandas as pd

from core.chunker import SlidingWindowChunker, Chunk
from core.tokenizer import count_tokens, calculate_token_telemetry, compute_context_budget
from core.embedder import TFIDFEmbedder, DenseEmbedder
from core.retriever import VectorRetriever
from core.visualizer import (
    create_vector_space_plot,
    create_score_distribution_plot,
    create_context_budget_gauge,
)
from core.generator import GroundedGenerator, PERSONAS

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Advanced RAG Studio | Dynamic Telemetry Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    .response-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 14px;
        padding: 22px;
        margin-top: 15px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .badge-persona {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
    
    .badge-provider {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.4);
        margin-left: 8px;
    }
    
    .diff-unique-a {
        background-color: rgba(59, 130, 246, 0.15);
        border-left: 3px solid #3b82f6;
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    
    .diff-shared {
        background-color: rgba(245, 158, 11, 0.22);
        border-left: 3px solid #f59e0b;
        padding: 8px 12px;
        border-radius: 4px;
        font-weight: 500;
        margin-bottom: 6px;
    }
    
    .diff-unique-b {
        background-color: rgba(16, 185, 129, 0.15);
        border-left: 3px solid #10b981;
        padding: 8px 12px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Corpus Data
# ---------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_corpus(filename: str) -> str:
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ---------------------------------------------------------
# Sidebar: Domain, Hyperparameters & Settings
# ---------------------------------------------------------
st.sidebar.markdown("### 🛠️ RAG Engineering Studio")

# 1. Use Case Selector
domain_choice = st.sidebar.selectbox(
    "Target Domain & Persona",
    options=["Use Case A: Exam Study Buddy", "Use Case B: E-Commerce Support Bot"],
    index=0,
    help="Switches active knowledge corpus and persona grounding prompt."
)

if "Study Buddy" in domain_choice:
    persona_key = "study_buddy"
    corpus_file = "os_study_guide.txt"
    default_query = "Which scheduling algorithm causes the convoy effect and why?"
    domain_badge = "Exam Study Buddy (Dense Technical OS Domain)"
else:
    persona_key = "ecommerce"
    corpus_file = "ecommerce_policy.txt"
    default_query = "Can I return a backpack after 20 days for a refund?"
    domain_badge = "E-Commerce Support Bot (Structured Policy Domain)"

raw_text = load_corpus(corpus_file)

# 2. Embedding Engine Representation
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧮 Vector Representation")
embedder_mode = st.sidebar.radio(
    "Embedding Mode",
    options=["Sparse TF-IDF (Exploratory)", "Dense Transformer (Semantic)"],
    index=1,
    help="Sparse TF-IDF exposes feature vocabulary and sparsity; Dense Transformer provides continuous latent embeddings."
)

# 3. Dynamic Indexing Hyperparameters
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Indexing Hyperparameters")

chunk_size = st.sidebar.slider(
    "Chunk Size (Words)",
    min_value=20,
    max_value=300,
    value=80,
    step=10,
    help="Configures the sliding window chunk length in words."
)

overlap_pct = st.sidebar.slider(
    "Overlap Ratio (%)",
    min_value=0,
    max_value=50,
    value=20,
    step=5,
    help="Configures chunk overlap percentage [0% to 50%]."
) / 100.0

top_k = st.sidebar.slider(
    "Top-K Passages",
    min_value=1,
    max_value=10,
    value=3,
    help="Maximum candidate passages passed to the generation context."
)

similarity_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.15,
    step=0.05,
    help="Suppresses chunks scoring below this cosine similarity threshold."
)

# 4. Optional API Key
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 Inference Engine")
api_key_input = st.sidebar.text_input(
    "Anthropic API Key (Optional)",
    type="password",
    help="Leave blank to use High-Fidelity Local Grounded Synthesizer (Zero outside hallucination)."
)

# ---------------------------------------------------------
# Dynamic Ingestion & Indexing Pipeline
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_dense_embedder_instance():
    return DenseEmbedder()

chunker = SlidingWindowChunker(mode="words")
chunks = chunker.chunk_text(
    raw_text,
    chunk_size=chunk_size,
    overlap_percentage=overlap_pct,
    token_counter=count_tokens
)

# Instantiate Embedder
if "TF-IDF" in embedder_mode:
    embedder = TFIDFEmbedder()
    doc_matrix = embedder.fit_transform([c.text for c in chunks])
else:
    embedder = get_dense_embedder_instance()
    doc_matrix = embedder.fit_transform([c.text for c in chunks])

retriever = VectorRetriever()
retriever.index(chunks, doc_matrix)

# Telemetry data
token_telemetry = calculate_token_telemetry(raw_text, chunks)
embedding_telemetry = embedder.get_telemetry()

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown('<div class="main-title">Dynamic RAG Explorer & Telemetry Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Advanced RAG Studio • Live Indexing Hyperparameter Tuning • Pipeline Telemetry & Grounding Verification</div>',
    unsafe_allow_html=True
)

# Top Domain / Status Bar
st.markdown(f'<span class="badge-persona">{domain_badge}</span> <span class="badge-provider">Engine: {embedder_mode}</span>', unsafe_allow_html=True)

# Quick Test Scenario Buttons (TC-01 to TC-03)
st.markdown("#### 🧪 Preset Evaluation Scenarios")
scen_col1, scen_col2, scen_col3 = st.columns(3)

with scen_col1:
    if st.button("📌 TC-01: Convoy Effect (OS)", use_container_width=True):
        st.session_state["query_input"] = "Which scheduling algorithm causes the convoy effect and why?"

with scen_col2:
    if st.button("⚠️ TC-02: Memory Segmentation (Out-of-Domain)", use_container_width=True):
        st.session_state["query_input"] = "What is memory segmentation?"

with scen_col3:
    if st.button("📦 TC-03: 20-Day Backpack Refund", use_container_width=True):
        st.session_state["query_input"] = "Can I return a backpack after 20 days for a refund?"

# Query Input Field
initial_val = st.session_state.get("query_input", default_query)
query_text = st.text_input("💬 Enter Question or Query:", value=initial_val, key="active_query")

# Execute Search & Generation
q_vector = embedder.transform_query(query_text)
retrieved_results, all_scores = retriever.search(
    q_vector,
    top_k=top_k,
    threshold=similarity_threshold
)

generator = GroundedGenerator(api_key=api_key_input)
gen_result = generator.generate(
    query=query_text,
    retrieved_chunks=retrieved_results,
    persona_key=persona_key,
    custom_api_key=api_key_input
)

# ---------------------------------------------------------
# Generation & Grounded Output Panel
# ---------------------------------------------------------
st.markdown("### 🤖 Grounded Generation & Citation Output")

response_html = f"""
<div class="response-card">
    <div style="margin-bottom: 12px;">
        <span class="badge-persona">Persona: {PERSONAS[persona_key]['name']}</span>
        <span class="badge-provider">{gen_result['provider']}</span>
    </div>
    <div style="color: #f1f5f9; font-size: 1.05rem; line-height: 1.6; white-space: pre-wrap;">
{gen_result['response']}
    </div>
</div>
"""
st.markdown(response_html, unsafe_allow_html=True)

# Sources Citation Bar
if retrieved_results:
    st.markdown("##### 📚 Retrieved Context Citations:")
    citation_cols = st.columns(len(retrieved_results))
    for idx, (col, r) in enumerate(zip(citation_cols, retrieved_results)):
        with col:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px;">
                <div style="color: #34d399; font-weight: 700; font-size: 0.85rem;">[Source {r['chunk_id']}] • Rank #{r['rank']}</div>
                <div style="color: #94a3b8; font-size: 0.8rem; margin: 4px 0;">Similarity: <b>{r['score']:.4f}</b> | Margin: <b>{r['margin_to_next']:.4f}</b></div>
                <div style="color: #cbd5e1; font-size: 0.75rem; max-height: 80px; overflow-y: auto;"><i>"{r['snippet']}"</i></div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("⚠️ No chunks exceeded the similarity threshold. Fallback guardrail successfully suppressed generation.")

# ---------------------------------------------------------
# Deep Telemetry & Metrics Laboratory (5 Tabs)
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Pipeline Telemetry & Laboratory Diagnostics")

tab_tokens, tab_embeddings, tab_retrieval, tab_overlap, tab_corpus = st.tabs([
    "🔤 1. Token Telemetry",
    "🌐 2. Embedding Space & Topology",
    "🎯 3. Retrieval Metrics & Distribution",
    "🔍 4. Overlap & Diff Inspector",
    "📑 5. Chunk Corpus Explorer"
])

# ---------------------------------------------------------
# Tab 1: Token Telemetry
# ---------------------------------------------------------
with tab_tokens:
    st.markdown("#### Token Distribution & Context Window Budget")
    
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{token_telemetry['raw_token_count']:,}</div>
            <div class="metric-label">Raw Corpus Tokens</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_t2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{token_telemetry['total_chunk_tokens']:,}</div>
            <div class="metric-label">Total Indexed Tokens</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {'#34d399' if token_telemetry['inflation_ratio'] < 1.3 else '#f59e0b'};">
                {token_telemetry['inflation_ratio']:.2f}x
            </div>
            <div class="metric-label">Overlap Inflation Factor</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{token_telemetry['mean_chunk_tokens']}</div>
            <div class="metric-label">Mean Tokens / Chunk (Min: {token_telemetry['min_chunk_tokens']}, Max: {token_telemetry['max_chunk_tokens']})</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Calculate retrieved tokens
    retrieved_tokens_sum = sum(r["token_count"] for r in retrieved_results)
    budget_gauge_fig = create_context_budget_gauge(
        prompt_tokens=250,
        retrieved_tokens=retrieved_tokens_sum,
        completion_budget=500,
        max_context=4096
    )
    st.plotly_chart(budget_gauge_fig, use_container_width=True)

# ---------------------------------------------------------
# Tab 2: Embedding Space & Semantic Topology
# ---------------------------------------------------------
with tab_embeddings:
    st.markdown("#### Dimensionality, Sparsity & 2D Vector Projection")

    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(chunks)}</div>
            <div class="metric-label">Total Chunks (N)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_e2:
        dim_str = f"{embedding_telemetry['dimension']:,}" if embedding_telemetry['dimension'] else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{dim_str}</div>
            <div class="metric-label">Feature Vector Dim (D)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_e3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {'#38bdf8' if embedding_telemetry['sparsity_pct'] > 0 else '#34d399'};">
                {embedding_telemetry['sparsity_pct']}%
            </div>
            <div class="metric-label">Matrix Sparsity (% Zeroes)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_e4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{str(embedding_telemetry['total_vocabulary'])[:15]}</div>
            <div class="metric-label">Vocabulary Mode</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2D PCA Projection Plot
    pca_fig = create_vector_space_plot(
        doc_matrix=doc_matrix,
        query_vector=q_vector,
        chunks=chunks,
        retrieved_results=retrieved_results,
        query_text=query_text
    )
    st.plotly_chart(pca_fig, use_container_width=True)

    # If TF-IDF is active, display top feature keywords per retrieved chunk
    if isinstance(embedder, TFIDFEmbedder) and retrieved_results:
        st.markdown("##### 🏷️ Salient Feature Weights (Top-10 TF-IDF Terms in Retrieved Chunks)")
        kw_data = []
        for r in retrieved_results:
            top_kws = embedder.get_top_keywords_for_chunk(r["chunk_index"], top_n=6)
            kw_str = ", ".join([f"**{kw}** ({wt:.3f})" for kw, wt in top_kws])
            kw_data.append({"Chunk": f"Source {r['chunk_id']}", "Top Salient Terms": kw_str})
        st.table(pd.DataFrame(kw_data))

# ---------------------------------------------------------
# Tab 3: Retrieval Metrics & Distribution
# ---------------------------------------------------------
with tab_retrieval:
    st.markdown("#### Ranked Cosine Similarities & Separation from Noise Floor")

    # Ranked results table
    if retrieved_results:
        table_rows = []
        for r in retrieved_results:
            table_rows.append({
                "Rank": f"#{r['rank']}",
                "Chunk ID": f"Source {r['chunk_id']}",
                "Cosine Similarity": f"{r['score']:.4f}",
                "Margin to Next": f"{r['margin_to_next']:.4f}",
                "Word Count": r["word_count"],
                "Token Count": r["token_count"],
                "Snippet Preview": r["snippet"]
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No chunks matched the current threshold.")

    # Score Distribution Histogram
    hist_fig = create_score_distribution_plot(all_scores, threshold=similarity_threshold, chunks=chunks)
    st.plotly_chart(hist_fig, use_container_width=True)

# ---------------------------------------------------------
# Tab 4: Contiguous Chunk Overlap Diff Viewer
# ---------------------------------------------------------
with tab_overlap:
    st.markdown("#### Boundary Continuity & Overlap Diff Viewer")
    st.markdown(
        "Inspect how contiguous chunks ($C_i$ and $C_{i+1}$) share boundary sentences to prevent split-semantic information loss."
    )

    if len(chunks) > 1:
        pair_options = [f"Chunk {i+1} ↔ Chunk {i+2}" for i in range(len(chunks) - 1)]
        selected_pair_idx = st.selectbox("Select Contiguous Pair to Inspect:", range(len(pair_options)), format_func=lambda x: pair_options[x])

        chunk_a = chunks[selected_pair_idx]
        chunk_b = chunks[selected_pair_idx + 1]

        diff_info = SlidingWindowChunker.get_overlap_diff(chunk_a, chunk_b)

        st.markdown(f"**Overlap Word Count**: `{diff_info['overlap_words']} words`")

        col_diff_a, col_diff_b = st.columns(2)
        with col_diff_a:
            st.markdown(f"##### Chunk {chunk_a.chunk_id}")
            if diff_info["unique_a"]:
                st.markdown(f'<div class="diff-unique-a">{diff_info["unique_a"]}</div>', unsafe_allow_html=True)
            if diff_info["shared"]:
                st.markdown(f'<div class="diff-shared"><b>[SHARED BOUNDARY]</b>: {diff_info["shared"]}</div>', unsafe_allow_html=True)
            else:
                st.caption("No overlapping boundary text with next chunk (0% overlap).")

        with col_diff_b:
            st.markdown(f"##### Chunk {chunk_b.chunk_id}")
            if diff_info["shared"]:
                st.markdown(f'<div class="diff-shared"><b>[SHARED BOUNDARY]</b>: {diff_info["shared"]}</div>', unsafe_allow_html=True)
            if diff_info["unique_b"]:
                st.markdown(f'<div class="diff-unique-b">{diff_info["unique_b"]}</div>', unsafe_allow_html=True)
            else:
                st.caption("No unique content after boundary.")
    else:
        st.info("At least 2 chunks are needed to inspect boundary overlap.")

# ---------------------------------------------------------
# Tab 5: Chunk Corpus Explorer
# ---------------------------------------------------------
with tab_corpus:
    st.markdown(f"#### Complete Index Explorer ({len(chunks)} Chunks)")
    for c in chunks:
        with st.expander(f"Chunk {c.chunk_id} • ({c.word_count} words | {c.token_count} tokens)"):
            st.markdown(f"```text\n{c.text}\n```")
            if c.overlap_with_next:
                st.caption(f"Shared boundary with Chunk {c.chunk_id + 1}: *{c.overlap_with_next[:90]}...*")
