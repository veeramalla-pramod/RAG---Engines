# Dynamic RAG Explorer & Telemetry Engine (Advanced RAG Studio)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-3.0+-yellow.svg)](https://www.sbert.net/)

An interactive, production-grade Retrieval-Augmented Generation (RAG) engineering laboratory and observability studio. Built strictly according to the specifications in [PRD.md](PRD.md).

---

## 🚀 Key Features & Highlights

1. **Reactive Indexing Hyperparameter Tuning**:
   - **Chunk Size Slider ($N$)**: Adjust chunk length in real-time ($20$ to $300$ words).
   - **Overlap Ratio ($O$)**: Adjust sliding-window overlap ($0\%$ to $50\%$).
   - **Top-$K$ Candidate Selector**: Return $1$ to $10$ top context passages.
   - **Cosine Similarity Threshold**: Dynamic cutoff filter ($0.00$ to $1.00$) suppressing low-scoring semantic noise.

2. **Dual-Mode Vector Representation**:
   - **Base Mode (Exploratory / Sparse)**: Scikit-Learn `TfidfVectorizer` (with stop-word filtering) exposing feature vocabulary, dimensionality, and matrix sparsity.
   - **Advanced Mode (Semantic / Dense)**: Dense transformer embeddings (`all-MiniLM-L6-v2`) in continuous latent space.

3. **Dual Real-World Application Domains**:
   - **Use Case A: Exam Study Buddy**: Dense academic notes on Operating Systems & CPU scheduling (FCFS, SJF, Round Robin, Multilevel Feedback Queues).
   - **Use Case B: Mini E-Commerce Support Bot**: Structured store policy for ApexGear Direct (15-day refund window, 1-year warranty, price matching, luggage specs).

4. **Multi-Tier Live Telemetry Laboratory**:
   - **Token Telemetry**: Raw corpus tokens, indexed chunk tokens, overlap inflation factor $\left(\frac{\sum \text{Tokens in Chunks}}{\text{Tokens in Source Text}}\right)$, and LLM Context Window Budget Gauge.
   - **Embedding Space & Topology**: Matrix sparsity %, feature vector dimensions, 2D PCA vector scatter plot (mapping chunks, query vector, and top-K retrieved neighbors), and Top-10 salient keyword weights.
   - **Retrieval Metrics**: Ranked cosine similarity table with margin-to-next rank and corpus score distribution histogram with threshold marker.
   - **Contiguous Chunk Diff Inspector**: Visual diff highlighter displaying shared boundary text between consecutive chunks ($C_i \leftrightarrow C_{i+1}$) to verify boundary continuity.

5. **Strict Grounding & Citation Guardrails**:
   - Zero hallucination policy: If evidence is below the similarity threshold or out-of-domain, the engine enforces strict refusal.
   - Explicit inline citations (`[Source 1]`, `[Source 2]`) with interactive context preview cards.
   - Flexible multi-backend inference (Anthropic Claude, OpenAI, or the built-in deterministic local grounded engine).

---

## 📁 Repository Structure

```
RAG - Engines/
├── PRD.md                     # Product Requirements Document
├── README.md                  # System overview and quickstart guide
├── requirements.txt           # Python dependency manifest
├── run_app.bat                # 1-Click Windows execution script
├── app.py                     # Interactive Streamlit Laboratory Dashboard
│
├── core/                      # Modular Core RAG Engine
│   ├── __init__.py
│   ├── chunker.py             # Sliding-window chunker with boundary tracking & diff viewer
│   ├── tokenizer.py           # TikToken counter, inflation ratio & budget gauge logic
│   ├── embedder.py            # Dual-mode vectorizer (Sparse TF-IDF & Dense Transformer)
│   ├── retriever.py           # In-memory cosine similarity search and ranker
│   ├── visualizer.py          # Interactive Plotly 2D PCA plots, histograms, gauges
│   └── generator.py           # Grounded persona generator with citation extraction
│
├── data/                      # Dual-domain curated knowledge bases
│   ├── os_study_guide.txt     # Dense technical OS & CPU scheduling notes
│   └── ecommerce_policy.txt   # Structured store policies & product catalog
│
└── tests/                     # Automated Test Suite
    └── test_scenarios.py      # Verification tests for TC-01 through TC-05
```

---

## ⚡ Quickstart Guide

### 1. Run the Application
You can launch the application directly using the included batch file:
```cmd
run_app.bat
```
Or run directly in PowerShell / Command Prompt:
```powershell
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

### 2. Run the Automated Evaluation Test Suite
The test suite validates all evaluation scenarios specified in **PRD Section 7 (TC-01 through TC-05)**:
```powershell
python -m unittest discover -s tests
```

---

## 🧪 Evaluation Scenarios (PRD Section 7)

| Test Case | Scenario / Query | Expected Engine Behavior | Status |
| --- | --- | --- | --- |
| **TC-01 (Study Buddy)** | *"Which scheduling algorithm causes the convoy effect and why?"* | High similarity score for FCFS chunk; explains convoy effect with `[Source X]` citation. |  **PASSED** |
| **TC-02 (Study Buddy)** | *"What is memory segmentation?"* (Out-of-domain) | Score falls below relevance threshold; returns strict refusal without hallucination. |  **PASSED** |
| **TC-03 (E-Commerce)** | *"Can I return a backpack after 20 days for a refund?"* | Retrieves 15-day return policy chunk; refuses refund and cites 15-day window constraint. |  **PASSED** |
| **TC-04 (Dynamic Tuning)** | Drag chunk size slider ($150 \to 40$ words) | Chunk count dynamically multiplies; token gauge updates; PCA vector plot updates in real-time. |  **PASSED** |
| **TC-05 (Boundary Diff)** | Zero vs. 25% overlap comparison | Contiguous diff inspector shows zero shared words on $0\%$ overlap and highlights shared sentences on $>0\%$. |  **PASSED** |
