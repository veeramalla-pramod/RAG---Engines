# Product Requirements Document (PRD)

---

## 1. Executive Summary & Document Overview

* **Product Name**: Dynamic RAG Explorer & Telemetry Engine (Advanced RAG Studio)
* **Status**: Draft / Ready for Review
* **Target Audience**: AI/ML Engineers, Product Teams, Technical Interviewers, and Solutions Architects
* **Core Value Proposition**: An advanced, production-grade Retrieval-Augmented Generation (RAG) platform that elevates baseline RAG pipelines into an interactive engineering laboratory. The system features real-time tuning of indexing hyperparameters (chunk size and overlap), live telemetry across the full pipeline (Tokens, Embeddings, and Vectors), and dual real-world application domains.



---

## 2. Selected Use Cases

To demonstrate the adaptability of the unified architecture across distinct textual structures and domains, the engine natively implements two contrasting use cases:

### Use Case A: Exam Study Buddy (Dense Technical & Academic Domain)

* **Persona**: *"A patient, pedagogical study partner helping a student revise complex technical concepts through structured, step-by-step explanations."*

* **Content Characteristics**: High information density, formal terminology, definitions, algorithmic trade-offs (e.g., CPU Scheduling, Operating Systems notes).
* **Operational Sensitivity**: Requires smaller chunk windows to isolate single technical concepts and prevent cross-topic semantic dilution.



### Use Case B: Mini E-Commerce & Customer Support Bot (Structured Policy & Product Domain)

* **Persona**: *"A polite, policy-accurate customer support representative strictly enforcing store terms, guarantees, and product specifications."*

* **Content Characteristics**: Structured bullet points, numerical criteria (prices, dimensions, warranty duration, return cutoff windows).
* **Operational Sensitivity**: Requires precise preservation of numerical constraints and policy boundaries to prevent unauthorized concessions or incorrect return approvals.

---

## 3. High-Level Architecture

The system separates offline/dynamic indexing operations from real-time query execution, instrumenting every transition point with real-time telemetry:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             STREAMLIT DASHBOARD                             │
│  [Use Case Selector]  │  [Chunk Size Slider]  │  [Overlap Slider]  │  [Top-K]│
└───────┬──────────────────────────┬───────────────────────────┬──────────────┘
        │                          │                           │
        ▼                          ▼                           ▼
┌──────────────────┐    ┌─────────────────────┐    ┌──────────────────────────┐
│  Phase 1: Ingest │    │ Phase 2: Indexing   │    │ Phase 3: Query & LLM     │
│  & Tokenization  │───▶│ & Embedding Space   │───▶│ Augmentation & Generation│
└──────────────────┘    └─────────────────────┘    └──────────────────────────┘
        │                          │                           │
        └──────────────────────────┼───────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       REAL-TIME TELEMETRY PANEL                             │
│  • Token Metrics      │  • Embedding Space  │  • Vector Similarity & Ranking│
│  • Cost/Context Gauge │  • 2D PCA/t-SNE Plot│  • Citation Grounding Verification
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 4. Functional Requirements & Key Features

### 4.1. Dynamic Ingestion & Real-Time Hyperparameter Tuning

* **FR-1.1: Live Parameter Controls**:
* **Chunk Size Slider ($N$)**: Configurable from 20 words to 300 words (or 30 to 400 tokens) with real-time reactive re-chunking.


* **Overlap Slider ($O$)**: Configurable from 0% to 50% of the selected chunk size (word/token-based sliding window).


* **Top-$K$ Selector**: Configurable from 1 to 10 returned passages.


* **Similarity Threshold Filter**: Slider ($0.00$ to $1.00$) to suppress low-scoring noise.




* **FR-1.2: Instant State Synchronization**: Adjusting parameters immediately triggers chunk recreation, sparse/dense matrix updates, and visual state refresh without requiring a backend server restart.

### 4.2. Dual-Engine Embeddings & Similarity Engine

* **FR-2.1: Dual Mode Representation**:
* *Base Mode (Exploratory)*: Scikit-Learn `TfidfVectorizer` (with stop-word filtering) for transparent vocabulary analysis and sparsity observation.


* *Advanced Mode (Semantic)*: Transformer dense embeddings (e.g., `text-embedding-3-small` or `sentence-transformers/all-MiniLM-L6-v2`) for true semantic matching.




* **FR-2.2: Vector Storage & Nearest-Neighbor Search**:
* In-memory index calculating Cosine Similarity $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\Vert{}\mathbf{u}\Vert{}_2 \Vert{}\mathbf{v}\Vert{}_2}$ across all stored chunks.


* Dynamic ranking returning top-$K$ chunks with formatted confidence scores ($0.000$ to $1.000$).





### 4.3. Grounded Prompt Augmentation & Strict Generation

* **FR-3.1: Persona Injection**: Auto-switch prompt system instructions based on the active domain (Study Buddy vs. E-Commerce Support).


* **FR-3.2: Strict Grounding & Fallback Enforcement**: Prompts enforce zero outside-knowledge synthesis: *"Answer using ONLY the supplied context. If not found, explicitly reply that the information is unavailable"*.


* **FR-3.3: Inline Citation Mapping**: LLM generates explicit citations (e.g., `[Source 1]`, `[Source 2]`) mapped directly to retrieved chunks.



---

## 5. Telemetry & Metrics Specification (The Dynamic Dashboard)

The UI must present deep, inspectable metrics across the three core tiers:

### 5.1. Token Telemetry

* **Corpus & Chunk Token Counts**:
* Total token count of the raw corpus.
* Per-chunk token distribution (min, mean, max tokens per chunk).


* **Context Budget Gauge**:
* Real-time progress bar tracking token consumption within the target LLM's context window (Prompt overhead + Retrieved Chunks + Completion Budget).




* **Token Overhead & Inflation**:
* Calculates overlap redundancy factor: $\text{Inflation Ratio} = \frac{\sum \text{Tokens in Chunks}}{\text{Tokens in Source Raw Text}}$.



### 5.2. Embedding Telemetry

* **Dimensionality & Vocabulary Statistics**:
* TF-IDF vocabulary matrix dimension $(N_{\text{chunks}} \times D_{\text{features}})$ and matrix sparsity percentage.


* Dense vector length (e.g., 384 or 1536 dimensions).


* **Vocabulary Extraction**:
* Top-10 most heavily weighted feature keywords in each retrieved chunk.


* **Semantic Topology Visualization**:
* 2D projection scatter plot (via PCA or t-SNE) plotting all document chunks in vector space, highlighting the user query vector and the top-$K$ retrieved neighbors.



### 5.3. Vector & Retrieval Metrics

* **Ranked Similarity Breakdown**:
* Tabular display of retrieved chunks showing Chunk ID, Preview Snippet, Raw Cosine Score, and Margin to Next Rank.




* **Score Distribution Histogram**:
* Visual score spread across the entire chunk index to inspect retrieval sharpness (showing separation between relevant matches and the semantic noise floor).




* **Overlap Inspection / Diff Viewer**:
* Textual highlighters visually showing overlapping sentences across contiguous chunks (Chunk $i$ and Chunk $i+1$) to verify boundary continuity.





---

## 6. Technical Stack & Implementation Architecture

| Layer | Recommended Technology | Purpose |
| --- | --- | --- |
| **Frontend & UI** | Streamlit or Gradio | Interactive parameter sliders, live dashboard layout, and chat interface. |
| **Data Visualization** | Plotly / Altair | Interactive 2D vector projections, similarity score charts, and token gauges. |
| **Chunking & NLP** | Custom Sliding Window / TikToken / NLTK | Word and subword tokenization with precise boundary overlap.

 |
| **Vectorization** | `scikit-learn` & `sentence-transformers` | Dual-mode representation (Sparse TF-IDF and Dense Embeddings).

 |
| **Vector Indexing** | NumPy / In-Memory Flat Index | Rapid dot product / cosine similarity matrix computations.

 |
| **LLM Inference** | Anthropic Claude SDK (`claude-sonnet-4-6`) | Context-grounded synthesis and structured citation output.

 |

---

## 7. Test Plan & Evaluation Scenarios

| Test Case | Scenario / Query | Expected Engine Behavior | Failure Condition |
| --- | --- | --- | --- |
| **TC-01 (Study Buddy)** | *"Which scheduling algorithm causes the convoy effect and why?"*<br> | High similarity score for FCFS chunk; explains convoy effect with `[Source X]` citation.

 | Hallucinates an explanation without citing sources.

 |
| **TC-02 (Study Buddy)** | *"What is memory segmentation?"* (Out-of-domain query) | Score falls below relevance threshold; returns *"I don't have that information."*

 | Synthesizes an answer from the model's pre-training weights.

 |
| **TC-03 (E-Commerce)** | *"Can I return a backpack after 20 days for a refund?"*<br> | Retrieves 15-day return policy chunk; refuses refund and cites 15-day window.

 | Confirms return eligibility or ignores policy constraints.

 |
| **TC-04 (Dynamic Tuning)** | Drag chunk size from 150 words down to 40 words | Document count dynamically multiplies; token gauge updates; PCA vector plot expands. | App freezes, crashes, or fails to update chunk indices. |
| **TC-05 (Zero Overlap Boundary)** | Split a critical rule exactly across two chunks ($0$ overlap) | Lower retrieval confidence due to split semantics. Demonstrates value of turning overlap to 15-20%.

 | - |

---

## 8. Rollout Phases & Milestones

1. **Milestone 1 (Core Pipeline Refactor)**: Wrap the single-engine script into an OOP service class accepting parameterized chunk sizes, overlaps, and domain knowledge bases.
2. **Milestone 2 (Telemetry & Metrics Layer)**: Build the token counter, TF-IDF / vector inspectability hooks, and cosine score extractors.
3. **Milestone 3 (Interactive Dashboard UI)**: Implement Streamlit controls, dual-persona toggle, Plotly vector mapping, and live response citation rendering.


4. **Milestone 4 (Telemetry Benchmarking & Verification)**: Run regression tests on both academic and support datasets to validate grounding and zero-hallucination guardrails.