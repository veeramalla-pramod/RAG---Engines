"""Interactive Plotly telemetry visualizations: 2D PCA Projections, Score Histograms, and Context Gauges."""

from typing import List, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def create_vector_space_plot(
    doc_matrix: np.ndarray,
    query_vector: np.ndarray,
    chunks: list,
    retrieved_results: List[Dict[str, Any]],
    query_text: str = ""
) -> go.Figure:
    """
    Project document chunks and query vector into 2D space via PCA and render scatter plot.
    """
    fig = go.Figure()

    if doc_matrix.size == 0 or len(chunks) == 0:
        fig.update_layout(title="No vector data available for projection.")
        return fig

    n_samples = doc_matrix.shape[0]
    has_query = query_vector is not None and query_vector.size > 0 and np.linalg.norm(query_vector) > 0

    # Stack docs and query for joint PCA projection
    if has_query:
        combined_matrix = np.vstack([doc_matrix, query_vector.reshape(1, -1)])
    else:
        combined_matrix = doc_matrix

    total_pts = combined_matrix.shape[0]

    if total_pts < 2:
        # Fallback if only 1 point exists
        coords = np.zeros((total_pts, 2))
    elif total_pts == 2:
        pca = PCA(n_components=1)
        reduced_1d = pca.fit_transform(combined_matrix)
        coords = np.column_stack([reduced_1d, np.zeros(total_pts)])
    else:
        n_comp = min(2, combined_matrix.shape[1], combined_matrix.shape[0])
        pca = PCA(n_components=n_comp)
        projected = pca.fit_transform(combined_matrix)
        if n_comp == 1:
            coords = np.column_stack([projected, np.zeros(total_pts)])
        else:
            coords = projected

    doc_coords = coords[:n_samples]
    retrieved_indices = {r["chunk_index"]: r for r in retrieved_results}

    # Group chunks into Unretrieved vs Retrieved
    unretrieved_x, unretrieved_y, unretrieved_hover, unretrieved_text = [], [], [], []
    retrieved_x, retrieved_y, retrieved_hover, retrieved_text = [], [], [], []

    for i, chunk in enumerate(chunks):
        x_val, y_val = float(doc_coords[i, 0]), float(doc_coords[i, 1])
        preview = (chunk.text[:100] + "...") if len(chunk.text) > 100 else chunk.text

        if i in retrieved_indices:
            res = retrieved_indices[i]
            retrieved_x.append(x_val)
            retrieved_y.append(y_val)
            retrieved_text.append(f"Source {chunk.chunk_id}")
            retrieved_hover.append(
                f"<b>Source {chunk.chunk_id}</b> (Rank #{res['rank']})<br>"
                f"Cosine Score: <b>{res['score']:.4f}</b><br>"
                f"Tokens: {chunk.token_count}<br>"
                f"<i>{preview}</i>"
            )
        else:
            unretrieved_x.append(x_val)
            unretrieved_y.append(y_val)
            unretrieved_text.append(f"C{chunk.chunk_id}")
            unretrieved_hover.append(
                f"<b>Chunk {chunk.chunk_id}</b><br>"
                f"Tokens: {chunk.token_count}<br>"
                f"<i>{preview}</i>"
            )

    # 1. Unretrieved chunks (Neutral slate)
    if unretrieved_x:
        fig.add_trace(go.Scatter(
            x=unretrieved_x,
            y=unretrieved_y,
            mode="markers+text",
            name="Corpus Chunks",
            text=unretrieved_text,
            textposition="top center",
            hoverinfo="text",
            hovertext=unretrieved_hover,
            marker=dict(
                size=12,
                color="#64748b",
                opacity=0.65,
                line=dict(color="#334155", width=1.5)
            )
        ))

    # 2. Retrieved Chunks (Accent Emerald / Green)
    if retrieved_x:
        fig.add_trace(go.Scatter(
            x=retrieved_x,
            y=retrieved_y,
            mode="markers+text",
            name="Retrieved Chunks (Top-K)",
            text=retrieved_text,
            textposition="top center",
            hoverinfo="text",
            hovertext=retrieved_hover,
            marker=dict(
                size=18,
                color="#10b981",
                symbol="circle",
                line=dict(color="#047857", width=2.5)
            )
        ))

    # 3. Query Vector (Accent Amber Star)
    if has_query:
        q_x, q_y = float(coords[-1, 0]), float(coords[-1, 1])
        q_preview = (query_text[:60] + "...") if len(query_text) > 60 else query_text
        fig.add_trace(go.Scatter(
            x=[q_x],
            y=[q_y],
            mode="markers+text",
            name="User Query Vector",
            text=["Query"],
            textposition="bottom center",
            hoverinfo="text",
            hovertext=[f"<b>User Query</b><br><i>{q_preview}</i>"],
            marker=dict(
                size=22,
                color="#f59e0b",
                symbol="star",
                line=dict(color="#b45309", width=2)
            )
        ))

    fig.update_layout(
        title=dict(
            text="<b>2D Semantic Topology (PCA Projection)</b>",
            font=dict(size=15, color="#f8fafc")
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#cbd5e1")
        ),
        margin=dict(l=25, r=25, t=55, b=25),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", zerolinecolor="#334155"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", zerolinecolor="#334155"),
        height=420,
    )

    return fig


def create_score_distribution_plot(
    all_scores: np.ndarray,
    threshold: float = 0.0,
    chunks: list = None
) -> go.Figure:
    """
    Render distribution of similarity scores across the chunk corpus with threshold marker.
    """
    fig = go.Figure()

    if all_scores.size == 0:
        fig.update_layout(title="No similarity scores calculated yet.")
        return fig

    # Score breakdown
    above_idx = np.where(all_scores >= threshold)[0]
    below_idx = np.where(all_scores < threshold)[0]

    # Bar chart for each chunk
    chunk_labels = [f"C{i+1}" for i in range(len(all_scores))]
    bar_colors = ["#10b981" if s >= threshold else "#64748b" for s in all_scores]

    fig.add_trace(go.Bar(
        x=chunk_labels,
        y=all_scores,
        marker_color=bar_colors,
        text=[f"{s:.2f}" for s in all_scores],
        textposition="outside",
        hovertext=[
            f"<b>Chunk {i+1}</b><br>Score: {all_scores[i]:.4f}<br>{'Status: Passed' if all_scores[i] >= threshold else 'Status: Suppressed'}"
            for i in range(len(all_scores))
        ],
        hoverinfo="text",
        name="Cosine Score"
    ))

    # Add similarity threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_position="top left",
        annotation_font=dict(color="#f87171", size=11)
    )

    fig.update_layout(
        title=dict(
            text="<b>Retrieval Sharpness & Corpus Score Spread</b>",
            font=dict(size=15, color="#f8fafc")
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=25, r=25, t=55, b=25),
        xaxis=dict(title="Chunk ID", gridcolor="#1e293b"),
        yaxis=dict(title="Cosine Similarity", range=[0.0, 1.05], gridcolor="#1e293b"),
        height=320,
    )

    return fig


def create_context_budget_gauge(
    prompt_tokens: int,
    retrieved_tokens: int,
    completion_budget: int,
    max_context: int = 4096
) -> go.Figure:
    """
    Render stacked context window budget progress gauge.
    """
    consumed = prompt_tokens + retrieved_tokens + completion_budget
    remaining = max(0, max_context - consumed)
    utilization_pct = min(100.0, round((consumed / max_context) * 100, 1))

    fig = go.Figure()

    # Stacked horizontal bar
    fig.add_trace(go.Bar(
        y=["Context Window"],
        x=[prompt_tokens],
        name="Prompt Overhead",
        orientation="h",
        marker=dict(color="#3b82f6"),
        hovertext=[f"Prompt System Overhead: {prompt_tokens} tokens"],
        hoverinfo="text"
    ))
    fig.add_trace(go.Bar(
        y=["Context Window"],
        x=[retrieved_tokens],
        name="Retrieved Chunks",
        orientation="h",
        marker=dict(color="#10b981"),
        hovertext=[f"Retrieved Chunks: {retrieved_tokens} tokens"],
        hoverinfo="text"
    ))
    fig.add_trace(go.Bar(
        y=["Context Window"],
        x=[completion_budget],
        name="Completion Budget",
        orientation="h",
        marker=dict(color="#f59e0b"),
        hovertext=[f"Target Completion Budget: {completion_budget} tokens"],
        hoverinfo="text"
    ))
    fig.add_trace(go.Bar(
        y=["Context Window"],
        x=[remaining],
        name="Available Headroom",
        orientation="h",
        marker=dict(color="#334155"),
        hovertext=[f"Available Headroom: {remaining} tokens"],
        hoverinfo="text"
    ))

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=f"<b>LLM Context Window Budget Gauge</b> ({consumed:,} / {max_context:,} tokens — {utilization_pct}%)",
            font=dict(size=14, color="#f8fafc")
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=25, r=25, t=45, b=25),
        xaxis=dict(title="Tokens", range=[0, max_context], gridcolor="#1e293b"),
        yaxis=dict(showticklabels=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#cbd5e1")
        ),
        height=180,
    )

    return fig
