"""
visualization.py

This module provides tools for creating interactive network visualizations using Plotly.
The function plot_network_interactive() creates an interactive graph plot with customizable
node colors, sizes, and highlights for activated nodes and seed nodes.
"""

import plotly.graph_objects as go
import networkx as nx


def plot_network_interactive(G, node_color=None, node_size=None, activated_nodes=None, seeds=None):
    """
    Create an interactive Plotly visualization of graph G.

    Parameters:
      - G: a NetworkX graph.
      - node_color: a list or dictionary of values for coloring nodes.
      - node_size: a list or dictionary specifying node sizes.
      - activated_nodes: nodes to highlight (e.g., from diffusion simulation).
      - seeds: seed nodes to highlight.

    The plot includes hover information with node ID and value,
    a color scale indicator, and customized legends.
    """
    pos = nx.spring_layout(G, seed=42)
    x_values = [pos[node][0] for node in G.nodes()]
    y_values = [pos[node][1] for node in G.nodes()]
    if isinstance(node_color, dict):
        node_color = [node_color[node] for node in G.nodes()]
    if isinstance(node_size, dict):
        node_size = [node_size[node] for node in G.nodes()]

    # Create edge trace
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node trace with hover labels
    hover_text = [f"Node {node}<br>Value: {node_color[i]:.4f}"
                  for i, node in enumerate(G.nodes())]
    node_trace = go.Scatter(
        x=x_values, y=y_values,
        mode='markers',
        hoverinfo='text',
        text=hover_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=node_color if node_color is not None else [0.5] * len(G.nodes()),
            size=node_size if node_size is not None else 10,
            colorbar=dict(
                thickness=15,
                title="Centrality Value",
                xanchor="left"
            )
        ),
        name="Network Nodes"
    )

    data = [edge_trace, node_trace]

    # Highlight activated nodes if provided
    if activated_nodes:
        activated_x = [pos[node][0] for node in activated_nodes]
        activated_y = [pos[node][1] for node in activated_nodes]
        activated_trace = go.Scatter(
            x=activated_x, y=activated_y,
            mode='markers',
            marker=dict(color='orange', size=20, symbol='circle', line=dict(width=2, color='black')),
            name='Activated Nodes'
        )
        data.append(activated_trace)

    # Highlight seed nodes if provided
    if seeds:
        seed_x = [pos[node][0] for node in seeds]
        seed_y = [pos[node][1] for node in seeds]
        seed_trace = go.Scatter(
            x=seed_x, y=seed_y,
            mode='markers',
            marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='black')),
            name='Seed Nodes'
        )
        data.append(seed_trace)

    fig = go.Figure(data=data)
    fig.update_layout(
        title="Interactive Network Visualization",
        title_font=dict(size=16),
        showlegend=True,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )
    fig.show()
