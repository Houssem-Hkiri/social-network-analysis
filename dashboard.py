"""
dashboard.py

This Streamlit dashboard enables interactive exploration of social network data,
centrality measures, and diffusion-based simulation of information propagation.
Users can load datasets, select centrality metrics, run diffusion simulations,
and obtain diffusion-based influencer rankings with real-time progress updates.
The dashboard uses interactive Plotly visualizations and provides explanatory text
to guide users through each analysis section.
"""

import streamlit as st
import networkx as nx
import data
import centrality
import diffusion
import visualization
import metrics
import random

# -------------------------------------------------------------------
# Page Configuration and Custom CSS
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Social Network Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

custom_css = """
<style>
    .reportview-container {
        background-color: #FAFAFA;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #FFFFFF;
        padding: 1rem;
    }
    h1 {
        color: #333333;
        font-size: 2.5rem;
        font-weight: bold;
    }
    h2, h3, h4 {
        color: #444444;
        margin-top: 1.5rem;
    }
    p, .css-1d391kg {
        color: #555555;
        font-size: 1rem;
    }
    hr {
        border: 1px solid #DDDDDD;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -------------------------------------------------------------------
# Top Header: University Logos and Title/Student Information
# -------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image("UTM.png", width=80)  # UTM logo; verify file path.
with col2:
    st.title("Social Network Analysis Dashboard")
    st.subheader("By: Houssem HKIRI & Rachid ZGHAL")
with col3:
    st.image("ENIT.png", width=80)  # ENIT logo; verify file path.

st.markdown("---")

# -------------------------------------------------------------------
# Dataset Selection and Loading
# -------------------------------------------------------------------
st.sidebar.header("Dataset Options")
data_file = st.sidebar.file_uploader("Upload Custom Dataset (edge list as txt)", type=["txt"])
if data_file is not None:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(data_file.read())
        dataset_path = tmp.name
    G = data.load_custom_dataset(dataset_path)
    dataset_choice = "Custom Uploaded Dataset"
else:
    dataset_choice = st.sidebar.selectbox("Select Dataset", ["Facebook (Circles)", "Twitter (Ego Network)"])
    if "Facebook" in dataset_choice:
        G = data.load_dataset("facebook")
    else:
        G = data.load_dataset("twitter")

st.write(f"**Dataset:** {dataset_choice}. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
st.markdown("---")

# -------------------------------------------------------------------
# Centrality Analysis Section
# -------------------------------------------------------------------
st.header("Centrality Analysis")
cent_choice = st.selectbox("Select Centrality Measure", ["Degree", "Betweenness", "Closeness", "PageRank", "Composite"])

if cent_choice == "Degree":
    cent_values = centrality.degree_centrality(G)
elif cent_choice == "Betweenness":
    cent_values = centrality.betweenness_centrality(G)
elif cent_choice == "Closeness":
    cent_values = centrality.closeness_centrality(G)
elif cent_choice == "PageRank":
    cent_values = centrality.pagerank_centrality(G)
elif cent_choice == "Composite":
    d = centrality.degree_centrality(G)
    b = centrality.betweenness_centrality(G)
    c = centrality.closeness_centrality(G)
    p = centrality.pagerank_centrality(G)
    cent_values = {node: (d[node] + b[node] + c[node] + p[node]) / 4.0 for node in G.nodes()}

top5 = sorted(cent_values.items(), key=lambda x: x[1], reverse=True)[:5]
st.write(f"**Top 5 nodes by {cent_choice} centrality:**",
         [f"Node {nid} ({val:.4f})" for nid, val in top5])
st.markdown("---")

# -------------------------------------------------------------------
# Section Selector (Dashboard Functions)
# -------------------------------------------------------------------
graph_section = st.sidebar.radio(
    "Select Analysis Section",
    ["Centrality Visualization", "Run Diffusion Simulation", "Enhanced Influencer Ranking"]
)

# -------------------------------------------------------------------
# Case 1: Centrality Visualization
# -------------------------------------------------------------------
if graph_section == "Centrality Visualization":
    if st.button("Show Centrality Graph"):
        st.header(f"Network Visualization (Centrality: {cent_choice})")
        node_size = [cent_values[node] * 10 for node in G.nodes()]
        node_color = [cent_values[node] for node in G.nodes()]
        visualization.plot_network_interactive(G, node_color=node_color, node_size=node_size)
        st.markdown("**Note:** Larger and more intensely colored nodes have higher centrality values, indicating greater importance.")

# -------------------------------------------------------------------
# Case 2: Diffusion Simulation
# -------------------------------------------------------------------
if graph_section == "Run Diffusion Simulation":
    st.header("Diffusion Simulation Setup")
    diff_method = st.selectbox("Select Diffusion Model", ["Independent Cascade", "Linear Threshold"])
    seeds_input = st.text_input("Enter Seed Node IDs (comma-separated)", "0")
    try:
        seed_nodes = [int(x.strip()) for x in seeds_input.split(",") if x.strip() != ""]
    except Exception as e:
        seed_nodes = []
        st.error("Error: Please enter valid comma-separated integer node IDs.")

    if st.button("Run Diffusion Simulation"):
        if diff_method == "Independent Cascade":
            model = diffusion.IndependentCascadeModel(G, p=0.1)
        else:
            model = diffusion.LinearThresholdModel(G)
        final_active = model.simulate(seed_nodes)
        st.write(f"**Initial Seeds:** {seed_nodes}")
        st.write(f"**Total Activated Nodes:** {len(final_active)}")
        st.header("Diffusion Outcome Visualization")
        node_size = [cent_values[node] * 10 for node in G.nodes()]
        node_color_list = [cent_values[node] for node in G.nodes()]
        visualization.plot_network_interactive(G, node_color=node_color_list, node_size=node_size,
                                               activated_nodes=final_active, seeds=seed_nodes)
        st.markdown("**Interpretation:** The simulation visualizes how influence propagates from the seed nodes. In the IC model, active nodes attempt to activate neighbors with a specified probability, while in the LT model, nodes activate when cumulative influence exceeds their threshold.")

# -------------------------------------------------------------------
# Case 3: Enhanced Influencer Ranking via Diffusion Simulations
# -------------------------------------------------------------------
if graph_section == "Enhanced Influencer Ranking":
    st.header("Enhanced Influencer Ranking via Diffusion Simulations")
    num_runs = st.number_input("Set number of simulation runs per node:", min_value=1, max_value=200, value=10, step=1)
    st.markdown("Select a subset of nodes to evaluate (to reduce runtime):")
    max_nodes = st.number_input("Maximum nodes to evaluate:", min_value=1, max_value=len(G.nodes()), value=min(len(G.nodes()), 50))
    subset_method = st.selectbox("Choose Node Subset Method:", ["Random Sample", "Top Degree Nodes", "All Nodes"])

    if subset_method == "All Nodes":
        node_subset = list(G.nodes())
    elif subset_method == "Random Sample":
        node_list = list(G.nodes())
        random.shuffle(node_list)
        node_subset = node_list[:max_nodes]
    else:  # Top Degree Nodes
        deg_centrality = centrality.degree_centrality(G)
        deg_sorted = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)
        node_subset = [node for node, _ in deg_sorted[:max_nodes]]

    st.markdown("Click the button below to compute diffusion-based influence scores on the selected subset of nodes.")
    if st.button("Run Enhanced Influencer Ranking"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"Processed {current} of {total} nodes...")

        analyzer = diffusion.CompositeInfluenceAnalyzer(
            G,
            diffusion_model_class=diffusion.IndependentCascadeModel,
            model_params={'p': 0.1},
            runs=int(num_runs)
        )

        st.write("Running diffusion simulations (this may take some time)...")
        influence_scores = analyzer.compute_influence_scores(node_subset=node_subset, progress_callback=progress_callback)
        top_influencers = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        st.success("Enhanced Influencer Ranking completed!")
        st.markdown("### Top Influencers based on Diffusion Simulations")
        for node, score in top_influencers:
            st.write(f"Node {node}: Average Diffusion Spread = {score:.2f}")

        st.markdown("**Interpretation:** The diffusion-based ranking uses multiple simulation runs to estimate each node's dynamic influence."
                    " A higher average spread indicates that the node can trigger larger cascades when used as a seed.")


# -------------------------------------------------------------------
# End of Dashboard
# -------------------------------------------------------------------
st.markdown("---")
st.write("Developed by Houssem HKIRI & Rachid ZGHAL, supervised by Ms. Myriam GHARBI & Ms. Nabila BITRI.")