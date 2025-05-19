social_network_analysis/  
├── __init__.py               # This file makes our directory work as a Python package  
├── centrality.py             # Contains various centrality calculations like degree, closeness, betweenness and PageRank  
├── diffusion.py              # Houses our diffusion models - both Independent Cascade and Linear Threshold implementations  
├── visualization.py          # All the visualization tools - from static NetworkX/Matplotlib plots to interactive graphics  
├── data.py                   # Handles loading and cleaning up Twitter and Facebook network datasets  
├── metrics.py                # Tools for measuring and analyzing centrality effects and diffusion results  
├── dashboard.py              # Our Streamlit dashboard for interactively exploring networks and simulating cascades  
└── datasets/                 # Where we store our example network data files (Facebook and Twitter networks)   