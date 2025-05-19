"""
data.py

This module handles data loading and preprocessing for social network datasets.
It provides functions to load Facebook and Twitter datasets, a custom dataset via file path,
and to scrape data from a URL.
"""

import networkx as nx
import os
import requests
from io import StringIO

# Directory containing the datasets (adjust as necessary)
DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")

def load_facebook_social_network():
    """Load the Facebook social circles network into a NetworkX graph."""
    path = os.path.join(DATA_DIR, "facebook_combined.txt")
    G = nx.read_edgelist(path, nodetype=int)
    G = nx.convert_node_labels_to_integers(G)
    return G

def load_twitter_social_network():
    """Load the Twitter ego network into a NetworkX DiGraph."""
    path = os.path.join(DATA_DIR, "twitter_combined.txt")
    G = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph())
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def load_custom_dataset(file_path):
    """Load a custom dataset from the specified file path into a NetworkX graph."""
    if not os.path.exists(file_path):
        raise ValueError(f"Dataset not found at path: {file_path}")
    return nx.read_edgelist(file_path, nodetype=int)

def load_dataset(name="facebook"):
    """
    General dataset loader.
    'name' can be 'facebook', 'twitter', or a path to a custom dataset.
    Returns a NetworkX graph.
    """
    name = name.lower()
    if name == "facebook":
        return load_facebook_social_network()
    elif name == "twitter":
        return load_twitter_social_network()
    elif os.path.exists(name):
        return load_custom_dataset(name)
    else:
        raise ValueError(f"Unknown dataset name or path: {name}")

def scrape_dataset(url):
    """
    Scrape a dataset from a URL containing an edge list in plain text format.
    Returns a NetworkX graph constructed from the scraped data.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch data from {url}")
    data_str = response.text
    file_like = StringIO(data_str)
    G = nx.read_edgelist(file_like, nodetype=int)
    return G
