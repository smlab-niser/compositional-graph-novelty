"""
Motif extraction and canonicalization
"""

import networkx as nx
from typing import List


def extract_motifs(G: nx.Graph, k: int) -> List[str]:
    """
    Extract k-hop neighborhood motifs from graph

    Args:
        G: Input graph
        k: Hop distance for neighborhood extraction

    Returns:
        List of canonical string representations of motifs
    """
    motifs = []

    for node in G.nodes():
        # Get k-hop subgraph around node
        try:
            subgraph_nodes = nx.single_source_shortest_path_length(
                G, node, cutoff=k
            ).keys()
            subgraph = G.subgraph(subgraph_nodes)

            # Convert to canonical form (handles isomorphism)
            canonical = canonicalize_graph(subgraph)
            motifs.append(canonical)
        except:
            # Handle disconnected components or other issues
            continue

    return motifs


def canonicalize_graph(G: nx.Graph) -> str:
    """
    Convert graph to canonical string representation
    Handles graph isomorphism using Weisfeiler-Lehman hash

    Args:
        G: Input graph

    Returns:
        Canonical string representation
    """
    try:
        # Use Weisfeiler-Lehman graph hash for canonical form
        return nx.weisfeiler_lehman_graph_hash(G)
    except:
        # Fallback: simple hash based on structure
        return f"nodes_{G.number_of_nodes()}_edges_{G.number_of_edges()}"
