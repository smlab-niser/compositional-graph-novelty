"""
Edge-type novelty computation
"""

import networkx as nx
import numpy as np
from collections import Counter
from typing import Dict, Tuple


def compute_edge_type_novelty(
    G_new: nx.Graph,
    corpus_edge_types: Counter
) -> float:
    """
    Compute novelty based on edge type combinations

    Args:
        G_new: New graph to evaluate
        corpus_edge_types: Counter of edge type combinations in corpus

    Returns:
        Edge-type novelty score [0, 1]
    """
    if G_new.number_of_edges() == 0:
        return 0.0

    rarities = []
    total_edges = sum(corpus_edge_types.values())

    for u, v, data in G_new.edges(data=True):
        edge_type = data.get('type', 'default')
        u_type = G_new.nodes[u].get('type', 'default')
        v_type = G_new.nodes[v].get('type', 'default')

        combo = (u_type, edge_type, v_type)
        frequency = corpus_edge_types.get(combo, 0)

        if frequency == 0:
            rarity = 1.0
        else:
            rarity = 1.0 - (frequency / total_edges)

        rarities.append(rarity)

    return np.mean(rarities)
