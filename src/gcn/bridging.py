"""
Bridging novelty computation
"""

import networkx as nx
import numpy as np
from typing import Dict, Tuple


def compute_bridging_novelty(
    G_new: nx.Graph,
    semantic_distances: Dict[Tuple[str, str], float]
) -> float:
    """
    Compute novelty based on semantic bridging
    (connecting distant concepts)

    Args:
        G_new: New graph to evaluate
        semantic_distances: Pre-computed semantic distances between node types

    Returns:
        Bridging novelty score [0, 1]
    """
    if G_new.number_of_edges() == 0:
        return 0.0

    bridging_scores = []

    for u, v, data in G_new.edges(data=True):
        u_type = G_new.nodes[u].get('type', 'default')
        v_type = G_new.nodes[v].get('type', 'default')

        # Get semantic distance
        distance = semantic_distances.get(
            (u_type, v_type),
            0.5  # default if not found
        )

        bridging_scores.append(distance)

    # Higher average distance = more bridging = more novel
    mean_dist = np.mean(bridging_scores) if bridging_scores else 0.0

    # Normalize by max distance to ensure score in [0,1] range
    max_dist = max(semantic_distances.values()) if semantic_distances else 1.0

    # Handle edge case where max_dist is 0 (all types identical)
    if max_dist == 0:
        return 0.0

    return mean_dist / max_dist
