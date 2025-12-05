"""
Core Graph Compositional Novelty implementation
"""

import networkx as nx
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
from .motifs import extract_motifs, canonicalize_graph
from .edge_novelty import compute_edge_type_novelty
from .bridging import compute_bridging_novelty


class GraphCompositionalNovelty:
    """
    Measures compositional novelty of graphs based on three components:
    1. Structural novelty (motif-based)
    2. Edge-type novelty (relationship combinations)
    3. Bridging novelty (semantic distance)
    """

    def __init__(
        self,
        corpus_graphs: List[nx.Graph],
        k: int = 3,
        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
    ):
        """
        Args:
            corpus_graphs: List of known graphs to compare against
            k: Size of motifs to extract (k=3 for triangles, etc.)
            weights: (w_structural, w_edge, w_bridging) - must sum to 1.0
        """
        if not np.isclose(sum(weights), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

        self.corpus_graphs = corpus_graphs
        self.k = k
        self.weights = weights

        # Pre-compute corpus statistics
        print("Extracting corpus motifs...")
        self.corpus_motifs = self._extract_corpus_motifs()
        print(f"Found {len(self.corpus_motifs)} unique motifs")

        print("Extracting corpus edge types...")
        self.corpus_edge_types = self._extract_corpus_edge_types()
        print(f"Found {len(self.corpus_edge_types)} unique edge type combinations")

        print("Computing semantic distances...")
        self.semantic_distances = self._compute_semantic_distances()

    def compute_novelty(self, G_new: nx.Graph) -> Dict[str, float]:
        """
        Compute compositional novelty of new graph

        Returns:
            Dictionary with component scores and overall novelty
        """
        # Component 1: Structural novelty
        structural = self._compute_structural_novelty(G_new)

        # Component 2: Edge-type novelty
        edge_novelty = compute_edge_type_novelty(
            G_new, self.corpus_edge_types
        )

        # Component 3: Bridging novelty
        bridging = compute_bridging_novelty(
            G_new, self.semantic_distances
        )

        # Weighted combination
        overall = (
            self.weights[0] * structural +
            self.weights[1] * edge_novelty +
            self.weights[2] * bridging
        )

        return {
            'structural_novelty': structural,
            'edge_novelty': edge_novelty,
            'bridging_novelty': bridging,
            'overall_novelty': overall
        }

    def _extract_corpus_motifs(self) -> Counter:
        """Extract all k-hop motifs from corpus"""
        all_motifs = Counter()

        for G in self.corpus_graphs:
            motifs = extract_motifs(G, self.k)
            all_motifs.update(motifs)

        return all_motifs

    def _compute_structural_novelty(self, G_new: nx.Graph) -> float:
        """
        Compute novelty based on motif rarity
        """
        new_motifs = extract_motifs(G_new, self.k)

        if not new_motifs:
            return 0.0

        # Compute rarity of each motif
        rarities = []
        total_corpus_motifs = sum(self.corpus_motifs.values())

        for motif in new_motifs:
            frequency = self.corpus_motifs.get(motif, 0)

            # Rarity = 1 - (frequency / total)
            # Unseen motifs get rarity = 1.0
            if frequency == 0:
                rarity = 1.0
            else:
                rarity = 1.0 - (frequency / total_corpus_motifs)

            rarities.append(rarity)

        # Return average rarity
        return np.mean(rarities)

    def _extract_corpus_edge_types(self) -> Counter:
        """Extract edge type combinations from corpus"""
        edge_type_combos = Counter()

        for G in self.corpus_graphs:
            for u, v, data in G.edges(data=True):
                edge_type = data.get('type', 'default')

                # Get node types
                u_type = G.nodes[u].get('type', 'default')
                v_type = G.nodes[v].get('type', 'default')

                # Create edge type combination
                combo = (u_type, edge_type, v_type)
                edge_type_combos[combo] += 1

        return edge_type_combos

    def _compute_semantic_distances(self) -> Dict[Tuple[str, str], float]:
        """
        Pre-compute semantic distances between node types
        Uses co-occurrence-based approach: types that co-occur frequently are closer
        """
        from collections import Counter

        # Count co-occurrences of node types in same graph
        cooccurrence = Counter()
        type_counts = Counter()

        for G in self.corpus_graphs:
            # Get all types in this graph
            types_in_graph = [G.nodes[n].get('type', 'default') for n in G.nodes()]
            type_counts.update(types_in_graph)

            # Count pairs that appear together
            unique_types = set(types_in_graph)
            for t1 in unique_types:
                for t2 in unique_types:
                    if t1 != t2:
                        # Store as ordered tuple for consistency
                        pair = tuple(sorted([t1, t2]))
                        cooccurrence[pair] += 1

        # Convert to distances: high co-occurrence = low distance
        distances = {}
        node_types = list(type_counts.keys())
        max_cooccur = max(cooccurrence.values()) if cooccurrence else 1.0

        for t1 in node_types:
            for t2 in node_types:
                if t1 == t2:
                    distances[(t1, t2)] = 0.0
                else:
                    pair = tuple(sorted([t1, t2]))
                    co = cooccurrence.get(pair, 0)
                    # Distance = 1 - normalized co-occurrence
                    # High co-occurrence → low distance (semantically close)
                    distances[(t1, t2)] = 1.0 - (co / max_cooccur)

        return distances
