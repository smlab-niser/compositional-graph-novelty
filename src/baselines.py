"""
Baseline novelty metrics for comparison
"""

import networkx as nx
import numpy as np
from typing import List
from collections import Counter


def set_membership_novelty(eval_graphs: List[nx.Graph], corpus_graphs: List[nx.Graph]) -> np.ndarray:
    """
    Simple binary novelty: 1 if graph not in corpus, 0 otherwise
    Uses graph isomorphism checking
    """
    scores = []

    # Create WL hashes for corpus graphs (faster than isomorphism)
    corpus_hashes = set()
    for G in corpus_graphs:
        try:
            h = nx.weisfeiler_lehman_graph_hash(G, node_attr='type', edge_attr='type')
            corpus_hashes.add(h)
        except:
            # Fallback if attributes missing
            h = nx.weisfeiler_lehman_graph_hash(G)
            corpus_hashes.add(h)

    # Check each eval graph
    for G in eval_graphs:
        try:
            h = nx.weisfeiler_lehman_graph_hash(G, node_attr='type', edge_attr='type')
        except:
            h = nx.weisfeiler_lehman_graph_hash(G)

        # Novel if hash not seen in corpus
        scores.append(1.0 if h not in corpus_hashes else 0.0)

    return np.array(scores)


def graph_edit_distance_novelty(eval_graphs: List[nx.Graph], corpus_graphs: List[nx.Graph]) -> np.ndarray:
    """
    Novelty based on minimum graph edit distance to corpus
    Normalized by graph size
    """
    scores = []

    for G_eval in eval_graphs:
        # Compute distance to nearest corpus graph
        min_distance = float('inf')

        # Sample corpus to make tractable (GED is expensive)
        sample_size = min(100, len(corpus_graphs))
        indices = np.random.choice(len(corpus_graphs), size=sample_size, replace=False)
        sampled_corpus = [corpus_graphs[i] for i in indices]

        for G_corpus in sampled_corpus:
            # Simple distance: symmetric difference of edge sets
            edges_eval = set(G_eval.edges())
            edges_corpus = set(G_corpus.edges())

            symmetric_diff = len(edges_eval.symmetric_difference(edges_corpus))

            # Normalize by max graph size
            max_edges = max(len(edges_eval), len(edges_corpus))
            distance = symmetric_diff / max_edges if max_edges > 0 else 0.0

            min_distance = min(min_distance, distance)

        scores.append(min_distance)

    return np.array(scores)


def mmd_novelty(eval_graphs: List[nx.Graph], corpus_graphs: List[nx.Graph]) -> np.ndarray:
    """
    Maximum Mean Discrepancy in feature space
    Uses simple graph statistics as features
    """
    def extract_features(G):
        """Extract simple graph statistics"""
        n = G.number_of_nodes()
        m = G.number_of_edges()

        if n == 0:
            return np.zeros(5)

        density = m / (n * (n - 1) / 2) if n > 1 else 0.0
        avg_degree = 2 * m / n if n > 0 else 0.0

        # Degree distribution moments
        degrees = [d for _, d in G.degree()]
        degree_std = np.std(degrees) if degrees else 0.0

        # Clustering coefficient
        try:
            clustering = nx.average_clustering(G)
        except:
            clustering = 0.0

        return np.array([n, density, avg_degree, degree_std, clustering])

    # Extract features
    corpus_features = np.array([extract_features(G) for G in corpus_graphs])
    eval_features = np.array([extract_features(G) for G in eval_graphs])

    # Compute MMD for each eval graph against corpus distribution
    scores = []
    corpus_mean = np.mean(corpus_features, axis=0)
    corpus_std = np.std(corpus_features, axis=0) + 1e-6

    for feat in eval_features:
        # Mahalanobis-like distance
        diff = (feat - corpus_mean) / corpus_std
        mmd_score = np.sqrt(np.sum(diff ** 2))
        scores.append(mmd_score)

    # Normalize to [0, 1]
    scores = np.array(scores)
    if len(scores) > 0 and scores.max() > 0:
        scores = scores / scores.max()

    return scores


def kernel_distance_novelty(eval_graphs: List[nx.Graph], corpus_graphs: List[nx.Graph]) -> np.ndarray:
    """
    Weisfeiler-Lehman kernel distance
    """
    from collections import Counter

    def wl_features(G, iterations=3):
        """Extract WL subtree features"""
        # Initialize labels
        labels = {}
        for node in G.nodes():
            labels[node] = G.nodes[node].get('type', 'default')

        feature_vectors = []

        for _ in range(iterations):
            new_labels = {}
            for node in G.nodes():
                # Collect neighbor labels
                neighbor_labels = sorted([labels[n] for n in G.neighbors(node)])
                # Create new label
                new_label = str(labels[node]) + ''.join(neighbor_labels)
                new_labels[node] = new_label

            labels = new_labels

            # Count label occurrences
            label_counts = Counter(labels.values())
            feature_vectors.append(label_counts)

        # Flatten to single feature vector
        all_features = Counter()
        for vec in feature_vectors:
            all_features.update(vec)

        return all_features

    # Extract features
    corpus_features = [wl_features(G) for G in corpus_graphs]
    eval_features = [wl_features(G) for G in eval_graphs]

    # Compute distance for each eval graph
    scores = []

    for eval_feat in eval_features:
        # Find minimum distance to corpus
        min_distance = float('inf')

        # Sample corpus for efficiency
        sample_size = min(100, len(corpus_features))
        sampled_corpus = np.random.choice(len(corpus_features), size=sample_size, replace=False)

        for idx in sampled_corpus:
            corpus_feat = corpus_features[idx]

            # Cosine distance between feature vectors
            all_keys = set(eval_feat.keys()) | set(corpus_feat.keys())

            if len(all_keys) == 0:
                distance = 0.0
            else:
                dot_product = sum(eval_feat.get(k, 0) * corpus_feat.get(k, 0) for k in all_keys)
                norm_eval = np.sqrt(sum(v**2 for v in eval_feat.values()))
                norm_corpus = np.sqrt(sum(v**2 for v in corpus_feat.values()))

                if norm_eval == 0 or norm_corpus == 0:
                    distance = 1.0
                else:
                    cosine_sim = dot_product / (norm_eval * norm_corpus)
                    distance = 1.0 - cosine_sim

            min_distance = min(min_distance, distance)

        scores.append(min_distance)

    return np.array(scores)


def embedding_distance_novelty(eval_graphs: List[nx.Graph], corpus_graphs: List[nx.Graph]) -> np.ndarray:
    """
    Distance in learned embedding space
    Uses simple degree-based embeddings as placeholder
    """
    def embed_graph(G):
        """Simple graph embedding: degree distribution histogram"""
        degrees = [d for _, d in G.degree()]
        if not degrees:
            return np.zeros(10)

        # Histogram of degrees (bins 0-9+)
        hist, _ = np.histogram(degrees, bins=range(11))
        # Normalize
        hist = hist.astype(float)
        if hist.sum() > 0:
            hist = hist / hist.sum()

        return hist

    # Compute embeddings
    corpus_embeddings = np.array([embed_graph(G) for G in corpus_graphs])
    eval_embeddings = np.array([embed_graph(G) for G in eval_graphs])

    # Compute distance for each eval graph
    scores = []

    for eval_emb in eval_embeddings:
        # Find minimum distance to corpus
        distances = np.linalg.norm(corpus_embeddings - eval_emb, axis=1)
        min_distance = np.min(distances)
        scores.append(min_distance)

    # Normalize to [0, 1]
    scores = np.array(scores)
    if len(scores) > 0 and scores.max() > 0:
        scores = scores / scores.max()

    return scores
