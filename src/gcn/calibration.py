"""
Human-calibrated graph compositional novelty
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from .core import GraphCompositionalNovelty


class HumanCalibratedGCN:
    """
    Human-calibrated graph compositional novelty scorer
    """

    def __init__(self):
        self.base_gcn = None
        self.calibration_model = None
        self.feature_names = None

    def train_calibration(
        self,
        graphs: List[nx.Graph],
        human_ratings: List[float],
        corpus_graphs: List[nx.Graph],
        k: int = 3
    ):
        """
        Train calibration model on human-rated data

        Args:
            graphs: List of graphs that were rated
            human_ratings: Human novelty ratings (1-10 scale)
            corpus_graphs: Reference corpus
            k: Motif size for base GCN
        """
        # Initialize base GCN scorer
        self.base_gcn = GraphCompositionalNovelty(corpus_graphs, k=k)

        # Extract features for each graph
        features_list = []

        print("Extracting features for calibration...")
        for i, G in enumerate(graphs):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(graphs)} graphs")

            features = self._extract_features(G)
            features_list.append(features)

        # Convert to DataFrame
        X = pd.DataFrame(features_list)
        y = np.array(human_ratings)

        self.feature_names = X.columns.tolist()

        # Train calibration model
        print("Training calibration model...")
        self.calibration_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        self.calibration_model.fit(X, y)

        # Report cross-validation performance
        cv_scores = cross_val_score(
            self.calibration_model, X, y,
            cv=5,
            scoring='neg_mean_squared_error'
        )

        r2_score = self.calibration_model.score(X, y)

        print(f"\nCalibration Model Performance:")
        print(f"  CV MSE: {-np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        print(f"  R²: {r2_score:.3f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.calibration_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 5 Important Features:")
        print(feature_importance.head())

        return {
            'cv_mse': -np.mean(cv_scores),
            'r2': r2_score,
            'feature_importance': feature_importance
        }

    def predict_novelty(self, G_new: nx.Graph) -> float:
        """
        Predict human-aligned novelty score for new graph

        Returns:
            Predicted novelty score (1-10 scale)
        """
        if self.calibration_model is None:
            raise ValueError("Model not trained. Call train_calibration first.")

        features = self._extract_features(G_new)
        X = pd.DataFrame([features])

        # Ensure columns match training data
        X = X[self.feature_names]

        prediction = self.calibration_model.predict(X)[0]

        # Clip to valid range [1, 10]
        return np.clip(prediction, 1.0, 10.0)

    def _extract_features(self, G: nx.Graph) -> Dict[str, float]:
        """Extract features from graph for calibration"""
        # Get base GCN scores
        gcn_scores = self.base_gcn.compute_novelty(G)

        # Compute graph statistics
        features = {
            'structural_novelty': gcn_scores['structural_novelty'],
            'edge_novelty': gcn_scores['edge_novelty'],
            'bridging_novelty': gcn_scores['bridging_novelty'],

            # Graph statistics
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G) if G.number_of_nodes() > 0 else 0.0,
        }

        # Add average degree
        if G.number_of_nodes() > 0:
            features['avg_degree'] = np.mean([d for n, d in G.degree()])
        else:
            features['avg_degree'] = 0.0

        # Add clustering coefficient
        try:
            features['clustering_coef'] = nx.average_clustering(G)
        except:
            features['clustering_coef'] = 0.0

        # Add diameter (for connected graphs)
        try:
            if nx.is_connected(G) and G.number_of_nodes() > 1:
                features['diameter'] = nx.diameter(G)
            else:
                features['diameter'] = -1
        except:
            features['diameter'] = -1

        # Add assortativity
        try:
            features['assortativity'] = nx.degree_assortativity_coefficient(G)
        except:
            features['assortativity'] = 0.0

        # Add heterogeneity measures
        features['num_node_types'] = len(set(
            G.nodes[n].get('type', 'default')
            for n in G.nodes()
        ))

        features['num_edge_types'] = len(set(
            data.get('type', 'default')
            for u, v, data in G.edges(data=True)
        ))

        return features
