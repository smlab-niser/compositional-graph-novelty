"""
Data loading utilities for various graph datasets
"""

import networkx as nx
import numpy as np
from typing import List, Tuple
from pathlib import Path
import sys


def load_dataset(
    dataset_name: str,
    corpus_size: int = None,
    eval_size: int = None,
    data_dir: str = "data/raw"
) -> Tuple[List[nx.Graph], List[nx.Graph]]:
    """
    Load a graph dataset and split into corpus and evaluation sets

    Args:
        dataset_name: One of 'qm9', 'arxiv', 'reddit', 'protein', 'zinc', 'cora',
                      'citeseer', 'enzymes', 'er', 'ba', 'ws'
        corpus_size: Number of graphs for corpus (None = use 90%)
        eval_size: Number of graphs for evaluation (None = use 10%)
        data_dir: Directory containing raw data

    Returns:
        (corpus_graphs, eval_graphs)
    """

    if dataset_name == 'qm9':
        graphs = load_qm9(data_dir)
    elif dataset_name == 'arxiv':
        graphs = load_arxiv(data_dir)
    elif dataset_name == 'reddit':
        graphs = load_reddit(data_dir)
    elif dataset_name == 'protein':
        graphs = load_protein(data_dir)
    elif dataset_name == 'zinc':
        graphs = load_zinc(data_dir)
    elif dataset_name == 'cora':
        graphs = load_cora(data_dir)
    elif dataset_name == 'citeseer':
        graphs = load_citeseer(data_dir)
    elif dataset_name == 'enzymes':
        graphs = load_enzymes(data_dir)
    elif dataset_name == 'er':
        graphs = load_erdos_renyi(data_dir)
    elif dataset_name == 'ba':
        graphs = load_barabasi_albert(data_dir)
    elif dataset_name == 'ws':
        graphs = load_watts_strogatz(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Shuffle
    np.random.shuffle(graphs)

    # Split into corpus and eval
    if corpus_size is None:
        corpus_size = int(len(graphs) * 0.9)
    if eval_size is None:
        eval_size = len(graphs) - corpus_size

    # Handle case where requested sizes exceed available graphs
    total_requested = corpus_size + eval_size
    total_available = len(graphs)

    if total_requested > total_available:
        print(f"Warning: Requested {total_requested} graphs but only {total_available} available")
        print(f"  Adjusting: corpus_size={int(total_available*0.9)}, eval_size={total_available - int(total_available*0.9)}")
        corpus_size = int(total_available * 0.9)
        eval_size = total_available - corpus_size

    corpus_graphs = graphs[:corpus_size]
    eval_graphs = graphs[corpus_size:corpus_size + eval_size]

    print(f"Loaded {dataset_name}: {len(corpus_graphs)} corpus + {len(eval_graphs)} eval = {len(corpus_graphs) + len(eval_graphs)} total")

    return corpus_graphs, eval_graphs


def load_qm9(data_dir: str) -> List[nx.Graph]:
    """Load QM9 molecular graphs"""
    print(f"Loading QM9 from {data_dir}")

    try:
        from torch_geometric.datasets import QM9
        from torch_geometric.utils import to_networkx

        qm9_dir = Path(data_dir) / 'qm9'

        # Load PyG dataset
        dataset = QM9(root=str(qm9_dir))

        print(f"  Converting {len(dataset)} PyG graphs to NetworkX...")

        # Convert to NetworkX graphs
        graphs = []
        for i, data in enumerate(dataset):
            # Convert to undirected NetworkX graph
            G = to_networkx(data, node_attrs=['z'], edge_attrs=None, to_undirected=True)

            # Add node types (atom symbols)
            # z contains atomic numbers
            atom_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
            for node in G.nodes():
                z_value = G.nodes[node].get('z', 6)  # default to Carbon
                if isinstance(z_value, (list, tuple)):
                    z_value = z_value[0] if len(z_value) > 0 else 6
                G.nodes[node]['type'] = atom_symbols.get(int(z_value), 'C')

            # Add edge types (all single bonds for QM9 - could be enhanced)
            for u, v in G.edges():
                G.edges[u, v]['type'] = 'single'

            graphs.append(G)

            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1}/{len(dataset)} molecules...")

        print(f"  ✓ Loaded {len(graphs)} QM9 molecules")
        return graphs

    except ImportError:
        print("  ⚠ PyTorch Geometric not available, using synthetic data")
        return _load_synthetic_qm9()
    except Exception as e:
        print(f"  ⚠ Error loading QM9: {e}")
        print("  Falling back to synthetic data")
        return _load_synthetic_qm9()


def load_arxiv(data_dir: str) -> List[nx.Graph]:
    """Load ArXiv citation network subgraphs"""
    print(f"Loading ArXiv from {data_dir}")

    try:
        from ogb.nodeproppred import PygNodePropPredDataset
        import torch
        import torch_geometric.data.data
        import torch_geometric.data.storage

        arxiv_dir = Path(data_dir) / 'arxiv'

        # Fix for PyTorch 2.6+
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
            torch.serialization.add_safe_globals([torch_geometric.data.data.Data])
            torch.serialization.add_safe_globals([torch_geometric.data.data.DataTensorAttr])
            torch.serialization.add_safe_globals([torch_geometric.data.storage.GlobalStorage])

        # Load OGB dataset
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=str(arxiv_dir))
        data = dataset[0]

        print(f"  ArXiv graph: {data.num_nodes} nodes, {data.num_edges} edges")
        print(f"  Extracting ego-networks as individual graphs...")

        # Extract ego-networks (k-hop neighborhoods) as individual graphs
        graphs = []
        edge_index = data.edge_index

        # Sample nodes for ego networks
        num_egos = min(10000, data.num_nodes)  # Limit to 10K ego networks
        sampled_nodes = np.random.choice(data.num_nodes, size=num_egos, replace=False)

        for idx, center_node in enumerate(sampled_nodes):
            # Get 2-hop neighborhood
            neighbors = set([int(center_node)])
            current_frontier = set([int(center_node)])

            for hop in range(2):
                next_frontier = set()
                for node in current_frontier:
                    # Find neighbors
                    mask = (edge_index[0] == node) | (edge_index[1] == node)
                    neighbor_edges = edge_index[:, mask]
                    next_frontier.update(neighbor_edges[0].tolist())
                    next_frontier.update(neighbor_edges[1].tolist())

                neighbors.update(next_frontier)
                current_frontier = next_frontier

            # Create subgraph
            neighbors = list(neighbors)
            if len(neighbors) > 100:  # Limit size
                neighbors = neighbors[:100]

            # Build NetworkX graph
            G = nx.DiGraph()

            # Add nodes with types
            node_labels = data.y[neighbors].squeeze().tolist() if data.y is not None else [0] * len(neighbors)
            subject_areas = ['ML', 'CV', 'NLP', 'Theory', 'Systems']

            for i, node in enumerate(neighbors):
                label = node_labels[i] if isinstance(node_labels, list) else node_labels
                G.add_node(node, type=subject_areas[int(label) % len(subject_areas)])

            # Add edges
            for i in range(edge_index.size(1)):
                src, dst = int(edge_index[0, i]), int(edge_index[1, i])
                if src in neighbors and dst in neighbors:
                    G.add_edge(src, dst, type='citation')

            if G.number_of_nodes() > 5:  # Only keep non-trivial graphs
                graphs.append(G)

            if (idx + 1) % 1000 == 0:
                print(f"    Extracted {idx+1}/{num_egos} ego networks...")

        print(f"  ✓ Extracted {len(graphs)} ArXiv citation subgraphs")
        return graphs

    except Exception as e:
        print(f"  ⚠ Error loading ArXiv: {e}")
        print("  Falling back to synthetic data")
        return _load_synthetic_arxiv()


def load_reddit(data_dir: str) -> List[nx.Graph]:
    """Load Reddit discussion thread graphs"""
    print(f"Loading Reddit from {data_dir}")

    try:
        from torch_geometric.datasets import Reddit2
        import torch

        reddit_dir = Path(data_dir) / 'reddit'

        # Load dataset
        dataset = Reddit2(root=str(reddit_dir))
        data = dataset[0]

        print(f"  Reddit graph: {data.num_nodes} nodes, {data.num_edges} edges")
        print(f"  Extracting community subgraphs...")

        # Extract community-based subgraphs
        graphs = []
        edge_index = data.edge_index
        labels = data.y.numpy() if data.y is not None else np.zeros(data.num_nodes)

        # Group by community labels
        unique_communities = np.unique(labels)
        print(f"  Found {len(unique_communities)} communities")

        # Sample communities to limit dataset size
        num_communities = min(1000, len(unique_communities))
        sampled_communities = np.random.choice(unique_communities, size=num_communities, replace=False)

        for idx, community_id in enumerate(sampled_communities):
            # Get nodes in this community
            community_nodes = np.where(labels == community_id)[0]

            if len(community_nodes) < 10:
                continue

            # Sample if too large
            if len(community_nodes) > 100:
                community_nodes = np.random.choice(community_nodes, size=100, replace=False)

            # Build subgraph
            G = nx.Graph()

            # Add nodes
            for node in community_nodes:
                G.add_node(int(node), type='post')

            # Add edges within community
            for i in range(edge_index.size(1)):
                src, dst = int(edge_index[0, i]), int(edge_index[1, i])
                if src in community_nodes and dst in community_nodes:
                    G.add_edge(src, dst, type='reply')

            if G.number_of_nodes() > 5:
                graphs.append(G)

            if (idx + 1) % 100 == 0:
                print(f"    Extracted {idx+1}/{num_communities} communities...")

        print(f"  ✓ Extracted {len(graphs)} Reddit community subgraphs")
        return graphs

    except Exception as e:
        print(f"  ⚠ Error loading Reddit: {e}")
        print("  Falling back to synthetic data")
        return _load_synthetic_reddit()


def load_protein(data_dir: str) -> List[nx.Graph]:
    """Load protein-protein interaction networks"""
    print(f"Loading Protein PPI from {data_dir}")

    try:
        from torch_geometric.datasets import TUDataset
        from torch_geometric.utils import to_networkx

        protein_dir = Path(data_dir) / 'proteins'

        # Load PROTEINS dataset
        dataset = TUDataset(root=str(protein_dir), name='PROTEINS')

        print(f"  Converting {len(dataset)} protein graphs to NetworkX...")

        graphs = []
        for i, data in enumerate(dataset):
            # Convert to NetworkX
            G = to_networkx(data, node_attrs=['x'], edge_attrs=None, to_undirected=True)

            # Add node types
            protein_types = ['kinase', 'receptor', 'enzyme', 'transcription_factor']
            for node in G.nodes():
                # Use node features if available
                x = G.nodes[node].get('x', [0])
                if isinstance(x, (list, tuple)):
                    type_idx = int(x[0]) if len(x) > 0 else 0
                else:
                    type_idx = int(x)
                G.nodes[node]['type'] = protein_types[type_idx % len(protein_types)]

            # Add edge types
            interaction_types = ['binding', 'phosphorylation', 'activation', 'inhibition']
            for u, v in G.edges():
                G.edges[u, v]['type'] = interaction_types[np.random.randint(0, len(interaction_types))]

            graphs.append(G)

        print(f"  ✓ Loaded {len(graphs)} protein graphs")
        return graphs

    except Exception as e:
        print(f"  ⚠ Error loading PROTEINS: {e}")
        print("  Falling back to synthetic data")
        return _load_synthetic_protein()


# Fallback synthetic data generators

def _load_synthetic_qm9() -> List[nx.Graph]:
    """Generate synthetic molecular graphs"""
    print("  Using synthetic QM9 data (100 graphs)")
    graphs = []
    for i in range(100):
        G = nx.erdos_renyi_graph(n=15, p=0.3)
        for node in G.nodes():
            G.nodes[node]['type'] = np.random.choice(['C', 'N', 'O', 'F'])
        for u, v in G.edges():
            G.edges[u, v]['type'] = np.random.choice(['single', 'double', 'triple'])
        graphs.append(G)
    return graphs


def _load_synthetic_arxiv() -> List[nx.Graph]:
    """Generate synthetic citation networks"""
    print("  Using synthetic ArXiv data (100 graphs)")
    graphs = []
    for i in range(100):
        G = nx.DiGraph()
        n_nodes = np.random.randint(20, 50)
        for node in range(n_nodes):
            G.add_node(node, type=np.random.choice(['ML', 'CV', 'NLP', 'Theory', 'Systems']))
        for node in range(1, n_nodes):
            n_citations = np.random.randint(1, 6)
            targets = np.random.choice(range(node), size=min(n_citations, node), replace=False)
            for target in targets:
                G.add_edge(node, target, type='citation')
        graphs.append(G)
    return graphs


def _load_synthetic_reddit() -> List[nx.Graph]:
    """Generate synthetic discussion threads"""
    print("  Using synthetic Reddit data (100 graphs)")
    graphs = []
    for i in range(100):
        G = nx.Graph()
        n_nodes = np.random.randint(10, 30)
        for node in range(n_nodes):
            G.add_node(node, type=np.random.choice(['post', 'comment']))
        for node in range(1, n_nodes):
            parent = np.random.randint(0, node)
            G.add_edge(node, parent, type='reply')
        graphs.append(G)
    return graphs


def _load_synthetic_protein() -> List[nx.Graph]:
    """Generate synthetic PPI networks"""
    print("  Using synthetic Protein data (100 graphs)")
    graphs = []
    for i in range(100):
        G = nx.erdos_renyi_graph(n=25, p=0.15)
        for node in G.nodes():
            G.nodes[node]['type'] = np.random.choice(['kinase', 'receptor', 'enzyme', 'transcription_factor'])
        for u, v in G.edges():
            G.edges[u, v]['type'] = np.random.choice(['binding', 'phosphorylation', 'activation', 'inhibition'])
        graphs.append(G)
    return graphs


# ============================================================================
# NEW DATASETS
# ============================================================================

def load_zinc(data_dir: str) -> List[nx.Graph]:
    """Load ZINC drug-like molecules"""
    print(f"Loading ZINC from {data_dir}")

    try:
        from torch_geometric.datasets import ZINC
        from torch_geometric.utils import to_networkx

        zinc_dir = Path(data_dir) / 'zinc'

        # Load ZINC dataset (subset=True for manageable size)
        dataset = ZINC(root=str(zinc_dir), subset=True, split='train')

        print(f"  Converting {len(dataset)} PyG graphs to NetworkX...")

        graphs = []
        atom_types = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'S', 5: 'Cl', 6: 'Br', 7: 'I', 8: 'P'}

        for i, data in enumerate(dataset):
            G = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True)

            # Add node types from atom features
            for node in G.nodes():
                x = G.nodes[node].get('x', [0])
                if isinstance(x, (list, tuple)) and len(x) > 0:
                    atom_idx = int(x[0]) if isinstance(x[0], (int, float)) else 0
                else:
                    atom_idx = 0
                G.nodes[node]['type'] = atom_types.get(atom_idx, 'C')

            # Add edge types
            for u, v in G.edges():
                edge_attr = G.edges[u, v].get('edge_attr', [0])
                if isinstance(edge_attr, (list, tuple)) and len(edge_attr) > 0:
                    bond_type = int(edge_attr[0]) if isinstance(edge_attr[0], (int, float)) else 0
                else:
                    bond_type = 0
                bond_names = {0: 'single', 1: 'double', 2: 'triple', 3: 'aromatic'}
                G.edges[u, v]['type'] = bond_names.get(bond_type, 'single')

            graphs.append(G)

            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1}/{len(dataset)} molecules...")

        print(f"  ✓ Loaded {len(graphs)} ZINC molecules")
        return graphs

    except Exception as e:
        print(f"  ⚠ Error loading ZINC: {e}")
        print("  Falling back to synthetic data")
        return _load_synthetic_qm9()  # Similar to QM9


def load_cora(data_dir: str) -> List[nx.Graph]:
    """Load Cora citation network as ego-networks"""
    print(f"Loading Cora from {data_dir}")

    try:
        from torch_geometric.datasets import Planetoid
        import torch

        cora_dir = Path(data_dir) / 'cora'

        dataset = Planetoid(root=str(cora_dir), name='Cora')
        data = dataset[0]

        print(f"  Cora graph: {data.num_nodes} nodes, {data.num_edges} edges")
        print(f"  Extracting ego-networks...")

        graphs = []
        edge_index = data.edge_index

        # Sample nodes for ego networks
        num_egos = min(2000, data.num_nodes)
        sampled_nodes = np.random.choice(data.num_nodes, size=num_egos, replace=False)

        for idx, center_node in enumerate(sampled_nodes):
            # Get 2-hop neighborhood
            neighbors = set([int(center_node)])
            current_frontier = set([int(center_node)])

            for hop in range(2):
                next_frontier = set()
                for node in current_frontier:
                    mask = (edge_index[0] == node) | (edge_index[1] == node)
                    neighbor_edges = edge_index[:, mask]
                    next_frontier.update(neighbor_edges[0].tolist())
                    next_frontier.update(neighbor_edges[1].tolist())
                neighbors.update(next_frontier)
                current_frontier = next_frontier

            neighbors = list(neighbors)
            if len(neighbors) > 50:
                neighbors = neighbors[:50]

            # Build NetworkX graph
            G = nx.Graph()

            # Add nodes with types (paper categories)
            categories = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                         'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
            node_labels = data.y[neighbors].tolist() if data.y is not None else [0] * len(neighbors)

            for i, node in enumerate(neighbors):
                label = node_labels[i] if isinstance(node_labels, list) else node_labels
                G.add_node(node, type=categories[int(label) % len(categories)])

            # Add edges
            for i in range(edge_index.size(1)):
                src, dst = int(edge_index[0, i]), int(edge_index[1, i])
                if src in neighbors and dst in neighbors:
                    G.add_edge(src, dst, type='citation')

            if G.number_of_nodes() > 5:
                graphs.append(G)

            if (idx + 1) % 500 == 0:
                print(f"    Extracted {idx+1}/{num_egos} ego networks...")

        print(f"  ✓ Extracted {len(graphs)} Cora citation subgraphs")
        return graphs

    except Exception as e:
        print(f"  ⚠ Error loading Cora: {e}")
        print("  Falling back to synthetic data")
        return _load_synthetic_arxiv()


def load_citeseer(data_dir: str) -> List[nx.Graph]:
    """Load CiteSeer citation network as ego-networks"""
    print(f"Loading CiteSeer from {data_dir}")

    try:
        from torch_geometric.datasets import Planetoid

        citeseer_dir = Path(data_dir) / 'citeseer'

        dataset = Planetoid(root=str(citeseer_dir), name='CiteSeer')
        data = dataset[0]

        print(f"  CiteSeer graph: {data.num_nodes} nodes, {data.num_edges} edges")
        print(f"  Extracting ego-networks...")

        graphs = []
        edge_index = data.edge_index

        num_egos = min(2000, data.num_nodes)
        sampled_nodes = np.random.choice(data.num_nodes, size=num_egos, replace=False)

        for idx, center_node in enumerate(sampled_nodes):
            neighbors = set([int(center_node)])
            current_frontier = set([int(center_node)])

            for hop in range(2):
                next_frontier = set()
                for node in current_frontier:
                    mask = (edge_index[0] == node) | (edge_index[1] == node)
                    neighbor_edges = edge_index[:, mask]
                    next_frontier.update(neighbor_edges[0].tolist())
                    next_frontier.update(neighbor_edges[1].tolist())
                neighbors.update(next_frontier)
                current_frontier = next_frontier

            neighbors = list(neighbors)
            if len(neighbors) > 50:
                neighbors = neighbors[:50]

            G = nx.Graph()
            categories = ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI']
            node_labels = data.y[neighbors].tolist() if data.y is not None else [0] * len(neighbors)

            for i, node in enumerate(neighbors):
                label = node_labels[i] if isinstance(node_labels, list) else node_labels
                G.add_node(node, type=categories[int(label) % len(categories)])

            for i in range(edge_index.size(1)):
                src, dst = int(edge_index[0, i]), int(edge_index[1, i])
                if src in neighbors and dst in neighbors:
                    G.add_edge(src, dst, type='citation')

            if G.number_of_nodes() > 5:
                graphs.append(G)

            if (idx + 1) % 500 == 0:
                print(f"    Extracted {idx+1}/{num_egos} ego networks...")

        print(f"  ✓ Extracted {len(graphs)} CiteSeer citation subgraphs")
        return graphs

    except Exception as e:
        print(f"  ⚠ Error loading CiteSeer: {e}")
        print("  Falling back to synthetic data")
        return _load_synthetic_arxiv()


def load_enzymes(data_dir: str) -> List[nx.Graph]:
    """Load ENZYMES protein tertiary structure dataset"""
    print(f"Loading ENZYMES from {data_dir}")

    try:
        from torch_geometric.datasets import TUDataset
        from torch_geometric.utils import to_networkx

        enzymes_dir = Path(data_dir) / 'enzymes'

        dataset = TUDataset(root=str(enzymes_dir), name='ENZYMES')

        print(f"  Converting {len(dataset)} enzyme graphs to NetworkX...")

        graphs = []
        enzyme_classes = ['EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6']

        for i, data in enumerate(dataset):
            G = to_networkx(data, node_attrs=['x'], edge_attrs=None, to_undirected=True)

            # Add node types
            for node in G.nodes():
                x = G.nodes[node].get('x', [0])
                if isinstance(x, (list, tuple)) and len(x) > 0:
                    type_idx = int(x[0]) % len(enzyme_classes)
                else:
                    type_idx = 0
                G.nodes[node]['type'] = enzyme_classes[type_idx]

            # Add edge types
            for u, v in G.edges():
                G.edges[u, v]['type'] = 'structure'

            graphs.append(G)

        print(f"  ✓ Loaded {len(graphs)} ENZYMES graphs")
        return graphs

    except Exception as e:
        print(f"  ⚠ Error loading ENZYMES: {e}")
        print("  Falling back to synthetic data")
        return _load_synthetic_protein()


def load_erdos_renyi(data_dir: str, n_graphs: int = 1000) -> List[nx.Graph]:
    """Generate Erdős-Rényi random graphs (benchmark)"""
    print(f"Generating {n_graphs} Erdős-Rényi random graphs")

    graphs = []
    for i in range(n_graphs):
        n_nodes = np.random.randint(10, 50)
        p = np.random.uniform(0.1, 0.4)

        G = nx.erdos_renyi_graph(n=n_nodes, p=p)

        # Add random node types
        node_types = ['A', 'B', 'C', 'D']
        for node in G.nodes():
            G.nodes[node]['type'] = np.random.choice(node_types)

        # Add random edge types
        edge_types = ['e1', 'e2', 'e3']
        for u, v in G.edges():
            G.edges[u, v]['type'] = np.random.choice(edge_types)

        graphs.append(G)

    print(f"  ✓ Generated {len(graphs)} ER graphs")
    return graphs


def load_barabasi_albert(data_dir: str, n_graphs: int = 1000) -> List[nx.Graph]:
    """Generate Barabási-Albert scale-free graphs (benchmark)"""
    print(f"Generating {n_graphs} Barabási-Albert scale-free graphs")

    graphs = []
    for i in range(n_graphs):
        n_nodes = np.random.randint(15, 60)
        m = np.random.randint(2, 5)  # edges to attach from new node

        G = nx.barabasi_albert_graph(n=n_nodes, m=m)

        # Add random node types
        node_types = ['A', 'B', 'C', 'D']
        for node in G.nodes():
            G.nodes[node]['type'] = np.random.choice(node_types)

        # Add random edge types
        edge_types = ['e1', 'e2', 'e3']
        for u, v in G.edges():
            G.edges[u, v]['type'] = np.random.choice(edge_types)

        graphs.append(G)

    print(f"  ✓ Generated {len(graphs)} BA graphs")
    return graphs


def load_watts_strogatz(data_dir: str, n_graphs: int = 1000) -> List[nx.Graph]:
    """Generate Watts-Strogatz small-world graphs (benchmark)"""
    print(f"Generating {n_graphs} Watts-Strogatz small-world graphs")

    graphs = []
    for i in range(n_graphs):
        n_nodes = np.random.randint(15, 60)
        k = min(np.random.randint(4, 10), n_nodes - 1)
        p = np.random.uniform(0.1, 0.5)

        G = nx.watts_strogatz_graph(n=n_nodes, k=k, p=p)

        # Add random node types
        node_types = ['A', 'B', 'C', 'D']
        for node in G.nodes():
            G.nodes[node]['type'] = np.random.choice(node_types)

        # Add random edge types
        edge_types = ['e1', 'e2', 'e3']
        for u, v in G.edges():
            G.edges[u, v]['type'] = np.random.choice(edge_types)

        graphs.append(G)

    print(f"  ✓ Generated {len(graphs)} WS graphs")
    return graphs
