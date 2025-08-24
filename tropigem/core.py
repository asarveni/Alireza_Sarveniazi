import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import deque
import logging
import random
from scipy.linalg import eigh  # NEU: Für Eigenvektorzerlegung

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GraphEmbeddingSystem')
logger.setLevel(logging.INFO)

# --- TROPISCHE ALGEBRA FUNKTIONEN ---
def tropical_addition(a, b):
    """Tropical Addition: a ⊕ b = max(a, b)"""
    return np.maximum(a, b)

def tropical_multiplication(a, b):
    """Tropical Multiplication: a ⊗ b = a + b"""
    return a + b

def tropical_matrix_mult(A, B):
    """Tropical Matrix Multiplication using (⊕, ⊗) algebra"""
    n, m = A.shape
    p = B.shape[1]
    result = np.full((n, p), -np.inf)
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                product = tropical_multiplication(A[i, k], B[k, j])
                result[i, j] = tropical_addition(result[i, j], product)
    
    return result

def compute_tropical_distances(graph):
    """Berechnet tropische Distanzen zwischen allen Knotenpaaren"""
    n_nodes = len(graph.nodes())
    dist_matrix = np.full((n_nodes, n_nodes), np.inf)
    
    # Initialisiere Diagonale mit 0 (neutrales Element für ⊕)
    np.fill_diagonal(dist_matrix, 0)
    
    # Setze direkte Kantengewichte (hier: 1 für ungewichtete Graphen)
    for i, j in graph.edges():
        dist_matrix[i, j] = 1
        dist_matrix[j, i] = 1
    
    # Tropische Matrix-Potenzierung für kürzeste Pfade
    # (Tropische Version des Floyd-Warshall Algorithmus)
    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                # Tropische Algebra: dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])
                dist_matrix[i, j] = min(dist_matrix[i, j], dist_matrix[i, k] + dist_matrix[k, j])
    
    return dist_matrix

# --- DYNAMIC EMBEDDER KLASSE ---
class DynamicEmbedder:
    def __init__(self, initial_graph, embedding_dim=2):
        self.graph = initial_graph
        self.embedding_dim = embedding_dim
        self.embeddings = None
        logger.info(f"DynamicEmbedder initialized with {len(initial_graph.nodes())} nodes")
    
    def tropical_fit(self):
        """Fortgeschrittenes tropisches Embedding mit Eigenvektorzerlegung"""
        logger.info("Starting ADVANCED tropical embedding...")
        
        n_nodes = len(self.graph.nodes())
        
        # 1. Tropische Distanzmatrix berechnen
        dist_matrix = compute_tropical_distances(self.graph)
        logger.info(f"Tropical distance matrix computed: {dist_matrix.shape}")
        
        # 2. Tropische Zentrierung (tropical double centering)
        row_means = np.mean(dist_matrix, axis=1)
        col_means = np.mean(dist_matrix, axis=0)
        grand_mean = np.mean(dist_matrix)
        
        # Zentrierungsmatrix
        centered_matrix = dist_matrix - row_means[:, np.newaxis] - col_means[np.newaxis, :] + grand_mean
        
        # 3. Eigenvektorzerlegung (tropische PCA-ähnlich)
        eigenvalues, eigenvectors = eigh(-0.5 * centered_matrix)
        
        # 4. Wähle die wichtigsten Eigenvektoren für Embedding
        self.embeddings = eigenvectors[:, -self.embedding_dim:]
        
        logger.info("Advanced tropical embedding completed")
        return self
    
    def fit(self):
        """Train the embedding model"""
        return self.tropical_fit()
    
    def fit_transform(self):
        """Convenience method: fit and return embeddings"""
        self.fit()
        return self.embeddings
    
    def visualize(self, title="Tropical Graph Embedding"):
        """Visualize the embeddings"""
        if self.embeddings is None:
            self.fit()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.embeddings[:, 0], self.embeddings[:, 1], alpha=0.7)
        
        # Knotenbeschriftungen hinzufügen
        for i, (x, y) in enumerate(self.embeddings):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        plt.title(title)
        plt.xlabel("Tropical Dimension 1")
        plt.ylabel("Tropical Dimension 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()