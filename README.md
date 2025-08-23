# TropiGEM - Tropical Graph Embedding Machine 🌴

**The world's first tropical math-based graph embedding library**

```python
from tropigem import DynamicEmbedder
import networkx as nx

G = nx.karate_club_graph()
embedder = DynamicEmbedder(G, dim=2)
embedder.visualize()
```

## Authorship
Developed by Alireza Sarve Niazi in August 2025.
All rights reserved.

[GitHub Repository](https://github.com/AlirezaSarveNiazi/tropigem)
[build-system]
requires = ["setuptools>=61.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "tropigem"
version = "1.0.0"
description = "TropiGEM: Tropical Graph Embedding Machine"
authors = [{ name = "Alireza Sarve Niazi", email = "alireza.sarveniazi@gmail.com" }]
dependencies = [
    "numpy",
    "scipy", 
    "scikit-learn",
    "matplotlib",
    "networkx"
]