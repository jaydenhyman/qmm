# Getting Started

## Installation

### With pip

```sh
pip install qmm-core
```

### With uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first:

=== "Mac/Linux"

    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```sh
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

Then create a project and add qmm:

```sh
uv init my-project
cd my-project
uv add qmm-core
```

## Using with Jupyter

### With pip

```sh
pip install qmm-core jupyterlab
jupyter lab
```

### With uv

```sh
uv add jupyterlab
uv run jupyter lab
```

## Basic Usage

```python
import qmm

# Load a built-in example model
G = qmm.load_digraph("snowshoe")

# View the model structure
qmm.create_matrix(G, form="signed")
# Matrix([
# [-1, -1,  0],
# [ 1,  0, -1],
# [ 1,  1, -1]])

# Analyse feedback cycles
qmm.cycles_table(G)
#    Length                             Cycle Sign
# 0       1                   P $\multimap$ P    −
# 1       1                   R $\multimap$ R    −
# 2       2   C $\rightarrow$ P $\multimap$ C    −
# 3       2   C $\multimap$ R $\rightarrow$ C    −

# Generate qualitative predictions
qmm.qualitative_predictions(G)
# Matrix([
# [+, −, +],
# [?, +, −],
# [+, ?, +]])
```

## Creating Models

### Using Digraph Builder

The easiest way to create models is with [Digraph Builder](https://www.digraphbuilder.com/), an interactive web application for building signed digraphs.

1. Create your model in Digraph Builder
2. Export as JSON
3. Load in Python:

```python
G = qmm.load_digraph("path/to/model.json")
```

### From an adjacency matrix

Use `list_to_digraph` to create a model from a signed adjacency matrix:

```python
from qmm import list_to_digraph

# Snowshoe hare model: V (vegetation), H (hare), P (predator)
# Matrix rows/cols are in order: V, H, P
G = list_to_digraph(
    [[-1, -1,  0],   # V: self-regulation, suppressed by H
     [ 1,  0, -1],   # H: benefits from V, suppressed by P
     [ 1,  1, -1]],  # P: benefits from V and H, self-regulation
    ids=['V', 'H', 'P']
)
```

### Using NetworkX directly

```python
import networkx as nx

# Snowshoe hare model: V (vegetation), H (hare), P (predator)
G = nx.DiGraph()

# Add state nodes
G.add_node("V", category="state")  # Vegetation
G.add_node("H", category="state")  # Hare
G.add_node("P", category="state")  # Predator

# Add edges with signs (+1 or -1)
G.add_edge("V", "V", sign=-1)  # V self-regulation
G.add_edge("V", "H", sign=1)   # V increases H
G.add_edge("V", "P", sign=1)   # V increases P
G.add_edge("H", "V", sign=-1)  # H decreases V
G.add_edge("H", "P", sign=1)   # H increases P
G.add_edge("P", "H", sign=-1)  # P decreases H
G.add_edge("P", "P", sign=-1)  # P self-regulation
```

## Next Steps

- Explore the [API Reference](reference.md) for all available functions
- Try the built-in example models: `snowshoe`, `snowshoe_io`, `mesocosm`
