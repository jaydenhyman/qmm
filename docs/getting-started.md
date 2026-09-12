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

The examples use the three-node `snowshoe_rp` model. Matrix rows are affected
nodes and columns are affecting nodes, in the order R (resource), C (consumer),
P (predator). Call `configure_pandas_display()` explicitly to show complete
tables; QMM leaves pandas display settings unchanged on import.

```pycon
>>> import qmm
>>> qmm.configure_pandas_display()
>>> G = qmm.load_digraph("snowshoe_rp")

>>> qmm.create_matrix(G, form="signed")
Matrix([
[-1, -1,  0],
[ 1,  0, -1],
[ 1,  1, -1]])

>>> qmm.cycles_table(G)
   Length                                          Cycle Sign
0       1                                P $\multimap$ P    −
1       1                                R $\multimap$ R    −
2       2                C $\rightarrow$ P $\multimap$ C    −
3       2                C $\multimap$ R $\rightarrow$ C    −
4       3  C $\multimap$ R $\rightarrow$ P $\multimap$ C    +

>>> qmm.qualitative_predictions(G)
Matrix([
[+, −, +],
[?, +, −],
[+, ?, +]])

```

## Creating Models

### Using Digraph Builder

The easiest way to create models is with [Digraph Builder](https://www.digraphbuilder.com/), an interactive web application for building signed digraphs.

1. Create your model in Digraph Builder
2. Export as JSON
3. Load in Python:

```python
G = qmm.import_digraph("path/to/model.json")
```

### From an adjacency matrix

Use `list_to_digraph` to create a model from a signed adjacency matrix:

```python
from qmm import list_to_digraph

# Same snowshoe_rp model: R (resource), C (consumer), P (predator)
# Matrix rows/cols are in order: R, C, P
G = list_to_digraph(
    [[-1, -1,  0],
     [ 1,  0, -1],
     [ 1,  1, -1]],
    ids=['R', 'C', 'P']
)
```

### Using NetworkX directly

```python
import networkx as nx

# Same snowshoe_rp model
G = nx.DiGraph()

# Add state nodes
G.add_node("R", category="state")  # Resource
G.add_node("C", category="state")  # Consumer
G.add_node("P", category="state")  # Predator

# Add edges with signs (+1 or -1)
G.add_edge("R", "R", sign=-1)  # R self-regulation
G.add_edge("R", "C", sign=1)   # R increases C
G.add_edge("R", "P", sign=1)   # R increases P
G.add_edge("C", "R", sign=-1)  # C decreases R
G.add_edge("C", "P", sign=1)   # C increases P
G.add_edge("P", "C", sign=-1)  # P decreases C
G.add_edge("P", "P", sign=-1)  # P self-regulation
```

## Next Steps

- Explore the [API Reference](reference.md) for all available functions
- Try the built-in example models: `snowshoe`, `snowshoe_rp`, `snowshoe_io`, `mesocosm`
