"""Define model structure in graph, matrix or equation forms."""

import json
from typing import Union, List, Dict, Tuple, Literal
import networkx as nx
import pandas as pd
import sympy as sp
from .helper import get_nodes


def import_digraph(data: Union[str, dict], file_path: bool = True) -> nx.DiGraph:
    """Import a JSON model and convert to a NetworkX DiGraph with sign attributes.

    Args:
        data: Path to JSON file or dictionary containing model structure
        file_path: If True, data is a file path. If False, data is a dictionary

    Returns:
        nx.DiGraph: Signed directed graph (signed digraph)

    Examples:
        ```python
        from qmm import import_digraph
        model_dict = {
        "nodes": [{"id": "A"}, {"id": "B"}],
        "edges": [{"from": "A", "to": "B", "arrows": {"to": {"type": "triangle"}}}]
        }
        list(import_digraph(model_dict, file_path=False).nodes())
        # ['A', 'B']

        G = import_digraph(model_dict, file_path=False)
        list(G.edges(data='sign'))
        # [('A', 'B', 1)]
        ```
    """
    if file_path:
        with open(data, "r") as file:
            data = json.load(file)
    G = nx.DiGraph()
    for node in data["nodes"]:
        att = {k: v for k, v in node.items() if k != "id"}
        if "title" not in att:
            att["title"] = None
        G.add_node(str(node["id"]), **att)
    for edge in data["edges"]:
        source, target = str(edge["from"]), str(edge["to"])
        att = {k: v for k, v in edge.items() if k not in ["from", "to", "arrows"]}
        arr = edge.get("arrows", {}).get("to", {})
        if isinstance(arr, dict):
            arr_type = arr.get("type")
            if arr_type == "triangle":
                att["sign"] = 1
            elif arr_type == "circle":
                att["sign"] = -1
        if "dashes" not in att:
            att["dashes"] = False
        if "title" not in att:
            att["title"] = None
        G.add_edge(source, target, **att)
    nx.set_node_attributes(G, "state", "category")
    nx.freeze(G)
    return G


def create_matrix(
    G: nx.DiGraph,
    form: Literal["symbolic", "signed", "binary"] = "symbolic",
    matrix_type: Literal["A", "B", "C", "D"] = "A",
) -> sp.Matrix:
    """Create an interaction matrix from a signed digraph in symbolic, signed, or binary form.

    Args:
        G: NetworkX DiGraph representing a signed digraph model
        form: Type of matrix elements ('symbolic', 'signed', or 'binary')
        matrix_type: Type of matrix to create ('A', 'B', 'C', or 'D')

    Returns:
        sp.Matrix: Interaction matrix

    References:
        - Levins, R. (1968). Evolution in changing environments; some theoretical explorations.
        - Dambacher, J.M., Li, H.W., Rossignol, P.A. (2002). Relevance of Community Structure in Assessing Indeterminacy of Ecological Predictions. Ecology 83, 1372–1385.

    Examples:
        ```python
        from qmm import load_digraph, create_matrix
        create_matrix(load_digraph("snowshoe"), form='symbolic')
        # Matrix([
        # [-a_V,V, -a_V,H,      0],
        # [ a_H,V,      0, -a_H,P],
        # [ a_P,V,  a_P,H, -a_P,P]])

        create_matrix(load_digraph("snowshoe"), form='signed')
        # Matrix([
        # [-1, -1,  0],
        # [ 1,  0, -1],
        # [ 1,  1, -1]])

        create_matrix(load_digraph("snowshoe"), form='binary')
        # Matrix([
        # [1, 1, 0],
        # [1, 0, 1],
        # [1, 1, 1]])
        ```
    """

    def sym(source: str, target: str, prefix: str) -> sp.Symbol:
        return sp.Symbol(f"{prefix}_{target},{source}")

    def sign(source: str, target: str, prefix: str) -> Union[sp.Symbol, int]:
        if form == "symbolic":
            return sym(source, target, prefix) * G[source][target].get("sign", 1)
        elif form == "signed":
            return G[source][target].get("sign", 1)
        else:  # form == 'binary'
            return int(G.has_edge(source, target))

    def product(path: List[str]) -> Union[sp.Symbol, int]:
        effect = 1
        for i in range(len(path) - 1):
            effect *= sign(path[i], path[i + 1], prefix)
        return effect

    state_n = get_nodes(G, "state")
    input_n = get_nodes(G, "input")
    output_n = get_nodes(G, "output")
    matrix_configs: Dict[str, Tuple[List[str], List[str], str, str]] = {
        "A": (state_n, state_n, "a", "state"),
        "B": (state_n, input_n, "b", "input"),
        "C": (output_n, state_n, "c", "output"),
        "D": (output_n, input_n, "d", "input"),
    }
    rows, cols, prefix, category = matrix_configs[matrix_type]
    matrix = sp.zeros(len(rows), len(cols))
    if matrix_type == "D":
        for inp in input_n:
            for out in output_n:
                if G.has_edge(inp, out):
                    raise ValueError(
                        f"Direct input to output edge ({inp} to {out}) not supported."
                    )
        return matrix
    for i, target in enumerate(rows):
        for j, source in enumerate(cols):
            if matrix_type == "A":
                if G.has_edge(source, target):
                    matrix[i, j] = sign(source, target, prefix)
            else:
                paths = nx.all_simple_paths(G, source, target)
                valid = [p for p in paths if all(G.nodes[n]["category"] == category for n in p[1:-1])]
                matrix[i, j] = sum(product(path) for path in valid)
    return matrix


def create_equations(G: nx.DiGraph, form: Literal["state", "output"] = "state") -> sp.Matrix:
    """Create linear system of differential equations from a signed digraph.

    Args:
        G: NetworkX DiGraph representing a signed digraph model
        form: Type of equations to create ('state' or 'output')

    Returns:
        sp.Matrix: Linear system of differential equations

    References:
        Levins, R. (1974). The qualitative analysis of partially specified systems. Annals of the New York Academy of Sciences 231, 123–138.

    Examples:
        ```python
        from qmm import load_digraph, create_equations
        create_equations(load_digraph("snowshoe"), form='state')
        # Matrix([
        # [           -a_V,H*x_H - a_V,V*x_V],
        # [           -a_H,P*x_P + a_H,V*x_V],
        # [a_P,H*x_H - a_P,P*x_P + a_P,V*x_V]])
        ```
    """
    A = create_matrix(G, form="symbolic", matrix_type="A")
    B = create_matrix(G, form="symbolic", matrix_type="B")
    C = create_matrix(G, form="symbolic", matrix_type="C")
    D = create_matrix(G, form="symbolic", matrix_type="D")
    state_nodes = get_nodes(G, "state")
    input_nodes = get_nodes(G, "input")
    output_nodes = get_nodes(G, "output")
    x = sp.Matrix([sp.Symbol(f"x_{i}") for i in state_nodes])
    u = sp.Matrix([sp.Symbol(f"u_{i}") for i in input_nodes]) if input_nodes else None
    if form == "state":
        equations = A * x
        if B.shape[1] > 0 and u is not None:
            equations += B * u
        return equations
    if form != "output":
        raise ValueError("form must be either 'state' or 'output'")
    if not output_nodes:
        raise ValueError("No output nodes found in graph")
    equations = C * x
    if D.shape[1] > 0 and u is not None:
        equations += D * u
    return equations


def nodes_table(G: nx.DiGraph) -> pd.DataFrame:
    """Create a table of node metadata.

    Args:
        G: NetworkX DiGraph representing signed digraph model

    Returns:
        pd.DataFrame: Node, label, category, and description

    Examples:
        ```python
        from qmm import load_digraph, nodes_table
        nodes_table(load_digraph("snowshoe"))
        #       Node Label Category Description
        # 0  $x_{V}$     V    State        None
        # 1  $x_{H}$     H    State        None
        # 2  $x_{P}$     P    State        None
        ```
    """
    rows = []
    for node_id, data in G.nodes(data=True):
        category = data.get("category", "state")
        if category == "input":
            symbol = f"u_{{{node_id}}}"
        elif category == "output":
            symbol = f"y_{{{node_id}}}"
        else:
            symbol = f"x_{{{node_id}}}"
        if category == "input":
            category_label = "Input"
        elif category == "output":
            category_label = "Output"
        else:
            category_label = "State"
        rows.append(
            {
                "Node": f"${symbol}$",
                "Label": data.get("label", str(node_id)),
                "Category": category_label,
                "Description": data.get("title"),
            }
        )
    return pd.DataFrame(rows)


def edges_table(G: nx.DiGraph) -> pd.DataFrame:
    """Create a table of edge metadata.

    Args:
        G: NetworkX DiGraph representing signed digraph model

    Returns:
        pd.DataFrame: Edge, from, sign, to, dashes, and description

    Examples:
        ```python
        from qmm import load_digraph, edges_table
        edges_table(load_digraph("snowshoe"))
        #         Edge From Sign To  Dashes Description
        # 0  $a_{V,V}$    V    -  V   False        None
        # 1  $a_{H,V}$    V    +  H   False        None
        # 2  $a_{P,V}$    V    +  P   False        None
        # 3  $a_{V,H}$    H    -  V   False        None
        # 4  $a_{P,H}$    H    +  P   False        None
        # 5  $a_{H,P}$    P    -  H   False        None
        # 6  $a_{P,P}$    P    -  P   False        None
        ```
    """
    rows = []
    for source, target, data in G.edges(data=True):
        source_cat = G.nodes[source].get("category", "state")
        target_cat = G.nodes[target].get("category", "state")
        if source_cat == "input" and target_cat == "output":
            prefix = "d"
        elif source_cat == "input":
            prefix = "b"
        elif target_cat == "output":
            prefix = "c"
        else:
            prefix = "a"
        sign_val = data.get("sign", 1)
        if sign_val == 1:
            sign_label = "+"
        elif sign_val == -1:
            sign_label = "-"
        else:
            sign_label = str(sign_val)
        rows.append(
            {
                "Edge": f"${prefix}_{{{target},{source}}}$",
                "From": G.nodes[source].get("label", str(source)),
                "Sign": sign_label,
                "To": G.nodes[target].get("label", str(target)),
                "Dashes": data.get("dashes", False),
                "Description": data.get("title"),
            }
        )
    return pd.DataFrame(rows)
