import pandas as pd
import networkx as nx
import sympy as sp
from functools import cache
from ..core.structure import create_matrix
from ..core.stability import system_feedback, net_feedback, absolute_feedback, weighted_feedback
from ..core.helper import get_nodes, get_weight, get_positive, get_negative, sign_string, arrows

@cache
def get_paths(G, source=None, target=None, form="symbolic"):
    nodes = get_nodes(G, "state")
    A = create_matrix(G, form=form)

    def calculate_paths(s, t):
        if s not in nodes or t not in nodes:
            raise ValueError("Source or target node not found in the graph")
        path_nodes = list(nx.all_simple_paths(G, s, t))
        if s == t:
            return [A[nodes.index(s), nodes.index(t)]]
        elif not path_nodes:
            return [sp.Integer(0)]
        return [
            sp.prod(
                A[nodes.index(p[i + 1]), nodes.index(p[i])] for i in range(len(p) - 1)
            )
            for p in path_nodes
        ]

    if source is None and target is None:
        return sp.Matrix([[sum(calculate_paths(s, t)) for s in nodes] for t in nodes])
    elif source is not None and target is not None:
        if source not in nodes or target not in nodes:
            raise ValueError("Source or target node not found in the graph")
        if source == target:
            return sp.Matrix([A[nodes.index(source), nodes.index(target)]])
        elif not nx.has_path(G, source, target):
            return sp.Matrix([sp.Integer(0)])
        else:
            return sp.Matrix(calculate_paths(source, target))
    else:
        raise ValueError("Both source and target must be specified.")


def paths_table(G, source=None, target=None, form="symbolic"):
    nodes = get_nodes(G, "state")

    def calculate_paths(s, t):
        if s not in nodes or t not in nodes:
            raise ValueError("Source or target node not found in the graph")
        if s == t:
            return [[s, s]] if G.has_edge(s, s) else None
        if not nx.has_path(G, s, t):
            return None
        paths = list(nx.all_simple_paths(G, s, t))
        return paths

    all_paths = []
    if source is None and target is None:
        for s in nodes:
            for t in nodes:
                paths = calculate_paths(s, t)
                if paths is not None:
                    all_paths.extend(paths)
    elif source is not None and target is not None:
        all_paths = calculate_paths(source, target)
    else:
        raise ValueError("Both source and target must be specified.")
    if not all_paths:
        return None
    net_fb = complementary_feedback(G, source=source, target=target, form="signed")
    absolute_fb = complementary_feedback(G, source=source, target=target, form="binary")

    positive_fb = get_positive(net_fb, absolute_fb)
    negative_fb = get_negative(net_fb, absolute_fb)
    weighted_fb = get_weight(net_fb, absolute_fb, 0)
    prediction_weight = weighted_paths(G, source, target, input="positive")

    paths_df = pd.DataFrame(
        {
            "Length": [len(path) - 1 for path in all_paths],
            "Path": [arrows(G, path) for path in all_paths],
            "Path sign": [sign_string(G, path) for path in all_paths],
            "Positive feedback": [positive_fb[i] for i in range(len(all_paths))],
            "Negative feedback": [negative_fb[i] for i in range(len(all_paths))],
            "Weighted feedback": [weighted_fb[i] for i in range(len(all_paths))],
            "Weighted path": [prediction_weight[i] for i in range(len(all_paths))],
        }
    )
    return paths_df.sort_values(["Length", "Path"]).reset_index(drop=True)


def get_cycles(G):
    A = create_matrix(G, form="symbolic")
    nodes = get_nodes(G, "state")
    node_id = {n: i for i, n in enumerate(nodes)}
    cycle_list = nx.simple_cycles(G)
    cycle_nodes = sorted([c for c in cycle_list], key=lambda x: len(x))
    C = [c + [c[0]] for c in cycle_nodes]
    cycles = sp.Matrix(
        [
            sp.prod([A[node_id[c[i + 1]], node_id[c[i]]] for i in range(len(c) - 1)])
            for c in C
        ]
    )
    return cycles


def cycles_table(G):
    cycle_nodes = sorted(
        [path for path in nx.simple_cycles(G)], key=lambda x: (len(x), x)
    )
    all_cycles = [cycle + [cycle[0]] for cycle in cycle_nodes]
    cycle_signs = [sign_string(G, path) for path in all_cycles]
    cycles_df = pd.DataFrame(
        {
            "Length": [len(nodes) for nodes in cycle_nodes],
            "Cycle": [arrows(G, path) for path in all_cycles],
            "Sign": cycle_signs,
        }
    )
    return cycles_df


@cache
def complementary_feedback(G, source=None, target=None, form="symbolic"):
    nodes = get_nodes(G, "state")
    n = len(nodes)

    def calculate_feedback(s, t):
        path_nodes = list(nx.all_simple_paths(G, s, t))
        feedback = []
        for path in path_nodes:
            path_nodes_set = set(path)
            subsystem_nodes = [node for node in nodes if node not in path_nodes_set]
            subsystem = G.subgraph(subsystem_nodes).copy()
            level = n - len(path)
            if form == "symbolic":
                feedback.append(system_feedback(subsystem, level=level)[0])
            elif form == "signed":
                feedback.append(net_feedback(subsystem, level=level)[0])
            elif form == "binary":
                feedback.append(absolute_feedback(subsystem, level=level)[0])
            else:
                raise ValueError(
                    "Invalid form. Choose 'symbolic', 'signed', or 'binary'."
                )
        return [sp.expand_mul(f) for f in feedback]

    if source is None and target is None:
        return sp.Matrix(
            [[sum(calculate_feedback(s, t)) for s in nodes] for t in nodes]
        )
    elif source is not None and target is not None:
        return sp.Matrix(calculate_feedback(source, target))
    else:
        raise ValueError(
            "Both source and target must be specified, or both must be None."
        )


@cache
def system_paths(G, source=None, target=None, input="positive", form="symbolic"):
    def calculate_effect(s, t):
        if s == t:
            feedback = complementary_feedback(G, s, t, form=form)
            if form == "binary":
                return feedback
            else:
                return feedback / sp.Integer(-1)
        else:
            path = get_paths(G, s, t, form=form)
            feedback = complementary_feedback(G, s, t, form=form)
            if path.shape == (0, 0) or feedback.shape == (0, 0):
                return sp.Matrix([[0]])
            if form == "binary":
                effect = path.multiply_elementwise(feedback)
            else:
                effect = path.multiply_elementwise(feedback) / sp.Integer(-1)
            input_multiplier = sp.Integer(1) if input == "positive" else sp.Integer(-1)
            effect = effect.applyfunc(lambda x: x * input_multiplier)
            return sp.Matrix([sp.expand_mul(e) for e in effect])

    nodes = get_nodes(G, "state")
    if source is None and target is None:
        return sp.Matrix([[sum(calculate_effect(s, t)) for s in nodes] for t in nodes])
    elif source is not None and target is not None:
        return calculate_effect(source, target)
    else:
        raise ValueError(
            "Both source and target must be specified, or both must be None."
        )


@cache
def weighted_paths(G, source, target, input="positive"):
    nodes = get_nodes(G, "state")
    A_sgn = create_matrix(G, form="signed")
    input_multiplier = sp.Integer(1) if input == "positive" else sp.Integer(-1)

    def calculate_weighted_paths(s, t):
        path_nodes = list(nx.all_simple_paths(G, s, t))
        wgt_effects = []
        for path in path_nodes:
            subsystem_nodes = [node for node in nodes if node not in path]
            if not subsystem_nodes:
                feedback = sp.Integer(-1)
            else:
                subsystem = G.subgraph(subsystem_nodes).copy()
                feedback = weighted_feedback(subsystem, level=len(nodes) - len(path))
                if feedback[0] == sp.nan:
                    feedback = sp.Integer(0)
            if s == t:
                wgt_effect = sp.Integer(-1) * feedback
            else:
                sign = sp.prod(
                    A_sgn[nodes.index(path[i + 1]), nodes.index(path[i])]
                    for i in range(len(path) - 1)
                )
                wgt_effect = sp.Integer(-1) * sign * feedback * input_multiplier
            wgt_effects.append(wgt_effect)
        return sp.Matrix(wgt_effects)

    return calculate_weighted_paths(source, target)
