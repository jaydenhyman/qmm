"""Analyse cumulative effects from perturbation scenarios with multiple-inputs and multiple-outputs."""

import warnings
import numpy as np
import pandas as pd
import sympy as sp
import networkx as nx
from functools import cache
from ..core.helper import (
    get_nodes,
    get_weight,
    get_positive,
    get_negative,
    sign_determinacy,
    _random_sampler,
    _parse_perturbations,
    _parse_observations,
    get_dashed_alternatives,
    _check_signs,
    _check_direct_io_edges,
    _edge_prefix,
)
from ..core.structure import create_matrix
from ..core.press import (
    adjoint_matrix,
    absolute_feedback_matrix,
)
from typing import Callable, Dict, Optional, Any, Tuple, Literal, Union


def define_input_output(G: nx.DiGraph, remove_disconnected: bool = True) -> nx.DiGraph:
    """Classify nodes as state, input or output from topology (any pre-set category is overwritten).

    Sources (and source-chains) become inputs, sinks (and sink-chains) outputs; a
    self-loop or feedback cycle keeps a node as state.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        remove_disconnected: Remove all but the largest weakly-connected
            component (warns about dropped nodes)

    Returns:
        nx.DiGraph: Model with input, state and output classification

    Examples:
        ```python
        from qmm import define_input_output, load_digraph
        G = load_digraph("snowshoe_io")
        G_io = define_input_output(G)
        (G_io.nodes['Inp1']['category'], G_io.nodes['Out1']['category'])
        # ('input', 'output')
        ```
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Input must be a networkx.DiGraph.")
    _check_signs(G)
    G_def = G.copy()
    if remove_disconnected:
        components = list(nx.connected_components(G_def.to_undirected()))
        if len(components) > 1:
            largest = max(components, key=lambda c: (len(c), sorted(c)))
            dropped = sorted(n for c in components if c != largest for n in c)
            warnings.warn(
                f"define_input_output: dropping {len(dropped)} node(s) in "
                f"{len(components) - 1} smaller disconnected component(s): {dropped}"
            )
            G_def.remove_nodes_from(dropped)
    nx.set_node_attributes(G_def, "state", "category")

    # Inputs then outputs, each a fixpoint (order-independent); self-loop/feedback nodes stay state.
    def classify(role, here, there):
        changed = True
        while changed:
            changed = False
            for node in G_def.nodes():
                if G_def.nodes[node]["category"] != "state" or G_def.has_edge(node, node) or not list(there(node)):
                    continue
                anchor = list(here(node))
                if not anchor or all(G_def.nodes[n]["category"] == role for n in anchor):
                    G_def.nodes[node]["category"] = role
                    changed = True

    classify("input", G_def.predecessors, G_def.successors)
    classify("output", G_def.successors, G_def.predecessors)

    _check_direct_io_edges(G_def)
    nx.freeze(G_def)
    return G_def


@cache
def direct_effects(
    G: nx.DiGraph,
    form: Literal["net", "absolute", "positive", "negative"] = "net",
) -> sp.Matrix:
    """Calculate direct effects from the signed digraph structure.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        form: Type of direct effects ('net', 'absolute', 'positive', 'negative')

    Returns:
        sp.Matrix: Direct effects for state/input columns and state/output rows
    """
    if form not in ("net", "absolute", "positive", "negative"):
        raise ValueError("Invalid form. Choose 'net', 'absolute', 'positive', 'negative'.")

    def block(f):
        m = {t: create_matrix(G, form=f, matrix_type=t) for t in "ABCD"}
        return sp.BlockMatrix([[m["A"], m["B"]], [m["C"], m["D"]]]).as_explicit()

    if form == "net":
        return block("signed")
    if form == "absolute":
        return block("binary")
    signed, binary = block("signed"), block("binary")
    return get_positive(signed, binary) if form == "positive" else get_negative(signed, binary)


@cache
def cumulative_effects(
    G: nx.DiGraph,
    form: Literal["symbolic", "signed", "binary"] = "symbolic",
) -> sp.Matrix:
    """Calculate cumulative effects to multiple inputs using state-space representation.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        form: Type of computation ('symbolic', 'signed', or 'binary')

    Returns:
        sp.Matrix: Cumulative effects on state variables and outputs

    Examples:
        ```python
        from qmm import load_digraph, cumulative_effects
        cumulative_effects(load_digraph("snowshoe_io"), form='symbolic')[3, 3]
        # a_C,R*a_P,C*b_R,Inp1*c_Out1,P - a_C,R*a_P,P*b_R,Inp1*c_Out1,C - a_P,C*a_R,R*b_C,Inp1*c_Out1,P + a_P,P*a_R,R*b_C,Inp1*c_Out1,C

        cumulative_effects(load_digraph("snowshoe_io"), form='signed')
        # Matrix([
        # [1, -1,  1, 2, -1],
        # [1,  1, -1, 0,  1],
        # [1,  1,  1, 0, -1],
        # [0,  0,  2, 0, -2],
        # [1,  1, -1, 0,  1]])

        cumulative_effects(load_digraph("snowshoe_io"), form='binary')
        # Matrix([
        # [1, 1, 1, 2, 1],
        # [1, 1, 1, 2, 1],
        # [1, 1, 1, 2, 1],
        # [2, 2, 2, 4, 2],
        # [1, 1, 1, 2, 1]])
        ```
    """
    if form not in ("symbolic", "signed", "binary"):
        raise ValueError("Invalid form. Choose 'symbolic', 'signed', 'binary'.")
    B = create_matrix(G, form=form, matrix_type="B")
    C = create_matrix(G, form=form, matrix_type="C")
    D = create_matrix(G, form=form, matrix_type="D")
    effects = absolute_feedback_matrix(G) if form == "binary" else adjoint_matrix(G, form=form)
    cemat = sp.BlockMatrix([[effects, effects * B], [C * effects, C * effects * B + D]]).as_explicit()
    if form != "symbolic":
        cemat = cemat.subs({sym: 1 for sym in cemat.free_symbols})
    return sp.expand(cemat)


def _tabulate_effects(
    G: nx.DiGraph,
    effects: Union[sp.MatrixBase, np.ndarray],
) -> pd.DataFrame:
    """Format effects as a table with state/input columns and state/output rows."""
    state = get_nodes(G, "state")
    inputs = get_nodes(G, "input")
    outputs = get_nodes(G, "output")
    columns = state + inputs
    index = state + outputs

    values = np.array(effects.tolist() if isinstance(effects, sp.MatrixBase) else effects, dtype=object)

    df = pd.DataFrame(values, index=index, columns=columns)
    col_groups = ["State"] * len(state) + ["Input"] * len(inputs)
    row_groups = ["State"] * len(state) + ["Output"] * len(outputs)
    df.columns = pd.MultiIndex.from_arrays([col_groups, columns])
    df.index = pd.MultiIndex.from_arrays([row_groups, index])
    return df


@cache
def net_effects(G: nx.DiGraph) -> sp.Matrix:
    """Calculate net cumulative effects from multiple inputs.

    Args:
        G: NetworkX DiGraph representing signed digraph model

    Returns:
        sp.Matrix: Net effects on state variables and outputs

    Examples:
        ```python
        from qmm import load_digraph, net_effects
        net_effects(load_digraph("snowshoe_io"))
        # Matrix([
        # [1, -1,  1, 2, -1],
        # [1,  1, -1, 0,  1],
        # [1,  1,  1, 0, -1],
        # [0,  0,  2, 0, -2],
        # [1,  1, -1, 0,  1]])
        ```
    """
    return cumulative_effects(G, form="signed")


@cache
def absolute_effects(G: nx.DiGraph) -> sp.Matrix:
    """Calculate absolute effects from multiple inputs.

    Args:
        G: NetworkX DiGraph representing signed digraph model

    Returns:
        sp.Matrix: Total effects on state variables and outputs

    Examples:
        ```python
        from qmm import load_digraph, absolute_effects
        absolute_effects(load_digraph("snowshoe_io"))
        # Matrix([
        # [1, 1, 1, 2, 1],
        # [1, 1, 1, 2, 1],
        # [1, 1, 1, 2, 1],
        # [2, 2, 2, 4, 2],
        # [1, 1, 1, 2, 1]])
        ```
    """
    return cumulative_effects(G, form="binary")


@cache
def weighted_effects(G: nx.DiGraph) -> sp.Matrix:
    """Calculate ratio of net to total terms for predicting cumulative effects.

    Args:
        G: NetworkX DiGraph representing signed digraph model

    Returns:
        sp.Matrix: Ratio of net to total effects

    Examples:
        ```python
        from qmm import load_digraph, weighted_effects
        weighted_effects(load_digraph("snowshoe_io"))
        # Matrix([
        # [1, -1,  1, 1, -1],
        # [1,  1, -1, 0,  1],
        # [1,  1,  1, 0, -1],
        # [0,  0,  1, 0, -1],
        # [1,  1, -1, 0,  1]])
        ```
    """
    return get_weight(net_effects(G), absolute_effects(G))


@cache
def sign_determinacy_effects(
    G: nx.DiGraph,
    method: Literal["average", "95_bound"] = "average",
) -> sp.Matrix:
    """Calculate probability of correct sign prediction for cumulative effects.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        method: Method for computing determinacy ('average', '95_bound')

    Returns:
        sp.Matrix: Sign determinacy probabilities for effects

    Examples:
        ```python
        from qmm import load_digraph, sign_determinacy_effects
        sign_determinacy_effects(load_digraph("snowshoe_io"), method='average')
        # Matrix([
        # [  1,  -1,  1,   1, -1],
        # [  1,   1, -1, 1/2,  1],
        # [  1,   1,  1, 1/2, -1],
        # [1/2, 1/2,  1, 1/2, -1],
        # [  1,   1, -1, 1/2,  1]])

        sign_determinacy_effects(load_digraph("snowshoe_io"), method='95_bound')
        # Matrix([
        # [  1,  -1,  1,   1, -1],
        # [  1,   1, -1, 1/2,  1],
        # [  1,   1,  1, 1/2, -1],
        # [1/2, 1/2,  1, 1/2, -1],
        # [  1,   1, -1, 1/2,  1]])
        ```
    """
    absolute = absolute_effects(G)
    return sign_determinacy(weighted_effects(G), absolute, method=method)


@cache
def get_simulations(
    G: nx.DiGraph,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    perturb: Optional[Tuple[str, int]] = None,
    observe: Optional[Tuple[Tuple[str, int], ...]] = None,
    presample: Optional[Callable[[Tuple[sp.Symbol, ...]], Dict[sp.Symbol, Any]]] = None,
    return_samples: bool = False,
    average_uncertain: bool = False,
) -> Dict[str, Any]:
    """Calculate average proportion of positive and negative effects from stable numerical simulations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        n_sim: Number of simulations
        dist: Distribution for sampling
        seed: Random seed
        perturb: Optional perturbations (node, sign)
        observe: Optional observations (node, sign)
        presample: Optional callable that receives the tuple of free symbols and
            returns a mapping of symbol substitutions to apply before sampling.
        return_samples: If True, include dict mapping symbol names to arrays of sampled values
        average_uncertain: If True, sample edges marked dashes=True in/out each draw (structure averaging)

    Returns:
        Dict containing effects, valid_sims, all_nodes, tmat, prop_stable, attempts, and optionally samples.

    Examples:
        ```python
        from qmm import get_simulations, load_digraph
        result = get_simulations(load_digraph("snowshoe_io"), n_sim=1000, perturb=('Inp1', 1))
        result['effects'][0]
        # array([ 1.29385476,  2.55918625,  3.12917778, -1.74767706,  2.48217995])
        ```
    """

    rng = np.random.RandomState(seed)

    A_sym, B_sym, C_sym, D_sym = (create_matrix(G, form="symbolic", matrix_type=m) for m in "ABCD")
    symbols_all = {s for m in (A_sym, B_sym, C_sym, D_sym) if m for s in m.free_symbols}
    symbols_all = tuple(sorted(symbols_all, key=str))
    fixed_subs = {}

    if presample and symbols_all and (subs := presample(symbols_all)):
        fixed_subs = {sym: subs[sym] for sym in symbols_all if sym in subs}
        A_sym, B_sym, C_sym, D_sym = (m.subs(subs) if m else None for m in (A_sym, B_sym, C_sym, D_sym))
        symbols_sampled = tuple(sorted({s for m in (A_sym, B_sym, C_sym, D_sym) if m for s in m.free_symbols}, key=str))
    else:
        symbols_sampled = symbols_all

    state, inputs, outputs = (get_nodes(G, t) for t in ("state", "input", "output"))
    all_nodes = state + inputs + outputs
    n_x, n_u, n_y = len(state), len(inputs), len(outputs)

    response_idx = {node: i for i, node in enumerate(state + outputs)}
    if observe:
        unknown = [n for n, _ in observe if n not in response_idx]
        if unknown:
            raise ValueError(f"Unknown observation node(s): {unknown}. Valid response nodes: {list(response_idx)}")
    perturb_nodes = state + inputs
    if perturb and perturb[0] not in perturb_nodes:
        raise ValueError(f"Perturbation node '{perturb[0]}' not found.")
    p_idx, p_sign = (perturb_nodes.index(perturb[0]), perturb[1]) if perturb else (None, 1)

    tmat = sp.matrix2numpy(absolute_effects(G)).astype(int)

    A_fn = sp.lambdify(symbols_sampled, A_sym)
    B_fn = sp.lambdify(symbols_sampled, B_sym) if n_u and B_sym else None
    C_fn = sp.lambdify(symbols_sampled, C_sym) if n_y and C_sym else None
    D_fn = sp.lambdify(symbols_sampled, D_sym) if D_sym and D_sym.shape != (0, 0) else None

    def compute_sample(values):
        A = A_fn(*values)
        if not np.all(np.real(np.linalg.eigvals(A)) < 0):
            return None
        try:
            inv_A = np.linalg.inv(-A)
        except np.linalg.LinAlgError:
            return None
        B = B_fn(*values) if B_fn else np.zeros((n_x, 0))
        C = C_fn(*values) if C_fn else np.zeros((0, n_x))
        D = D_fn(*values) if D_fn else np.zeros((n_y, n_u))
        E = np.block([[inv_A, inv_A @ B], [C @ inv_A, C @ inv_A @ B + D]]) if n_x else np.zeros((n_y, n_u))
        effect = E[:, p_idx] * p_sign if p_idx is not None else E
        return effect

    def is_valid(effect, tmat_ref):
        if not observe or tmat_ref is None:
            return True
        for node, obs in observe:
            idx = response_idx[node]
            expected = tmat_ref[idx, p_idx] != 0
            if (expected and (obs == 0 or np.sign(effect[idx]) != obs)) or (not expected and obs != 0):
                return False
        return True

    uncertain = [(u, v, symbols_sampled.index(sp.Symbol(f"{_edge_prefix(G, u, v)}_{v},{u}")))
                 for u, v, d in G.edges(data=True) if d.get("dashes")] if average_uncertain else []
    base_cls, checked = {n: d.get("category", "state") for n, d in G.nodes(data=True)}, set()
    effects, valid_sims, samples = [], [], []
    attempts, max_attempts = 0, n_sim * 100
    while len(effects) < n_sim and attempts < max_attempts:
        attempts += 1
        values = _random_sampler(dist, len(symbols_sampled), rng)
        if uncertain:
            keep = rng.uniform(size=len(uncertain)) < rng.uniform()
            if (k := tuple(keep)) not in checked:
                variant = nx.DiGraph(G)
                variant.remove_edges_from([(u, v) for ke, (u, v, _) in zip(keep, uncertain) if not ke])
                if {n: d.get("category", "state") for n, d in define_input_output(variant, remove_disconnected=False).nodes(data=True)} != base_cls:
                    raise ValueError("Excluding an uncertain edge re-classifies a node; structure averaging halted.")
                checked.add(k)
            values[[i for ke, (_, _, i) in zip(keep, uncertain) if not ke]] = 0.0
        if (effect := compute_sample(values)) is None:
            continue
        effects.append(effect)
        valid_sims.append(is_valid(effect, tmat))
        if return_samples:
            samples.append(values)

    if len(effects) < n_sim:
        raise RuntimeError(f"Maximum iterations reached. Stable proportion: {len(effects) / max_attempts:.4f}")

    result = {
        "effects": effects,
        "valid_sims": valid_sims,
        "all_nodes": all_nodes,
        "tmat": tmat,
        "prop_stable": len(effects) / attempts,
        "attempts": attempts,
        "n_stable": len(effects),
    }
    if return_samples:
        n_samples = len(samples)
        sampled_index = {sym: i for i, sym in enumerate(symbols_sampled)}
        result_samples = {}
        for sym in symbols_all:
            if sym in sampled_index:
                idx = sampled_index[sym]
                result_samples[str(sym)] = np.array([s[idx] for s in samples])
            elif sym in fixed_subs:
                result_samples[str(sym)] = np.full(n_samples, fixed_subs[sym])
        result["samples"] = result_samples
    return result


def _sign_counts(effects) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.array(effects)
    return np.sum(arr > 0, axis=0), np.sum(arr < 0, axis=0)


def simulation_effects(
    G: nx.DiGraph,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    positive_only: bool = False,
    presample: Optional[Callable[[Tuple[sp.Symbol, ...]], Dict[sp.Symbol, Any]]] = None,
    average_uncertain: bool = False,
) -> sp.Matrix:
    """Performs numerical simulations of cumulative effects using random interaction strengths.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        n_sim: Number of simulations
        dist: Distribution for sampling ("uniform", "weak", "moderate", "strong")
        seed: Random seed
        positive_only: Return just the proportion of positive responses instead of sign-dominant proportions
        presample: Optional callable passed through to get_simulations
        average_uncertain: Passed through to get_simulations (structure averaging over uncertain links)

    Returns:
        SymPy Matrix containing simulation results

    Examples:
        ```python
        from qmm import load_digraph, simulation_effects
        simulation_effects(load_digraph("snowshoe_io"), n_sim=1000)
        # Matrix([
        # [   1.0,   -1.0,  1.0,    1.0, -1.0],
        # [   1.0,    1.0, -1.0, -0.513,  1.0],
        # [   1.0,    1.0,  1.0, -0.513, -1.0],
        # [-0.517, -0.517,  1.0,  0.526, -1.0],
        # [   1.0,    1.0, -1.0, -0.513,  1.0]])

        simulation_effects(load_digraph("snowshoe_io"), n_sim=1000, positive_only=True)
        # Matrix([
        # [  1.0,   0.0, 1.0,   1.0, 0.0],
        # [  1.0,   1.0, 0.0, 0.487, 1.0],
        # [  1.0,   1.0, 1.0, 0.487, 0.0],
        # [0.483, 0.483, 1.0, 0.526, 0.0],
        # [  1.0,   1.0, 0.0, 0.487, 1.0]])
        ```
    """
    sims = get_simulations(G, n_sim, dist, seed, presample=presample, average_uncertain=average_uncertain)
    tmat = sims["tmat"]
    n_rows, n_cols = tmat.shape

    positive, negative = _sign_counts(sims["effects"])

    smat = positive / n_sim if positive_only else np.where(
        negative > positive, -negative / n_sim, positive / n_sim
    )
    smat = [[sp.nan if not tmat[i, j] else smat[i, j] for j in range(n_cols)] for i in range(n_rows)]
    return sp.Matrix(smat)


def simulations_table(
    G: nx.DiGraph,
    perturb: str,
    observe: str = "",
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    combinations: bool = True,
    presample: Optional[Callable[[Tuple[sp.Symbol, ...]], Dict[sp.Symbol, Any]]] = None,
) -> pd.DataFrame:
    """Summarise simulation effects across model variants for each response node.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        perturb: Node and sign to perturb (perturbation string)
        observe: Observation string (node:sign, comma-separated allowed) to filter simulations
        n_sim: Number of simulations
        dist: Distribution for sampling
        seed: Random seed
        combinations: If True, evaluate every combination of dashed edges
        presample: Optional callable passed through to get_simulations

    Returns:
        pd.DataFrame: Table of counts for negative, no effect, and positive responses

    Examples:
        ```python
        from qmm import load_digraph, simulations_table
        simulations_table(load_digraph("snowshoe_io"), perturb='Inp1:+', n_sim=1000)
        #    model effect_on  negative  no_effect  positive  valid_sims  stable_sims  attempts
        # 0      1         R         0          0      1000        1000         1000      1000
        # 1      1         C       513          0       487        1000         1000      1000
        # 2      1         P       513          0       487        1000         1000      1000
        # 3      1      Out1       474          0       526        1000         1000      1000
        # 4      1      Out2       513          0       487        1000         1000      1000
        ```
    """
    variants = get_dashed_alternatives(G, combinations=combinations)
    observations = _parse_observations(observe) if observe else None
    rows = []

    for model_idx, g in enumerate(variants, start=1):
        graph, pert = _parse_perturbations(g, perturb)
        sims = get_simulations(
            graph,
            n_sim=n_sim,
            dist=dist,
            seed=seed,
            perturb=pert,
            observe=observations,
            presample=presample,
        )

        response_nodes = get_nodes(g, "state") + get_nodes(g, "output")
        if not response_nodes:
            continue
        node_count = len(response_nodes)
        p_idx = sims["all_nodes"].index(pert[0])
        tmat = sims["tmat"][:node_count, :]

        valid_effects = [effect[:node_count] for effect, valid in zip(sims["effects"], sims["valid_sims"]) if valid]
        valid_count = len(valid_effects)
        if valid_count:
            positive, negative = (c.astype(int) for c in _sign_counts(valid_effects))
        else:
            negative = np.zeros(node_count, dtype=int)
            positive = np.zeros(node_count, dtype=int)
        has_effect = tmat[:, p_idx] != 0
        no_effect = np.where(has_effect, 0, valid_count).astype(int)
        negative = np.where(has_effect, negative, 0).astype(int)
        positive = np.where(has_effect, positive, 0).astype(int)

        for i, node in enumerate(response_nodes):
            row = {
                "model": model_idx,
                "effect_on": node,
                "negative": int(negative[i]),
                "no_effect": int(no_effect[i]),
                "positive": int(positive[i]),
                "valid_sims": int(valid_count),
                "stable_sims": int(sims["n_stable"]),
                "attempts": int(sims["attempts"]),
            }
            rows.append(row)

    cols = ["model", "effect_on", "negative", "no_effect", "positive", "valid_sims", "stable_sims", "attempts"]
    return pd.DataFrame(rows, columns=cols)


def table_of_direct_effects(
    G: nx.DiGraph,
    form: Literal["net", "absolute", "positive", "negative"] = "net",
) -> pd.DataFrame:
    """Create a table of direct effects with state/input columns and state/output rows."""
    return _tabulate_effects(G, direct_effects(G, form=form))


_EFFECT_GENERATORS = {
    "net_effects": net_effects,
    "absolute_effects": absolute_effects,
    "weighted_effects": weighted_effects,
    "sign_determinacy_effects": sign_determinacy_effects,
    "simulation_effects": simulation_effects,
}


def table_of_effects(
    G: nx.DiGraph,
    generator: Union[
        Callable[..., Union[sp.Matrix, np.ndarray, pd.DataFrame]],
        Literal[
            "net_effects",
            "absolute_effects",
            "weighted_effects",
            "sign_determinacy_effects",
            "simulation_effects",
        ],
    ] = net_effects,
    decimals: Optional[int] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    if isinstance(generator, str):
        generator = _EFFECT_GENERATORS.get(generator)
    if not callable(generator):
        raise ValueError(f"Generator must be callable, got: {type(generator)}")
    effects = generator(G, **kwargs)
    if decimals is not None and isinstance(effects, sp.MatrixBase):
        effects = effects.evalf(decimals)
    return _tabulate_effects(G, effects)
