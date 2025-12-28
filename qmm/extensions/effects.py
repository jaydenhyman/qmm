"""Analyse cumulative effects from perturbation scenarios with multiple-inputs and multiple-outputs."""

import numpy as np
import pandas as pd
import sympy as sp
import networkx as nx
from functools import cache
from ..core.helper import (
    get_nodes,
    get_weight,
    sign_determinacy,
    _random_sampler,
    _parse_perturbations,
    _parse_observations,
    get_dashed_alternatives,
    _check_direct_io_edges,
    _check_acyclic_inputs,
    _check_acyclic_outputs,
)
from ..core.structure import create_matrix
from ..core.press import (
    adjoint_matrix,
    absolute_feedback_matrix,
    numerical_simulations,
    weighted_predictions_matrix,
    sign_determinacy_matrix,
)
from ..core.prediction import qualitative_predictions, matrix_to_predictions
from typing import Callable, Dict, Optional, Any, Tuple, Literal, Union


def define_input_output(G: nx.DiGraph, remove_disconnected: bool = True) -> nx.DiGraph:
    """Define model components as state variables, inputs and outputs.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        remove_disconnected: Remove disconnected components

    Returns:
        nx.DiGraph: Model with input, state and output classification

    Examples:
        ```python
        from qmm import define_input_output, load_digraph
        G = load_digraph("snowshoe_io")
        G_io = define_input_output(G)
        (G_io.nodes['I']['category'], G_io.nodes['O']['category'])
        # ('input', 'output')
        ```
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Input must be a networkx.DiGraph.")
    G_def = G.copy()
    if remove_disconnected:
        G_undirected = G_def.to_undirected()
        connected = list(nx.connected_components(G_undirected))
        if len(connected) > 1:
            largest = max(connected, key=len)
            nodes_to_remove = [node for system in connected if system != largest for node in system]
            G_def.remove_nodes_from(nodes_to_remove)
    nx.set_node_attributes(G_def, "state", "category")
    while True:
        reclassified = False
        for node in list(G_def.nodes()):
            if G_def.nodes[node]["category"] == "state":
                if all(G_def.nodes[pred]["category"] == "input" for pred in G_def.predecessors(node)):
                    G_def.nodes[node]["category"] = "input"
                    reclassified = True
                elif all(G_def.nodes[succ]["category"] == "output" for succ in G_def.successors(node)):
                    G_def.nodes[node]["category"] = "output"
                    reclassified = True
        if not reclassified:
            break

    _check_acyclic_inputs(G_def)
    _check_acyclic_outputs(G_def)
    _check_direct_io_edges(G_def)

    nx.freeze(G_def)
    return G_def


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
        # a_H,P*a_P,H*b_V,I*c_O,V - a_H,P*a_P,V*b_V,I*c_O,H - a_H,P*a_V,H*b_P,I*c_O,V + a_H,P*a_V,V*b_P,I*c_O,H + a_H,V*a_P,H*b_V,I*c_O,P + a_H,V*a_P,P*b_V,I*c_O,H - a_H,V*a_V,H*b_P,I*c_O,P + a_P,H*a_V,V*b_H,I*c_O,P - a_P,P*a_V,H*b_H,I*c_O,V + a_P,P*a_V,V*b_H,I*c_O,H - a_P,V*a_V,H*b_H,I*c_O,P

        cumulative_effects(load_digraph("snowshoe_io"), form='signed')
        # Matrix([
        # [1, -1,  1, -1],
        # [0,  1, -1,  2],
        # [1,  0,  1,  0],
        # [2,  0,  1,  1]])

        cumulative_effects(load_digraph("snowshoe_io"), form='binary')
        # Matrix([
        # [1, 1, 1,  3],
        # [2, 1, 1,  4],
        # [1, 2, 1,  4],
        # [4, 4, 3, 11]])
        ```
    """
    B = create_matrix(G, form=form, matrix_type="B")
    C = create_matrix(G, form=form, matrix_type="C")
    D = create_matrix(G, form=form, matrix_type="D")
    if form == "symbolic":
        effects = adjoint_matrix(G, form="symbolic")
    elif form == "signed":
        effects = adjoint_matrix(G, form="signed")
    elif form == "binary":
        effects = absolute_feedback_matrix(G)
    else:
        raise ValueError("Invalid form. Choose 'symbolic', 'signed', 'binary'.")
    cemat = sp.BlockMatrix([[effects, effects * B], [C * effects, C * effects * B + D]]).as_explicit()
    if form != "symbolic":
        cemat = cemat.subs({sym: 1 for sym in cemat.free_symbols})
    return sp.expand(cemat)


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
        # [1, -1,  1, -1],
        # [0,  1, -1,  2],
        # [1,  0,  1,  0],
        # [2,  0,  1,  1]])
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
        # [1, 1, 1,  3],
        # [2, 1, 1,  4],
        # [1, 2, 1,  4],
        # [4, 4, 3, 11]])
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
        # [  1, -1,   1, -1/3],
        # [  0,  1,  -1,  1/2],
        # [  1,  0,   1,    0],
        # [1/2,  0, 1/3, 1/11]])
        ```
    """
    net = cumulative_effects(G, form="signed")
    absolute = cumulative_effects(G, form="binary")
    return get_weight(net, absolute)


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
        # [                1,  -1,                 1, -0.766271554878084],
        # [              1/2,   1,                -1,  0.857923586573834],
        # [                1, 1/2,                 1,                1/2],
        # [0.857923586573834, 1/2, 0.766271554878084,  0.586297666302382]])

        sign_determinacy_effects(load_digraph("snowshoe_io"), method='95_bound')
        # Matrix([
        # [  1,  -1,   1, -1/2],
        # [1/2,   1,  -1,  1/2],
        # [  1, 1/2,   1,  1/2],
        # [1/2, 1/2, 1/2,  1/2]])
        ```
    """
    weighted = weighted_effects(G)
    absolute = cumulative_effects(G, form="binary")
    return sign_determinacy(weighted, absolute, method=method)


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

    Returns:
        Dict containing effects, valid_sims, all_nodes, tmat, prop_stable, attempts, and optionally samples.

    Examples:
        ```python
        from qmm import get_simulations, load_digraph
        result = get_simulations(load_digraph("snowshoe_io"), n_sim=1000, perturb=('V', 1))
        result['effects'][0]
        # array([  7.97294786, -33.66459468,   6.1386996 , -12.69217711])
        ```
    """

    np.random.seed(seed)

    A_sym, B_sym, C_sym, D_sym = (create_matrix(G, form="symbolic", matrix_type=m) for m in "ABCD")
    symbols_all = {s for m in (A_sym, B_sym, C_sym, D_sym) if m for s in m.free_symbols}
    symbols_all |= {sp.Symbol(f"a_{u},{v}") for u, v in G.edges}
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
    perturb_nodes = state + inputs
    if perturb and perturb[0] not in perturb_nodes:
        raise KeyError(f"Perturbation node '{perturb[0]}' not found.")
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

    effects, valid_sims, samples = [], [], []
    attempts, max_attempts = 0, n_sim * 100
    while len(effects) < n_sim and attempts < max_attempts:
        attempts += 1
        values = _random_sampler(dist, len(symbols_sampled))
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
        "n_valid": int(sum(valid_sims)),
        "n_attempts": attempts,
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


def simulation_effects(
    G: nx.DiGraph,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    positive_only: bool = False,
    presample: Optional[Callable[[Tuple[sp.Symbol, ...]], Dict[sp.Symbol, Any]]] = None,
) -> sp.Matrix:
    """Performs numerical simulations of cumulative effects using random interaction strengths.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        n_sim: Number of simulations
        dist: Distribution for sampling ("uniform", "weak", "moderate", "strong")
        seed: Random seed
        positive_only: Return just the proportion of positive responses instead of sign-dominant proportions
        presample: Optional callable passed through to get_simulations

    Returns:
        SymPy Matrix containing simulation results

    Examples:
        ```python
        from qmm import load_digraph, simulation_effects
        simulation_effects(load_digraph("snowshoe_io"), n_sim=1000)
        # Matrix([
        # [  1.0,  -1.0,   1.0, -0.697],
        # [0.651,   1.0,  -1.0,  0.962],
        # [  1.0, 0.638,   1.0,   0.61],
        # [0.952,  0.62, 0.694,  0.716]])

        simulation_effects(load_digraph("snowshoe_io"), n_sim=1000, positive_only=True)
        # Matrix([
        # [  1.0,   0.0,   1.0, 0.303],
        # [0.651,   1.0,   0.0, 0.962],
        # [  1.0, 0.638,   1.0,  0.61],
        # [0.952,  0.62, 0.694, 0.716]])
        ```
    """
    sims = get_simulations(G, n_sim, dist, seed, presample=presample)
    tmat = sims["tmat"]
    n_rows, n_cols = tmat.shape

    effects = np.array(sims["effects"])
    positive = np.sum(effects > 0, axis=0)
    negative = np.sum(effects < 0, axis=0)

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
        simulations_table(load_digraph("snowshoe_io"), perturb='I:+', n_sim=1000)
        #    model effect_on  negative  no_effect  positive  valid_sims  stable_sims  attempts
        # 0      1         V       697          0       303        1000         1000      1230
        # 1      1         H        38          0       962        1000         1000      1230
        # 2      1         P       390          0       610        1000         1000      1230
        # 3      1         O       284          0       716        1000         1000      1230
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
            effect_arr = np.array(valid_effects)
            negative = np.sum(effect_arr < 0, axis=0).astype(int)
            positive = np.sum(effect_arr > 0, axis=0).astype(int)
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


def table_of_predictions(
    G: nx.DiGraph,
    generator: Callable[..., Union[sp.Matrix, np.ndarray]] = simulation_effects,
    t1: float = 0.8,
    t2: float = 0.95,
) -> pd.DataFrame:
    """Create a table of qualitative predictions with thresholds for ambiguity.

    This function works with all prediction generators including effects-based ones
    (weighted_effects, sign_determinacy_effects, simulation_effects) and core press
    generators (numerical_simulations, weighted_predictions_matrix, sign_determinacy_matrix).

    Args:
        G (nx.DiGraph): Graph input for the matrix generator
        generator (Callable): Matrix generator function. Can be from core.press or
            extensions.effects modules
        t1 (float): Lower threshold for predictions
        t2 (float): Higher threshold for predictions

    Returns:
        pd.DataFrame: Qualitative predictions table. For effects generators,
            returns a MultiIndex DataFrame with State/Input columns and State/Output rows.
            For press generators, returns a simple DataFrame with state variables only.

    Raises:
        ValueError: If generator is not callable or thresholds are invalid

    References:
        - Puccia, C.J., Levins, R. (1985). Qualitative modeling of complex systems: an introduction to loop analysis and time averaging. Harvard University Press, Cambridge, MA.
        - Dambacher, J.M., Li, H.W., Rossignol, P.A. (2002). Relevance of Community Structure in Assessing Indeterminacy of Ecological Predictions. Ecology 83, 1372–1385.

    Examples:
        ```python
        from qmm import load_digraph, weighted_effects, table_of_predictions
        table_of_predictions(load_digraph("snowshoe_io"), generator=weighted_effects, t1=0.5, t2=1.0)
        #          State       Input
        #              V  H  P     I
        # State  V     +  −  +     ?
        #        H     ?  +  −   (+)
        #        P     +  ?  +     ?
        # Output O   (+)  ?  ?     ?
        ```
    """
    if not callable(generator):
        raise ValueError(f"Generator must be callable, got: {type(generator)}")

    generator_name = getattr(generator, "__name__", None)
    pred_matrix = qualitative_predictions(G, generator=generator, t1=t1, t2=t2)
    state = get_nodes(G, "state")

    # Check if this is an effects generator that produces extended matrices
    if generator_name in (
        "sign_determinacy_effects",
        "simulation_effects",
        "weighted_effects",
    ):
        inputs = get_nodes(G, "input")
        outputs = get_nodes(G, "output")
        columns = state + inputs
        index = state + outputs
        df = matrix_to_predictions(pred_matrix, t1=t1, t2=t2, index=index, columns=columns)
        col_groups = ["State"] * len(state) + ["Input"] * len(inputs)
        row_groups = ["State"] * len(state) + ["Output"] * len(outputs)
        df.columns = pd.MultiIndex.from_arrays([col_groups, columns])
        df.index = pd.MultiIndex.from_arrays([row_groups, index])
        return df

    return matrix_to_predictions(pred_matrix, t1=t1, t2=t2, index=state, columns=state)
