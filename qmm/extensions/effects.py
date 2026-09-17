"""Analyse cumulative effects from perturbation scenarios with multiple-inputs and multiple-outputs."""

import numpy as np
import pandas as pd
import sympy as sp
import itertools
import networkx as nx
from ..core.helper import (
    _build_model_variant,
    _check_direct_io_edges,
    _group_uncertain_edges,
    get_nodes,
    get_weight,
    get_positive,
    get_negative,
    sign_determinacy,
    _random_sampler,
    _parse_perturbations,
    _parse_observations,
    get_dashed_alternatives,
    _edge_prefix,
)
from ..core.structure import create_matrix, define_input_output
from ..core.press import (
    adjoint_matrix,
    absolute_feedback_matrix,
)
from typing import Callable, Dict, Optional, Any, Tuple, Literal, Union, Iterator


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
    _check_direct_io_edges(G)
    effects = absolute_feedback_matrix(G) if form == "binary" else adjoint_matrix(G, form=form)
    cemat = sp.BlockMatrix([[effects, effects * B], [C * effects, C * effects * B]]).as_explicit()
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


def get_simulations(
    G: nx.DiGraph,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    perturb: Optional[Union[Tuple[str, int], Tuple[Tuple[str, int], ...]]] = None,
    observe: Optional[Tuple[Tuple[str, int], ...]] = None,
    presample: Optional[Callable[[Tuple[sp.Symbol, ...]], Dict[sp.Symbol, Any]]] = None,
    return_samples: bool = False,
    uncertain_interactions: Literal["sample", "enumerate"] = "sample",
    pair_reciprocal: bool = True,
    max_attempts: Optional[int] = None,
    condition: bool = True,
) -> Dict[str, Any]:
    """Collect numerical simulations; see iter_simulations for sampling options.

    Args:
        G: Signed digraph with state, input and output categories.
        n_sim: Target stable draws per batch; observation matches when condition=True.
        dist: Distribution of interaction strengths.
        seed: Random seed.
        perturb: One (node, sign) pair or a tuple of pairs for simultaneous unit presses.
        observe: Optional (node, sign) pairs; signs are -1, 0 or 1.
        presample: Callable receiving coefficient symbols and returning substitutions.
        return_samples: Include coefficient strengths for each stable draw.
        uncertain_interactions: Sample uncertain interactions or enumerate every structure.
        pair_reciprocal: Keep or drop reciprocal dashed edges together.
        max_attempts: Maximum draws attempted per batch; defaults to 100 * n_sim.
        condition: Require observation matches to reach n_sim; otherwise count all stable draws.

    Returns:
        Dictionary with effects, valid_sims, all_nodes, tmat, prop_stable,
        attempts, n_stable, structures, perturb, interactions, signs, and optionally samples.
        All stable draws are retained; valid_sims flags observation matches. structures
        contains the bit mask of present uncertain interactions for each draw, with bit i
        for the edges in interactions[i]. perturb holds the (node, sign) presses, and
        signs holds the sign of each edge.

    Raises:
        ValueError: Invalid sampling options or incompatible node categories.
        RuntimeError: A batch cannot reach n_sim within max_attempts.

    Examples:
        ```python
        from qmm import get_simulations, load_digraph
        result = get_simulations(load_digraph("snowshoe"), n_sim=10, seed=42)
        len(result["effects"])
        # 10
        result["effects"][0].shape
        # (3, 3)
        ```
    """
    batches = iter_simulations(G, n_sim, dist, seed, perturb, observe, presample,
                               return_samples, uncertain_interactions, pair_reciprocal,
                               condition=condition, max_attempts=max_attempts)
    result = next(batches)
    samples = [result["samples"]] if return_samples else []
    for batch in batches:
        for key in ("effects", "valid_sims", "structures"):
            result[key].extend(batch[key])
        result["attempts"] += batch["attempts"]
        result["n_stable"] += batch["n_stable"]
        if return_samples:
            samples.append(batch["samples"])
    result["prop_stable"] = result["n_stable"] / result["attempts"]
    if return_samples:
        result["samples"] = {name: np.concatenate([sample[name] for sample in samples])
                             for name in result["samples"]}
    return result


def iter_simulations(
    G: nx.DiGraph,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    perturb: Optional[Union[Tuple[str, int], Tuple[Tuple[str, int], ...]]] = None,
    observe: Optional[Tuple[Tuple[str, int], ...]] = None,
    presample: Optional[Callable[[Tuple[sp.Symbol, ...]], Dict[sp.Symbol, Any]]] = None,
    return_samples: bool = False,
    uncertain_interactions: Literal["sample", "enumerate"] = "sample",
    pair_reciprocal: bool = True,
    condition: bool = True,
    max_attempts: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield stable simulations, one batch per enumerated structure or one sampled batch.

    Args:
        G: Signed digraph with state, input and output categories.
        n_sim: Target stable draws matching observe. With condition=False, target
            stable draws regardless of observations. The target applies to each batch.
        dist: Distribution of interaction strengths.
        seed: Random seed.
        perturb: One (node, sign) pair or a tuple of pairs for simultaneous unit presses.
        observe: Optional (node, sign) pairs; signs are -1, 0 or 1.
        presample: Callable receiving coefficient symbols and returning substitutions.
        return_samples: Include the actual coefficient strengths for each stable draw.
        uncertain_interactions: 'sample' draws a probability uniformly from 0 to 1 for each attempt,
            then keeps each uncertain interaction independently with that probability.
            'enumerate' visits every combination and collects n_sim draws for each.
        pair_reciprocal: Keep or drop reciprocal dashed edges together.
        condition: Require observation matches to reach n_sim. Use False to estimate
            likelihoods from a fixed number of stable draws.
        max_attempts: Maximum draws attempted per batch; defaults to 100 * n_sim.

    Yields:
        Dictionaries with effects, valid_sims, all_nodes, tmat, prop_stable,
        attempts, n_stable, structures, perturb, interactions, signs, and optionally samples.
        All stable draws are retained; valid_sims flags observation matches. structures
        identifies each draw by a bit mask of present uncertain interactions, with bit i
        for the edges in interactions[i]. perturb holds the (node, sign) presses, and
        signs holds the sign of each edge. tmat contains term counts for G; effects use
        each draw's structure to set cells without terms to exact zero before combining
        perturbations.

    Raises:
        ValueError: A draw limit is invalid, or a structure has invalid nodes or
            changes node categories.
        RuntimeError: A batch cannot reach n_sim within max_attempts.
    """
    if isinstance(n_sim, (bool, np.bool_)) or not isinstance(n_sim, (int, np.integer)) or n_sim <= 0:
        raise ValueError("n_sim must be a positive integer.")
    if max_attempts is None:
        max_attempts = 100 * int(n_sim)
    if isinstance(max_attempts, (bool, np.bool_)) or not isinstance(max_attempts, (int, np.integer)) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer.")
    rng = np.random.RandomState(seed)

    A_sym, B_sym, C_sym, D_sym = (create_matrix(G, form="symbolic", matrix_type=m) for m in "ABCD")
    symbols_all = {s for m in (A_sym, B_sym, C_sym, D_sym) if m for s in m.free_symbols}
    symbols_all = tuple(sorted(symbols_all, key=str))
    fixed_subs = {}
    if uncertain_interactions not in ("sample", "enumerate"):
        raise ValueError("uncertain_interactions must be 'sample' or 'enumerate'.")
    interactions = _group_uncertain_edges(G, pair_reciprocal)
    uncertain_symbols = {sp.Symbol(f"{_edge_prefix(G, u, v)}_{v},{u}") for group in interactions for u, v in group}
    matrix_subs = {}

    if presample and symbols_all and (subs := presample(symbols_all)):
        fixed_subs = {sym: subs[sym] for sym in symbols_all if sym in subs}
        matrix_subs = {sym: value for sym, value in subs.items() if sym not in uncertain_symbols}
        A_sym, B_sym, C_sym, D_sym = (m.subs(matrix_subs) if m else None for m in (A_sym, B_sym, C_sym, D_sym))
        symbols_sampled = tuple(sorted({s for m in (A_sym, B_sym, C_sym, D_sym) if m for s in m.free_symbols}, key=str))
    else:
        symbols_sampled = symbols_all
    fixed_uncertain = {sym: sym.subs(fixed_subs)
                       for sym in uncertain_symbols if sym in fixed_subs}
    symbols_sampled = tuple(sorted(set(symbols_sampled) | uncertain_symbols |
                                   {s for expr in fixed_uncertain.values() for s in expr.free_symbols}, key=str))
    fixed_indices = [symbols_sampled.index(sym) for sym in fixed_uncertain]
    fixed_fn = sp.lambdify(symbols_sampled, list(fixed_uncertain.values())) if fixed_uncertain else None

    state, inputs, outputs = (get_nodes(G, t) for t in ("state", "input", "output"))
    all_nodes = state + inputs + outputs
    n_x, n_u, n_y = len(state), len(inputs), len(outputs)

    response_idx = {node: i for i, node in enumerate(state + outputs)}
    if observe:
        unknown = [n for n, _ in observe if n not in response_idx]
        if unknown:
            raise ValueError(f"Unknown observation node(s): {unknown}. Valid response nodes: {list(response_idx)}")
    perturb_nodes = state + inputs
    presses = (perturb,) if perturb and isinstance(perturb[0], str) else tuple(perturb or ())
    for node, _ in presses:
        if node not in perturb_nodes:
            raise ValueError(f"Perturbation node '{node}' not found.")
    p_cols = [perturb_nodes.index(node) for node, _ in presses]
    p_signs = [sign for _, sign in presses]
    if observe and not p_cols:
        raise ValueError("Observations require a perturbation.")

    def no_effect_cells(counts):
        return counts[:, p_cols] == 0 if p_cols else counts == 0

    tmat = sp.matrix2numpy(absolute_effects(G)).astype(int)
    no_effect = no_effect_cells(tmat)

    A_fn = sp.lambdify(symbols_sampled, A_sym)
    B_fn = sp.lambdify(symbols_sampled, B_sym) if n_u and B_sym else None
    C_fn = sp.lambdify(symbols_sampled, C_sym) if n_y and C_sym else None
    D_fn = sp.lambdify(symbols_sampled, D_sym) if D_sym and D_sym.shape != (0, 0) else None

    def evaluate_sample(values, mask):
        A = np.asarray(A_fn(*values), dtype=float).reshape(n_x, n_x)
        if not np.all(np.real(np.linalg.eigvals(A)) < 0):
            return None
        try:
            inv_A = np.linalg.inv(-A)
        except np.linalg.LinAlgError:
            return None
        B = B_fn(*values) if B_fn else np.zeros((n_x, 0))
        C = C_fn(*values) if C_fn else np.zeros((0, n_x))
        D = D_fn(*values) if D_fn else np.zeros((n_y, n_u))
        E = np.block([[inv_A, inv_A @ B], [C @ inv_A, C @ inv_A @ B + D]]) if n_u or n_y else inv_A
        if not np.isfinite(E).all():
            return None
        effect = np.where(mask, 0.0, E[:, p_cols] if p_cols else E)
        if p_cols:
            effect = np.sum(effect * p_signs, axis=1)
        return effect if np.isfinite(effect).all() else None

    group_idx = [[symbols_sampled.index(sp.Symbol(f"{_edge_prefix(G, u, v)}_{v},{u}")) for u, v in group]
                 for group in interactions]
    base_cls = {n: d.get("category", "state") for n, d in G.nodes(data=True)}
    masks: Dict[int, np.ndarray] = {}

    def get_zero_effect_mask(present):
        code = sum(int(keep) << i for i, keep in enumerate(present))
        if code not in masks:
            masks[code] = no_effect
            if interactions:
                variant = define_input_output(_build_model_variant(G, interactions, present))
                changed = [n for n, category in variant.nodes(data="category") if category != base_cls[n]]
                if changed:
                    raise ValueError(f"Nodes change category: {', '.join(map(str, changed))}")
                if not n_u and not n_y and all(variant.has_edge(node, node) for node in state):
                    reachable = [nx.descendants(variant, node) | {node}
                                 for node in ([node for node, _ in presses] if p_cols else state)]
                    masks[code] = np.array([[node not in reached for reached in reachable] for node in state])
                elif p_cols and not n_u and not n_y:
                    masks[code] = np.column_stack([
                        np.asarray(absolute_feedback_matrix(variant, node), dtype=int).ravel() == 0
                        for node, _ in presses
                    ])
                else:
                    masks[code] = no_effect_cells(np.asarray(absolute_effects(variant), dtype=int))
        return code, masks[code]

    sampled_index = {sym: i for i, sym in enumerate(symbols_sampled)}
    sample_functions = {
        sym: sp.lambdify(symbols_sampled, sym.subs(matrix_subs))
        for sym in fixed_subs if return_samples and sym not in uncertain_symbols
    }

    variants = itertools.product((False, True), repeat=len(interactions)) if uncertain_interactions == "enumerate" else (None,)
    for present in variants:
        effects, valid_sims, samples, structure_ids = [], [], [], []
        attempts, drawn = 0, 0
        if uncertain_interactions == "enumerate":
            masks.clear()
        while drawn < n_sim and attempts < max_attempts:
            attempts += 1
            sampled_structure = present
            values = _random_sampler(dist, len(symbols_sampled), rng)
            if fixed_fn:
                values[fixed_indices] = fixed_fn(*values)
            if sampled_structure is None:
                sampled_structure = tuple(rng.uniform(size=len(interactions)) < rng.uniform()) if interactions else ()
            code, mask = get_zero_effect_mask(sampled_structure)
            values[[i for keep, idx in zip(sampled_structure, group_idx) if not keep for i in idx]] = 0.0
            effect = evaluate_sample(values, mask)
            if effect is not None:
                effects.append(effect)
                structure_ids.append(code)
                valid_sims.append(all(np.sign(effect[response_idx[node]]) == obs for node, obs in observe or ()))
                if return_samples:
                    samples.append(values)
            drawn += effect is not None and (not condition or valid_sims[-1])
        if drawn < n_sim:
            label = f" for structure {get_zero_effect_mask(present)[0]}" if present is not None else ""
            raise RuntimeError(f"Maximum iterations reached{label}. Matched {drawn}/{n_sim} draws.")

        result = {
            "effects": effects,
            "valid_sims": valid_sims,
            "all_nodes": all_nodes,
            "tmat": tmat,
            "prop_stable": len(effects) / attempts,
            "attempts": attempts,
            "n_stable": len(effects),
            "structures": structure_ids,
            "perturb": presses,
            "interactions": interactions,
            "signs": {(u, v): sign for u, v, sign in G.edges(data="sign", default=1)},
        }
        if return_samples:
            n_samples = len(samples)
            sample_array = np.asarray(samples)
            result_samples = {}
            for sym in symbols_all:
                if sym in sample_functions:
                    values = sample_functions[sym](*sample_array.T)
                    result_samples[str(sym)] = np.broadcast_to(values, (n_samples,)).copy()
                elif sym in sampled_index:
                    idx = sampled_index[sym]
                    result_samples[str(sym)] = sample_array[:, idx].copy()
            result["samples"] = result_samples
        yield result


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
    uncertain_interactions: Literal["sample", "enumerate"] = "sample",
    pair_reciprocal: bool = True,
) -> sp.Matrix:
    """Performs numerical simulations of cumulative effects using random interaction strengths.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        n_sim: Stable draws per structure.
        dist: Distribution for sampling ("uniform", "weak", "moderate", "strong")
        seed: Random seed
        positive_only: Return just the proportion of positive responses instead of sign-dominant proportions
        presample: Optional callable passed through to get_simulations
        uncertain_interactions: Sample uncertain interactions, or average every structure equally.
        pair_reciprocal: Keep or drop reciprocal dashed edges together.

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
    positive, negative, count = 0, 0, 0
    for sims in iter_simulations(G, n_sim, dist, seed, presample=presample,
                                 uncertain_interactions=uncertain_interactions, pair_reciprocal=pair_reciprocal):
        pos, neg = _sign_counts(sims["effects"])
        positive += pos / sims["n_stable"]
        negative += neg / sims["n_stable"]
        count += 1
    positive, negative = positive / count, negative / count
    tmat = sims["tmat"]
    n_rows, n_cols = tmat.shape
    smat = positive if positive_only else np.where(negative > positive, -negative, positive)
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
    pair_reciprocal: bool = True,
) -> pd.DataFrame:
    """Summarise simulation effects across model variants for each response node.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        perturb: Comma-separated node:sign pairs applied simultaneously with equal unit magnitudes
        observe: Observation string (node:sign, comma-separated allowed) to filter simulations
        n_sim: Stable draws per structure.
        dist: Distribution for sampling
        seed: Random seed
        combinations: If True, evaluate every combination of dashed edges
        presample: Optional callable passed through to get_simulations
        pair_reciprocal: If True, a reciprocal pair of dashed edges is one uncertain interaction kept or dropped together

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
    G = define_input_output(G)
    variants = [define_input_output(g) for g in get_dashed_alternatives(G, combinations=combinations, pair_reciprocal=pair_reciprocal)]
    categories = dict(G.nodes(data="category"))
    for g in variants:
        changed = [n for n, c in g.nodes(data="category") if c != categories[n]]
        if changed:
            raise ValueError(f"Nodes change category: {', '.join(map(str, changed))}")
    observations = _parse_observations(observe) if observe else None
    rows = []

    for model_idx, g in enumerate(variants, start=1):
        response_nodes = get_nodes(g, "state") + get_nodes(g, "output")
        if not response_nodes:
            continue
        pert = _parse_perturbations(g, perturb)
        sims = next(iter_simulations(
            g,
            n_sim=n_sim,
            dist=dist,
            seed=seed,
            perturb=pert,
            observe=observations,
            presample=presample,
            condition=False,
        ))

        node_count = len(response_nodes)
        valid_effects = [effect[:node_count] for effect, valid in zip(sims["effects"], sims["valid_sims"]) if valid]
        valid_count = len(valid_effects)
        if valid_count:
            positive, negative = _sign_counts(valid_effects)
        else:
            negative = np.zeros(node_count, dtype=int)
            positive = np.zeros(node_count, dtype=int)
        no_effect = valid_count - negative - positive

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
