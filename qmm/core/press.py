"""Analyse direct and indirect effects of press perturbations."""

import numpy as np
import sympy as sp
from .structure import create_matrix
from .helper import (
    get_weight,
    get_nodes,
    sign_determinacy,
    _random_sampler,
    perm,
)

from typing import Optional, Literal
import networkx as nx

def adjoint_matrix(
    G: nx.DiGraph,
    form: Literal["symbolic", "signed"] = "symbolic",
    perturb: Optional[str] = None,
) -> sp.Matrix:
    """Calculate elements of classical adjoint matrix for press perturbation response.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        form: Type of computation ('symbolic', 'signed')
        perturb: Node to perturb (None for full matrix)

    Returns:
        sp.Matrix: Classical adjoint matrix elements

    References:
        - Dambacher, J.M., Li, H.W., Rossignol, P.A. (2002). Relevance of Community Structure in Assessing Indeterminacy of Ecological Predictions. Ecology 83, 1372–1385.

    Examples:
        ```python
        from qmm import load_digraph, adjoint_matrix
        adjoint_matrix(load_digraph("snowshoe_rp"), form='symbolic')
        # Matrix([
        # [               a_C,P*a_P,C,              -a_P,P*a_R,C,  a_C,P*a_R,C],
        # [-a_C,P*a_P,R + a_C,R*a_P,P,               a_P,P*a_R,R, -a_C,P*a_R,R],
        # [               a_C,R*a_P,C, a_P,C*a_R,R - a_P,R*a_R,C,  a_C,R*a_R,C]])

        adjoint_matrix(load_digraph("snowshoe_rp"), form='symbolic', perturb='R')
        # Matrix([
        # [               a_C,P*a_P,C],
        # [-a_C,P*a_P,R + a_C,R*a_P,P],
        # [               a_C,R*a_P,C]])

        adjoint_matrix(load_digraph("snowshoe_rp"), form='signed')
        # Matrix([
        # [1, -1,  1],
        # [0,  1, -1],
        # [1,  0,  1]])

        adjoint_matrix(load_digraph("snowshoe_rp"), form='signed', perturb='R')
        # Matrix([
        # [1],
        # [0],
        # [1]])
        ```
    """
    A = create_matrix(G, form=form)
    A = sp.Matrix(-A)
    nodes = get_nodes(G, "state")
    if perturb is not None and perturb not in nodes:
        raise ValueError(f"Perturbation node must be one of: {nodes}")
    n = len(nodes)
    if perturb is not None:
        src_id = nodes.index(perturb)
        return sp.Matrix([sp.Integer(-1) ** (src_id + j) * A.minor(src_id, j) for j in range(n)])
    adjoint_matrix = sp.expand(A.adjugate())
    return sp.Matrix(adjoint_matrix)


def absolute_feedback_matrix(G: nx.DiGraph, perturb: Optional[str] = None) -> sp.Matrix:
    """Calculate total number of both positive and negative terms for press perturbation response.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        perturb: Node to perturb (None for full matrix)

    Returns:
        sp.Matrix: Absolute feedback matrix elements

    References:
        - Dambacher, J.M., Li, H.W., Rossignol, P.A. (2002). Relevance of Community Structure in Assessing Indeterminacy of Ecological Predictions. Ecology 83, 1372–1385.

    Examples:
        ```python
        from qmm import load_digraph, absolute_feedback_matrix
        absolute_feedback_matrix(load_digraph("snowshoe_rp"))
        # Matrix([
        # [1, 1, 1],
        # [2, 1, 1],
        # [1, 2, 1]])

        absolute_feedback_matrix(load_digraph("snowshoe_rp"), perturb='R')
        # Matrix([
        # [1],
        # [2],
        # [1]])
        ```
    """
    A = sp.matrix2numpy(create_matrix(G, form="binary"), dtype=int)
    nodes = get_nodes(G, "state")
    if perturb is not None and perturb not in nodes:
        raise ValueError(f"Perturbation node must be one of: {nodes}")
    if perturb is not None:
        return sp.Matrix(perm(A, source=nodes.index(perturb)))
    return sp.Matrix([perm(A, source=j) for j in range(len(nodes))]).T

def weighted_predictions_matrix(G: nx.DiGraph, as_nan: bool = True, as_abs: bool = False, perturb: Optional[str] = None) -> sp.Matrix:
    """Calculate ratio of net to total terms for a press perturbation response.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        as_nan: Return NaN for undefined ratios
        as_abs: Return absolute values
        perturb: Node to perturb (None for full matrix)

    Returns:
        sp.Matrix: Prediction weights

    References:
        - Dambacher, J.M., Li, H.W., Rossignol, P.A. (2002). Relevance of Community Structure in Assessing Indeterminacy of Ecological Predictions. Ecology 83, 1372–1385.

    Examples:
        ```python
        from qmm import load_digraph, weighted_predictions_matrix
        weighted_predictions_matrix(load_digraph("snowshoe_rp"))
        # Matrix([
        # [1, -1,  1],
        # [0,  1, -1],
        # [1,  0,  1]])

        weighted_predictions_matrix(load_digraph("snowshoe_rp"), perturb='R')
        # Matrix([
        # [1],
        # [0],
        # [1]])

        weighted_predictions_matrix(load_digraph("snowshoe_rp"), as_abs=True)
        # Matrix([
        # [1, 1, 1],
        # [0, 1, 1],
        # [1, 0, 1]])

        weighted_predictions_matrix(load_digraph("snowshoe_rp"), as_abs=False)
        # Matrix([
        # [1, -1,  1],
        # [0,  1, -1],
        # [1,  0,  1]])
        ```
    """
    amat = adjoint_matrix(G, perturb=perturb, form="signed")
    if as_abs:
        amat = sp.Abs(amat)
    tmat = absolute_feedback_matrix(G, perturb=perturb)
    if as_nan:
        wmat = get_weight(amat, tmat)
    else:
        wmat = get_weight(amat, tmat, sp.Integer(1))
    return sp.Matrix(wmat)

def sign_determinacy_matrix(
    G: nx.DiGraph,
    method: Literal["average", "95_bound"] = "average",
    as_nan: bool = True,
    as_abs: bool = False,
    perturb: Optional[str] = None,
) -> sp.Matrix:
    """Calculate probability of a correct sign prediction (matches adjoint).

    Args:
        G: NetworkX DiGraph representing signed digraph model
        method: Method for computing determinacy ('average', '95_bound')
        as_nan: Return NaN for undefined ratios
        as_abs: Return absolute values
        perturb: Node to perturb (None for full matrix)

    Returns:
        sp.Matrix: Probability of sign determinacy

    References:
        - Hosack, G.R., Hayes, K.R., Dambacher, J.M. (2008). Assessing Model Structure Uncertainty Through an Analysis of System Feedback and Bayesian Networks. Ecological Applications 18, 1070–1082.

    Examples:
        ```python
        from qmm import load_digraph, sign_determinacy_matrix
        sign_determinacy_matrix(load_digraph("snowshoe_rp"), method='average')
        # Matrix([
        # [  1,  -1,  1],
        # [1/2,   1, -1],
        # [  1, 1/2,  1]])

        sign_determinacy_matrix(load_digraph("snowshoe_rp"), method='average', perturb='R')
        # Matrix([
        # [  1],
        # [1/2],
        # [  1]])

        sign_determinacy_matrix(load_digraph("snowshoe_rp"), method='average', as_abs=True)
        # Matrix([
        # [  1,   1, 1],
        # [1/2,   1, 1],
        # [  1, 1/2, 1]])

        sign_determinacy_matrix(load_digraph("snowshoe_rp"), method='average', as_abs=False)
        # Matrix([
        # [  1,  -1,  1],
        # [1/2,   1, -1],
        # [  1, 1/2,  1]])
        ```
    """
    wmat = weighted_predictions_matrix(G, perturb=perturb, as_nan=as_nan, as_abs=as_abs)
    tmat = sp.Matrix(absolute_feedback_matrix(G, perturb=perturb))
    pmat = sign_determinacy(wmat, tmat, method)
    return sp.Matrix(pmat)

def numerical_simulations(
    G: nx.DiGraph,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    as_nan: bool = True,
    mode: Literal["dominant", "absolute", "positive", "match_adjoint"] = "dominant",
) -> sp.Matrix:
    """Calculate proportion of positive and negative responses from stable simulations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        n_sim: Nonnegative integer number of simulations
        dist: Distribution for sampling ('uniform', 'weak', 'moderate', 'strong')
        seed: Random seed
        as_nan: Return NaN for undefined ratios
        mode: Response summary: 'dominant' (signed proportion of the dominant sign),
            'absolute' (unsigned proportion of the dominant sign), 'positive'
            (proportion of positive responses) or 'match_adjoint' (proportion
            matching the sign of the adjoint; ambiguous adjoint entries return 0.5).

    References:
        - Dambacher, J.M., Li, H.W., Rossignol, P.A. (2003). Qualitative predictions in model ecosystems. Ecological Modelling 161, 79–93.

    Returns:
        sp.Matrix: Average proportion of positive and negative responses

    Raises:
        ValueError: If mode is invalid or incompatible with as_nan.

    Examples:
        ```python
        from qmm import load_digraph, numerical_simulations
        numerical_simulations(load_digraph("snowshoe_rp"), n_sim=1000, seed=42)
        # Matrix([
        # [  1.0,  -1.0,  1.0],
        # [0.621,   1.0, -1.0],
        # [  1.0, 0.639,  1.0]])

        numerical_simulations(load_digraph("snowshoe_rp"), n_sim=1000, seed=42, mode='absolute')
        # Matrix([
        # [  1.0,   1.0, 1.0],
        # [0.621,   1.0, 1.0],
        # [  1.0, 0.639, 1.0]])

        numerical_simulations(load_digraph("snowshoe_rp"), n_sim=1000, seed=42, mode='match_adjoint')
        # Matrix([
        # [1.0, 1.0, 1.0],
        # [0.5, 1.0, 1.0],
        # [1.0, 0.5, 1.0]])

        numerical_simulations(load_digraph("snowshoe_rp"), n_sim=1000, seed=42, mode='positive')
        # Matrix([
        # [  1.0,   0.0, 1.0],
        # [0.621,   1.0, 0.0],
        # [  1.0, 0.639, 1.0]])
        ```
    """
    if isinstance(n_sim, (bool, np.bool_)) or not isinstance(n_sim, (int, np.integer)) or n_sim < 0:
        raise ValueError("n_sim must be a nonnegative integer")
    if mode not in ("dominant", "absolute", "positive", "match_adjoint"):
        raise ValueError("Invalid mode. Choose 'dominant', 'absolute', 'positive', 'match_adjoint'.")
    if mode in ("absolute", "positive") and not as_nan:
        raise ValueError(f"mode='{mode}' requires as_nan=True")

    rng = np.random.RandomState(seed)
    A = create_matrix(G, form="symbolic", matrix_type="A")
    state_nodes = get_nodes(G, "state")
    n = len(state_nodes)
    symbols = sorted(list(A.free_symbols), key=str)
    A_sp = sp.lambdify(symbols, A)
    positive = np.zeros((n, n), dtype=int)
    negative = np.zeros((n, n), dtype=int)
    total_simulations = 0
    if mode == "match_adjoint":
        adjoint_signs_np = np.array(
            adjoint_matrix(G, form="signed").tolist(), dtype=float
        )
    attempts, max_attempts = 0, n_sim * 100
    while total_simulations < n_sim and attempts < max_attempts:
        attempts += 1
        values = _random_sampler(dist, len(symbols), rng)
        sim_A = np.asarray(A_sp(*values), dtype=float).reshape(n, n)
        if np.all(np.real(np.linalg.eigvals(sim_A)) < 0):
            try:
                inv_A = np.linalg.inv(-sim_A)
                if not np.isfinite(inv_A).all():
                    continue
                positive += inv_A > 0
                negative += inv_A < 0
                total_simulations += 1
            except np.linalg.LinAlgError:
                continue
    if total_simulations < n_sim and n_sim > 0:
        raise RuntimeError(f"Maximum iterations reached. Stable proportion: {total_simulations / max_attempts:.4f}")
    if total_simulations == 0:
        smat = np.full((n, n), np.nan)
    elif mode == "positive":
        smat = positive / total_simulations
    elif mode == "match_adjoint":
        matches = np.where(adjoint_signs_np > 0, positive,
                  np.where(adjoint_signs_np < 0, negative, 0))
        smat = np.where(adjoint_signs_np == 0, 0.5,
                        matches / total_simulations)
    else:
        smat = np.where(negative > positive, -negative / total_simulations, positive / total_simulations)
    smat = sp.Matrix(smat.tolist())
    if total_simulations > 0:
        tmat = absolute_feedback_matrix(G)
        tmat_np = np.array(tmat.tolist(), dtype=bool)
        smat = sp.Matrix([[sp.nan if not tmat_np[i, j] else smat[i, j] for j in range(n)] for i in range(n)])
        if mode == "absolute":
            smat = sp.Matrix([[sp.Abs(x) if x != sp.nan else sp.nan for x in row] for row in smat.tolist()])

    if not as_nan:
        fill = sp.Rational(1, 2) if mode == "match_adjoint" else 0
        smat = sp.Matrix([[fill if sp.nan == x else x for x in row] for row in smat.tolist()])
    return sp.Matrix(smat)
