"""Analyse direct and indirect effects of press perturbations."""

import numpy as np
import sympy as sp
from functools import cache
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

@cache
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
        adjoint_matrix(load_digraph("snowshoe"), form='symbolic')
        # Matrix([
        # [               a_H,P*a_P,H,              -a_P,P*a_V,H,  a_H,P*a_V,H],
        # [-a_H,P*a_P,V + a_H,V*a_P,P,               a_P,P*a_V,V, -a_H,P*a_V,V],
        # [               a_H,V*a_P,H, a_P,H*a_V,V - a_P,V*a_V,H,  a_H,V*a_V,H]])

        adjoint_matrix(load_digraph("snowshoe"), form='symbolic', perturb='V')
        # Matrix([
        # [               a_H,P*a_P,H],
        # [-a_H,P*a_P,V + a_H,V*a_P,P],
        # [               a_H,V*a_P,H]])

        adjoint_matrix(load_digraph("snowshoe"), form='signed')
        # Matrix([
        # [1, -1,  1],
        # [0,  1, -1],
        # [1,  0,  1]])

        adjoint_matrix(load_digraph("snowshoe"), form='signed', perturb='V')
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

@cache
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
        absolute_feedback_matrix(load_digraph("snowshoe"))
        # Matrix([
        # [1, 1, 1],
        # [2, 1, 1],
        # [1, 2, 1]])

        absolute_feedback_matrix(load_digraph("snowshoe"), perturb='V')
        # Matrix([
        # [1],
        # [2],
        # [1]])
        ```
    """
    A = create_matrix(G, form="binary")
    A_np = np.array(sp.matrix2numpy(A), dtype=float)
    nodes = get_nodes(G, "state")
    if perturb is not None and perturb not in nodes:
        raise ValueError(f"Perturbation node must be one of: {nodes}")
    n = A_np.shape[0]
    if perturb is not None:
        perturb_index = nodes.index(perturb)
        result = np.zeros(n, dtype=int)
        for j in range(n):
            minor = np.delete(np.delete(A_np, perturb_index, 0), j, 1)
            result[j] = int(perm(minor))
        return sp.Matrix(result)
    tmat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(A_np, j, 0), i, 1)
            tmat[i, j] = int(perm(minor))
    return sp.Matrix(tmat)

@cache
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
        weighted_predictions_matrix(load_digraph("snowshoe"))
        # Matrix([
        # [1, -1,  1],
        # [0,  1, -1],
        # [1,  0,  1]])

        weighted_predictions_matrix(load_digraph("snowshoe"), perturb='V')
        # Matrix([
        # [1],
        # [0],
        # [1]])

        weighted_predictions_matrix(load_digraph("snowshoe"), as_abs=True)
        # Matrix([
        # [1, 1, 1],
        # [0, 1, 1],
        # [1, 0, 1]])

        weighted_predictions_matrix(load_digraph("snowshoe"), as_abs=False)
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

@cache
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
        sign_determinacy_matrix(load_digraph("snowshoe"), method='average')
        # Matrix([
        # [  1,  -1,  1],
        # [1/2,   1, -1],
        # [  1, 1/2,  1]])

        sign_determinacy_matrix(load_digraph("snowshoe"), method='average', perturb='V')
        # Matrix([
        # [  1],
        # [1/2],
        # [  1]])

        sign_determinacy_matrix(load_digraph("snowshoe"), method='average', as_abs=True)
        # Matrix([
        # [  1,   1, 1],
        # [1/2,   1, 1],
        # [  1, 1/2, 1]])

        sign_determinacy_matrix(load_digraph("snowshoe"), method='average', as_abs=False)
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

@cache
def numerical_simulations(
    G: nx.DiGraph,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    as_nan: bool = True,
    as_abs: bool = False,
    positive_only: bool = False,
) -> sp.Matrix:
    """Calculate proportion of positive and negative responses from stable simulations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        n_sim: Number of simulations
        dist: Distribution for sampling ('uniform', 'weak', 'moderate', 'strong')
        seed: Random seed
        as_nan: Return NaN for undefined ratios
        positive_only: Return just the proportion of positive responses instead of sign-dominant proportions.
        as_abs: Return absolute values

    References:
        - Dambacher, J.M., Li, H.W., Rossignol, P.A. (2003). Qualitative predictions in model ecosystems. Ecological Modelling 161, 79–93.

    Returns:
        sp.Matrix: Average proportion of positive and negative responses

    Raises:
        ValueError: If invalid parameter combinations are used.

    Examples:
        ```python
        from qmm import load_digraph, numerical_simulations
        numerical_simulations(load_digraph("snowshoe"), n_sim=1000, seed=42)
        # Matrix([
        # [  1.0,  -1.0,  1.0],
        # [0.621,   1.0, -1.0],
        # [  1.0, 0.639,  1.0]])

        numerical_simulations(load_digraph("snowshoe"), n_sim=1000, seed=42, as_abs=True)
        # Matrix([
        # [  1.0,   1.0, 1.0],
        # [0.621,   1.0, 1.0],
        # [  1.0, 0.639, 1.0]])

        numerical_simulations(load_digraph("snowshoe"), n_sim=1000, seed=42, as_abs=False)
        # Matrix([
        # [  1.0,  -1.0,  1.0],
        # [0.621,   1.0, -1.0],
        # [  1.0, 0.639,  1.0]])

        numerical_simulations(load_digraph("snowshoe"), n_sim=1000, seed=42, positive_only=True)
        # Matrix([
        # [  1.0,   0.0, 1.0],
        # [0.621,   1.0, 0.0],
        # [  1.0, 0.639, 1.0]])
        ```
    """
    if positive_only and not as_nan:
        raise ValueError("Invalid parameter combination: positive_only=True requires as_nan=True")
    if as_abs and not as_nan:
        raise ValueError("Invalid parameter combination: as_abs=True requires as_nan=True")

    np.random.seed(seed)
    A = create_matrix(G, form="symbolic", matrix_type="A")
    state_nodes = get_nodes(G, "state")
    n = len(state_nodes)
    symbols = sorted(list(A.free_symbols), key=str)
    A_sp = sp.lambdify(symbols, A)
    positive = np.zeros((n, n), dtype=int)
    negative = np.zeros((n, n), dtype=int)
    total_simulations = 0
    while total_simulations < n_sim:
        values = _random_sampler(dist, len(symbols))
        sim_A = A_sp(*values)
        if np.all(np.real(np.linalg.eigvals(sim_A)) < 0):
            try:
                inv_A = np.linalg.inv(-sim_A)
                positive += inv_A > 0
                negative += inv_A < 0
                total_simulations += 1
            except np.linalg.LinAlgError:
                continue
    if total_simulations == 0:
        smat = np.full((n, n), np.nan)
    elif positive_only:
        smat = positive / total_simulations
    else:
        smat = np.where(negative > positive, -negative / total_simulations, positive / total_simulations)
    smat = sp.Matrix(smat.tolist())
    if total_simulations > 0:
        tmat = absolute_feedback_matrix(G)
        tmat_np = np.array(tmat.tolist(), dtype=bool)
        smat = sp.Matrix([[sp.nan if not tmat_np[i, j] else smat[i, j] for j in range(n)] for i in range(n)])
        if as_abs:
            smat = sp.Matrix([[sp.Abs(x) if x != sp.nan else sp.nan for x in row] for row in smat.tolist()])

    if not as_nan:
        smat = sp.Matrix([[0 if sp.nan == x else x for x in row] for row in smat.tolist()])
    return sp.Matrix(smat)
