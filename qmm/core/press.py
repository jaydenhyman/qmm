"""Analyse direct and indirect effects of press perturbations."""

import numpy as np
import sympy as sp
from functools import cache
from scipy.stats import truncnorm
from .structure import create_matrix
from .helper import get_weight, get_nodes, sign_determinacy
from thewalrus import perm

from typing import Optional
import networkx as nx

@cache
def adjoint_matrix(G: nx.DiGraph, form: str = "symbolic", perturb: Optional[str] = None) -> sp.Matrix:
    """Calculate elements of classical adjoint matrix for press perturbation response.
    
    Args:
        G: NetworkX DiGraph representing signed digraph model
        form: Type of computation ('symbolic', 'signed')
        perturb: Node to perturb (None for full matrix)
        
    Returns:
        sp.Matrix: Classical adjoint matrix elements
    """
    A = create_matrix(G, form=form)
    A = sp.Matrix(-A)
    nodes = get_nodes(G, "state")
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
    """
    A = create_matrix(G, form="binary")
    A_np = np.array(sp.matrix2numpy(A), dtype=int)
    nodes = get_nodes(G, "state")
    n = A_np.shape[0]
    if perturb is not None:
        perturb_index = nodes.index(perturb)
        result = np.zeros(n, dtype=int)
        for j in range(n):
            minor = np.delete(np.delete(A_np, perturb_index, 0), j, 1)
            result[j] = int(perm(minor.astype(float)))
        return sp.Matrix(result)
    tmat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(A_np, j, 0), i, 1)
            tmat[i, j] = int(perm(minor.astype(float)))
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
def sign_determinacy_matrix(G: nx.DiGraph, method: str = "average", as_nan: bool = True, as_abs: bool = False, perturb: Optional[str] = None) -> sp.Matrix:
    """Calculate probability of a correct sign prediction (matches adjoint).

    Args:
        G: NetworkX DiGraph representing signed digraph model
        method: Method for computing determinacy ('average', '95_bound', 'simulation')
        as_nan: Return NaN for undefined ratios
        as_abs: Return absolute values
        perturb: Node to perturb (None for full matrix)
        
    Returns:
        sp.Matrix: Probability of sign determinacy
    """
    wmat = weighted_predictions_matrix(G, perturb=perturb, as_nan=as_nan, as_abs=as_abs)
    tmat = sp.Matrix(absolute_feedback_matrix(G, perturb=perturb))
    pmat = sign_determinacy(wmat, tmat, method)
    return sp.Matrix(pmat)

@cache
def numerical_simulations(G: nx.DiGraph, n_sim: int = 10000, dist: str = "uniform", seed: int = 42, 
                         as_nan: bool = True, as_abs: bool = False, positive_only: bool = False, 
                         match_adjoint: bool = False) -> sp.Matrix:
    """Calculate proportion of positive and negative responses from stable simulations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        n_sim: Number of simulations
        dist: Distribution for sampling ('uniform', 'weak', 'moderate', 'strong')
        seed: Random seed
        as_nan: Return NaN for undefined ratios
        positive_only: Return just the proportion of positive responses instead of sign-dominant proportions.
        as_abs: Return absolute values of proportions
        match_adjoint: If true, counts the proportion of simulations matching the sign of the adjoint matrix predictions.
        
    Returns:
        sp.Matrix: Average proportion of positive and negative responses
        
    Raises:
        ValueError: If invalid parameter combinations are used.
    """
    if positive_only and not as_nan:
        raise ValueError("Invalid parameter combination: positive_only=True requires as_nan=False")
    if as_abs and not as_nan:
        raise ValueError("Invalid parameter combination: as_abs=True requires as_nan=True")
    if match_adjoint:
        if as_abs:
            raise ValueError("Invalid parameter combination: match_adjoint=True requires as_abs=False")
        if not as_nan:
            raise ValueError("Invalid parameter combination: match_adjoint=True requires as_nan=True")
        if positive_only:
            raise ValueError("Invalid parameter combination: match_adjoint=True requires positive_only=False")

    np.random.seed(seed)
    A = create_matrix(G, form="symbolic", matrix_type="A")
    state_nodes = get_nodes(G, "state")
    n = len(state_nodes)
    symbols = list(A.free_symbols)
    A_sp = sp.lambdify(symbols, A)
    dist_funcs = {
        "uniform": lambda size: np.random.uniform(0, 1, size),
        "weak": lambda size: np.random.beta(1, 3, size),
        "moderate": lambda size: np.random.beta(2, 2, size),
        "strong": lambda size: np.random.beta(3, 1, size),
        "normal_weak": lambda size: truncnorm.rvs(a=0, b=3, loc=0, scale=1/3, size=size),
        "normal_moderate": lambda size: truncnorm.rvs(a=-3, b=3, loc=0.5, scale=1/6, size=size),
        "normal_strong": lambda size: truncnorm.rvs(a=-3, b=0, loc=1, scale=1/3, size=size)
    }
    if match_adjoint:
        adj_mat = adjoint_matrix(G, form="signed")
        adj_sign_np = np.array(adj_mat.applyfunc(sp.sign).tolist(), dtype=float)
        matches = np.zeros((n, n), dtype=int)
    else:
        positive = np.zeros((n, n), dtype=int)
        negative = np.zeros((n, n), dtype=int)
    total_simulations = 0
    while total_simulations < n_sim:
        values = dist_funcs[dist](len(symbols))
        sim_A = A_sp(*values)
        if np.all(np.real(np.linalg.eigvals(sim_A)) < 0):
            try:
                inv_A = np.linalg.inv(-sim_A)
                if match_adjoint:
                    matches += (np.sign(inv_A) == adj_sign_np)
                else:
                    positive += inv_A > 0
                    negative += inv_A < 0
                total_simulations += 1
            except np.linalg.LinAlgError:
                continue
    if total_simulations == 0:
        smat = np.full((n, n), np.nan)
    elif match_adjoint:
        smat = matches / total_simulations
    elif positive_only:
        smat = positive / total_simulations
    else:
        smat = np.where(negative > positive, -negative / total_simulations, positive / total_simulations)
    smat = sp.Matrix(smat.astype(float).tolist())
    if total_simulations > 0:
        tmat = absolute_feedback_matrix(G)
        tmat_np = np.array(tmat.tolist(), dtype=bool)
        smat = sp.Matrix([[sp.nan if not tmat_np[i, j] else smat[i, j] for j in range(n)] for i in range(n)])
        if as_abs and not match_adjoint:
            smat = sp.Matrix([[sp.Abs(x) if x != sp.nan else sp.nan for x in row] for row in smat.tolist()])

    if not as_nan:
        smat = sp.Matrix([[0 if sp.nan == x else x for x in row] for row in smat.tolist()])
    return sp.Matrix(smat)
