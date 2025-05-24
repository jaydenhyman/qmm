"""Analyse the stability properties of a system based on its structure."""

import numpy as np
import pandas as pd
import networkx as nx
import sympy as sp
from itertools import combinations
from functools import cache
from .structure import create_matrix
from .helper import get_positive, get_negative, get_weight
from thewalrus import perm
from typing import Optional

def _colour_test(G) -> str:
    A = create_matrix(G, form="signed")
    n = A.shape[0]
    colour = {i: "black" if A[i, i] != 0 else "white" for i in range(n)}
    if n <= 4 or "white" not in colour.values():
        return "Fail"
    else:
        while "white" in colour.values():
            progress_made = False
            for i in [i for i, c in colour.items() if c == "white"]:
                neighbours = [(j, colour[j]) for j in range(n) if A[i, j] * A[j, i] < 0]
                white_neighbours = [j for j, c in neighbours if c == "white"]
                if not white_neighbours or any(
                    sum(1 for k in range(n) if A[j, k] * A[k, j] < 0 and colour[k] == "white") <= 1
                    for j in [j for j, c in neighbours if c == "black"]
                ):
                    colour[i] = "black"
                    progress_made = True
                    break
            if not progress_made:
                return "Pass"
        return "Fail"

def sign_stability(G: nx.DiGraph) -> pd.DataFrame:
    """Evaluate necessary and sufficient conditions for sign stability including color test.
    
    Args:
        G: NetworkX DiGraph representing signed digraph model
        
    Returns:
        pd.DataFrame: Test results for sign stability conditions
    """
    A = sp.matrix2numpy(create_matrix(G, form="signed")).astype(int)
    n = A.shape[0]
    conditions = [
        all(A[i, i] <= 0 for i in range(n)),
        any(A[i, i] < 0 for i in range(n)),
        all(A[i, j] * A[j, i] <= 0 for i in range(n) for j in range(n) if i != j),
        all(len(cycle) < 3 for cycle in nx.simple_cycles(nx.DiGraph(A))),
        np.linalg.det(A) != 0,
    ]
    colour_result = _colour_test(G) == "Fail"
    is_sign_stable = all(conditions) and colour_result
    return pd.DataFrame(
        {
            "Test": [
                "Condition i",
                "Condition ii",
                "Condition iii",
                "Condition iv",
                "Condition v",
                "Colour test",
                "Sign stable",
            ],
            "Definition": [
                "No positive self-effects",
                "At least one node is self-regulating",
                "The product of any pairwise interaction is non-positive",
                "No cycles greater than length two",
                "Non-zero determinant (all nodes have at least " + "one incoming and outgoing link)",
                "Fails Jeffries' colour test",
                "Satisfies necessary and sufficient conditions for sign stability",
            ],
            "Result": conditions + [colour_result] + [is_sign_stable],
        }
    )

@cache
def system_feedback(G: nx.DiGraph, level: Optional[int] = None, form: str = "symbolic") -> sp.Matrix:
    """Calculate the product of conjunct and disjunct feedback cycles for any level of the system (coefficients of the characteristic polynomial).

    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Level of feedback to compute (None for all levels)
        form: Type of feedback ('symbolic', 'signed', or 'binary')

    Returns:
        sp.Matrix: Feedback cycle products at specified levels
    """
    A = create_matrix(G, form=form)
    if level == 0:
        return sp.Matrix([-1])
    n = A.shape[0]
    lam = sp.symbols("lambda")
    p = A.charpoly(lam).as_expr()
    if level is None:
        fb = [-p.coeff(lam, n - k) for k in range(n + 1)]
    else:
        fb = [-p.coeff(lam, n - level)]
    return sp.Matrix(fb)

@cache
def net_feedback(G: nx.DiGraph, level: Optional[int] = None) -> sp.Matrix:
    """Calculate net feedback at a specified level of the system.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Level of feedback to compute (None for all levels)
        
    Returns:
        sp.Matrix: Net feedback at specified levels
    """
    return system_feedback(G, level=level, form="signed")

@cache
def absolute_feedback(G: nx.DiGraph, level: Optional[int] = None, method: str = "combinations") -> sp.Matrix:
    """Calculate absolute feedback at a specified level of the system.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Level of feedback to compute (None for all levels) 
        method: Method for computing feedback ('combinations' or 'polynomial')
        
    Returns:
        sp.Matrix: Total number of feedback terms at specified levels
    """
    A = create_matrix(G, form="signed")
    if level == 0:
        return sp.Matrix([1])
    n = A.shape[0]
    if method == "combinations":
        A = sp.matrix2numpy(A).astype(int)
        A = np.abs(A)
        if level is None:
            fb = []
            for k in range(n + 1):
                fb_k = sum(perm(A[np.ix_(c, c)], method="glynn") for c in combinations(range(n), k))
                fb.append(int(fb_k))
        else:
            fb_k = sum(perm(A[np.ix_(c, c)], method="glynn") for c in combinations(range(n), level))
            fb = [int(fb_k)]
    elif method == "polynomial":
        lam = sp.Symbol("lambda")
        A_abs = sp.Matrix(sp.Abs(A) + lam * sp.eye(n))
        P = sp.per(A_abs)
        if level is None:
            fb = [P.coeff(lam, n - k) for k in range(n + 1)]
        else:
            fb = [P.coeff(lam, n - level)]
    return sp.Matrix(fb)

@cache
def weighted_feedback(G: nx.DiGraph, level: Optional[int] = None) -> sp.Matrix:
    """Calculate ratio of net to total feedback terms at each level of the system.
    
    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Level to compute weighted feedback (None for all levels)
        
    Returns:
        sp.Matrix: Weighted feedback metrics for each level
    """
    net_fb = net_feedback(G, level=level)
    tot_fb = absolute_feedback(G, level=level)
    return get_weight(net_fb, tot_fb)

def _hurwitz_matrix(fb, level) -> sp.Matrix:
    fb_pos = fb * sp.Integer(-1)
    if level == 0:
        return sp.Matrix([fb_pos[0]])
    H = sp.zeros(level, level)
    for i in range(level):
        for j in range(level):
            index = 2 * j - i + 1
            if 0 <= index < len(fb_pos):
                H[i, j] = fb_pos[index]
    return H

@cache
def feedback_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """Calculate net, absolute and weighted feedback metrics at each level of the system.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        
    Returns:
        pd.DataFrame: Feedback metrics for each system level
    """
    net = net_feedback(G)
    absolute = absolute_feedback(G)
    positive = get_positive(net, absolute)
    negative = get_negative(net, absolute)
    weighted = weighted_feedback(G)
    n = len(positive)
    levels = [str(i) for i in range(n)]

    df = {
        "Feedback level": levels,
        "Net": [net[i, 0] for i in range(n)],
        "Absolute": [absolute[i, 0] for i in range(n)],
        "Positive": [positive[i, 0] for i in range(n)],
        "Negative": [negative[i, 0] for i in range(n)],
        "Weighted": [weighted[i, 0] for i in range(n)],
    }

    return pd.DataFrame(df)

@cache
def hurwitz_determinants(G: nx.DiGraph, level: Optional[int] = None, form: str = "symbolic") -> sp.Matrix:
    """Calculate Hurwitz determinants for analysing system stability.

    Args:
        G: NetworkX DiGraph representing signed digraph model  
        level: Level to compute determinants (None for all Hurwitz determinants)
        form: Type of computation ('symbolic', 'signed', or 'binary')

    Returns:
        sp.Matrix: Hurwitz determinants at specified levels
    """
    fb = system_feedback(G, level=None, form=form)
    n = len(fb) - 1
    if n > 5 and form == "symbolic":
        raise ValueError("Limited to systems with five or fewer variables.")
    if level is None:
        h = _hurwitz_matrix(fb, n)
        hd = sp.Matrix([sp.det(h[:k, :k]) for k in range(0, n + 1)])
    else:
        h = _hurwitz_matrix(fb, level)
        hd = sp.Matrix([sp.det(h[:level, :level])])
    return sp.Matrix(hd)

@cache
def net_determinants(G: nx.DiGraph, level: Optional[int] = None) -> sp.Matrix:
    """Calculate net terms in Hurwitz determinants.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Level to compute determinants (None for all Hurwitz determinants)
        
    Returns:
        sp.Matrix: Net terms in Hurwitz determinants
    """
    return hurwitz_determinants(G, level=level, form="signed")

@cache
def absolute_determinants(G: nx.DiGraph, level: Optional[int] = None) -> sp.Matrix:
    """Calculate absolute terms in Hurwitz determinants.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Level to compute determinants (None for all Hurwitz determinants)
        
    Returns:
        sp.Matrix: Absolute terms in Hurwitz determinants
    """
    tot_fb = absolute_feedback(G)
    n = tot_fb.shape[0] - 1
    h = _hurwitz_matrix(tot_fb, n)
    if level is None:
        td = [sp.Integer(1)]
        for k in range(1, n + 1):
            h_k = np.array(h[:k, :k].tolist(), dtype=float)
            td.append(sp.Abs(sp.Integer(int(perm(h_k)))))
    else:
        if level < 0 or level > n:
            raise ValueError(f"Level must be between 0 and {n}")
        if level == 0:
            td = [sp.Integer(1)]
        else:
            H_k = np.array(h[:level, :level].tolist(), dtype=float)
            td = [sp.Abs(sp.Integer(int(perm(H_k))))]
    return sp.Matrix(td)

@cache
def weighted_determinants(G: nx.DiGraph, level: Optional[int] = None) -> sp.Matrix:
    """Calculate ratio of net to total terms for Hurwitz determinants.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Level to compute determinants (None for all Hurwitz determinants)
        
    Returns:
        sp.Matrix: Ratio of net to total terms for Hurwitz determinants
    """
    net_det = net_determinants(G, level=level)
    tot_det = absolute_determinants(G, level=level)
    wgt_det = get_weight(net_det, tot_det)
    return wgt_det

@cache
def determinants_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """Calculate net, absolute and weighted Hurwitz determinant metrics.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        
    Returns:
        pd.DataFrame: Hurwitz determinant metrics
    """
    net = net_determinants(G)
    absolute = absolute_determinants(G)
    weighted = weighted_determinants(G)
    n = len(net)
    levels = [str(i) for i in range(n)]
    df = {
        "Hurwitz determinant": levels,
        "Net": [net[i, 0] for i in range(n)],
        "Absolute": [absolute[i, 0] for i in range(n)],
        "Weighted": [weighted[i, 0] for i in range(n)],
    }
    return pd.DataFrame(df)

@cache
def _create_model_c(n: int) -> nx.DiGraph:
    C = nx.DiGraph()
    for i in range(n):
        C.add_node(i)
    for i in range(1, n):
        C.add_edge(i - 1, i, sign=-1)
        C.add_edge(i, i - 1, sign=1)
    C.add_edge(n - 1, n - 1, sign=-1)
    nx.set_node_attributes(C, "state", "category")
    return C

@cache
def conditional_stability(G: nx.DiGraph) -> pd.DataFrame:
    """Analyse conditional stability metrics and model stability class.

    Args:
        G: NetworkX DiGraph representing signed digraph model

    Returns:
        pd.DataFrame: Conditional stability metrics and model class
    """
    A = create_matrix(G, form="signed")
    n = A.shape[0]
    w_fb = weighted_feedback(G)
    w_det = weighted_determinants(G, level=n - 1)[0]
    C = _create_model_c(n)
    w_det_c = weighted_determinants(C, level=n - 1)[0]
    ratio_C = w_det / w_det_c
    max_fb_n = np.max(w_fb) == w_fb[-1]
    kmax = len(w_fb) - 1 - np.argmax(w_fb[::-1])
    is_sign_stable = sign_stability(G)["Result"].iloc[-1]
    if is_sign_stable:
        model_class = "Sign stable"
    elif max_fb_n and ratio_C >= 1:
        model_class = "Class I"
    else:
        model_class = "Class II"
    stability_metrics = pd.DataFrame(
        {
            "Test": [
                "Weighted feedback",
                "Weighted determinant",
                "Ratio to model-c system",
                "Model class",
            ],
            "Definition": [
                f"Maximum weighted feedback (level {kmax})",
                "n-1 weighted determinant at level",
                "Ratio to a 'model-c' type system",
                "Class of the model based on conditional stability metrics",
            ],
            "Result": [
                np.max(w_fb).evalf(2),
                w_det.evalf(2),
                ratio_C.evalf(2),
                model_class,
            ],
        }
    )
    return stability_metrics

@cache
def simulation_stability(G: nx.DiGraph, n_sim: int = 10000) -> pd.DataFrame:
    """Analyse stability using randomly sampled interaction strengths from a uniform distribution.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        n_sim: Number of simulations to perform (default 10000)
        
    Returns:
        pd.DataFrame: Proportion of stable matrices and proportion that fail Hurwitz criteria
    """
    A = create_matrix(G, "signed")
    A = sp.matrix2numpy(A).astype(int)
    n_stable = 0
    n_unstable = 0
    n_hurwitz_i_fail = 0
    n_hurwitz_ii_fail = 0
    n_hurwitz_i_only_fail = 0
    n_hurwitz_ii_only_fail = 0
    for _ in range(n_sim):
        M = np.random.rand(*A.shape)
        S = A * M
        if np.all(np.real(np.linalg.eigvals(S)) < 0):
            n_stable += 1
        else:
            n_unstable += 1
        pc = np.poly(S)
        hurwitz_i = np.all(pc[1:] > 0) or np.all(pc[1:] < 0)
        n = len(pc)
        H = np.zeros((n - 1, n - 1))
        for i in range(1, n):
            for j in range(1, n):
                index = 2 * j - i
                if 0 <= index < n:
                    H[i - 1, j - 1] = pc[index]
        hd = [np.linalg.det(H[: k + 1, : k + 1]) for k in range(n - 1)]
        hurwitz_ii = np.all(np.array(hd[1:-1]) > 0)
        if not hurwitz_i:
            n_hurwitz_i_fail += 1
            if hurwitz_ii:
                n_hurwitz_i_only_fail += 1
        if not hurwitz_ii:
            n_hurwitz_ii_fail += 1
            if hurwitz_i:
                n_hurwitz_ii_only_fail += 1
    prop_stable = n_stable / n_sim
    prop_unstable = n_unstable / n_sim
    prop_hurwitz_i_fail = n_hurwitz_i_fail / n_sim
    prop_hurwitz_ii_fail = n_hurwitz_ii_fail / n_sim
    prop_hurwitz_i_only_fail = n_hurwitz_i_only_fail / n_sim
    prop_hurwitz_ii_only_fail = n_hurwitz_ii_only_fail / n_sim
    sim_df = pd.DataFrame(
        {
            "Test": [
                "Stable matrices",
                "Unstable matrices",
                "Hurwitz criterion i",
                "Hurwitz criterion ii",
                "Hurwitz criterion i only",
                "Hurwitz criterion ii only",
            ],
            "Definition": [
                "Proportion where all eigenvalues have negative real parts",
                "Proportion where one or more eigenvalues have positive real parts",
                "Proportion where polynomial coefficients are not " + "all of the same sign",
                "Proportion where Hurwitz determinants are not all positive",
                "Proportion where only Hurwitz criterion i fails",
                "Proportion where only Hurwitz criterion ii fails",
            ],
            "Result": [
                f"{prop_stable:.2%}",
                f"{prop_unstable:.2%}",
                f"{prop_hurwitz_i_fail:.2%}",
                f"{prop_hurwitz_ii_fail:.2%}",
                f"{prop_hurwitz_i_only_fail:.2%}",
                f"{prop_hurwitz_ii_only_fail:.2%}",
            ],
        }
    )
    return sim_df
