"""Analyse the stability properties of a system based on its structure."""

import numpy as np
import pandas as pd
import networkx as nx
import sympy as sp
from itertools import combinations
from functools import cache
from .structure import create_matrix
from .helper import get_positive, get_negative, get_weight, perm, _random_sampler
from typing import Optional, Literal, Union

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

    References:
        - May, R.M. (1973). Qualitative Stability in Model Ecosystems. Ecology 54, 638–641.
        - Jeffries, C. (1974). Qualitative Stability and Digraphs in Model Ecosystems. Ecology 55, 1415–1419.

    Examples:
        ```python
        from qmm import load_digraph, sign_stability
        sign_stability(load_digraph("snowshoe"))
        #             Test                                                                     Definition  Result
        # 0    Condition i                                                       No positive self-effects    True
        # 1   Condition ii                                           At least one node is self-regulating    True
        # 2  Condition iii                        The product of any pairwise interaction is non-positive    True
        # 3   Condition iv                                              No cycles greater than length two   False
        # 4    Condition v  Non-zero determinant (all nodes have at least one incoming and outgoing link)    True
        # 5    Colour test                                                    Fails Jeffries' colour test    True
        # 6    Sign stable               Satisfies necessary and sufficient conditions for sign stability   False
        ```
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
def system_feedback(
    G: nx.DiGraph,
    level: Optional[int] = None,
    form: Literal["symbolic", "signed"] = "symbolic",
) -> sp.Matrix:
    """Calculate the product of conjunct and disjunct feedback cycles for any level of the system (coefficients of the characteristic polynomial).

    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Level of feedback to compute (None for all levels)
        form: Type of feedback ('symbolic', 'signed', or 'binary')

    Returns:
        sp.Matrix: Feedback cycle products at specified levels

    References:
        - Lyapunov, A.M. (1892). The general problem of the stability of motion. International Journal of Control 55, 531–534.
        - Levins, R. (1974). The qualitative analysis of partially specified systems. Annals of the New York Academy of Sciences 231, 123–138.
        - Puccia, C.J., Levins, R. (1985). Qualitative Modeling of Complex Systems: An Introduction to Loop Analysis and Time Averaging. Harvard University Press.

    Examples:
        ```python
        from qmm import load_digraph, system_feedback
        system_feedback(load_digraph("snowshoe"), level=2, form='symbolic')
        # Matrix([[-a_H,P*a_P,H - a_H,V*a_V,H - a_P,P*a_V,V]])

        system_feedback(load_digraph("snowshoe"), level=3, form='symbolic')
        # Matrix([[-a_H,P*a_P,H*a_V,V + a_H,P*a_P,V*a_V,H - a_H,V*a_P,P*a_V,H]])

        system_feedback(load_digraph("snowshoe"), form='symbolic')
        # Matrix([
        # [                                                        -1],
        # [                                            -a_P,P - a_V,V],
        # [                  -a_H,P*a_P,H - a_H,V*a_V,H - a_P,P*a_V,V],
        # [-a_H,P*a_P,H*a_V,V + a_H,P*a_P,V*a_V,H - a_H,V*a_P,P*a_V,H]])
        ```
    """
    A = create_matrix(G, form=form)
    if level == 0:
        return sp.Matrix([-1])
    n = A.shape[0]
    if form not in ("symbolic", "signed"):
        raise ValueError("form must be either 'symbolic' or 'signed'")
    if level is not None and (level < 0 or level > n):
        raise ValueError(f"Level must be between 0 and {n}")
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

    References:
        - Dambacher, J.M., Luh, H.-K., Li, H.W., Rossignol, P.A. (2003). Qualitative stability and ambiguity in model ecosystems. The American Naturalist 161, 876–888.

    Examples:
        ```python
        from qmm import load_digraph, net_feedback
        net_feedback(load_digraph("snowshoe"), level=2)
        # Matrix([[-3]])

        net_feedback(load_digraph("snowshoe"))
        # Matrix([
        # [-1],
        # [-2],
        # [-3],
        # [-1]])
        ```
    """
    return system_feedback(G, level=level, form="signed")

@cache
def absolute_feedback(
    G: nx.DiGraph,
    level: Optional[int] = None,
    method: Literal["combinations", "polynomial"] = "combinations",
) -> sp.Matrix:
    """Calculate absolute feedback at a specified level of the system.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Level of feedback to compute (None for all levels)
        method: Method for computing feedback ('combinations' or 'polynomial')

    Returns:
        sp.Matrix: Total number of feedback terms at specified levels

    References:
        - Dambacher, J.M., Luh, H.-K., Li, H.W., Rossignol, P.A. (2003). Qualitative stability and ambiguity in model ecosystems. The American Naturalist 161, 876–888.

    Examples:
        ```python
        from qmm import load_digraph, absolute_feedback
        absolute_feedback(load_digraph("snowshoe"), level=2)
        # Matrix([[3]])

        absolute_feedback(load_digraph("snowshoe"))
        # Matrix([
        # [1],
        # [2],
        # [3],
        # [3]])
        ```
    """
    A = create_matrix(G, form="signed")
    if level == 0:
        return sp.Matrix([1])
    n = A.shape[0]
    if level is not None and (level < 0 or level > n):
        raise ValueError(f"Level must be between 0 and {n}")
    if method == "combinations":
        A = sp.matrix2numpy(A).astype(int)
        A = np.abs(A)
        if level is None:
            fb = []
            for k in range(n + 1):
                fb_k = sum(perm(A[np.ix_(c, c)], method="bbfg") for c in combinations(range(n), k))
                fb.append(int(fb_k))
        else:
            fb_k = sum(perm(A[np.ix_(c, c)], method="bbfg") for c in combinations(range(n), level))
            fb = [int(fb_k)]
    elif method == "polynomial":
        lam = sp.Symbol("lambda")
        A_abs = sp.Matrix(sp.Abs(A) + lam * sp.eye(n))
        P = sp.per(A_abs)
        if level is None:
            fb = [P.coeff(lam, n - k) for k in range(n + 1)]
        else:
            fb = [P.coeff(lam, n - level)]
    else:
        raise ValueError("method must be either 'combinations' or 'polynomial'")
    return sp.Matrix(fb)

@cache
def weighted_feedback(G: nx.DiGraph, level: Optional[int] = None) -> sp.Matrix:
    """Calculate ratio of net to total feedback terms at each level of the system.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Level to compute weighted feedback (None for all levels)

    Returns:
        sp.Matrix: Weighted feedback metrics for each level

    References:
        - Dambacher, J.M., Luh, H.-K., Li, H.W., Rossignol, P.A. (2003). Qualitative stability and ambiguity in model ecosystems. The American Naturalist 161, 876–888.

    Examples:
        ```python
        from qmm import load_digraph, weighted_feedback
        weighted_feedback(load_digraph("snowshoe"), level=2)
        # Matrix([[-1]])

        weighted_feedback(load_digraph("snowshoe"))
        # Matrix([
        # [  -1],
        # [  -1],
        # [  -1],
        # [-1/3]])
        ```
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

    References:
        - Dambacher, J.M., Luh, H.-K., Li, H.W., Rossignol, P.A. (2003). Qualitative stability and ambiguity in model ecosystems. The American Naturalist 161, 876–888.

    Examples:
        ```python
        from qmm import load_digraph, feedback_metrics
        feedback_metrics(load_digraph("snowshoe"))
        #   Feedback level Net Absolute Positive Negative Weighted
        # 0              0  -1        1        0        1       -1
        # 1              1  -2        2        0        2       -1
        # 2              2  -3        3        0        3       -1
        # 3              3  -1        3        1        2     -1/3
        ```
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
def hurwitz_determinants(
    G: nx.DiGraph,
    level: Optional[int] = None,
    form: Literal["symbolic", "signed"] = "symbolic",
) -> sp.Matrix:
    """Calculate Hurwitz determinants for analysing system stability.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Level to compute determinants (None for all Hurwitz determinants)
        form: Type of computation ('symbolic' or 'signed')

    Returns:
        sp.Matrix: Hurwitz determinants at specified levels

    References:
        - Hurwitz, A. (1895). On the conditions under which an equation has only roots with negative real parts. Mathematische Annelen 65, 273–284.
        - Puccia, C.J., Levins, R. (1985). Qualitative Modeling of Complex Systems: An Introduction to Loop Analysis and Time Averaging. Harvard University Press.

    Examples:
        ```python
        from qmm import load_digraph, hurwitz_determinants
        hurwitz_determinants(load_digraph("snowshoe"), level=2, form='symbolic')
        # Matrix([[a_H,P*a_P,H*a_P,P + a_H,P*a_P,V*a_V,H + a_H,V*a_V,H*a_V,V + a_P,P**2*a_V,V + a_P,P*a_V,V**2]])
        ```
    """
    fb = system_feedback(G, level=None, form=form)
    n = len(fb) - 1
    if n > 5 and form == "symbolic":
        raise ValueError("Limited to systems with five or fewer variables.")
    if level is not None and (level < 0 or level > n):
        raise ValueError(f"Level must be between 0 and {n}")
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

    References:
        - Dambacher, J.M., Luh, H.-K., Li, H.W., Rossignol, P.A. (2003). Qualitative stability and ambiguity in model ecosystems. The American Naturalist 161, 876–888.

    Examples:
        ```python
        from qmm import load_digraph, net_determinants
        net_determinants(load_digraph("snowshoe"), level=2)
        # Matrix([[5]])

        net_determinants(load_digraph("snowshoe"))
        # Matrix([
        # [1],
        # [2],
        # [5],
        # [5]])
        ```
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

    References:
        - Dambacher, J.M., Luh, H.-K., Li, H.W., Rossignol, P.A. (2003). Qualitative stability and ambiguity in model ecosystems. The American Naturalist 161, 876–888.

    Examples:
        ```python
        from qmm import load_digraph, absolute_determinants
        absolute_determinants(load_digraph("snowshoe"), level=2)
        # Matrix([[9]])

        absolute_determinants(load_digraph("snowshoe"))
        # Matrix([
        # [ 1],
        # [ 2],
        # [ 9],
        # [27]])
        ```
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

    References:
        - Dambacher, J.M., Luh, H.-K., Li, H.W., Rossignol, P.A. (2003). Qualitative stability and ambiguity in model ecosystems. The American Naturalist 161, 876–888.

    Examples:
        ```python
        from qmm import load_digraph, weighted_determinants
        weighted_determinants(load_digraph("snowshoe"), level=2)
        # Matrix([[5/9]])

        weighted_determinants(load_digraph("snowshoe"))
        # Matrix([
        # [   1],
        # [   1],
        # [ 5/9],
        # [5/27]])
        ```
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

    References:
        - Dambacher, J.M., Luh, H.-K., Li, H.W., Rossignol, P.A. (2003). Qualitative stability and ambiguity in model ecosystems. The American Naturalist 161, 876–888.

    Examples:
        ```python
        from qmm import load_digraph, determinants_metrics
        determinants_metrics(load_digraph("snowshoe"))
        #   Hurwitz determinant Net Absolute Weighted
        # 0                   0   1        1        1
        # 1                   1   2        2        1
        # 2                   2   5        9      5/9
        # 3                   3   5       27     5/27
        ```
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
    nx.freeze(C)
    return C

@cache
def conditional_stability(G: nx.DiGraph) -> pd.DataFrame:
    """Analyse conditional stability metrics and model stability class.

    Args:
        G: NetworkX DiGraph representing signed digraph model

    Returns:
        pd.DataFrame: Conditional stability metrics and model class

    References:
        - Dambacher, J.M., Luh, H.-K., Li, H.W., Rossignol, P.A. (2003). Qualitative stability and ambiguity in model ecosystems. The American Naturalist 161, 876–888.

    Examples:
        ```python
        from qmm import load_digraph, conditional_stability
        conditional_stability(load_digraph("snowshoe"))
        #                       Test                                                 Definition   Result
        # 0        Weighted feedback                        Maximum weighted feedback (level 3)    -0.33
        # 1     Weighted determinant                          n-1 weighted determinant at level     0.56
        # 2  Ratio to model-c system                           Ratio to a 'model-c' type system      1.7
        # 3              Model class  Class of the model based on conditional stability metrics  Class I
        ```
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

def simulation_stability(
    G: nx.DiGraph,
    n_sim: int = 10000,
    distribution: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    presample: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Analyse stability using randomly sampled interaction strengths from a specified distribution.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        n_sim: Number of simulations to perform (default 10000)
        distribution: Distribution to sample from (default 'uniform'):
            - "uniform": Uniform(0, 1) - no assumption about interaction strength
            - "weak": Beta(1, 3) - weak interactions predominate
            - "moderate": Beta(2, 2) - moderate interactions predominate
            - "strong": Beta(3, 1) - strong interactions predominate
            - "uniform_two_oom": Uniform(0.01, 1)
        presample: Optional pre-sampled values of shape (n_sim, n, n) to use instead of random sampling

    Returns:
        pd.DataFrame: Proportion of stable matrices and proportion that fail Hurwitz criteria

    References:
        - May, R.M. (1973). Qualitative Stability in Model Ecosystems. Ecology 54, 638–641.
        - Dambacher, J.M., Luh, H.-K., Li, H.W., Rossignol, P.A. (2003). Qualitative stability and ambiguity in model ecosystems. The American Naturalist 161, 876–888.

    Examples:
        ```python
        from qmm import load_digraph, simulation_stability
        simulation_stability(load_digraph("snowshoe"), n_sim=1000)
        #                         Test                                                             Definition  Result
        # 0            Stable matrices              Proportion where all eigenvalues have negative real parts  79.10%
        # 1          Unstable matrices      Proportion where one or more eigenvalues have positive real parts  20.90%
        # 2        Hurwitz criterion i  Proportion where polynomial coefficients are not all of the same sign  20.90%
        # 3       Hurwitz criterion ii             Proportion where Hurwitz determinants are not all positive   0.00%
        # 4   Hurwitz criterion i only                        Proportion where only Hurwitz criterion i fails  20.90%
        # 5  Hurwitz criterion ii only                       Proportion where only Hurwitz criterion ii fails   0.00%
        ```
    """
    A = create_matrix(G, "signed")
    A = sp.matrix2numpy(A).astype(int)

    if presample is not None:
        if presample.shape != (n_sim, *A.shape):
            raise ValueError(f"presample must have shape ({n_sim}, {A.shape[0]}, {A.shape[1]})")

    n_stable = 0
    n_unstable = 0
    n_hurwitz_i_fail = 0
    n_hurwitz_ii_fail = 0
    n_hurwitz_i_only_fail = 0
    n_hurwitz_ii_only_fail = 0

    for i in range(n_sim):
        if presample is not None:
            M = presample[i]
        else:
            M = _random_sampler(distribution, A.size).reshape(A.shape)
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
