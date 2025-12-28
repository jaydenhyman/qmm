"""Generate qualitative predictions of system response to press perturbations with thresholds for ambiguity."""

import numpy as np
import sympy as sp
import pandas as pd
import networkx as nx
from typing import Union, List, Optional, Callable
from .helper import get_nodes
from .press import (
    numerical_simulations,
    weighted_predictions_matrix,
    sign_determinacy_matrix,
)

def _apply_thresholds(
    M: Union[sp.Matrix, np.ndarray],
    t1: float,
    t2: float,
) -> list[list[str]]:
    if not (0.5 <= t1 <= 1):
        raise ValueError("t1 must be between 0.5 and 1")
    if not (0.5 <= t2 <= 1):
        raise ValueError("t2 must be between 0.5 and 1")
    if t1 > t2:
        raise ValueError("t1 must be less than or equal to t2")
    if isinstance(M, sp.Matrix):
        M = sp.matrix2numpy(M, dtype=float)

    def label(value: float) -> str:
        try:
            is_nan = bool(np.isnan(value))
        except TypeError:
            is_nan = (value == sp.nan)
        if is_nan:
            return "0"
        if value >= t1:
            return "+" if value >= t2 else "(+)"
        if value <= -t1:
            return "−" if value <= -t2 else "(−)"
        return "?"

    rows, cols = M.shape
    return [[label(M[i, j]) for j in range(cols)] for i in range(rows)]

def qualitative_predictions(
    G: nx.DiGraph,
    generator: Callable[..., Union[sp.Matrix, np.ndarray]] = numerical_simulations,
    t1: float = 0.8,
    t2: float = 0.95,
) -> sp.Matrix:
    """Create a sympy matrix of qualitative predictions with thresholds for ambiguity.

    Args:
        G (nx.DiGraph): Graph input for the matrix generator
        generator (Callable): Matrix generator function from core.press module
        t1 (float): Lower threshold for predictions
        t2 (float): Higher threshold for predictions

    Returns:
        sp.Matrix: Qualitative predictions

    Raises:
        ValueError: If generator is not callable or thresholds are invalid

    References:
        - Dambacher, J.M., Li, H.W., Rossignol, P.A. (2002). Relevance of Community Structure in Assessing Indeterminacy of Ecological Predictions. Ecology 83, 1372–1385.
        - Dambacher, J.M., Li, H.W., Rossignol, P.A. (2003). Qualitative predictions in model ecosystems. Ecological Modelling 161, 79–93.
        - Hosack, G.R., Hayes, K.R., Dambacher, J.M. (2008). Assessing Model Structure Uncertainty Through an Analysis of System Feedback and Bayesian Networks. Ecological Applications 18, 1070–1082.

    Examples:
        ```python
        from qmm import load_digraph, qualitative_predictions, weighted_predictions_matrix
        qualitative_predictions(load_digraph("snowshoe"), generator=weighted_predictions_matrix, t1=0.5, t2=1.0)
        # Matrix([
        # [+, −, +],
        # [?, +, −],
        # [+, ?, +]])
        ```
    """
    if not callable(generator):
        raise ValueError(f"Generator must be callable, got: {type(generator)}")

    matrix = generator(G)
    predictions = _apply_thresholds(matrix, t1, t2)
    rows = [[sp.Integer(0) if vals == "0" else sp.Symbol(vals) for vals in row] for row in predictions]
    return sp.Matrix(rows)


def matrix_to_predictions(
    M: Union[sp.Matrix, np.ndarray],
    t1: float = 0.8,
    t2: float = 0.95,
    index: Optional[List[str]] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create a table of qualitative predictions with thresholds for ambiguity.

    Args:
        M (Union[sp.Matrix, np.ndarray]): Matrix of predictions from press perturbation analysis
        t1 (float): Lower threshold for predictions
        t2 (float): Higher threshold for predictions
        index (Optional[List[str]]): Row labels
        columns (Optional[List[str]]): Column labels

    Returns:
        pd.DataFrame: Qualitative predictions

    References:
        - Dambacher, J.M., Li, H.W., Rossignol, P.A. (2002). Relevance of Community Structure in Assessing Indeterminacy of Ecological Predictions. Ecology 83, 1372–1385.
        - Dambacher, J.M., Li, H.W., Rossignol, P.A. (2003). Qualitative predictions in model ecosystems. Ecological Modelling 161, 79–93.
        - Hosack, G.R., Hayes, K.R., Dambacher, J.M. (2008). Assessing Model Structure Uncertainty Through an Analysis of System Feedback and Bayesian Networks. Ecological Applications 18, 1070–1082.

    Examples:
        ```python
        from qmm import load_digraph, weighted_predictions_matrix, matrix_to_predictions
        W = weighted_predictions_matrix(load_digraph("snowshoe"))
        nodes = ['V', 'H', 'P']
        matrix_to_predictions(W, t1=0.5, t2=1.0, index=nodes, columns=nodes)
        #    V  H  P
        # V  +  −  +
        # H  ?  +  −
        # P  +  ?  +
        ```
    """
    if isinstance(M, sp.Matrix) and any(isinstance(v, (sp.Symbol, str)) for v in M):
        return pd.DataFrame(M.tolist(), index=index, columns=columns)
    predictions = _apply_thresholds(M, t1, t2)
    return pd.DataFrame(predictions, index=index, columns=columns)

def compare_predictions(M1: pd.DataFrame, M2: pd.DataFrame) -> pd.DataFrame:
    """Compare predictions between alternative models or prediction methods.

    Args:
        M1 (pd.DataFrame): First matrix of predictions
        M2 (pd.DataFrame): Second matrix of predictions

    Returns:
        pd.DataFrame: Combined predictions showing differences and agreements

    Examples:
        ```python
        from qmm import load_digraph, weighted_predictions_matrix, matrix_to_predictions, compare_predictions
        G1 = load_digraph("snowshoe")
        G2 = G1.copy()
        G2.remove_edge('V', 'P')
        nodes = ['V', 'H', 'P']
        W1 = weighted_predictions_matrix(G1)
        W2 = weighted_predictions_matrix(G2)
        pred1 = matrix_to_predictions(W1, t1=0.5, t2=1.0, index=nodes, columns=nodes)
        pred2 = matrix_to_predictions(W2, t1=0.5, t2=1.0, index=nodes, columns=nodes)
        compare_predictions(pred1, pred2)
        #       V     H  P
        # V     +     −  +
        # H  ?, +     +  −
        # P     +  ?, +  +
        ```
    """
    if not M1.index.equals(M2.index) or not M1.columns.equals(M2.columns):
        raise ValueError("M1 and M2 must have the same index and columns")
    M1_str = M1.astype(str)
    M2_str = M2.astype(str)
    combined = pd.DataFrame(
        index=M1.index,
        columns=M1.columns,
        data=np.where(M1_str.values == M2_str.values, M1_str.values, M1_str.values + ", " + M2_str.values),
    )
    return combined
