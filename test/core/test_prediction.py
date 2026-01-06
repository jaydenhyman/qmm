"""Tests for qmm.core.prediction module."""

import pytest
import pandas as pd
import pandas.testing as pdt
import numpy as np
import sympy as sp

from qmm.core.helper import get_nodes
from qmm.core.press import weighted_predictions_matrix, numerical_simulations
from qmm.core.prediction import (
    matrix_to_predictions,
    qualitative_predictions,
    compare_predictions,
)
from qmm.extensions.effects import simulation_effects


# =============================================================================
# table_of_predictions()
# =============================================================================

def test_table_of_predictions_default_thresholds_snowshoe(snowshoe):
    expected = pd.DataFrame(
        [["+", "−", "+"], ["+", "+", "−"], ["+", "+", "+"]],
        index=["R", "C", "P"],
        columns=["R", "C", "P"],
    )
    nodes = get_nodes(snowshoe, "state")
    result = matrix_to_predictions(
        weighted_predictions_matrix(snowshoe),
        t1=0.5,
        t2=1.0,
        index=nodes,
        columns=nodes,
    )
    pdt.assert_frame_equal(result, expected)


def test_table_of_predictions_default_thresholds_chain(chain):
    expected = pd.DataFrame(
        [
            ["+", "−", "+", "−", "+"],
            ["+", "+", "−", "+", "−"],
            ["+", "+", "+", "−", "+"],
            ["+", "+", "+", "+", "−"],
            ["+", "+", "+", "+", "+"],
        ],
        index=["1", "2", "3", "4", "5"],
        columns=["1", "2", "3", "4", "5"],
    )
    nodes = get_nodes(chain, "state")
    result = matrix_to_predictions(
        weighted_predictions_matrix(chain),
        t1=0.5,
        t2=1.0,
        index=nodes,
        columns=nodes,
    )
    pdt.assert_frame_equal(result, expected)


def test_table_of_predictions_default_thresholds_snowshoe_na(snowshoe_na):
    expected = pd.DataFrame(
        [["+", "−", "?"], ["0", "+", "−"], ["0", "0", "+"]],
        index=["1", "2", "3"],
        columns=["1", "2", "3"],
    )
    nodes = get_nodes(snowshoe_na, "state")
    result = matrix_to_predictions(
        weighted_predictions_matrix(snowshoe_na),
        t1=0.5,
        t2=1.0,
        index=nodes,
        columns=nodes,
    )
    pdt.assert_frame_equal(result, expected)


def test_table_of_predictions_simulation_effects_snowshoe_io_na(snowshoe_io_na):
    expected = pd.DataFrame(
        [
            ["+", "−", "+", "+", "+", "−"],
            ["+", "+", "−", "+", "?", "+"],
            ["+", "+", "+", "+", "?", "−"],
            ["0", "0", "0", "+", "0", "0"],
            ["?", "?", "+", "?", "?", "−"],
            ["+", "+", "−", "+", "?", "+"],
        ],
        index=["R", "C", "P", "N", "Out1", "Out2"],
        columns=["R", "C", "P", "N", "Inp1", "Inp2"],
    )
    rows = get_nodes(snowshoe_io_na, "state") + get_nodes(snowshoe_io_na, "output")
    cols = get_nodes(snowshoe_io_na, "state") + get_nodes(snowshoe_io_na, "input")
    result = matrix_to_predictions(
        simulation_effects(snowshoe_io_na, n_sim=500, seed=42),
        t1=0.8,
        t2=1.0,
        index=rows,
        columns=cols,
    )
    pdt.assert_frame_equal(result, expected)


def test_table_of_predictions_default_thresholds_mesocosm(mesocosm):
    expected = pd.DataFrame(
        [
            ["+",   "?",   "?",   "−",   "?",   "+",   "?",   "?"],
            ["?", "(+)", "(−)",   "?", "(−)",   "?", "(+)", "(−)"],
            ["?",   "?", "(+)",   "?", "(+)",   "?",   "?", "(+)"],
            ["+",   "?",   "?", "(+)",   "?",   "+",   "?",   "?"],
            ["?", "(+)", "(−)",   "?",   "?",   "?",   "?",   "?"],
            ["+",   "?", "(+)",   "−",   "?",   "+",   "?",   "?"],
            ["?",   "?", "(−)",   "?",   "?",   "?", "(+)", "(−)"],
            ["?", "(+)", "(−)",   "?",   "?",   "?", "(+)",   "?"],
        ],
        index=["P", "A1", "A2", "AP", "H1", "H2", "C1", "C2"],
        columns=["P", "A1", "A2", "AP", "H1", "H2", "C1", "C2"],
    )
    nodes = get_nodes(mesocosm, "state")
    result = matrix_to_predictions(
        numerical_simulations(mesocosm, n_sim=500, seed=42),
        t1=0.8,
        t2=1.0,
        index=nodes,
        columns=nodes,
    )
    pdt.assert_frame_equal(result, expected)

def test_table_of_predictions_threshold_boundaries(snowshoe):
    values = np.array([[1.0, 0.95, 0.8, 0.0, -0.8, -0.95, -1.0, np.nan]])
    expected = pd.DataFrame(
        [["+", "+", "(+)", "?", "(−)", "−", "−", "0"]],
        index=["row1"],
        columns=["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"],
    )
    result = matrix_to_predictions(
        values,
        t1=0.8,
        t2=0.95,
        index=expected.index.tolist(),
        columns=expected.columns.tolist(),
    )
    pdt.assert_frame_equal(result, expected)

def test_qualitative_predictions_matrix_output(snowshoe):
    matrix = np.array([[1.0, 0.95, 0.9, 0.8, 0.0, -0.8, -0.9, -0.95, -1.0, np.nan]])
    expected = sp.Matrix([[
        sp.Symbol("+"),
        sp.Symbol("+"),
        sp.Symbol("(+)"),
        sp.Symbol("(+)"),
        sp.Symbol("?"),
        sp.Symbol("(−)"),
        sp.Symbol("(−)"),
        sp.Symbol("−"),
        sp.Symbol("−"),
        sp.Integer(0),
    ]])
    result = qualitative_predictions(
        snowshoe,
        lambda G: matrix,
        t1=0.8,
        t2=0.95,
    )
    assert result == expected

def test_table_of_predictions_symbolic_matrix(snowshoe):
    a, b, c, d = sp.Symbol("a"), sp.Symbol("b"), sp.Symbol("c"), sp.Symbol("d")
    matrix = sp.Matrix([[a, b], [c, d]])
    expected = pd.DataFrame([[a, b], [c, d]], index=["r1", "r2"], columns=["c1", "c2"])
    result = matrix_to_predictions(
        matrix,
        t1=0.8,
        t2=1.0,
        index=["r1", "r2"],
        columns=["c1", "c2"],
    )
    pdt.assert_frame_equal(result, expected)

def test_table_of_predictions_typeerror_nan_check(snowshoe):
    class NanLike:
        def __eq__(self, other):
            return other == sp.nan

    matrix = np.array([[NanLike()]], dtype=object)
    expected = pd.DataFrame([["0"]], index=["r1"], columns=["c1"])
    result = matrix_to_predictions(
        matrix,
        t1=0.8,
        t2=1.0,
        index=["r1"],
        columns=["c1"],
    )
    pdt.assert_frame_equal(result, expected)


def test_table_of_predictions_threshold_consistency_snowshoe(snowshoe):
    nodes = get_nodes(snowshoe, "state")
    wpm = weighted_predictions_matrix(snowshoe)
    base = matrix_to_predictions(wpm, t1=0.5, t2=1.0, index=nodes, columns=nodes)
    repeat = matrix_to_predictions(wpm, t1=0.5, t2=1.0, index=nodes, columns=nodes)
    pdt.assert_frame_equal(repeat, base)


# =============================================================================
# compare_predictions()
# =============================================================================

def test_compare_predictions_identical_tables_snowshoe(snowshoe):
    nodes = get_nodes(snowshoe, "state")
    preds = matrix_to_predictions(
        weighted_predictions_matrix(snowshoe),
        t1=0.5,
        t2=1.0,
        index=nodes,
        columns=nodes,
    )
    result = compare_predictions(preds, preds.copy())
    pdt.assert_frame_equal(result, preds)


def test_compare_predictions_identical_tables_chain(chain):
    nodes = get_nodes(chain, "state")
    preds = matrix_to_predictions(
        weighted_predictions_matrix(chain),
        t1=0.5,
        t2=1.0,
        index=nodes,
        columns=nodes,
    )
    result = compare_predictions(preds, preds.copy())
    pdt.assert_frame_equal(result, preds)


def test_compare_predictions_identical_tables_mesocosm(mesocosm):
    nodes = get_nodes(mesocosm, "state")
    preds = matrix_to_predictions(
        numerical_simulations(mesocosm, n_sim=500, seed=42),
        t1=0.8,
        t2=1.0,
        index=nodes,
        columns=nodes,
    )
    result = compare_predictions(preds, preds.copy())
    pdt.assert_frame_equal(result, preds)


def test_compare_predictions_variant_mismatch_mesocosm_alt_models(mesocosm_alt_models):
    base, alt = mesocosm_alt_models
    nodes = get_nodes(base, "state")
    base_preds = matrix_to_predictions(
        numerical_simulations(base, n_sim=500, seed=42),
        t1=0.8,
        t2=1.0,
        index=nodes,
        columns=nodes,
    )
    alt_preds = matrix_to_predictions(
        numerical_simulations(alt, n_sim=500, seed=42),
        t1=0.8,
        t2=1.0,
        index=nodes,
        columns=nodes,
    )
    result = compare_predictions(base_preds, alt_preds)
    assert result.loc["P", "P"] == "+"
    assert result.loc["P", "C2"] == "?, (−)"
    assert result.loc["A1", "A1"] == "(+), +"
    assert result.loc["C2", "P"] == "?, (+)"


def test_compare_predictions_single_difference_snowshoe(snowshoe):
    nodes = get_nodes(snowshoe, "state")
    preds = matrix_to_predictions(
        weighted_predictions_matrix(snowshoe),
        t1=0.5,
        t2=1.0,
        index=nodes,
        columns=nodes,
    )
    modified = preds.copy()
    modified.loc["R", "R"] = "−"
    result = compare_predictions(preds, modified)
    assert result.loc["R", "R"] == "+, −"
    assert result.loc["C", "R"] == preds.loc["C", "R"]


def test_compare_predictions_label_mismatch_snowshoe(snowshoe):
    nodes = get_nodes(snowshoe, "state")
    preds = matrix_to_predictions(
        weighted_predictions_matrix(snowshoe),
        t1=0.5,
        t2=1.0,
        index=nodes,
        columns=nodes,
    )
    shuffled = matrix_to_predictions(
        weighted_predictions_matrix(snowshoe),
        t1=0.5,
        t2=1.0,
        index=nodes[::-1],
        columns=nodes,
    )
    with pytest.raises(ValueError):
        compare_predictions(preds, shuffled)


def test_compare_predictions_mismatched_columns():
    df1 = pd.DataFrame({'A': ['+', '-'], 'B': ['+', '+']}, index=['X', 'Y'])
    df2 = pd.DataFrame({'A': ['+', '-'], 'C': ['+', '+']}, index=['X', 'Y'])
    with pytest.raises(ValueError, match="same index and columns"):
        compare_predictions(df1, df2)


def test_compare_predictions_mismatched_index():
    df1 = pd.DataFrame({'A': ['+', '-'], 'B': ['+', '+']}, index=['X', 'Y'])
    df2 = pd.DataFrame({'A': ['+', '-'], 'B': ['+', '+']}, index=['X', 'Z'])
    with pytest.raises(ValueError, match="same index and columns"):
        compare_predictions(df1, df2)


def test_compare_predictions_different_sizes():
    df1 = pd.DataFrame({'A': ['+', '-', '?']}, index=['X', 'Y', 'Z'])
    df2 = pd.DataFrame({'A': ['+', '-']}, index=['X', 'Y'])
    with pytest.raises(ValueError, match="same index and columns"):
        compare_predictions(df1, df2)


def test_compare_predictions_single_element():
    df1 = pd.DataFrame({'A': ['+']}, index=['X'])
    df2 = pd.DataFrame({'A': ['-']}, index=['X'])
    result = compare_predictions(df1, df2)
    assert result.loc['X', 'A'] == '+, -'


def test_compare_predictions_all_different():
    df1 = pd.DataFrame({'A': ['+', '+'], 'B': ['+', '+']}, index=['X', 'Y'])
    df2 = pd.DataFrame({'A': ['-', '-'], 'B': ['-', '-']}, index=['X', 'Y'])
    result = compare_predictions(df1, df2)
    assert result.loc['X', 'A'] == '+, -'
    assert result.loc['Y', 'B'] == '+, -'


def test_compare_predictions_numeric_values():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['X', 'Y'])
    df2 = pd.DataFrame({'A': [1, 5], 'B': [3, 6]}, index=['X', 'Y'])
    result = compare_predictions(df1, df2)
    assert result.loc['X', 'A'] == '1'
    assert result.loc['Y', 'A'] == '2, 5'


# =============================================================================
# Additional tests for error handling and coverage
# =============================================================================

def test_apply_thresholds_invalid_t1_too_low(snowshoe):
    with pytest.raises(ValueError, match="t1 must be between 0 and 1"):
        matrix_to_predictions(
            weighted_predictions_matrix(snowshoe),
            t1=-0.1,
            t2=0.95
        )

def test_apply_thresholds_invalid_t1_too_high(snowshoe):
    with pytest.raises(ValueError, match="t1 must be between 0 and 1"):
        matrix_to_predictions(
            weighted_predictions_matrix(snowshoe),
            t1=1.2,
            t2=0.95
        )

def test_apply_thresholds_invalid_t2_too_low(snowshoe):
    with pytest.raises(ValueError, match="t2 must be between 0 and 1"):
        matrix_to_predictions(
            weighted_predictions_matrix(snowshoe),
            t1=0.8,
            t2=-0.1
        )

def test_apply_thresholds_invalid_t2_too_high(snowshoe):
    with pytest.raises(ValueError, match="t2 must be between 0 and 1"):
        matrix_to_predictions(
            weighted_predictions_matrix(snowshoe),
            t1=0.8,
            t2=1.2
        )

def test_apply_thresholds_t1_greater_than_t2(snowshoe):
    with pytest.raises(ValueError, match="t1 must be less than or equal to t2"):
        matrix_to_predictions(
            weighted_predictions_matrix(snowshoe),
            t1=0.95,
            t2=0.8
        )

def test_matrix_to_predictions_dataframe_input(snowshoe):
    nodes = get_nodes(snowshoe, "state")
    M_matrix = weighted_predictions_matrix(snowshoe)
    M_df = pd.DataFrame(M_matrix.tolist(), index=nodes, columns=nodes)
    result = matrix_to_predictions(M_df, t1=0.5, t2=1.0, index=nodes, columns=nodes)
    expected = matrix_to_predictions(M_matrix, t1=0.5, t2=1.0, index=nodes, columns=nodes)
    pdt.assert_frame_equal(result, expected)


def test_qualitative_predictions_non_callable_generator(snowshoe):
    with pytest.raises(ValueError, match="Generator must be callable"):
        qualitative_predictions(snowshoe, generator="invalid_string")
