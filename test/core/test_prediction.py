"""Tests for qmm.core.prediction module."""

import pytest
import pandas as pd
import pandas.testing as pdt
import numpy as np
import sympy as sp

from qmm import (
    get_nodes,
    weighted_predictions_matrix,
    table_of_predictions,
    matrix_to_predictions,
    qualitative_predictions,
    compare_predictions,
    numerical_simulations,
    simulation_effects,
)


# =============================================================================
# table_of_predictions()
# =============================================================================

def test_table_of_predictions_default_thresholds_snowshoe(snowshoe):
    """Test table_of_predictions with weighted predictions on the snowshoe model."""
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
    """Test table_of_predictions with weighted predictions on the chain model."""
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
    """Test table_of_predictions with weighted predictions on the snowshoe_na model."""
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
    """Test table_of_predictions with simulation_effects output for snowshoe IO NaN graph."""
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
    """Test table_of_predictions with numerical_simulations for the mesocosm model."""
    expected = pd.DataFrame(
        [
            ["+", "?", "?", "−", "?", "+", "?", "?"],
            ["?", "(+)", "(−)", "?", "(−)", "?", "(+)", "(−)"],
            ["?", "?", "(+)", "?", "(+)", "?", "?", "(+)"],
            ["+", "?", "?", "(+)", "?", "+", "?", "?"],
            ["?", "(+)", "(−)", "?", "?", "?", "?", "?"],
            ["+", "?", "(+)", "−", "?", "+", "?", "?"],
            ["?", "?", "(−)", "?", "?", "?", "(+)", "(−)"],
            ["?", "(+)", "(−)", "?", "?", "?", "(+)", "?"],
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
    """Test threshold labeling on boundary values."""
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
    """Test qualitative_predictions returns a symbol matrix."""
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
    """Test table_of_predictions returns symbol matrices unchanged."""
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
    """Test table_of_predictions handles non-numpy NaN checks."""
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
    """Test table_of_predictions keeps determinate results when thresholds repeat."""
    nodes = get_nodes(snowshoe, "state")
    wpm = weighted_predictions_matrix(snowshoe)
    base = matrix_to_predictions(wpm, t1=0.5, t2=1.0, index=nodes, columns=nodes)
    repeat = matrix_to_predictions(wpm, t1=0.5, t2=1.0, index=nodes, columns=nodes)
    pdt.assert_frame_equal(repeat, base)


# =============================================================================
# compare_predictions()
# =============================================================================

def test_compare_predictions_identical_tables_snowshoe(snowshoe):
    """Test compare_predictions returns identical table for snowshoe inputs."""
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
    """Test compare_predictions returns identical table for chain inputs."""
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
    """Test compare_predictions returns identical table for mesocosm inputs."""
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
    """Test compare_predictions highlights differing entries across mesocosm variants."""
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
    """Test compare_predictions reports comma-separated differences for snowshoe."""
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
    """Test compare_predictions raises ValueError when labels differ for snowshoe tables."""
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


# =============================================================================
# Additional tests for error handling and coverage
# =============================================================================

def test_apply_thresholds_invalid_t1_too_low(snowshoe):
    """Test _apply_thresholds raises ValueError for t1 < 0."""
    with pytest.raises(ValueError, match="t1 must be between 0 and 1"):
        matrix_to_predictions(
            weighted_predictions_matrix(snowshoe),
            t1=-0.1,
            t2=0.95
        )

def test_apply_thresholds_invalid_t1_too_high(snowshoe):
    """Test _apply_thresholds raises ValueError for t1 > 1."""
    with pytest.raises(ValueError, match="t1 must be between 0 and 1"):
        matrix_to_predictions(
            weighted_predictions_matrix(snowshoe),
            t1=1.2,
            t2=0.95
        )

def test_apply_thresholds_invalid_t2_too_low(snowshoe):
    """Test _apply_thresholds raises ValueError for t2 < 0."""
    with pytest.raises(ValueError, match="t2 must be between 0 and 1"):
        matrix_to_predictions(
            weighted_predictions_matrix(snowshoe),
            t1=0.8,
            t2=-0.1
        )

def test_apply_thresholds_invalid_t2_too_high(snowshoe):
    """Test _apply_thresholds raises ValueError for t2 > 1."""
    with pytest.raises(ValueError, match="t2 must be between 0 and 1"):
        matrix_to_predictions(
            weighted_predictions_matrix(snowshoe),
            t1=0.8,
            t2=1.2
        )

def test_apply_thresholds_t1_greater_than_t2(snowshoe):
    """Test _apply_thresholds raises ValueError when t1 > t2."""
    with pytest.raises(ValueError, match="t1 must be less than or equal to t2"):
        matrix_to_predictions(
            weighted_predictions_matrix(snowshoe),
            t1=0.95,
            t2=0.8
        )

def test_qualitative_predictions_non_callable_generator(snowshoe):
    """Test qualitative_predictions raises ValueError for non-callable generator."""
    with pytest.raises(ValueError, match="Generator must be callable"):
        qualitative_predictions(snowshoe, generator="invalid_string")

def test_table_of_predictions_with_string_generator(snowshoe):
    """Test table_of_predictions with string generator name."""
    result = table_of_predictions(snowshoe, generator=weighted_predictions_matrix, t1=0.5, t2=1.0)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)

def test_table_of_predictions_with_effects_generator(snowshoe_io):
    """Test table_of_predictions with effects generator producing MultiIndex."""
    from qmm import weighted_effects
    result = table_of_predictions(snowshoe_io, generator=weighted_effects, t1=0.8, t2=1.0)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.columns, pd.MultiIndex)
    assert isinstance(result.index, pd.MultiIndex)
