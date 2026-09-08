"""Press-response checks for the omnivory model."""

import networkx as nx
import sympy as sp

from qmm.core.press import (
    adjoint_matrix,
    absolute_feedback_matrix,
    weighted_predictions_matrix,
)


def test_adjoint_matrix_form_signed_omnivory(omnivory):
    result = adjoint_matrix(omnivory, form='signed')
    expected = sp.Matrix([
        [ 1, -1,  1],
        [-1,  1, -2],
        [ 1,  0,  1]])
    assert result == expected


def test_adjoint_matrix_form_symbolic_perturb_2_omnivory(omnivory):
    result = adjoint_matrix(omnivory, form='symbolic', perturb='2')
    a_11 = sp.Symbol('a_1,1')
    a_12 = sp.Symbol('a_1,2')
    a_13 = sp.Symbol('a_1,3')
    a_31 = sp.Symbol('a_3,1')
    a_32 = sp.Symbol('a_3,2')
    expected = sp.Matrix([
        [-a_13*a_32],
        [a_31*a_13],
        [a_11*a_32 - a_12*a_31]])
    assert result == expected


def test_absolute_feedback_matrix_default_omnivory(omnivory):
    result = absolute_feedback_matrix(omnivory)
    expected = sp.Matrix([
        [1, 1, 1],
        [1, 1, 2],
        [1, 2, 1]])
    assert result == expected


def test_absolute_feedback_matrix_perturb_2_omnivory(omnivory):
    result = absolute_feedback_matrix(omnivory, perturb='2')
    expected = sp.Matrix([[1], [1], [2]])
    assert result == expected


def test_weighted_predictions_matrix_as_abs_true_omnivory(omnivory):
    result = weighted_predictions_matrix(omnivory, as_abs=True)
    expected = sp.Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 1]])
    assert result == expected


def test_press_results_follow_node_order_omnivory(omnivory):
    G = nx.DiGraph()
    G.add_nodes_from((node, omnivory.nodes[node]) for node in ['3', '1', '2'])
    G.add_edges_from(omnivory.edges(data=True))
    result = (
        adjoint_matrix(G, form='signed'),
        absolute_feedback_matrix(G),
        weighted_predictions_matrix(G, as_abs=True),
    )
    expected = (
        sp.Matrix([
            [ 1,  1,  0],
            [ 1,  1, -1],
            [-2, -1,  1]]),
        sp.Matrix([
            [1, 1, 2],
            [1, 1, 1],
            [2, 1, 1]]),
        sp.Matrix([
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1]]),
    )
    assert result == expected
