import pytest
import sympy as sp
import networkx as nx
import pandas as pd
from qmm import *

# Test file
test_json_path = "test/mesocosm.json"

@pytest.fixture
def test_graph():
    return import_digraph(test_json_path)

def test_import_digraph(test_graph):
    assert isinstance(test_graph, nx.DiGraph)
    assert len(test_graph.nodes) == 8
    assert len(test_graph.edges) == 23

def test_create_matrix(test_graph):
    matrix = create_matrix(test_graph, form='signed')
    assert isinstance(matrix, sp.Matrix)
    assert matrix.shape == (8, 8)

def test_sign_stability(test_graph):
    result = sign_stability(test_graph)
    assert isinstance(result, pd.DataFrame)
    assert 'Test' in result.columns
    assert 'Result' in result.columns

def test_feedback_metrics(test_graph):
    result = feedback_metrics(test_graph)
    assert isinstance(result, pd.DataFrame)
    assert 'Feedback level' in result.columns
    assert 'Net' in result.columns

def test_determinants_metrics(test_graph):
    result = determinants_metrics(test_graph)
    assert isinstance(result, pd.DataFrame)
    assert 'Hurwitz determinant' in result.columns
    assert 'Net' in result.columns

def test_conditional_stability(test_graph):
    result = conditional_stability(test_graph)
    assert isinstance(result, pd.DataFrame)
    assert 'Test' in result.columns
    assert 'Result' in result.columns

def test_simulation_stability(test_graph):
    result = simulation_stability(test_graph, n_sim=100)
    assert isinstance(result, pd.DataFrame)
    assert 'Test' in result.columns
    assert 'Result' in result.columns

def test_adjoint_matrix(test_graph):
    result = adjoint_matrix(test_graph, form='signed')
    assert isinstance(result, sp.Matrix)
    assert result.shape == (8, 8)

def test_absolute_feedback_matrix(test_graph):
    result = absolute_feedback_matrix(test_graph)
    assert isinstance(result, sp.Matrix)
    assert result.shape == (8, 8)

def test_weighted_predictions_matrix(test_graph):
    result = weighted_predictions_matrix(test_graph, as_abs=True)
    assert isinstance(result, sp.Matrix)
    assert result.shape == (8, 8)

def test_sign_determinacy_matrix(test_graph):
    result = sign_determinacy_matrix(test_graph, method='average', as_abs=True)
    assert isinstance(result, sp.Matrix)
    assert result.shape == (8, 8)

def test_numerical_simulations(test_graph):
    result = numerical_simulations(test_graph, n_sim=100, dist="uniform", as_abs=True)
    assert isinstance(result, sp.Matrix)
    assert result.shape == (8, 8)

def test_table_of_predictions(test_graph):
    matrix = weighted_predictions_matrix(test_graph)
    result = table_of_predictions(matrix, t1=0.5, t2=1)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (8, 8)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])