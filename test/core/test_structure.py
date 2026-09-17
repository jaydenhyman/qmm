"""Tests for qmm.core.structure module."""

import json
import warnings
from pathlib import Path
from unittest.mock import patch
import networkx as nx
import pytest
import sympy as sp
from qmm.core.helper import get_nodes
from qmm.core.press import adjoint_matrix
from qmm.extensions.effects import define_input_output as from_effects
from qmm.extensions.life import life_expectancy_change

from qmm.core.structure import (
    import_digraph,
    define_input_output,
    create_matrix,
    create_equations,
    nodes_table,
    edges_table,
)


# =============================================================================
# import_digraph()
# =============================================================================

@pytest.mark.parametrize("attributes", [
    {}, {"sign": 0}, {"sign": True}, {"sign": "-1"}, {"sign": None},
    {"arrows": None}, {"arrows": "to"}, {"arrows": {}},
    {"arrows": {"to": "to"}, "sign": -1},
    {"arrows": {"to": {"type": "box"}}, "sign": -1},
    {"arrows": {"to": {"type": []}}},
])
def test_import_digraph_rejects_invalid_signs_and_arrows(attributes):
    data = {
        "nodes": [{"id": "A"}],
        "edges": [{"from": "A", "to": "A", **attributes}],
    }
    with pytest.raises(ValueError, match="A.*A"):
        import_digraph(data)


@pytest.mark.parametrize("reverse", [False, True])
def test_import_digraph_derives_roles(reverse):
    data = {
        "nodes": [{"id": n, "category": "output", "label": "Same"}
                  for n in ("I", "S", "O")],
        "edges": [{"from": a, "to": b, "sign": -1}
                  for a, b in [("I", "S"), ("S", "S"), ("S", "O")]],
    }
    if reverse:
        data["nodes"].reverse()
        data["edges"].reverse()
    graph = import_digraph(data)
    result = nx.get_node_attributes(graph, "category")
    expected = {"I": "input", "S": "state", "O": "output"}
    assert result == expected
    assert nx.is_frozen(graph)
    assert all(node["category"] == "output" for node in data["nodes"])


def test_import_digraph_rejects_disconnected_model():
    data = {
        "nodes": [{"id": n} for n in ("A", "B", "Z")],
        "edges": [{"from": a, "to": b, "sign": -1}
                  for a, b in [("A", "A"), ("A", "B"), ("B", "A"), ("Z", "Z")]],
    }
    expected = r"Disconnected model: \['A', 'B'\]; \['Z'\]"
    with pytest.raises(ValueError, match=expected):
        import_digraph(data)


@pytest.mark.parametrize("first, second", [("B", "B"), ("A", "A"), (1, "1")])
def test_import_digraph_rejects_duplicate_edges(first, second):
    data = {
        "nodes": [{"id": "A"}, {"id": "B"}, {"id": "1"}],
        "edges": [{"from": "A", "to": target, "sign": 1} for target in (first, second)],
    }
    expected = f"Duplicate edge: A -> {second}"
    with pytest.raises(ValueError, match=expected):
        import_digraph(data)


@pytest.mark.parametrize("data, expected", [
    ({"nodes": [{"id": "A"}]}, "Model needs nodes and edges"),
    ({"edges": []}, "Model needs nodes and edges"),
    ({"nodes": [{"label": "A"}], "edges": []}, "Node needs id: {'label': 'A'}"),
    ({"nodes": [{"id": "A"}], "edges": [{"to": "A", "sign": 1}]}, "Edge needs from and to: {'to': 'A', 'sign': 1}"),
    ({"nodes": [{"id": "A"}], "edges": [{"from": "A", "to": "B", "sign": 1}]}, "Unknown node: A -> B"),
    ({"nodes": [{"id": "A"}], "edges": [{"from": "B", "to": "A", "sign": 1}]}, "Unknown node: B -> A"),
])
def test_import_digraph_rejects_incomplete_models(data, expected):
    with pytest.raises(ValueError) as error:
        import_digraph(data)
    result = str(error.value)
    assert result == expected


@pytest.mark.parametrize("first, second", [("A", "A"), (1, "1")])
def test_import_digraph_rejects_duplicate_node_ids(first, second):
    data = {"nodes": [{"id": first}, {"id": second}], "edges": []}
    expected = f"Duplicate node: {second}"
    with pytest.raises(ValueError, match=expected):
        import_digraph(data)


@pytest.mark.parametrize("attributes, expected, corrected", [
    ({"sign": 1}, 1, False),
    ({"sign": -1.0}, -1, False),
    ({"arrows": {"to": {"type": "triangle"}}}, 1, False),
    ({"arrows": {"to": {"type": "circle"}}, "sign": -1}, -1, False),
    ({"arrows": {"to": {"type": "triangle"}}, "sign": -1}, 1, True),
    ({"arrows": {"to": {"type": "circle"}}, "sign": 1}, -1, True),
])
def test_import_digraph_from_dict(attributes, expected, corrected):
    data = {
        "nodes": [{"id": "A"}],
        "edges": [{"id": "e1", "from": "A", "to": "A", **attributes}],
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        G = import_digraph(data)
    result = create_matrix(G, form="signed")
    assert result == sp.Matrix([[expected]])
    assert create_matrix(G, form="symbolic") == sp.Matrix([[expected * sp.Symbol("a_A,A")]])
    assert G["A"]["A"] == {**attributes, "id": "e1", "sign": expected,
                            "dashes": False, "title": None}
    assert len(caught) == int(corrected)
    if corrected:
        assert "A -> A" in str(caught[0].message)


@pytest.mark.parametrize("path_type", [str, Path])
def test_import_digraph_from_file(tmp_path, path_type):
    data = {"nodes": [{"id": "X"}], "edges": [{"from": "X", "to": "X", "sign": -1}], "meta": {"title": "X"}}
    path = tmp_path / "model.json"
    path.write_text(json.dumps(data))
    result = import_digraph(path_type(path))
    expected = import_digraph(data)
    assert result.graph == expected.graph
    assert list(result.nodes(data=True)) == list(expected.nodes(data=True))
    assert list(result.edges(data=True)) == list(expected.edges(data=True))


def test_import_digraph_does_not_parse_json_text():
    data = {"nodes": [{"id": "X"}], "edges": [{"from": "X", "to": "X", "sign": -1}]}
    with pytest.raises(OSError):
        import_digraph(json.dumps(data))


def test_import_digraph_keeps_attributes():
    data = {
        "nodes": [{"id": "A", "label": "Node A"}, {"id": "B"}],
        "edges": [{"from": "A", "to": "A", "sign": -1}, {"from": "A", "to": "B", "sign": 1}],
        "meta": {"title": "Model", "description": "Context"},
        "references": ["Source"],
        "custom": {"units": "biomass"},
    }
    G = import_digraph(data)
    result = G.nodes['A']['label']
    expected = 'Node A'
    assert result == expected
    result = G.graph
    expected = {
        "meta": {"title": "Model", "description": "Context"},
        "references": ["Source"],
        "custom": {"units": "biomass"},
    }
    assert result == expected

# =============================================================================
# define_input_output()
# =============================================================================

def test_define_input_output_categories_snowshoe_io(snowshoe_io):
    categorized = define_input_output(snowshoe_io)
    result = {node: data['category'] for node, data in categorized.nodes(data=True)}
    expected = {
        'R': 'state',
        'C': 'state',
        'P': 'state',
        'Inp1': 'input',
        'Inp2': 'input',
        'Out1': 'output',
        'Out2': 'output',
    }
    assert result == expected


def test_define_input_output_rejects_invalid_nodes(disconnected_graph):
    disconnected_graph.add_edge('D', 'E', sign=1)
    disconnected_graph.add_node('F')
    with pytest.raises(ValueError, match=r"Invalid nodes: \['C', 'D', 'E', 'F'\]"):
        define_input_output(disconnected_graph)
    assert list(disconnected_graph) == ['A', 'B', 'C', 'D', 'E', 'F']


def test_define_input_output_equal_components_keep_state(snowshoe):
    G = nx.compose(nx.DiGraph(snowshoe), nx.relabel_nodes(nx.DiGraph(snowshoe), {'R': 'X', 'C': 'Y', 'P': 'Z'}))
    result = set(nx.get_node_attributes(define_input_output(G), "category").values())
    expected = {'state'}
    assert result == expected


def test_define_input_output_rejects_non_graph():
    with pytest.raises(TypeError) as exc_info:
        define_input_output("not a graph")
    result = str(exc_info.value)
    expected = "Input must be a networkx.DiGraph."
    assert result == expected


def test_define_input_output_cyclic_inputs_become_state(cyclic_inputs_graph):
    G = define_input_output(cyclic_inputs_graph)
    assert set(get_nodes(G, "state")) >= {"I1", "I2"}


def test_define_input_output_classifies_feedthrough(snowshoe_io_with_direct_edge):
    graph = define_input_output(snowshoe_io_with_direct_edge)
    result = nx.get_node_attributes(graph, "category")
    expected = {
        "R": "state", "C": "state", "P": "state",
        "Inp1": "input", "Inp2": "input", "Out1": "output", "Out2": "output",
    }
    assert result == expected
    with pytest.raises(ValueError, match="Direct input to output edge"):
        create_matrix(graph, matrix_type="D")


@pytest.mark.parametrize("nodes", ['A', 'AB', 'ABCD'])
def test_define_input_output_rejects_feedback_free_chain(nodes):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(zip(nodes, nodes[1:]), sign=1)
    with pytest.raises(ValueError) as error:
        define_input_output(graph)
    result = str(error.value)
    expected = f"Invalid nodes: {list(nodes)}"
    assert result == expected


def test_define_input_output_rejects_non_unit_signs():
    G = nx.DiGraph()
    G.add_edge('A', 'B', sign=0.5)
    with pytest.raises(ValueError, match="Edge signs must be"):
        define_input_output(G)


def test_define_input_output_overwrites_preset_categories():
    G = nx.DiGraph()
    G.add_edge('R', 'R', sign=-1)
    G.add_edge('Inp', 'R', sign=1)
    G.add_edge('R', 'Out', sign=1)
    G.nodes['Inp']['category'] = 'output'
    Gd = define_input_output(G)
    assert (Gd.nodes['Inp']['category'], Gd.nodes['R']['category'], Gd.nodes['Out']['category']) == ('input', 'state', 'output')


def test_define_input_output_importable():
    assert define_input_output is from_effects

# =============================================================================
# create_matrix()
# =============================================================================

@pytest.mark.parametrize("function", [create_matrix, adjoint_matrix, life_expectancy_change])
def test_create_matrix_rejects_invalid_form(snowshoe, function):
    with pytest.raises(ValueError, match="^Invalid form"):
        function(snowshoe, form="invalid")


def test_create_matrix_rejects_invalid_matrix_type(snowshoe):
    with pytest.raises(ValueError, match="^Invalid matrix type. Choose 'A', 'B', 'C', 'D'.$"):
        create_matrix(snowshoe, matrix_type="invalid")

def test_create_matrix_form_signed_snowshoe(snowshoe):
    A = create_matrix(snowshoe, form='signed')
    result = A
    expected = sp.Matrix([
        [-1, -1,  0],
        [ 1,  0, -1],
        [ 0,  1, -1]])
    assert result == expected


def test_create_matrix_form_symbolic_snowshoe(snowshoe):
    result = create_matrix(snowshoe, form='symbolic')
    a_RR = sp.Symbol('a_R,R')
    a_RC = sp.Symbol('a_R,C')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    expected = sp.Matrix([
        [-a_RR, -a_RC,     0],
        [ a_CR,     0, -a_CP],
        [    0,  a_PC, -a_PP]])
    assert result == expected


def test_create_matrix_form_binary_snowshoe(snowshoe):
    A = create_matrix(snowshoe, form='binary')
    result = A
    expected = sp.Matrix([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]])
    assert result == expected


def test_create_matrix_form_signed_matrix_B_snowshoe_io(snowshoe_io):
    with patch('qmm.core.structure.nx.all_simple_paths', wraps=nx.all_simple_paths) as paths:
        B = create_matrix(snowshoe_io, form='signed', matrix_type='B')
    for call in paths.call_args_list:
        graph, source, target = call.args
        assert all(node in (source, target) or data['category'] == 'input'
                   for node, data in graph.nodes(data=True))
    result = (B.shape, B)
    expected = ((3, 2), sp.Matrix([
        [1, 0],
        [-1, 0],
        [0, -1]
    ]))
    assert result == expected


def test_create_matrix_form_signed_matrix_C_snowshoe_io(snowshoe_io):
    with patch('qmm.core.structure.nx.all_simple_paths', wraps=nx.all_simple_paths) as paths:
        C = create_matrix(snowshoe_io, form='signed', matrix_type='C')
    for call in paths.call_args_list:
        graph, source, target = call.args
        assert all(node in (source, target) or data['category'] == 'output'
                   for node, data in graph.nodes(data=True))
    result = (C.shape, C)
    expected = ((2, 3), sp.Matrix([
        [0, -1, 1],
        [0, 1, 0]
    ]))
    assert result == expected


def test_create_matrix_form_signed_matrix_D_snowshoe_io(snowshoe_io):
    D = create_matrix(snowshoe_io, form='signed', matrix_type='D')
    result = (D.shape, D)
    expected = ((2, 2), sp.Matrix([
        [0, 0],
        [0, 0]
    ]))
    assert result == expected


def test_create_matrix_form_symbolic_matrix_B_snowshoe_io(snowshoe_io):
    result = create_matrix(snowshoe_io, form='symbolic', matrix_type='B')
    b_R_Inp1 = sp.Symbol('b_R,Inp1')
    b_C_Inp1 = sp.Symbol('b_C,Inp1')
    b_P_Inp2 = sp.Symbol('b_P,Inp2')
    expected = sp.Matrix([
        [ b_R_Inp1,         0],
        [-b_C_Inp1,         0],
        [        0, -b_P_Inp2]])
    assert result == expected


def test_create_matrix_form_symbolic_matrix_C_snowshoe_io(snowshoe_io):
    result = create_matrix(snowshoe_io, form='symbolic', matrix_type='C')
    c_Out1_C = sp.Symbol('c_Out1,C')
    c_Out1_P = sp.Symbol('c_Out1,P')
    c_Out2_C = sp.Symbol('c_Out2,C')
    expected = sp.Matrix([
        [0, -c_Out1_C, c_Out1_P],
        [0,  c_Out2_C,        0]])
    assert result == expected


def test_create_matrix_form_symbolic_matrix_D_snowshoe_io(snowshoe_io):
    result = create_matrix(snowshoe_io, form='symbolic', matrix_type='D')
    expected = sp.Matrix([
        [0, 0],
        [0, 0]])
    assert result == expected


def test_create_matrix_form_signed_chain(chain):
    A = create_matrix(chain, form='signed')
    result = A
    expected = sp.Matrix([
        [-1, -1,  0,  0,  0],
        [ 1, -1, -1,  0,  0],
        [ 0,  1, -1, -1,  0],
        [ 0,  0,  1, -1, -1],
        [ 0,  0,  0,  1, -1]])
    assert result == expected


def test_create_matrix_form_binary_chain(chain):
    A = create_matrix(chain, form='binary')
    result = A
    expected = sp.Matrix([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1]])
    assert result == expected


def test_create_matrix_form_symbolic_chain(chain):
    result = create_matrix(chain, form='symbolic')
    a_11 = sp.Symbol('a_1,1')
    a_12 = sp.Symbol('a_1,2')
    a_21 = sp.Symbol('a_2,1')
    a_22 = sp.Symbol('a_2,2')
    a_23 = sp.Symbol('a_2,3')
    a_32 = sp.Symbol('a_3,2')
    a_33 = sp.Symbol('a_3,3')
    a_34 = sp.Symbol('a_3,4')
    a_43 = sp.Symbol('a_4,3')
    a_44 = sp.Symbol('a_4,4')
    a_45 = sp.Symbol('a_4,5')
    a_54 = sp.Symbol('a_5,4')
    a_55 = sp.Symbol('a_5,5')
    expected = sp.Matrix([
        [-a_11, -a_12,     0,     0,     0],
        [ a_21, -a_22, -a_23,     0,     0],
        [    0,  a_32, -a_33, -a_34,     0],
        [    0,     0,  a_43, -a_44, -a_45],
        [    0,     0,     0,  a_54, -a_55]])
    assert result == expected


def test_create_matrix_form_symbolic_mesocosm(mesocosm):
    result = create_matrix(mesocosm, form='symbolic')
    a_PP = sp.Symbol('a_P,P')
    a_PA1 = sp.Symbol('a_P,A1')
    a_PA2 = sp.Symbol('a_P,A2')
    a_PAP = sp.Symbol('a_P,AP')
    a_A1P = sp.Symbol('a_A1,P')
    a_A1H1 = sp.Symbol('a_A1,H1')
    a_A1H2 = sp.Symbol('a_A1,H2')
    a_A2P = sp.Symbol('a_A2,P')
    a_A2H2 = sp.Symbol('a_A2,H2')
    a_APP = sp.Symbol('a_AP,P')
    a_APAP = sp.Symbol('a_AP,AP')
    a_H1A1 = sp.Symbol('a_H1,A1')
    a_H1C1 = sp.Symbol('a_H1,C1')
    a_H1C2 = sp.Symbol('a_H1,C2')
    a_H2A1 = sp.Symbol('a_H2,A1')
    a_H2A2 = sp.Symbol('a_H2,A2')
    a_H2C2 = sp.Symbol('a_H2,C2')
    a_C1H1 = sp.Symbol('a_C1,H1')
    a_C1C2 = sp.Symbol('a_C1,C2')
    a_C2H1 = sp.Symbol('a_C2,H1')
    a_C2H2 = sp.Symbol('a_C2,H2')
    a_C2C1 = sp.Symbol('a_C2,C1')
    a_C2C2 = sp.Symbol('a_C2,C2')
    expected = sp.Matrix([
        [-a_PP, -a_PA1, -a_PA2,  -a_PAP,       0,       0,       0,       0],
        [a_A1P,      0,      0,       0, -a_A1H1, -a_A1H2,       0,       0],
        [a_A2P,      0,      0,       0,       0, -a_A2H2,       0,       0],
        [a_APP,      0,      0, -a_APAP,       0,       0,       0,       0],
        [    0, a_H1A1,      0,       0,       0,       0, -a_H1C1, -a_H1C2],
        [    0, a_H2A1, a_H2A2,       0,       0,       0,       0, -a_H2C2],
        [    0,      0,      0,       0,  a_C1H1,       0,       0, -a_C1C2],
        [    0,      0,      0,       0,  a_C2H1,  a_C2H2,  a_C2C1, -a_C2C2]])
    assert result == expected

# =============================================================================
# create_equations()
# =============================================================================

def test_create_equations_form_output_without_inputs():
    graph = nx.DiGraph()
    graph.add_node('X', category='state')
    graph.add_node('Y', category='output')
    graph.add_edge('X', 'X', sign=-1)
    graph.add_edge('X', 'Y', sign=1)
    assert create_equations(graph, form='output') == sp.Matrix([
        sp.Symbol('c_Y,X') * sp.Symbol('x_X')])


def test_create_equations_form_state_snowshoe(snowshoe):
    result = create_equations(snowshoe, form='state')
    x_R = sp.Symbol('x_R')
    x_C = sp.Symbol('x_C')
    x_P = sp.Symbol('x_P')
    a_RR = sp.Symbol('a_R,R')
    a_RC = sp.Symbol('a_R,C')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    expected = sp.Matrix([
        -a_RR * x_R - a_RC * x_C,
        a_CR * x_R - a_CP * x_P,
        a_PC * x_C - a_PP * x_P
    ])
    assert result == expected


def test_create_equations_form_state_chain(chain):
    result = create_equations(chain, form='state')
    x_1 = sp.Symbol('x_1')
    x_2 = sp.Symbol('x_2')
    x_3 = sp.Symbol('x_3')
    x_4 = sp.Symbol('x_4')
    x_5 = sp.Symbol('x_5')
    a_11 = sp.Symbol('a_1,1')
    a_12 = sp.Symbol('a_1,2')
    a_21 = sp.Symbol('a_2,1')
    a_22 = sp.Symbol('a_2,2')
    a_23 = sp.Symbol('a_2,3')
    a_32 = sp.Symbol('a_3,2')
    a_33 = sp.Symbol('a_3,3')
    a_34 = sp.Symbol('a_3,4')
    a_43 = sp.Symbol('a_4,3')
    a_44 = sp.Symbol('a_4,4')
    a_45 = sp.Symbol('a_4,5')
    a_54 = sp.Symbol('a_5,4')
    a_55 = sp.Symbol('a_5,5')
    expected = sp.Matrix([
        -a_11 * x_1 - a_12 * x_2,
        a_21 * x_1 - a_22 * x_2 - a_23 * x_3,
        a_32 * x_2 - a_33 * x_3 - a_34 * x_4,
        a_43 * x_3 - a_44 * x_4 - a_45 * x_5,
        a_54 * x_4 - a_55 * x_5
    ])
    assert result == expected


def test_create_equations_form_state_snowshoe_io(snowshoe_io):
    result = create_equations(snowshoe_io, form='state')
    x_R = sp.Symbol('x_R')
    x_C = sp.Symbol('x_C')
    x_P = sp.Symbol('x_P')
    u_Inp1 = sp.Symbol('u_Inp1')
    u_Inp2 = sp.Symbol('u_Inp2')
    a_RR = sp.Symbol('a_R,R')
    a_RC = sp.Symbol('a_R,C')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    b_R_Inp1 = sp.Symbol('b_R,Inp1')
    b_C_Inp1 = sp.Symbol('b_C,Inp1')
    b_P_Inp2 = sp.Symbol('b_P,Inp2')
    expected = sp.Matrix([
        -a_RR * x_R - a_RC * x_C + b_R_Inp1 * u_Inp1,
        a_CR * x_R - a_CP * x_P - b_C_Inp1 * u_Inp1,
        a_PC * x_C - a_PP * x_P - b_P_Inp2 * u_Inp2
    ])
    assert result == expected


def test_create_equations_form_output_snowshoe_io(snowshoe_io):
    result = create_equations(snowshoe_io, form='output')
    x_C = sp.Symbol('x_C')
    x_P = sp.Symbol('x_P')
    c_Out1_C = sp.Symbol('c_Out1,C')
    c_Out1_P = sp.Symbol('c_Out1,P')
    c_Out2_C = sp.Symbol('c_Out2,C')
    expected = sp.Matrix([
        -c_Out1_C * x_C + c_Out1_P * x_P,
        c_Out2_C * x_C
    ])
    assert result == expected


def test_create_equations_form_output_rejects_no_outputs(snowshoe):
    with pytest.raises(ValueError, match="No output nodes"):
        create_equations(snowshoe, form='output')

# =============================================================================
# nodes_table() and edges_table()
# =============================================================================

def test_nodes_table_category_counts_snowshoe_io(snowshoe_io):
    result = nodes_table(snowshoe_io)
    assert len(result) == 7
    assert 'Node' in result.columns
    assert 'Label' in result.columns
    assert 'Category' in result.columns
    assert 'Description' in result.columns
    state_rows = result[result['Category'] == 'State']
    assert len(state_rows) == 3
    input_rows = result[result['Category'] == 'Input']
    assert len(input_rows) == 2
    output_rows = result[result['Category'] == 'Output']
    assert len(output_rows) == 2

def test_edges_table_columns_snowshoe_io(snowshoe_io):
    result = edges_table(snowshoe_io)
    assert len(result) > 0
    assert 'Edge' in result.columns
    assert 'From' in result.columns
    assert 'Sign' in result.columns
    assert 'To' in result.columns
    assert 'Dashes' in result.columns
    assert 'Description' in result.columns
    assert '+' in result['Sign'].values or '-' in result['Sign'].values

def test_create_matrix_rejects_invalid_nodes(disconnected_graph):
    disconnected_graph.nodes['C']['category'] = 'invalid'
    with pytest.raises(ValueError, match=r"Invalid nodes: \['C'\]"):
        create_matrix(disconnected_graph)


def test_create_equations_rejects_invalid_form(snowshoe):
    with pytest.raises(ValueError, match="form must be either 'state' or 'output'"):
        create_equations(snowshoe, form='invalid')


# =============================================================================
# Additional coverage tests
# =============================================================================

def test_edges_table_input_to_output_edge(direct_input_output_graph):
    result = edges_table(direct_input_output_graph)
    d_edges = result[result['Edge'].str.contains(r'\$d_')]
    assert len(d_edges) > 0


def test_edges_table_non_standard_sign(non_standard_sign_graph):
    result = edges_table(non_standard_sign_graph)
    assert '0.5' in result['Sign'].values


def test_define_input_output_remove_disconnected(snowshoe):
    G = nx.DiGraph(snowshoe)
    G.add_edge('Z', 'Z', sign=-1)
    result = nx.get_node_attributes(define_input_output(G), "category")
    expected = {'R': 'state', 'C': 'state', 'P': 'state', 'Z': 'state'}
    assert result == expected
    with pytest.warns(UserWarning, match=r"Dropped nodes: \['Z'\]"):
        result = nx.get_node_attributes(define_input_output(G, remove_disconnected=True), "category")
    expected = {'R': 'state', 'C': 'state', 'P': 'state'}
    assert result == expected
    assert list(G) == ['R', 'C', 'P', 'Z']
