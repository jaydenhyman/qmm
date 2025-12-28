"""Tests for qmm.core.structure module."""

import json
import pytest
import sympy as sp

from qmm import import_digraph, create_matrix, create_equations, define_input_output

# =============================================================================
# import_digraph()
# =============================================================================

def test_import_digraph_from_dict_inline_data():
    """Test import_digraph with inline dictionary data to build a graph."""
    data = {
        "nodes": [{"id": "A"}, {"id": "B"}],
        "edges": [
            {"from": "A", "to": "B", "arrows": {"to": {"type": "triangle"}}},
        {"from": "B", "to": "A", "arrows": {"to": {"type": "circle"}}}
    ]
    }
    G = import_digraph(data, file_path=False)
    result = (G['A']['B']['sign'], G['B']['A']['sign'])
    expected = (1, -1)
    assert result == expected


def test_import_digraph_from_file_tmp_path(tmp_path):
    """Test import_digraph with JSON file path input."""
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"nodes": [{"id": "X"}], "edges": []}))
    G = import_digraph(str(path), file_path=True)
    result = G.number_of_nodes()
    expected = 1
    assert result == expected


def test_import_digraph_node_attributes_inline_data():
    """Test import_digraph preserves node attributes from dictionaries."""
    data = {
        "nodes": [{"id": "A", "label": "Node A"}, {"id": "B"}],
        "edges": []
    }
    G = import_digraph(data, file_path=False)
    result = G.nodes['A']['label']
    expected = 'Node A'
    assert result == expected

# =============================================================================
# create_matrix()
# =============================================================================

def test_create_matrix_form_signed_snowshoe(snowshoe):
    """Test create_matrix with signed form for the snowshoe model."""
    A = create_matrix(snowshoe, form='signed')
    result = A
    expected = sp.Matrix([
        [-1, -1, 0],
        [1, 0, -1],
        [0, 1, -1]
    ])
    assert result == expected


def test_create_matrix_form_symbolic_snowshoe(snowshoe):
    """Test create_matrix with symbolic form for the snowshoe model."""
    result = create_matrix(snowshoe, form='symbolic')
    a_RR = sp.Symbol('a_R,R')
    a_RC = sp.Symbol('a_R,C')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    expected = sp.Matrix([
        [-a_RR, -a_RC, 0],
        [a_CR, 0, -a_CP],
        [0, a_PC, -a_PP]
    ])
    assert result == expected


def test_create_matrix_form_binary_snowshoe(snowshoe):
    """Test create_matrix with binary form for the snowshoe model."""
    A = create_matrix(snowshoe, form='binary')
    result = A
    expected = sp.Matrix([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ])
    assert result == expected


def test_create_matrix_form_signed_matrix_B_snowshoe_io(snowshoe_io):
    """Test create_matrix with signed form for input matrix B."""
    B = create_matrix(snowshoe_io, form='signed', matrix_type='B')
    result = (B.shape, B)
    expected = ((3, 2), sp.Matrix([
        [1, 0],
        [-1, 0],
        [0, -1]
    ]))
    assert result == expected


def test_create_matrix_form_signed_matrix_C_snowshoe_io(snowshoe_io):
    """Test create_matrix with signed form for output matrix C."""
    C = create_matrix(snowshoe_io, form='signed', matrix_type='C')
    result = (C.shape, C)
    expected = ((2, 3), sp.Matrix([
        [0, -1, 1],
        [0, 1, 0]
    ]))
    assert result == expected


def test_create_matrix_form_signed_matrix_D_snowshoe_io(snowshoe_io):
    """Test create_matrix with signed form for feedthrough matrix D."""
    D = create_matrix(snowshoe_io, form='signed', matrix_type='D')
    result = (D.shape, D)
    expected = ((2, 2), sp.Matrix([
        [0, 0],
        [0, 0]
    ]))
    assert result == expected


def test_create_matrix_form_symbolic_matrix_B_snowshoe_io(snowshoe_io):
    """Test create_matrix with symbolic form for input matrix B."""
    result = create_matrix(snowshoe_io, form='symbolic', matrix_type='B')
    b_R_Inp1 = sp.Symbol('b_R,Inp1')
    b_C_Inp1 = sp.Symbol('b_C,Inp1')
    b_P_Inp2 = sp.Symbol('b_P,Inp2')
    expected = sp.Matrix([
        [b_R_Inp1, 0],
        [-b_C_Inp1, 0],
        [0, -b_P_Inp2]
    ])
    assert result == expected


def test_create_matrix_form_symbolic_matrix_C_snowshoe_io(snowshoe_io):
    """Test create_matrix with symbolic form for output matrix C."""
    result = create_matrix(snowshoe_io, form='symbolic', matrix_type='C')
    c_Out1_C = sp.Symbol('c_Out1,C')
    c_Out1_P = sp.Symbol('c_Out1,P')
    c_Out2_C = sp.Symbol('c_Out2,C')
    expected = sp.Matrix([
        [0, -c_Out1_C, c_Out1_P],
        [0, c_Out2_C, 0]
    ])
    assert result == expected


def test_create_matrix_form_symbolic_matrix_D_snowshoe_io(snowshoe_io):
    """Test create_matrix with symbolic form for feedthrough matrix D."""
    result = create_matrix(snowshoe_io, form='symbolic', matrix_type='D')
    expected = sp.Matrix([
        [0, 0],
        [0, 0]
    ])
    assert result == expected


def test_create_matrix_form_signed_chain(chain):
    """Test create_matrix with signed form for the chain model."""
    A = create_matrix(chain, form='signed')
    result = A
    expected = sp.Matrix([
        [-1, -1, 0, 0, 0],
        [1, -1, -1, 0, 0],
        [0, 1, -1, -1, 0],
        [0, 0, 1, -1, -1],
        [0, 0, 0, 1, -1]
    ])
    assert result == expected


def test_create_matrix_form_binary_chain(chain):
    """Test create_matrix with binary form for the chain model."""
    A = create_matrix(chain, form='binary')
    result = A
    expected = sp.Matrix([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1]
    ])
    assert result == expected


def test_create_matrix_form_symbolic_chain(chain):
    """Test create_matrix with symbolic form for the chain model."""
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
        [-a_11, -a_12, 0, 0, 0],
        [a_21, -a_22, -a_23, 0, 0],
        [0, a_32, -a_33, -a_34, 0],
        [0, 0, a_43, -a_44, -a_45],
        [0, 0, 0, a_54, -a_55]
    ])
    assert result == expected


def test_create_matrix_form_symbolic_mesocosm(mesocosm):
    """Test create_matrix with symbolic form for the mesocosm model."""
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
        [-a_PP, -a_PA1, -a_PA2, -a_PAP, 0, 0, 0, 0],
        [a_A1P, 0, 0, 0, -a_A1H1, -a_A1H2, 0, 0],
        [a_A2P, 0, 0, 0, 0, -a_A2H2, 0, 0],
        [a_APP, 0, 0, -a_APAP, 0, 0, 0, 0],
        [0, a_H1A1, 0, 0, 0, 0, -a_H1C1, -a_H1C2],
        [0, a_H2A1, a_H2A2, 0, 0, 0, 0, -a_H2C2],
        [0, 0, 0, 0, a_C1H1, 0, 0, -a_C1C2],
        [0, 0, 0, 0, a_C2H1, a_C2H2, a_C2C1, -a_C2C2]
    ])
    assert result == expected

# =============================================================================
# create_equations()
# =============================================================================

def test_create_equations_form_state_snowshoe(snowshoe):
    """Test create_equations with state form for the snowshoe model."""
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
    """Test create_equations with state form for the chain model."""
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
    """Test create_equations with state form for the IO snowshoe model."""
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
    """Test create_equations with output form for the IO snowshoe model."""
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


def test_create_equations_form_output_no_outputs_snowshoe(snowshoe):
    """Test create_equations raising ValueError for missing output nodes."""
    with pytest.raises(ValueError, match="No output nodes"):
        create_equations(snowshoe, form='output')

# =============================================================================
# define_input_output()
# =============================================================================

def test_define_input_output_categories_snowshoe_io(snowshoe_io):
    """Test define_input_output classification for snowshoe IO graph."""
    result = {node: snowshoe_io.nodes[node]['category'] for node in ['R', 'C', 'P', 'Inp1', 'Inp2', 'Out1', 'Out2']}
    expected = {
        'R': 'state',
        'C': 'state',
        'P': 'state',
        'Inp1': 'input',
        'Inp2': 'input',
        'Out1': 'output',
        'Out2': 'output'
    }
    assert result == expected


def test_define_input_output_no_io_nodes_snowshoe(snowshoe):
    """Test define_input_output defaults nodes to state category when IO is absent."""
    result_graph = define_input_output(snowshoe)
    result = tuple(result_graph.nodes[node]['category'] for node in sorted(result_graph.nodes()))
    expected = tuple('state' for _ in result)
    assert result == expected


def test_define_input_output_invalid_input_type_no_fixture():
    """Test define_input_output raising TypeError for non-graph input."""
    with pytest.raises(TypeError):
        define_input_output("not a graph")

# =============================================================================
# nodes_table() and edges_table()
# =============================================================================

def test_nodes_table_snowshoe_io(snowshoe_io):
    """Test nodes_table creates proper metadata table for snowshoe IO model."""
    from qmm import nodes_table
    result = nodes_table(snowshoe_io)
    assert len(result) == 7  # 3 state + 2 input + 2 output
    assert 'Node' in result.columns
    assert 'Label' in result.columns
    assert 'Category' in result.columns
    assert 'Description' in result.columns
    # Check that state nodes are formatted as x_{id}
    state_rows = result[result['Category'] == 'State']
    assert len(state_rows) == 3
    # Check that input nodes are formatted as u_{id}
    input_rows = result[result['Category'] == 'Input']
    assert len(input_rows) == 2
    # Check that output nodes are formatted as y_{id}
    output_rows = result[result['Category'] == 'Output']
    assert len(output_rows) == 2

def test_edges_table_snowshoe_io(snowshoe_io):
    """Test edges_table creates proper metadata table for snowshoe IO model."""
    from qmm import edges_table
    result = edges_table(snowshoe_io)
    assert len(result) > 0
    assert 'Edge' in result.columns
    assert 'From' in result.columns
    assert 'Sign' in result.columns
    assert 'To' in result.columns
    assert 'Dashes' in result.columns
    assert 'Description' in result.columns
    # Check sign labels
    assert '+' in result['Sign'].values or '-' in result['Sign'].values

def test_create_equations_form_invalid_snowshoe(snowshoe):
    """Test create_equations raising ValueError for invalid form."""
    with pytest.raises(ValueError, match="form must be either 'state' or 'output'"):
        create_equations(snowshoe, form='invalid')


# =============================================================================
# Additional coverage tests
# =============================================================================

def test_edges_table_input_to_output_edge():
    """Test edges_table handles direct input-to-output edges (D matrix)."""
    import networkx as nx
    from qmm import edges_table, define_input_output

    # Create graph with direct D matrix edge
    G = nx.DiGraph()
    G.add_node('X', category='state')
    G.add_node('U', category='input')
    G.add_node('Y', category='output')
    G.add_edge('U', 'X', sign=1)
    G.add_edge('U', 'Y', sign=1)  # Direct input to output
    G.add_edge('X', 'Y', sign=1)
    nx.freeze(G)

    result = edges_table(G)
    # Check that the edge from input to output has prefix 'd'
    d_edges = result[result['Edge'].str.contains(r'\$d_')]
    assert len(d_edges) > 0


def test_edges_table_non_standard_sign():
    """Test edges_table handles non +1/-1 sign values."""
    import networkx as nx
    from qmm import edges_table

    G = nx.DiGraph()
    G.add_node('A')
    G.add_node('B')
    G.add_edge('A', 'B', sign=0.5)  # Non-standard sign

    result = edges_table(G)
    assert '0.5' in result['Sign'].values
