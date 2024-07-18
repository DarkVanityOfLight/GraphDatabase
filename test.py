import unittest

from util import *

graph1, attributes1, automaton1, global_vars1 = parse_json_file('example1.json')
graph2, attributes2, automaton2, global_vars2 = parse_json_file('example2.json')
graph3, attributes3, automaton3, global_vars3 = parse_json_file('example3.json')
graph4, attributes4, automaton4, global_vars4 = parse_json_file('example4.json')


def test_algorithm(algorithm):
    assert algorithm(attributes1, automaton1, graph1, global_vars1, 1, 5)
    assert algorithm(attributes1, automaton1, graph1, global_vars1, 1, 4)
    assert algorithm(attributes1, automaton1, graph1, global_vars1, 2, 5)
    assert not algorithm(attributes1, automaton1, graph1, global_vars1, 2, 3)

    assert algorithm(attributes2, automaton2, graph2, global_vars2, 2, 3)
    assert not algorithm(attributes2, automaton2, graph2, global_vars2, 3, 1)

    assert algorithm(attributes3, automaton3, graph3, global_vars3, 4, 6)
    assert not algorithm(attributes3, automaton3, graph3, global_vars3, 2, 5)
    assert not algorithm(attributes3, automaton3, graph3, global_vars3, 2, 3)

    assert algorithm(attributes4, automaton4, graph4, global_vars4, 1, 13)
    assert algorithm(attributes4, automaton4, graph4, global_vars4, 1, 10)
    assert algorithm(attributes4, automaton4, graph4, global_vars4, 3, 10)
    assert not algorithm(attributes4, automaton4, graph4, global_vars4, 1, 7)
    assert not algorithm(attributes4, automaton4, graph4, global_vars4, 3, 7)
    assert not algorithm(attributes4, automaton4, graph4, global_vars4, 9, 7)


class Test(unittest.TestCase):

    def test_graph_exploration(self):
        assert graph1.get_paths(1, 5)
        assert graph2.get_paths(1, 4)
        assert graph3.get_paths(2, 5)
        assert not graph3.get_paths(6, 5)
        assert not graph3.get_paths(2, 1)

    def test_naive_algorithm(self):
        test_algorithm(query_with_naive_algorithm)

    def test_macro_state_algorithm(self):
        test_algorithm(query_with_macro_state)

    def test_minimize_constraints(self):
        vars = global_vars3
        constraints = []
        formula0 = z3.parse_smt2_string("(assert (< p1 3))", decls=vars)[0]
        constraints = minimize_global_constraints(constraints, formula0)
        assert constraints == [formula0]
        formula1 = z3.parse_smt2_string("(assert (< p1 1))", decls=vars)[0]
        constraints = minimize_global_constraints(constraints, formula1)
        assert constraints == [formula1]
        _formula2 = z3.parse_smt2_string("(assert (< p1 -1))", decls=vars)[0]
        constraints = minimize_global_constraints(constraints, formula1)
        assert constraints == [formula1]
