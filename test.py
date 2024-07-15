from util import *
import unittest
file_path1 = 'example1.json'  # Path to your JSON file
graph1, attributes1, automaton1, global_vars1 = parse_json_file(file_path1)
file_path2 = 'example2.json'  # Path to your JSON file
graph2, attributes2, automaton2, global_vars2 = parse_json_file(file_path2)
file_path3 = 'example3.json'  # Path to your JSON file
graph3, attributes3, automaton3, global_vars3 = parse_json_file(file_path3)
file_path4 = 'example4.json'  # Path to your JSON file
graph4, attributes4, automaton4, global_vars4 = parse_json_file(file_path4)


class Test(unittest.TestCase):

    def test_graph_exploration(self):
        assert graph1.get_paths(1, 5)
        assert graph2.get_paths(1, 4)
        assert graph3.get_paths(2, 5)
        assert not graph3.get_paths(6, 5)
        assert not graph3.get_paths(2, 1)

    def test_naive_algorithm(self):
        assert query_with_naive_algorithm(attributes1, automaton1, graph1, global_vars1, 1, 5)
        assert query_with_naive_algorithm(attributes1, automaton1, graph1, global_vars1, 1, 4)
        assert query_with_naive_algorithm(attributes1, automaton1, graph1, global_vars1, 2, 5)
        assert not query_with_naive_algorithm(attributes1, automaton1, graph1, global_vars1, 2, 3)

        assert not query_with_naive_algorithm(attributes2, automaton2, graph2, global_vars2, 3, 1)
        assert query_with_naive_algorithm(attributes2, automaton2, graph2, global_vars2, 2, 3)

        assert query_with_naive_algorithm(attributes3, automaton3, graph3, global_vars3, 4, 6)
        assert not query_with_naive_algorithm(attributes3, automaton3, graph3, global_vars3, 2, 5)
        assert not query_with_naive_algorithm(attributes3, automaton3, graph3, global_vars3, 2, 3)

        assert query_with_naive_algorithm(attributes4, automaton4, graph4, global_vars4, 1, 13)
        assert query_with_naive_algorithm(attributes4, automaton4, graph4, global_vars4, 1, 10)
        assert query_with_naive_algorithm(attributes4, automaton4, graph4, global_vars4, 3, 10)
        assert not query_with_naive_algorithm(attributes4, automaton4, graph4, global_vars4, 1, 7)
        assert not query_with_naive_algorithm(attributes4, automaton4, graph4, global_vars4, 3, 7)
        assert not query_with_naive_algorithm(attributes4, automaton4, graph4, global_vars4, 9, 7)

    def test_macro_state_algorithm(self):
        assert query_with_macro_state(attributes1, automaton1, graph1, global_vars1, 1, 5)
        assert query_with_macro_state(attributes1, automaton1, graph1, global_vars1, 1, 4)
        assert not query_with_macro_state(attributes1, automaton1, graph1, global_vars1, 1, 2)
        assert not query_with_macro_state(attributes1, automaton1, graph1, global_vars1, 2, 3)

        assert not query_with_macro_state(attributes2, automaton2, graph2, global_vars2, 3, 1)
        assert query_with_macro_state(attributes2, automaton2, graph2, global_vars2, 2, 3)

        assert query_with_macro_state(attributes3, automaton3, graph3, global_vars3, 4, 6)
        assert not query_with_macro_state(attributes3, automaton3, graph3, global_vars3, 2, 5)
        assert not query_with_macro_state(attributes3, automaton3, graph3, global_vars3, 2, 3)

        assert query_with_macro_state(attributes4, automaton4, graph4, global_vars4, 1, 13)
        assert query_with_macro_state(attributes4, automaton4, graph4, global_vars4, 1, 10)
        assert query_with_macro_state(attributes4, automaton4, graph4, global_vars4, 3, 10)
        assert not query_with_macro_state(attributes4, automaton4, graph4, global_vars4, 1, 7)
        assert not query_with_macro_state(attributes4, automaton4, graph4, global_vars4, 3, 7)
        assert not query_with_macro_state(attributes4, automaton4, graph4, global_vars4, 9, 7)
