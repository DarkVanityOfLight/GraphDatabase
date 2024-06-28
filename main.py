#!/bin/env python3

import json
import z3

class Graph:
    def __init__(self):
        self.nodes = set()
        self.adjacency_map = {}

    def add_node(self, node):
        self.nodes.add(node)
        if node not in self.adjacency_map:
            self.adjacency_map[node] = []

    def add_edge(self, from_node, to_node):
        self.add_node(from_node)
        self.add_node(to_node)
        self.adjacency_map[from_node].append(to_node)

    def __str__(self):
        result = "Graph:\n"
        for node in sorted(self.nodes):
            result += f"{node}: {', '.join(map(str, self.adjacency_map.get(node, [])))}\n"
        return result

    def find_path(self, start, end):
        # This is retarded as set, but we can't be sure about the node numbering
        visited = set()
        stack = [(start, [start])]

        while len(stack) != 0:

            (node, path) = stack.pop()

            # don't visit nodes twice
            if node not in visited:
                if node == end:
                    return path

                visited.add(node)

                # visit every neighbor
                for neighbor in self.adjacency_map[node]:
                    stack.append((neighbor, path + [neighbor]))

        return None

        

class AutomatonTransition:
    def __init__(self, from_state, to_state, formula):
        self.from_state = from_state
        self.to_state = to_state
        self.formula = formula

    def __str__(self):
        return f"From: {self.from_state}, To: {self.to_state}, Formula: {self.formula}"

class NodeAttributes:
    def __init__(self):
        self.alphabet = {}
        self.attribute_map = {}

    def add_variable(self, var_name, value):
        if isinstance(value, (int, float)):
            self.alphabet[var_name] = z3.Real(var_name)
        elif isinstance(value, str):
            self.alphabet[var_name] = z3.String(var_name)
        else:
            raise ValueError("Unsupported attribute type")


    def get_variable(self, var_name):
        return self.alphabet.get(var_name, None)
    
    def __str__(self):
        output = "Node Attributes:\n"
        for var_name, value in self.attribute_map.items():
            output += f"{var_name}: {value}\n"
        return output

class Automaton:
    def __init__(self):
        self.initial_state = None
        self.transitions = []
        self.final_states = set()
    def __str__(self):
        transitions_str = "\n".join(str(transition) for transition in self.transitions)
        return f"Initial State: {self.initial_state}, Transitions:\n{transitions_str}, Final States: {self.final_states}"


def create_global_var(var_name, type):
        if type == "Real":
            return z3.Real(var_name)
        elif type == "String":
            return z3.String(var_name)
        else:
            raise ValueError("Unsupported attribute type")

def merge_dicts(dict1, dict2):
    common_keys = set(dict1.keys()) & set(dict2.keys())
    if common_keys:
        raise ValueError(f"Key(s) {common_keys} are present in both dictionaries and would be overwritten")

    return {**dict1, **dict2}

def parse_json_file(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    graph_db = Graph()
    attributes = NodeAttributes()
    automaton = Automaton()
    attribute_map = {}
    global_vars = {}

    # Parse Graph Database
    for edge in json_data["Graph Database"]["Edges"]:
        from_node, to_node = map(int, edge.split(" -> "))
        graph_db.add_edge(from_node, to_node)

    # Parse Attributes
    for vertex, attr in json_data["Attributes"].items():
        for attr_name, attr_value in attr.items():
            attributes.add_variable(attr_name, attr_value)

    for vertex, attr in json_data["Attributes"].items():
        attr_tuple = tuple(attr.values())
        attribute_map[vertex] = attr_tuple
    attributes.attribute_map = attribute_map
    # Parse Automaton
    automaton.initial_state = json_data["Automaton"]["Initial State"]
    automaton.final_states = set(json_data["Automaton"]["Final States"])
    automaton.transitions = [
        AutomatonTransition(t['from'], t['to'], t['formula']) for t in json_data["Automaton"]["Transitions"]
    ]

    # Parse Global Variables
    for name, type in json_data["Global Variables"].items():
        global_vars[name] = create_global_var(name, type)

    return graph_db, attributes, automaton, global_vars


if __name__ == '__main__':
    solver = z3.Solver()
    file_path = 'example1.json'  # Path to your JSON file

    # Parse JSON file
    parsed_graph, parsed_attributes, parsed_automaton, global_vars = parse_json_file(file_path)

    # Example usage to access the parsed data
    print("Graph Database:", parsed_graph)
    print("Automaton:", parsed_automaton)
    print("Attributes: ", parsed_attributes)
    print("Alphabet: ", parsed_attributes.alphabet)
    print("Formula: ", parsed_automaton.transitions[0].formula)
    print("Global Vars", global_vars)

    all_variables = merge_dicts(parsed_attributes.alphabet, global_vars)
    # Parse smt2 string with declared vars; returns vector of assertions, in our case always 1
    test0 = z3.parse_smt2_string(parsed_automaton.transitions[0].formula, decls=all_variables)[0]
    solver.add(test0)
    print("test0: ", test0)
    solver.check()
    print("model 1:",solver.model())
    test = z3.parse_smt2_string(parsed_automaton.transitions[1].formula, decls=all_variables)[0]
    print("test:", test)
    solver.add(test)
    # Check model
    solver.check()
    print("model 2: ", solver.model())

    # Replace age by value 2
    test4 = (parsed_attributes.alphabet['age'])
    test5 = (global_vars['p1'])
    # test0[0] is the first assert in the z3 ast vector
    expr2 = z3.substitute(test0, (test4, z3.RealVal(2.0)))
    print("Substitute age by 2: ", expr2)

    print("==========================================================")

    print(parsed_graph.find_path(1, 3))

