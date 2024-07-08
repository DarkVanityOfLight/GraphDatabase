import json
import z3
from typing import Any, List, Set, Iterable
Node = Any


def to_z3_val(input: str | int | float) -> Any:
    if isinstance(input, str):
        return z3.StringVal(input)
    elif isinstance(input, int) or isinstance(input, float):
        return z3.RealVal(input)


class Graph:
    def __init__(self):
        self.nodes = set()
        self.adjacency_map = {}

    def add_node(self, node):
        self.nodes.add(node)
        if node not in self.adjacency_map:
            self.adjacency_map[node] = []

    # TODO I think there is something off here don't know when it will break tho
    def get_paths(self, source, target) -> List[List[Node]]:
        stack = [(source, [source])]
        paths = set()

        while stack:
            node, path = stack.pop()

            if node == target:
                # Use tuple to store paths in a set (tuples are hashable)
                paths.add(tuple(path))

            for neighbor in self.adjacency_map.get(node, []):
                if neighbor not in path:  # Avoid cycles
                    stack.append((neighbor, path + [neighbor]))

        # Make sure we return the correct thing xD
        return [list(path) for path in paths]

    def add_edge(self, from_node, to_node):
        self.add_node(from_node)
        self.add_node(to_node)
        self.adjacency_map[from_node].append(to_node)

    def __str__(self):
        result = "Graph:\n"
        for node in sorted(self.nodes):
            result += f"{node}: {', '.join(map(str, self.adjacency_map.get(node, [])))}\n"
        return result


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

    def transitions_from(self, state: int) -> Iterable[AutomatonTransition]:
        return filter(lambda x: x.from_state == state, self.transitions)

    def transitions_to(self, state: int) -> Iterable[AutomatonTransition]:
        return filter(lambda x: x.to_state == state, self.transitions)

    def transitions_from_to(self, from_state: int, to_state: int) -> Iterable[AutomatonTransition]:
        return filter(lambda x: x.from_state == from_state and x.to_state == to_state, self.transitions)


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
        attribute_map[vertex] = attr
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


def query_with_naive_algorithm(
        attribute: NodeAttributes,
        aut: Automaton,
        graph: Graph,
        vars,
        source,
        target) -> bool:

    all_variables = merge_dicts(vars, attribute.alphabet)
    visited = set()
    stack: list[tuple[int, list[int], list[Any], int]] = [
        (source, [source], [], aut.initial_state)]
    # candidate_solutions: list[tuple[list[int], list[Any]]] = []

    while len(stack) != 0:
        (node, path, constraints, state) = stack.pop()

        if (node, state) not in visited:
            if node == target:
                # Check if state is final if yes we are done
                if state in aut.final_states:
                    return True
                    # candidate_solutions.append((path, constraints))

            visited.add((node, state))

            for neighbor in graph.adjacency_map[node]:
                transitions = aut.transitions_from(state)

                # For each possible transition add it with the parameters replaced by the values
                for transition in transitions:
                    # Parse the formula
                    transition_formula = z3.parse_smt2_string(
                        transition.formula, decls=all_variables)[0]

                    # Replace all variables in the formula
                    for name, variable in attribute.alphabet.items():
                        value = attribute.attribute_map[str(
                            neighbor)][name]
                        substitution = (variable, to_z3_val(value))
                        transition_formula = z3.substitute(
                            transition_formula, substitution)

                    solver = z3.Solver()
                    # Add all constraints we had before on this path
                    solver.add(constraints)
                    # Add the new constraint
                    solver.add(transition_formula)

                    r = solver.check()
                    if r == z3.sat:
                        # TODO throw out formulas already evaluated, aka not containing global variables
                        # TODO Task 3, can be done here, just replace upper and lower bounds for a specific global variable
                        stack.append(
                            (neighbor, path + [neighbor], constraints + [transition_formula], transition.to_state))

    return False


def query_with_macro_state(
        attribute: NodeAttributes,
        aut: Automaton,
        graph: Graph,
        vars,
        source,
        target
) -> bool:
    pass
# TODO To be implemented for the third task
