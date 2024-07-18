import json
import copy
import itertools
from typing import Any, Iterable

import z3

Node = Any


def to_z3_val(input: str | int | float) -> z3.SeqRef | z3.RatNumRef:
    if isinstance(input, str):
        return z3.StringVal(input)
    elif isinstance(input, int) or isinstance(input, float):
        return z3.RealVal(input)


class Graph:
    def __init__(self):
        self.nodes = set()
        self.adjacency_map: dict[GraphNode, list[GraphNode]] = {}

    def add_node(self, node):
        self.nodes.add(node)
        if node not in self.adjacency_map:
            self.adjacency_map[node] = []

    def get_paths(self, source, target) -> list[list[Node]]:

        if source not in self.nodes or target not in self.nodes:
            return []

        # This is retarded as set, but we can't be sure about the node numbering
        visited = set()
        stack = [source]

        while len(stack) != 0:

            node = stack.pop()

            # don't visit nodes twice
            if node not in visited:
                if node == target:
                    return stack

                visited.add(node)

                # visit every neighbor
                for neighbor in self.adjacency_map[node]:
                    stack.append(neighbor)

        return stack

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
    def __init__(self, initial_state: int, transitions: list[AutomatonTransition], final_states: set[int]):
        self.initial_state: int = initial_state 
        self.transitions: list[AutomatonTransition] = transitions
        self.final_states: set[int] = final_states

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


def parse_json_file(file_path) -> tuple[Graph, NodeAttributes, Automaton, dict[str, z3.ExprRef]]:
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    graph_db = Graph()
    attributes = NodeAttributes()
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
    initial_state = json_data["Automaton"]["Initial State"]
    final_states = set(json_data["Automaton"]["Final States"])
    transitions = [
        AutomatonTransition(t['from'], t['to'], t['formula']) for t in json_data["Automaton"]["Transitions"]
    ]
    automaton = Automaton(initial_state, transitions, final_states)

    # Parse Global Variables
    for name, type in json_data["Global Variables"].items():
        global_vars[name] = create_global_var(name, type)

    return graph_db, attributes, automaton, global_vars


def get_vars(f):
    r = set()

    def collect(f):
        if z3.is_const(f):
            if f.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                r.add(f)
        else:
            for c in f.children():
                collect(c)
    collect(f)
    return r


def split_and(formula):
    conjuncts = []
    stack = [formula]
    while stack:
        current = stack.pop()
        if z3.is_and(current):
            stack.extend(current.children())
        else:
            conjuncts.append(current)
    return conjuncts


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

    while len(stack) != 0:
        (node, path, constraints, state) = stack.pop()

        s_constraints = tuple(sorted(map(str, constraints)))
        if (node, s_constraints) not in visited:
            if node == target:
                # Check if state is final if yes we are done
                if state in aut.final_states:
                    return True

            visited.add((node, s_constraints))

            for neighbor in graph.adjacency_map[node]:
                transitions = aut.transitions_from(state)

                # For each possible transition add it with the parameters replaced by the values
                for transition in transitions:
                    # Parse the formula
                    transition_formula = z3.parse_smt2_string(
                        transition.formula, decls=all_variables)[0]

                    # Replace all variables in the formula
                    for name, variable in attribute.alphabet.items():
                        value = attribute.attribute_map[str(neighbor)][name]
                        substitution = (variable, to_z3_val(value))
                        transition_formula = z3.substitute(
                            transition_formula, substitution)

                    transition_formulas = split_and(transition_formula)

                    solver = z3.Solver()
                    # Add all constraints we had before on this path
                    solver.add(constraints)
                    # Add the new constraint
                    solver.add(transition_formulas)

                    r = solver.check()
                    if r == z3.sat:
                        # Don't append formulas that don't contain any global variables, aka variables that haven't been replaced
                        to_append = list(filter(lambda formula: len(get_vars(formula)) >= 1, transition_formulas))

                        stack.append(
                            (neighbor, path + [neighbor], constraints + to_append, transition.to_state))

    return False


def is_upper_bound(formula, variable):
    if isinstance(formula, z3.BoolRef):
        if formula.decl().kind() in [z3.Z3_OP_LE, z3.Z3_OP_LT, z3.Z3_OP_GE, z3.Z3_OP_GT]:
            lhs = formula.arg(0)
            rhs = formula.arg(1)
            # Check if the variable is on the left-hand side
            if lhs == variable:
                return formula.decl().kind() in [z3.Z3_OP_LE, z3.Z3_OP_LT]
            # Check if the variable is on the right-hand side
            elif rhs == variable:
                return formula.decl().kind() in [z3.Z3_OP_GE, z3.Z3_OP_GT]
    return False


def is_lower_bound(formula, variable):
    if isinstance(formula, z3.BoolRef):
        if formula.decl().kind() in [z3.Z3_OP_LE, z3.Z3_OP_LT, z3.Z3_OP_GE, z3.Z3_OP_GT]:
            lhs = formula.arg(0)
            rhs = formula.arg(1)
            # Check if the variable is on the left-hand side
            if lhs == variable:
                return formula.decl().kind() in [z3.Z3_OP_GE, z3.Z3_OP_GT]
            # Check if the variable is on the right-hand side
            elif rhs == variable:
                return formula.decl().kind() in [z3.Z3_OP_LE, z3.Z3_OP_LT]
    return False


def query_with_macro_state(
        attribute: NodeAttributes,
        automaton: Automaton,
        graph: Graph,
        vars,
        source,
        target
) -> bool:
    all_variables = merge_dicts(vars, attribute.alphabet)
    visited = set()
    # (Node, path, {variable: (upperbound, lowerbound)}, state)
    stack: list[tuple[int, list[int], Any, int]] = [
        (source, [source], {var: (None, None) for var in vars.values()}, automaton.initial_state)]

    while len(stack) != 0:
        (node, path, constraints, state) = stack.pop()
        print(stack)

        if (node, state) in visited:
            continue

        if node == target:
            # Check if state is final if yes we are done
            if state in automaton.final_states:
                return True

        visited.add((node, state))
        constraints = copy.deepcopy(constraints)

        for neighbor in graph.adjacency_map[node]:

            # For each possible transition add it with the parameters replaced by the values
            for transition in automaton.transitions_from(state):
                # Parse the formula
                transition_formula = z3.parse_smt2_string(
                    transition.formula, decls=all_variables)[0]

                # Replace all variables in the formula
                substitutions = [(variable, to_z3_val(attribute.attribute_map[str(neighbor)][name]))
                                 for name, variable in attribute.alphabet.items()]
                transition_formula = z3.substitute(transition_formula, *substitutions)

                # Split up conjuncts, so we only store the necessary conjuncts
                transition_formulas = split_and(transition_formula)

                # Check if our path is satisfiable

                solver = z3.Solver()
                # Add all constraints we had before on this path
                for _, (upper, lower) in constraints.items():
                    if upper is not None:
                        solver.add(upper)
                    if lower is not None:
                        solver.add(lower)

                # Add the new constraint
                solver.add(transition_formulas)

                r = solver.check()

                # If we are still satisfiable continue this path, else discard it
                if r == z3.sat:

                    # Don't append formulas that don't contain any global variables, aka variables that haven't been replaced
                    global_bounds = filter(lambda formula: len(get_vars(formula)) >= 1, transition_formulas)

                    for bound in global_bounds:
                        # Should be only one variable
                        var = list(get_vars(bound))[0]

                        current_upper, current_lower = constraints[var]

                        if current_upper is None and is_upper_bound(bound, var):
                            current_upper = bound
                            constraints[var] = (bound, current_lower)

                        if current_lower is None and is_lower_bound(bound, var):
                            current_lower = bound
                            constraints[var] = (current_upper, bound)

                        # A bound is stronger then the other if it implies the other
                        # thus we need to check for validity of the implication bound => current_{upper, lower}
                        # for this we check the unsatisfiability of the negation of the implication

                        solver = z3.Solver()
                        if current_upper is not None:
                            solver.add(z3.Not(z3.Implies(bound, current_upper)))

                            # We found a new upper bound since the new bound implies the old one
                            if solver.check() == z3.unsat:
                                constraints[var] = (bound, current_lower)

                            solver.reset()

                        if current_lower is not None:
                            solver.add(z3.Not(z3.Implies(bound, current_lower)))

                            # We found a new lower bound since the new bound implies the old one
                            if solver.check() == z3.unsat:
                                (upper, _) = constraints[var]
                                constraints[var] = (upper, bound)

                    stack.append(
                        (neighbor, path + [neighbor], constraints, transition.to_state))

    return False

type AutomatonState = int
type GraphNode = int
type GlobalVars = dict[str, z3.ExprRef]
type MetaNode = tuple[AutomatonState, GraphNode]
type Path = list[MetaNode]
type GlobalConstraints = list[Any]
type Formula = z3.AstVector

def check_transition(formula, global_constraints: GlobalConstraints) -> bool:
    solver = z3.Solver()
    solver.add(formula)
    for bound in global_constraints:
        if bound is not None:
            solver.add(bound)
    return solver.check() == z3.sat

# returns True iff bound1 implies bound2
def implies(formulas, bound2) -> bool:
    s = z3.Solver()
    s.add(z3.Not(z3.Implies(z3.And(formulas), bound2)))
    return s.check() == z3.unsat

def minimize_global_constraints(global_constraints: GlobalConstraints, formula) -> GlobalConstraints: 
    global_constraints = copy.deepcopy(global_constraints)
    for constraint in split_and(formula):
        if not implies(z3.And(global_constraints), constraint):
            global_constraints.append(constraint)
    length = len(global_constraints)
    for index in reversed(range(length)):
        if implies(global_constraints[:index] + global_constraints[index + 1:], global_constraints[index]):
            global_constraints.pop(index)
    return global_constraints

def dk_query_with_macro_state(
        graph_attributes: NodeAttributes,
        automaton: Automaton,
        graph: Graph,
        global_vars: GlobalVars,
        source_node: GraphNode,
        target_node: GraphNode,
) -> bool:
    initial = ([(automaton.initial_state, source_node)], [])
    stack: list[tuple[Path, GlobalConstraints]] = [initial]
    visited: list[tuple[MetaNode, GlobalConstraints]] = []
    all_variables: dict[str, z3.ExprRef] = merge_dicts(graph_attributes.alphabet, global_vars)
    while stack:
        state = stack.pop()
        (path, global_constraints) = state
        meta_node = path[-1]
        visited_entry = (meta_node, global_constraints)
        if visited_entry in visited:
            continue
        else:
            visited.append(visited_entry)
        (automaton_state, graph_node) = meta_node
        for (automaton_transition, next_graph_node) in itertools.product(automaton.transitions_from(automaton_state), graph.adjacency_map[graph_node]):
            next_automaton_state = automaton_transition.to_state
            transition_formula = z3.parse_smt2_string(
                automaton_transition.formula, decls=all_variables)[0]

            # Replace all variables in the formula
            substitutions = [(variable, to_z3_val(graph_attributes.attribute_map[str(next_graph_node)][name]))
                             for name, variable in graph_attributes.alphabet.items()]
            transition_formula = z3.substitute(transition_formula, *substitutions)
            next_path = path + [(next_automaton_state, next_graph_node)]
            if not check_transition(transition_formula, global_constraints):
                # print("notttttt trans", next_path, transition_formula, global_constraints)
                continue
            # print("accepted trans", next_path, transition_formula, global_constraints)
            if next_graph_node == target_node and automaton_transition.to_state in automaton.final_states:
                # print("found", next_path)
                return True
            next_global_constraints = minimize_global_constraints(global_constraints, transition_formula)
            next_state = (next_path, next_global_constraints)
            if not next_state in stack:
                stack.append(next_state)
    return False
