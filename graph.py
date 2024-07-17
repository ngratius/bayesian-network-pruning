import pysmile
import random
import metrics
import kl_divergence


def dfs(net, node, visited):
    """Perform DFS to mark all reachable nodes from the given node."""
    visited.add(node)  # Mark the current node as visited

    # Traverse all children of the current node
    children = net.get_children(node)
    for child in children:
        if child not in visited:
            dfs(net, child, visited)

    # Traverse all parents of the current node (to ensure undirected-like traversal)
    parents = net.get_parents(node)
    for parent in parents:
        if parent not in visited:
            dfs(net, parent, visited)


def is_connected(net):
    """Check if all nodes are reachable from the first node."""
    visited = set()  # Set to keep track of visited nodes

    first_node = net.get_first_node()  # Get the first node in the network
    dfs(net, first_node, visited)  # Perform DFS starting from the first node

    # Count total number of nodes in the network
    n = net.get_first_node()
    total_nodes = 0
    while n >= 0:
        total_nodes += 1
        n = net.get_next_node(n)

    # If the number of visited nodes equals the total number of nodes, the network is connected (output is True)
    return len(visited) == total_nodes


def duplicate_network(net):
    net_copy = pysmile.Network()

    # Copy nodes
    node_map = {}  # Mapping from original node IDs to new node IDs
    n = net.get_first_node()
    while n >= 0:
        node_type = net.get_node_type(n)
        node_name = net.get_node_name(n)
        # print(f"Adding node: type={node_type}, name={node_name}")

        # Add the node with its type and name
        new_node_id = net_copy.add_node(pysmile.NodeType(node_type), node_name)
        node_map[n] = new_node_id
        n = net.get_next_node(n)

    # Copy arcs using the node_map
    for original_parent, new_parent in node_map.items():
        children = net.get_children(original_parent)
        for child in children:
            new_child = node_map[child]
            net_copy.add_arc(new_parent, new_child)

    return net_copy


def non_cut_entities(net):

    # Initialize sets to store non-cut nodes and non-cut arcs
    non_cut_nodes = []
    non_cut_arcs = []
    cut_arcs = []

    # Iterate over all nodes and their children to list all arcs
    n = net.get_first_node()
    while n >= 0:
        # Create a copy of the network to work on (avoid restoring arcs without CPDs or lost nodes)
        net_copy = duplicate_network(net)
        # Test arcs starting from that node:
        # node_name = net_copy.get_node_name(n)
        children = net_copy.get_children(n)
        for child in children:
            # child_name = net_copy.get_node_name(child)
            net_copy.delete_arc(n, child)  # Temporarily remove the arc
            if is_connected(net_copy):
                # print(f"{node_name} -> {child_name}")
                non_cut_arcs.append((n, child))
            else:
                cut_arcs.append((n, child))
            net_copy.add_arc(n, child)  # Restore the arc

        # Test the node itself:
        net_copy.delete_node(n)  # Remove the node
        if is_connected(net_copy):
            non_cut_nodes.append(n)

        n = net.get_next_node(n)

    return non_cut_nodes, non_cut_arcs, cut_arcs


def get_number_of_independent_parameters(net):
    """Calculate the total number of independent parameters in the Bayesian network."""
    total_independent_parameters = 0

    n = net.get_first_node()
    while n >= 0:
        num_states = net.get_outcome_count(n)  # Number of states of the node
        parents = net.get_parents(n)
        if len(parents) == 0:
            # If the node has no parents, there are (num_states - 1) independent parameters
            total_independent_parameters += (num_states - 1)
        else:
            parent_configurations = 1
            for parent in parents:
                parent_configurations *= net.get_outcome_count(parent)
            # Number of independent parameters for this node
            total_independent_parameters += parent_configurations * (num_states - 1)

        n = net.get_next_node(n)

    return total_independent_parameters


def get_parameter_nodes(net):
    """Return all leaf nodes from the network except for GATEWAY"""
    parameter_nodes = []
    n = net.get_first_node()
    while n >= 0:
        # Check if the node has no children and if it's not GATEWAY
        if not net.get_children(n) and not net.get_node_id(n) == "GATEWAY":
            parameter_nodes.append(n)
        n = net.get_next_node(n)
    return parameter_nodes


def get_system_nodes(net):
    """Return all leaf nodes from the network except for GATEWAY"""
    system_nodes = []
    parameter_nodes = get_parameter_nodes(net)
    n = net.get_first_node()
    while n >= 0:
        # Check if the node has no children and if it's not GATEWAY
        if n not in parameter_nodes:
            system_nodes.append(n)
        n = net.get_next_node(n)
    return system_nodes


def get_all_nodes(net):
    """Return all nodes from the network"""
    system_nodes = []
    parameter_nodes = get_parameter_nodes(net)
    n = net.get_first_node()
    while n >= 0:
        parameter_nodes.append(n)
        n = net.get_next_node(n)
    return parameter_nodes


def generate_data(net, num_records, file_path, node_type):
    """Generate parameter data from the Bayesian network and save it to a text file."""
    # Get the parameter nodes
    if node_type == "parameters":
        nodes = get_parameter_nodes(net)
    elif node_type == "systems":
        nodes = get_system_nodes(net)
    elif node_type == "all":
        nodes = get_all_nodes(net)

    with open(file_path, 'w') as file:
        # Write headers representing node names
        headers = [net.get_node_name(node_id) for node_id in nodes]
        file.write(' '.join(headers) + '\n')

        # Generate data records
        for _ in range(num_records):
            record = []
            for node_id in nodes:
                node_name = net.get_node_name(node_id)
                # Get the node state based on conditional probabilities
                num_outcomes = net.get_outcome_count(node_id)
                probabilities = net.get_node_definition(node_id)[:num_outcomes]  # Trim probabilities list
                state_index = random.choices(range(num_outcomes), probabilities)[0]
                state_name = net.get_outcome_id(node_id, state_index)
                record.append(state_name)
            file.write(' '.join(record) + '\n')


def get_net_performance(net):
    n = net.get_first_node()
    Utility = 0
    Observability = 0
    Knowledge = 0

    while n >= 0:
        if net.get_node_id(n) in metrics.utility :
            Utility += metrics.utility[net.get_node_id(n)]
        elif net.get_node_id(n) in metrics.observability :
            Observability += metrics.observability[net.get_node_id(n)]
        n = net.get_next_node(n)

    net.write_file("net_copy_perf.xdsl")  # Required to update node numbering
    net.read_file("net_copy_perf.xdsl")  # Required to update node numbering

    non_cut_nodes, non_cut_arcs, cut_arcs = non_cut_entities(net)
    arcs = non_cut_arcs + cut_arcs

    generate_data(net, num_records=1000, file_path="modified_data.txt", node_type="parameters")  # previously 200
    kl_div = kl_divergence.compute_kl("original_data.txt", "modified_data.txt")

    for arc in arcs:
        key = (net.get_node_name(arc[0]), net.get_node_name(arc[1]))
        if key in metrics.knowledge:
            Knowledge += metrics.knowledge[key]

    BN_parameters = get_number_of_independent_parameters(net)

    U = Utility
    O = Observability
    K = Knowledge
    kl = (10**3)*kl_div
    P_count = 0.1*BN_parameters

    performance = U + O + K - kl - P_count

    print(f"Perf: {int(performance)} = U:{int(U)} + O:{int(O)} + K:{int(K)} - KL:{int(kl)} - P_count:{int(P_count)}")

    return performance
