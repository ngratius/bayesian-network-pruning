import pysmile
import pysmile_license
import metrics
import graph
import kl_divergence
import plot
import os
import glob
import shutil
import random


from testing import *
from graph import *


def get_best_kl_arc(net, candidate_arcs):
    # Initialize with placeholder values
    smallest_kl_div = 100
    best_arc = candidate_arcs[0]

    for index, arc in enumerate(candidate_arcs):
        net.write_file("net_copy.xdsl")  # Save network before trial pruning
        net.delete_arc(arc[0], arc[1])

        graph.generate_data(net, num_records=1000, file_path="modified_data.txt", node_type="all")
        kl_div = kl_divergence.compute_kl("original_data.txt", "modified_data.txt")
        # print(kl_div)

        # print(
        #     f"\rArc tested: {index + 1}/{len(candidate_arcs)} ({net.get_node_id(arc[0])} -> {net.get_node_id(arc[1])})"
        #     , end="")

        if kl_div < smallest_kl_div:
            smallest_kl_div = kl_div
            best_arc = arc

        net.read_file("net_copy.xdsl")  # Reset network after trial pruning
    return best_arc, smallest_kl_div


def get_best_kl_node(net, candidate_nodes):
    # Initialize with placeholder values
    smallest_kl_div = 1000
    best_node = 0

    for index, node in enumerate(candidate_nodes):
        # print(f"\rNode tested: {index + 1}/{len(candidate_nodes)} ({net.get_node_id(node)})", end="")
        net.write_file("net_copy.xdsl")  # Save network before trial pruning
        net.delete_node(node)

        graph.generate_data(net, num_records=1000, file_path="modified_data.txt", node_type="all")
        kl_div = kl_divergence.compute_kl("original_data.txt", "modified_data.txt")
        # print(kl_div)

        if kl_div < smallest_kl_div:
            smallest_kl_div = kl_div
            best_node = node

        net.read_file("net_copy.xdsl")  # Reset network after trial pruning

    return best_node, smallest_kl_div


def get_best_perf_node(net, candidate_nodes):
    # Initialize with placeholder values
    largest_performance = 0
    best_node = 0

    for index, node in enumerate(candidate_nodes):
        # print(f"\rNode tested: {index + 1}/{len(candidate_nodes)} ({net.get_node_id(node)})", end="")
        print(f"Node tested: ({net.get_node_id(node)})")
        net.write_file("net_copy.xdsl")  # Save network before trial pruning
        net.delete_node(node)

        performance = graph.get_net_performance(net)

        if performance > largest_performance:
            largest_performance = performance
            best_node = node

        net.read_file("net_copy.xdsl")  # Reset network after trial pruning

    return best_node, largest_performance


def get_best_perf_arc(net, candidate_arcs):
    # Initialize with placeholder values
    largest_performance = 0
    best_arc = candidate_arcs[0]

    for index, arc in enumerate(candidate_arcs):
        net.write_file("net_copy.xdsl")  # Save network before trial pruning
        net.delete_arc(arc[0], arc[1])

        # print(
        #     f"\rArc tested: {index + 1}/{len(candidate_arcs)} ({net.get_node_id(arc[0])} -> {net.get_node_id(arc[1])})"
        #     , end="")

        print(f"Arc tested: ({net.get_node_id(arc[0])} -> {net.get_node_id(arc[1])})")
        performance = graph.get_net_performance(net)

        if performance > largest_performance:
            largest_performance = performance
            best_arc = arc

        net.read_file("net_copy.xdsl")  # Reset network after trial pruning
    return best_arc, largest_performance


def pruning(net_zero, target, method):

    performance_scores = []
    performance_scores.append(graph.get_net_performance(net_zero))

    parameter_counts = []
    parameter_counts.append(graph.get_number_of_independent_parameters(net_zero))  # Compute target

    parameter_target = int(target * parameter_counts[0])

    print(f"Target reduction : {parameter_counts[0]} parameters -> {parameter_target} parameters")
    print(f"Initial performance score: {performance_scores[0]:.2f}\n")

    pruning_iteration = 0

    # Clear repo
    if method == "baseline":
        [os.remove(file) for file in glob.glob(os.path.join("iterations_baseline", '*')) if os.path.isfile(file)]
    elif method == "proposed":
        [os.remove(file) for file in glob.glob(os.path.join("iterations_proposed", '*')) if os.path.isfile(file)]

    # Initialize
    net = pysmile.Network()
    net.read_file("g_zero.xdsl")

    while parameter_counts[-1] > parameter_target:
        pruning_iteration += 1

        candidate_nodes, candidate_arcs, cut_arcs = graph.non_cut_entities(net)

        if method == "baseline":

            if len(candidate_arcs) > 0:
                best_arc, kl_div_arc = get_best_kl_arc(net, candidate_arcs)
                net.delete_arc(best_arc[0], best_arc[1])
                print(f"\rPruned arc: {net.get_node_id(best_arc[0])} -> {net.get_node_id(best_arc[1])}")
            else:  # All non-cut arcs have been pruned; Starting to prune nodes based on kl divergence
                best_node, smallest_kl_div = get_best_kl_node(net, candidate_nodes)
                print(f"\rPruned node: {net.get_node_id(best_node)}")
                net.delete_node(best_node)

            net.write_file(os.path.join("iterations_baseline", f"net_pruned_{pruning_iteration}.xdsl"))

        elif method == "proposed":

            best_node, perf_node = get_best_perf_node(net, candidate_nodes)
            if len(candidate_arcs) == 0:
                print(f"\r> Pruned node: {net.get_node_id(best_node)}")
                net.delete_node(best_node)
            else:
                best_arc, perf_arc = get_best_perf_arc(net, candidate_arcs)
                if perf_arc > perf_node:
                    print(f"\rPruned arc: {net.get_node_id(best_arc[0])} -> {net.get_node_id(best_arc[1])}")
                    net.delete_arc(best_arc[0], best_arc[1])
                else:
                    print(f"\rPruned node: {net.get_node_id(best_node)}")
                    net.delete_node(best_node)

            net.write_file(os.path.join("iterations_proposed", f"net_pruned_{pruning_iteration}.xdsl"))

        parameter_counts.append(graph.get_number_of_independent_parameters(net))
        net.write_file("net_copy.xdsl")  # Required to update node numbering
        net.read_file("net_copy.xdsl")  # Required to update node numbering
        performance_scores.append(graph.get_net_performance(net))

        print(f"> New parameter count: {parameter_counts[-1]}, New performance score: {performance_scores[-1]:.2f}")

    if method == "baseline":
        net.write_file(os.path.join("iterations_final", f"{target}_net_pruned_b_{pruning_iteration}.xdsl"))
    elif method == "proposed":
        net.write_file(os.path.join("iterations_final", f"{target}_net_pruned_p_{pruning_iteration}.xdsl"))

    return net, parameter_counts, performance_scores, parameter_target


def main():

    # Define lists to store results for both methods
    parameter_counts = [[], []]
    performance_scores = [[], []]

    # Initial network
    net_zero = pysmile.Network()
    net_zero.read_file("g_zero.xdsl")

    results = {}

    [os.remove(file) for file in glob.glob(os.path.join("iterations_final", '*')) if os.path.isfile(file)]

    graph.generate_data(net_zero, num_records=10000, file_path="original_data.txt", node_type="parameters")

    for t in [0.95, 0.9, 0.8]:

        # PRUNING

        print(f"\nBASELINE PRUNING: Target is {t}\n")
        net_zero.read_file("g_zero.xdsl")
        net_b, parameter_counts[0], performance_scores[0], parameter_target_b = pruning(net_zero, target=t, method="baseline")

        print(f"\nPROPOSED PRUNING: Target is {t}\n")
        net_zero.read_file("g_zero.xdsl")
        net_p, parameter_counts[1], performance_scores[1], parameter_target_p = pruning(net_zero, target=t, method="proposed")

        plot.plot(parameter_counts, performance_scores, parameter_target_p)

        net_pruned_baseline = pysmile.Network()
        net_pruned_baseline.read_file("iterations_baseline/" + get_latest_file("iterations_baseline"))

        net_pruned_proposed = pysmile.Network()
        net_pruned_proposed.read_file("iterations_proposed/" + get_latest_file("iterations_proposed"))

        plt.savefig(f'verification_{str(t)}.png')
        # TESTING

        net_true = pysmile.Network()
        net_true.read_file("g_true.xdsl")
        query_count, b_correct, p_correct = assess(net_true, net_pruned_baseline, net_pruned_proposed, diagnosis_count=1000)

        results[t] = query_count, b_correct, p_correct

        print(f"RESULT {t}: Proposed:{p_correct/query_count} Baseline:{b_correct/query_count}")

    return results


# net = pysmile.Network()

results = main()
print(f"results: {results}")

accuracies = {}

for key, value in results.items():
    # (accuracy of proposed, accuacy of baseline)
    updated_value = (value[2] / value[0], value[1] / value[0])
    accuracies[key] = updated_value

print(f"success_rates: {accuracies}")

plot_accuracy(accuracies)

