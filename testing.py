import random
from graph import *
import os
import matplotlib.pyplot as plt
import numpy as np


def filter_observations(net):
    # Access the observability dictionary from the metrics module
    from metrics import observability

    # Read the observations.txt file
    with open('observations.txt', 'r') as file:
        lines = file.readlines()

    # Extract the header and the data
    header = lines[0].strip().split()
    data_lines = lines[1:]

    # Initialize a list to store the filtered data
    filtered_data = []

    # Iterate over each data line
    for line in data_lines:
        data = line.strip().split()
        filtered_line = []

        # Iterate over each entry in the row
        for i, entry in enumerate(data):
            system_node_name = header[i]

            # Get the node handle for the system node
            system_node = net.get_node(system_node_name)

            # Check if the system node is valid
            if system_node >= 0:
                # Modify the state based on the observability probability
                if system_node_name in observability:
                    if random.random() > observability[system_node_name]:
                        # Get the state names for the current system node
                        state_names = net.get_outcome_ids(system_node)
                        state_names.remove(entry)  # Remove the current state

                        # Pick a random state that is not the same as the current entry
                        new_state = random.choice(state_names)
                        filtered_line.append(new_state)
                    else:
                        filtered_line.append(entry)
                else:
                    filtered_line.append(entry)
            else:
                # Handle the case where the system node is invalid
                filtered_line.append(entry)

        # Add the filtered line to the filtered_data list
        filtered_data.append(filtered_line)

    # Write the filtered data to a new file called "filtered_observations.txt"
    with open('filtered_observations.txt', 'w') as file:
        file.write(' '.join(header) + '\n')
        for line in filtered_data:
            file.write(' '.join(line) + '\n')

    # print("Filtering complete. The filtered data has been written to 'filtered_observations.txt'.")


def sample_states(net_true):
    # Generate data and filter observations
    generate_data(net_true, num_records=1, file_path="observations.txt", node_type="systems")
    filter_observations(net_true)


def infer_parameters(net, pruned):
    # Read filtered_observations.txt
    with open('filtered_observations.txt', 'r') as file:
        lines = file.readlines()

    # Extract the header and the data
    header = lines[0].strip().split()
    data_lines = lines[1:]

    # Dictionary to store beliefs for parameter nodes
    parameter_beliefs = {}
    # Dictionary to store evidence
    evidence = {}

    # Iterate over each data line
    for line in data_lines:
        data = line.strip().split()

        # Iterate over each entry
        for i, entry in enumerate(data):
            node_id = header[i]

            # If the value is not "nominal", set evidence and update beliefs
            if entry != "nominal":
                try:
                    net.set_evidence(node_id, entry)
                    # Add to evidence dictionary
                    evidence[node_id] = entry
                except pysmile.SMILEException as e:
                    # print(f"Skipping node {node_id} as it was pruned")
                    pass
                # if pruned is False:
                #     print(f"Evidence: {node_id}; {entry}")

        # Update beliefs after setting evidence
        net.update_beliefs()

        # Store updated beliefs for each parameter node
        parameter_nodes = get_parameter_nodes(net)
        for n in parameter_nodes:
            beliefs = net.get_node_value(n)
            most_likely_index = beliefs.index(max(beliefs))
            outcome_id = net.get_outcome_id(n, most_likely_index)
            belief_value = round(beliefs[most_likely_index], 2)
            parameter_beliefs[net.get_node_name(n)] = (outcome_id, belief_value)

    return parameter_beliefs, evidence


def get_latest_file(folder_path):
    # Get a list of all files in the folder that match the pattern
    files = [f for f in os.listdir(folder_path) if f.startswith("net_pruned_") and f.endswith(".xdsl")]

    if not files:
        # If no matching files found, return None
        return None

    # Extract the numbers from the filenames
    numbers = [int(f.split("_")[2].split(".")[0]) for f in files]

    # Find the file with the highest number
    latest_number = max(numbers)
    latest_file = f"net_pruned_{latest_number}.xdsl"
    return latest_file


def get_query(evidence_true, net):
    parameters_queried = []

    parameter_nodes = get_parameter_nodes(net)

    for node in evidence_true.keys():
        children = net.get_children(node)

        for child in children:
            if child in parameter_nodes:
                parameters_queried.append(net.get_node_name(child))

    return parameters_queried


def compare(parameters_queried, beliefs_true, beliefs_baseline, beliefs_proposed):
    baseline_correct = 0
    baseline_incorrect = 0
    proposed_correct = 0
    proposed_incorrect = 0

    for parameter in parameters_queried:
        true_value = beliefs_true[parameter][0]

        # Check baseline values
        if parameter in beliefs_baseline:
            baseline_value = beliefs_baseline[parameter][0]
            if baseline_value == true_value:
                baseline_correct += 1
            else:
                baseline_incorrect += 1
                print(f"incorrect baseline: {parameter}")
        else:
            baseline_incorrect += 1
            print(f"missing baseline: {parameter}")

        # Check proposed values
        if parameter in beliefs_proposed:
            proposed_value = beliefs_proposed[parameter][0]
            if proposed_value == true_value:
                proposed_correct += 1
            else:
                proposed_incorrect += 1
                print(f"incorrect proposed: {parameter}")
        else:
            proposed_incorrect += 1
            print(f"missing proposed: {parameter}")

    return baseline_correct, baseline_incorrect, proposed_correct, proposed_incorrect


def assess(net_true, net_pruned_baseline, net_pruned_proposed, diagnosis_count):

    query_count = 0
    total_b_correct = 0
    total_p_correct = 0

    for i in range(diagnosis_count):

        sample_states(net_true)

        beliefs_true, evidence_true = infer_parameters(net_true, pruned=False)
        # print(f"evidence_true: {evidence_true}")
        beliefs_baseline, evidence_baseline = infer_parameters(net_pruned_baseline, pruned=True)
        beliefs_proposed, evidence_proposed = infer_parameters(net_pruned_proposed, pruned=True)

        parameters_queried = get_query(evidence_true, net_true)

        baseline_correct, baseline_incorrect, proposed_correct, proposed_incorrect =\
            compare(parameters_queried, beliefs_true, beliefs_baseline, beliefs_proposed)

        query_count += len(parameters_queried)
        total_b_correct += baseline_correct
        total_p_correct += proposed_correct

    return query_count, total_b_correct, total_p_correct


def plot_accuracy(data_dict):
    """
    Plot a clustered column chart from a dictionary of data.

    Parameters:
        data_dict (dict): A dictionary where keys are clusters and values are tuples of data for each cluster.
    """
    # Extract keys and values from the dictionary
    keys = list(data_dict.keys())
    values = list(data_dict.values())

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Define the width of each bar
    bar_width = 0.35

    # Define the index for the x-axis
    index = range(len(keys))

    # Define lighter colors
    lighter_colors = [(0.6, 0.8, 0.6), (1.0, 0.8, 0.6)]

    # Plot the bars for each cluster
    for i in range(len(values[0])):
        if i == 0:
            label = 'Proposed'
        else:
            label = 'Baseline'
        ax.bar([x + i * bar_width for x in index], [v[i] for v in values], bar_width, label=label, color=lighter_colors[i])
        # Add annotations for the value of each column
        for j, v in enumerate(values):
            ax.text(index[j] + i * bar_width, v[i] / 2, f'{v[i]:.2f}', ha='center', va='center', color='black')

    # Set labels and title
    ax.set_xlabel('Target Reduction')
    ax.set_ylabel('Accuracy')
    ax.set_xticks([x + bar_width / 2 for x in index])
    ax.set_xticklabels(keys)
    ax.legend()

    plt.savefig(f'accuracy.png')

    # Show the plot
    plt.show()
