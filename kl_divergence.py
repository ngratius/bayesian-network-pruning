from collections import defaultdict
import numpy as np
import math


def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    header = lines[0].strip().split()  # Extract header
    data = [line.strip().split() for line in lines[1:]]  # Extract data
    return header, data


def compute_joint_probability(data):
    joint_prob = defaultdict(float)  # Update to accept float values
    total_samples = len(data)

    for row in data:
        joint_prob[tuple(row)] += 1

    # Normalize probabilities
    for key in joint_prob:
        joint_prob[key] = round(joint_prob[key] / total_samples, 10)

    # Ensure the total probability sums up to exactly 1
    total_prob = sum(joint_prob.values())
    if total_prob != 1:
        # Adjust the last probability to compensate for rounding
        last_key = next(reversed(joint_prob))  # Get the last key
        joint_prob[last_key] += 1 - total_prob

    return joint_prob


def common_columns(header1, header2):
    return list(set(header1) & set(header2))


# def filter_data(data, header, common_cols):
#     filtered_data = []
#     for row in data:
#         filtered_row = [row[i] for i, col in enumerate(header) if col in common_cols]
#         filtered_data.append(filtered_row)
#     return filtered_data

def filter_data(data, header, common_cols):
    # Filter the header to keep only the common columns
    filtered_header = [col for col in header if col in common_cols]

    # Find the indices of these columns in the original headers
    indices = [header.index(col) for col in filtered_header]

    # Order the filtered header based on the order of common_cols
    ordered_header = sorted(filtered_header, key=lambda col: common_cols.index(col))

    # Create a map from old index to new index
    old_to_new_index = {header.index(col): ordered_header.index(col) for col in ordered_header}

    # Reorder and filter the data based on the indices and old_to_new_index map
    ordered_data = []
    for row in data:
        ordered_row = [None] * len(ordered_header)
        for old_index, new_index in old_to_new_index.items():
            ordered_row[new_index] = row[old_index]
        ordered_data.append(ordered_row)

    return ordered_data, ordered_header


def kl_divergence(p, q):
    epsilon = 1e-10  # Small value to avoid logarithm of zero
    kl_div = 0
    for key, value in p.items():
        if key in q:
            if value == 0:
                continue  # Skip term if probability is zero
            kl_div += value * np.log2((value + epsilon) / (q[key] + epsilon))
            # print("kl_update")
    return kl_div


def normalize_probabilities(joint_prob):
    total_prob = sum(joint_prob.values())
    normalized_prob = {k: v / total_prob for k, v in joint_prob.items()}
    return normalized_prob


def compute_kl(file1, file2):  # E.g., file1 = "full_bn.txt", file2 = "pruned_bn.txt"

    header1, data1 = read_file(file1)
    header2, data2 = read_file(file2)

    common_cols = common_columns(header1, header2)

    filtered_data1, filtered_header1 = filter_data(data1, header1, common_cols)
    filtered_data2, filtered_header2 = filter_data(data2, header2, common_cols)

    joint_prob1 = compute_joint_probability(filtered_data1)
    joint_prob2 = compute_joint_probability(filtered_data2)

    total_prob1 = sum(joint_prob1.values())
    total_prob2 = sum(joint_prob2.values())

    # print("Joint Probability 1:", joint_prob1)
    # print("Total Probability 1:", total_prob1)
    # print("Joint Probability 2:", joint_prob2)
    # print("Total Probability 2:", total_prob2)

    kl_div = kl_divergence(joint_prob1, joint_prob2)

    # print("KL Divergence between the two joint probabilities:", kl_div)

    # if kl_div == 0:
    #     raise "KL Divergence is zero. Try increasing the number of samples."

    return abs(kl_div)







