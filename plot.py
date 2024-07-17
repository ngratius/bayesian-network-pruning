import matplotlib.pyplot as plt


def plot(parameter_counts, performance_scores, parameter_target):
    # Create the plot
    plt.figure(figsize=(7, 6))
    plt.rcParams.update({'font.size': 14})

    # Define labels and colors for each method
    methods = ['Baseline', 'Proposed']
    colors = ['b', 'g']

    for j in range(len(parameter_counts)):
        # Plot the performance_scores against parameter_counts for each method
        labels = ["Start"] + [f"{'p' if j == 1 else 'b'}{i}" for i in range(1, len(parameter_counts[j]))]  # Updated label format
        for i, label in enumerate(labels):
            plt.plot(parameter_counts[j][i], performance_scores[j][i], marker='o', linestyle='', color=colors[j])
            plt.text(parameter_counts[j][i], performance_scores[j][i], label, fontsize=14, color=colors[j], ha='left', va='center',
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

        # Plot the lines between points
        plt.plot(parameter_counts[j], performance_scores[j], linestyle='-', color=colors[j], label=methods[j])

    # Plot the vertical line for parameter_target
    plt.axvline(x=parameter_target, color='r', linestyle='--', label='Computability target')

    # Set labels and title
    plt.xlabel('Parameter Count (Decreasing)', fontsize=14)
    plt.ylabel('Performance Scores (Increasing)', fontsize=14)

    # Invert the x-axis
    plt.gca().invert_xaxis()

    # Add legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    # plt.figure(figsize=(8, 6))

    # plt.show()