import itertools
from py_script_exec import py_run
import json


# Function to execute the simple_cnn.py script with given parameters
def run_cnn_script(epochs, lr, batch_size):
    args = [
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--batch_size",
        str(batch_size),
        "--train",
    ]

    # Execute the script using py_run from py_script_exec
    exit_code, stdout, stderr = py_run("simple_cnn.py", args)

    # Return results
    return exit_code, stdout, stderr


# List of different hyperparameters to try
epochs_list = [30]
lr_list = [0.0001, 0.001, 0.01]
batch_size_list = [16, 32, 64]

# Generate all combinations of hyperparameters using itertools.product
param_combinations = itertools.product(epochs_list, lr_list, batch_size_list)

# List to store results
all_results = []

# Loop over each combination of parameters
for epochs, lr, batch_size in param_combinations:
    print(f"Running with epochs={epochs}, lr={lr}, batch_size={batch_size}")

    # Run the CNN script with the current parameters
    exit_code, stdout, stderr = run_cnn_script(epochs, lr, batch_size)

    # Store the result, including exit code and outputs
    result = {
        "Epochs": epochs,
        "Learning Rate": lr,
        "Batch Size": batch_size,
        "Exit Code": exit_code,
        "Stdout": stdout,
        "Stderr": stderr,
    }
    all_results.append(result)

    # Print the output of the script
    print(f"Exit Code: {exit_code}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")

# Write all results to a JSON file for later analysis
with open("cnn_training_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("Finished running all experiments.")
