import os
import subprocess
import itertools

# Hyperparameter grid
learning_rates = [0.01, 0.001]
num_layers = [2, 3]
batch_sizes = [64, 128, 256]

# Iterate over all combinations
for lr, nl, bs in itertools.product(learning_rates, num_layers, batch_sizes):
    env = os.environ.copy()
    env["learning_rate"] = str(lr)
    env["numLayers"] = str(nl)
    env["batch_size"] = str(bs)

    print(f"\nRunning: lr={lr}, numLayers={nl}, batch_size={bs}")

    subprocess.run(
        ["python3.11", "main_new.py"],
        env=env,
        check=True
    )
