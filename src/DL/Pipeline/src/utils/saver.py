import os
import json

def save_results(config, metrics):

    path = os.path.join("logs", config.EXPERIMENT_NAME)
    os.makedirs(path, exist_ok=True)

    # save config
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)

    # save metrics
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)