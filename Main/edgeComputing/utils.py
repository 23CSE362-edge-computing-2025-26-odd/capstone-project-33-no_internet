import os
import json

def save_log(log_dict, folder="logs", filename="training_log.json"):
    """
    Save round-wise training logs as JSON.
    """
    os.makedirs(folder, exist_ok=True)  # create folder if it doesn't exist
    filepath = os.path.join(folder, filename)
    # If file exists, append the new log; else create
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(log_dict)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Training log updated at {filepath}")