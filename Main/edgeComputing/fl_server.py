import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters


## lsof -ti :8080 <- this is for killing the process running on port 8080
## kill -9 <PID> <- this is for killing the process with the given PID
# Import the necessary components from your other files
from model import CNNModel
from custom_strategy import SmartSchedulingStrategy

# --- Step 1: Load Initial Model Parameters ---
# The custom strategy needs to know the starting point of the global model.
# We load it from the same file your BFT strategy was using.
print("--- Loading initial global model for the strategy ---")
initial_model = CNNModel()

# This path is relative to the project's root folder (YAFS Simulation),
# which is where you should run this script from.
weights_path = "src/current_weights.pth"
try:
    # We need to explicitly tell torch.load it's safe because the file was saved in a different structure.
    # This is a standard security practice in newer PyTorch versions.
    initial_model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    print(f"Successfully loaded initial weights from: {weights_path}")
except FileNotFoundError:
    print(f"WARNING: Initial weights file not found at '{weights_path}'. The server will start with random model weights.")
except Exception as e:
    print(f"An error occurred loading weights: {e}. Starting with random model weights.")

# Convert the PyTorch model's state_dict into Flower's Parameters format
initial_parameters = ndarrays_to_parameters(
    [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
)


if __name__ == "__main__":
    # --- Step 2: Instantiate Your Custom Strategy ---
    # Create an instance of your SmartSchedulingStrategy, passing the
    # initial model parameters it requires.
    strategy = SmartSchedulingStrategy(
        initial_parameters=initial_parameters,
        k_bft=900.0,
        # You can also configure FedAvg parameters here, for example:
        min_fit_clients=2,        # Minimum clients to train in a round
        min_available_clients=2,  # Wait for at least this many clients to be connected
        fraction_fit=1.0,         # Use 100% of available clients for training
    )

    # --- Step 3: Start the Server with the Custom Strategy ---
    # Instead of the default server, we tell Flower to use our custom strategy
    # which contains all the BFT+PSO logic.
    print("\n--- Starting Flower server with SMART Scheduling Strategy ---")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=5), # Set the total number of rounds
        strategy=strategy # <-- This is the crucial part
    )

    print("--- Server finished ---")
