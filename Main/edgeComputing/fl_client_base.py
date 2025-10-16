import flwr as fl
import torch
# Import the NEW baseline function we will create
from client import train_one_round_baseline, evaluate_model 
from model import CNNModel
import os

client_id = int(os.getenv("CLIENT_ID", 0))
print(f"[Baseline Client] Running client {client_id}")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = CNNModel()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        results_dir = os.getenv("RESULTS_DIR", "./results")

        # Call the new, simple baseline function
        new_state_dict, num_samples, metrics = train_one_round_baseline(
            self.model.state_dict(),
            client_id=client_id,
            results_dir=results_dir
        )
        
        self.model.load_state_dict(new_state_dict)
        return self.get_parameters({}), int(num_samples), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = evaluate_model(self.model.state_dict())
        return 0.0, 1000, metrics

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())