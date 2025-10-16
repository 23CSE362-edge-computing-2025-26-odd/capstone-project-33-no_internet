import flwr as fl
import torch
from client import train_one_round, evaluate_model
from model import CNNModel
import os
import random
client_id = int(os.getenv("CLIENT_ID", 0))
print(f"[Client] Running client {client_id}")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = CNNModel()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(),
                              [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
    # set current global params
        self.set_parameters(parameters)

    # try to get client info from config (fallback to env vars)
        client_id = int(config.get("client_id")) if config and "client_id" in config else int(os.environ.get("CLIENT_ID", "0"))
        results_dir = config.get("results_dir", os.environ.get("RESULTS_DIR", "./results"))
        '''bw = None
        if config and "bw_mbps" in config:
            try:
                bw = float(config["bw_mbps"])
            except Exception:
                bw = None'''
        # Read bandwidth directly from the environment variable set by YAFS
        '''bw_str = os.getenv("BW_MBPS")
        bw = float(bw_str) if bw_str is not None else None
        print(f"[Client {client_id}] Using bandwidth = {bw} Mbps")'''
        bw = random.uniform(0.1, 15.0)
        print(f"[Client {client_id}] Current round bandwidth = {bw:.2f} Mbps")

    # run one local training round with decision logic (skip/compress/full)
        new_state_dict, num_samples, metrics = train_one_round(
            self.model.state_dict(),
            epochs=1,
            client_id=client_id,
            results_dir=results_dir,
            bw_mbps=bw
        )
        #print(f"[Client {client_id}] Bandwidth = {os.getenv('BW_MBPS', 'not set')} Mbps")
    # If train_one_round returned None (skip), use current params and report zero samples
        if new_state_dict is None:
            # no update sent this round
            return self.get_parameters({}), int(num_samples), {"decision": "skip"}

    # otherwise update local model and return parameters to server
        self.model.load_state_dict(new_state_dict)
        return self.get_parameters({}), int(num_samples), metrics

    '''def fit(self, parameters, config):
        self.set_parameters(parameters)
        new_state_dict, num_samples, metrics = train_one_round(self.model.state_dict(), epochs=1)
        self.model.load_state_dict(new_state_dict)
        return self.get_parameters({}), num_samples, metrics'''

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = evaluate_model(self.model.state_dict())
        return 0.0, 1000, metrics  # loss placeholder

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
