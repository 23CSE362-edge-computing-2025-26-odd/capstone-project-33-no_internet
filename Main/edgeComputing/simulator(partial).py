import torch
import os
import random
import subprocess
import hashlib
from client import Client
from server import Server
from dataset import get_mnist_loaders
from partition import non_iid_partition
from compression import topk_sparsify, should_skip
from model import load_initial_weights
from model import CNNModel, save_initial_weights

# --- Global Paths (relative to project root) ---
CLIENT_UPDATES_FOLDER = "client_updates"
os.makedirs(CLIENT_UPDATES_FOLDER, exist_ok=True)
EDITOR_LOG_PATH = "../edgeComputing/editor_log.txt"
GLOBAL_WEIGHTS_PATH = "../src/current_weights.pth"


class Simulator:
    def __init__(self, num_clients=3, rounds=200, local_epochs=1):
        self.num_clients = num_clients
        self.rounds = rounds
        self.local_epochs = local_epochs

        train_loader, _ = get_mnist_loaders()
        train_dataset = train_loader.dataset
        client_subsets = non_iid_partition(train_dataset, num_clients=self.num_clients)

        self.clients = [
            Client(i, data_subset=client_subsets[i], local_epochs=self.local_epochs)
            for i in range(self.num_clients)
        ]

        self.server = Server(num_clients=self.num_clients)
        self.server.model = CNNModel()

    def simulate_bandwidth(self):
        return [random.uniform(0.01, 0.1) for _ in range(self.num_clients)]

    def save_client_update(self, client):
        filepath = os.path.join(CLIENT_UPDATES_FOLDER, f"client{client.client_id}.pth")
        torch.save(client.model.state_dict(), filepath)
        print(f"[Simulator] Client {client.client_id} updated weights saved at {filepath}")

    def write_editor_log(self, allowed_clients):
        os.makedirs(os.path.dirname(EDITOR_LOG_PATH), exist_ok=True)
        with open(EDITOR_LOG_PATH, "w") as f:
            for cid in allowed_clients:
                f.write(f"client{cid}\n")
        print(f"[Simulator] editor_log.txt updated with allowed clients: {allowed_clients}")

    def get_weights_hash(self, filepath):
        """Computes the SHA256 hash of a file to check for changes."""
        if not os.path.exists(filepath):
            return None
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def run(self):
        # Get initial weights for comparison (only once before all rounds)
        load_initial_weights(self.server.model, filepath=GLOBAL_WEIGHTS_PATH)
        initial_weights = {name: param.clone() for name, param in self.server.model.named_parameters()}

        for round_num in range(1, self.rounds + 1):
            print(f"\n--- Round {round_num} ---")

            # --- Per-round logic for file hash check ---
            initial_hash = self.get_weights_hash(GLOBAL_WEIGHTS_PATH)

            allowed_clients = []
            bandwidths = self.simulate_bandwidth()

            for client, bw in zip(self.clients, bandwidths):
                print(f"\n[Simulator] Client {client.client_id} bandwidth: {bw:.3f} MB")
                load_initial_weights(client.model, filepath=GLOBAL_WEIGHTS_PATH)
                client.train()

                update_size = 1.0
                if should_skip(bw, update_size):
                    print(f"[Simulator] Client {client.client_id} skipped due to low bandwidth")
                    continue

                self.save_client_update(client)
                allowed_clients.append(client.client_id)

            self.write_editor_log(allowed_clients)

            print(f"\n[Simulator] Round {round_num} complete. Starting BFT+PSO screening...")

            command = ['python', '../ci/bft_pso_strategy.py']
            try:
                subprocess.run(command, check=True)
                print("[Simulator] BFT+PSO screening and aggregation complete.")
            except subprocess.CalledProcessError as e:
                print(f"[Simulator] Error during BFT+PSO execution: {e}")
                print(f"Stdout: {e.stdout.decode('utf-8')}")
                print(f"Stderr: {e.stderr.decode('utf-8')}")
                break

            # --- Per-round logic for file hash check ---
            final_hash = self.get_weights_hash(GLOBAL_WEIGHTS_PATH)

            print(f"\n[Verification] Global weights file hash before round {round_num}: {initial_hash}")
            print(f"[Verification] Global weights file hash after round {round_num}: {final_hash}")

            if initial_hash and initial_hash != final_hash:
                print(f"[Verification] The global model was successfully updated in round {round_num}. 🎉")
            else:
                print(f"[Verification] No change detected in the global model after round {round_num}. 😔")

        # --- FINAL CALCULATION AFTER ALL ROUNDS ---
        print(f"\n--- Final Summary ---")

        # Get final weights after aggregation of the last round
        load_initial_weights(self.server.model, filepath=GLOBAL_WEIGHTS_PATH)
        final_weights = {name: param.clone() for name, param in self.server.model.named_parameters()}

        # Calculate the magnitude of change using L2 Norm
        total_change_magnitude = 0.0
        for name in initial_weights:
            if name in final_weights:
                diff = final_weights[name] - initial_weights[name]
                total_change_magnitude += torch.norm(diff, p=2).item() ** 2

        total_change_magnitude = total_change_magnitude ** 0.5

        print(f"Total magnitude of change across all {self.rounds} rounds (L2 norm): {total_change_magnitude:.6f}")


if __name__ == "__main__":
    sim = Simulator(num_clients=3, rounds=100, local_epochs=1)
    sim.run()
