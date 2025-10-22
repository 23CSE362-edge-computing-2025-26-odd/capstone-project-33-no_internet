# Main/edgeComputing/custom_strategy.py

import logging
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, Subset
from torchvision import datasets, transforms
from dataclasses import dataclass, field
from enum import Enum

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.server import FitRes, ClientProxy
from flwr.common import (
    Parameters,
    FitIns,
    FitRes,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

# Configure logging (optional, but helpful)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- PSO Parameters ---
W = 0.5  # Inertia weight
C1 = 2.0  # Cognitive coefficient
C2 = 2.0  # Social coefficient

# --- Enums and Dataclasses (Copied from bft_pso_strategy.py) ---
class ClientStatus(Enum):
    ACTIVE = "active"
    FILTERED_BFT = "filtered_bft"
    FILTERED_NO_CHANGE = "filtered_no_change"
    SELECTED = "selected"
    ERROR = "error"

@dataclass
class ClientMetrics:
    accuracy: float = 0.0
    false_positive_rate: float = 0.0 # Note: Placeholder, may not be available from client
    response_time: float = 0.0 # Corresponds to train_time_s
    last_update: float = 0.0 # Timestamp of when update was received (can be added in aggregate_fit)

@dataclass
class ClientResult:
    client_id: str
    original_client_id: str
    parameters: Dict[str, torch.Tensor]  # Use state_dict format
    num_examples: int
    metrics: ClientMetrics
    status: ClientStatus = ClientStatus.ACTIVE

# --- CNN Model Definition (Copied from bft_pso_strategy.py) ---
class CNNModel(nn.Module):
    """A CNN model for MNIST."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Model Utilities (Copied from bft_pso_strategy.py) ---
class ModelUtils:
    """Utility functions for model parameter handling."""
    @staticmethod
    def ndarrays_to_state_dict(model_keys: List[str], parameters: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        """Converts Flower ndarrays to a PyTorch state_dict."""
        return {k: torch.Tensor(v) for k, v in zip(model_keys, parameters)}

    @staticmethod
    def state_dict_to_ndarrays(state_dict: Dict[str, torch.Tensor]) -> List[np.ndarray]:
        """Converts a PyTorch state_dict to Flower ndarrays."""
        return [val.cpu().numpy() for _, val in state_dict.items()]

# --- Reference Data Loader (Copied from bft_pso_strategy.py) ---
class ReferenceDataLoader:
    """Loads reference dataset for server-side training."""
    @staticmethod
    def load_reference_data(batch_size: int = 32, num_samples: int = 500, data_root: str = "./data/mnist_ref"):
        """Loads a small subset of MNIST as the server's trusted dataset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # Use standard MNIST normalization
        ])
        # Download to a specific folder to avoid conflicts
        try:
            dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
            # Ensure dataset has enough samples
            num_samples = min(num_samples, len(dataset))
            indices = torch.randperm(len(dataset))[:num_samples]
            subset_dataset = Subset(dataset, indices)
            return TorchDataLoader(subset_dataset, batch_size=batch_size, shuffle=True)
        except Exception as e:
            logger.error(f"Failed to load reference MNIST data from {data_root}: {e}")
            logger.error("Please ensure the directory is writable and the dataset can be downloaded.")
            # Return an empty loader or raise an error
            return None # Or raise SystemExit("Cannot load reference data")


# --- BFT Class (Copied from bft_pso_strategy.py) ---
class ByzantineFaultTolerance:
    """Implements Byzantine Fault Tolerance filtering."""
    @staticmethod
    def euclidean_distance(arr1: np.ndarray, arr2: np.ndarray) -> float:
        return np.linalg.norm(arr1 - arr2)

    @staticmethod
    def flatten_parameters(parameters: Dict[str, torch.Tensor]) -> np.ndarray:
        # Detach tensors before converting to numpy to avoid grad issues
        return np.concatenate([param.detach().cpu().numpy().flatten() for param in parameters.values()])

    @staticmethod
    def compute_update(old_params: Dict[str, torch.Tensor], new_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: new_params[k].detach() - old_params[k].detach() for k in old_params.keys()}

    def filter_clients(self,
                       client_results: List[ClientResult],
                       reference_update: Dict[str, torch.Tensor],
                       global_parameters: Dict[str, torch.Tensor],
                       k: float = 1.25) -> List[ClientResult]:
        if not reference_update:
             logger.warning("Reference update is empty. Skipping BFT filtering.")
             return client_results # Pass all clients if reference update failed

        reference_update_flat = self.flatten_parameters(reference_update)
        reference_norm = np.linalg.norm(reference_update_flat)
        if reference_norm == 0:
            logger.warning("Reference update norm is zero. Skipping BFT filtering.")
            return client_results # Pass all if reference update is zero

        cutoff_distance = k * reference_norm
        logger.info(f"[BFT] Reference Update Norm: {reference_norm:.4f}, Cutoff Distance: {cutoff_distance:.4f}")

        filtered_results = []
        bft_passed_clients = []

        for client_result in client_results:
            try:
                client_update = self.compute_update(global_parameters, client_result.parameters)
                client_update_flat = self.flatten_parameters(client_update)
                distance = self.euclidean_distance(client_update_flat, reference_update_flat)

                logger.info(f"[BFT] Client {client_result.client_id} Update Distance: {distance:.4f}")

                if distance <= cutoff_distance:
                    filtered_results.append(client_result)
                    bft_passed_clients.append(client_result.client_id)
                else:
                    logger.warning(f"[BFT] Client {client_result.client_id} filtered. Distance too high: {distance:.4f}")
                    client_result.status = ClientStatus.FILTERED_BFT
            except Exception as e:
                logger.error(f"[BFT] Error filtering client {client_result.client_id}: {e}")
                client_result.status = ClientStatus.ERROR

        logger.info(f"[BFT] Candidates permitted: {bft_passed_clients}")
        return filtered_results


# --- PSO Classes (Copied from bft_pso_strategy.py) ---
class Particle:
    """Represents a particle in the PSO swarm."""
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.position = np.random.rand() # Position is fitness score
        self.velocity = np.random.rand() * 0.1 # Small initial velocity
        self.fitness = -np.inf
        self.personal_best_position = self.position
        self.personal_best_fitness = -np.inf

    def update_fitness(self, new_fitness: float):
        self.fitness = new_fitness
        if self.fitness > self.personal_best_fitness:
            self.personal_best_fitness = self.fitness
            self.personal_best_position = self.position # Update best position to current fitness

    def update_velocity_and_position(self, global_best_position: float):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive = C1 * r1 * (self.personal_best_position - self.position)
        social = C2 * r2 * (global_best_position - self.position)
        self.velocity = W * self.velocity + cognitive + social
        # Position update is implicitly handled by fitness update in this adaptation
        # self.position = self.position + self.velocity # Not needed if position IS fitness


class ParticleSwarmOptimization:
    """Implements PSO for intelligent client selection."""
    def __init__(self):
        self.particles: Dict[str, Particle] = {}
        self.global_best_position: Optional[float] = None # Best fitness score found
        self.global_best_fitness = -np.inf

    @staticmethod
    def compute_fitness_score(metrics: ClientMetrics, w1: float = 0.99, w2: float = 0.0, w3: float = 0.01) -> float:
        """
        Calculates a composite fitness score.
        Massively prioritizes accuracy (99%) and uses time (1%) as a minor tie-breaker.
        w2 (FPR) is set to 0.0 as it is not being used.
        """
        
        # 1. Normalize metrics
        accuracy_norm = min(max(metrics.accuracy, 0.0), 1.0)
        
        # We set w2 to 0.0, but we'll normalize fpr just in case.
        fpr_norm = min(max(metrics.false_positive_rate, 0.0), 1.0)
        
        # Normalize time. A 60-second response time will be normalized to 1.0.
        # This is a more reasonable upper bound for your simulation.
        time_norm = max(metrics.response_time, 0.0) / 60.0 

        # 2. Calculate the score using new weights
        # Note: w1=0.99, w2=0.0, w3=0.01
        score = (w1 * accuracy_norm) - (w2 * fpr_norm) - (w3 * time_norm)
        
        # Return -inf for invalid scores so PSO properly discards them
        return score if np.isfinite(score) else -np.inf

    def initialize_particles(self, client_results: List[ClientResult]):
        for client_result in client_results:
            if client_result.client_id not in self.particles:
                self.particles[client_result.client_id] = Particle(client_result.client_id)

    def update_fitness_and_positions(self, client_results: List[ClientResult]):
        """Update fitness scores for all particles based on latest metrics."""
        for client_result in client_results:
            if client_result.client_id in self.particles:
                # Use the actual metrics from the ClientResult
                fitness = self.compute_fitness_score(client_result.metrics)
                self.particles[client_result.client_id].update_fitness(fitness)
                # In this adaptation, position IS the fitness score
                self.particles[client_result.client_id].position = fitness

    def update_global_best(self):
        current_best_particle_fitness = -np.inf
        current_best_particle_position = None
        if not self.particles: return # No particles to update from

        for particle in self.particles.values():
            if particle.fitness > current_best_particle_fitness:
                 current_best_particle_fitness = particle.fitness
                 current_best_particle_position = particle.position # Position is fitness

        # Update global best only if a better fitness is found
        if current_best_particle_fitness > self.global_best_fitness:
            self.global_best_fitness = current_best_particle_fitness
            self.global_best_position = current_best_particle_position # Best fitness score
            logger.info(f"[PSO] New Global Best Fitness: {self.global_best_fitness:.4f}")


    def update_particle_velocities(self):
        """Update particle velocities based on personal and global bests."""
        if self.global_best_position is not None:
            for particle in self.particles.values():
                # Pass the global best *fitness score* as the global best position
                particle.update_velocity_and_position(self.global_best_position)


    def select_best_clients(self, client_results: List[ClientResult], num_clients: int) -> List[ClientResult]:
        """Select best clients based on PSO fitness scores."""
        # Filter results to only include clients that have corresponding particles
        valid_results = [res for res in client_results if res.client_id in self.particles]
        
        if not valid_results:
            return []

        # Sort by the current fitness of the corresponding particle
        selected_results = sorted(
            valid_results,
            key=lambda x: self.particles[x.client_id].fitness,
            reverse=True
        )[:num_clients]

        for result in selected_results:
            result.status = ClientStatus.SELECTED

        pso_passed_clients = [r.client_id for r in selected_results]
        logger.info(f"[PSO] Candidates selected: {pso_passed_clients}")
        return selected_results


# ---
#
# The Custom Flower Strategy
#
# ---

class SmartSchedulingStrategy(FedAvg):
    """Custom Flower Strategy combining BFT filtering and PSO selection."""

    def __init__(
        self,
        *,
        k_bft: float = 1.25,
        min_pso_clients: int = 2,
        initial_parameters: Parameters,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        fraction_fit: float = 1.0,
        **kwargs # Pass other FedAvg args
    ):
        # Pass relevant FedAvg parameters to the superclass
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            fraction_fit=fraction_fit,
             # Pass initial_parameters if FedAvg needs it, otherwise manage internally
            initial_parameters=initial_parameters,
            **kwargs
        )
        self.k_bft = k_bft
        self.min_pso_clients = min_pso_clients
        
        # Internal state for the strategy
        self._model = CNNModel() # Need model structure for conversions
        self._model_keys = list(self._model.state_dict().keys()) # Get keys once
        self.current_global_parameters_state_dict: Dict[str, torch.Tensor] = {} # Store as state_dict
        
        # Initialize BFT and PSO components
        self.bft = ByzantineFaultTolerance()
        self.pso = ParticleSwarmOptimization()
        self.reference_data_loader = ReferenceDataLoader.load_reference_data()
        
        if self.reference_data_loader is None:
            logger.warning("Reference data loader is None. BFT filtering might be skipped.")

        # Set initial parameters
        initial_ndarrays = parameters_to_ndarrays(initial_parameters)
        self.current_global_parameters_state_dict = ModelUtils.ndarrays_to_state_dict(
            self._model_keys, initial_ndarrays
        )
        logger.info("SmartSchedulingStrategy initialized.")

# In custom_strategy.py

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        print(f"\n==================== ROUND {server_round}: Aggregation Cycle ====================") # Clear round separator

        if not results:
            logger.warning("aggregate_fit: No successful client results received.")
            print("==================== ROUND END (No Results) ====================")
            return None, {}

        # 1. Convert received parameters and gather metrics
        print("\n--- 1. Processing Client Results ---")
        client_results_for_strategy: List[ClientResult] = []
        for client_proxy, fit_res in results:
            try:
                client_id = client_proxy.cid # Use Flower's internal ID
                params_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                client_state_dict = ModelUtils.ndarrays_to_state_dict(self._model_keys, params_ndarrays)

                acc = fit_res.metrics.get("accuracy", 0.0) # Default 0 if missing
                t_time = fit_res.metrics.get("train_time_s", 0.0) # Default 0

                client_metrics = ClientMetrics(
                    accuracy=float(acc),
                    response_time=float(t_time),
                    last_update=time.time()
                )

                '''client_results_for_strategy.append(ClientResult(
                    client_id=client_id,
                    parameters=client_state_dict,
                    num_examples=fit_res.num_examples,
                    metrics=client_metrics
                ))'''
                ##HERE TO 
                original_id_from_client = fit_res.metrics.get("client_id", "Unknown") # Read the ID

                client_results_for_strategy.append(ClientResult(
                    client_id=client_id, # Flower's ID
                    original_client_id=str(original_id_from_client), # Store the original ID
                    parameters=client_state_dict,
                    num_examples=fit_res.num_examples,
                    metrics=client_metrics
                ))
            # REPLACE THE OLD PRINT STATEMENT WITH THIS ONE:
                print(f"  > Received from Client {original_id_from_client} [{client_id[-6:]}]: {fit_res.num_examples} samples, Acc={acc:.3f}, Time={t_time:.1f}s")
                ##HERE
                #print(f"  > Received from Client {client_id[-6:]}: {fit_res.num_examples} samples, Acc={acc:.3f}, Time={t_time:.1f}s") # Shortened Client ID

            except Exception as e:
                logger.error(f"Error processing result from client {client_proxy.cid}: {e}")

        if not client_results_for_strategy:
             logger.warning("aggregate_fit: No valid client results after processing.")
             print("==================== ROUND END (Processing Error) ====================")
             return None, {}

        # 2. Compute Reference Update
        print("\n--- 2. Computing BFT Reference Update ---")
        reference_update = self.compute_reference_update()

        # 3. Apply BFT Filtering
        print("\n--- 3. Applying Byzantine Fault Tolerance (BFT) Filter ---")
        bft_filtered_results = self.bft.filter_clients(
            client_results_for_strategy,
            reference_update,
            self.current_global_parameters_state_dict,
            self.k_bft
        )

        if not bft_filtered_results:
            logger.warning("aggregate_fit: All clients were filtered by BFT. Skipping aggregation.")
            print("--------------------------------------------------")
            print(">> BFT Result: All clients filtered out.")
            print("==================== ROUND END (BFT Filtered All) ====================")
            return None, {"bft_filtered_all": 1}
        else:
            print(f">> BFT Result: {len(bft_filtered_results)} clients passed.")
            print("--------------------------------------------------")


        # 4. Apply PSO Selection
        print("\n--- 4. Applying Particle Swarm Optimization (PSO) Selection ---")
        self.pso.initialize_particles(bft_filtered_results)
        self.pso.update_fitness_and_positions(bft_filtered_results)
        self.pso.update_global_best()
        self.pso.update_particle_velocities()

        num_to_select = min(max(self.min_pso_clients, self.min_fit_clients), len(bft_filtered_results))
        print(f"  > Attempting to select top {num_to_select} clients based on fitness.")
        pso_selected_results = self.pso.select_best_clients(bft_filtered_results, num_to_select)

        if not pso_selected_results:
            logger.warning("aggregate_fit: No clients were selected by PSO. Skipping aggregation.")
            print("--------------------------------------------------")
            print(">> PSO Result: No clients selected.")
            print("==================== ROUND END (PSO Filtered All) ====================")
            return None, {"pso_filtered_all": 1}
        else:
             #READABLE CHANGES
             pso_passed_clients = [r.client_id for r in pso_selected_results] # Make sure this list is created
             logger.info(f"[PSO] Candidates selected: {pso_passed_clients}")

             # ADD THIS BLOCK for human-readable output:
             print(">> PSO Result: Human-readable selected list:")
             readable_list = [f"Client {res.original_client_id} [{res.client_id[-6:]}]" for res in pso_selected_results]
             print(f"     {readable_list}")

            #READABLE CHANGES
             print(f">> PSO Result: Selected {len(pso_selected_results)} clients:")
             for res in pso_selected_results:
                 print(f"     - Client {res.original_client_id} [{res.client_id[-6:]}] (Fitness: {self.pso.particles[res.client_id].fitness:.4f})")
             print("--------------------------------------------------")

        # 5. Aggregate ONLY the selected clients
        print("\n--- 5. Aggregating Selected Client Models ---")
        aggregated_state_dict = self.weighted_average(pso_selected_results)
        print("  > Weighted averaging complete.")

        # 6. Convert back to Flower's Parameters format and update stored parameters
        aggregated_ndarrays = ModelUtils.state_dict_to_ndarrays(aggregated_state_dict)
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        self.current_global_parameters_state_dict = aggregated_state_dict # Update server's state

        # Aggregate custom metrics
        selected_accuracies = [res.metrics.accuracy for res in pso_selected_results]
        metrics_aggregated = {
            "selected_clients": len(pso_selected_results),
            "avg_selected_accuracy": np.mean(selected_accuracies) if selected_accuracies else 0.0
        }

        print(f"--- Aggregation complete for Round {server_round}. ---")
        print(f"==================== ROUND {server_round}: END ====================\n")

        return aggregated_parameters, metrics_aggregated

    # --- (Keep your existing compute_reference_update and weighted_average helper methods below) ---
    # --- Helper methods adapted from bft_pso_strategy.py ---

    def compute_reference_update(self) -> Dict[str, torch.Tensor]:
        """Compute server's reference update using trusted dataset."""
        if self.reference_data_loader is None:
             logger.warning("Cannot compute reference update: reference data loader is not available.")
             return {} # Return empty if no data

        # Use a copy of the current global model state for training
        model_copy = CNNModel()
        model_copy.load_state_dict(self.current_global_parameters_state_dict)
        original_parameters = {k: v.clone().detach() for k, v in model_copy.state_dict().items()}

        criterion = nn.CrossEntropyLoss()
        # Use a small learning rate for server-side refinement
        optimizer = optim.SGD(model_copy.parameters(), lr=0.001, momentum=0.9)
        device = next(model_copy.parameters()).device # Use the model's device

        model_copy.train()
        # Perform a small number of updates on the server
        batches_done = 0
        max_batches = 5 # Limit server computation
        total_loss = 0.0
        try:
            for images, labels in self.reference_data_loader:
                if batches_done >= max_batches:
                    break
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model_copy(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches_done += 1
            avg_loss = total_loss / batches_done if batches_done > 0 else 0
            logger.info(f"Computed reference update over {batches_done} batches. Avg Loss: {avg_loss:.4f}")
        except Exception as e:
            logger.error(f"Error during reference update computation: {e}")
            return {} # Return empty on error


        updated_parameters = model_copy.state_dict()
        return self.bft.compute_update(original_parameters, updated_parameters)


    def weighted_average(self, client_results: List[ClientResult]) -> Dict[str, torch.Tensor]:
        """Compute weighted average of client parameters (state_dict format)."""
        if not client_results:
             logger.warning("Weighted average called with no client results.")
             return self.current_global_parameters_state_dict # Return current params if none selected

        total_examples = sum(res.num_examples for res in client_results)
        if total_examples == 0:
             logger.warning("Weighted average: Total number of examples is zero.")
             return self.current_global_parameters_state_dict

        # Initialize aggregated parameters based on the global model structure
        aggregated_params = {k: torch.zeros_like(v) for k, v in self.current_global_parameters_state_dict.items()}
        
        # Use the device of the global parameters if available, else cpu
        agg_device = next(iter(aggregated_params.values())).device if aggregated_params else torch.device("cpu")

        for result in client_results:
             weight = result.num_examples / total_examples
             for name, param in result.parameters.items():
                  if name in aggregated_params:
                      # Ensure parameters are on the same device before adding
                      aggregated_params[name] += weight * param.to(agg_device).detach()
                  else:
                      logger.warning(f"Parameter '{name}' from client {result.client_id} not found in global model structure.")

        return aggregated_params