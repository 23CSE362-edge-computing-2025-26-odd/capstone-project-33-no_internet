class Particle:
    """Represents a particle in the PSO swarm."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        # PSO position will be a single float representing the fitness score
        self.position = np.random.rand()
        self.velocity = np.random.rand()
        self.fitness = -np.inf
        self.personal_best_position = self.position
        self.personal_best_fitness = -np.inf
        self.metrics = ClientMetrics()

    def update_fitness(self, new_fitness: float):
        """Updates particle fitness and personal best."""
        self.fitness = new_fitness
        if self.fitness > self.personal_best_fitness:
            self.personal_best_fitness = self.fitness
            self.personal_best_position = self.position

    def update_velocity_and_position(self, global_best_position: float):
        """Updates particle velocity and position using PSO equations."""
        r1 = np.random.rand()
        r2 = np.random.rand()

        cognitive_component = C1 * r1 * (self.personal_best_position - self.position)
        social_component = C2 * r2 * (global_best_position - self.position)
        self.velocity = W * self.velocity + cognitive_component + social_component
        self.position = self.position + self.velocity


class ParticleSwarmOptimization:
    """Implements PSO for intelligent client selection."""

    def __init__(self):
        self.particles: Dict[str, Particle] = {}
        self.global_best_position: Optional[float] = None
        self.global_best_fitness = -np.inf

    @staticmethod
    def compute_fitness_score(accuracy: float,
                              false_positive_rate: float,
                              response_time: float,
                              w1: float = 0.5,
                              w2: float = 0.3,
                              w3: float = 0.2) -> float:
        """Calculates composite fitness score."""
        return (w1 * accuracy) - (w2 * false_positive_rate) - (w3 * response_time)

    def initialize_particles(self, client_results: List[ClientResult]):
        """Initialize particles for new clients."""
        for client_result in client_results:
            if client_result.client_id not in self.particles:
                self.particles[client_result.client_id] = Particle(client_result.client_id)

    def update_fitness(self, client_results: List[ClientResult]):
        """Update fitness scores for all particles."""
        for client_result in client_results:
            if client_result.client_id in self.particles:
                fitness = self.compute_fitness_score(
                    client_result.metrics.accuracy,
                    client_result.metrics.false_positive_rate,
                    client_result.metrics.response_time
                )
                self.particles[client_result.client_id].update_fitness(fitness)

    def update_global_best(self):
        """Update global best position and fitness."""
        for particle in self.particles.values():
            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position

    def update_particles(self):
        """Update all particle positions and velocities."""
        if self.global_best_position is not None:
            for particle in self.particles.values():
                particle.update_velocity_and_position(self.global_best_position)

    def select_best_clients(self, client_results: List[ClientResult], num_clients: int) -> List[ClientResult]:
        """Select best clients based on PSO fitness scores."""
        selected_results = sorted(
            client_results,
            key=lambda x: self.particles.get(x.client_id, Particle("", )).fitness,
            reverse=True
        )[:num_clients]

        for result in selected_results:
            result.status = ClientStatus.SELECTED

        pso_passed_clients = [r.client_id for r in selected_results]
        print("---")
        print("Candidates permitted by PSO:")
        print(pso_passed_clients)
        print("---")
        return selected_results


class HybridStrategy:
    """
    Standalone Hybrid Strategy combining BFT filtering and PSO for intelligent client selection.
    """

    def __init__(self,
                 k: float = 29.0,
                 min_clients: int = 2,
                 github_owner: str = "EdgeComputing-x-CI",
                 github_repo: str = "EdgeComputing-x-CI",
                 github_token: str = "YOUR_GITHUB_TOKEN",
                 check_interval: int = 5):
        self.k = k
        self.min_clients = min_clients
        self.check_interval = check_interval
        self.global_parameters: Optional[Dict[str, torch.Tensor]] = None
        self.github_owner = github_owner
        self.github_repo = github_repo

        # Initialize components
        self.storage = LocalStorage()
        self.bft = ByzantineFaultTolerance()
        self.pso = ParticleSwarmOptimization()
        self.model = CNNModel()

        # Load reference data
        self.reference_data_loader = DataLoader.load_reference_data()

    def initialize_global_model(self, model_filename: str = "../src/current_weights.pth"):
        """Initialize global model parameters from local file, or locally if not found."""
        params = self.storage.load_state_dict(model_filename)
        if params:
            self.global_parameters = params
            self.model.load_state_dict(params)
            logger.info("Successfully loaded initial global model from local file.")
        else:
            logger.warning("Failed to load initial model from local file. Initializing a new model locally.")
            self.global_parameters = self.model.state_dict()

    def compute_reference_update(self) -> Dict[str, torch.Tensor]:
        """Compute server's reference update using trusted dataset."""
        logger.info("Computing reference update using trusted dataset.")
        model = CNNModel()
        model.load_state_dict(self.global_parameters)
        original_parameters = {k: v.clone() for k, v in model.state_dict().items()}

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        model.train()
        for images, labels in self.reference_data_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        updated_parameters = model.state_dict()
        return self.bft.compute_update(original_parameters, updated_parameters)

    def weighted_average(self, client_results: List[ClientResult]) -> Dict[str, torch.Tensor]:
        """Compute weighted average of client parameters."""
        if not client_results:
            logger.warning("No clients selected for aggregation. Returning current global model.")
            return self.global_parameters

        total_examples = sum(result.num_examples for result in client_results)
        if total_examples == 0:
            logger.warning("Total number of examples is zero. Returning current global model.")
            return self.global_parameters

        # Initialize aggregated parameters with zeros
        aggregated_params = {k: torch.zeros_like(v) for k, v in self.global_parameters.items()}

        # Weighted averaging
        for result in client_results:
            weight = result.num_examples / total_examples
            for name, param in result.parameters.items():
                aggregated_params[name] += weight * param

        return aggregated_params

    def run_full_cycle(self, client_updates_folder: str = "../client_updates"):
        """
        Executes a full cycle of client selection and model aggregation.
        1. Lists all client update files.
        2. Loads each client's model state.
        3. Applies BFT filtering.
        4. Applies PSO selection.
        5. Aggregates the selected client models.
        6. Saves the new global model.
        """
        logger.info(f"Starting full cycle for client updates in '{client_updates_folder}' folder.")

        # New step: Load the editor log file to find eligible clients
        log_file_path = "../edgeComputing/editor_log.txt"
        eligible_client_ids = self.storage.load_eligible_clients_from_log(log_file_path)
        print(f"Eligible Candidates: {eligible_client_ids}")
        if not eligible_client_ids:
            logger.warning("No eligible clients found in editor_log.txt. Exiting.")
            return

        # Phase 1: Access and load client data
        client_files = self.storage.list_folder_contents(client_updates_folder)
        if not client_files:
            logger.warning("No .pth files found in the client_updates folder. Exiting.")
            return

        # New step: Filter client files based on the editor log
        eligible_client_files = [
            file_name for file_name in client_files
            if file_name.replace(".pth", "") in eligible_client_ids
        ]

        if not eligible_client_files:
            logger.warning("No client update files match the eligible list. Skipping aggregation.")
            return

        client_results = []
        for file_name in eligible_client_files:
            file_path = os.path.join(client_updates_folder, file_name)
            client_id = file_name.replace(".pth", "")

            logger.info(f"Attempting to load eligible model from file: {file_path}")
            state_dict = self.storage.load_state_dict(file_path)

            if state_dict:
                # We assume some placeholder metrics and num_examples here
                # In a real-world scenario, this data would be sent along with the model.
                metrics = ClientMetrics(
                    accuracy=np.random.rand(),  # Placeholder
                    false_positive_rate=np.random.rand() * 0.1,  # Placeholder
                    response_time=np.random.randint(100, 500),  # Placeholder
                    last_update=time.time()
                )
                client_results.append(ClientResult(
                    client_id=client_id,
                    parameters=state_dict,
                    num_examples=np.random.randint(100, 1000),  # Placeholder
                    metrics=metrics
                ))

        if not client_results:
            logger.warning("No valid client models were loaded. Skipping aggregation.")
            return

        # Phase 2: BFT Filtering
        reference_update = self.compute_reference_update()
        filtered_results = self.bft.filter_clients(client_results, reference_update, self.global_parameters, self.k)

        if not filtered_results:
            logger.warning("All clients were filtered by BFT. No aggregation will occur.")
            return

        # Phase 3: PSO Client Selection
        self.pso.initialize_particles(filtered_results)
        self.pso.update_fitness(filtered_results)
        self.pso.update_global_best()
        self.pso.update_particles()
        selected_results = self.pso.select_best_clients(filtered_results, min(self.min_clients, len(filtered_results)))

        if not selected_results:
            logger.warning("PSO selection resulted in no clients. No aggregation will occur.")
            return

        # Phase 4: Model Aggregation
        aggregated_state_dict = self.weighted_average(selected_results)
        self.global_parameters = aggregated_state_dict

        # Phase 5: Write back the model updates to src folder
        self.storage.save_state_dict(aggregated_state_dict, "../src/current_weights.pth")

        logger.info("Full aggregation cycle complete. The new global model is now available locally.")


# Example usage
if __name__ == "__main__":
    # Configuration
    # This path is correct because the script is being run from ci/
    CLIENT_UPDATES_FOLDER = "../client_updates"

    # Initialize strategy
    strategy = HybridStrategy()

    # Initialize the global model from the repository
    strategy.initialize_global_model()

    # Run a single full aggregation and update cycle
    strategy.run_full_cycle(client_updates_folder=CLIENT_UPDATES_FOLDER)
