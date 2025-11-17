from typing import Dict, Any
from queue_simulator import QueueSimulator, QueueMetrics, QueueDiscipline


class Evaluator:
    """
    Evaluates queuing system configurations by running simulations
    and computing efficiency metrics.
    """

    def __init__(
        self,
        target_arrival_rate: float = 0.8,
        simulation_duration: float = 10.0,
        seed: int = 42,
    ):
        """
        Initialize the evaluator.

        Args:
            target_arrival_rate: The arrival rate to test against (lambda)
            simulation_duration: How long to run each simulation
            seed: Random seed for reproducibility
        """
        self.target_arrival_rate = target_arrival_rate
        self.simulation_duration = simulation_duration
        self.seed = seed

    def evaluate(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a set of queuing hyperparameters.

        Args:
            hyperparameters: Dictionary containing:
                - num_servers: int (number of servers)
                - service_rate: float (mu - service rate per server)
                - queue_discipline: str ("FIFO", "LIFO", or "PRIORITY")

        Returns:
            Dictionary with:
                - efficiency_score: float (lower is better, 0 is perfect)
                - metrics: QueueMetrics as dict
                - observations: str (human-readable feedback)
        """
        # Extract and validate hyperparameters
        num_servers = int(hyperparameters.get("num_servers", 1))
        service_rate = float(hyperparameters.get("service_rate", 1.0))
        queue_discipline_str = hyperparameters.get("queue_discipline", "FIFO").upper()

        # Convert queue discipline string to enum
        try:
            queue_discipline = QueueDiscipline[queue_discipline_str]
        except KeyError:
            queue_discipline = QueueDiscipline.FIFO

        # Validate parameters
        num_servers = max(1, num_servers)
        service_rate = max(0.01, service_rate)

        # Run simulation
        simulator = QueueSimulator(
            num_servers=num_servers,
            service_rate=service_rate,
            arrival_rate=self.target_arrival_rate,
            queue_discipline=queue_discipline,
            simulation_duration=self.simulation_duration,
            seed=self.seed,
        )

        metrics = simulator.simulate()

        # Compute efficiency score
        # Lower is better. We want to minimize:
        # - Average wait time (primary concern)
        # - Average queue length
        # - But also consider server utilization (don't want too many idle servers)

        # Normalize metrics for scoring
        # Ideal: low wait time, low queue length, reasonable utilization
        wait_time_penalty = metrics.average_wait_time * 10  # Weight wait time heavily
        queue_length_penalty = metrics.average_queue_length * 5
        utilization_penalty = (
            abs(metrics.server_utilization - 0.8) * 2
        )  # Prefer ~80% utilization

        efficiency_score = (
            wait_time_penalty + queue_length_penalty + utilization_penalty
        )

        # Generate observations
        observations = self._generate_observations(
            metrics, num_servers, service_rate, queue_discipline_str
        )

        return {
            "efficiency_score": efficiency_score,
            "metrics": metrics.to_dict(),
            "observations": observations,
        }

    def _generate_observations(
        self,
        metrics: QueueMetrics,
        num_servers: int,
        service_rate: float,
        queue_discipline: str,
    ) -> str:
        """Generate human-readable feedback about the configuration"""
        observations = []

        # Wait time analysis
        if metrics.average_wait_time < 1.0:
            observations.append("Excellent wait times - agents are served quickly!")
        elif metrics.average_wait_time < 5.0:
            observations.append("Good wait times - acceptable performance")
        elif metrics.average_wait_time < 10.0:
            observations.append("Moderate wait times - room for improvement")
        else:
            observations.append("High wait times - system is struggling")

        # Queue length analysis
        if metrics.average_queue_length < 1.0:
            observations.append("Queue stays short - good capacity")
        elif metrics.average_queue_length < 3.0:
            observations.append("Moderate queue lengths")
        else:
            observations.append(
                f"Long queues (avg: {metrics.average_queue_length:.1f}) - consider more servers or faster service"
            )

        # Utilization analysis
        if metrics.server_utilization > 0.95:
            observations.append("Very high server utilization - servers are overloaded")
        elif metrics.server_utilization > 0.8:
            observations.append("High server utilization - efficient use of resources")
        elif metrics.server_utilization > 0.5:
            observations.append("Moderate server utilization - balanced")
        else:
            observations.append("Low server utilization - may have excess capacity")

        # Throughput analysis
        if metrics.agents_served < 10:
            observations.append("Low throughput - few agents served")
        else:
            observations.append(f"Throughput: {metrics.agents_served} agents served")

        # Stability check (rho = lambda / (c * mu))
        rho = self.target_arrival_rate / (num_servers * service_rate)
        if rho >= 1.0:
            observations.append(
                "WARNING: System is unstable (arrival rate >= service capacity) - queues will grow indefinitely!"
            )
        elif rho > 0.9:
            observations.append(
                "WARNING: System is near capacity - high risk of instability"
            )
        elif rho < 0.3:
            observations.append("System has excess capacity - may be over-provisioned")

        return " | ".join(observations)
