"""
Simplified Queue Simulation System using AgentModel from emergent.main

Simulates agents waiting in queues with configurable hyperparameters:
- Number of servers (c)
- Service rate (mu) - how fast each server processes agents
- Arrival rate (lambda) - how fast agents arrive
- Queue discipline (FIFO, LIFO, priority)
- Simulation duration
"""

import random
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from emergent.main import AgentModel


class QueueDiscipline(Enum):
    FIFO = "FIFO"  # First In First Out
    LIFO = "LIFO"  # Last In First Out
    PRIORITY = "PRIORITY"  # Priority-based (lower number = higher priority)


@dataclass
class QueueMetrics:
    """Metrics collected from a queue simulation"""

    total_agents: int = 0
    agents_served: int = 0
    total_wait_time: float = 0.0
    total_service_time: float = 0.0
    max_queue_length: int = 0
    average_queue_length: float = 0.0
    server_utilization: float = 0.0

    @property
    def average_wait_time(self) -> float:
        """Average time agents wait in queue"""
        if self.agents_served == 0:
            return 0.0
        return self.total_wait_time / self.agents_served

    @property
    def average_service_time(self) -> float:
        """Average service time"""
        if self.agents_served == 0:
            return 0.0
        return self.total_service_time / self.agents_served

    @property
    def average_time_in_system(self) -> float:
        """Average total time in system"""
        return self.average_wait_time + self.average_service_time

    @property
    def throughput(self) -> float:
        """Agents served per unit time"""
        return self.agents_served

    def to_dict(self) -> dict:
        """Convert metrics to dictionary"""
        return {
            "total_agents": self.total_agents,
            "agents_served": self.agents_served,
            "average_wait_time": self.average_wait_time,
            "average_service_time": self.average_service_time,
            "average_time_in_system": self.average_time_in_system,
            "max_queue_length": self.max_queue_length,
            "average_queue_length": self.average_queue_length,
            "server_utilization": self.server_utilization,
            "throughput": self.throughput,
        }


class QueueSimulator:
    """
    Simplified queue simulator using AgentModel from emergent.main.

    Each agent is a node in the graph with state:
    - "waiting": in queue
    - "serving": being processed by a server
    - "completed": finished and left system
    """

    def __init__(
        self,
        num_servers: int = 1,
        service_rate: float = 1.0,  # mu: agents per time unit
        arrival_rate: float = 0.5,  # lambda: arrivals per time unit
        queue_discipline: QueueDiscipline = QueueDiscipline.FIFO,
        simulation_duration: float = 10.0,
        seed: Optional[int] = None,
    ):
        self.num_servers = num_servers
        self.service_rate = service_rate  # mu
        self.arrival_rate = arrival_rate  # lambda
        self.queue_discipline = queue_discipline
        self.simulation_duration = simulation_duration
        self.random = random.Random(seed)

        # Validate parameters
        if num_servers < 1:
            raise ValueError("Number of servers must be at least 1")
        if service_rate <= 0:
            raise ValueError("Service rate must be positive")
        if arrival_rate < 0:
            raise ValueError("Arrival rate must be non-negative")
        if simulation_duration <= 0:
            raise ValueError("Simulation duration must be positive")

        # Initialize AgentModel
        self.model = AgentModel()
        self.model.update_parameters(
            {
                "num_nodes": 0,  # Will grow as agents arrive
                "max_timesteps": int(
                    simulation_duration * 10
                ),  # Use discrete timesteps
            }
        )

        # Track simulation state
        self.current_time = 0.0
        self.time_step = 0.1  # Discrete time steps
        self.next_arrival_time = 0.0
        self.agent_counter = 0
        self.queue: List[int] = []  # List of agent IDs waiting
        self.servers: Dict[int, int] = {}  # server_idx -> agent_id being served
        self.server_busy_until: Dict[int, float] = {}  # server_idx -> time when free
        self.server_busy_time: Dict[int, float] = {}  # server_idx -> total busy time

        # Metrics tracking
        self.metrics = QueueMetrics()
        self.queue_length_samples: List[float] = []  # Track queue length over time

        # Set up AgentModel functions
        self.model.set_initial_data_function(self._initial_data_function)
        self.model.set_timestep_function(self._timestep_function)

    def _initial_data_function(self, model: AgentModel) -> Dict[str, Any]:
        """Initialize agent data when created"""
        return {
            "state": "waiting",
            "arrival_time": self.current_time,
            "service_start_time": None,
            "service_end_time": None,
            "priority": self.random.random(),
            "wait_time": 0.0,
            "service_time": 0.0,
        }

    def _timestep_function(self, model: AgentModel) -> None:
        """Update model state each timestep"""
        graph = model.get_graph()

        # Update current time
        self.current_time += self.time_step

        # Generate new arrivals
        while self.next_arrival_time <= self.current_time:
            if self.current_time < self.simulation_duration:
                # Create new agent node
                agent_id = self.agent_counter
                self.agent_counter += 1

                initial_data = self._initial_data_function(model)
                graph.add_node(agent_id, **initial_data)

                # Add to queue
                self.queue.append(agent_id)
                self.metrics.total_agents += 1

                # Schedule next arrival
                if self.arrival_rate > 0:
                    self.next_arrival_time += self.random.expovariate(self.arrival_rate)
                else:
                    self.next_arrival_time = float("inf")
            else:
                break

        # Track busy time for all active servers
        for server_idx in self.servers.keys():
            self.server_busy_time[server_idx] = (
                self.server_busy_time.get(server_idx, 0) + self.time_step
            )

        # Process servers - check if any finish service
        for server_idx in list(self.servers.keys()):
            if self.current_time >= self.server_busy_until.get(server_idx, 0):
                # Server finished, agent leaves
                agent_id = self.servers.pop(server_idx)
                self.server_busy_until.pop(server_idx, None)

                if agent_id in graph.nodes():
                    agent_data = graph.nodes[agent_id]
                    agent_data["state"] = "completed"
                    agent_data["service_end_time"] = self.current_time
                    agent_data["service_time"] = (
                        agent_data["service_end_time"]
                        - agent_data["service_start_time"]
                    )
                    agent_data["wait_time"] = (
                        agent_data["service_start_time"] - agent_data["arrival_time"]
                    )

                    # Update metrics
                    self.metrics.agents_served += 1
                    self.metrics.total_wait_time += agent_data["wait_time"]
                    self.metrics.total_service_time += agent_data["service_time"]

        # Assign waiting agents to free servers
        for server_idx in range(self.num_servers):
            if server_idx not in self.servers and self.queue:
                # Select agent from queue based on discipline
                agent_id = self._select_agent_from_queue()
                if agent_id is not None:
                    # Start service
                    service_time = self.random.expovariate(self.service_rate)
                    self.servers[server_idx] = agent_id
                    self.server_busy_until[server_idx] = (
                        self.current_time + service_time
                    )

                    if agent_id in graph.nodes():
                        agent_data = graph.nodes[agent_id]
                        agent_data["state"] = "serving"
                        agent_data["service_start_time"] = self.current_time

        # Track queue length
        self.queue_length_samples.append(len(self.queue))
        self.metrics.max_queue_length = max(
            self.metrics.max_queue_length, len(self.queue)
        )

        model.set_graph(graph)

    def _select_agent_from_queue(self) -> Optional[int]:
        """Select next agent to serve based on queue discipline"""
        if not self.queue:
            return None

        if self.queue_discipline == QueueDiscipline.FIFO:
            return self.queue.pop(0)
        elif self.queue_discipline == QueueDiscipline.LIFO:
            return self.queue.pop()
        elif self.queue_discipline == QueueDiscipline.PRIORITY:
            # Find agent with lowest priority value (highest priority)
            graph = self.model.get_graph()
            min_priority = float("inf")
            min_idx = -1
            for i, agent_id in enumerate(self.queue):
                if agent_id in graph.nodes():
                    priority = graph.nodes[agent_id].get("priority", 0.5)
                    if priority < min_priority:
                        min_priority = priority
                        min_idx = i
            if min_idx >= 0:
                return self.queue.pop(min_idx)
            return self.queue.pop(0)  # Fallback
        else:
            return self.queue.pop(0)  # Default to FIFO

    def simulate(self) -> QueueMetrics:
        """
        Run the queue simulation.

        Returns:
            QueueMetrics with simulation results
        """
        # Initialize graph
        self.model.initialize_graph()

        # Run simulation
        num_timesteps = int(self.simulation_duration / self.time_step)
        for _ in range(num_timesteps):
            self.model.timestep()
            if self.current_time >= self.simulation_duration:
                break

        # Calculate final metrics
        if self.queue_length_samples:
            self.metrics.average_queue_length = sum(self.queue_length_samples) / len(
                self.queue_length_samples
            )

        # Calculate server utilization
        # server_busy_time already tracks time spent busy during simulation
        total_server_time = sum(self.server_busy_time.values())
        total_available_time = self.num_servers * self.simulation_duration
        if total_available_time > 0:
            self.metrics.server_utilization = max(
                0.0, min(1.0, total_server_time / total_available_time)
            )

        return self.metrics
