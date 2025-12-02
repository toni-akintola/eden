from typing import Callable, Annotated, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum

# --- Type Aliases for State-Dependent Functions ---

# Function for a state-dependent rate: k (queue length) -> rate (float)
RateFunction = Callable[[int], float]

# Function for the entry probability: k (queue length) -> entry_probability [0, 1]
EntryProbFunction = Callable[[int], float]

# Function for the exit rule: (queue_length k, position l) -> (rate y, prob z)
ExitRuleFunction = Callable[[int, int], tuple[float, float]]

# --- Agent and Queue Structure ---


class Agent:
    """
    Represents an agent in the system with Bayesian belief updating.

    The agent maintains a belief distribution γ̃_t^ℓ over their position ℓ in the queue,
    which is updated based on service events according to the queue discipline.
    """

    def __init__(
        self,
        arrival_time: float,
        initial_position: int,
        v: Optional[float] = None,
        c: Optional[float] = None,
    ):
        """
        Initialize an agent.

        Args:
            arrival_time: Time when agent arrived
            initial_position: Initial position in queue (1-indexed, 1 = front)
            v: Agent's private value of service (if None, uses global V)
            c: Agent's private waiting cost per period (if None, uses global C)
        """
        self.arrival_time = arrival_time
        self.v = v  # Private value of service
        self.c = c  # Private waiting cost per period

        # Bayesian belief state: γ̃_t^ℓ = P(position = ℓ | not served by time t)
        # Initially, agent knows their exact position with certainty
        self.belief: Dict[int, float] = {initial_position: 1.0}

        # Track expected remaining wait time based on current belief
        self.expected_remaining_wait: float = 0.0

        # Track time spent waiting (for utility calculation)
        self.time_in_queue: float = 0.0

    def update_belief_on_service(
        self,
        queue_discipline: "QueueDiscipline",
        service_occurred: bool,
        current_queue_length: int,
        service_rate: float,
    ):
        """
        Update belief distribution after a service event using Bayes' rule.

        From the paper, the belief update formula is:

        γ̃_{t+1}^ℓ = (γ̃_t^ℓ · μ_B + γ̃_t^{ℓ+1} · μ_A) / (γ̃_t^1 · μ_B + Σ_{i=2}^{K_A} γ̃_t^i)

        Where:
        - μ_A = probability of moving up (agent ahead served)
        - μ_B = probability of NOT moving up (staying in place)

        For FCFS when a service event occurs:
        - Agent definitely moves up one position (μ_A = 1 for the event)
        - Conditional on not being served: belief shifts down by 1

        Args:
            queue_discipline: The queue discipline in effect
            service_occurred: Whether a service completion just happened
            current_queue_length: Queue length after the event
            service_rate: The service rate μ
        """
        if not service_occurred or not self.belief:
            return

        new_belief: Dict[int, float] = {}
        k_before = current_queue_length + 1  # Queue length before service

        if queue_discipline == QueueDiscipline.FCFS:
            # Under FCFS, service always removes from position 1 (front)
            # If service occurred and agent is still here, they moved up one position
            #
            # Paper's formula applied to FCFS service event:
            # γ̃_{t+1}^ℓ = γ̃_t^{ℓ+1} / (1 - γ̃_t^1)
            #
            # Because: μ_A = 1 (definitely moved up), μ_B = 0 (can't stay in place)
            # Denominator: probability of NOT being served = 1 - P(was at position 1)

            prob_was_at_front = self.belief.get(1, 0.0)
            denominator = 1.0 - prob_was_at_front

            if denominator > 0:
                for pos, prob in self.belief.items():
                    if pos > 1:
                        # Agent was at position pos, now at pos-1
                        new_pos = pos - 1
                        # Bayes: P(was at pos | not served) = P(was at pos) / P(not served)
                        new_belief[new_pos] = (
                            new_belief.get(new_pos, 0.0) + prob / denominator
                        )
                    # If pos == 1, agent would have been served (excluded)

        elif queue_discipline == QueueDiscipline.LIFO:
            # Under LIFO, service removes from position k (back)
            # If service occurred and agent is still here, they weren't at the back
            # Position unchanged for all remaining agents
            #
            # Paper's formula applied to LIFO:
            # γ̃_{t+1}^ℓ = γ̃_t^ℓ / (1 - γ̃_t^k)  for ℓ < k

            prob_was_at_back = self.belief.get(k_before, 0.0)
            denominator = 1.0 - prob_was_at_back

            if denominator > 0:
                for pos, prob in self.belief.items():
                    if pos < k_before:
                        # Agent wasn't at back, position unchanged
                        new_belief[pos] = new_belief.get(pos, 0.0) + prob / denominator
                    # If pos == k_before (back), agent would have been served

        elif queue_discipline == QueueDiscipline.SIRO:
            # Under SIRO, any position could have been served with equal probability 1/k
            # If service occurred and agent is still here, they weren't the one selected
            #
            # Paper's formula for SIRO:
            # P(position ℓ | not served) = P(not served | position ℓ) * P(position ℓ) / P(not served)
            # P(not served | position ℓ) = (k-1)/k for all ℓ (equal chance of being selected)
            # P(not served) = (k-1)/k (since exactly one agent is served)
            #
            # Result: belief distribution is unchanged, but positions may shift
            # Actually in SIRO, positions don't have meaning - just renormalize

            # The served agent was removed, so we need to account for that
            # Under SIRO, the agent's position belief remains proportionally the same
            # but we need to exclude the possibility they were the one served

            if k_before > 1:
                # Each position had 1/k chance of being served
                # P(not served) = (k-1)/k
                for pos, prob in self.belief.items():
                    # Belief stays proportionally the same
                    new_belief[pos] = prob

        # Normalize the belief distribution
        total = sum(new_belief.values())
        if total > 0:
            self.belief = {pos: prob / total for pos, prob in new_belief.items()}

    def update_expected_wait(
        self, service_rate: float, queue_discipline: "QueueDiscipline" = None
    ):
        """
        Update expected remaining wait time based on current belief.

        From the paper:
        E[T | not served by t] = Σ_ℓ γ̃_t^ℓ * E[T | position = ℓ]

        For FCFS with position ℓ:
        - E[T | ℓ] = ℓ / μ (expected time for ℓ services to complete)
        - This is because ℓ agents ahead must be served first

        For LIFO with position ℓ:
        - Expected wait depends on future arrivals and is more complex
        - Approximation: E[T | ℓ] ≈ 1/μ (next service might be you if at back)

        For SIRO with position ℓ:
        - E[T | ℓ] = k/μ where k is queue length (position doesn't matter)

        Args:
            service_rate: The service rate μ
            queue_discipline: The queue discipline (affects wait time calculation)
        """
        if service_rate <= 0:
            self.expected_remaining_wait = float("inf")
            return

        # Default to FCFS if not specified
        if queue_discipline is None:
            queue_discipline = QueueDiscipline.FCFS

        expected_wait = 0.0

        if queue_discipline == QueueDiscipline.FCFS:
            # E[T | position ℓ] = ℓ / μ for FCFS
            # Agent at position ℓ waits for ℓ services (including their own)
            for pos, prob in self.belief.items():
                expected_wait += prob * (pos / service_rate)

        elif queue_discipline == QueueDiscipline.LIFO:
            # Under LIFO, expected wait is highly variable
            # If at back (position k), you're served next: E[T] = 1/μ
            # If at front (position 1), you wait until queue empties
            # Approximation: weight by how close to back
            max_pos = max(self.belief.keys()) if self.belief else 1
            for pos, prob in self.belief.items():
                # Closer to back = shorter wait
                # This is a rough approximation
                relative_pos = (max_pos - pos + 1) / max_pos if max_pos > 0 else 1
                expected_wait += prob * (1 / service_rate) / relative_pos

        elif queue_discipline == QueueDiscipline.SIRO:
            # Under SIRO, position doesn't matter - each agent has equal chance
            # E[T] = E[k] / μ where E[k] is expected queue length when served
            # Approximation: use expected position as proxy for queue length
            exp_pos = self.get_expected_position()
            expected_wait = exp_pos / service_rate

        self.expected_remaining_wait = expected_wait

    def get_expected_position(self) -> float:
        """Get the expected position E[ℓ] based on current belief."""
        return sum(pos * prob for pos, prob in self.belief.items())

    def should_abandon(self, V_global: float, C_global: float) -> bool:
        """
        Check if agent should voluntarily abandon based on incentive constraint.

        Agent abandons if: v - c * E[T | not served] < 0

        Args:
            V_global: Global value of service (used if agent has no private v)
            C_global: Global waiting cost (used if agent has no private c)

        Returns:
            True if agent should abandon, False otherwise
        """
        v = self.v if self.v is not None else V_global
        c = self.c if self.c is not None else C_global

        expected_utility = v - c * self.expected_remaining_wait
        return expected_utility < 0

    def __repr__(self):
        exp_pos = self.get_expected_position()
        return f"Agent(arr={self.arrival_time:.2f}, E[ℓ]={exp_pos:.1f})"


# --- Designer's Choices (Enums) ---


class QueueDiscipline(Enum):
    """Specifies the order in which agents are selected for service."""

    FCFS = "First Come First Serve"
    LIFO = "Last Come First Serve"
    SIRO = "Service In Random Order"  # Service in Random Order


class InformationRule(Enum):
    """Specifies the level of information agents receive."""

    NO_INFORMATION_BEYOND_REC = "No Information Beyond Recommendation"
    FULL_INFORMATION = "Full Information (Observable Queue)"
    COARSE_INFORMATION = "Coarse Information"


# --- Designer's Policy Functions ---


class EntryExitRule(BaseModel):
    """
    Represents the designer's policy for entry (x) and exit (y, z).
    These functions are the control variables the optimization agent seeks to define.
    """

    entry_rule_fn: EntryProbFunction = Field(
        ..., description="Function for x_k: queue_length -> entry_probability [0, 1]"
    )
    exit_rule_fn: ExitRuleFunction = Field(
        ...,
        description="Function for (y_k,l, z_k,l): (queue_length, position) -> (rate y >= 0, prob z in [0, 1])",
    )

    class Config:
        arbitrary_types_allowed = True


class PrimitiveProcess(BaseModel):
    """Represents the fundamental arrival and service processes (λ, μ)."""

    arrival_rate_fn: RateFunction = Field(
        ..., description="Function for λ_k: queue_length -> arrival_rate"
    )
    service_rate_fn: RateFunction = Field(
        ..., description="Function for μ_k: queue_length -> service_rate"
    )

    class Config:
        arbitrary_types_allowed = True


# --- Complete Model Definition ---


class CheTercieuxQueueModel(BaseModel):
    """
    A class representing the full queue model and design parameters.
    """

    # Core Preferences/Utility (These are constants in the simulation)
    V_surplus: Annotated[
        float, Field(description="Net surplus V from service (V > 0).", gt=0)
    ]
    C_waiting_cost: Annotated[
        float, Field(description="Per-period waiting cost C (C > 0).", gt=0)
    ]
    R_provider_profit: Annotated[
        float, Field(description="Profit R for each served agent (R > 0).", gt=0)
    ]

    # Designer's Objective Weight
    alpha_weight: Annotated[
        float,
        Field(description="Weight on agents' welfare (alpha in [0, 1]).", ge=0, le=1),
    ]

    # Process and Policy Definitions
    primitive_process: PrimitiveProcess
    design_rules: EntryExitRule
    queue_discipline: QueueDiscipline
    information_rule: InformationRule

    class Config:
        arbitrary_types_allowed = True


# --- Simulation Output Structure ---


class SimulationResults(BaseModel):
    """Stores the time-weighted results from a single simulation run."""

    total_run_time: Annotated[float, Field(gt=0)]
    time_spent_at_k: Dict[int, float]

    # Core Metrics for Welfare Calculation (Calculated based on time_spent_at_k)
    # The expected number of agents in the system: E[k] = sum(k * p_k)
    expected_queue_length_E_k: float = 0.0
    # The expected flow rate contribution: E[μ_k] = sum(p_k * μ_k)
    expected_service_flow_E_mu_k: float = 0.0

    # Service and exit metrics
    num_served: int = 0
    num_voluntary_abandonment: int = 0  # Agents who left due to incentive constraint
    num_designer_exit: int = 0  # Agents removed by designer exit rule

    # Waiting cost tracking
    total_waiting_cost: float = 0.0
    avg_wait_time_served: float = 0.0  # Average wait time for served agents
