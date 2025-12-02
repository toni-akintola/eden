from typing import Callable, Annotated, Dict
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
    """Represents an agent in the system."""

    def __init__(self, arrival_time: float):
        self.arrival_time = arrival_time
        # In a full simulation, you might add self.type_V, self.type_C, etc.

    def __repr__(self):
        return f"Agent(arr={self.arrival_time:.2f})"


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

    # Optionally track other metrics for debugging/analysis
    num_served: int = 0
    total_waiting_cost: float = 0.0
