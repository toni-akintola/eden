import random
import math
from collections import deque
from typing import Dict, Deque, List, Union

# Import all necessary types from the local file
from evolve_types import (
    CheTercieuxQueueModel,
    SimulationResults,
    Agent,
    QueueDiscipline,
    InformationRule,  # Added for completeness
)


class QueueSimulator:
    """
    Implements a Discrete-Event Simulation (DES) based on the Che and Tercieux
    queueing model, allowing for state-dependent arrival/service/entry/exit rules.
    """

    def __init__(self, model: CheTercieuxQueueModel):
        self.model = model
        self.reset_state()

    def reset_state(self):
        """Resets the simulation state variables."""
        self.current_time: float = 0.0
        self.queue_agents: Deque[Agent] = deque()
        # time_spent_at_k[k] stores the total time spent in state k
        self.time_spent_at_k: Dict[int, float] = {0: 0.0}
        self.num_served: int = 0

    def _draw_next_event_time(self, total_rate: float) -> float:
        """
        Calculates the time until the next event using the Exponential distribution.

        Args:
            total_rate: The sum of all possible event rates (Lambda).

        Returns:
            The time step delta_t.
        """
        if total_rate == 0:
            return float("inf")

        # Time = -ln(U) / Lambda, where U is uniform(0, 1)
        return -math.log(random.random()) / total_rate

    def run_simulation(self, max_time: float) -> SimulationResults:
        """
        Runs the Discrete-Event Simulation (DES) loop.
        """
        self.reset_state()

        while self.current_time < max_time:
            # k is the state (queue length) for the next delta_t interval
            k = len(self.queue_agents)

            # --- 1. Calculate Event Rates ---

            # Get designer-defined functions
            lambda_k = self.model.primitive_process.arrival_rate_fn(k)
            mu_k = self.model.primitive_process.service_rate_fn(k)
            x_k = self.model.design_rules.entry_rule_fn(k)

            # 1. Effective Arrival Rate (R_A): Arrival * Entry Prob
            R_A = lambda_k * x_k

            # 2. Service Rate (R_S): Max Service Rate (Active only if k > 0)
            R_S = mu_k if k > 0 else 0.0

            # 3. Designer Exit Rate (R_E): Sum of removal rates for all agents
            R_E = 0.0
            agent_removal_rates = []
            for l in range(k):
                # l is 0-indexed position, the paper uses 1-indexed l
                y_k_l, z_k_l = self.model.design_rules.exit_rule_fn(k, l + 1)
                R_E += y_k_l
                agent_removal_rates.append(y_k_l)

            # 4. Total Rate (Lambda)
            total_rate = R_A + R_S + R_E

            if total_rate == 0.0:
                print(
                    f"Simulation ended early at T={self.current_time:.2f}. Total rate is zero."
                )
                break

            # --- 2. Determine Time to Next Event (Δt) ---
            delta_t = self._draw_next_event_time(total_rate)

            # --- 3. Update Time and Record Time-in-State ---

            # Bound delta_t so we don't exceed max_time
            if self.current_time + delta_t > max_time:
                delta_t = max_time - self.current_time
                self.current_time = max_time
                # Record the remaining time in the current state k
                self.time_spent_at_k[k] = self.time_spent_at_k.get(k, 0.0) + delta_t
                break

            # Record time spent in state k
            self.current_time += delta_t
            self.time_spent_at_k[k] = self.time_spent_at_k.get(k, 0.0) + delta_t

            # --- 4. Select and Execute the Event ---

            U = random.uniform(0, total_rate)

            if U <= R_A:
                # --- EVENT: Agent Arrives and Joins ---
                new_agent = Agent(arrival_time=self.current_time)
                self.queue_agents.append(new_agent)

            elif U <= R_A + R_S:
                # --- EVENT: Service Completion ---

                served_agent: Agent
                if not self.queue_agents:
                    continue

                # Keeping track of served agent in case we want to log/debug later
                if self.model.queue_discipline == QueueDiscipline.FCFS:
                    served_agent = self.queue_agents.popleft()
                elif self.model.queue_discipline == QueueDiscipline.LIFO:
                    served_agent = self.queue_agents.pop()
                elif self.model.queue_discipline == QueueDiscipline.SIRO:
                    idx = random.randrange(len(self.queue_agents))
                    served_agent = self.queue_agents.pop(idx)

                self.num_served += 1

            else:  # U > R_A + R_S
                # --- EVENT: Designer-Induced Exit ---
                U_exit = U - (R_A + R_S)  # Remap U to [0, R_E]

                cumulative_rate = 0.0
                removed_index = -1

                for idx, rate in enumerate(agent_removal_rates):
                    cumulative_rate += rate
                    if U_exit <= cumulative_rate:
                        removed_index = idx
                        break

                if removed_index != -1:
                    self.queue_agents.pop(removed_index)

        # After loop, compile results and calculate expected values
        return self._calculate_final_results(self.current_time, self.time_spent_at_k)

    def _calculate_final_results(
        self, T: float, time_spent_at_k: Dict[int, float]
    ) -> SimulationResults:
        """Calculates expected values (E[k] and E[μ_k]) from time-in-state data."""

        E_k_sum = 0.0
        E_mu_k_sum = 0.0
        mu_k_fn = self.model.primitive_process.service_rate_fn

        for k, T_k in time_spent_at_k.items():
            if T_k > 0:
                p_k_estimate = T_k / T

                # IMPORTANT: mu_k must reflect the actual service capacity used in state k.
                # The model's mu_k_fn returns the MAX service rate, but only if k > 0.
                # In the simulation: mu_k should be 0 if k=0, and mu_k_fn(k) if k>0.

                # Let's rely on the definition: E[mu_k] = sum(p_k * mu_k_fn(k))
                # The provided mu_k_fn for M/M/1 was faulty, assuming mu_k_fn(0) should be 0.
                # If we trust the function *passed in*, we use it:
                mu_k_raw = mu_k_fn(k)

                # BUT, since the definition of the M/M/1 example was wrong and must be fixed,
                # we must calculate E[mu_k] correctly: the *actual* service rate is 0 if k=0.
                # The designer's mu_k_fn(k) is the *potential* rate.

                # Fix: The effective rate is the user's rate only if k > 0, otherwise 0.
                mu_k_effective = mu_k_raw if k > 0 else 0.0

                E_k_sum += k * p_k_estimate
                E_mu_k_sum += p_k_estimate * mu_k_effective

        return SimulationResults(
            total_run_time=T,
            time_spent_at_k=time_spent_at_k,
            num_served=self.num_served,
            expected_queue_length_E_k=E_k_sum,
            expected_service_flow_E_mu_k=E_mu_k_sum,
        )

    @staticmethod
    def evaluate_designer_performance(
        model: CheTercieuxQueueModel, results: SimulationResults
    ) -> float:
        """
        Calculates the designer's objective score (W) using the calculated expected
        values, which approximates the long-run time average.

        W = (1-alpha)R * E[μ_k] + alpha * (E[μ_k]V - E[k]C)
        """

        alpha = model.alpha_weight
        R = model.R_provider_profit
        V = model.V_surplus
        C = model.C_waiting_cost

        E_mu_k = results.expected_service_flow_E_mu_k
        E_k = results.expected_queue_length_E_k

        # 1. Provider's Profit Term: (1 - α) * R * E[μ_k]
        provider_profit_term = (1.0 - alpha) * R * E_mu_k

        # 2. Agent's Utility Term: α * (E[μ_k] * V - E[k] * C)
        # Note: The expected flow rate E[μ_k] * V is the expected gross surplus
        # The expected cost E[k] * C is the expected waiting cost (as k agents incur cost C per period)
        agent_utility_term = alpha * (E_mu_k * V - E_k * C)

        W_score = provider_profit_term + agent_utility_term

        return W_score
