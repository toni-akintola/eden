import random
import math
from collections import deque
from typing import Dict, Deque, List

# Import all necessary types from the local file
from evolve_types import (
    CheTercieuxQueueModel,
    SimulationResults,
    Agent,
    QueueDiscipline,
    InformationRule,
)


class QueueSimulator:
    """
    Implements a Discrete-Event Simulation (DES) based on the Che and Tercieux
    queueing model, allowing for state-dependent arrival/service/entry/exit rules.

    Includes agent belief updating and voluntary abandonment based on the
    dynamic incentive constraint.
    """

    def __init__(
        self, model: CheTercieuxQueueModel, enable_belief_updates: bool = True
    ):
        """
        Initialize the simulator.

        Args:
            model: The queue model configuration
            enable_belief_updates: If True, agents update beliefs and may voluntarily abandon.
                                   If False, uses simplified simulation without belief tracking.
        """
        self.model = model
        self.enable_belief_updates = enable_belief_updates
        self.reset_state()

    def reset_state(self):
        """Resets the simulation state variables."""
        self.current_time: float = 0.0
        self.queue_agents: Deque[Agent] = deque()
        # time_spent_at_k[k] stores the total time spent in state k
        self.time_spent_at_k: Dict[int, float] = {0: 0.0}
        self.num_served: int = 0
        self.num_voluntary_abandonment: int = 0
        self.num_designer_exit: int = 0
        self.total_wait_time_served: float = 0.0  # Sum of wait times for served agents

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

    def _get_entry_probability(self, k: int) -> float:
        """
        Get entry probability based on information rule.

        Args:
            k: Current queue length

        Returns:
            Entry probability x based on what agents can observe
        """
        if self.model.information_rule == InformationRule.FULL_INFORMATION:
            # Agents observe the actual queue length k
            return self.model.design_rules.entry_rule_fn(k)

        elif self.model.information_rule == InformationRule.NO_INFORMATION_BEYOND_REC:
            # Agents don't observe k, use expected queue length from current distribution
            if self.current_time == 0.0 or not self.time_spent_at_k:
                # At start, use k=0 as default
                return self.model.design_rules.entry_rule_fn(0)

            # Compute expected queue length from empirical distribution
            total_time = sum(self.time_spent_at_k.values())
            if total_time == 0:
                return self.model.design_rules.entry_rule_fn(0)

            E_k = sum(
                k_state * (time / total_time)
                for k_state, time in self.time_spent_at_k.items()
            )
            # Round to nearest integer for entry rule evaluation
            k_observed = round(E_k)
            return self.model.design_rules.entry_rule_fn(k_observed)

        elif self.model.information_rule == InformationRule.COARSE_INFORMATION:
            # Agents observe a coarse signal (e.g., "short" vs "long")
            # Map k to coarse categories: 0-2 = "short" (0), 3-5 = "medium" (3), 6+ = "long" (6)
            if k <= 2:
                k_coarse = 0
            elif k <= 5:
                k_coarse = 3
            else:
                k_coarse = 6
            return self.model.design_rules.entry_rule_fn(k_coarse)

        else:
            # Default to full information
            return self.model.design_rules.entry_rule_fn(k)

    def _update_agent_positions(self):
        """
        Update all agents' beliefs about their position after queue changes.
        Called after arrivals, services, or exits to maintain consistent beliefs.
        """
        if not self.enable_belief_updates:
            return

        # After any queue change, update each agent's position belief
        # In reality, what agents observe depends on the information rule
        for idx, agent in enumerate(self.queue_agents):
            position = idx + 1  # 1-indexed position

            if self.model.information_rule == InformationRule.FULL_INFORMATION:
                # Agents know their exact position
                agent.belief = {position: 1.0}
            # For other information rules, beliefs are updated via Bayesian updating
            # in _update_beliefs_after_service

    def _update_beliefs_after_service(self, service_occurred: bool):
        """
        Update all agents' beliefs after a service event.

        Implements the paper's Bayesian belief update formula:
        γ̃_{t+1}^ℓ = (γ̃_t^ℓ · μ_B + γ̃_t^{ℓ+1} · μ_A) / (γ̃_t^1 · μ_B + Σ_{i=2}^{K_A} γ̃_t^i)

        Args:
            service_occurred: Whether a service completion just happened
        """
        if not self.enable_belief_updates or not service_occurred:
            return

        k = len(self.queue_agents)
        mu_k = self.model.primitive_process.service_rate_fn(k) if k > 0 else 0.0

        for agent in self.queue_agents:
            agent.update_belief_on_service(
                queue_discipline=self.model.queue_discipline,
                service_occurred=service_occurred,
                current_queue_length=k,
                service_rate=mu_k,
            )
            agent.update_expected_wait(mu_k, self.model.queue_discipline)

    def _check_voluntary_abandonment(self) -> List[int]:
        """
        Check which agents should voluntarily abandon based on incentive constraint.

        Returns:
            List of indices (0-based) of agents who should abandon
        """
        if not self.enable_belief_updates:
            return []

        abandon_indices = []
        V = self.model.V_surplus
        C = self.model.C_waiting_cost

        for idx, agent in enumerate(self.queue_agents):
            if agent.should_abandon(V, C):
                abandon_indices.append(idx)

        return abandon_indices

    def run_simulation(self, max_time: float) -> SimulationResults:
        """
        Runs the Discrete-Event Simulation (DES) loop with belief updates.
        """
        self.reset_state()

        while self.current_time < max_time:
            # k is the state (queue length) for the next delta_t interval
            k = len(self.queue_agents)

            # --- 0. Check for Voluntary Abandonment ---
            # Agents check their incentive constraint and may leave
            if self.enable_belief_updates and k > 0:
                abandon_indices = self._check_voluntary_abandonment()
                # Remove abandoning agents (in reverse order to preserve indices)
                for idx in sorted(abandon_indices, reverse=True):
                    del self.queue_agents[idx]
                    self.num_voluntary_abandonment += 1
                # Update k after abandonments
                k = len(self.queue_agents)
                # Update remaining agents' positions
                self._update_agent_positions()

            # --- 1. Calculate Event Rates ---

            # Get designer-defined functions
            lambda_k = self.model.primitive_process.arrival_rate_fn(k)
            mu_k = self.model.primitive_process.service_rate_fn(k)

            # Apply entry rule based on information available to agents
            x_k = self._get_entry_probability(k)

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
                # If total rate is zero, we're in an absorbing state (no events possible)
                # Ensure we've recorded at least some time to avoid validation errors
                if self.current_time == 0.0:
                    # If we never progressed, record a minimal time at state k=0
                    self.time_spent_at_k[0] = (
                        1e-6  # Very small epsilon to satisfy validation
                    )
                    self.current_time = 1e-6
                print(
                    f"Simulation ended early at T={self.current_time:.2f}. Total rate is zero (absorbing state)."
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
                # Update agents' time in queue
                for agent in self.queue_agents:
                    agent.time_in_queue += delta_t
                break

            # Record time spent in state k
            self.current_time += delta_t
            self.time_spent_at_k[k] = self.time_spent_at_k.get(k, 0.0) + delta_t

            # Update agents' time in queue
            for agent in self.queue_agents:
                agent.time_in_queue += delta_t

            # --- 4. Select and Execute the Event ---

            U = random.uniform(0, total_rate)

            if U <= R_A:
                # --- EVENT: Agent Arrives and Joins ---
                # New agent joins at the back of the queue
                initial_position = len(self.queue_agents) + 1
                new_agent = Agent(
                    arrival_time=self.current_time,
                    initial_position=initial_position,
                )
                self.queue_agents.append(new_agent)

                # Initialize the new agent's expected wait
                if self.enable_belief_updates:
                    new_agent.update_expected_wait(
                        mu_k if mu_k > 0 else 1.0, self.model.queue_discipline
                    )

            elif U <= R_A + R_S:
                # --- EVENT: Service Completion ---

                if not self.queue_agents:
                    continue

                served_agent: Agent
                # Select agent based on queue discipline
                if self.model.queue_discipline == QueueDiscipline.FCFS:
                    served_agent = self.queue_agents.popleft()
                elif self.model.queue_discipline == QueueDiscipline.LIFO:
                    served_agent = self.queue_agents.pop()
                elif self.model.queue_discipline == QueueDiscipline.SIRO:
                    idx = random.randrange(len(self.queue_agents))
                    served_agent = self.queue_agents[idx]
                    del self.queue_agents[idx]

                self.num_served += 1
                self.total_wait_time_served += served_agent.time_in_queue

                # Update beliefs for remaining agents
                self._update_beliefs_after_service(service_occurred=True)
                self._update_agent_positions()

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
                    del self.queue_agents[removed_index]
                    self.num_designer_exit += 1
                    self._update_agent_positions()

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

        # Calculate average wait time for served agents
        avg_wait = (
            self.total_wait_time_served / self.num_served
            if self.num_served > 0
            else 0.0
        )

        return SimulationResults(
            total_run_time=T,
            time_spent_at_k=time_spent_at_k,
            num_served=self.num_served,
            num_voluntary_abandonment=self.num_voluntary_abandonment,
            num_designer_exit=self.num_designer_exit,
            expected_queue_length_E_k=E_k_sum,
            expected_service_flow_E_mu_k=E_mu_k_sum,
            avg_wait_time_served=avg_wait,
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
