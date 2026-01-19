from evolve_types import CheTercieuxQueueModel, SimulationResults


def evaluate_designer_performance(
    model: CheTercieuxQueueModel, results: SimulationResults
) -> float:
    """
    Calculates the designer's objective score (W) using the calculated expected
    values, which approximates the long-run time average.

    W = (1-alpha)R * E[μ_k] + alpha * (E[μ_k]V - E[k]C) - D * E[R_E]

    Where E[R_E] is the expected exit rate (time-weighted average), not the actual
    number of exits. This ensures consistent penalization of high exit rates.
    """

    alpha = model.alpha_weight
    R = model.R_provider_profit
    V = model.V_surplus
    C = model.C_waiting_cost
    D = model.D_exit_disutility

    E_mu_k = results.expected_service_flow_E_mu_k
    E_k = results.expected_queue_length_E_k
    E_R_E = results.expected_exit_rate_E_R_E

    # 1. Provider's Profit Term: (1 - α) * R * E[μ_k]
    provider_profit_term = (1.0 - alpha) * R * E_mu_k

    # 2. Agent's Utility Term: α * (E[μ_k] * V - E[k] * C) - D * E[R_E]
    # Note: The expected flow rate E[μ_k] * V is the expected gross surplus
    # The expected cost E[k] * C is the expected waiting cost (as k agents incur cost C per period)
    # Subtract disutility D times expected exit rate E[R_E] (time-weighted average exit rate)
    # This penalizes mechanisms with high exit rates, even if few exits occur in a finite simulation
    # Note: E[R_E] is in units of exits per unit time, so D should be disutility per exit per unit time
    # If D is meant to be disutility per exit, we'd multiply by simulation time T, but typically
    # we think of D as a rate penalty (disutility per exit per unit time)
    total_disutility = D * E_R_E
    agent_utility_term = alpha * (E_mu_k * V - E_k * C) - total_disutility
    # print(f"Total disutility: {total_disutility}")
    # print(f"Expected exit rate: {results.expected_exit_rate_E_R_E}")
    # print(f"Number of designer exits: {results.num_designer_exit}")

    W_score = provider_profit_term + agent_utility_term

    return W_score
