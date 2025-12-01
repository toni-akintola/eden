from evolve_types import CheTercieuxQueueModel, SimulationResults


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
