from evolve_types import CheTercieuxQueueModel, QueueDiscipline, InformationRule
from queue_simulator import QueueSimulator


def test_simulator():
    # --- Example of running the simulator for testing ---
    # --- Example M/M/1 Setup (Optimal Policy is no-rationing/cutoff=infinity) ---
    def lambda_k_mm1(k: int) -> float:
        return 2.0  # Lambda = 2

    # FIXED: mu_k_mm1 must return 0.0 when k=0, as the server is idle.
    # The analytical definition of service rate for M/M/1 is min(k, c) * mu, where c=1.
    def mu_k_mm1(k: int) -> float:
        return 3.0 if k > 0 else 0.0  # Mu = 3 only if server is busy

    def x_k_always_join(k: int) -> float:
        return 1.0  # Always join

    def yz_k_l_never_remove(k: int, l: int) -> tuple[float, float]:
        return (0.0, 0.0)  # Never remove

    example_model = CheTercieuxQueueModel(
        V_surplus=10.0,
        C_waiting_cost=1.0,
        R_provider_profit=5.0,
        alpha_weight=0.5,
        primitive_process={
            "arrival_rate_fn": lambda_k_mm1,
            "service_rate_fn": mu_k_mm1,
        },
        design_rules={
            "entry_rule_fn": x_k_always_join,
            "exit_rule_fn": yz_k_l_never_remove,
        },
        queue_discipline=QueueDiscipline.FCFS,
        information_rule=InformationRule.NO_INFORMATION_BEYOND_REC,
    )

    simulator = QueueSimulator(example_model)
    # INCREASED MAX_TIME to 100,000 to better approximate steady-state
    print("Starting simulation (T=1,000,000)...")
    sim_results = simulator.run_simulation(max_time=1000000.0)

    print("\n--- Simulation Results ---")
    print(f"Total Run Time: {sim_results.total_run_time:.2f}")
    print(f"Total Served: {sim_results.num_served}")

    # Check the estimated expected values
    print(f"\n--- Estimated Expected Values (Target E[k]=2.0, Target E[μ_k]=2.0) ---")
    print(f"E[k] (Queue Length): {sim_results.expected_queue_length_E_k:.4f}")
    print(f"E[μ_k] (Service Flow): {sim_results.expected_service_flow_E_mu_k:.4f}")

    # Calculate the Score (Welfare)
    welfare_score = QueueSimulator.evaluate_designer_performance(
        example_model, sim_results
    )
    print(f"\n--- Evaluation ---")
    print(f"Estimated Long-Run Social Welfare (W): {welfare_score:.4f}")

    # Expected analytical W for M/M/1:
    # W = (1-0.5)*5*2.0 + 0.5 * (2.0*10 - 2.0*1.0) = 5.0 + 0.5 * (18.0) = 14.0
    print(f"(Expected W for M/M/1: 14.0)")


if __name__ == "__main__":
    test_simulator()
