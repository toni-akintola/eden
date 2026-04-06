"""
regretnet_inference.py

Loads a pretrained RegretNet (from dimonenka/optimaler) for the 2-bidder, 2-item
additive uniform setting and wraps it as a Mechanism compatible with auction.py.

Usage
-----
    from regretnet_inference import load_regretnet
    from auction import AuctionInstance, evaluate

    mechanism = load_regretnet(seed=1)          # loads pretrained weights, seed in {0,1,2}
    inst      = AuctionInstance()
    profiles  = inst.sample(n=10_000)
    result    = evaluate(mechanism, profiles, n_misreports=500, penalty=1.0, instance=inst)
    print(result)

Requirements
------------
    pip install torch easydict scipy
    The optimaler repo must be cloned alongside this file (or OPTIMALER_PATH set).

Directory layout expected:
    optimaler/                          <- cloned from github.com/dimonenka/optimaler
        core/nets/additive_net.py
        core/configs/additive_2x2_uniform_config.py
        target_nets/RegretNet/setting_additive_2x2_uniform/seed_{0,1,2}/model_200000
    regretnet_inference.py              <- this file
    auction.py
"""

import os
import sys
import numpy as np
import torch
from copy import deepcopy
from typing import Tuple

# ---------------------------------------------------------------------------
# Path setup — point at the optimaler repo
# ---------------------------------------------------------------------------
OPTIMALER_PATH = os.environ.get(
    "OPTIMALER_PATH", os.path.join(os.path.dirname(__file__), "optimaler")
)
if OPTIMALER_PATH not in sys.path:
    sys.path.insert(0, OPTIMALER_PATH)

try:
    from optimaler.core.nets.additive_net import AdditiveNet
    from optimaler.core.configs.additive_2x2_uniform_config import cfg as _base_cfg
except ImportError as e:
    raise ImportError(
        f"Could not import from optimaler at '{OPTIMALER_PATH}'. "
        "Clone https://github.com/dimonenka/optimaler next to this file, "
        "or set the OPTIMALER_PATH environment variable.\n"
        f"Original error: {e}"
    )

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _build_config():
    """Return a clean 2x2 uniform config."""
    cfg = deepcopy(_base_cfg)
    # these are already set by additive_2x2_uniform_config but be explicit
    cfg.num_agents = 2
    cfg.num_items = 2
    return cfg


def load_regretnet(seed: int = 1, device: str = "cpu") -> Tuple:
    """
    Load a pretrained RegretNet for the additive 2x2 uniform setting.

    Parameters
    ----------
    seed   : int in {0, 1, 2} — which of the three pretrained seeds to load
    device : torch device string

    Returns
    -------
    (alloc_fn, pay_fn) — a Mechanism tuple compatible with auction.py evaluate()

    alloc_fn(v1, v2) -> (a1, a2)
        v1, v2 : np.ndarray shape (2,)  — reported types
        a1, a2 : np.ndarray shape (2,)  — allocation probabilities

    pay_fn(v1, v2) -> (p1, p2)
        p1, p2 : float — payments
    """
    assert seed in (0, 1, 2), f"seed must be 0, 1, or 2; got {seed}"

    weights_path = os.path.join(
        OPTIMALER_PATH,
        "target_nets",
        "RegretNet",
        "setting_additive_2x2_uniform",
        f"seed_{seed}",
        "model_200000",
    )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Pretrained weights not found at:\n  {weights_path}\n"
            "Make sure the optimaler repo is fully cloned (including target_nets/)."
        )

    cfg = _build_config()
    net = AdditiveNet(cfg, device)
    state_dict = torch.load(weights_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    return _wrap_network(net, device)


# ---------------------------------------------------------------------------
# Network wrapper — converts (v1, v2) numpy interface to torch forward pass
# ---------------------------------------------------------------------------


def _wrap_network(net: AdditiveNet, device: str):
    """
    Wrap an AdditiveNet into the (alloc_fn, pay_fn) interface expected by auction.py.

    The network forward pass operates on batches; the wrapper handles the
    single-profile case (called per-profile inside evaluate()'s Python loop)
    by adding/removing the batch dimension.

    Network I/O recap
    -----------------
    Input  : x  shape (batch, n_agents, n_items) = (batch, 2, 2)
    Output : alloc  shape (batch, n_agents, n_items)  — softmax over agents+dummy
             pay    shape (batch, n_agents)            — sigmoid * (v · alloc)
    """

    def _forward_single(v1: np.ndarray, v2: np.ndarray):
        """Run one profile through the network. Returns (alloc, pay) as tensors."""
        # shape: (1, 2, 2)
        x = torch.tensor(
            np.stack([v1, v2], axis=0)[np.newaxis],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            alloc, pay = net(x)  # (1,2,2), (1,2)
        return alloc[0], pay[0]  # (2,2), (2,)

    def alloc_fn(v1: np.ndarray, v2: np.ndarray):
        alloc, _ = _forward_single(v1, v2)
        a1 = alloc[0].cpu().numpy()  # bidder 0's allocations for items 0,1
        a2 = alloc[1].cpu().numpy()  # bidder 1's allocations for items 0,1
        return a1, a2

    def pay_fn(v1: np.ndarray, v2: np.ndarray):
        _, pay = _forward_single(v1, v2)
        p1 = float(pay[0].cpu().item())
        p2 = float(pay[1].cpu().item())
        return p1, p2

    return alloc_fn, pay_fn


# ---------------------------------------------------------------------------
# Batched evaluation (fast path — avoids Python loop over profiles)
# ---------------------------------------------------------------------------


def evaluate_regretnet_batched(
    net_or_seed,
    profiles: np.ndarray,
    n_misreports: int = 500,
    penalty: float = 1.0,
    device: str = "cpu",
    rng=None,
):
    """
    Faster evaluation that runs the full profile batch through the network in
    one forward pass (vectorised allocation/payment), then computes regret with
    a random-misreport search using batched network calls.

    Parameters
    ----------
    net_or_seed : AdditiveNet or int seed (0/1/2)
    profiles    : np.ndarray shape (n, 2, 2)
    n_misreports: number of random misreport candidates per bidder per profile
    penalty     : lambda in fitness = revenue - penalty * regret
    device      : torch device string
    rng         : optional np.random.Generator

    Returns
    -------
    dict with keys: revenue, regret, welfare, fitness
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- load network if seed given ---
    if isinstance(net_or_seed, int):
        alloc_fn, pay_fn = load_regretnet(seed=net_or_seed, device=device)
        # recover the underlying net for batched calls
        cfg = _build_config()
        net = AdditiveNet(cfg, device)
        weights_path = os.path.join(
            OPTIMALER_PATH,
            "target_nets",
            "RegretNet",
            "setting_additive_2x2_uniform",
            f"seed_{net_or_seed}",
            "model_200000",
        )
        net.load_state_dict(torch.load(weights_path, map_location=device))
        net.eval()
    else:
        net = net_or_seed

    n = len(profiles)

    def _batch_forward(bids: np.ndarray):
        """bids: (batch, 2, 2) -> alloc (batch,2,2), pay (batch,2)"""
        x = torch.tensor(bids, dtype=torch.float32, device=device)
        with torch.no_grad():
            alloc, pay = net(x)
        return alloc.cpu().numpy(), pay.cpu().numpy()

    # --- truthful pass ---
    alloc_t, pay_t = _batch_forward(profiles)  # (n,2,2), (n,2)
    revenue = pay_t.sum(axis=1).mean()  # scalar
    welfare = (profiles * alloc_t).sum(axis=(1, 2)).mean()  # v_i · a_i summed

    # --- regret estimation via random misreport sampling ---
    # For each bidder, sample n_misreports alternative reports and find max utility gain
    regrets = np.zeros((n, 2))

    for bidder_idx in range(2):
        # truthful utility for this bidder across all profiles
        u_truth = (profiles[:, bidder_idx, :] * alloc_t[:, bidder_idx, :]).sum(
            axis=1
        ) - pay_t[
            :, bidder_idx
        ]  # shape (n,)

        best_gain = np.zeros(n)

        for _ in range(n_misreports):
            # sample one misreport per profile (uniform [0,1] per item)
            mr = rng.uniform(0, 1, size=(n, 2))  # (n, 2)

            # build misreported profiles: replace bidder_idx's type with mr
            mis_profiles = profiles.copy()
            mis_profiles[:, bidder_idx, :] = mr

            alloc_mr, pay_mr = _batch_forward(mis_profiles)

            # utility at TRUE value but misreported bid
            u_mr = (profiles[:, bidder_idx, :] * alloc_mr[:, bidder_idx, :]).sum(
                axis=1
            ) - pay_mr[
                :, bidder_idx
            ]  # shape (n,)

            gain = np.maximum(0.0, u_mr - u_truth)
            best_gain = np.maximum(best_gain, gain)

        regrets[:, bidder_idx] = best_gain

    mean_regret = regrets.mean()
    fitness = float(revenue) - penalty * float(mean_regret)

    return {
        "revenue": float(revenue),
        "regret": float(mean_regret),
        "welfare": float(welfare),
        "fitness": fitness,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    sys.path.insert(0, os.path.dirname(__file__))
    from auction import AuctionInstance, evaluate, _wrap_baseline, second_price_bundle

    print("=== RegretNet inference smoke test ===\n")

    rng = np.random.default_rng(42)
    inst = AuctionInstance()

    # small sample for the slow per-profile evaluator
    profiles_small = inst.sample(n=500, rng=rng)
    # larger sample for the fast batched evaluator
    profiles_large = inst.sample(n=5_000, rng=rng)

    for seed in (0, 1, 2):
        print(f"--- Seed {seed} (per-profile evaluator, n=500) ---")
        mechanism = load_regretnet(seed=seed)
        t0 = time.time()
        result = evaluate(
            mechanism,
            profiles_small,
            n_misreports=200,
            penalty=1.0,
            rng=np.random.default_rng(seed),
            instance=inst,
        )
        print(result)
        print(f"  elapsed: {time.time()-t0:.1f}s\n")

    print("--- Seed 1 (batched evaluator, n=5000, n_misreports=300) ---")
    t0 = time.time()
    result_fast = evaluate_regretnet_batched(
        1, profiles_large, n_misreports=300, penalty=1.0, rng=np.random.default_rng(1)
    )
    print(result_fast)
    print(f"  elapsed: {time.time()-t0:.1f}s\n")

    print("--- Baseline: grand-bundle second-price (n=500) ---")
    bundle_mech = _wrap_baseline(second_price_bundle)
    print(
        evaluate(
            bundle_mech,
            profiles_small,
            n_misreports=200,
            penalty=1.0,
            rng=np.random.default_rng(0),
            instance=inst,
        )
    )
