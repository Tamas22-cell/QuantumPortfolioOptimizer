"""Ideal QAOA simulator for the Quantum Portfolio Optimizer.

This first simulator step runs a small binary portfolio-selection problem
with Qiskit's statevector sampler (noise-free). It is intentionally kept
small and deterministic so we can benchmark noisy simulation later.
"""

from __future__ import annotations

import time

import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer


ASSETS = ["BTC", "ETH", "SOL", "NVDA"]
EXPECTED_RETURNS = np.array([0.18, 0.15, 0.22, 0.12])
COVARIANCE = np.array(
    [
        [0.090, 0.050, 0.060, 0.020],
        [0.050, 0.070, 0.055, 0.018],
        [0.060, 0.055, 0.120, 0.022],
        [0.020, 0.018, 0.022, 0.045],
    ]
)

RISK_AVERSION = 0.50
BUDGET = 2
SEED = 42


def build_qubo() -> QuadraticProgram:
    """Build a binary mean-variance portfolio model."""
    qp = QuadraticProgram("quantum_portfolio_optimizer")

    for asset in ASSETS:
        qp.binary_var(asset)

    # Minimize: risk - lambda * expected return
    linear = {
        asset: -RISK_AVERSION * float(EXPECTED_RETURNS[i])
        for i, asset in enumerate(ASSETS)
    }

    quadratic: dict[tuple[str, str], float] = {}
    for i, asset_i in enumerate(ASSETS):
        for j, asset_j in enumerate(ASSETS):
            if j < i:
                continue
            coefficient = float(COVARIANCE[i, j])
            if i != j:
                coefficient *= 2.0
            quadratic[(asset_i, asset_j)] = coefficient

    qp.minimize(linear=linear, quadratic=quadratic)
    qp.linear_constraint(
        linear={asset: 1 for asset in ASSETS},
        sense="==",
        rhs=BUDGET,
        name="budget",
    )

    return qp


def run_ideal_qaoa() -> None:
    """Run noise-free QAOA and print a reproducible benchmark summary."""
    np.random.seed(SEED)

    qp = build_qubo()
    sampler = StatevectorSampler(seed=SEED)
    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=100),
        reps=1,
        initial_point=np.array([0.5, 0.5]),
    )
    optimizer = MinimumEigenOptimizer(qaoa)

    started = time.perf_counter()
    result = optimizer.solve(qp)
    runtime = time.perf_counter() - started

    selected_assets = [
        ASSETS[i] for i, value in enumerate(result.x) if int(round(value)) == 1
    ]

    print("=== Ideal QAOA Portfolio Simulator ===")
    print(f"Assets: {ASSETS}")
    print(f"Budget: {BUDGET}")
    print(f"Selected assets: {selected_assets}")
    print(f"Binary solution: {result.x.astype(int).tolist()}")
    print(f"Objective value: {result.fval:.6f}")
    print(f"Runtime: {runtime:.3f} s")
    print(f"QAOA reps: 1")
    print(f"Seed: {SEED}")


if __name__ == "__main__":
    run_ideal_qaoa()
