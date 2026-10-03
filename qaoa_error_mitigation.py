"""Readout-error mitigation benchmark for the Quantum Portfolio Optimizer.

Runs the same noisy QAOA portfolio problem as ``qaoa_noisy_simulator.py`` and
post-processes the measured QAOA probability distribution with an inverse
readout-confusion matrix. This mitigates the *measurement/readout* component
of the noise model; depolarizing gate noise remains present.

The script reports both the raw noisy solution and the highest-probability
feasible portfolio after readout mitigation so it can be compared with the
ideal and classical reference paths.
"""

from __future__ import annotations

import time

import numpy as np
from qiskit.primitives import BackendSamplerV2
from qiskit_aer import AerSimulator
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qaoa_noisy_simulator import READOUT_ERROR, SHOTS, build_noise_model
from qaoa_simulator import ASSETS, BUDGET, SEED, build_qubo


N_QUBITS = len(ASSETS)


def readout_assignment_matrix(n_qubits: int, error_rate: float) -> np.ndarray:
    """Return the independent symmetric readout assignment matrix.

    For one qubit, rows are measured states and columns are true states:

        [[P(0|0), P(0|1)],
         [P(1|0), P(1|1)]]

    The full-system matrix is the Kronecker product over all qubits.
    """
    one_qubit = np.array(
        [
            [1.0 - error_rate, error_rate],
            [error_rate, 1.0 - error_rate],
        ],
        dtype=float,
    )

    matrix = np.array([[1.0]])
    for _ in range(n_qubits):
        matrix = np.kron(matrix, one_qubit)
    return matrix


def vector_index(bits: np.ndarray) -> int:
    """Convert a binary vector to a probability-vector index."""
    bit_string = "".join(str(int(round(v))) for v in bits)
    return int(bit_string, 2)


def index_bits(index: int, n_qubits: int) -> np.ndarray:
    """Convert a probability-vector index back into a binary vector."""
    return np.array([int(c) for c in f"{index:0{n_qubits}b}"], dtype=int)


def measured_distribution(result) -> np.ndarray:
    """Build a normalized measured probability vector from QAOA samples."""
    probabilities = np.zeros(2**N_QUBITS, dtype=float)

    samples = getattr(result, "samples", None)
    if not samples:
        raise RuntimeError("QAOA result did not expose optimization samples for mitigation.")

    for sample in samples:
        probabilities[vector_index(np.asarray(sample.x))] += float(sample.probability)

    total = probabilities.sum()
    if total <= 0:
        raise RuntimeError("Measured QAOA sample probabilities sum to zero.")
    return probabilities / total


def mitigate_readout(probabilities: np.ndarray) -> np.ndarray:
    """Apply pseudo-inverse readout mitigation, then project to probabilities."""
    assignment = readout_assignment_matrix(N_QUBITS, READOUT_ERROR)
    corrected = np.linalg.pinv(assignment) @ probabilities

    # Matrix inversion can create small negative quasi-probabilities. For this
    # portfolio-selection benchmark we project them back onto the probability
    # simplex before selecting the most likely feasible bitstring.
    corrected = np.clip(corrected, 0.0, None)
    total = corrected.sum()
    if total <= 0:
        raise RuntimeError("Mitigated probability distribution is empty.")
    return corrected / total


def highest_probability_feasible(probabilities: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the most probable bitstring that satisfies the K-asset budget."""
    ranked = np.argsort(probabilities)[::-1]
    for index in ranked:
        bits = index_bits(int(index), N_QUBITS)
        if int(bits.sum()) == BUDGET:
            return bits, float(probabilities[index])
    raise RuntimeError("No feasible portfolio found in mitigated distribution.")


def run_mitigated_qaoa() -> None:
    """Run noisy QAOA and apply readout-error mitigation to its samples."""
    np.random.seed(SEED)

    qp = build_qubo()
    backend = AerSimulator(
        noise_model=build_noise_model(),
        seed_simulator=SEED,
    )
    sampler = BackendSamplerV2(
        backend=backend,
        options={
            "default_shots": SHOTS,
            "seed_simulator": SEED,
        },
    )

    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=100),
        reps=1,
        initial_point=np.array([0.5, 0.5]),
    )
    optimizer = MinimumEigenOptimizer(qaoa)

    started = time.perf_counter()
    result = optimizer.solve(qp)
    measured = measured_distribution(result)
    mitigated = mitigate_readout(measured)
    mitigated_bits, mitigated_probability = highest_probability_feasible(mitigated)
    runtime = time.perf_counter() - started

    raw_bits = result.x.astype(int)
    raw_assets = [ASSETS[i] for i, value in enumerate(raw_bits) if value == 1]
    mitigated_assets = [
        ASSETS[i] for i, value in enumerate(mitigated_bits) if value == 1
    ]
    mitigated_objective = float(qp.objective.evaluate(mitigated_bits))

    print("=== Error-Mitigated QAOA Portfolio Simulator ===")
    print(f"Assets: {ASSETS}")
    print(f"Budget: {BUDGET}")
    print(f"Shots: {SHOTS}")
    print(f"Seed: {SEED}")
    print("Mitigation: inverse readout assignment matrix (pseudo-inverse)")
    print("Gate depolarizing noise: NOT mitigated")
    print()
    print(f"Raw noisy selected assets: {raw_assets}")
    print(f"Raw noisy binary solution: {raw_bits.tolist()}")
    print(f"Raw noisy objective value: {result.fval:.6f}")
    print()
    print(f"Mitigated selected assets: {mitigated_assets}")
    print(f"Mitigated binary solution: {mitigated_bits.tolist()}")
    print(f"Mitigated objective value: {mitigated_objective:.6f}")
    print(f"Mitigated solution probability: {mitigated_probability:.6f}")
    print(f"Readout error corrected: {READOUT_ERROR:.3%}")
    print(f"Runtime including mitigation: {runtime:.3f} s")


if __name__ == "__main__":
    run_mitigated_qaoa()
