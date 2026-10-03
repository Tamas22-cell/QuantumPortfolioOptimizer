"""IBM Quantum hardware runner for the Quantum Portfolio Optimizer.

This module takes the same small portfolio QUBO used by the simulator paths,
converts it to an Ising cost Hamiltonian, builds a p=1 QAOA circuit, transpiles
it for a real IBM Quantum backend, executes it with the IBM Sampler primitive,
and reports hardware metrics plus the best feasible sampled portfolio.

Important: this file is hardware-ready code. It does not claim that a hardware
run occurred until the script is executed with a valid IBM Quantum account.

Authentication (recommended): save your IBM Quantum account once with
QiskitRuntimeService.save_account(...), then run this script normally.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_ibm_runtime import QiskitRuntimeService

try:
    # qiskit-ibm-runtime >= 0.50 client-side Sampler.
    from qiskit_ibm_runtime.executor_sampler import Sampler
except ImportError:  # pragma: no cover - compatibility with older installs
    from qiskit_ibm_runtime import SamplerV2 as Sampler

from qaoa_simulator import (
    ASSETS,
    BUDGET,
    COVARIANCE,
    EXPECTED_RETURNS,
    RISK_AVERSION,
    SEED,
    build_qubo,
)

SHOTS = 2048
QAOA_REPS = 1
TRANSPILER_OPTIMIZATION_LEVEL = 3


def portfolio_objective(bits: np.ndarray) -> float:
    """Evaluate the original constrained portfolio objective."""
    risk = float(bits @ COVARIANCE @ bits)
    reward = float(EXPECTED_RETURNS @ bits)
    return risk - RISK_AVERSION * reward


def bitstring_to_vector(bitstring: str) -> np.ndarray:
    """Convert Qiskit's displayed classical bit order to variable order."""
    clean = bitstring.replace(" ", "")
    return np.array([int(bit) for bit in clean[::-1][: len(ASSETS)]], dtype=int)


def best_feasible_from_counts(counts: dict[str, int]) -> tuple[str, np.ndarray, int, float] | None:
    """Return the most frequent feasible portfolio from hardware samples."""
    feasible: list[tuple[str, np.ndarray, int, float]] = []
    for bitstring, count in counts.items():
        bits = bitstring_to_vector(bitstring)
        if int(bits.sum()) != BUDGET:
            continue
        feasible.append((bitstring, bits, int(count), portfolio_objective(bits)))

    if not feasible:
        return None

    # Prefer the best objective; break ties with higher sample count.
    return min(feasible, key=lambda item: (item[3], -item[2]))


def build_hardware_circuit(beta: float, gamma: float):
    """Build the fixed-parameter QAOA circuit for one hardware sampling run."""
    constrained_qp = build_qubo()
    qubo = QuadraticProgramToQubo().convert(constrained_qp)
    cost_operator, offset = qubo.to_ising()

    ansatz = QAOAAnsatz(cost_operator=cost_operator, reps=QAOA_REPS)
    parameters = list(ansatz.parameters)
    if len(parameters) != 2:
        raise RuntimeError(f"Expected 2 QAOA parameters for p=1, got {len(parameters)}")

    # For p=1 QAOAAnsatz exposes one beta and one gamma parameter.
    bound = ansatz.assign_parameters([beta, gamma])
    bound.measure_all()
    return bound, offset


def run_on_ibm_hardware(beta: float, gamma: float, shots: int) -> None:
    service = QiskitRuntimeService()
    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=len(ASSETS),
    )

    circuit, offset = build_hardware_circuit(beta=beta, gamma=gamma)
    pass_manager = generate_preset_pass_manager(
        optimization_level=TRANSPILER_OPTIMIZATION_LEVEL,
        backend=backend,
        seed_transpiler=SEED,
    )
    isa_circuit = pass_manager.run(circuit)

    sampler = Sampler(mode=backend)

    started = time.perf_counter()
    job = sampler.run([isa_circuit], shots=shots)
    result = job.result()
    wall_time = time.perf_counter() - started

    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())
    best = best_feasible_from_counts(counts)
    top_counts = Counter(counts).most_common(8)

    print("=== IBM Quantum Hardware QAOA ===")
    print(f"Backend: {backend.name}")
    print(f"Job ID: {job.job_id()}")
    print(f"Assets: {ASSETS}")
    print(f"Budget: {BUDGET}")
    print(f"QAOA reps: {QAOA_REPS}")
    print(f"Parameters: beta={beta:.6f}, gamma={gamma:.6f}")
    print(f"Shots: {shots}")
    print(f"Transpiler optimization level: {TRANSPILER_OPTIMIZATION_LEVEL}")
    print(f"ISA circuit depth: {isa_circuit.depth()}")
    print(f"ISA circuit operations: {dict(isa_circuit.count_ops())}")
    print(f"Ising offset: {offset:.6f}")
    print(f"Client wall time: {wall_time:.3f} s")

    if best is None:
        print("Best feasible sample: none observed")
    else:
        bitstring, bits, count, objective = best
        selected = [ASSETS[i] for i, value in enumerate(bits) if value == 1]
        print(f"Best feasible bitstring: {bitstring}")
        print(f"Selected assets: {selected}")
        print(f"Binary solution: {bits.tolist()}")
        print(f"Original objective value: {objective:.6f}")
        print(f"Sample count: {count}/{total}")
        print(f"Sample probability: {count / total:.3%}")

    print("Top measured bitstrings:")
    for bitstring, count in top_counts:
        print(f"  {bitstring}: {count} ({count / total:.3%})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the portfolio QAOA circuit on IBM Quantum hardware.")
    parser.add_argument("--beta", type=float, default=0.5, help="p=1 QAOA beta parameter")
    parser.add_argument("--gamma", type=float, default=0.5, help="p=1 QAOA gamma parameter")
    parser.add_argument("--shots", type=int, default=SHOTS, help="hardware sampling shots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_on_ibm_hardware(beta=args.beta, gamma=args.gamma, shots=args.shots)
