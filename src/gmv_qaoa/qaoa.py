"""Ideal Penalty/X and ordered-ring XY/Dicke simulation and optimisation."""
from __future__ import annotations
import itertools
import math
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector
from .model import all_state_vectors, objective_all_states, exact_reference, build_qubo, qubo_matrix_energies, vector_to_state_index


def model_arrays(instance, alpha=0.5, penalty_factor=1.1, tolerance=1e-10):
    if instance.n > 12:
        raise ValueError("Quantum benchmarks are limited to n <= 12")
    vectors = all_state_vectors(instance.n)
    objective = objective_all_states(instance, vectors, alpha)
    qubo = build_qubo(instance, alpha, penalty_factor)
    reference = exact_reference(instance, alpha, tolerance)
    basis = fixed_weight_basis(instance.n, instance.k)
    return ModelArrays(instance.n, instance.k, vectors, objective,
                       qubo_matrix_energies(qubo, vectors), reference, basis,
                       len(reference['optimal_state_indices']) / math.comb(instance.n, instance.k),
                       float(qubo.penalty))


def state_and_expectation(arrays, method, depth):
    if method == 'penalty_x':
        def state(parameters):
            return qaoa_statevector(arrays.penalty_energies, arrays.n, depth, parameters)
        def expected(probabilities):
            return float(probabilities @ arrays.penalty_energies)
    elif method == 'xy_dicke':
        energies = arrays.objective_b[arrays.basis.state_indices]
        def state(parameters):
            subspace = fixed_weight_qaoa_state(energies, arrays.basis, depth, parameters)
            return expand_to_full_state(subspace, arrays.basis)
        def expected(probabilities):
            return float(probabilities[arrays.basis.state_indices] @ energies)
    else:
        raise ValueError(f'Unknown method: {method}')
    return state, expected


def final_distribution(arrays, method, depth, parameters):
    state, expected = state_and_expectation(arrays, method, depth)
    probabilities = np.square(np.abs(state(np.asarray(parameters, dtype=float))))
    probabilities /= probabilities.sum()
    return probabilities, expected(probabilities)


def outside_domain(values, depth, gamma_upper, beta_upper):
    return bool(np.any(values[:depth] < 0) or np.any(values[:depth] >= gamma_upper)
                or np.any(values[depth:] < 0) or np.any(values[depth:] >= beta_upper))


def optimise(arrays, method, depth, initial, optimizer, cap, wrapped, study):
    state, expected = state_and_expectation(arrays, method, depth)
    gamma_upper = study['initial_angles']['gamma'][1]
    beta_upper = study['initial_angles']['beta'][1]
    settings = study['optimizer']
    calls = 0
    outside = 0

    def cost(parameters):
        nonlocal calls, outside
        values = np.asarray(parameters, dtype=float)
        calls += 1
        outside += int(outside_domain(values, depth, gamma_upper, beta_upper))
        if wrapped:
            values = canonicalise(values, depth, gamma_upper, beta_upper)
        return expected(np.square(np.abs(state(values))))

    if optimizer == 'COBYLA':
        result = minimize(cost, np.asarray(initial, dtype=float), method='COBYLA', tol=settings['tolerance'],
                          options={'maxiter': int(cap), 'rhobeg': settings['rhobeg'], 'disp': False})
    elif optimizer == 'Powell' and wrapped:
        bounds = [(0.0, np.nextafter(gamma_upper, 0.0))] * depth + [(0.0, np.nextafter(beta_upper, 0.0))] * depth
        result = minimize(cost, canonicalise(initial, depth, gamma_upper, beta_upper), method='Powell', bounds=bounds,
                          options={'maxfev': int(cap), 'xtol': settings['powell_xtol'], 'ftol': settings['powell_ftol'], 'disp': False})
    elif optimizer == 'Powell':
        result = minimize(cost, np.asarray(initial, dtype=float), method='Powell',
                          options={'maxfev': int(cap), 'xtol': settings['powell_xtol'], 'ftol': settings['powell_ftol'], 'disp': False})
    else:
        raise ValueError('Unsupported optimiser/protocol combination')
    if calls != result.nfev or calls > cap:
        raise AssertionError('Objective-call accounting failed')
    final = np.asarray(result.x, dtype=float)
    if wrapped:
        final = canonicalise(final, depth, gamma_upper, beta_upper)
    probabilities, final_energy = final_distribution(arrays, method, depth, final)
    return final, probabilities, {
        'optimizer_success': bool(result.success), 'optimizer_status': int(result.status),
        'optimizer_nfev': calls, 'training_expected_energy': float(result.fun),
        'final_expected_energy': final_energy, 'evaluations_outside_original_domain': outside,
        'final_outside_original_domain': outside_domain(final, depth, gamma_upper, beta_upper),
    }

def initial_parameters(
    depth: int,
    seed: int,
    gamma_range: tuple[float, float],
    beta_range: tuple[float, float],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gammas = rng.uniform(gamma_range[0], gamma_range[1], size=depth)
    betas = rng.uniform(beta_range[0], beta_range[1], size=depth)
    return np.concatenate([gammas, betas])


def apply_mixer(state: np.ndarray, beta: float, n: int) -> np.ndarray:
    output = np.asarray(state, dtype=np.complex128).copy()
    cosine = np.cos(beta)
    sine = np.sin(beta)
    for qubit in range(n):
        stride = 1 << qubit
        blocks = output.reshape(-1, 2, stride)
        zero = blocks[:, 0, :].copy()
        one = blocks[:, 1, :].copy()
        blocks[:, 0, :] = cosine * zero - 1j * sine * one
        blocks[:, 1, :] = -1j * sine * zero + cosine * one
    return output


def qaoa_statevector(
    energies: np.ndarray, n: int, depth: int, parameters: np.ndarray
) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    if values.shape != (2 * depth,):
        raise ValueError(
            f"Expected {2 * depth} QAOA parameters, received {values.shape}"
        )
    if len(energies) != 2**n:
        raise ValueError(f"Expected {2**n} energies, received {len(energies)}")
    gammas = values[:depth]
    betas = values[depth:]
    state = Statevector.from_label("+" * n).data.copy()
    for gamma, beta in zip(gammas, betas, strict=True):
        state *= np.exp(-1j * gamma * energies)
        state = apply_mixer(state, beta, n)
    return state


def qiskit_reference_statevector(
    energies: np.ndarray, n: int, depth: int, parameters: np.ndarray
) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    for gamma, beta in zip(values[:depth], values[depth:], strict=True):
        circuit.append(
            DiagonalGate(np.exp(-1j * gamma * np.asarray(energies))),
            list(range(n)),
        )
        circuit.rx(2.0 * beta, range(n))
    return Statevector.from_instruction(circuit).data


def verify_qiskit_equivalence(
    energies: np.ndarray,
    n: int,
    depth: int,
    parameters: np.ndarray,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    kernel = qaoa_statevector(energies, n, depth, parameters)
    qiskit = qiskit_reference_statevector(energies, n, depth, parameters)
    kernel_state = Statevector(kernel)
    reference_state = Statevector(qiskit)
    probability_error = float(
        np.max(np.abs(np.square(np.abs(kernel)) - np.square(np.abs(qiskit))))
    )
    equivalent = bool(kernel_state.equiv(reference_state, atol=tolerance))
    if not equivalent or probability_error > tolerance:
        raise AssertionError(
            "Analytic diagonal/mixer kernel does not match Qiskit "
            f"(equivalent={equivalent}, probability_error={probability_error})"
        )
    return {
        "statevectors_equivalent_up_to_global_phase": equivalent,
        "maximum_probability_error": probability_error,
        "tolerance": tolerance,
        "result": "PASS",
    }

@dataclass(frozen=True)
class FixedWeightBasis:
    n: int
    k: int
    selected_tuples: tuple[tuple[int, ...], ...]
    vectors: np.ndarray
    state_indices: np.ndarray
    index_by_state: dict[int, int]


def fixed_weight_basis(n: int, k: int) -> FixedWeightBasis:
    if not 0 <= k <= n:
        raise ValueError("Require 0 <= k <= n")
    selected = tuple(itertools.combinations(range(n), k))
    vectors = np.zeros((len(selected), n), dtype=np.int8)
    for row, positions in enumerate(selected):
        vectors[row, list(positions)] = 1
    state_indices = np.asarray(
        [vector_to_state_index(vector) for vector in vectors], dtype=np.int64
    )
    return FixedWeightBasis(
        n=n,
        k=k,
        selected_tuples=selected,
        vectors=vectors,
        state_indices=state_indices,
        index_by_state={
            int(state_index): row
            for row, state_index in enumerate(state_indices)
        },
    )


def dicke_state(basis: FixedWeightBasis) -> np.ndarray:
    return np.full(
        len(basis.state_indices),
        1.0 / math.sqrt(len(basis.state_indices)),
        dtype=np.complex128,
    )


def ring_edges(n: int) -> tuple[tuple[int, int], ...]:
    if n < 3:
        raise ValueError("Ring mixer requires at least three qubits")
    return tuple((index, index + 1) for index in range(n - 1)) + ((n - 1, 0),)


def apply_xy_edge(
    state: np.ndarray,
    basis: FixedWeightBasis,
    left: int,
    right: int,
    beta: float,
) -> np.ndarray:
    """Apply exp[-i beta (XX+YY)/2] to one ordered ring edge."""
    output = np.asarray(state, dtype=np.complex128).copy()
    cosine = np.cos(beta)
    sine = np.sin(beta)
    for row, vector in enumerate(basis.vectors):
        if vector[left] != 1 or vector[right] != 0:
            continue
        partner_state = int(basis.state_indices[row] ^ (1 << left) ^ (1 << right))
        partner = basis.index_by_state[partner_state]
        left_amplitude = output[row]
        right_amplitude = output[partner]
        output[row] = cosine * left_amplitude - 1j * sine * right_amplitude
        output[partner] = cosine * right_amplitude - 1j * sine * left_amplitude
    return output


def apply_ring_xy(
    state: np.ndarray, basis: FixedWeightBasis, beta: float
) -> np.ndarray:
    output = np.asarray(state, dtype=np.complex128).copy()
    for left, right in ring_edges(basis.n):
        output = apply_xy_edge(output, basis, left, right, beta)
    return output


def fixed_weight_qaoa_state(
    energies: np.ndarray,
    basis: FixedWeightBasis,
    depth: int,
    parameters: np.ndarray,
    *,
    return_layer_leakage: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[float]]:
    values = np.asarray(parameters, dtype=float)
    if values.shape != (2 * depth,):
        raise ValueError(f"Expected {2 * depth} parameters")
    feasible_energies = np.asarray(energies, dtype=float)
    if feasible_energies.shape != (len(basis.state_indices),):
        raise ValueError("Energy table does not match fixed-weight basis")
    state = dicke_state(basis)
    leakage = []
    for gamma, beta in zip(values[:depth], values[depth:], strict=True):
        state *= np.exp(-1j * gamma * feasible_energies)
        state = apply_ring_xy(state, basis, beta)
        leakage.append(abs(1.0 - float(np.vdot(state, state).real)))
    if return_layer_leakage:
        return state, leakage
    return state


def expand_to_full_state(
    state: np.ndarray, basis: FixedWeightBasis
) -> np.ndarray:
    full = np.zeros(2**basis.n, dtype=np.complex128)
    full[basis.state_indices] = np.asarray(state, dtype=np.complex128)
    return full


def qiskit_reference_state(
    full_energies: np.ndarray,
    basis: FixedWeightBasis,
    depth: int,
    parameters: np.ndarray,
) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    initial = expand_to_full_state(dicke_state(basis), basis)
    circuit = QuantumCircuit(basis.n)
    circuit.initialize(initial, range(basis.n))
    for gamma, beta in zip(values[:depth], values[depth:], strict=True):
        circuit.append(
            DiagonalGate(np.exp(-1j * gamma * np.asarray(full_energies))),
            range(basis.n),
        )
        for left, right in ring_edges(basis.n):
            circuit.rxx(beta, left, right)
            circuit.ryy(beta, left, right)
    return Statevector.from_instruction(circuit).data


def reference_comparison(
    full_energies: np.ndarray,
    basis: FixedWeightBasis,
    depth: int,
    parameters: np.ndarray,
    tolerance: float = 1e-10,
) -> dict[str, float | bool | str]:
    custom_subspace = fixed_weight_qaoa_state(
        np.asarray(full_energies)[basis.state_indices],
        basis,
        depth,
        parameters,
    )
    custom = expand_to_full_state(custom_subspace, basis)
    reference = qiskit_reference_state(
        full_energies, basis, depth, parameters
    )
    probability_error = float(
        np.max(
            np.abs(np.square(np.abs(custom)) - np.square(np.abs(reference))),
            initial=0.0,
        )
    )
    equivalent = bool(
        Statevector(custom).equiv(Statevector(reference), atol=tolerance)
    )
    if not equivalent or probability_error > tolerance:
        raise AssertionError(
            "Custom XY/Dicke path differs from Qiskit reference: "
            f"equivalent={equivalent}, error={probability_error}"
        )
    return {
        "result": "PASS",
        "statevectors_equivalent_up_to_global_phase": equivalent,
        "maximum_probability_error": probability_error,
        "tolerance": tolerance,
    }

@dataclass
class ModelArrays:
    n: int
    k: int
    vectors: np.ndarray
    objective_b: np.ndarray
    penalty_energies: np.ndarray
    reference: dict[str, Any]
    basis: FixedWeightBasis
    uniform_optimum_probability: float
    penalty: float

def canonicalise(
    parameters: np.ndarray, depth: int, gamma_period: float, beta_period: float
) -> np.ndarray:
    values = np.asarray(parameters, dtype=float).copy()
    if values.shape != (2 * depth,):
        raise ValueError(f"Expected {2 * depth} parameters, found {values.shape}")
    values[:depth] = np.mod(values[:depth], gamma_period)
    values[depth:] = np.mod(values[depth:], beta_period)
    return values


def interpolate_p1_to_p2(parameters: np.ndarray) -> np.ndarray:
    """Zhou et al. Appendix B interpolation specialised from p=1 to p=2."""
    values = np.asarray(parameters, dtype=float)
    if values.shape != (2,):
        raise ValueError("p=1 interpolation requires [gamma_1, beta_1]")
    gamma, beta = values
    return np.array([gamma, gamma, beta, beta], dtype=float)
