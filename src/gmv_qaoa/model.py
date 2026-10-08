"""Candidate-set objectives, QUBO/Ising encoding, and bit conventions."""
from __future__ import annotations
import itertools
import math
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd


def published_float_array(values: np.ndarray) -> np.ndarray:
    """Define model values using pandas CSV parsing in memory.

    Pandas' default decimal parser differs by an ULP from round-trip parsing. That difference can change a local optimiser's path, so the numerical protocol serialises and parses derived arrays consistently.
    """
    array = np.asarray(values, dtype=float)
    serialized = pd.DataFrame(array).to_csv(index=False)
    return pd.read_csv(StringIO(serialized)).to_numpy(float).reshape(array.shape)


def load_instances(path: str | Path, study: dict[str, Any]) -> dict[int, ModelInstance]:
    table = pd.read_csv(path, float_precision="round_trip")
    required = {"rank", "locality_name", "avg_qff_per_trap", "centroid_x_m", "centroid_y_m", "geography_order", "distance_from_shepparton_km"}
    if set(table.columns) != required or len(table) != 39:
        raise ValueError("Expected the documented 39-row locality input schema")
    if not table.locality_name.is_unique or table.locality_name.isna().any():
        raise ValueError("Locality names must be present and unique")
    if sorted(table.geography_order) != list(range(1, 40)) or sorted(table['rank']) != list(range(1, 40)):
        raise ValueError("Source ranks and geographic ranks must each cover 1–39")
    numeric = table.drop(columns="locality_name").to_numpy(float)
    if not np.isfinite(numeric).all() or (table.avg_qff_per_trap < 0).any():
        raise ValueError("Inputs must be finite and activity must be nonnegative")
    anchor = table.loc[table.locality_name == study['instances']['anchor']]
    if len(anchor) != 1:
        raise ValueError("The geographic anchor must occur exactly once")
    xy = table[['centroid_x_m', 'centroid_y_m']].to_numpy()
    anchor_xy = anchor[['centroid_x_m', 'centroid_y_m']].to_numpy()[0]
    distance = np.sqrt(np.square(xy - anchor_xy).sum(axis=1)) / 1000.0
    if not np.allclose(distance, table.distance_from_shepparton_km, rtol=0, atol=1e-12):
        raise ValueError("Stored geographic distances do not match the centroids")
    geographic = table.assign(_distance=distance).sort_values(['_distance', 'locality_name'], kind='mergesort')
    if geographic.geography_order.tolist() != list(range(1, 40)):
        raise ValueError("Membership ordering must depend only on geometry")
    instances = {}
    for n in study['instances']['sizes']:
        subset = table.sort_values('locality_name') if n == 39 else geographic.iloc[:n]
        coords = subset[['centroid_x_m', 'centroid_y_m']].to_numpy()
        offsets = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt(np.square(offsets).sum(axis=2)) / 1000.0
        activity = subset.avg_qff_per_trap.to_numpy(float)
        risk = np.log1p(activity)
        risk /= float(risk.max())
        # The full-set analysis loaded distances before deriving its scale.
        if n == 39:
            distances = published_float_array(distances)
        tau = float(np.median(distances[np.triu_indices(n, 1)]))
        proximity = np.exp(-distances / tau)
        np.fill_diagonal(proximity, 0.0)
        if n != 39:
            risk = published_float_array(risk)
            proximity = published_float_array(proximity)
            distances = published_float_array(distances)
        instances[n] = ModelInstance(n, int(study['instances']['budgets'][str(n)]), tuple(subset.locality_name), risk, activity, distances, proximity, tau)
    return instances

@dataclass(frozen=True)
class ModelInstance:
    n: int
    k: int
    localities: tuple[str, ...]
    risk: np.ndarray
    activity: np.ndarray
    distance_km: np.ndarray
    proximity: np.ndarray
    tau_km: float


def vector_to_canonical_bitstring(x: Iterable[int]) -> str:
    return "".join(str(int(value)) for value in x)


def canonical_bitstring_to_vector(bitstring: str) -> np.ndarray:
    if not bitstring or set(bitstring) - {"0", "1"}:
        raise ValueError(f"Invalid canonical bit string: {bitstring!r}")
    return np.fromiter((int(value) for value in bitstring), dtype=np.int8)


def canonical_to_qiskit_bitstring(bitstring: str) -> str:
    """Qiskit displays q_(n-1)..q_0; canonical order is x_0..x_(n-1)."""
    canonical_bitstring_to_vector(bitstring)
    return bitstring[::-1]


def qiskit_to_canonical_bitstring(bitstring: str) -> str:
    canonical_bitstring_to_vector(bitstring)
    return bitstring[::-1]


def state_index_to_vector(index: int, n: int) -> np.ndarray:
    if index < 0 or index >= 2**n:
        raise ValueError(f"State index {index} is outside n={n}")
    return ((index >> np.arange(n)) & 1).astype(np.int8)


def vector_to_state_index(x: Iterable[int]) -> int:
    vector = np.asarray(tuple(x), dtype=np.int8)
    if not np.isin(vector, [0, 1]).all():
        raise ValueError("Binary vector contains a non-binary entry")
    return int(np.dot(vector, 1 << np.arange(len(vector))))


def all_state_vectors(n: int) -> np.ndarray:
    states = np.arange(2**n, dtype=np.uint32)
    return ((states[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(
        np.int8
    )


def feasible_objective(
    instance: ModelInstance, x: np.ndarray, alpha: float
) -> float:
    vector = np.asarray(x, dtype=float)
    if vector.shape != (instance.n,):
        raise ValueError(f"Expected vector shape {(instance.n,)}, got {vector.shape}")
    risk_term = -(alpha / instance.k) * float(instance.risk @ vector)
    pair_sum = 0.0
    for i in range(instance.n):
        for j in range(i + 1, instance.n):
            pair_sum += instance.proximity[i, j] * vector[i] * vector[j]
    pair_term = ((1.0 - alpha) / math.comb(instance.k, 2)) * pair_sum
    return risk_term + pair_term


def objective_all_states(
    instance: ModelInstance, vectors: np.ndarray, alpha: float
) -> np.ndarray:
    vectors_float = np.asarray(vectors, dtype=float)
    risk_term = -(alpha / instance.k) * (vectors_float @ instance.risk)
    pair_sum = np.zeros(len(vectors_float), dtype=float)
    for i in range(instance.n):
        for j in range(i + 1, instance.n):
            pair_sum += (
                instance.proximity[i, j]
                * vectors_float[:, i]
                * vectors_float[:, j]
            )
    return risk_term + ((1.0 - alpha) / math.comb(instance.k, 2)) * pair_sum


def selected_indices(x: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(x, dtype=np.int8))


def solution_metrics(
    instance: ModelInstance,
    x: np.ndarray,
    alpha: float,
    top_detection_total: float | None = None,
) -> dict[str, Any]:
    vector = np.asarray(x, dtype=np.int8)
    indices = selected_indices(vector)
    if top_detection_total is None:
        top_detection_total = float(
            np.sort(instance.risk)[-instance.k :].sum()
        )
    detection = float(instance.risk[indices].sum())
    if len(indices) >= 2:
        pair_distances = np.array(
            [
                instance.distance_km[i, j]
                for i, j in itertools.combinations(indices, 2)
            ],
            dtype=float,
        )
        mean_distance = float(pair_distances.mean())
        minimum_distance = float(pair_distances.min())
    else:
        mean_distance = 0.0
        minimum_distance = 0.0
    canonical = vector_to_canonical_bitstring(vector)
    return {
        "canonical_bitstring": canonical,
        "qiskit_bitstring": canonical_to_qiskit_bitstring(canonical),
        "cardinality": int(vector.sum()),
        "selected_localities": [instance.localities[index] for index in indices],
        "objective": feasible_objective(instance, vector, alpha),
        "detection_score": detection,
        "detection_retention": (
            detection / top_detection_total if top_detection_total > 0 else 1.0
        ),
        "mean_pairwise_distance_km": mean_distance,
        "minimum_pairwise_distance_km": minimum_distance,
    }


def exact_reference(
    instance: ModelInstance, alpha: float, tolerance: float
) -> dict[str, Any]:
    vectors = all_state_vectors(instance.n)
    objectives = objective_all_states(instance, vectors, alpha)
    cardinalities = vectors.sum(axis=1)
    feasible_mask = cardinalities == instance.k
    feasible_values = objectives[feasible_mask]
    best = float(feasible_values.min())
    worst = float(feasible_values.max())
    optimum_mask = feasible_mask & np.isclose(
        objectives, best, rtol=0.0, atol=tolerance
    )
    optimum_vectors = vectors[optimum_mask]
    optimum_strings = sorted(
        vector_to_canonical_bitstring(vector) for vector in optimum_vectors
    )
    return {
        "vectors": vectors,
        "objectives": objectives,
        "cardinalities": cardinalities,
        "feasible_mask": feasible_mask,
        "best_objective": best,
        "worst_objective": worst,
        "optimal_canonical_bitstrings": optimum_strings,
        "optimal_state_indices": sorted(
            vector_to_state_index(vector) for vector in optimum_vectors
        ),
        "total_state_count": int(len(vectors)),
        "feasible_state_count": int(feasible_mask.sum()),
    }

@dataclass(frozen=True)
class QuboModel:
    n: int
    k: int
    alpha: float
    a_min: float
    penalty_factor: float
    penalty: float
    constant: float
    upper_matrix: np.ndarray


@dataclass(frozen=True)
class IsingModel:
    constant: float
    h: np.ndarray
    coupling: np.ndarray


def calculate_a_min(
    instance: ModelInstance, alpha: float, tolerance: float = 1e-12
) -> float:
    vectors = all_state_vectors(instance.n)
    objective = objective_all_states(instance, vectors, alpha)
    cardinality = vectors.sum(axis=1)
    feasible = cardinality == instance.k
    best = float(objective[feasible].min())
    infeasible = ~feasible
    violation = np.square(cardinality[infeasible] - instance.k)
    improvement = np.maximum(0.0, best - objective[infeasible])
    ratios = improvement / violation
    a_min = float(ratios.max(initial=0.0))
    if a_min < tolerance:
        return 0.0
    return a_min


def build_qubo(
    instance: ModelInstance, alpha: float, penalty_factor: float
) -> QuboModel:
    a_min = calculate_a_min(instance, alpha)
    penalty = penalty_factor * a_min
    upper = np.zeros((instance.n, instance.n), dtype=float)
    upper[np.diag_indices(instance.n)] = (
        -(alpha / instance.k) * instance.risk
        + penalty * (1 - 2 * instance.k)
    )
    pair_scale = (1.0 - alpha) / math.comb(instance.k, 2)
    for i in range(instance.n):
        for j in range(i + 1, instance.n):
            upper[i, j] = (
                pair_scale * instance.proximity[i, j] + 2.0 * penalty
            )
    return QuboModel(
        n=instance.n,
        k=instance.k,
        alpha=alpha,
        a_min=a_min,
        penalty_factor=penalty_factor,
        penalty=penalty,
        constant=penalty * instance.k**2,
        upper_matrix=upper,
    )


def qubo_direct_energies(
    instance: ModelInstance,
    vectors: np.ndarray,
    alpha: float,
    penalty: float,
) -> np.ndarray:
    objective = objective_all_states(instance, vectors, alpha)
    violation = np.square(vectors.sum(axis=1) - instance.k)
    return objective + penalty * violation


def qubo_matrix_energies(model: QuboModel, vectors: np.ndarray) -> np.ndarray:
    vectors_float = np.asarray(vectors, dtype=float)
    return (
        np.einsum(
            "bi,ij,bj->b",
            vectors_float,
            model.upper_matrix,
            vectors_float,
        )
        + model.constant
    )


def qubo_to_ising(model: QuboModel) -> IsingModel:
    linear = np.diag(model.upper_matrix).copy()
    quadratic = np.triu(model.upper_matrix, k=1)
    h = -0.5 * linear
    coupling = np.zeros_like(model.upper_matrix)
    constant = model.constant + 0.5 * float(linear.sum())
    for i in range(model.n):
        for j in range(i + 1, model.n):
            coefficient = quadratic[i, j]
            coupling[i, j] = coefficient / 4.0
            h[i] -= coefficient / 4.0
            h[j] -= coefficient / 4.0
            constant += coefficient / 4.0
    return IsingModel(constant=constant, h=h, coupling=coupling)


def ising_energies(model: IsingModel, vectors: np.ndarray) -> np.ndarray:
    z = 1.0 - 2.0 * np.asarray(vectors, dtype=float)
    energies = model.constant + z @ model.h
    for i in range(z.shape[1]):
        for j in range(i + 1, z.shape[1]):
            energies += model.coupling[i, j] * z[:, i] * z[:, j]
    return energies
