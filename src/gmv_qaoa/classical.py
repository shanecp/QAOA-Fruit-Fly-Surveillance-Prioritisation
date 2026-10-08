"""Exact feasible enumeration, classical heuristics, and allocation breakpoints."""
from __future__ import annotations
import itertools
import math
from typing import Any
import numpy as np
from .model import ModelInstance, feasible_objective, solution_metrics, exact_reference


def classical_results(instances, study):
    exact_rows, baseline_rows = [], []
    full_values = None
    for n, instance in instances.items():
        combinations = combination_array(n, instance.k)
        values = combination_coefficients(combinations, instance.risk, instance.distance_km, instance.tau_km, instance.k)
        alphas = study['objective']['full_set_alphas'] if n == 39 else [study['objective']['alpha']]
        for alpha in alphas:
            objectives = alpha * values['detection_coefficient'] + (1 - alpha) * values['proximity_coefficient']
            best = float(objectives.min())
            ties = np.flatnonzero(np.isclose(objectives, best, atol=study['tolerances']['objective'], rtol=0))
            for index in ties:
                vector = np.zeros(n, dtype=np.int8)
                vector[combinations[index]] = 1
                metrics = solution_metrics(instance, vector, alpha)
                exact_rows.append({'n': n, 'k': instance.k, 'alpha': alpha, 'feasible_count': len(combinations),
                                   'optimum_count': len(ties), **metrics,
                                   'selected_localities': '|'.join(metrics['selected_localities'])})
            if alpha == study['objective']['alpha']:
                for method, vector in [
                    ('top_k', top_detection_baseline(instance)),
                    ('max_min_dispersion', dispersion_baseline(instance)),
                    ('greedy_composite', greedy_composite_baseline(instance, alpha)),
                ]:
                    metrics = solution_metrics(instance, vector, alpha)
                    baseline_rows.append({'n': n, 'k': instance.k, 'alpha': alpha, 'method': method,
                                          'exact_objective': best, 'objective_gap': metrics['objective'] - best,
                                          **metrics, 'selected_localities': '|'.join(metrics['selected_localities'])})
        if n == 39:
            full_values = (instance, combinations, values)
    instance, combinations, values = full_values
    intercepts = values['proximity_coefficient']
    slopes = values['detection_coefficient'] - intercepts
    tolerance = study['tolerances']['objective']
    intervals, _ = envelope_intervals(lower_envelope(intercepts, slopes, tolerance), intercepts, slopes,
                                      combinations, values, list(instance.localities), instance.k, 1.0,
                                      float(np.sort(instance.risk)[-instance.k:].sum()), tolerance)
    return exact_rows, baseline_rows, intervals

def top_detection_baseline(instance: ModelInstance) -> np.ndarray:
    order = sorted(
        range(instance.n),
        key=lambda index: (-instance.risk[index], instance.localities[index]),
    )
    vector = np.zeros(instance.n, dtype=np.int8)
    vector[order[: instance.k]] = 1
    return vector


def dispersion_baseline(instance: ModelInstance, tolerance: float = 1e-12) -> np.ndarray:
    pairs = list(itertools.combinations(range(instance.n), 2))
    farthest = max(instance.distance_km[i, j] for i, j in pairs)
    tied_pairs = [
        (i, j)
        for i, j in pairs
        if math.isclose(
            instance.distance_km[i, j], farthest, abs_tol=tolerance, rel_tol=0
        )
    ]
    selected = list(
        min(
            tied_pairs,
            key=lambda pair: tuple(
                sorted((instance.localities[pair[0]], instance.localities[pair[1]]))
            ),
        )
    )
    while len(selected) < instance.k:
        candidate_scores = []
        for candidate in range(instance.n):
            if candidate in selected:
                continue
            minimum_distance = min(
                instance.distance_km[candidate, current] for current in selected
            )
            candidate_scores.append((candidate, minimum_distance))
        best_score = max(score for _, score in candidate_scores)
        tied = [
            candidate
            for candidate, score in candidate_scores
            if math.isclose(score, best_score, abs_tol=tolerance, rel_tol=0)
        ]
        selected.append(min(tied, key=lambda index: instance.localities[index]))
    vector = np.zeros(instance.n, dtype=np.int8)
    vector[selected] = 1
    return vector


def greedy_composite_baseline(
    instance: ModelInstance, alpha: float, tolerance: float = 1e-12
) -> np.ndarray:
    selected: list[int] = []
    current = np.zeros(instance.n, dtype=np.int8)
    current_value = feasible_objective(instance, current, alpha)
    while len(selected) < instance.k:
        marginal: list[tuple[int, float]] = []
        for candidate in range(instance.n):
            if candidate in selected:
                continue
            trial = current.copy()
            trial[candidate] = 1
            marginal.append(
                (
                    candidate,
                    feasible_objective(instance, trial, alpha) - current_value,
                )
            )
        best_value = min(value for _, value in marginal)
        tied = [
            candidate
            for candidate, value in marginal
            if math.isclose(value, best_value, abs_tol=tolerance, rel_tol=0)
        ]
        chosen = min(tied, key=lambda index: instance.localities[index])
        selected.append(chosen)
        current[chosen] = 1
        current_value = feasible_objective(instance, current, alpha)
    return current

def combination_array(n: int, k: int, limit: int | None = None) -> np.ndarray:
    iterator = itertools.combinations(range(n), k)
    if limit is not None:
        iterator = itertools.islice(iterator, limit)
    return np.asarray(list(iterator), dtype=np.int16)


def combination_coefficients(
    combinations: np.ndarray,
    risk: np.ndarray,
    distance: np.ndarray,
    tau: float,
    k: int,
) -> dict[str, np.ndarray]:
    proximity = np.exp(-distance / tau)
    np.fill_diagonal(proximity, 0.0)
    detection_sum = risk[combinations].sum(axis=1)
    proximity_sum = np.zeros(len(combinations))
    distance_sum = np.zeros(len(combinations))
    minimum_distance = np.full(len(combinations), np.inf)
    for left, right in itertools.combinations(range(k), 2):
        pair_proximity = proximity[
            combinations[:, left], combinations[:, right]
        ]
        pair_distance = distance[
            combinations[:, left], combinations[:, right]
        ]
        proximity_sum += pair_proximity
        distance_sum += pair_distance
        minimum_distance = np.minimum(minimum_distance, pair_distance)
    pair_count = math.comb(k, 2)
    return {
        "detection_sum": detection_sum,
        "detection_coefficient": -detection_sum / k,
        "proximity_coefficient": proximity_sum / pair_count,
        "mean_pairwise_distance_km": distance_sum / pair_count,
        "minimum_pairwise_distance_km": minimum_distance,
    }


def locality_set(combination: np.ndarray, names: list[str]) -> str:
    return "|".join(names[int(index)] for index in combination)


def direct_ties(
    intercepts: np.ndarray,
    slopes: np.ndarray,
    alpha: float,
    tolerance: float,
) -> tuple[float, np.ndarray]:
    objectives = intercepts + alpha * slopes
    best = float(objectives.min())
    ties = np.flatnonzero(
        np.isclose(objectives, best, atol=tolerance, rtol=0)
    )
    return best, ties


def lower_envelope(
    intercepts: np.ndarray,
    slopes: np.ndarray,
    tolerance: float = 1e-10,
) -> list[dict[str, float | int]]:
    """Slope-sorted lower hull; query alpha increases from zero to one."""
    if len(intercepts) != len(slopes) or not len(intercepts):
        raise ValueError("Line arrays must be non-empty and equally sized")
    order = np.lexsort((intercepts, -slopes))
    representatives: list[int] = []
    last_slope_key: float | None = None
    for index in order:
        slope_key = round(float(slopes[index]), 12)
        if slope_key != last_slope_key:
            representatives.append(int(index))
            last_slope_key = slope_key
        # Later same-slope lines have equal or larger intercept and are dominated.
    hull: list[dict[str, float | int]] = []
    for index in representatives:
        slope = float(slopes[index])
        intercept = float(intercepts[index])
        start = -math.inf
        while hull:
            previous = hull[-1]
            denominator = float(previous["slope"]) - slope
            if denominator <= 0:
                raise AssertionError("Envelope slopes are not strictly descending")
            start = (intercept - float(previous["intercept"])) / denominator
            if start > float(previous["start"]) + tolerance:
                break
            hull.pop()
        if not hull:
            start = -math.inf
        hull.append(
            {
                "line_index": index,
                "slope": slope,
                "intercept": intercept,
                "start": start,
            }
        )
    return hull


def envelope_intervals(
    hull: list[dict[str, float | int]],
    intercepts: np.ndarray,
    slopes: np.ndarray,
    combinations: np.ndarray,
    values: dict[str, np.ndarray],
    names: list[str],
    k: int,
    factor: float,
    top_detection_total: float,
    tolerance: float,
) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    zero_width = 0
    relevant = []
    for position, line in enumerate(hull):
        raw_start = float(line["start"])
        raw_end = (
            float(hull[position + 1]["start"])
            if position + 1 < len(hull)
            else math.inf
        )
        start = max(0.0, raw_start)
        end = min(1.0, raw_end)
        if end < start - tolerance:
            continue
        if end - start <= tolerance:
            zero_width += 1
            continue
        relevant.append((line, start, end))
    previous_set: set[str] | None = None
    for interval_index, (line, start, end) in enumerate(relevant):
        midpoint = (start + end) / 2.0
        best, ties = direct_ties(
            intercepts, slopes, midpoint, tolerance
        )
        predicted = float(line["intercept"]) + midpoint * float(line["slope"])
        if not math.isclose(best, predicted, abs_tol=tolerance, rel_tol=0):
            raise AssertionError("Hull prediction disagrees at interval midpoint")
        sets = sorted(locality_set(combinations[index], names) for index in ties)
        start_best, start_ties = direct_ties(
            intercepts, slopes, start, tolerance
        )
        end_best, end_ties = direct_ties(intercepts, slopes, end, tolerance)
        representative = int(ties[0])
        current_set = set(sets[0].split("|"))
        changed = (
            ""
            if previous_set is None
            else "|".join(sorted(previous_set.symmetric_difference(current_set)))
        )
        output.append(
            {
                "analysis_id": f"k{k:02d}-tf{round(factor * 100):03d}",
                "k": k,
                "tau_factor": factor,
                "interval_index": interval_index,
                "alpha_start": start,
                "alpha_end": end,
                "alpha_midpoint": midpoint,
                "optimal_sets": ";".join(sets),
                "interior_tie_count": len(sets),
                "start_tied_sets": ";".join(
                    sorted(
                        locality_set(combinations[index], names)
                        for index in start_ties
                    )
                ),
                "end_tied_sets": ";".join(
                    sorted(
                        locality_set(combinations[index], names)
                        for index in end_ties
                    )
                ),
                "start_objective": start_best,
                "midpoint_objective": best,
                "end_objective": end_best,
                "detection_sum": float(values["detection_sum"][representative]),
                "detection_retention": float(
                    values["detection_sum"][representative]
                    / top_detection_total
                ),
                "proximity_coefficient": float(
                    values["proximity_coefficient"][representative]
                ),
                "mean_pairwise_distance_km": float(
                    values["mean_pairwise_distance_km"][representative]
                ),
                "minimum_pairwise_distance_km": float(
                    values["minimum_pairwise_distance_km"][representative]
                ),
                "changed_localities_from_previous": changed,
            }
        )
        previous_set = current_set
    return output, zero_width
