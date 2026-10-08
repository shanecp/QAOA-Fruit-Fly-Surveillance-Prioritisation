"""Distribution-level metrics and reproducible paired statistics."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .qaoa import ModelArrays


SUMMARY_METRICS = [
    'exact_optimum_probability', 'feasible_probability_mass',
    'conditional_exact_optimum_probability', 'overall_optimum_ratio_to_uniform',
    'conditional_optimum_ratio_to_uniform', 'near_1pct_probability',
    'near_5pct_probability', 'expected_best_regret_b4096',
]
PAIRED_METRICS = [
    'exact_optimum_probability', 'overall_optimum_ratio_to_uniform',
    'near_1pct_probability', 'near_5pct_probability',
    'feasible_probability_mass', 'expected_best_regret_b4096',
]
GROUP_KEYS = ['experiment', 'n', 'k', 'depth', 'method', 'optimizer', 'evaluation_cap', 'penalty_factor', 'angle_handling']


def summarize_runs(rows):
    frame = pd.DataFrame(rows)
    output = []
    for key, group in frame.groupby(GROUP_KEYS, sort=True):
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy(float)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            output.append(dict(zip(GROUP_KEYS, key, strict=True)) | {
                'metric': metric, 'run_count': len(values), 'q1': q1, 'median': median, 'q3': q3,
                'optimizer_success_count': int(group.optimizer_success.sum()),
                'evaluation_cap_count': int((group.optimizer_nfev == group.evaluation_cap).sum()),
            })
    return output


def bootstrap_interval(values, rng, resamples):
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    return np.quantile(np.median(values[indices], axis=1), [0.025, 0.975])


def paired_comparisons(rows, study):
    frame = pd.DataFrame(rows)
    output = []
    for experiment in ['wrapped_comparison', 'direct_comparison']:
        rng = np.random.default_rng(study['experiments'][experiment]['bootstrap_seed'])
        subset = frame[frame.experiment == experiment]
        if experiment == 'direct_comparison':
            subset = subset[subset.depth == 2]
        for (n, depth), group in subset.groupby(['n', 'depth'], sort=True):
            penalty = group[group.method == 'penalty_x'].sort_values('seed').set_index('seed')
            xy = group[group.method == 'xy_dicke'].sort_values('seed').set_index('seed')
            if not penalty.index.equals(xy.index):
                raise ValueError('Mismatched paired seeds')
            for metric in PAIRED_METRICS:
                differences = xy[metric].to_numpy(float) - penalty[metric].to_numpy(float)
                low, high = bootstrap_interval(differences, rng, study['bootstrap_resamples'])
                output.append({'experiment': experiment, 'n': int(n), 'depth': int(depth),
                               'comparison': 'xy_dicke minus penalty_x', 'metric': metric,
                               'optimizer': 'COBYLA', 'evaluation_cap': int(group.evaluation_cap.iloc[0]),
                               'pair_count': len(differences), 'median_difference': float(np.median(differences)),
                               'bootstrap_95_low': float(low), 'bootstrap_95_high': float(high),
                               'positive_pair_count': int((differences > 0).sum()),
                               'relative_gap_to_best': '', 'practical_tie': ''})
            if experiment == 'wrapped_comparison':
                # Skip one resampling block per group as part of the fixed RNG protocol.
                rng.integers(0, len(penalty), size=(study['bootstrap_resamples'], len(penalty)))
                if depth == 1:
                    # Advance over the configured reserved blocks before depth two.
                    # These offsets define the random stream used for the published intervals.
                    omitted = study['experiments'][experiment]['bootstrap_omitted_depth_one_groups']
                    for _ in range(omitted * (len(PAIRED_METRICS) + 1)):
                        rng.integers(0, len(penalty), size=(study['bootstrap_resamples'], len(penalty)))
    audit = frame[frame.experiment == 'optimizer_audit']
    candidates = list(audit.groupby(['optimizer', 'evaluation_cap'], sort=True))
    best_key, best = max(candidates, key=lambda item: float(item[1].exact_optimum_probability.median()))
    best = best.sort_values(['n', 'seed'])
    best_median = float(best.exact_optimum_probability.median())
    settings = study['experiments']['optimizer_audit']
    rng = np.random.default_rng(settings['bootstrap_seed'])
    for (optimizer, cap), group in candidates:
        group = group.sort_values(['n', 'seed'])
        differences = group.exact_optimum_probability.to_numpy(float) - best.exact_optimum_probability.to_numpy(float)
        low, high = bootstrap_interval(differences, rng, study['bootstrap_resamples'])
        gap = (best_median - float(group.exact_optimum_probability.median())) / best_median
        output.append({'experiment': 'optimizer_audit', 'n': '10,12', 'depth': 1,
                       'comparison': f'{optimizer}/{cap} minus {best_key[0]}/{best_key[1]}',
                       'metric': 'exact_optimum_probability', 'optimizer': optimizer, 'evaluation_cap': int(cap),
                       'pair_count': len(differences), 'median_difference': float(np.median(differences)),
                       'bootstrap_95_low': float(low), 'bootstrap_95_high': float(high),
                       'positive_pair_count': int((differences > 0).sum()), 'relative_gap_to_best': gap,
                       'practical_tie': bool(gap <= settings['practical_tie_relative_tolerance'] and low <= 0 <= high)})
    return output

def expected_best(
    regrets: np.ndarray, probabilities: np.ndarray, budget: int
) -> float:
    order = np.argsort(regrets, kind="mergesort")
    values, starts = np.unique(regrets[order], return_index=True)
    masses = np.add.reduceat(probabilities[order], starts)
    cumulative = np.cumsum(masses)
    inclusive = 1.0 - np.concatenate(([0.0], cumulative[:-1]))
    exclusive = np.maximum(0.0, 1.0 - cumulative)
    return float(values @ (inclusive**budget - exclusive**budget))


def distribution_metrics(
    arrays: ModelArrays,
    probabilities: np.ndarray,
    shot_budgets: list[int],
) -> dict[str, Any]:
    feasible = arrays.vectors.sum(axis=1) == arrays.k
    optimum = np.asarray(arrays.reference["optimal_state_indices"], dtype=int)
    denominator = (
        arrays.reference["worst_objective"]
        - arrays.reference["best_objective"]
    )
    feasible_regrets = (
        np.zeros(int(feasible.sum()))
        if denominator <= 0
        else np.clip(
            (
                arrays.objective_b[feasible]
                - arrays.reference["best_objective"]
            )
            / denominator,
            0.0,
            1.0,
        )
    )
    regrets = np.ones(len(probabilities))
    regrets[feasible] = feasible_regrets
    feasible_mass = float(probabilities[feasible].sum())
    optimum_probability = float(probabilities[optimum].sum())
    feasible_b = (
        float(
            probabilities[feasible]
            @ arrays.objective_b[feasible]
            / feasible_mass
        )
        if feasible_mass
        else None
    )
    metrics: dict[str, Any] = {
        "probability_sum": float(probabilities.sum()),
        "feasible_probability_mass": feasible_mass,
        "infeasible_probability_mass": float(max(0.0, 1.0 - feasible_mass)),
        "exact_optimum_probability": optimum_probability,
        "conditional_exact_optimum_probability": (
            optimum_probability / feasible_mass if feasible_mass else 0.0
        ),
        "uniform_optimum_probability": arrays.uniform_optimum_probability,
        "overall_optimum_ratio_to_uniform": (
            optimum_probability / arrays.uniform_optimum_probability
        ),
        "conditional_optimum_ratio_to_uniform": (
            optimum_probability / feasible_mass / arrays.uniform_optimum_probability
            if feasible_mass
            else 0.0
        ),
        "near_1pct_probability": float(
            probabilities[feasible][feasible_regrets <= 0.01 + 1e-12].sum()
        ),
        "near_5pct_probability": float(
            probabilities[feasible][feasible_regrets <= 0.05 + 1e-12].sum()
        ),
        "expected_feasible_objective_conditional": feasible_b,
        "exact_best_objective": float(arrays.reference["best_objective"]),
        "exact_worst_objective": float(arrays.reference["worst_objective"]),
        "exact_optimum_count": len(arrays.reference["optimal_state_indices"]),
    }
    metrics["shot_metrics"] = [
        {
            "shot_budget": int(budget),
            "exact_hit_probability": float(
                1.0 - (1.0 - optimum_probability) ** int(budget)
            ),
            "expected_best_normalised_regret": expected_best(
                regrets, probabilities, int(budget)
            ),
            "no_feasible_sample_probability": float(
                (1.0 - feasible_mass) ** int(budget)
            ),
        }
        for budget in shot_budgets
    ]
    return metrics
