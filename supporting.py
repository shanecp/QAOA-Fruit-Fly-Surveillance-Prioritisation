"""Reproduce the mixer-period checks, cost-period counterexamples and 20-seed comparison."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd

from reproduce import ROOT, sha256, angle_checks, run_experiments
from gmv_qaoa.model import load_instances
from gmv_qaoa.qaoa import model_arrays, final_distribution
from gmv_qaoa.metrics import bootstrap_interval

TABLE_NAMES = ['qaoa-runs.csv', 'robustness-summary.csv', 'mixer-period-checks.csv', 'cost-period-counterexamples.csv']
RESULT_NAMES = TABLE_NAMES + ['qaoa-probabilities.npz', 'symmetry-validation.json']


def load_study():
    study = json.loads((ROOT / 'study.json').read_text())
    study.update(json.loads((ROOT / 'supporting-study.json').read_text()))
    return study


def period_checks(instances, study):
    spec = study['symmetries']
    design = study['experiments']['robustness_comparison']
    rng = np.random.default_rng(spec['test_seed'])
    rows, costs = [], []
    for n in design['sizes']:
        arrays = model_arrays(instances[n], study['objective']['alpha'], study['objective']['penalty_factor'])
        for method in design['methods']:
            period = spec['mixer_periods'][method]
            energies = arrays.penalty_energies if method == 'penalty_x' else arrays.objective_b[arrays.basis.state_indices]
            residual = float(np.max(np.abs(np.exp(-1j * spec['proposed_cost_period'] * (energies - energies[0])) - 1)))
            if residual <= spec['probability_tolerance']:
                raise ValueError('Expected counterexample to the proposed cost period')
            costs.append({'n': n, 'method': method, 'proposed_cost_period': spec['proposed_cost_period'],
                          'max_relative_phase_error': residual, 'is_period': False})
            for depth in design['depths']:
                for draw in range(spec['draws_per_size_depth_method']):
                    values = rng.uniform(*spec['parameter_range'], size=2 * depth)
                    baseline, _ = final_distribution(arrays, method, depth, values)
                    for layer in range(depth):
                        shifted = values.copy()
                        shifted[depth + layer] += period
                        candidate, _ = final_distribution(arrays, method, depth, shifted)
                        error = float(np.max(np.abs(baseline - candidate)))
                        if not np.isfinite(error) or error > spec['probability_tolerance']:
                            raise ValueError('Mixer period did not preserve probabilities')
                        rows.append({'n': n, 'method': method, 'depth': depth, 'draw': draw,
                                     'shifted_layer': layer, 'period': period,
                                     'parameters': json.dumps(values.tolist()), 'max_probability_error': error})
    if len(rows) != spec['expected_checks']:
        raise ValueError('Mixer-period check count differs')
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    eigenvalues = np.linalg.eigvalsh((np.kron(x, x) + np.kron(y, y)) / 2)
    if not np.allclose(eigenvalues, [-1, 0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError('XY edge spectrum differs')
    invalid = {'n': study['angle_check']['n'], 'depth': study['angle_check']['depth']}
    labels = {'cost_gamma_plus_2pi': 'gamma_plus_2pi_probability_l1',
              'mixer_beta_plus_pi_over_2': 'beta_plus_pi_over_2_probability_l1'}
    for row in angle_checks(instances, study):
        if row['probability_l1'] <= spec['probability_tolerance']:
            raise ValueError('Expected invalid-period probability counterexample')
        invalid.setdefault(row['method'], {})[labels[row['shift']]] = row['probability_l1']
    summary = {
        'status': 'PASS', 'safe_mixer_shift_checks': len(rows),
        'max_safe_shift_probability_difference': max(row['max_probability_error'] for row in rows),
        'xy_edge_eigenvalues': eigenvalues.tolist(), 'invalid_period_counterexamples': invalid,
        'interpretation': 'X beta+pi is a global phase; ordered XY beta+2*pi repeats every edge. Cost T requires all T*(E-E_ref)/(2*pi) integral. Counterexamples reject the proposed periods only; no claim of aperiodicity or smallest period.',
    }
    return rows, costs, summary


def robustness_summary(rows, study):
    frame = pd.DataFrame(rows)
    spec = study['experiments']['robustness_comparison']
    output = []
    for optimizer in spec['optimizers']:
        for n in spec['sizes']:
            for depth in spec['depths']:
                group = frame[(frame.optimizer == optimizer) & (frame.n == n) & (frame.depth == depth)]
                penalty, xy = [group[group.method == method].sort_values('seed') for method in spec['methods']]
                if penalty.seed.tolist() != study['seeds'] or xy.seed.tolist() != study['seeds']:
                    raise ValueError('Missing or mismatched paired seeds')
                differences = xy.exact_optimum_probability.to_numpy(float) - penalty.exact_optimum_probability.to_numpy(float)
                low, high = bootstrap_interval(differences, np.random.default_rng(spec['bootstrap_seed']), spec['bootstrap_resamples'])
                output.append({
                    'optimizer': optimizer, 'n': n, 'k': int(penalty.k.iloc[0]), 'depth': depth,
                    'paired_seed_count': len(penalty), 'uniform_optimum_probability': float(penalty.uniform_optimum_probability.iloc[0]),
                    'penalty_median': float(penalty.exact_optimum_probability.median()),
                    'xy_median': float(xy.exact_optimum_probability.median()),
                    'median_paired_xy_minus_penalty': float(np.median(differences)),
                    'paired_bootstrap_95_low': float(low), 'paired_bootstrap_95_high': float(high),
                    'xy_higher_pair_count': int((differences > 0).sum()),
                    'penalty_higher_pair_count': int((differences < 0).sum()), 'ties': int((differences == 0).sum()),
                    'bootstrap_seed': spec['bootstrap_seed'], 'bootstrap_resamples': spec['bootstrap_resamples'],
                })
    return output


def write_metadata(output, inputs, study):
    metadata = {
        'schema_version': 1, 'input_sha256': sha256(inputs), 'study_sha256': sha256(ROOT / 'study.json'),
        'supporting_study_sha256': sha256(ROOT / 'supporting-study.json'),
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'qiskit']},
        'run_count': study['expected_runs'], 'tolerances': study['tolerances'],
        'csv_records': {name: len(pd.read_csv(output / name)) for name in TABLE_NAMES},
        'files': {name: sha256(output / name) for name in RESULT_NAMES},
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'data/processed/localities.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/supporting')
    args = parser.parse_args()
    if (ROOT / 'outputs').resolve() not in args.output.resolve().parents:
        parser.error('Write fresh supporting results under outputs/.')
    study = load_study()
    instances = load_instances(args.inputs, study)
    mixer, costs, symmetry = period_checks(instances, study)
    rows, probabilities, _ = run_experiments(instances, study)
    tables = {'qaoa-runs.csv': rows, 'robustness-summary.csv': robustness_summary(rows, study),
              'mixer-period-checks.csv': mixer, 'cost-period-counterexamples.csv': costs}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, records in tables.items():
        pd.DataFrame(records).to_csv(args.output / name, index=False)
    np.savez_compressed(args.output / 'qaoa-probabilities.npz', **probabilities)
    (args.output / 'symmetry-validation.json').write_text(json.dumps(symmetry, indent=2) + '\n')
    write_metadata(args.output, args.inputs, study)
    print(f'Reproduced {len(rows)} supporting runs and {len(mixer)} mixer-period checks. Results: {args.output}')


if __name__ == '__main__':
    main()
