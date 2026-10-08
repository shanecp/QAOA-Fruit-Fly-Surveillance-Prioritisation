"""Reproduce the published numerical results from the frozen locality input."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from gmv_qaoa.model import load_instances
from gmv_qaoa.classical import classical_results
from gmv_qaoa.qaoa import model_arrays, initial_parameters, interpolate_p1_to_p2, optimise, final_distribution
from gmv_qaoa.metrics import distribution_metrics, summarize_runs, paired_comparisons


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_plan(study):
    plan = []
    for name, settings in study['experiments'].items():
        for n in settings['sizes']:
            for depth in settings['depths']:
                for factor in settings.get('factors', [study['objective']['penalty_factor']]):
                    for optimizer in settings.get('optimizers', [settings.get('optimizer')]):
                        caps = settings.get('caps', [settings.get('cap')])
                        if isinstance(caps, dict):
                            caps = [caps[str(depth)]]
                        for cap in caps:
                            for method in settings['methods']:
                                for seed in study['seeds']:
                                    wrapped = name in ('optimizer_audit', 'wrapped_comparison')
                                    row = {'experiment': name, 'n': n, 'k': study['instances']['budgets'][str(n)],
                                           'depth': depth, 'penalty_factor': factor, 'optimizer': optimizer,
                                           'evaluation_cap': cap, 'method': method, 'seed': seed,
                                           'angle_handling': 'wrapped' if wrapped else 'direct',
                                           'initialisation': 'duplicate_p1' if name.endswith('_comparison') and depth == 2 else 'random'}
                                    row['run_id'] = f'{name}-{method}-n{n:02}-p{depth}-{optimizer.lower()}-b{cap:04}-a{round(factor * 100):03}-s{seed:02}'
                                    plan.append(row)
    if len(plan) != study['expected_runs'] or len({r['run_id'] for r in plan}) != len(plan):
        raise ValueError('Experiment matrix does not match the declared run count')
    return plan


def run_experiments(instances, study):
    plan = run_plan(study)
    arrays_cache, depth_one = {}, {}
    rows, probabilities_by_run, shot_rows = [], {}, []
    for position, spec in enumerate(plan, 1):
        key = (spec['n'], spec['penalty_factor'])
        if key not in arrays_cache:
            arrays_cache[key] = model_arrays(instances[spec['n']], study['objective']['alpha'], spec['penalty_factor'], study['tolerances']['objective'])
        arrays = arrays_cache[key]
        parent_key = (spec['experiment'], spec['n'], spec['method'], spec['optimizer'], spec['seed'])
        if spec['initialisation'] == 'duplicate_p1':
            initial = interpolate_p1_to_p2(depth_one[parent_key])
        else:
            initial = initial_parameters(spec['depth'], spec['seed'], tuple(study['initial_angles']['gamma']), tuple(study['initial_angles']['beta']))
        final, probabilities, optimizer = optimise(arrays, spec['method'], spec['depth'], initial, spec['optimizer'],
                                                    spec['evaluation_cap'], spec['angle_handling'] == 'wrapped', study)
        if spec['depth'] == 1 and spec['experiment'].endswith('_comparison'):
            depth_one[parent_key] = final
        metrics = distribution_metrics(arrays, probabilities, study['shot_budgets'])
        shots = metrics.pop('shot_metrics')
        rows.append(spec | optimizer | metrics | {
            'alpha': study['objective']['alpha'], 'penalty_A': arrays.penalty,
            'initial_parameters': json.dumps(initial.tolist()), 'final_parameters': json.dumps(final.tolist()),
            'expected_best_regret_b4096': next(s['expected_best_normalised_regret'] for s in shots if s['shot_budget'] == 4096),
        })
        probabilities_by_run[spec['run_id']] = probabilities
        shot_rows.extend({'run_id': spec['run_id'], **s} for s in shots)
        if position == 1 or position % 20 == 0 or position == len(plan):
            print(f'Optimisation {position}/{len(plan)}: {spec["run_id"]}', flush=True)
    for n in study['experiments'].get('penalty_diagnostic', {}).get('sizes', []):
        arrays = arrays_cache[(n, study['objective']['penalty_factor'])]
        probabilities = np.zeros(2 ** n)
        probabilities[arrays.basis.state_indices] = 1.0 / len(arrays.basis.state_indices)
        uniform = distribution_metrics(arrays, probabilities, study['shot_budgets'])
        shot_rows.extend({'run_id': f'uniform-valid-n{n:02}', **s} for s in uniform['shot_metrics'])
    return rows, probabilities_by_run, shot_rows


def angle_checks(instances, study):
    settings = study['angle_check']
    arrays = model_arrays(instances[settings['n']], study['objective']['alpha'], study['objective']['penalty_factor'])
    parameters = np.array(settings['parameters'])
    rows = []
    for method in ['penalty_x', 'xy_dicke']:
        base, _ = final_distribution(arrays, method, settings['depth'], parameters)
        for label, index, shift in [('cost_gamma_plus_2pi', 0, 2 * np.pi), ('mixer_beta_plus_pi_over_2', settings['depth'], np.pi / 2)]:
            shifted = parameters.copy()
            shifted[index] += shift
            probabilities, _ = final_distribution(arrays, method, settings['depth'], shifted)
            rows.append({'n': settings['n'], 'depth': settings['depth'], 'method': method, 'shift': label,
                         'probability_l1': float(np.abs(base - probabilities).sum())})
    return rows


def write_results(output, inputs, study_path, study, exact, baselines, intervals, rows, probabilities, shots, angles):
    output.mkdir(parents=True, exist_ok=True)
    tables = {'exact-solutions.csv': exact, 'classical-baselines.csv': baselines, 'policy-breakpoints.csv': intervals,
              'qaoa-runs.csv': rows, 'qaoa-summary.csv': summarize_runs(rows), 'paired-comparisons.csv': paired_comparisons(rows, study),
              'sampling-metrics.csv': shots, 'angle-checks.csv': angles}
    for name, records in tables.items():
        pd.DataFrame(records).to_csv(output / name, index=False)
    np.savez_compressed(output / 'qaoa-probabilities.npz', **probabilities)
    metadata = {
        'schema_version': 1, 'input_sha256': sha256(inputs), 'study_sha256': sha256(study_path),
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'qiskit', 'geopandas', 'shapely', 'pyproj', 'pypdf']},
        'run_count': len(rows), 'state_index_convention': 'Array index z encodes x_i=(z>>i)&1; canonical bitstrings list x_0 first.',
        'parameter_order': 'gamma_1 ... gamma_p, beta_1 ... beta_p',
        'csv_records': {name: len(records) for name, records in tables.items()},
        'tolerances': study['tolerances'],
        'files': {name: sha256(output / name) for name in [*tables, 'qaoa-probabilities.npz']},
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'data/processed/localities.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/reproduced')
    parser.add_argument('--report', type=Path, help='Optionally rebuild inputs from the separately obtained report')
    parser.add_argument('--geometry', type=Path, help='Original frozen geometry; required with --report')
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / 'results').resolve() or (ROOT / 'results').resolve() in args.output.resolve().parents:
        parser.error('Write fresh results outside the published results directory.')
    if bool(args.report) != bool(args.geometry):
        parser.error('Supply both --report and --geometry to run ETL.')
    if args.report:
        from etl import rebuild
        args.inputs = rebuild(args.report, args.geometry, args.output / 'inputs/localities.csv')
    study_path = ROOT / 'study.json'
    study = json.loads(study_path.read_text())
    instances = load_instances(args.inputs, study)
    print('Computing exact solutions, classical baselines and policy breakpoints.', flush=True)
    exact, baselines, intervals = classical_results(instances, study)
    rows, probabilities, shots = run_experiments(instances, study)
    write_results(args.output, args.inputs, study_path, study, exact, baselines, intervals,
                  rows, probabilities, shots, angle_checks(instances, study))
    print(f'Reproduced {len(rows)} runs. Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
