"""Check supplied numerical evidence and compare a fresh reproduction."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from gmv_qaoa.model import load_instances, all_state_vectors, build_qubo, qubo_direct_energies, qubo_matrix_energies, qubo_to_ising, ising_energies
from gmv_qaoa.qaoa import model_arrays, final_distribution, initial_parameters, interpolate_p1_to_p2, verify_qiskit_equivalence, reference_comparison
from gmv_qaoa.metrics import distribution_metrics, summarize_runs, paired_comparisons
from gmv_qaoa.classical import classical_results
from reproduce import run_plan, angle_checks
import supporting

TABLE_NAMES = ['exact-solutions.csv', 'classical-baselines.csv', 'policy-breakpoints.csv',
               'qaoa-runs.csv', 'qaoa-summary.csv', 'paired-comparisons.csv', 'sampling-metrics.csv', 'angle-checks.csv']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, float_precision='round_trip')


def close(actual, expected, tolerance, label):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, f'{label}: shape differs')
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), f'{label}: non-finite value')
    error = float(np.max(np.abs(actual - expected), initial=0.0))
    require(error <= tolerance, f'{label}: error {error:.3g} exceeds {tolerance}')
    return error


def compare_frames(actual, expected, tolerance, label):
    require(list(actual.columns) == list(expected.columns) and actual.shape == expected.shape, f'{label}: table schema or row count differs')
    maximum = 0.0
    for column in actual.columns:
        left, right = actual[column], expected[column]
        if column in ('initial_parameters', 'final_parameters', 'parameters'):
            for a, b in zip(left, right, strict=True):
                maximum = max(maximum, close(json.loads(a), json.loads(b), 1e-9, f'{label}/{column}'))
        elif pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            maximum = max(maximum, close(left, right, tolerance, f'{label}/{column}'))
        else:
            # Object columns can contain a mix of optional numbers and blanks.
            for row, (a, b) in enumerate(zip(left, right, strict=True)):
                if str(a) == str(b):
                    continue
                try:
                    a_number, b_number = float(a), float(b)
                except (ValueError, TypeError):
                    raise ValueError(f'{label}/{column}: text differs at row {row}') from None
                maximum = max(maximum, close(a_number, b_number, tolerance, f'{label}/{column}/{row}'))
    return maximum


def compare_json(actual, expected, tolerance, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f'{label}: JSON fields differ')
        for key in expected:
            compare_json(actual[key], expected[key], tolerance, f'{label}/{key}')
    elif isinstance(expected, (int, float, list)) and not isinstance(expected, bool):
        close(actual, expected, tolerance, label)
    else:
        require(actual == expected, f'{label}: value differs')


def metadata_checks(folder, study, *, support=False):
    metadata = json.loads((folder / 'metadata.json').read_text())
    tables = supporting.TABLE_NAMES if support else TABLE_NAMES
    files = supporting.RESULT_NAMES if support else TABLE_NAMES + ['qaoa-probabilities.npz']
    require(metadata['input_sha256'] == sha256(ROOT / 'data/processed/localities.csv'), 'Processed input checksum differs from the published snapshot')
    require(metadata['study_sha256'] == sha256(ROOT / 'study.json'), 'Study configuration checksum differs')
    if support:
        require(metadata['supporting_study_sha256'] == sha256(ROOT / 'supporting-study.json'), 'Supporting configuration checksum differs')
    require(metadata['run_count'] == study['expected_runs'], 'Run count differs')
    require(set(metadata['files']) == set(files), 'Result manifest coverage differs')
    require(set(metadata['csv_records']) == set(tables), 'CSV row-count coverage differs')
    for name, expected in metadata['files'].items():
        require(sha256(folder / name) == expected, f'Checksum mismatch: {name}')
    for name, expected in metadata['csv_records'].items():
        require(len(read_csv(folder / name)) == expected, f'Record count mismatch: {name}')
    for name in ['numpy', 'pandas', 'scipy', 'qiskit']:
        require(importlib.metadata.version(name) == metadata['packages'][name], f'{name} version differs; install the locked environment')
    return metadata


def scientific_checks(folder, study, *, support=False):
    instances = load_instances(ROOT / 'data/processed/localities.csv', study)
    max_probability_error = 0.0
    for n in ([] if support else [6, 8, 10, 12]):
        instance = instances[n]
        vectors = all_state_vectors(n)
        for factor in [0.5, 1.1, 2.0]:
            qubo = build_qubo(instance, 0.5, factor)
            direct = qubo_direct_energies(instance, vectors, 0.5, qubo.penalty)
            close(qubo_matrix_energies(qubo, vectors), direct, 1e-10, f'n={n} QUBO')
            close(ising_energies(qubo_to_ising(qubo), vectors), direct, 1e-10, f'n={n} Ising')
        arrays = model_arrays(instance)
        for depth in [1, 2]:
            parameters = initial_parameters(depth, 47, (0, 2 * np.pi), (0, np.pi / 2))
            verify_qiskit_equivalence(arrays.penalty_energies, n, depth, parameters)
            reference_comparison(arrays.objective_b, arrays.basis, depth, parameters)
        print(f'Validated n={n}: QUBO/Ising energies and both Qiskit reference circuits.', flush=True)
    rows = read_csv(folder / 'qaoa-runs.csv').to_dict('records')
    plan = run_plan(study)
    require([r['run_id'] for r in rows] == [r['run_id'] for r in plan], 'Run IDs/order differ from the declared matrix')
    arrays_cache, depth_one = {}, {}
    shot_rows = []
    with np.load(folder / 'qaoa-probabilities.npz', allow_pickle=False) as distributions:
        require(set(distributions.files) == {r['run_id'] for r in rows}, 'Missing or additional probability vectors')
        for spec, row in zip(plan, rows, strict=True):
            run_id = row['run_id']
            for key, value in spec.items():
                require(row[key] == value, f'{run_id}: configuration field {key} differs')
            key = (row['n'], row['penalty_factor'])
            if key not in arrays_cache:
                arrays_cache[key] = model_arrays(instances[row['n']], study['objective']['alpha'], row['penalty_factor'])
            arrays = arrays_cache[key]
            final = np.array(json.loads(row['final_parameters']))
            initial = np.array(json.loads(row['initial_parameters']))
            parent_key = (row['experiment'], row['n'], row['method'], row['optimizer'], row['seed'])
            expected_initial = (interpolate_p1_to_p2(depth_one[parent_key]) if row['initialisation'] == 'duplicate_p1'
                                else initial_parameters(row['depth'], row['seed'], tuple(study['initial_angles']['gamma']), tuple(study['initial_angles']['beta'])))
            close(initial, expected_initial, 1e-12, f'{run_id}: initialisation')
            if row['depth'] == 1 and row['experiment'].endswith('_comparison'):
                depth_one[parent_key] = final
            probabilities = distributions[run_id]
            require(probabilities.shape == (2 ** row['n'],), f'{run_id}: incorrect state count')
            require(np.isfinite(probabilities).all() and np.all(probabilities >= 0), f'{run_id}: invalid probabilities')
            close(probabilities.sum(), 1, 1e-12, f'{run_id}: normalization')
            predicted, energy = final_distribution(arrays, row['method'], row['depth'], final)
            max_probability_error = max(max_probability_error, close(probabilities, predicted, 1e-9, run_id))
            close(row['final_expected_energy'], energy, 1e-10, f'{run_id}: expected energy')
            close(row['training_expected_energy'], energy, 1e-10, f'{run_id}: training energy')
            require(0 < row['optimizer_nfev'] <= row['evaluation_cap'], f'{run_id}: evaluation cap violated')
            require(0 <= row['evaluations_outside_original_domain'] <= row['optimizer_nfev'], f'{run_id}: invalid outside-domain count')
            if row['method'] == 'xy_dicke':
                invalid = arrays.vectors.sum(axis=1) != arrays.k
                close(probabilities[invalid].sum(), 0, 1e-12, f'{run_id}: feasible-subspace leakage')
            calculated = distribution_metrics(arrays, probabilities, study['shot_budgets'])
            shots = calculated.pop('shot_metrics')
            for metric, value in calculated.items():
                close(row[metric], value, 1e-10, f'{run_id}/{metric}')
            close(row['expected_best_regret_b4096'], shots[-1]['expected_best_normalised_regret'], 1e-10, f'{run_id}: regret')
            shot_rows.extend({'run_id': run_id, **s} for s in shots)
    for n in ([] if support else [6, 8, 10, 12]):
        arrays = arrays_cache[(n, study['objective']['penalty_factor'])]
        probabilities = np.zeros(2 ** n)
        probabilities[arrays.basis.state_indices] = 1 / len(arrays.basis.state_indices)
        shots = distribution_metrics(arrays, probabilities, study['shot_budgets'])['shot_metrics']
        shot_rows.extend({'run_id': f'uniform-valid-n{n:02}', **s} for s in shots)
    report = {'validated_runs': len(rows), 'maximum_probability_reconstruction_error': max_probability_error}
    if support:
        mixer, costs, symmetry = supporting.period_checks(instances, study)
        calculated_tables = {'robustness-summary.csv': supporting.robustness_summary(rows, study),
                             'mixer-period-checks.csv': mixer, 'cost-period-counterexamples.csv': costs}
        compare_json(json.loads((folder / 'symmetry-validation.json').read_text()), symmetry,
                     study['symmetries']['probability_tolerance'], 'symmetry-validation.json')
        report.update(mixer_period_checks=len(mixer), cost_period_counterexamples=len(costs),
                      paired_groups=len(calculated_tables['robustness-summary.csv']))
    else:
        exact, baselines, intervals = classical_results(instances, study)
        calculated_tables = {'exact-solutions.csv': exact, 'classical-baselines.csv': baselines, 'policy-breakpoints.csv': intervals,
                             'qaoa-summary.csv': summarize_runs(rows), 'paired-comparisons.csv': paired_comparisons(rows, study),
                             'sampling-metrics.csv': shot_rows, 'angle-checks.csv': angle_checks(instances, study)}
        report.update(reference_circuit_checks=16, classical_baseline_rows=len(baselines), allocation_intervals=len(intervals))
    for name, records in calculated_tables.items():
        compare_frames(read_csv(folder / name), pd.DataFrame(records), 1e-10, name)
    return report


def compare_results(actual, reference, study, *, support=False):
    maximum_table_error = 0.0
    for name in (supporting.TABLE_NAMES if support else TABLE_NAMES):
        maximum_table_error = max(maximum_table_error, compare_frames(read_csv(actual / name), read_csv(reference / name), study['tolerances']['metrics'], name))
    maximum_probability_error = 0.0
    with np.load(actual / 'qaoa-probabilities.npz', allow_pickle=False) as a, np.load(reference / 'qaoa-probabilities.npz', allow_pickle=False) as b:
        require(set(a.files) == set(b.files), 'Probability archive run IDs differ')
        for key in a.files:
            maximum_probability_error = max(maximum_probability_error, close(a[key], b[key], study['tolerances']['probabilities'], key))
    if support:
        compare_json(json.loads((actual / 'symmetry-validation.json').read_text()),
                     json.loads((reference / 'symmetry-validation.json').read_text()),
                     study['symmetries']['probability_tolerance'], 'symmetry-validation.json')
    return {'maximum_table_error': maximum_table_error, 'maximum_probability_error': maximum_probability_error}


def check_original_overlap(original_folder, supporting_folder):
    original = read_csv(original_folder / 'qaoa-runs.csv')
    original = original[original.experiment == 'direct_comparison']
    expanded = read_csv(supporting_folder / 'qaoa-runs.csv')
    expanded = expanded[(expanded.optimizer == 'COBYLA') & (expanded.seed < 10)]
    require(len(original) == len(expanded) == 160, 'Original ten-seed overlap is incomplete')
    columns = [c for c in original.columns if c not in ('experiment', 'run_id')]
    compare_frames(original[columns], expanded[columns], 1e-10, 'Original ten-seed overlap')
    return len(expanded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--supporting-results', type=Path, default=ROOT / 'results/supporting')
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs/verification.json')
    args = parser.parse_args()
    report_path = args.report.resolve()
    if (ROOT / 'outputs').resolve() not in report_path.parents:
        parser.error('Write verification reports under outputs/; published files are protected.')
    report = {'result': 'FAIL'}
    try:
        study = json.loads((ROOT / 'study.json').read_text())
        metadata_checks(ROOT / 'results', study)
        metadata_checks(args.results, study)
        report.update(scientific_checks(args.results, study))
        if args.results.resolve() != (ROOT / 'results').resolve():
            report['reference_comparison'] = compare_results(args.results, ROOT / 'results', study)
        extra_study = supporting.load_study()
        metadata_checks(ROOT / 'results/supporting', extra_study, support=True)
        metadata_checks(args.supporting_results, extra_study, support=True)
        report['supporting'] = scientific_checks(args.supporting_results, extra_study, support=True)
        report['supporting']['original_runs_preserved'] = check_original_overlap(args.results, args.supporting_results)
        if args.supporting_results.resolve() != (ROOT / 'results/supporting').resolve():
            report['supporting']['reference_comparison'] = compare_results(args.supporting_results, ROOT / 'results/supporting', extra_study, support=True)
        report['result'] = 'PASS'
    except Exception as error:
        report['error'] = str(error)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['result'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
