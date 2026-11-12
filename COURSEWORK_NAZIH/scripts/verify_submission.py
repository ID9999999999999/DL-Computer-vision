"""Read-only integrity checks for the archived KSDD2 coursework submission.

Works with the submitted CODE.zip + RESULTS.zip. Requires only Python's
standard library; does not require GPU, dataset download, or model training.
The only optional heavyweight operation is validating the 26 checkpoint files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def get(zipf: zipfile.ZipFile, name: str):
    return json.loads(zipf.read(name).decode('utf-8'))


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def validate(results_zip: Path, models_dir: Path | None = None):
    source = read_json(ROOT / 'input' / 'target_manifest.json')
    test = read_json(ROOT / 'input' / 'test_manifest.json')
    check(len(source) == 70, 'Missing frozen baseline manifest')
    check(Counter(x['label'] for x in source) == {0: 35, 1: 35}, 'Bad original class distribution')
    check(len(test) == 1004, 'Bad official test manifest')
    check(Counter(x['label'] for x in test) == {0: 894, 1: 110}, 'Bad official test labels')
    for x in source:
        p = ROOT / 'input' / 'train' / Path(x['zip_path']).name
        check(p.is_file() and sha256_file(p) == x['sha256'], f'Missing/modified 70-image reference: {p}')
    test_hashes = {x['sha256'] for x in test}
    check(not {x['sha256'] for x in source} & test_hashes, 'Train/test file overlap')

    with zipfile.ZipFile(results_zip) as z:
        check(z.testzip() is None, 'RESULTS.zip CRC error')
        paths = set(z.namelist())
        for seed in (20261028, 20261029, 20261030):
            for loss in ('ce', 'poly1'):
                name = f'historical/baseline_eval/eval_{seed}_{loss}_epoch12.json'
                train = f'historical/baseline_train/target_train_{seed}_{loss}.json'
                check(name in paths and train in paths, f'Missing historical record: {name}')
                record = get(z, name)
                ref = get(z, train)
                check(record['seed'] == seed and record['arm'] == loss and ref['seed'] == seed, name)
                check(record['checkpoint_sha256'] == ref['checkpoint_sha256'], 'Historical checkpoint SHA mismatch')
                check(len(record['predictions']) == 1004, 'Historical prediction count')
                check([p['sha256'] for p in record['predictions']] == [x['sha256'] for x in test], 'Historical sample mismatch')

        evaluated = 0
        fixed_models = []
        for size in (140, 280, 420):
            for seed in (20261028, 20261029, 20261030):
                for loss in ('ce', 'poly1'):
                    stub = f'size{size}_seed{seed}_{loss}'
                    ev = f'evaluations/eval_{stub}.json'
                    tr = f'training_metadata/{stub}.json'
                    check(ev in paths and tr in paths, f'Missing fixed-budget run {stub}')
                    a = get(z, ev)
                    b = get(z, tr)
                    check((a['size'], a['seed'], a['loss']) == (size, seed, loss), ev)
                    check((b['size'], b['seed'], b['loss']) == (size, seed, loss), tr)
                    check(a['checkpoint_sha256'] == b['checkpoint_sha256'], f'Fixed model mismatch: {stub}')
                    check(b['total_optimizer_steps'] == 216 and len(b['train_history']) == 12, f'Incorrect budget {stub}')
                    check(len(a['predictions']) == 1004, f'Incorrect prediction count {stub}')
                    check([r['sha256'] for r in a['predictions']] == [r['sha256'] for r in test], f'Wrong test order {stub}')
                    correct = 0
                    cm = [[0, 0], [0, 0]]
                    for pred, src in zip(a['predictions'], test):
                        check(pred['label'] == src['label'] and pred['path'] == src['path'], f'Test metadata mismatch {stub}')
                        check(pred['prediction'] in (0, 1), f'Bad binary pred {stub}')
                        check(math.isfinite(pred['defect_score']), f'Nonfinite score {stub}')
                        check(pred['prediction'] == int(pred['defect_score'] > 0), f'Score/decision mismatch {stub}')
                        cm[pred['label']][pred['prediction']] += 1
                        correct += pred['label'] == pred['prediction']
                    check(cm == a['metrics']['confusion_matrix'], f'Wrong confusion matrix {stub}')
                    check(abs(correct / 1004 - a['metrics']['accuracy']) < 1e-12, f'Wrong accuracy {stub}')
                    f1 = sum(2 * cm[i][i] / (sum(cm[i]) + cm[0][i] + cm[1][i]) for i in (0, 1)) / 2
                    check(abs(f1 - a['metrics']['macro_f1']) < 1e-12, f'Wrong Macro-F1 {stub}')
                    evaluated += 1
                    fixed_models.append((f'fixed216_{stub}.pt', b['checkpoint_sha256']))

        check(evaluated == 18, 'Must have 18 recent fixed-budget evaluations')
        validation_reference = None
        full_models = []
        full_runs = 0
        for size in (70, 140, 280, 420):
            for loss in ('ce', 'poly1'):
                run = f'n{size}_seed20261028_{loss}'
                complete = get(z, f'full_epochs/{run}/COMPLETE.json')
                split = get(z, f'full_epochs/{run}/split.json')
                ident = complete['identity']
                check((ident['size'], ident['seed'], ident['loss']) == (size, 20261028, loss), run)
                hist = complete['history']
                check(len(hist) == 30, f'Not 30 true full epochs: {run}')
                steps = math.ceil(size / 4)
                check(ident['total_planned_optimizer_steps'] == steps * 30, f'Wrong expected steps {run}')
                check(all(x['full_epoch'] == i + 1 and x['training_optimizer_steps'] == steps and x['training_samples_seen'] == size
                          for i, x in enumerate(hist)), f'Invalid full-epoch history {run}')
                check(complete['official_test_evaluated'] is False, f'Test policy violated {run}')
                check(sha256_bytes(z.read(f'full_epochs/{run}/split.json')) == complete['split_sha256'], f'Split SHA256 mismatch {run}')
                train = split['train_rows'] if 'train_rows' in split else split['training_rows']
                val = split['validation_rows']
                check(len(train) == size and len(val) == 180, f'Wrong split size {run}')
                check(Counter(x['label'] for x in train) == {0: size//2, 1: size//2}, f'Unbalanced training {run}')
                check(Counter(x['label'] for x in val) == {0: 150, 1: 30}, f'Invalid validation distribution {run}')
                check(not {x['sha256'] for x in train} & {x['sha256'] for x in val}, f'Train/validation leakage {run}')
                check(not {x['sha256'] for x in val} & test_hashes, f'Validation/test exact overlap {run}')
                if validation_reference is None:
                    validation_reference = val
                check(val == validation_reference, f'Validation not frozen identically: {run}')
                best = complete['best_epoch']
                check(1 <= best <= 30, f'Invalid best epoch {run}')
                highest = max(x['validation']['macro_f1'] for x in hist)
                check(abs(highest - complete['best_validation_macro_f1']) < 1e-12,
                      f'Best metric does not match history: {run}')
                # Earliest epoch that achieved the best Macro-F1.
                earliest = next(i+1 for i, x in enumerate(hist)
                                if abs(x['validation']['macro_f1'] - highest) < 1e-12)
                check(earliest == best, f'Selection rule not followed {run}')
                full_models.append((f'full30_{run}.pt', complete['best_model_sha256']))
                full_runs += 1
        check(full_runs == 8, 'Not all 8 full-epoch runs checked')

    checkpoint_status = 'not checked (no --models-dir)'
    if models_dir is not None:
        expected = fixed_models + full_models
        for filename, sha in expected:
            file = models_dir / filename
            check(file.is_file(), f'Missing model: {filename}')
            check(sha256_file(file) == sha, f'Model SHA256 mismatch: {filename}')
        checkpoint_status = f'{len(expected)} model SHA256 values verified'

    print('PASS: 70 original train images / 6 historical records / 18 trained+evaluated fixed-budget models')
    print('PASS: 8 complete 30-full-epoch histories / identical 180-image validation / frozen splits')
    print('PASS: decisions, confusion matrices, F1 and saved metadata independently recalculated')
    print(f'MODELS: {checkpoint_status}')
    print('NOTE: these checks verify archived records; they do not retrain the models.')
    print('NOTE: official test results were exploratory; this does not establish group independence.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results-zip',type=Path,required=True)
    p.add_argument('--models-dir',type=Path)
    a = p.parse_args()
    try:
        validate(a.results_zip,a.models_dir)
    except (AssertionError, ValueError, FileNotFoundError, KeyError, zipfile.BadZipFile) as exc:
        sys.exit(f'FAIL: {exc}')
