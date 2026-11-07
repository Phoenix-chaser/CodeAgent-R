"""Content-verified receipts for the single-GPU experiment runner (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(paths):
    result = {}
    for name in paths:
        path = Path(name).resolve()
        if path.is_file():
            result[str(path)] = file_hash(path)
        elif path.is_dir():
            files = sorted(item for item in path.rglob('*') if item.is_file()
                           and '__pycache__' not in item.parts and item.suffix != '.pyc')
            if not files:
                raise ValueError(f'Empty artifact directory: {path}')
            result[str(path)] = {str(item.relative_to(path)): file_hash(item) for item in files}
        else:
            raise ValueError(f'Missing artifact: {path}')
    return result


def check_receipt(receipt, inputs, outputs):
    path = Path(receipt)
    if not path.exists():
        return False, 'no completion receipt'
    try:
        previous = json.loads(path.read_text(encoding='utf-8'))
        current = {'inputs': snapshot(inputs), 'outputs': snapshot(outputs)}
    except (ValueError, OSError) as error:
        return False, str(error)
    return (previous == current, 'verified' if previous == current else 'input/output content changed')


def mark_receipt(receipt, inputs, outputs):
    data = {'inputs': snapshot(inputs), 'outputs': snapshot(outputs)}
    path = Path(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2), encoding='utf-8')
    temporary.replace(path)


def initialize():
    env = os.environ
    exp = Path(env.get('EXP_DIR', 'data/experiment-v2')).resolve()
    config = {'schema_version': 1, 'experiment_dir': str(exp)}
    defaults = {
        'MODEL': 'Qwen/Qwen2.5-Coder-7B-Instruct',
        'MODEL_REVISION': 'c03e6d358207e414f1eca0bb1891e29f1db0e242',
        'TRAIN_CANDIDATES': '12', 'DEV_CANDIDATES': '4', 'MAX_PER_TASK': '1',
        'EPOCHS': '3', 'LR': '2e-5', 'BATCH_SIZE': '1', 'GRAD_ACCUM': '8',
        'MAX_LENGTH': '2048', 'LORA_R': '16', 'LORA_ALPHA': '32',
        'LORA_DROPOUT': '0.05', 'PATIENCE': '2', 'MIN_DELTA': '0.001',
        'SEED': '42', 'MAX_NEW_TOKENS': '2048', 'QUANTIZATION': 'nf4',
        'TEACHER_MODEL': 'teacher-api-model',
        'TEACHER_BASE_URL': 'https://teacher-api.example/api/paas/v4',
        'REVALIDATE_STALE': '0',
    }
    config.update({key: env.get(key, default) for key, default in defaults.items()})
    for key in ('CKPT_DIR', 'SFT_DIR', 'EVAL_OUTPUT_DIR', 'TRAIN_CANDIDATES_FILE',
                'DEV_CANDIDATES_FILE'):
        config[key] = str(Path(env[key]).resolve()) if env.get(key) else None
    run_id = canonical_hash(config)[:20]
    state = Path('.state/experiments').resolve() / run_id
    state.mkdir(parents=True, exist_ok=True)
    config_path = state / 'configuration.json'
    encoded = json.dumps(config, sort_keys=True, indent=2)
    if config_path.exists() and config_path.read_text(encoding='utf-8') != encoded:
        raise ValueError('Run fingerprint collision; refusing to reuse state')
    config_path.write_text(encoded, encoding='utf-8')
    values = [run_id, str(state), str(exp),
              config['CKPT_DIR'] or str(Path(f'checkpoints/sft-{run_id}').resolve()),
              config['SFT_DIR'] or str(exp / 'sft' / run_id),
              config['EVAL_OUTPUT_DIR'] or str(exp / 'results' / run_id),
              config['TRAIN_CANDIDATES_FILE'] or str(exp / 'candidates/train.jsonl'),
              config['DEV_CANDIDATES_FILE'] or str(exp / 'candidates/dev.jsonl')]
    if any('\n' in value or '\t' in value for value in values):
        raise ValueError('Paths must not contain tabs or newlines')
    print('\t'.join(values))


def validate_split(directory, pool, seed=None):
    directory = Path(directory)
    frozen = [json.loads(line) for line in (directory / 'tasks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    manifest = json.loads((directory / 'split-manifest.json').read_text(encoding='utf-8'))
    if canonical_hash(frozen) != manifest['tasks_sha256']:
        raise ValueError('Frozen tasks differ from split-manifest.json; use a new EXP_DIR')
    if seed is not None and (manifest.get('seed') != seed
                             or manifest.get('dev_ratio') != 0.15
                             or manifest.get('test_ratio') != 0.15):
        raise ValueError('Frozen split configuration differs; use a new EXP_DIR')
    audit_path = directory / 'audit-manifest.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if audit.get('tasks_sha256') != manifest['tasks_sha256'] or any(
            audit.get(key) != len(frozen) for key in ('reference_passes', 'bugs_reproduced')
        ):
            raise ValueError('Frozen audit does not cover all tasks; use a new EXP_DIR')
    if pool:
        source = [json.loads(line) for line in Path(pool).read_text(encoding='utf-8').splitlines() if line.strip()]
        def without_split(records):
            return sorted(({key: value for key, value in record.items() if key != 'split'}
                           for record in records), key=lambda item: item['task_id'])
        if without_split(source) != without_split(frozen):
            raise ValueError('Task pool changed since freeze; use a new EXP_DIR')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    for name in ('check', 'mark'):
        child = sub.add_parser(name)
        child.add_argument('--receipt', required=True)
        child.add_argument('--inputs', nargs='+', required=True)
        child.add_argument('--outputs', nargs='+', required=True)
    child = sub.add_parser('validate-split')
    child.add_argument('--directory', required=True)
    child.add_argument('--pool')
    child.add_argument('--seed', type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            initialize()
        elif args.command == 'validate-split':
            validate_split(args.directory, args.pool, args.seed)
        elif args.command == 'mark':
            mark_receipt(args.receipt, args.inputs, args.outputs)
        else:
            valid, reason = check_receipt(args.receipt, args.inputs, args.outputs)
            print(reason)
            return 0 if valid else 3
    except (ValueError, KeyError, OSError) as error:
        print(f'Run-state error: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
