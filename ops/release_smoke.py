"""Run the archive-to-demo integration against the local Azure ML runtime."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def command(*args, env, cwd=ROOT):
    print('+', *args, flush=True)
    subprocess.run(args, check=True, cwd=cwd, env=env)


def local_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def http(url, data=None, kind=None):
    headers = {'Content-Type': kind} if kind else {}
    try:
        with urlopen(Request(url, data=data, headers=headers), timeout=15) as response:
            return response.status, response.read()
    except HTTPError as error:
        with error:
            return error.code, error.read()


@contextmanager
def service(args, env, log, health_url, cwd=ROOT):
    print('+', *args, flush=True)
    with log.open('w') as output:
        process = subprocess.Popen(args, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f'Service exited early. See {log}')
                try:
                    if http(health_url)[0] == 200:
                        break
                except URLError:
                    pass
                time.sleep(.25)
            else:
                raise RuntimeError(f'Service did not start. See {log}')
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def check_dataset(root, rows):
    splits, frames = defaultdict(set), defaultdict(list)
    for row in rows:
        image = root / row['image_path']
        assert row['template_id'] == image.relative_to(root).parts[1]
        assert image.is_file() and not row['quality_status']
        assert 'rus_internalpassport' not in row['image_path']
        splits[row['template_id']].add(row['split'])
        frames[row['document_id']].append(int(image.stem))
    assert len({row['image_id'] for row in rows}) == len(rows) == 54
    assert len(splits) == 9 and all(len(value) == 1 for value in splits.values())
    assert len(frames) == 18
    for indices in frames.values():
        ordered = sorted(indices)
        assert len(ordered) == 3 and min(b-a for a, b in zip(ordered, ordered[1:])) >= 10
    for split in ('train', 'validation', 'test'):
        assert {r['label'] for r in rows if r['split'] == split} == {'ID_Card', 'Passport'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    out = options.output.resolve()
    out.mkdir()
    source = options.source.resolve(strict=True)
    initial = source.stat()
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               MLFLOW_ALLOW_FILE_STORE='true', MLFLOW_DISABLE_AGENT_HINT='1')
    for key in ('EKYC_SCORING_KEY', 'EKYC_SCORING_DEPLOYMENT'):
        env.pop(key, None)

    py = sys.executable
    dataset = out / 'dataset'
    manifest = dataset / 'manifest.csv'
    command(py, '-m', 'data_pipeline.midv2020', '--source', str(source), '--output',
            str(dataset), '--max-clips-per-template', '2', env=env)
    rows = read_rows(manifest)
    check_dataset(dataset, rows)
    tasks_path = out / 'tasks.json'
    command(py, '-m', 'labeling.label_studio', 'make-tasks', '--manifest', str(manifest),
            '--dataset-root', str(dataset), '--media-root', str(out), '--output',
            str(tasks_path), env=env)
    tasks = json.loads(tasks_path.read_text())
    assert {task['data']['image_id'] for task in tasks} == {row['image_id'] for row in rows}
    # A fabricated review verifies ingestion only. Training always uses the original manifest.
    tasks[0]['annotations'] = [{'result': [{'from_name': 'quality', 'to_name': 'image',
        'type': 'choices', 'value': {'choices': ['Blur']}}]}]
    export = out / 'fixture.json'
    export.write_text(json.dumps(tasks))
    reviewed = out / 'fixture-reviewed.csv'
    command(py, '-m', 'labeling.label_studio', 'apply-reviews', '--manifest', str(manifest),
            '--review-export', str(export), '--output', str(reviewed), env=env)
    assert Counter(row['quality_status'] for row in read_rows(reviewed)) == {'': 53, 'Blur': 1}

    stage = out / 'stage'
    command(py, 'azure/stage.py', str(stage), env=env)
    bundle = out / 'bundle'
    tracking = out / 'mlruns'
    command(py, '-m', 'modeling.train', '--dataset-root', str(dataset), '--manifest',
            str(manifest), '--output-dir', str(bundle), '--epochs', '1', '--batch-size',
            '4', '--device', 'cpu', '--image-size', '64', '--tracking-uri', tracking.as_uri(),
            '--experiment', 'release-smoke', cwd=stage/'src', env=env)
    assert {f.name for f in bundle.iterdir()} == {'metadata.json', 'weights.pt'}
    from mlflow import MlflowClient
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    client = MlflowClient(tracking_uri=tracking.as_uri())
    experiment = client.get_experiment_by_name('release-smoke')
    run, = client.search_runs([experiment.experiment_id])
    assert run.info.status == 'FINISHED'
    assert run.data.metrics['test_quality_samples'] == 0
    assert run.data.metrics['test_samples'] == 12

    model_port, demo_port = local_port(), local_port()
    while demo_port == model_port:
        demo_port = local_port()
    endpoint = f'http://127.0.0.1:{model_port}'
    demo = f'http://127.0.0.1:{demo_port}'
    env.update(EKYC_SCORING_URL=endpoint+'/score', GUNICORN_CMD_ARGS=f'--bind 127.0.0.1:{model_port}')
    server = str(Path(py).with_name('azmlinfsrv'))
    scorer_args = [server, '--entry_script', 'score.py', '--model_dir', str(bundle),
                   '--port', str(model_port)]
    scoring_code = stage / 'src' / 'modeling'
    held_out = next(r for r in rows if r['split'] == 'test')
    image = (dataset / held_out['image_path']).read_bytes()

    def predict(_=None):
        status, content = http(demo+'/predict', image, 'image/jpeg')
        return status, json.loads(content)

    with service([py, '-m', 'demo.server', '--port', str(demo_port)], env, out/'demo.log', demo):
        with service(scorer_args, env, out/'scorer.log', endpoint, scoring_code):
            assert http(endpoint+'/score', b'{"image_base64":"!"}', 'application/json')[0] == 400
            assert http(demo+'/predict', b'bad', 'text/plain')[0] == 415
            with ThreadPoolExecutor(max_workers=4) as pool:
                predictions = list(pool.map(predict, range(6)))
            status, answer = predictions[0]
            assert all(result == (status, answer) for result in predictions)
            assert status == 200 and set(answer) == {'label', 'status', 'confidence', 'needs_review'}
            assert answer['label'] in {'ID_Card', 'Passport'} and 0 <= answer['confidence'] <= 1
            assert answer['status'] is None and answer['needs_review'] is True
        unavailable, _ = predict()
        assert unavailable == 502
        with service(scorer_args, env, out/'scorer-restarted.log', endpoint, scoring_code):
            assert predict() == (status, answer)
    final = source.stat()
    assert (initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns) == (
        final.st_size, final.st_mtime_ns, final.st_ctime_ns)
    summary = {
        'scope': 'local Azure ML inference runtime and disposable review fixture',
        'archive_bytes': final.st_size,
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'splits': dict(Counter(row['split'] for row in rows)),
        'review_fixture': dict(Counter(row['quality_status'] for row in read_rows(reviewed))),
        'mlflow_run_id': run.info.run_id,
        'mlflow_metrics': run.data.metrics,
        'repeat_and_concurrent_requests': len(predictions),
        'upstream_down_status': unavailable,
        'successful_response': answer,
        'scorer_restart': 'same prediction after restart',
    }
    raw = json.dumps(summary, indent=2) + '\n'
    (out/'summary.json').write_text(raw)
    print(raw, end='')


if __name__ == '__main__':
    main()
