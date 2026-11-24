"""Exercise the queued review workflow on MIDV mock data and a disposable Label Studio project."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from secrets import token_hex, token_urlsafe
from xml.etree import ElementTree

from labeling.smoke import IMAGE, Studio, container_command


def label_studio_export(media_root, storage_path, tasks, config_path, labels_by_event, runtime):
    """Import actual tasks and generate a raw export using a disposable fixture reviewer."""
    name = f'ekyc-hitl-fixture-{token_hex(6)}'
    config = config_path.read_text()
    xml = ElementTree.fromstring(config)
    choice_groups = {
        frozenset(choice.attrib['value'] for choice in group.findall('Choice')): group.attrib
        for group in xml.iter('Choices')
    }
    document_control = next(control for values, control in choice_groups.items()
                            if {'ID_Card', 'Passport'} <= values)
    quality_control = next(control for values, control in choice_groups.items()
                           if {'Clear', 'Blur', 'Glare', 'Occluded'} <= values)
    media_mount = '/label-studio/hitl'
    try:
        container_command(
            runtime, 'run', '--detach', '--name', name, '--publish', '127.0.0.1::8080',
            '--volume', f'{media_root}:{media_mount}:ro,Z',
            '--env', 'LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true',
            '--env', f'LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={media_mount}', IMAGE,
        )
        port = container_command(runtime, 'port', name, '8080/tcp').stdout.strip()
        if not re.fullmatch(r'127\.0\.0\.1:\d+', port):
            raise RuntimeError('Disposable Label Studio must bind to loopback')
        studio = Studio(f'http://{port}')
        studio.healthy()
        studio.request('/user/signup/')
        studio.request('/user/signup/', {
            'email': 'hitl-fixture@mlops-ekyc.invalid', 'password': token_urlsafe(32),
        }, form=True)
        project_status, project_response = studio.request('/api/projects/', {
            'title': 'Disposable eKYC queued review fixture',
            'label_config': config,
        })
        project = json.loads(project_response)['id']
        storage_status, _ = studio.request('/api/storages/localfiles/', {
            'project': project, 'title': 'Queued mock images',
            'path': f'{media_mount}/{storage_path.as_posix()}', 'use_blob_urls': False,
        })
        import_status, response = studio.request(
            f'/api/projects/{project}/import?return_task_ids=true', tasks)
        imported = json.loads(response)
        assert project_status == storage_status == import_status == 201
        assert imported['task_count'] == len(tasks)
        assert len(set(imported['task_ids'])) == len(tasks)

        observed = set()
        for task_id in imported['task_ids']:
            _, raw_task = studio.request(f'/api/tasks/{task_id}/')
            task = json.loads(raw_task)
            data = task['data']
            status, image = studio.request(data['image'])
            event = hashlib.sha256(image).hexdigest()
            assert status == 200 and event in labels_by_event and event not in observed
            assert not task['annotations']
            observed.add(event)
            label, quality = labels_by_event[event]

            def choice(control, value):
                return {'from_name': control['name'], 'to_name': control['toName'],
                        'type': 'choices', 'value': {'choices': [value]}}

            choices = [choice(document_control, label)]
            if quality is not None:
                choices.append(choice(quality_control, quality))
            status, _ = studio.request(f'/api/tasks/{task_id}/annotations/', {'result': choices})
            assert status == 201
        assert observed == set(labels_by_event)
        status, export = studio.request(f'/api/projects/{project}/export?exportType=JSON')
        assert status == 200
        print(f'Disposable Label Studio project {project}: imported {len(tasks)} queued tasks, '
              f'verified {len(observed)} images, wrote fixture annotations, exported raw JSON.',
              flush=True)
        return export, project
    finally:
        if container_command(runtime, 'inspect', name, check=False).returncode == 0:
            container_command(runtime, 'rm', '--force', '--volumes', name)
            print('Removed disposable Label Studio project container and volume.', flush=True)


def smoke(dataset, output, runtime):
    import os
    import subprocess
    import sys
    from urllib.request import Request, urlopen

    from PIL import Image

    from checks.release_smoke import ROOT, command, http, local_port, read_rows, service

    output.mkdir()
    dataset = dataset.resolve(strict=True)
    manifest = dataset / 'manifest.csv'
    rows = read_rows(manifest)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               MLFLOW_ALLOW_FILE_STORE='true', MLFLOW_DISABLE_AGENT_HINT='1')
    for key in ('EKYC_SCORING_KEY', 'EKYC_SCORING_DEPLOYMENT'):
        env.pop(key, None)
    executable = sys.executable
    source_bundle = output / 'initial-model'

    def train(input_manifest, model_dir, experiment):
        command(executable, '-m', 'modeling.train', '--dataset-root', str(dataset),
                '--manifest', str(input_manifest), '--output-dir', str(model_dir),
                '--device', 'cpu', '--epochs', '1', '--batch-size', '4', '--image-size', '64',
                '--tracking-uri', (output/'mlruns').as_uri(), '--experiment', experiment, env=env)

    def hitl(action, *options):
        command(executable, '-m', 'labeling.hitl', action, *map(str, options), env=env)

    train(manifest, source_bundle, 'initial')
    queue = output / 'queue'
    model_port, demo_port = local_port(), local_port()
    while demo_port == model_port:
        demo_port = local_port()
    scorer = f'http://127.0.0.1:{model_port}'
    demo = f'http://127.0.0.1:{demo_port}'
    env.update(EKYC_SCORING_URL=scorer+'/score', GUNICORN_CMD_ARGS=f'--bind 127.0.0.1:{model_port}')
    launcher = str(Path(executable).with_name('azmlinfsrv'))
    scorer_args = [launcher, '--entry_script', 'score.py', '--model_dir', str(source_bundle),
                   '--port', str(model_port)]
    demo_args = [executable, '-m', 'demo.server', '--port', str(demo_port), '--review-queue',
                 str(queue), '--review-manifest', str(manifest), '--review-dataset-root', str(dataset)]
    train_rows = [r for r in rows if r['split'] == 'train']
    selected = [next(r for r in train_rows if r['label'] == label) for label in ('Passport', 'ID_Card')]
    labels_by_event, queued_predictions = {}, {}
    report_body = None
    excluded_input = BytesIO()
    Image.new('RGB', (8, 8), color='lightblue').save(excluded_input, format='PNG')
    with service(scorer_args, env, output/'scorer.log', scorer, ROOT/'modeling'):
        with service(demo_args, env, output/'demo.log', demo):
            assert b'Report incorrect prediction' in http(demo)[1]
            assert http(demo+'/predict', b'unreadable', 'image/jpeg')[0] == 400
            assert http(demo+'/report', b'{"token":"invalid"}', 'application/json')[0] in (400, 403)
            unknown = Request(demo+'/predict', excluded_input.getvalue(), {'Content-Type': 'image/png'})
            with urlopen(unknown, timeout=15) as response:
                assert response.status == 200
                assert response.headers.get('X-EKYC-Review-Status') == 'not_allowlisted'
                assert response.headers.get('X-EKYC-Report-Token') is None
            for index, row in enumerate(selected):
                image = (dataset / row['image_path']).read_bytes()
                digest = hashlib.sha256(image).hexdigest()
                request = Request(demo+'/predict', image, {'Content-Type': 'image/jpeg'})
                with urlopen(request, timeout=15) as response:
                    assert response.status == 200
                    prediction = json.load(response)
                    token = response.headers.get('X-EKYC-Report-Token')
                assert prediction['needs_review'] and token
                queued_predictions[digest] = prediction
                labels_by_event[digest] = (row['label'], 'Clear') if index == 0 else ('Outside_Scope', None)
                if index == 0:
                    report_body = json.dumps({'token': token}).encode()
                    for _ in range(2):
                        status, _ = http(demo+'/report', report_body, 'application/json')
                        assert 200 <= status < 300
                    assert http(demo+'/predict', image, 'image/jpeg')[0] == 200

        # A new demo process must preserve the durable queue and invalidate old receipts.
        with service(demo_args, env, output/'demo-restarted.log', demo):
            assert http(demo+'/report', report_body, 'application/json')[0] == 403
            image = (dataset / selected[0]['image_path']).read_bytes()
            assert http(demo+'/predict', image, 'image/jpeg')[0] == 200

    events = {path.name: json.loads((path/'event.json').read_text())
              for path in (queue/'events').iterdir() if path.is_dir() and not path.name.startswith('.')}
    assert set(events) == set(labels_by_event)
    for event, record in events.items():
        assert record['event_id'] == record['image_sha256'] == event
        assert record['prediction'] == queued_predictions[event]
        assert hashlib.sha256((queue/'events'/event/record['image_file']).read_bytes()).hexdigest() == event
        expected = {'needs_review', 'reported_incorrect'} if labels_by_event[event][1] else {'needs_review'}
        assert set(record['reasons']) == expected
    batch = output / 'batch'
    hitl('batch', '--queue-root', queue, '--base-manifest', manifest,
         '--dataset-root', dataset, '--output', batch)
    tasks = json.loads((batch/'tasks.json').read_text())
    assert len(tasks) == len(events)
    raw_export, project = label_studio_export(
        batch, Path('images'), tasks, ROOT/'labeling'/'hitl.xml', labels_by_event, runtime)
    export = output / 'fixture-export.json'
    export.write_bytes(raw_export)
    approval = output / 'fixture-approval.json'
    versions = output / 'versions'

    def publish(export_path):
        hitl('publish', '--batch', batch, '--review-export', export_path, '--approval', approval,
             '--base-manifest', manifest, '--dataset-root', dataset, '--versions-root', versions)

    try:
        publish(export)
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError('Publishing without explicit approval unexpectedly succeeded')
    assert not versions.exists() or not list(versions.iterdir())
    hitl('approve', '--batch', batch, '--review-export', export,
         '--reviewer', 'fixture:orchestrator', '--output', approval)
    tampered = output / 'tampered-export.json'
    tampered.write_bytes(raw_export + b'\n ')
    try:
        publish(tampered)
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError('Publishing changed export unexpectedly succeeded')
    publish(export)
    publish(export)
    version, = (path for path in versions.iterdir() if path.is_dir())
    published = version/'manifest.csv'
    lineage = json.loads((version/'lineage.json').read_text())
    by_id = {row['image_id']: row for row in read_rows(published)}
    assert selected[1]['image_id'] not in by_id
    reviewed = by_id[selected[0]['image_id']]
    assert reviewed['label'] == selected[0]['label'] and reviewed['quality_status'] == 'Clear'
    for row in rows:
        if row['split'] in ('test', 'validation'):
            assert all(by_id[row['image_id']][key] == value for key, value in row.items())
    from mlflow import MlflowClient

    # The approval/publish steps do not start training. Submit the trainer explicitly.
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    client = MlflowClient(tracking_uri=(output/'mlruns').as_uri())
    assert client.get_experiment_by_name('approved-fixture-manual') is None
    train(published, output/'reviewed-model', 'approved-fixture-manual')
    experiment = client.get_experiment_by_name('approved-fixture-manual')
    run, = client.search_runs([experiment.experiment_id])
    assert run.info.status == 'FINISHED'
    assert run.data.metrics['train_quality_samples'] == 1
    assert run.data.metrics['test_quality_samples'] == 0
    summary = {
        'scope': 'local technical workflow using approved mock-input allowlist and fixture reviewer',
        'queue_records': list(events.values()),
        'disposable_label_studio_project': project,
        'unallowlisted_valid_image': 'scored without queue record or report receipt',
        'restart': 'durable event reused and prior report receipt rejected',
        'reviewer': 'fixture:orchestrator',
        'export_sha256': hashlib.sha256(raw_export).hexdigest(),
        'publication_without_approval': 'rejected',
        'changed_export_after_approval': 'rejected',
        'version_id': version.name,
        'version_manifest_sha256': hashlib.sha256(published.read_bytes()).hexdigest(),
        'lineage': lineage,
        'manual_training': {'run_id': run.info.run_id, 'metrics': run.data.metrics},
    }
    result = json.dumps(summary, indent=2)+'\n'
    (output/'summary.json').write_text(result)
    print(result, end='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--runtime', default='podman', choices=('podman', 'docker'))
    options = parser.parse_args()
    smoke(options.dataset_root.resolve(), options.output.resolve(), options.runtime)


if __name__ == '__main__':
    main()
