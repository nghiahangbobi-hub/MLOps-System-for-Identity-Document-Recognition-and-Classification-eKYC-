"""Render the course drafts and reconcile tables and chart with local-results.json."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from xml.etree import ElementTree
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    out = options.output.resolve()
    out.mkdir()
    report = DOCS / 'bao-cao-ekyc.docx'
    deck = DOCS / 'trinh-bay-ekyc.pptx'
    evidence = json.loads((DOCS / 'local-results.json').read_text())
    runs = evidence['comparison']['variants']
    chosen = min(runs, key=lambda run: min(run['validation_loss']))

    cloud_path = DOCS / 'azure-evidence' / 'staging-benchmark.json'
    uit = DOCS / 'bao-cao-ekyc-uit.docx'
    office_files = (report, deck, uit) if cloud_path.is_file() else (report, deck)
    for item in office_files:
        with ZipFile(item) as artifact:
            assert artifact.testzip() is None
    subprocess.run(['libreoffice', f'-env:UserInstallation={(out/"profile").as_uri()}',
                    '--headless', '--convert-to', 'pdf', '--outdir', str(out),
                    *(str(item) for item in office_files)], check=True)

    extracted = {}
    for item in (report, deck):
        pdf = out / f'{item.stem}.pdf'
        body = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-'], text=True)
        (out / f'{item.stem}.txt').write_text(body)
        pages = [page for page in body.split('\f') if page.strip()]
        extracted[item.suffix] = (body, len(pages))
    report_text, report_pages = extracted['.docx']
    deck_text, slide_count = extracted['.pptx']
    latest = DOCS / 'model-quality-evidence' / 'held-out.json'
    has_appendix = latest.is_file()
    assert report_pages == (4 if has_appendix else 3)
    thesis_evidence = DOCS / 'thesis-evidence' / 'hitl-flow.png'
    has_thesis = thesis_evidence.is_file()
    assert slide_count == (14 if has_thesis else 9 if has_appendix else 7)
    word_ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with ZipFile(report) as artifact:
        document = ElementTree.fromstring(artifact.read('word/document.xml'))
    tables = document.findall('.//w:tbl', word_ns)
    assert len(tables) == (3 if has_appendix else 2)
    def table_rows(table):
        return [
            [''.join(node.text or '' for node in cell.findall('.//w:t', word_ns))
             for cell in row.findall('w:tc', word_ns)]
            for row in table.findall('w:tr', word_ns)
        ]

    def latency(value):
        return f'{value:.2f}'.replace('.', ',') + ' ms'

    filesystem = evidence['filesystem']['results']
    benchmark = evidence['scoring']['results']['scoring']
    assert table_rows(tables[0])[1:] == [
        [label, f"{result['successes']}/{result['attempts']}",
         latency(result['p50_ms']), latency(result['p95_ms'])]
        for label, result in [
            ('Đọc tệp 16 MiB', filesystem['read']),
            ('Ghi tệp 16 MiB', filesystem['write']),
            ('API qua loopback', benchmark),
        ]
    ]
    assert table_rows(tables[1])[1:] == [
        [f"{'ImageNet' if run['pretrained'] else 'Ngẫu nhiên'} / {run['rate']}",
         f"{min(run['validation_loss']):.4f}", f"{run['selected_validation_accuracy']:.1%}",
         f"{run['test_accuracy']:.1%}"]
        for run in runs
    ]
    for body in (report_text, deck_text):
        assert f"{evidence['comparison']['samples']} ảnh" in body
        assert f"{chosen['test_accuracy']:.1%}" in body
        assert f"{chosen['rate']}" in body
        assert 'Azure' in body
        assert 'duyệt' in body
        assert f"{benchmark['successes']}/{benchmark['attempts']}" in body
        assert f"{benchmark['p50_ms']:.2f}".replace('.', ',') in body
        assert f"{benchmark['p95_ms']:.2f}".replace('.', ',') in body
        for operation in ('read', 'write'):
            entry = evidence['filesystem']['results'][operation]
            assert f"{entry['p50_ms']:.2f}".replace('.', ',') in body
    if cloud_path.is_file():
        cloud = json.loads(cloud_path.read_text())['results']['scoring']
        events, = json.loads((cloud_path.parent / 'low-confidence-rate.json').read_text())
        browser = json.loads((cloud_path.parent / 'browser-managed.json').read_text())
        uit_text = subprocess.check_output(['pdftotext', '-layout', str(out / 'bao-cao-ekyc-uit.pdf'), '-'], text=True)
        final_pdf = subprocess.check_output(['pdftotext', '-layout', str(uit.with_suffix('.pdf')), '-'], text=True)
        assert final_pdf == uit_text
        uit_pages = [page for page in uit_text.split('\f') if page.strip()]
        has_evidence_appendix = "PHỤ LỤC B. HỒ SƠ MINH CHỨNG" in uit_text
        has_kyc_workflow = "Hình 1.1. Vai trò của cổng phân loại" in uit_text
        has_capture_evaluation = "Bảng 5.8. Hiệu năng mô hình đã khóa" in uit_text
        has_multisource_pilot = "Kết quả fine-tune đa nguồn" in uit_text
        assert len(uit_pages) == (32 if has_multisource_pilot else 29 if has_capture_evaluation else 28 if has_kyc_workflow else 26 if has_evidence_appendix else 22 if has_thesis else 9)
        if has_thesis:
            contents = uit_pages[3]
            entries = re.findall(r'^((?:TÓM TẮT|ABSTRACT|CHƯƠNG|TÀI LIỆU THAM KHẢO|PHỤ LỤC|Hình|Bảng)[^\n]*?)\.{3,}\s*(\d+)\s*$', contents, re.M)
            assert len(entries) == (25 if has_multisource_pilot else 24 if has_capture_evaluation or has_evidence_appendix else 23)
            for label, page in entries:
                target = re.sub(r'\s+', ' ', uit_pages[int(page)])
                assert re.sub(r'\s+', ' ', label).strip() in target
            with ZipFile(uit) as artifact:
                media = [name for name in artifact.namelist() if name.startswith('word/media/')]
                assert len(media) == (12 if has_multisource_pilot else 11 if has_kyc_workflow else 10 if has_evidence_appendix else 6)
            if has_kyc_workflow:
                for marker in ("Bảng 3.2.", "Bảng 5.3.", "Bảng 5.4.", "Bảng 5.6.", "1.178,48", "3.270,64", "0/600", "98/100"):
                    assert marker in uit_text
            if has_capture_evaluation:
                data = json.loads((DOCS / 'thesis-evidence' / 'capture-evaluation' / 'results.json').read_text())
                for result in data['results'].values():
                    metrics = result['metrics']
                    assert f"{metrics['correct']}/{metrics['samples']}" in uit_text
                    assert f"{metrics['accuracy']:.2%}".replace('.', ',') in uit_text
            if has_multisource_pilot:
                pilot_path = DOCS / 'thesis-evidence' / 'multisource' / 'result.json'
                pilot = json.loads(pilot_path.read_text())
                assert len(pilot['candidates']) == 4 and pilot['selected'] is None
                for row in [pilot['baseline'], *pilot['candidates']]:
                    assert f"{row['mean_domain_macro_f1']:.2%}".replace('.', ',') in uit_text
                assert "Hình 5.6." in uit_text and "Bảng 5.9." in uit_text
            if has_evidence_appendix:
                for marker in ('Hình B.1.', 'Hình B.2.', 'Hình B.3.', 'Hình B.4.', 'Bảng B.1.', 'Bảng B.2.'):
                    assert uit_text.count(marker) >= 1
            assert 'Demo cục bộ qua trình duyệt' in deck_text and 'Duyệt nhãn và phê duyệt bản xuất' in deck_text
        p95 = f"{cloud['p95_ms']:,.2f}".translate(str.maketrans(',.', '.,'))
        ratio = f"{events['LowConfidenceEvents']}/{events['SuccessfulPredictions']}"
        successful = f"{cloud['successes']}/{cloud['attempts']}"
        assert sum(sample > 3000 for sample in cloud['successful_attempt_ms']) == 2
        assert browser['predict_http'] == 200
        assert not json.loads((cloud_path.parent / 'app-traces-verified.json').read_text())
        for body in (report_text, deck_text, uit_text):
            assert all(value in body for value in (p95, ratio, successful, 'HTTP 200'))
    else:
        assert 'trước khi có kết quả Azure' in report_text
        assert 'bằng chứng cần bổ sung' in deck_text

    if has_appendix:
        held = json.loads(latest.read_text())
        metrics = held['document']
        policy = held['document_review_policy']
        accuracy = f"{metrics['accuracy']:.2%}"
        f1 = f"{metrics['macro_f1']:.2%}"
        recalls = [f"{value:.2%}" for value in metrics['per_class_recall'].values()]
        ece = f"{metrics['calibration']['ece']:.4f}"
        retained = f"{policy['coverage']:.2%}"
        gate = 'ĐẠT' if held['acceptance'] == 'pass' else 'CHƯA ĐẠT'
        count = metrics['samples']
        correct = sum(row[index] for index, row in enumerate(
            metrics['confusion_true_rows_predicted_columns']))
        assert table_rows(tables[2])[1:] == [
            ['Độ chính xác và cổng 85%', f'{accuracy} ({correct}/{count}) · {gate}'],
            ['Macro-F1', f1],
            ['Recall ID_Card / Passport', ' / '.join(recalls)],
            ['ECE / độ bao phủ theo ngưỡng tài liệu', f'{ece} / {retained}'],
        ]
        workflow = json.loads((DOCS / 'hitl-evidence' / 'integration.json').read_text())
        for body in (report_text, deck_text):
            assert accuracy in body and f1 in body and retained in body and ece in body
            assert f'{correct}/{count}' in body and gate in body
            assert workflow['version_id'][:12] in body and 'Outside_Scope' in body
    namespace = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
    with ZipFile(deck) as artifact:
        charts = [name for name in artifact.namelist() if re.fullmatch(r'ppt/charts/chart\d+\.xml', name)]
        assert len(charts) == 1
        chart = ElementTree.fromstring(artifact.read(charts[0]))
    series = chart.findall('.//c:ser', namespace)
    assert len(series) == len(runs)
    for line, run in zip(series, runs):
        assert [point.text for point in line.findall('c:cat//c:pt/c:v', namespace)] == [
            'Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4']
        label, = line.findall('c:tx//c:pt/c:v', namespace)
        assert label.text == f"{'ImageNet' if run['pretrained'] else 'Ngẫu nhiên'} / {run['rate']}"
        actual = [float(point.text) for point in line.findall('c:val//c:pt/c:v', namespace)]
        expected = run['validation_loss']
        assert len(actual) == len(expected)
        assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(actual, expected))
    print(json.dumps({
        'report_pages': report_pages, 'slides': slide_count, 'chart_series': len(series),
        'chart_points_checked': sum(len(run['validation_loss']) for run in runs),
        'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
        'deck_sha256': hashlib.sha256(deck.read_bytes()).hexdigest(),
        'hitl_and_current_model_appendix': has_appendix,
        'managed_staging_evidence': cloud_path.is_file(),
        'thesis_pages': len(uit_pages) if cloud_path.is_file() else None,
        'thesis_contents_entries': len(entries) if cloud_path.is_file() and has_thesis else None,
        'checked': ['train comparison', 'scoring request count and p50/p95',
                    'storage p50/p95 in Word and p50 in PowerPoint', 'human review status'],
    }, indent=2))


if __name__ == '__main__':
    main()
