import contextlib
import importlib.util
import http.server
import threading
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('bf16_cli_reference', Path(__file__).with_name('bf16_cli_reference.py'))
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)
UUIDS = ['GPU-735d11ff-3eb7-f0b3-0e5b-4cabe70cdc6a', 'GPU-2e0136cd-44e3-52c1-fd62-81a0b225286a']


def plan(root):
    return {'schemaVersion': 1, 'binary': {'path': str(root / 'mistralrs'), 'sha256': '0' * 64}, 'buildIdentityPath': str(root / 'build.json'), 'source': {'root': str(root), 'baseRevision': 'a' * 40, 'sourcePatchSha256': 'b' * 64}, 'model': {'root': str(root / 'model'), 'manifestSha256': 'c' * 64, 'repository': reference.MODEL_REPOSITORY, 'revision': reference.MODEL_REVISION}, 'request': {'path': str(root / 'request.json'), 'sha256': 'd' * 64}, 'gpuUuids': UUIDS, 'port': 19081, 'outputDirectory': str(root / 'result'), 'readinessSeconds': 2, 'requestSeconds': 2, 'totalSeconds': 10, 'outputTokens': [24]}


class ReferenceTests(unittest.TestCase):
    def test_cli_parent_options_and_pristine_bf16_mapping(self):
        argv = reference.cli_argv('/candidate/mistralrs', '/pinned/model', 19081)
        self.assertLess(argv.index('--host'), argv.index('multimodal'))
        self.assertLess(argv.index('--max-seqs'), argv.index('multimodal'))
        self.assertEqual(argv[argv.index('--dtype') + 1], 'bf16')
        self.assertEqual(argv[argv.index('--device-layers') + 1], '0:32;1:32')
        self.assertNotIn('--quant', argv)
        self.assertNotIn('--isq', argv)

    def test_preserves_serving_text_and_thinking_without_mutating_original(self):
        original = {'messages': [{'role': 'system', 'content': [{'kind': 'text', 'text': 'first'}, {'kind': 'text', 'text': 'second'}]}], 'thinking': False, 'jsonMode': False, 'maxOutputTokens': 24}
        before = reference.canonical(original)
        payload = reference.normalize_request(original, 'local', 48)
        self.assertEqual(payload['messages'], [{'role': 'system', 'content': 'first\nsecond'}])
        self.assertFalse(payload['enable_thinking'])
        self.assertEqual(payload['max_tokens'], 48)
        self.assertNotIn('response_format', payload)
        self.assertEqual(reference.canonical(original), before)

    def test_plan_rejects_overflow_duplicate_gpu_and_nonlocal_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = plan(root)
            reference.validate_plan(value)
            for key, bad in [('totalSeconds', 901), ('requestSeconds', 301), ('readinessSeconds', 601), ('gpuUuids', [UUIDS[0], UUIDS[0]]), ('outputDirectory', '/workspace/synemantic-state/result')]:
                invalid = dict(value, **{key: bad})
                with self.assertRaises(reference.Refusal):
                    reference.validate_plan(invalid)

    def test_output_traversal_symlink_and_publication_identity_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = plan(root)
            for output in ['/tmp/../workspace/live', str(root / 'nested/../escape')]:
                with self.assertRaisesRegex(reference.Refusal, 'traversal'):
                    reference.validate_plan(dict(value, outputDirectory=output))
            (root / 'link').symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(reference.Refusal, 'symlink'):
                reference.validate_plan(dict(value, outputDirectory=str(root / 'link/result')))
            output = root / 'result'
            with reference.fresh_output(output) as (directory, identity):
                output.rename(root / 'moved')
                output.mkdir()
                with self.assertRaisesRegex(reference.Refusal, 'identity_changed'):
                    reference.publish(output, {}, directory, identity)
            self.assertFalse((root / 'moved/report.json').exists())

    def test_absolute_request_deadline_stops_trickled_http_body(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Length', '1000')
                self.end_headers()
                try:
                    for _ in range(100):
                        self.wfile.write(b' ')
                        self.wfile.flush()
                        time.sleep(0.01)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(reference.Refusal, 'request_deadline_exceeded'):
                with reference.request_deadline(0.06):
                    reference.http_json(server.server_port, '/', 0.2)
            self.assertLess(time.monotonic() - started, 0.3)
        finally:
            thread.join(timeout=2)
            server.server_close()
        self.assertFalse(thread.is_alive())

    def test_gpu_memory_sample_identity_and_explicit_unavailable(self):
        rows = ''.join(f'{uuid}, RTX 6000 Ada, 580.0, 49140, 120, 49020, 25.5\n' for uuid in UUIDS).encode()
        with mock.patch.object(reference, 'checked_command', return_value=rows):
            sample = reference.gpu_memory_sample(UUIDS, time.monotonic() + 1, 'before_load')
            self.assertEqual(sample['devices'][0]['freeMiB'], 49020)
            self.assertIn('utcUnixNs', sample['boundary'])
        with mock.patch.object(reference, 'checked_command', side_effect=OSError):
            self.assertIn('unavailable', reference.gpu_memory_sample(UUIDS, time.monotonic() + 1, 'before_load'))

    def test_shared_uuid_lease_collision_idle_refusal_and_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            observed = []
            query = lambda uuid: observed.append(uuid) or ''
            with reference.gpu_leases(UUIDS, time.monotonic() + 2, directory, query):
                with self.assertRaisesRegex(reference.Refusal, 'gpu_already_leased'):
                    with reference.gpu_leases(list(reversed(UUIDS)), time.monotonic() + 2, directory, query):
                        self.fail('collision admitted')
            self.assertEqual(observed, sorted(UUIDS))
            for output in ['1234\n', 'N/A\n']:
                with self.assertRaisesRegex(reference.Refusal, 'gpu_not_idle'):
                    with reference.gpu_leases(UUIDS, time.monotonic() + 2, directory, lambda _uuid: output):
                        self.fail('busy or unknown GPU admitted')
            with reference.gpu_leases(UUIDS, time.monotonic() + 2, directory, lambda _uuid: ''):
                pass

    def test_lease_symlink_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / f'{sorted(UUIDS)[0]}.lock').symlink_to(directory / 'elsewhere')
            with self.assertRaises(OSError):
                with reference.gpu_leases(UUIDS, time.monotonic() + 2, directory, lambda _uuid: ''):
                    self.fail('symlink lease admitted')

    def test_readiness_deadline_and_process_group_escalation(self):
        with mock.patch.object(reference, 'http_json', side_effect=ConnectionRefusedError):
            process = mock.Mock()
            process.poll.return_value = None
            started = time.monotonic()
            with self.assertRaisesRegex(reference.Refusal, 'deadline'):
                reference.wait_ready(process, 19081, started + 0.2, 0.02, '/expected/model')
            self.assertLess(time.monotonic() - started, 0.5)
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / 'ready'
            code = "import signal,time,pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(__import__('sys').argv[1]).touch(); time.sleep(30)"
            process = subprocess.Popen([sys.executable, '-c', code, str(ready)], start_new_session=True)
            try:
                deadline = time.monotonic() + 2
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
                reference.stop_group(process, grace=0.02)
                self.assertEqual(process.returncode, -signal.SIGKILL)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)

    def test_foreign_model_and_foreign_listener_refused_before_prompt(self):
        process = mock.Mock()
        process.poll.return_value = None
        with mock.patch.object(reference, 'http_json', return_value=({'data': [{'id': 'default'}, {'id': '/foreign/model', 'status': 'loaded'}]}, b'')) as http_calls:
            with self.assertRaisesRegex(reference.Refusal, 'foreign_or_unexpected'):
                reference.wait_ready(process, 19081, time.monotonic() + 1, 1, '/expected/model')
            self.assertEqual(http_calls.call_count, 1)
            self.assertEqual(http_calls.call_args.args[1], '/v1/models')
        server = http.server.HTTPServer(('127.0.0.1', 0), http.server.BaseHTTPRequestHandler)
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            with self.assertRaisesRegex(reference.Refusal, 'listener_not_owned'):
                reference.http_json(server.server_port, '/v1/chat/completions', 1, {'messages': 'secret'}, process=child)
        finally:
            reference.stop_group(child)
            server.server_close()

    def test_report_retains_usage_without_payload_or_invented_decode_rate(self):
        value = {'usage': {'prompt_tokens': 5, 'completion_tokens': 2}, 'choices': [{'finish_reason': 'length', 'message': {'content': 'sensitive output'}}]}
        result = reference.response_summary(value, reference.canonical(value), 1.25)
        self.assertEqual(result['apiUsage'], value['usage'])
        self.assertEqual(result['httpWallSeconds'], 1.25)
        self.assertIsNone(result['decodeRate'])
        self.assertFalse(result['jsonParseSucceeded'])
        self.assertNotIn(b'sensitive output', reference.canonical(result))
        warning = reference.diagnostic(b'WARN CUDA shape error core/src/model.rs:12 sensitive prompt')
        self.assertIn('shape_mismatch', warning['classes'])
        self.assertNotIn(b'sensitive prompt', reference.canonical(warning))

    def test_technical_cuda_http_details_do_not_retain_payload(self):
        text = b'CUDA error: DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument") secret prompt'
        detail = reference.technical_detail(text)
        self.assertIn('CUDA_ERROR_INVALID_VALUE', detail['technicalNames'])
        self.assertIn('invalid argument', detail['technicalMessages'])
        self.assertNotIn(b'secret prompt', reference.canonical(detail))
        body = reference.canonical({'message': text.decode(), 'partial_response': {'choices': [{'message': {'content': 'secret output CUDA_ERROR_OUT_OF_MEMORY'}}]}})
        result = reference.http_error_detail(500, body)
        self.assertIn('CUDA_ERROR_INVALID_VALUE', result['technicalNames'])
        self.assertNotIn('CUDA_ERROR_OUT_OF_MEMORY', result['technicalNames'])
        self.assertNotIn(b'secret output', reference.canonical(result))
        self.assertNotIn(b'secret prompt', reference.canonical(result))
        self.assertEqual(result['bodySha256'], reference.digest(body))
        self.assertIsNotNone(reference.technical_detail(b'CUDA_ERROR_SECRET_PAYLOAD')['unavailable'])

    def test_http_500_retains_only_error_message_technical_detail(self):
        body = reference.canonical({'message': 'DriverError(CUDA_ERROR_INVALID_CONTEXT, "invalid device context") secret', 'partial_response': {'content': 'private'}})
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send_response(500)
                self.end_headers()
                self.wfile.write(body)
        server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        try:
            with self.assertRaises(reference.HttpFailure) as caught:
                reference.http_json(server.server_port, '/', 1)
            self.assertIn('CUDA_ERROR_INVALID_CONTEXT', caught.exception.detail['technicalNames'])
            self.assertNotIn(b'private', reference.canonical(caught.exception.detail))
        finally:
            thread.join(timeout=2)
            server.server_close()

    def test_operation_contexts_retain_only_fixed_metadata_from_safe_error_fields(self):
        layer = 'BF16_OP_CONTEXT{kind=layer;layer=32;stage=mlp_gate;device=cpu;ordinal=0;dtype=BF16;dims=1,2,3;strides=6,1,2}'
        gemm = 'BF16_OP_CONTEXT{kind=gemm;backend=cublaslt;device=cpu;ordinal=0;dtype=BF16;dims=1,2,3;strides=6,1,2;weight_device=cpu;weight_ordinal=0;weight_dtype=BF16;weight_dims=1,5,3;weight_strides=0,3,1}'
        message = layer + ' ' + gemm + ' CublasError(CUBLAS_STATUS_INTERNAL_ERROR) private text and backtrace'
        detail = reference.http_error_detail(500, reference.canonical({'message': message, 'partial_response': {'content': layer.replace('layer=32', 'layer=7') + ' private output'}}))
        self.assertEqual(len(detail['operationContexts']), 2)
        self.assertEqual(detail['operationContexts'][0], {'kind': 'layer', 'layer': 32, 'stage': 'mlp_gate', 'device': 'cpu', 'ordinal': 0, 'dtype': 'BF16', 'dims': [1, 2, 3], 'strides': [6, 1, 2]})
        self.assertEqual(detail['operationContexts'][1]['weight_strides'], [0, 3, 1])
        self.assertEqual(detail['operationContexts'][1]['weight_device'], 'cpu')
        self.assertEqual(detail['operationContexts'][1]['weight_ordinal'], 0)
        self.assertIn('CUBLAS_STATUS_INTERNAL_ERROR', detail['technicalNames'])
        self.assertNotIn(b'private', reference.canonical(detail))
        self.assertNotIn(b'backtrace', reference.canonical(detail))
        log = reference.diagnostic(('ERROR ' + message).encode())
        self.assertEqual(log['operationContexts'], detail['operationContexts'])
        partial_only = reference.http_error_detail(500, reference.canonical({'message': 'CublasError', 'partial_response': {'content': gemm}}))
        self.assertEqual(partial_only['operationContexts'], [])

    def test_operation_context_parser_rejects_noncanonical_or_unbounded_records(self):
        record = 'BF16_OP_CONTEXT{kind=layer;layer=32;stage=mlp_gate;device=cuda;ordinal=1;dtype=BF16;dims=1,2,3;strides=6,1,2}'
        invalid = [
            record.replace('kind=layer', 'kind=private'),
            record.replace('stage=mlp_gate', 'stage=private'),
            record.replace('device=cuda', 'device=private'),
            record.replace('dtype=BF16', 'dtype=private'),
            record.replace('layer=32', 'layer=4096'),
            record.replace('ordinal=1', 'ordinal=256'),
            record.replace('device=cuda', 'device=cpu'),
            record.replace('dims=1,2,3', 'dims=1,2,3,4,5'),
            record.replace('dims=1,2,3', 'dims=1000000001,2,3'),
            record.replace('dims=1,2,3', 'dims=-1,2,3'),
            record.replace('dims=1,2,3', 'dims=1.0,2,3'),
            record.replace('dims=1,2,3', 'dims=01,2,3'),
            record.replace('strides=6,1,2', 'strides=6,1'),
            record.replace('strides=6,1,2', 'strides=6,1,2;private=secret'),
            record.replace('layer=32', 'layer=32;layer=33'),
            'PREFIX_' + record,
        ]
        for value in invalid:
            with self.subTest(value=value):
                self.assertEqual(reference.operation_contexts(value), [])
        gemm = 'BF16_OP_CONTEXT{kind=gemm;backend=cublas;device=cuda;ordinal=1;dtype=BF16;dims=1,2,3;strides=6,3,1;weight_device=cuda;weight_ordinal=0;weight_dtype=BF16;weight_dims=1,5,3;weight_strides=0,3,1}'
        parsed = reference.operation_contexts(gemm)[0]
        self.assertEqual((parsed['ordinal'], parsed['weight_ordinal']), (1, 0))
        for value in [gemm.replace('backend=cublas', 'backend=private'), gemm.replace('weight_dtype=BF16', 'weight_dtype=private'), gemm.replace('weight_strides=0,3,1', 'weight_strides=0,3'), gemm.replace('weight_device=cuda', 'weight_device=private'), gemm.replace('weight_ordinal=0', 'weight_ordinal=256'), gemm.replace('weight_ordinal=0', 'weight_ordinal=-1'), gemm.replace('weight_device=cuda;weight_ordinal=0', 'weight_device=cpu;weight_ordinal=1'), gemm.replace(';weight_device=cuda;weight_ordinal=0', '')]:
            self.assertEqual(reference.operation_contexts(value), [])
        self.assertEqual(len(reference.operation_contexts((' ' + record) * 10)), 8)
        self.assertEqual(reference.operation_contexts('x' * 4096 + record), [])

    def test_compiled_receipt_allows_only_launcher_delta_and_binds_archive(self):
        def patch(path, content):
            return f'diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n{content}\n'.encode()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = patch('mistralrs-core/src/lib.rs', '+engine')
            old = engine + patch('scripts/bf16_cli_reference.py', '+old')
            new = engine + patch('scripts/bf16_cli_reference.py', '+new')
            archive = root / 'compiled.patch'
            archive.write_bytes(old)
            source = {'root': str(root), 'baseRevision': 'a' * 40, 'sourcePatchSha256': reference.digest(new), 'cargoLockSha256': 'b' * 64, 'cargoConfigSha256': 'c' * 64}
            compiled = dict(source, sourcePatchSha256=reference.digest(old))
            with mock.patch.object(reference, 'patch_bytes', return_value=new):
                self.assertTrue(reference.validate_build_source(compiled, source, str(archive), time.monotonic() + 1))
            for forbidden in [patch('Makefile', '+changed flags'), patch('mistralrs-core/src/lib.rs', '+changed engine')]:
                changed = new + forbidden
                with mock.patch.object(reference, 'patch_bytes', return_value=changed):
                    with self.assertRaisesRegex(reference.Refusal, 'outside_launcher'):
                        reference.validate_build_source(compiled, dict(source, sourcePatchSha256=reference.digest(changed)), str(archive), time.monotonic() + 1)
            archive.write_bytes(old + b'tampered')
            with self.assertRaisesRegex(reference.Refusal, 'archive_hash_mismatch'):
                reference.validate_build_source(compiled, source, str(archive), time.monotonic() + 1)

    def test_binary_digest_mismatch_prevents_any_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = plan(root)
            (root / 'mistralrs').write_bytes(b'not the candidate')
            with mock.patch.object(reference, 'source_identity', return_value=value['source']), mock.patch.object(reference.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(reference.Refusal, 'binary_identity_mismatch'):
                    reference.launch(value)
                popen.assert_not_called()

    def test_loopback_subprocess_reference_cleanup_and_fresh_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / 'mistralrs'
            binary.write_text(f'''#!{sys.executable}
import http.server,json,sys
if '--help' in sys.argv: raise SystemExit(0)
port=int(sys.argv[sys.argv.index('--port')+1])
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  value={{'data':[{{'id':'default'}},{{'id':sys.argv[sys.argv.index('--model-id')+1],'status':'loaded'}}]}}; body=json.dumps(value).encode(); self.send_response(200); self.end_headers(); self.wfile.write(body)
 def do_POST(self):
  request=json.loads(self.rfile.read(int(self.headers['Content-Length']))); value={{'usage':{{'prompt_tokens':8,'completion_tokens':request['max_tokens']}},'choices':[{{'finish_reason':'length','message':{{'content':'sensitive output'}}}}]}}; body=json.dumps(value).encode(); self.send_response(200); self.end_headers(); self.wfile.write(body)
http.server.HTTPServer(('127.0.0.1',port),Handler).serve_forever()
''')
            binary.chmod(0o700)
            value = plan(root)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                value['port'] = probe.getsockname()[1]
            value['binary']['sha256'] = reference.file_digest(binary)
            request = {'messages': [{'role': 'user', 'content': [{'kind': 'text', 'text': 'sensitive prompt'}]}], 'maxOutputTokens': 24, 'jsonMode': False, 'thinking': False}
            (root / 'request.json').write_bytes(reference.canonical(request))
            value['request']['sha256'] = reference.file_digest(root / 'request.json')
            source = dict(value['source'], cargoLockSha256='e' * 64, cargoConfigSha256='f' * 64)
            (root / 'build.json').write_bytes(reference.canonical({'source': source, 'binaryPath': str(binary), 'binarySha256': value['binary']['sha256'], 'features': ['cuda', 'nccl'], 'defaultFeatures': False}))
            leases = reference.gpu_leases
            with mock.patch.object(reference, 'source_identity', return_value=source), mock.patch.object(reference, 'validate_model', return_value={'loader': 'multimodal_qwen3_5'}), mock.patch.object(reference, 'gpu_memory_sample', return_value={'unavailable': 'portable_fixture'}), mock.patch.object(reference, 'gpu_leases', side_effect=lambda uuids, deadline: leases(uuids, deadline, root / 'leases', lambda _uuid: '')):
                self.assertTrue(reference.launch(value))
                report = (root / 'result/report.json').read_bytes()
                self.assertNotIn(b'sensitive prompt', report)
                self.assertNotIn(b'sensitive output', report)
                retained = json.loads(report)
                self.assertTrue(retained['cleanupCompletedBeforeLeaseRelease'])
                self.assertEqual(retained['attempts'][0]['apiUsage']['completion_tokens'], 24)
                with self.assertRaises(FileExistsError):
                    reference.launch(value)


if __name__ == '__main__':
    unittest.main()
