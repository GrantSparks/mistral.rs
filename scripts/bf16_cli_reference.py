#!/usr/bin/env python3
"""Finite, isolated BF16 reference launcher; payload text never enters reports."""
import argparse
import contextlib
import csv
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

MAX_TOTAL_SECONDS = 900
MAX_REQUEST_BYTES = 128 * 1024
MAX_RESPONSE_BYTES = 256 * 1024
MAX_WARN_BYTES = 64 * 1024
MAX_REPORT_BYTES = 256 * 1024
LEASE_DIRECTORY = Path('/var/lock/synemantic-gpu-qualification')
NVIDIA_SMI = '/usr/bin/nvidia-smi'
MODEL_REPOSITORY = 'Qwen/Qwen3.8-27B'
MODEL_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
UUID_PATTERN = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z')
SHA_PATTERN = re.compile(r'[0-9a-f]{64}\Z')


class Refusal(Exception):
    pass


class HttpFailure(Refusal):
    def __init__(self, status, detail):
        super().__init__(f'http_status_{status}')
        self.detail = detail


LAUNCHER_ONLY_PATHS = {
    b'scripts/bf16_cli_reference.py',
    b'scripts/test_bf16_cli_reference.py',
    b'docs/src/content/docs/guides/perf/bf16-reference.md',
}
TECHNICAL_NAMES = (
    'CUDA_ERROR_INVALID_VALUE', 'CUDA_ERROR_OUT_OF_MEMORY', 'CUDA_ERROR_NOT_INITIALIZED',
    'CUDA_ERROR_DEINITIALIZED', 'CUDA_ERROR_INVALID_DEVICE', 'CUDA_ERROR_INVALID_CONTEXT',
    'CUDA_ERROR_CONTEXT_ALREADY_CURRENT', 'CUDA_ERROR_PEER_ACCESS_UNSUPPORTED',
    'CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED', 'CUDA_ERROR_INVALID_HANDLE',
    'CUDA_ERROR_NOT_FOUND', 'CUDA_ERROR_NOT_READY', 'CUDA_ERROR_ILLEGAL_ADDRESS',
    'CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES', 'CUDA_ERROR_LAUNCH_TIMEOUT',
    'CUDA_ERROR_LAUNCH_FAILED', 'CUDA_ERROR_ASSERT', 'CUDA_ERROR_NOT_SUPPORTED',
    'CUDA_ERROR_INVALID_PTX', 'CUDA_ERROR_NO_BINARY_FOR_GPU', 'CUDA_ERROR_UNKNOWN',
    'CUBLAS_STATUS_NOT_INITIALIZED', 'CUBLAS_STATUS_ALLOC_FAILED',
    'CUBLAS_STATUS_INVALID_VALUE', 'CUBLAS_STATUS_ARCH_MISMATCH',
    'CUBLAS_STATUS_MAPPING_ERROR', 'CUBLAS_STATUS_EXECUTION_FAILED',
    'CUBLAS_STATUS_INTERNAL_ERROR', 'CUBLAS_STATUS_NOT_SUPPORTED',
    'DriverError', 'CublasError', 'CompileError',
)
TECHNICAL_PHRASES = (
    'invalid argument', 'invalid device context', 'invalid resource handle',
    'out of memory', 'an illegal memory access was encountered',
    'no kernel image is available for execution on the device',
    'peer access is not supported between these two devices',
    'operation not supported', 'misaligned address', 'device-side assert triggered',
    'unspecified launch failure', 'missing kernel', 'unsupported dtype',
    'matmul is only supported for contiguous tensors',
    'staging cross-gpu layer transfers through cpu',
)
OP_CONTEXT_PATTERN = re.compile(r'(?<![A-Za-z0-9_])BF16_OP_CONTEXT\{([^{}]{1,640})\}(?![A-Za-z0-9_])')
OP_CONTEXT_DTYPES = {'U8', 'U32', 'I16', 'I32', 'I64', 'BF16', 'F16', 'F32', 'F64', 'F8E4M3', 'F6E2M3', 'F6E3M2', 'F4', 'F8E8M0'}
OP_CONTEXT_STAGES = {'attention_qkv', 'attention_sdpa', 'attention_output', 'gdn', 'mlp_gate', 'mlp_up', 'mlp_down', 'full_attention_layer', 'linear_attention_layer', 'lm_head'}
OP_CONTEXT_MAX_INTEGER = 1_000_000_000
OP_CONTEXT_MAX_RECORDS = 8


def operation_contexts(text):
    text = text[:4096]
    contexts = []
    for match in OP_CONTEXT_PATTERN.finditer(text):
        parts = match[1].split(';')
        if any(part.count('=') != 1 for part in parts):
            continue
        fields = dict(part.split('=', 1) for part in parts)
        if len(fields) != len(parts):
            continue
        kind = fields.get('kind')
        schema = {
            'layer': ('kind', 'layer', 'stage', 'device', 'ordinal', 'dtype', 'dims', 'strides'),
            'gemm': ('kind', 'backend', 'device', 'ordinal', 'dtype', 'dims', 'strides', 'weight_device', 'weight_ordinal', 'weight_dtype', 'weight_dims', 'weight_strides'),
        }.get(kind)
        if schema is None or tuple(fields) != schema:
            continue
        if fields['device'] not in {'cpu', 'cuda', 'metal'} or fields['dtype'] not in OP_CONTEXT_DTYPES:
            continue
        if kind == 'layer' and fields['stage'] not in OP_CONTEXT_STAGES:
            continue
        if kind == 'gemm' and (fields['backend'] not in {'cublas', 'cublaslt'} or fields['weight_device'] not in {'cpu', 'cuda', 'metal'} or fields['weight_dtype'] not in OP_CONTEXT_DTYPES):
            continue
        parsed = dict(fields)
        valid = True
        for key, bound in [('ordinal', 255)] + ([('layer', 4095)] if kind == 'layer' else [('weight_ordinal', 255)]):
            if not re.fullmatch(r'0|[1-9][0-9]{0,9}', fields[key]) or int(fields[key]) > bound:
                valid = False
                break
            parsed[key] = int(fields[key])
        if not valid or (fields['device'] == 'cpu' and parsed['ordinal'] != 0):
            continue
        if kind == 'gemm' and fields['weight_device'] == 'cpu' and parsed['weight_ordinal'] != 0:
            continue
        vectors = ('dims', 'strides') + (('weight_dims', 'weight_strides') if kind == 'gemm' else ())
        for key in vectors:
            if not re.fullmatch(r'(?:0|[1-9][0-9]{0,9})(?:,(?:0|[1-9][0-9]{0,9})){0,3}', fields[key]):
                valid = False
                break
            values = [int(value) for value in fields[key].split(',')]
            if any(value > OP_CONTEXT_MAX_INTEGER for value in values):
                valid = False
                break
            parsed[key] = values
        if not valid or len(parsed['dims']) != len(parsed['strides']):
            continue
        if kind == 'gemm' and len(parsed['weight_dims']) != len(parsed['weight_strides']):
            continue
        contexts.append(parsed)
        if len(contexts) == OP_CONTEXT_MAX_RECORDS:
            break
    return contexts


def technical_detail(data):
    text = data[:4096].decode('utf-8', errors='replace')
    lowered = text.lower()
    names = [name for name in TECHNICAL_NAMES if re.search(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])', text)]
    phrases = [phrase for phrase in TECHNICAL_PHRASES if phrase in lowered]
    contexts = operation_contexts(text)
    return {'technicalNames': names, 'technicalMessages': phrases, 'operationContexts': contexts, 'unavailable': None if names or phrases or contexts else 'no_allowlisted_technical_detail', 'freeTextRetained': False}


def http_error_detail(status, data):
    message = b''
    try:
        value = json.loads(data)
        # BaseJsonModelError.partial_response can contain generated text: never inspect it.
        if isinstance(value, dict) and isinstance(value.get('message'), str):
            message = value['message'].encode()[:4096]
    except (ValueError, UnicodeDecodeError):
        pass
    return {'status': status, 'bodySha256': digest(data), 'bodyBytes': len(data), 'messageSha256': digest(message), **technical_detail(message)}


def patch_bytes(root, deadline):
    patch = checked_command(['git', '-C', str(root), 'diff', '--binary', '--no-ext-diff', '--no-textconv'], deadline, 8 * 1024 * 1024)
    names = checked_command(['git', '-C', str(root), 'ls-files', '--others', '--exclude-standard', '-z'], deadline)
    for raw_name in sorted(name for name in names.split(b'\0') if name):
        result = subprocess.run(['git', '-C', str(root), 'diff', '--no-index', '--binary', '--no-ext-diff', '--no-textconv', '--', '/dev/null', os.fsdecode(raw_name)], capture_output=True, timeout=remaining(deadline, 10), check=False)
        if result.returncode not in (0, 1) or len(result.stdout) > 8 * 1024 * 1024:
            raise Refusal('untracked_source_diff_failed')
        patch += result.stdout
        if len(patch) > 8 * 1024 * 1024:
            raise Refusal('source_patch_exceeds_bound')
    return patch


def compiled_patch_bytes(patch):
    result = []
    for chunk in re.split(rb'(?m)(?=^diff --git )', patch):
        if not chunk:
            continue
        match = re.match(rb'diff --git a/(\S+) b/(\S+)\n', chunk)
        if not match or match[1] != match[2]:
            raise Refusal('compiled_patch_header_invalid')
        if match[1] not in LAUNCHER_ONLY_PATHS:
            result.append(chunk)
    return b''.join(result)


def validate_build_source(build_source, source, archive, deadline):
    if build_source == source:
        return False
    if not archive or any(build_source.get(key) != value for key, value in source.items() if key != 'sourcePatchSha256'):
        raise Refusal('binary_build_source_identity_mismatch')
    path = Path(archive)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise Refusal('compiled_source_archive_invalid')
    old = path.read_bytes()
    if digest(old) != build_source.get('sourcePatchSha256'):
        raise Refusal('compiled_source_archive_hash_mismatch')
    current = patch_bytes(source['root'], deadline)
    if digest(current) != source['sourcePatchSha256'] or compiled_patch_bytes(old) != compiled_patch_bytes(current):
        raise Refusal('compiled_source_changed_outside_launcher_allowlist')
    return True


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path, deadline=None):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        while chunk := source.read(1024 * 1024):
            if deadline is not None:
                remaining(deadline)
            result.update(chunk)
    return result.hexdigest()


def remaining(deadline, cap=None):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise Refusal('total_deadline_exceeded')
    return min(seconds, cap) if cap is not None else seconds


def checked_command(argv, deadline, maximum=128 * 1024):
    try:
        result = subprocess.run(argv, capture_output=True, timeout=remaining(deadline, 10), check=False)
    except subprocess.TimeoutExpired as error:
        raise Refusal('identity_or_gpu_query_timeout') from error
    if result.returncode != 0 or len(result.stdout) > maximum or len(result.stderr) > maximum:
        raise Refusal('identity_or_gpu_query_failed')
    return result.stdout


def source_identity(root, deadline):
    root = Path(root).resolve(strict=True)
    revision = checked_command(['git', '-C', str(root), 'rev-parse', 'HEAD'], deadline).decode().strip()
    patch = patch_bytes(root, deadline)
    lock = root / 'Cargo.lock'
    config = root / '.cargo/config.toml'
    return {'root': str(root), 'baseRevision': revision, 'sourcePatchSha256': digest(patch), 'cargoLockSha256': file_digest(lock, deadline), 'cargoConfigSha256': file_digest(config, deadline)}


def read_json(path, maximum=MAX_REQUEST_BYTES):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise Refusal('input_file_invalid_or_over_bound')
    data = path.read_bytes()
    try:
        return json.loads(data), data
    except (ValueError, UnicodeDecodeError) as error:
        raise Refusal('input_json_invalid') from error


def require_sha(value):
    if not isinstance(value, str) or not SHA_PATTERN.fullmatch(value):
        raise Refusal('sha256_must_be_lowercase_hex')
    return value


def validate_plan(plan):
    required = {'schemaVersion', 'binary', 'buildIdentityPath', 'source', 'model', 'request', 'gpuUuids', 'port', 'outputDirectory', 'readinessSeconds', 'requestSeconds', 'totalSeconds', 'outputTokens'}
    if not isinstance(plan, dict) or set(plan) not in (required, required | {'compiledSourcePatchPath'}) or plan['schemaVersion'] != 1:
        raise Refusal('plan_schema_invalid')
    if set(plan['binary']) != {'path', 'sha256'} or set(plan['source']) != {'root', 'baseRevision', 'sourcePatchSha256'}:
        raise Refusal('binary_or_source_identity_invalid')
    if set(plan['model']) != {'root', 'manifestSha256', 'repository', 'revision'} or set(plan['request']) != {'path', 'sha256'}:
        raise Refusal('model_or_request_identity_invalid')
    for value in [plan['binary']['sha256'], plan['source']['sourcePatchSha256'], plan['model']['manifestSha256'], plan['request']['sha256']]:
        require_sha(value)
    if not re.fullmatch(r'[0-9a-f]{40}', plan['source']['baseRevision']):
        raise Refusal('base_revision_invalid')
    if plan['model']['repository'] != MODEL_REPOSITORY or plan['model']['revision'] != MODEL_REVISION:
        raise Refusal('unexpected_reference_model')
    uuids = plan['gpuUuids']
    if not isinstance(uuids, list) or len(uuids) != 2 or len(set(uuids)) != 2 or not all(UUID_PATTERN.fullmatch(uuid) for uuid in uuids):
        raise Refusal('exactly_two_distinct_gpu_uuids_required')
    for key, cap in [('readinessSeconds', 600), ('requestSeconds', 300), ('totalSeconds', MAX_TOTAL_SECONDS)]:
        if type(plan[key]) is not int or not (10 if key == 'totalSeconds' else 1) <= plan[key] <= cap:
            raise Refusal('timeout_out_of_bounds')
    if type(plan['port']) is not int or not 1024 <= plan['port'] <= 65535:
        raise Refusal('port_out_of_bounds')
    lengths = plan['outputTokens']
    if not isinstance(lengths, list) or not 1 <= len(lengths) <= 2 or not all(type(length) is int and 1 <= length <= 512 for length in lengths):
        raise Refusal('output_lengths_out_of_bounds')
    for path in [plan['binary']['path'], plan['buildIdentityPath'], plan['source']['root'], plan['model']['root'], plan['request']['path'], plan['outputDirectory']]:
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise Refusal('absolute_paths_required')
    if 'compiledSourcePatchPath' in plan and (not isinstance(plan['compiledSourcePatchPath'], str) or not Path(plan['compiledSourcePatchPath']).is_absolute()):
        raise Refusal('absolute_compiled_source_archive_required')
    output = Path(plan['outputDirectory'])
    if '..' in output.parts or any(parent.is_symlink() for parent in [output, *output.parents]):
        raise Refusal('output_traversal_or_symlink_refused')
    resolved = output.resolve(strict=False)
    if not (resolved.is_relative_to(Path('/tmp').resolve()) or resolved.is_relative_to((Path(plan['source']['root']) / 'target').resolve())):
        raise Refusal('output_must_be_fresh_tmp_or_source_target_directory')
    return plan


def validate_model(model, deadline):
    root = Path(model['root']).resolve(strict=True)
    manifest, raw = read_json(root / 'manifest.json')
    if digest(raw) != model['manifestSha256']:
        raise Refusal('model_manifest_identity_mismatch')
    if manifest.get('kind') != 'pinnedBf16Base' or manifest.get('format') != 'huggingface-safetensors-bf16' or manifest.get('toolVersions', {}).get('source-repository') != MODEL_REPOSITORY:
        raise Refusal('not_the_pristine_bf16_base')
    files = manifest.get('files', [])
    if not isinstance(files, list) or not 1 <= len(files) <= 128:
        raise Refusal('model_manifest_file_bound_invalid')
    seen = set()
    for item in files:
        relative = Path(item['relativePath'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
            raise Refusal('model_manifest_path_invalid')
        seen.add(str(relative))
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item['byteLength']:
            raise Refusal('model_file_size_or_path_mismatch')
        if not path.name.endswith('.safetensors') and file_digest(path, deadline) != item['digest'].removeprefix('sha256:'):
            raise Refusal('model_metadata_digest_mismatch')
    config, _ = read_json(root / 'config.json')
    text = config.get('text_config', {})
    if config.get('architectures') != ['Qwen3_5ForConditionalGeneration'] or text.get('num_hidden_layers') != 64 or text.get('dtype') != 'bfloat16' or 'quantization_config' in config or 'quantization_config' in text:
        raise Refusal('model_architecture_layers_or_bf16_invalid')
    return {'manifestSha256': digest(raw), 'contentDigest': manifest.get('contentDigest'), 'byteLength': manifest.get('byteLength'), 'repository': MODEL_REPOSITORY, 'revision': MODEL_REVISION, 'revisionAuthority': 'owner_pinned_input', 'metadataHashesVerified': True, 'weightHashesVerified': False, 'weightIdentityAuthority': 'pinned_manifest_plus_sizes', 'layerCount': 64, 'loader': 'multimodal_qwen3_5'}


def normalize_request(request, model_id, length):
    allowed = {'messages', 'maxOutputTokens', 'jsonMode', 'thinking', 'temperature'}
    if not isinstance(request, dict) or not set(request).issubset(allowed) or not isinstance(request.get('messages'), list) or not 1 <= len(request['messages']) <= 32:
        raise Refusal('serving_request_schema_invalid')
    messages = []
    for message in request['messages']:
        if set(message) != {'role', 'content'} or message['role'] not in ['system', 'user', 'assistant'] or not isinstance(message['content'], list):
            raise Refusal('text_only_serving_message_required')
        text = []
        for part in message['content']:
            if set(part) != {'kind', 'text'} or part['kind'] != 'text' or not isinstance(part['text'], str):
                raise Refusal('text_only_serving_part_required')
            text.append(part['text'])
        messages.append({'role': message['role'], 'content': '\n'.join(text)})
    if type(request.get('thinking')) is not bool or type(request.get('jsonMode')) is not bool:
        raise Refusal('explicit_thinking_and_json_mode_required')
    payload = {'model': model_id, 'messages': messages, 'max_tokens': length, 'stream': False, 'enable_thinking': request['thinking'], 'temperature': 0.0, 'top_k': 1}
    if 'temperature' in request and request['temperature'] is not None and request['temperature'] > 0:
        payload['temperature'] = request['temperature']
    if request['jsonMode']:
        payload['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'reference', 'schema': {}}}
    return payload


def cli_argv(binary, model_root, port):
    return [str(binary), 'serve', '--max-seqs', '1', '--host', '127.0.0.1', '--port', str(port), '--no-ui', 'multimodal', '--model-id', str(model_root), '--dtype', 'bf16', '--device-layers', '0:32;1:32']


@contextlib.contextmanager
def gpu_leases(uuids, deadline, directory=LEASE_DIRECTORY, query=None):
    query = query or (lambda uuid: checked_command([NVIDIA_SMI, '--id', uuid, '--query-compute-apps=pid', '--format=csv,noheader,nounits'], deadline, 4096).decode())
    directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise Refusal('gpu_lease_directory_invalid')
    descriptors = []
    try:
        for uuid in sorted(uuids):
            fd = os.open(directory / f'{uuid}.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644)
            descriptors.append(fd)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise Refusal('gpu_lease_not_regular')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise Refusal('gpu_already_leased') from error
        for uuid in sorted(uuids):
            if query(uuid).strip():
                raise Refusal('gpu_not_idle_or_query_unknown')
        yield
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def require_listener_owner(process, port):
    # Fail closed on Linux if any listener at this port is foreign or unprovable.
    if process.poll() is not None:
        raise Refusal('cli_exited_before_http_request')
    listeners = set()
    try:
        for table in [Path('/proc/net/tcp'), Path('/proc/net/tcp6')]:
            for line in table.read_text().splitlines()[1:]:
                fields = line.split()
                if fields[3] == '0A' and int(fields[1].split(':')[1], 16) == port:
                    listeners.add(fields[9])
        owned = set()
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                if os.getpgid(int(entry.name)) != process.pid:
                    continue
                for descriptor in (entry / 'fd').iterdir():
                    try:
                        target = os.readlink(descriptor)
                        if target.startswith('socket:[') and target.endswith(']'):
                            owned.add(target[8:-1])
                    except OSError:
                        continue
            except (OSError, ProcessLookupError):
                continue
        if not listeners or not listeners.issubset(owned):
            raise Refusal('listener_not_owned_by_cli_process_group')
    except (OSError, ValueError, IndexError) as error:
        raise Refusal('listener_ownership_unavailable') from error


def http_json(port, path, timeout, payload=None, process=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        if process is not None:
            connection.connect()
            require_listener_owner(process, port)
        body = None if payload is None else canonical(payload)
        connection.request('GET' if body is None else 'POST', path, body=body, headers={} if body is None else {'Content-Type': 'application/json'})
        response = connection.getresponse()
        data = response.read(MAX_RESPONSE_BYTES + 1)
        if len(data) > MAX_RESPONSE_BYTES:
            raise Refusal('http_response_over_bound')
        if response.status != 200:
            raise HttpFailure(response.status, http_error_detail(response.status, data))
        try:
            return json.loads(data), data
        except (ValueError, UnicodeDecodeError) as error:
            raise Refusal('http_response_json_invalid') from error
    finally:
        connection.close()


def wait_ready(process, port, deadline, readiness_seconds, expected_model_id):
    ready_deadline = min(deadline, time.monotonic() + readiness_seconds)
    while True:
        if process.poll() is not None:
            raise Refusal('cli_exited_before_ready')
        try:
            with request_deadline(remaining(ready_deadline, 1)):
                value, _ = http_json(port, '/v1/models', remaining(ready_deadline, 1), process=process)
            models = value.get('data', [])
            if not isinstance(models, list) or len(models) != 2 or {item.get('id') for item in models} != {'default', expected_model_id}:
                raise Refusal('foreign_or_unexpected_served_model')
            concrete = next(item for item in models if item['id'] == expected_model_id)
            if concrete.get('status') != 'loaded':
                raise Refusal('expected_model_not_loaded')
            return expected_model_id
        except Refusal as error:
            if str(error) in ['foreign_or_unexpected_served_model', 'listener_not_owned_by_cli_process_group', 'listener_ownership_unavailable']:
                raise
            remaining(ready_deadline)
        except (OSError, http.client.HTTPException):
            remaining(ready_deadline)
        time.sleep(min(0.1, remaining(ready_deadline)))


def stop_group(process, grace=2):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=grace)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        # A CLI leader can exit while descendants still own the group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def diagnostic(line):
    if b'WARN' not in line and b'ERROR' not in line and b'error' not in line.lower():
        return None
    lowered = line.lower()
    categories = [name for name, pattern in [('out_of_memory', b'out of memory'), ('cuda_error', b'cuda'), ('shape_mismatch', b'shape'), ('dtype_mismatch', b'dtype'), ('nccl_error', b'nccl'), ('model_load_error', b'loading'), ('kernel_error', b'kernel'), ('file_io_error', b'file')] if pattern in lowered]
    locations = re.findall(rb'[A-Za-z0-9_/-]+\.rs:[0-9]+(?::[0-9]+)?', line)[:4]
    return {'level': 'WARN' if b'WARN' in line else 'ERROR', 'classes': categories or ['unclassified'], 'sourceLocations': [location.decode('ascii') for location in locations], 'lineSha256': digest(line), 'lineBytes': len(line), 'payloadTextRedacted': True, **technical_detail(line)}


class LogDrain:
    def __init__(self, stream):
        self.stream = stream
        self.bytes = 0
        self.warnings = []
        self.warning_bytes = 0
        self.dropped = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while line := self.stream.readline(4096):
            self.bytes += len(line)
            item = diagnostic(line)
            if item is not None:
                size = len(canonical(item))
                if self.warning_bytes + size <= MAX_WARN_BYTES:
                    self.warnings.append(item)
                    self.warning_bytes += size
                else:
                    self.dropped += 1

    def close(self):
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise Refusal('cli_log_pipe_did_not_close')
        self.stream.close()


def response_summary(value, raw, wall):
    usage = value.get('usage')
    if not isinstance(usage, dict):
        raise Refusal('api_usage_missing')
    choices = value.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        raise Refusal('api_choice_count_invalid')
    content = choices[0].get('message', {}).get('content')
    encoded = b'' if content is None else content.encode()
    parsed = False
    if isinstance(content, str):
        try:
            json.loads(content)
            parsed = True
        except ValueError:
            pass
    return {'apiUsage': usage, 'httpWallSeconds': wall, 'responseSha256': digest(raw), 'outputContentSha256': digest(encoded), 'outputContentBytes': len(encoded), 'finishReason': choices[0].get('finish_reason'), 'jsonParseSucceeded': parsed, 'qualitySchema': None, 'schemaValidationUnavailable': 'no_schema_supplied', 'decodeRate': None, 'decodeRateUnavailable': 'reference_reports_usage_and_http_wall_separately'}


@contextlib.contextmanager
def fresh_output(path):
    # Every ancestor is opened without following links; publication uses this FD.
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(path.parts[1:]):
            final = index == len(path.parts) - 2
            try:
                os.mkdir(part, mode=0o700, dir_fd=directory)
            except FileExistsError:
                if final:
                    raise
            next_directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        identity = os.fstat(directory)
        yield directory, (identity.st_dev, identity.st_ino)
    finally:
        os.close(directory)


def publish(output, report, directory, identity):
    current = output.stat(follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != identity:
        raise Refusal('output_directory_identity_changed')
    data = canonical(report)
    if len(data) > MAX_REPORT_BYTES:
        raise Refusal('report_over_bound')
    fd = os.open('report.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    with os.fdopen(fd, 'wb') as destination:
        destination.write(data + b'\n')


@contextlib.contextmanager
def request_deadline(seconds):
    # Socket idle timeouts alone permit an indefinitely trickling response.
    handler = signal.getsignal(signal.SIGALRM)
    previous, interval = signal.getitimer(signal.ITIMER_REAL)
    start = time.monotonic()
    request_owns_alarm = previous == 0 or seconds < previous
    def expired(signum, frame):
        if request_owns_alarm:
            raise Refusal('request_deadline_exceeded')
        handler(signum, frame)
    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, min(seconds, previous) if previous else seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)
        restored = previous - (time.monotonic() - start)
        if previous and restored > 0:
            signal.setitimer(signal.ITIMER_REAL, restored, interval)


def boundary():
    before = time.monotonic_ns()
    utc = time.time_ns()
    after = time.monotonic_ns()
    return {'monotonicNs': before, 'utcUnixNs': utc, 'utcAnchorUncertaintyNs': after - before}


def gpu_memory_sample(uuids, deadline, phase):
    result = {'phase': phase, 'boundary': boundary()}
    try:
        raw = checked_command([NVIDIA_SMI, '--id', ','.join(uuids), '--query-gpu=uuid,name,driver_version,memory.total,memory.used,memory.free,power.draw', '--format=csv,noheader,nounits'], deadline, 4096)
        rows = list(csv.reader(raw.decode().splitlines()))
        devices = []
        for row in rows:
            if len(row) != 7 or row[0].strip() not in uuids:
                raise Refusal('gpu_memory_sample_invalid')
            values = [float(value.strip()) for value in row[3:6]]
            if any(value < 0 or value != value or value == float('inf') for value in values):
                raise Refusal('gpu_memory_sample_invalid')
            try:
                power = float(row[6].strip())
                if power < 0 or power != power or power == float('inf'):
                    power = None
            except ValueError:
                power = None
            devices.append({'uuid': row[0].strip(), 'name': row[1].strip(), 'driverVersion': row[2].strip(), 'totalMiB': values[0], 'usedMiB': values[1], 'freeMiB': values[2], 'powerWatts': power})
        if len(devices) != 2 or {item['uuid'] for item in devices} != set(uuids):
            raise Refusal('gpu_memory_sample_invalid')
        result['devices'] = devices
    except (Refusal, OSError, UnicodeDecodeError, ValueError):
        result['unavailable'] = 'nvidia_smi_memory_query_unavailable_or_invalid'
    return result


def launch(plan):
    validate_plan(plan)
    start_boundary = boundary()
    started = time.monotonic()
    deadline = started + plan['totalSeconds'] - 8
    source = source_identity(plan['source']['root'], deadline)
    if any(source[key] != value for key, value in plan['source'].items()):
        raise Refusal('source_identity_mismatch')
    binary = Path(plan['binary']['path']).resolve(strict=True)
    if file_digest(binary, deadline) != plan['binary']['sha256']:
        raise Refusal('binary_identity_mismatch')
    build, _ = read_json(plan['buildIdentityPath'])
    launcher_delta = validate_build_source(build.get('source', {}), source, plan.get('compiledSourcePatchPath'), deadline)
    if build.get('binarySha256') != plan['binary']['sha256'] or build.get('binaryPath') != str(binary) or build.get('features') != ['cuda', 'nccl'] or build.get('defaultFeatures') is not False:
        raise Refusal('binary_build_identity_mismatch')
    model = validate_model(plan['model'], deadline)
    checked_command(cli_argv(binary, plan['model']['root'], plan['port']) + ['--help'], deadline)
    request, raw_request = read_json(plan['request']['path'])
    if digest(raw_request) != plan['request']['sha256']:
        raise Refusal('request_identity_mismatch')
    normalize_request(request, 'preflight', plan['outputTokens'][0])
    output = Path(plan['outputDirectory'])
    output_context = fresh_output(output)
    output_fd, output_identity = output_context.__enter__()
    report = {'schemaVersion': 1, 'kind': 'pristine_bf16_reference', 'source': source, 'compiledSource': build['source'], 'launcherOnlyDeltaVerified': launcher_delta, 'binary': plan['binary'], 'buildIdentitySha256': file_digest(plan['buildIdentityPath'], deadline), 'model': model, 'originalRequestSha256': digest(raw_request), 'gpuUuids': plan['gpuUuids'], 'requestedLayerMapping': '0:32;1:32', 'effectiveLayerMappingUnavailable': 'no_structured_loader_mapping_receipt', 'requestedWorldSize': 1, 'tensorParallelClaim': False, 'cliArgumentParserChecked': True, 'expectedServedModelId': plan['model']['root'], 'listenerOwnershipPolicy': 'linux_proc_socket_inode_cli_process_group_before_http_payload', 'attempts': [], 'status': 'failed', 'diagnostics': [], 'startBoundary': start_boundary, 'gpuMemorySamples': [], 'peakMemoryUnavailable': 'bounded_boundary_samples_only', 'cudaToolkitBuildEnvironment': build.get('environment', {}), 'cudaRuntimeVersionUnavailable': 'no_runtime_version_hook'}
    process = None
    drain = None
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', plan['port']))
        argv = cli_argv(binary, plan['model']['root'], plan['port'])
        environment = dict(os.environ)
        environment.update({'CUDA_VISIBLE_DEVICES': ','.join(plan['gpuUuids']), 'MISTRALRS_NO_NCCL': '1', 'MISTRALRS_MN_LOCAL_WORLD_SIZE': '1', 'MISTRALRS_MN_GLOBAL_WORLD_SIZE': '1', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'RUST_LOG': 'warn'})
        for key in ['MISTRALRS_MN_NODE_ID', 'MISTRALRS_MN_MASTER_ADDR', 'MISTRALRS_MN_MASTER_PORT']:
            environment.pop(key, None)
        with gpu_leases(plan['gpuUuids'], deadline):
            try:
                report['gpuMemorySamples'].append(gpu_memory_sample(plan['gpuUuids'], deadline, 'before_load'))
                report['processStartBoundary'] = boundary()
                process = subprocess.Popen(argv, env=environment, cwd=source['root'], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
                report['processIdentity'] = {'pid': process.pid, 'processGroupId': process.pid, 'binarySha256': plan['binary']['sha256']}
                drain = LogDrain(process.stdout)
                ready_start = time.monotonic()
                model_id = wait_ready(process, plan['port'], deadline, plan['readinessSeconds'], plan['model']['root'])
                report['readyWallSeconds'] = time.monotonic() - ready_start
                for length in plan['outputTokens']:
                    payload = normalize_request(request, model_id, length)
                    request_boundary = boundary()
                    request_start = time.monotonic()
                    attempt = {'requestStartBoundary': request_boundary, 'requestedOutputTokens': length, 'effectiveRequestSha256': digest(canonical(payload)), 'normalizedMessagesSha256': digest(canonical(payload['messages']))}
                    report['attempts'].append(attempt)
                    try:
                        with request_deadline(remaining(deadline, plan['requestSeconds'])):
                            value, raw_response = http_json(plan['port'], '/v1/chat/completions', remaining(deadline, plan['requestSeconds']), payload, process=process)
                        attempt.update(response_summary(value, raw_response, time.monotonic() - request_start))
                    finally:
                        attempt['requestEndBoundary'] = boundary()
                        attempt['httpWallSeconds'] = time.monotonic() - request_start
                report['gpuMemorySamples'].append(gpu_memory_sample(plan['gpuUuids'], deadline, 'after_requests_before_cleanup'))
                report['status'] = 'completed'
            finally:
                if process is not None:
                    stop_group(process)
                    report['cliExitCodeAfterCleanup'] = process.returncode
                if drain is not None:
                    drain.close()
                    report['diagnostics'] = drain.warnings
                    report['cliLogBytesDiscarded'] = drain.bytes
                    report['diagnosticsDropped'] = drain.dropped
                report['cleanupCompletedBeforeLeaseRelease'] = True
    except (Refusal, OSError, http.client.HTTPException, subprocess.TimeoutExpired) as error:
        report['failure'] = str(error) if isinstance(error, Refusal) else type(error).__name__
        if isinstance(error, HttpFailure):
            report['httpError'] = error.detail
    finally:
        report['totalWallSeconds'] = time.monotonic() - started
        try:
            publish(output, report, output_fd, output_identity)
        finally:
            output_context.__exit__(None, None, None)
    return report['status'] == 'completed'


def build_start(output, root):
    identity = {'source': source_identity(root, time.monotonic() + 30), 'features': ['cuda', 'nccl'], 'defaultFeatures': False, 'environment': {key: os.environ.get(key) for key in ['CUDA_HOME', 'CUDA_PATH', 'CUDA_COMPUTE_CAP', 'CUDA_NVCC_FLAGS', 'RUSTFLAGS', 'RUSTC_WRAPPER', 'CARGO_HOME', 'CARGO_TARGET_DIR']}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_bytes(canonical(identity) + b'\n')


def build_finish(start, binary, output):
    identity, _ = read_json(start)
    if source_identity(identity['source']['root'], time.monotonic() + 30) != identity['source']:
        raise Refusal('source_changed_during_build')
    identity['binaryPath'] = str(Path(binary).resolve(strict=True))
    identity['binarySha256'] = file_digest(binary)
    Path(output).write_bytes(canonical(identity) + b'\n')


@contextlib.contextmanager
def bounded_signals(seconds):
    def interrupted(signum, _frame):
        raise Refusal('total_deadline_exceeded' if signum == signal.SIGALRM else 'launcher_interrupted')
    previous = {number: signal.getsignal(number) for number in [signal.SIGALRM, signal.SIGINT, signal.SIGTERM]}
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        for number in previous:
            signal.signal(number, interrupted)
        signal.setitimer(signal.ITIMER_REAL, seconds - 8)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        for number, handler in previous.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('launch')
    run.add_argument('--plan', required=True)
    start = commands.add_parser('build-start')
    start.add_argument('--output', required=True)
    start.add_argument('--source-root', required=True)
    finish = commands.add_parser('build-finish')
    finish.add_argument('--start', required=True)
    finish.add_argument('--binary', required=True)
    finish.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'launch':
            plan, _ = read_json(args.plan)
            validate_plan(plan)
            with bounded_signals(plan['totalSeconds']):
                return 0 if launch(plan) else 1
        if args.command == 'build-start':
            build_start(args.output, args.source_root)
        else:
            build_finish(args.start, args.binary, args.output)
        return 0
    except (Refusal, OSError, ValueError, KeyError, TypeError) as error:
        # Input errors may contain payload text, so only our fixed refusal labels leave the process.
        print(f'BF16 reference refused: {str(error) if isinstance(error, Refusal) else type(error).__name__}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
