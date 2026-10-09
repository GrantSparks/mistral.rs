.PHONY: fmt docs-regen docs-check

fmt:
	cargo fmt
	ruff format
	find mistralrs-* -type f \( -name "*.metal" -o -name "*.c" -o -name "*.cu" -o -name "*.hpp" -o -name "*.h" -o -name "*.cpp" \) -exec clang-format -i {} +
docs-regen:
	cargo test -p mistralrs-cli regenerate_cli_reference -- --ignored
	cargo test -p mistralrs-server-core regenerate_openapi -- --ignored
	cargo test -p mistralrs-core regenerate_supported_models -- --ignored
	python3 docs/scripts/render_pyi.py
	python3 docs/scripts/render_examples.py

docs-check:
	cargo test -p mistralrs-cli docgen::cli_reference_matches_committed
	cargo test -p mistralrs-server-core openapi_matches_committed
	cargo test -p mistralrs-core supported_models_matches_committed
	python3 docs/scripts/render_pyi.py --check

.PHONY: observation-check observation-test observation-format
observation-check:
	cargo check -p mistralrs-core -p mistralrs-quant -p mistralrs
observation-test:
	cargo test -p mistralrs-core -p mistralrs-quant observation_contract --lib
observation-format:
	rustfmt --edition 2021 --config skip_children=true $(OBSERVATION_FILES)

BF16_CUDA_ROOT ?= /usr/local/cuda-13.3
BF16_CLI_TARGET_DIR ?= $(if $(CARGO_TARGET_DIR),$(CARGO_TARGET_DIR),target)
BF16_CLI_BINARY = $(BF16_CLI_TARGET_DIR)/release/mistralrs
BF16_CLI_BUILD_START = $(BF16_CLI_TARGET_DIR)/bf16-cli-build-start.json
BF16_CLI_BUILD_IDENTITY = $(BF16_CLI_TARGET_DIR)/bf16-cli-build-identity.json
BF16_CLI_ENV = CUDA_HOME="$(BF16_CUDA_ROOT)" CUDA_PATH="$(BF16_CUDA_ROOT)" PATH="$(BF16_CUDA_ROOT)/bin:$$PATH" LD_LIBRARY_PATH="$(BF16_CUDA_ROOT)/targets/x86_64-linux/lib:$${LD_LIBRARY_PATH:-}"
BF16_LAYER_DEVICE_CUDA_FILTER ?= bf16_layer_device

.PHONY: build-bf16-reference-cli test-bf16-reference-launcher run-bf16-reference
.PHONY: test-bf16-layer-devices test-bf16-layer-devices-cuda
test-bf16-layer-devices:
	cargo test --locked -p mistralrs-core --no-default-features bf16_layer_device --lib

test-bf16-layer-devices-cuda:
	env $(BF16_CLI_ENV) cargo test --locked -p mistralrs-core --no-default-features --features cuda,nccl $(BF16_LAYER_DEVICE_CUDA_FILTER) --lib -- --ignored --test-threads=1

build-bf16-reference-cli:
	env $(BF16_CLI_ENV) python3 scripts/bf16_cli_reference.py build-start --source-root "$(CURDIR)" --output "$(BF16_CLI_BUILD_START)"
	env $(BF16_CLI_ENV) cargo build --locked --release -p mistralrs-cli --no-default-features --features cuda,nccl
	env $(BF16_CLI_ENV) python3 scripts/bf16_cli_reference.py build-finish --start "$(BF16_CLI_BUILD_START)" --binary "$(BF16_CLI_BINARY)" --output "$(BF16_CLI_BUILD_IDENTITY)"

test-bf16-reference-launcher:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p 'test_bf16_cli_reference.py'

run-bf16-reference:
	@test -n "$(BF16_REFERENCE_PLAN)" || { echo 'Set BF16_REFERENCE_PLAN to the reviewed bounded launch plan'; exit 1; }
	python3 scripts/bf16_cli_reference.py launch --plan "$(BF16_REFERENCE_PLAN)"

# Read-only test fixtures; CUDA runs require the external owner lease wrapper.
FP8_LIBRARY_CUDA_FILTER ?= fp8_library_cuda_small
FP8_GEMM_CUDA_DEVICE ?= 0
.PHONY: test-fp8-library test-fp8-library-cuda format-fp8-library
test-fp8-library:
	cargo test --locked -p mistralrs-quant --no-default-features fp8_library_cpu --lib

test-fp8-library-cuda:
	@test "$(FP8_GEMM_GPU_LEASE_HELD)" = 1 || { echo 'Run through the owner wrapper with both UUID leases, idle checks and timeout'; exit 1; }
	env $(BF16_CLI_ENV) FP8_GEMM_CUDA_DEVICE="$(FP8_GEMM_CUDA_DEVICE)" cargo test --locked -p mistralrs-quant --no-default-features --features cuda,nccl $(FP8_LIBRARY_CUDA_FILTER) --lib -- --ignored --test-threads=1 --nocapture

format-fp8-library:
	$(MAKE) observation-format OBSERVATION_FILES="mistralrs-quant/src/fp8/tests.rs mistralrs-quant/src/blockwise_fp8/mod.rs"
