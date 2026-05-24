---
name: dcu-llm-regression
description: Build or run automated DCU LLM image-iteration regression workflows for vLLM and SGLang. Use when Codex needs to create a git-style test repository, generate Docker/container orchestration, launch multiple DCU inference services in parallel, evaluate accuracy with GSM8K, MATH500, HumanEval, and MMMU using OpenCompass or EvalScope, benchmark performance with vLLM/SGLang serving benchmarks and random datasets, compare against baseline images, and produce accuracy/performance reports.
---

# DCU LLM Regression

Use this skill to create or operate an automated regression workflow for multiple LLMs across iterative DCU inference images.

## Core Workflow

1. Inspect the target repository and confirm whether an automation scaffold already exists.
2. Create or update a matrix-driven workflow with these axes:
   - image tag or digest
   - framework: `vllm` or `sglang`
   - model path and served model name
   - DCU device binding
   - accuracy tasks: `gsm8k`, `math_500`, `humaneval`, `mmmu`
   - performance cases: random input/output lengths, request counts, request rate, concurrency
3. Generate service launch configs from templates and isolate each service by port and DCU card.
4. Start services in parallel only when device bindings do not conflict.
5. Wait for the OpenAI-compatible endpoint to become ready before running evaluations.
6. Run deterministic accuracy evaluations:
   - set `temperature=0`
   - keep prompt template, dataset version, tokenizer, and max tokens fixed
   - use EvalScope or OpenCompass through OpenAI-compatible API endpoints
7. Run serving benchmarks:
   - use `vllm bench serve` for vLLM
   - use `python -m sglang.bench_serving` for SGLang
   - use `random` data for controlled input/output lengths
8. Collect raw JSON/JSONL/log files, normalize metrics, compare with baseline, and render Markdown/HTML reports.
9. Fail the workflow when configured regression thresholds are exceeded.

## Repository Scaffold

When the user asks to create the automation repository, prefer generating this structure:

```text
configs/
  matrix.yaml
  models.yaml
  thresholds.yaml
templates/
  docker-compose.vllm.yaml.j2
  docker-compose.sglang.yaml.j2
scripts/
  build_images.py
  launch_services.py
  wait_ready.py
  run_accuracy.py
  run_perf.py
  collect_metrics.py
  compare_baseline.py
  render_report.py
results/
reports/
README.md
```

Use `scripts/create_scaffold.py` from this skill to create the initial files, then customize for the user's local image names, model paths, DCU runtime, and CI system.

## DCU Service Rules

- Follow the service launch patterns from `HYGON-AI/dcu-inference-cookbook/docs/model-deployment` when the user references that cookbook.
- Use host networking for benchmark stability unless the user's environment requires isolated Docker networks.
- Include DCU devices such as `/dev/kfd` and `/dev/dri` only when appropriate for the host runtime.
- Bind services with `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`.
- Avoid running two heavy cases on the same DCU card unless the user explicitly wants contention testing.
- Record image digest, framework version, model path, tensor parallel size, max context length, driver/runtime versions, and benchmark command lines in the report.

## Accuracy

Prefer EvalScope for a compact OpenAI API workflow. Use OpenCompass when the repo already standardizes on OpenCompass or when dataset/model support is better there.

Keep generation deterministic:

```bash
evalscope eval \
  --model <served-model-name> \
  --api-url http://127.0.0.1:<port>/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets gsm8k math_500 humaneval \
  --generation-config temperature=0,top_p=1,max_tokens=2048 \
  --work-dir results/accuracy/<case-id>
```

For MMMU, verify that image paths, multimodal message format, model chat template, and evaluator backend are identical across image iterations.

## Performance

Use the framework's own serving benchmark first.

vLLM:

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:<port> \
  --model <served-model-name> \
  --dataset-name random \
  --random-input-len <tokens> \
  --random-output-len <tokens> \
  --num-prompts <n> \
  --request-rate inf \
  --max-concurrency <n> \
  --save-result \
  --save-detailed \
  --result-dir results/perf/<case-id>
```

SGLang:

```bash
python -m sglang.bench_serving \
  --backend sglang-oai \
  --base-url http://127.0.0.1:<port> \
  --model <served-model-name> \
  --dataset-name random \
  --random-input-len <tokens> \
  --random-output-len <tokens> \
  --num-prompts <n> \
  --request-rate inf \
  --max-concurrency <n> \
  --output-file results/perf/<case-id>/sglang.jsonl
```

## References

- Read `references/workflow.md` when designing or explaining the end-to-end automation flow.
- Read `references/config-schema.md` when creating or editing matrix, model, or threshold configs.
- Use `scripts/create_scaffold.py` to generate a starter repository layout.
