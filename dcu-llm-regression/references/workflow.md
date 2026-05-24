# End-to-End Workflow

## Test Phases

1. Image preparation
   - Pull or build each image.
   - Record image tag and immutable digest.
   - Record vLLM/SGLang, Python, DTK/ROCm, driver, and kernel versions.

2. Service launch
   - Render one compose/service config per case.
   - Assign unique `port`, `case_id`, and DCU card list.
   - Start non-conflicting cases in parallel.
   - Save service stdout/stderr to `results/logs/<case_id>.log`.

3. Readiness check
   - Poll `GET /v1/models` or the framework health endpoint.
   - Run a one-prompt smoke request before expensive tests.
   - Stop the case early if startup fails or the first request is empty.

4. Accuracy evaluation
   - Run GSM8K, MATH500, HumanEval, and MMMU as configured.
   - Keep generation deterministic.
   - Store raw evaluator output under `results/accuracy/<case_id>/`.

5. Performance evaluation
   - Warm up the service before measuring.
   - Run random dataset cases across configured input/output lengths and concurrency.
   - Store raw benchmark output under `results/perf/<case_id>/`.

6. Metric collection
   - Normalize accuracy metrics to one row per `case_id`, `model`, `dataset`.
   - Normalize performance metrics to one row per `case_id`, `input_len`, `output_len`, `concurrency`.
   - Include raw file paths in normalized records.

7. Baseline comparison
   - Compare current image with the configured baseline image.
   - Mark accuracy drops, throughput drops, and latency increases.
   - Apply thresholds from `configs/thresholds.yaml`.

8. Report generation
   - Generate Markdown by default.
   - Include conclusion, environment, service commands, accuracy table, performance table, regression table, and links to raw files.

## Report Metrics

Accuracy:

- GSM8K: exact match or evaluator score.
- MATH500: exact match or evaluator score.
- HumanEval: pass@1.
- MMMU: accuracy by split and overall.

Performance:

- requests per second.
- output tokens per second.
- total tokens per second when available.
- TTFT p50/p90/p99.
- inter-token latency or TPOT p50/p90/p99.
- end-to-end latency p50/p90/p99.
- error rate.

## Regression Gate Defaults

Use conservative defaults unless the user provides their own:

- accuracy absolute drop greater than `0.005` fails.
- pass@1 absolute drop greater than `0.01` fails.
- output token throughput drop greater than `5%` fails.
- TTFT p99 increase greater than `10%` fails.
- TPOT p99 increase greater than `10%` fails.
- any benchmark error rate greater than `0` fails.
