# Config Schema

## `configs/matrix.yaml`

```yaml
images:
  - tag: dcu-vllm:iter-001
    framework: vllm
    baseline: true
  - tag: dcu-vllm:iter-002
    framework: vllm

models:
  - name: qwen2.5-7b-instruct
    served_model_name: qwen2.5-7b-instruct
    path: /models/Qwen2.5-7B-Instruct
    tp: 1
    cards: "0"
    max_model_len: 32768
    accuracy: [gsm8k, math_500, humaneval]
    perf:
      input_lens: [512, 2048, 8192]
      output_lens: [128, 512]
      concurrencies: [1, 4, 16, 64]
      num_prompts: 1000

  - name: qwen2.5-vl-7b-instruct
    served_model_name: qwen2.5-vl-7b-instruct
    path: /models/Qwen2.5-VL-7B-Instruct
    tp: 1
    cards: "1"
    max_model_len: 32768
    accuracy: [mmmu]
```

## `configs/thresholds.yaml`

```yaml
accuracy:
  default_abs_drop: 0.005
  humaneval_pass_at_1_abs_drop: 0.01
performance:
  throughput_drop_ratio: 0.05
  ttft_p99_increase_ratio: 0.10
  tpot_p99_increase_ratio: 0.10
  max_error_rate: 0
```

## Case ID

Generate stable case IDs:

```text
<framework>__<image-safe-tag>__<model-name>__tp<tp>__dcu<cards>
```

For performance subcases, append:

```text
__in<input_len>__out<output_len>__c<concurrency>
```
