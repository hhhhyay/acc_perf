# Scripts

- `run_opencompass_sglang_accuracy.py`: SGLang + OpenCompass accuracy runner for Qwen3.5. It waits for idle DCUs, starts model servers on disjoint card groups, runs OpenCompass clients in parallel, writes reports, and can repeat batches until a deadline.

- `dcu_regression.py`: 读取 `configs/matrix.yaml`，生成测试矩阵，启动服务，运行 OpenCompass 和 benchmark。
- `collect_report.py`: 汇总 benchmark JSON/JSONL，检查 SLA，和基线 JSON 对比，生成 Markdown 报告。

默认不执行实际命令；`dcu_regression.py` 只有带 `--execute` 才会运行 Docker、OpenCompass 和 benchmark。
