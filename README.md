# DCU LLM 镜像迭代自动化测试

这个仓库用于在海光 DCU 环境上，对多轮镜像迭代中的多个 LLM 模型进行 vLLM/SGLang 精度与性能回归测试。

覆盖范围：

- 测试环境：`nmz`、`bmz`、`kme`
- 推理框架：`vllm`、`sglang` 海光 DCU 镜像
- 模型类型：Qwen、DeepSeek/DPSK、GLM5、Kimi 2.5、MiniMax 等
- 启动模式：重新创建容器，或复用已有容器/已有 OpenAI API 服务
- 精度测试：OpenCompass 跑 `gsm8k`、`math500`、`humaneval`、`mmmu`
- 性能测试：框架自带 benchmark，使用 `random` 或 `random-ids`
- SLA：短输入 `(2k,1k)` 要求 `ttft <= 3ms`、`tpot <= 50ms`；长输入 `(64k,1k)` 要求 `ttft <= 30ms`、`tpot <= 100ms`
- 输出：当前结果、基线对比、SLA 判定、Markdown 报告

## 快速开始

1. 编辑 [configs/matrix.yaml](C:/Users/HP/Documents/acc_pref/configs/matrix.yaml)，填入真实镜像、模型路径、DCU 卡号、OpenCompass 命令模板。
2. 先生成 dry-run 计划：

```bash
python scripts/dcu_regression.py --config configs/matrix.yaml --plan-out results/run_plan.json
```

3. 在 DCU 测试机上执行：

```bash
python scripts/dcu_regression.py --config configs/matrix.yaml --execute
```

4. 汇总已有结果并生成报告：

```bash
python scripts/collect_report.py --results-dir results --baseline results/baseline/summary.json --output reports/report.md
```

## 运行模式

`create_container`：重新用镜像创建服务容器，适合镜像迭代回归。

`existing_endpoint`：直接复用已经启动好的 OpenAI-compatible 服务，只跑精度/性能。

`existing_container`：复用已有容器，在容器里执行服务启动命令；适合镜像已经常驻、只想快速换模型的场景。

## 关键约束

- 同一张 DCU 卡默认不要并行跑多个重负载服务。
- 性能测试前必须先完成 `/v1/models` ready 检查和 smoke 请求。
- 精度测试固定 `temperature=0`，并固定数据集版本、prompt 模板和模型 chat template。
- MMMU 必须保证图像路径、多模态输入格式、OpenCompass 配置完全一致，否则不同镜像不可比。
- 报告必须保存原始 benchmark/OpenCompass 输出，方便定位回归来源。

