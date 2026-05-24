#!/usr/bin/env python3
"""Create a starter DCU LLM regression repository scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent


FILES = {
    "configs/matrix.yaml": """
        images:
          - tag: dcu-vllm:iter-001
            framework: vllm
            baseline: true
          - tag: dcu-sglang:iter-001
            framework: sglang

        models:
          - name: qwen2.5-7b-instruct
            served_model_name: qwen2.5-7b-instruct
            path: /models/Qwen2.5-7B-Instruct
            tp: 1
            cards: "0"
            max_model_len: 32768
            accuracy: [gsm8k, math_500, humaneval]
            perf:
              input_lens: [512, 2048]
              output_lens: [128, 512]
              concurrencies: [1, 4, 16]
              num_prompts: 1000
    """,
    "configs/thresholds.yaml": """
        accuracy:
          default_abs_drop: 0.005
          humaneval_pass_at_1_abs_drop: 0.01
        performance:
          throughput_drop_ratio: 0.05
          ttft_p99_increase_ratio: 0.10
          tpot_p99_increase_ratio: 0.10
          max_error_rate: 0
    """,
    "templates/docker-compose.vllm.yaml.j2": """
        services:
          {{ case_id }}:
            image: {{ image }}
            container_name: {{ case_id }}
            network_mode: host
            ipc: host
            privileged: true
            devices:
              - /dev/kfd
              - /dev/dri
            volumes:
              - /models:/models:ro
              - ./results:/results
            environment:
              HIP_VISIBLE_DEVICES: "{{ cards }}"
              ROCR_VISIBLE_DEVICES: "{{ cards }}"
              HSA_FORCE_FINE_GRAIN_PCIE: "1"
            command: >
              vllm serve {{ model_path }}
              --host 0.0.0.0
              --port {{ port }}
              --served-model-name {{ served_model_name }}
              --tensor-parallel-size {{ tp }}
              --max-model-len {{ max_model_len }}
              --gpu-memory-utilization 0.90
              --trust-remote-code
              --disable-log-requests
    """,
    "templates/docker-compose.sglang.yaml.j2": """
        services:
          {{ case_id }}:
            image: {{ image }}
            container_name: {{ case_id }}
            network_mode: host
            ipc: host
            privileged: true
            devices:
              - /dev/kfd
              - /dev/dri
            volumes:
              - /models:/models:ro
              - ./results:/results
            environment:
              HIP_VISIBLE_DEVICES: "{{ cards }}"
              ROCR_VISIBLE_DEVICES: "{{ cards }}"
              HSA_FORCE_FINE_GRAIN_PCIE: "1"
            command: >
              python -m sglang.launch_server
              --model-path {{ model_path }}
              --host 0.0.0.0
              --port {{ port }}
              --served-model-name {{ served_model_name }}
              --tp-size {{ tp }}
              --context-length {{ max_model_len }}
              --trust-remote-code
    """,
    "scripts/wait_ready.py": """
        #!/usr/bin/env python3
        import argparse
        import json
        import time
        import urllib.error
        import urllib.request


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--base-url", required=True)
            parser.add_argument("--timeout", type=int, default=900)
            args = parser.parse_args()
            deadline = time.time() + args.timeout
            url = args.base_url.rstrip("/") + "/v1/models"
            last_error = None
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(url, timeout=10) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    if payload.get("data") is not None:
                        print("ready")
                        return
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    last_error = exc
                time.sleep(5)
            raise SystemExit(f"service not ready: {last_error}")


        if __name__ == "__main__":
            main()
    """,
    "scripts/run_accuracy.py": """
        #!/usr/bin/env python3
        import argparse
        import subprocess


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--model", required=True)
            parser.add_argument("--base-url", required=True)
            parser.add_argument("--datasets", nargs="+", required=True)
            parser.add_argument("--work-dir", required=True)
            args = parser.parse_args()
            cmd = [
                "evalscope", "eval",
                "--model", args.model,
                "--api-url", args.base_url.rstrip("/") + "/v1",
                "--api-key", "EMPTY",
                "--eval-type", "openai_api",
                "--datasets", *args.datasets,
                "--generation-config", "temperature=0,top_p=1,max_tokens=2048",
                "--work-dir", args.work_dir,
            ]
            subprocess.run(cmd, check=True)


        if __name__ == "__main__":
            main()
    """,
    "scripts/run_perf.py": """
        #!/usr/bin/env python3
        import argparse
        import subprocess
        from pathlib import Path


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--framework", choices=["vllm", "sglang"], required=True)
            parser.add_argument("--model", required=True)
            parser.add_argument("--base-url", required=True)
            parser.add_argument("--input-len", type=int, required=True)
            parser.add_argument("--output-len", type=int, required=True)
            parser.add_argument("--concurrency", type=int, required=True)
            parser.add_argument("--num-prompts", type=int, default=1000)
            parser.add_argument("--result-dir", required=True)
            args = parser.parse_args()
            result_dir = Path(args.result_dir)
            result_dir.mkdir(parents=True, exist_ok=True)
            if args.framework == "vllm":
                cmd = [
                    "vllm", "bench", "serve",
                    "--backend", "vllm",
                    "--base-url", args.base_url,
                    "--model", args.model,
                    "--dataset-name", "random",
                    "--random-input-len", str(args.input_len),
                    "--random-output-len", str(args.output_len),
                    "--num-prompts", str(args.num_prompts),
                    "--request-rate", "inf",
                    "--max-concurrency", str(args.concurrency),
                    "--save-result",
                    "--save-detailed",
                    "--result-dir", str(result_dir),
                ]
            else:
                cmd = [
                    "python", "-m", "sglang.bench_serving",
                    "--backend", "sglang-oai",
                    "--base-url", args.base_url,
                    "--model", args.model,
                    "--dataset-name", "random",
                    "--random-input-len", str(args.input_len),
                    "--random-output-len", str(args.output_len),
                    "--num-prompts", str(args.num_prompts),
                    "--request-rate", "inf",
                    "--max-concurrency", str(args.concurrency),
                    "--output-file", str(result_dir / "sglang.jsonl"),
                ]
            subprocess.run(cmd, check=True)


        if __name__ == "__main__":
            main()
    """,
    "scripts/render_report.py": """
        #!/usr/bin/env python3
        import argparse
        import csv
        from pathlib import Path


        def read_csv(path):
            if not path.exists():
                return []
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))


        def table(rows):
            if not rows:
                return "_No data found._"
            headers = list(rows[0].keys())
            out = [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
            for row in rows:
                out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
            return "\\n".join(out)


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--accuracy-csv", default="results/accuracy_summary.csv")
            parser.add_argument("--perf-csv", default="results/perf_summary.csv")
            parser.add_argument("--regression-csv", default="results/regression_summary.csv")
            parser.add_argument("--output", default="reports/report.md")
            args = parser.parse_args()
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            content = [
                "# DCU LLM Image Regression Report",
                "",
                "## Accuracy",
                table(read_csv(Path(args.accuracy_csv))),
                "",
                "## Performance",
                table(read_csv(Path(args.perf_csv))),
                "",
                "## Regression Gates",
                table(read_csv(Path(args.regression_csv))),
                "",
            ]
            output.write_text("\\n".join(content), encoding="utf-8")
            print(output)


        if __name__ == "__main__":
            main()
    """,
    "README.md": """
        # DCU LLM Regression

        Matrix-driven automation for vLLM/SGLang DCU image iteration testing.

        Typical flow:

        1. Edit `configs/matrix.yaml`.
        2. Start services with generated compose files or your CI runner.
        3. Run `scripts/wait_ready.py`.
        4. Run `scripts/run_accuracy.py`.
        5. Run `scripts/run_perf.py`.
        6. Normalize metrics, compare with baseline, and run `scripts/render_report.py`.
    """,
}


def write_file(root: Path, relative: str, content: str, force: bool) -> None:
    path = root / relative
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip(), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", default=".")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in FILES.items():
        write_file(root, relative, content, args.force)
    (root / "results").mkdir(exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    print(f"created scaffold under {root}")


if __name__ == "__main__":
    main()
