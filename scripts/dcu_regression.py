#!/usr/bin/env python3
"""Orchestrate DCU LLM image-iteration accuracy and performance tests."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: PyYAML for YAML config. Use configs/matrix.json or install with `pip install pyyaml`.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def safe_id(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def run_command(cmd: str, *, execute: bool, log_file: Path | None = None, env: dict[str, str] | None = None) -> int:
    print(cmd)
    if not execute:
        return 0
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"\n$ {cmd}\n")
            proc = subprocess.run(cmd, shell=True, text=True, stdout=handle, stderr=subprocess.STDOUT, env=env)
            return proc.returncode
    return subprocess.run(cmd, shell=True, env=env).returncode


@dataclass(frozen=True)
class Case:
    env_name: str
    image_name: str
    image: str
    framework: str
    mode: str
    model_name: str
    served_model_name: str
    model_path: str
    tp: int
    cards: str
    max_model_len: int
    port: int
    case_id: str
    baseline: bool
    existing_endpoint: str | None = None
    existing_container: str | None = None


def iter_cases(config: dict[str, Any]) -> list[Case]:
    base_port = int(config["run"].get("base_port", 18000))
    cases: list[Case] = []
    index = 0
    for env_cfg in config["environments"]:
        for image_cfg in config["images"]:
            framework = image_cfg["framework"]
            for model_cfg in config["models"]:
                if framework not in model_cfg.get("frameworks", [framework]):
                    continue
                port = int(image_cfg.get("port", base_port + index))
                raw_id = f"{env_cfg['name']}__{image_cfg['name']}__{model_cfg['name']}__tp{model_cfg['tp']}__dcu{model_cfg['cards']}"
                cases.append(
                    Case(
                        env_name=env_cfg["name"],
                        image_name=image_cfg["name"],
                        image=image_cfg.get("image", ""),
                        framework=framework,
                        mode=image_cfg.get("mode", "create_container"),
                        model_name=model_cfg["name"],
                        served_model_name=model_cfg["served_model_name"],
                        model_path=model_cfg["path"],
                        tp=int(model_cfg["tp"]),
                        cards=str(model_cfg["cards"]),
                        max_model_len=int(model_cfg.get("max_model_len", 32768)),
                        port=port,
                        case_id=safe_id(raw_id),
                        baseline=bool(image_cfg.get("baseline", False)),
                        existing_endpoint=image_cfg.get("endpoint"),
                        existing_container=image_cfg.get("container_name"),
                    )
                )
                index += 1
    return cases


def format_service_command(config: dict[str, Any], case: Case) -> str:
    key = "vllm_command" if case.framework == "vllm" else "sglang_command"
    return config["service"][key].format(
        model_path=case.model_path,
        port=case.port,
        served_model_name=case.served_model_name,
        tp=case.tp,
        max_model_len=case.max_model_len,
    )


def docker_run_command(config: dict[str, Any], case: Case, serve_cmd: str) -> str:
    common = config["service"]["docker_common"]
    parts = ["docker", "run", "-d", "--name", case.case_id]
    if common.get("network"):
        parts.extend(["--network", common["network"]])
    if common.get("ipc"):
        parts.extend(["--ipc", common["ipc"]])
    if common.get("privileged"):
        parts.append("--privileged")
    for device in common.get("devices", []):
        parts.extend(["--device", device])
    for volume in common.get("volumes", []):
        parts.extend(["-v", volume])
    parts.extend(["-e", f"HIP_VISIBLE_DEVICES={case.cards}", "-e", f"ROCR_VISIBLE_DEVICES={case.cards}"])
    for env_cfg in config.get("environments", []):
        if env_cfg["name"] == case.env_name:
            for key, value in env_cfg.get("device_env", {}).items():
                parts.extend(["-e", f"{key}={value}"])
    parts.extend([case.image, "bash", "-lc", serve_cmd])
    return shell_join(parts)


def docker_exec_command(case: Case, serve_cmd: str) -> str:
    if not case.existing_container:
        raise ValueError(f"{case.case_id}: existing_container mode requires container_name")
    return shell_join(["docker", "exec", "-d", case.existing_container, "bash", "-lc", serve_cmd])


def base_url(case: Case) -> str:
    return case.existing_endpoint or f"http://127.0.0.1:{case.port}"


def wait_ready(url: str, timeout_sec: int, execute: bool) -> None:
    print(f"wait ready: {url}/v1/models")
    if not execute:
        return
    deadline = time.time() + timeout_sec
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("data") is not None:
                return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(5)
    raise RuntimeError(f"service not ready: {url}; last_error={last_error}")


def smoke_request(case: Case, prompt: str, execute: bool, log_file: Path) -> None:
    payload = json.dumps(
        {
            "model": case.served_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 32,
        },
        ensure_ascii=False,
    )
    cmd = (
        "curl -s "
        + shlex.quote(base_url(case).rstrip("/") + "/v1/chat/completions")
        + " -H "
        + shlex.quote("Content-Type: application/json")
        + " -d "
        + shlex.quote(payload)
    )
    if run_command(cmd, execute=execute, log_file=log_file) != 0:
        raise RuntimeError(f"smoke request failed: {case.case_id}")


def accuracy_command(config: dict[str, Any], case: Case, datasets: list[str], work_dir: Path) -> tuple[str, dict[str, str]]:
    acc_cfg = config["accuracy"]
    dataset_map = acc_cfg.get("dataset_map", {})
    mapped = [dataset_map.get(item, item) for item in datasets]
    cmd = acc_cfg["command_template"].format(
        opencompass_model_config=acc_cfg["opencompass_model_config"],
        datasets=" ".join(mapped),
        work_dir=str(work_dir),
        base_url=base_url(case).rstrip("/") + "/v1",
        model=case.served_model_name,
    )
    env = os.environ.copy()
    env["OPENCOMPASS_API_BASE"] = base_url(case).rstrip("/") + "/v1"
    env["OPENCOMPASS_MODEL_NAME"] = case.served_model_name
    env["OPENCOMPASS_API_KEY"] = env.get("OPENCOMPASS_API_KEY", "EMPTY")
    return cmd, env


def perf_command(config: dict[str, Any], case: Case, perf_case: dict[str, Any], concurrency: int, result_dir: Path) -> str:
    perf_cfg = config["performance"]
    dataset = perf_cfg.get("dataset", "random")
    if case.framework == "vllm":
        parts = [
            "vllm", "bench", "serve",
            "--backend", "vllm",
            "--base-url", base_url(case),
            "--model", case.served_model_name,
            "--dataset-name", dataset,
            "--random-input-len", str(perf_case["input_len"]),
            "--random-output-len", str(perf_case["output_len"]),
            "--num-prompts", str(perf_cfg.get("num_prompts", 1000)),
            "--request-rate", str(perf_cfg.get("request_rate", "inf")),
            "--max-concurrency", str(concurrency),
            "--save-result",
            "--save-detailed",
            "--result-dir", str(result_dir),
        ]
    else:
        parts = [
            "python", "-m", "sglang.bench_serving",
            "--backend", "sglang-oai",
            "--base-url", base_url(case),
            "--model", case.served_model_name,
            "--dataset-name", dataset,
            "--random-input-len", str(perf_case["input_len"]),
            "--random-output-len", str(perf_case["output_len"]),
            "--num-prompts", str(perf_cfg.get("num_prompts", 1000)),
            "--request-rate", str(perf_cfg.get("request_rate", "inf")),
            "--max-concurrency", str(concurrency),
            "--output-file", str(result_dir / "sglang.jsonl"),
        ]
    return shell_join(parts)


def card_set(case: Case) -> set[str]:
    return {item.strip() for item in case.cards.split(",") if item.strip()}


def make_batches(cases: list[Case], max_parallel: int) -> list[list[Case]]:
    pending = list(cases)
    batches: list[list[Case]] = []
    while pending:
        used_cards: set[str] = set()
        batch: list[Case] = []
        remaining: list[Case] = []
        for case in pending:
            cards = card_set(case)
            if len(batch) < max_parallel and cards.isdisjoint(used_cards):
                batch.append(case)
                used_cards.update(cards)
            else:
                remaining.append(case)
        batches.append(batch)
        pending = remaining
    return batches


def write_plan(cases: list[Case], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([case.__dict__ for case in cases], indent=2, ensure_ascii=False), encoding="utf-8")


def run_case(config: dict[str, Any], case: Case, args: argparse.Namespace) -> None:
    out_dir = Path(config["run"].get("output_dir", "results"))
    case_dir = out_dir / "cases" / case.case_id
    log_file = out_dir / "logs" / f"{case.case_id}.log"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(json.dumps(case.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")

    launched_by_runner = False
    try:
        if not args.skip_launch and case.mode != "existing_endpoint":
            serve_cmd = format_service_command(config, case)
            if case.mode == "create_container":
                cmd = docker_run_command(config, case, serve_cmd)
                launched_by_runner = True
            elif case.mode == "existing_container":
                cmd = docker_exec_command(case, serve_cmd)
            else:
                raise ValueError(f"unsupported mode: {case.mode}")
            if run_command(cmd, execute=args.execute, log_file=log_file) != 0:
                raise RuntimeError(f"launch failed: {case.case_id}")

        wait_ready(base_url(case), int(config["run"].get("service_ready_timeout_sec", 900)), args.execute)
        smoke_request(case, config["run"].get("smoke_prompt", "hello"), args.execute, log_file)

        if not args.skip_accuracy:
            model_cfg = next(item for item in config["models"] if item["name"] == case.model_name)
            datasets = list(model_cfg.get("accuracy", []))
            if datasets:
                work_dir = Path(config["accuracy"].get("output_dir", "results/accuracy")) / case.case_id
                cmd, env = accuracy_command(config, case, datasets, work_dir)
                if run_command(cmd, execute=args.execute, log_file=log_file, env=env) != 0:
                    raise RuntimeError(f"accuracy failed: {case.case_id}")

        if not args.skip_perf:
            for perf_case in config["performance"].get("cases", []):
                for concurrency in config["performance"].get("concurrencies", [1]):
                    result_dir = (
                        Path(config["performance"].get("output_dir", "results/perf"))
                        / case.case_id
                        / f"{perf_case['name']}__c{concurrency}"
                    )
                    cmd = perf_command(config, case, perf_case, int(concurrency), result_dir)
                    if run_command(cmd, execute=args.execute, log_file=log_file) != 0:
                        raise RuntimeError(f"performance failed: {case.case_id} {perf_case['name']} c{concurrency}")
    finally:
        cleanup = bool(config["run"].get("cleanup_after_case", True))
        if cleanup and launched_by_runner:
            run_command(shell_join(["docker", "rm", "-f", case.case_id]), execute=args.execute, log_file=log_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/matrix.yaml")
    parser.add_argument("--execute", action="store_true", help="Actually run docker/OpenCompass/benchmark commands.")
    parser.add_argument("--plan-out", default="results/run_plan.json")
    parser.add_argument("--skip-launch", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--parallel-services", type=int, default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    out_dir = Path(config["run"].get("output_dir", "results"))
    cases = iter_cases(config)
    write_plan(cases, Path(args.plan_out))
    print(f"cases: {len(cases)}")
    max_parallel = args.parallel_services or int(config["run"].get("parallel_services", 1))
    if max_parallel <= 1:
        for case in cases:
            run_case(config, case, args)
        return

    for batch_index, batch in enumerate(make_batches(cases, max_parallel), start=1):
        print(f"batch {batch_index}: {', '.join(case.case_id for case in batch)}")
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = [pool.submit(run_case, config, case, args) for case in batch]
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
