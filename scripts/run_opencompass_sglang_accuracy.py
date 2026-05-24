#!/usr/bin/env python3
"""Run SGLang servers and OpenCompass accuracy jobs without taking busy DCUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def split_devices(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_capture(cmd: list[str], timeout: int = 60) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return proc.stdout


def read_hcu_state() -> dict[int, dict[str, float]]:
    output = ""
    for cmd in (["rocm-smi", "--showuse", "--showmemuse"], ["hy-smi"]):
        try:
            output = run_capture(cmd, timeout=60)
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
        if "HCU[" in output:
            break
    state: dict[int, dict[str, float]] = {}
    for line in output.splitlines():
        if "HCU[" not in line:
            continue
        try:
            idx = int(line.split("HCU[", 1)[1].split("]", 1)[0])
            value = float(line.rsplit(":", 1)[1].strip().split()[0])
        except (IndexError, ValueError):
            continue
        state.setdefault(idx, {})
        if "memory use" in line:
            state[idx]["mem"] = value
        elif "HCU use" in line:
            state[idx]["util"] = value
    return state


def devices_idle(devices: list[int], max_util: float, max_mem: float) -> tuple[bool, str]:
    state = read_hcu_state()
    if not state:
        return False, "no HCU state from rocm-smi/hy-smi"
    busy: list[str] = []
    for dev in devices:
        util = state.get(dev, {}).get("util", 100.0)
        mem = state.get(dev, {}).get("mem", 100.0)
        if util > max_util or mem > max_mem:
            busy.append(f"HCU[{dev}] util={util:.1f}% mem={mem:.1f}%")
    return not busy, "; ".join(busy) if busy else "idle"


def wait_for_idle(models: list[dict[str, Any]], run_cfg: dict[str, Any], deadline: dt.datetime | None, skip: bool) -> None:
    if skip:
        return
    devices: list[int] = []
    for model in models:
        devices.extend(split_devices(model["devices"]))
    devices = sorted(set(devices))
    max_util = float(run_cfg.get("idle_hcu_util_percent", 5))
    max_mem = float(run_cfg.get("idle_hcu_mem_percent", 5))
    interval = int(run_cfg.get("idle_check_interval_sec", 300))
    while True:
        idle, detail = devices_idle(devices, max_util, max_mem)
        print(f"[{dt.datetime.now():%F %T}] idle check devices={devices}: {detail}", flush=True)
        if idle:
            return
        if deadline and dt.datetime.now() + dt.timedelta(seconds=interval) >= deadline:
            raise TimeoutError(f"deadline reached while waiting for idle DCUs: {detail}")
        time.sleep(interval)


def build_server_command(model: dict[str, Any]) -> list[str]:
    cmd = ["sglang", "serve", "--model-path", model["model_path"]]
    cmd.extend(["--host", "0.0.0.0", "--port", str(model["port"])])
    cmd.extend(["--served-model-name", model["served_model_name"]])
    cmd.extend(model["server_args"])
    return cmd


def start_server(model: dict[str, Any], env_base: dict[str, str], log_path: Path, dry_run: bool) -> subprocess.Popen[str] | None:
    if not Path(model["model_path"]).exists():
        raise FileNotFoundError(f"model path does not exist: {model['model_path']}")
    cmd = build_server_command(model)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(env_base)
    env.update(model.get("server_env", {}))
    env["HIP_VISIBLE_DEVICES"] = model["devices"]
    env["CUDA_VISIBLE_DEVICES"] = model["devices"]
    print(f"SERVER: HIP_VISIBLE_DEVICES={model['devices']} " + " ".join(cmd), flush=True)
    if dry_run:
        return None
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(f"$ HIP_VISIBLE_DEVICES={model['devices']} CUDA_VISIBLE_DEVICES={model['devices']} " + " ".join(cmd) + "\n")
    log_handle.flush()
    proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=env)
    setattr(proc, "_acc_perf_log_handle", log_handle)
    return proc


def wait_ready(model: dict[str, Any], timeout_sec: int, dry_run: bool, proc: subprocess.Popen[str] | None = None) -> None:
    if dry_run:
        return
    url = f"http://127.0.0.1:{model['port']}/v1/models"
    deadline = time.time() + timeout_sec
    last_error: Exception | None = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited before ready: {model['name']} exit={proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("data") is not None:
                return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(10)
    raise RuntimeError(f"server not ready: {url}; last_error={last_error}")


def smoke(model: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    payload = json.dumps(
        {
            "model": model["served_model_name"],
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "temperature": 0,
            "max_tokens": 8,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{model['port']}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
    if "choices" not in body:
        raise RuntimeError(f"smoke response did not contain choices: {body[:500]}")


def write_opencompass_config(model: dict[str, Any], datasets: list[dict[str, str]], cfg_path: Path, work_dir: Path) -> None:
    imports = "\n".join(f"    {item['import']}" for item in datasets)
    api_base = f"http://127.0.0.1:{model['port']}/v1"
    content = f"""
from mmengine.config import read_base

with read_base():
{imports}
    from opencompass.configs.summarizers.example import summarizer

datasets = sum(
    [v for k, v in locals().items() if k.endswith("_datasets") or k == "datasets"],
    [],
)

work_dir = {str(work_dir)!r}

from opencompass.models import OpenAISDK

api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
)

models = [
    dict(
        abbr={model["abbr"]!r},
        type=OpenAISDK,
        path={model["served_model_name"]!r},
        openai_api_base={api_base!r},
        tokenizer_path={model["model_path"]!r},
        key="EMPTY",
        meta_template=api_meta_template,
        temperature=0,
        query_per_second={int(model.get("query_per_second", 64))},
        max_out_len={int(model.get("max_out_len", 32768))},
        max_seq_len={int(model.get("max_seq_len", 32768))},
        pred_postprocessor=dict(
            type="opencompass.utils.text_postprocessors.extract_non_reasoning_content"
        ),
        batch_size={int(model.get("batch_size", 32))},
    ),
]
"""
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def run_opencompass(opencompass_root: Path, cfg_path: Path, log_path: Path, dry_run: bool) -> int:
    cmd = [sys.executable, str(opencompass_root / "run.py"), str(cfg_path)]
    print("CLIENT:", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(opencompass_root), stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def stop_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    handle = getattr(proc, "_acc_perf_log_handle", None)
    if handle:
        handle.close()


def collect_report(report_path: Path, batch_name: str, run_id: str, results: dict[str, int], work_dirs: dict[str, Path]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {batch_name} {run_id}", "", "| model | exit_code | work_dir |", "|---|---:|---|"]
    for name, code in results.items():
        lines.append(f"| {name} | {code} | {work_dirs.get(name, '')} |")
    lines.append("")
    for name, work_dir in work_dirs.items():
        summaries = sorted(work_dir.rglob("*summary*"))[:20] if work_dir.exists() else []
        if summaries:
            lines.append(f"## {name} summaries")
            lines.extend(f"- {item}" for item in summaries)
            lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_deadline(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    now = dt.datetime.now()
    if len(value) == 5 and value[2] == ":":
        hour, minute = [int(part) for part in value.split(":", 1)]
        deadline = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if deadline <= now:
            deadline += dt.timedelta(days=1)
        return deadline
    return dt.datetime.fromisoformat(value)


def run_batch(config: dict[str, Any], batch: dict[str, Any], batch_index: int, args: argparse.Namespace, deadline: dt.datetime | None) -> None:
    run_cfg = config["run"]
    models_by_name = {model["name"]: model for model in config["models"]}
    models = [models_by_name[name] for name in batch["models"]]
    wait_for_idle(models, run_cfg, deadline, args.skip_wait)

    run_id = f"{now_stamp()}_batch{batch_index}_{batch['name']}"
    output_root = Path(run_cfg["output_root"]) / run_id
    report_root = Path(run_cfg["report_root"])
    opencompass_root = Path(run_cfg["opencompass_root"])
    server_procs: list[subprocess.Popen[str] | None] = []
    procs_by_name: dict[str, subprocess.Popen[str] | None] = {}
    work_dirs: dict[str, Path] = {}
    try:
        for model in models:
            log_path = output_root / model["name"] / "server.log"
            proc = start_server(model, config.get("server_env", {}), log_path, args.dry_run)
            server_procs.append(proc)
            procs_by_name[model["name"]] = proc
        ready_models: list[dict[str, Any]] = []
        failed: dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = [
                pool.submit(
                    wait_ready,
                    model,
                    int(run_cfg["server_ready_timeout_sec"]),
                    args.dry_run,
                    procs_by_name[model["name"]],
                )
                for model in models
            ]
            future_models = {future: model for future, model in zip(futures, models)}
            for future in concurrent.futures.as_completed(futures):
                model = future_models[future]
                try:
                    future.result()
                    ready_models.append(model)
                except Exception as exc:
                    failed[model["name"]] = str(exc)
        for model in ready_models:
            try:
                smoke(model, args.dry_run)
            except Exception as exc:
                failed[model["name"]] = f"smoke failed: {exc}"
        ready_models = [model for model in ready_models if model["name"] not in failed]
        if args.smoke_only:
            return
        results: dict[str, int] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(ready_models))) as pool:
            future_map = {}
            for model in ready_models:
                model_out = output_root / model["name"]
                cfg_path = model_out / "opencompass_config.py"
                work_dir = model_out / "work_dir"
                work_dirs[model["name"]] = work_dir
                write_opencompass_config(model, config["datasets"], cfg_path, work_dir)
                log_path = model_out / "opencompass.log"
                future = pool.submit(run_opencompass, opencompass_root, cfg_path, log_path, args.dry_run)
                future_map[future] = model["name"]
            for future in concurrent.futures.as_completed(future_map):
                results[future_map[future]] = future.result()
        for name, reason in failed.items():
            results[name] = 1
            fail_dir = output_root / name
            fail_dir.mkdir(parents=True, exist_ok=True)
            (fail_dir / "failure.txt").write_text(reason, encoding="utf-8")
        collect_report(report_root / f"{run_id}.md", batch["name"], run_id, results, work_dirs)
    finally:
        for proc in server_procs:
            stop_process(proc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/mnt11/task/acc_perf/configs/opencompass_sglang_qwen35_qwen36.json")
    parser.add_argument("--until", default=None, help="Deadline, for example 09:00 or 2026-05-25T09:00:00.")
    parser.add_argument("--once", action="store_true", help="Run configured batches once and exit.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    until_value = args.until or (None if args.once else config["run"].get("default_until"))
    deadline = parse_deadline(until_value)
    repeat_batches = 1 if args.once else int(config["run"].get("repeat_batches", 1))
    batch_index = 0
    while True:
        for _ in range(repeat_batches):
            for batch in config["batches"]:
                batch_index += 1
                if deadline and dt.datetime.now() >= deadline:
                    print(f"deadline reached before batch {batch_index}", flush=True)
                    return 0
                run_batch(config, batch, batch_index, args, deadline)
        if args.once or not deadline or dt.datetime.now() >= deadline:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
