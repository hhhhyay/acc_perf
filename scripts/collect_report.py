#!/usr/bin/env python3
"""Collect benchmark outputs, check SLA, compare baseline, and render a report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_ALIASES = {
    "ttft_ms": [
        "mean_ttft_ms",
        "median_ttft_ms",
        "ttft_ms",
        "time_to_first_token_ms",
        "Mean TTFT (ms)",
        "Median TTFT (ms)",
    ],
    "tpot_ms": [
        "mean_tpot_ms",
        "median_tpot_ms",
        "tpot_ms",
        "time_per_output_token_ms",
        "Mean TPOT (ms)",
        "Median TPOT (ms)",
    ],
    "request_throughput": [
        "request_throughput",
        "request_throughput_per_s",
        "requests_per_second",
        "Request throughput (req/s)",
    ],
    "output_throughput": [
        "output_throughput",
        "output_token_throughput",
        "output_tokens_per_second",
        "Output token throughput (tok/s)",
    ],
    "error_rate": ["error_rate", "failed_rate", "failure_rate"],
}


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_key = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(value, next_key))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            out.update(flatten(value, f"{prefix}.{index}"))
    else:
        out[prefix] = obj
    return out


def pick_metric(flat: dict[str, Any], names: list[str]) -> float | None:
    for wanted in names:
        for key, value in flat.items():
            if key == wanted or key.endswith("." + wanted):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
    return None


def collect_perf(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    perf_root = results_dir / "perf"
    if not perf_root.exists():
        return rows
    for path in perf_root.rglob("*.json"):
        payload = load_json(path)
        if payload is None:
            continue
        flat = flatten(payload)
        row = {"source": str(path)}
        for metric, aliases in METRIC_ALIASES.items():
            row[metric] = pick_metric(flat, aliases)
        row["sla_pass"] = sla_pass(row, str(path))
        rows.append(row)
    for path in perf_root.rglob("*.jsonl"):
        # SGLang can emit JSONL. Aggregate is framework-version dependent, so keep source.
        rows.append({"source": str(path), "ttft_ms": None, "tpot_ms": None, "request_throughput": None, "output_throughput": None, "error_rate": None, "sla_pass": "unknown"})
    return rows


def sla_pass(row: dict[str, Any], source: str) -> str:
    source_lower = source.lower()
    if "short_2k_1k" in source_lower:
        ttft_limit, tpot_limit = 3.0, 50.0
    elif "long_64k_1k" in source_lower:
        ttft_limit, tpot_limit = 30.0, 100.0
    else:
        return "unknown"
    ttft = row.get("ttft_ms")
    tpot = row.get("tpot_ms")
    if ttft is None or tpot is None:
        return "unknown"
    return "pass" if float(ttft) <= ttft_limit and float(tpot) <= tpot_limit else "fail"


def load_baseline(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def compare_with_baseline(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    # Baseline schema is intentionally simple:
    # {"<source-suffix-or-case-id>": {"ttft_ms": 1.2, "tpot_ms": 30, "output_throughput": 1000}}
    compared: list[dict[str, Any]] = []
    for row in rows:
        matched_key = None
        for key in baseline:
            if key in row["source"]:
                matched_key = key
                break
        item = {"source": row["source"], "baseline_key": matched_key or "", "sla_pass": row.get("sla_pass", "")}
        if matched_key:
            base = baseline[matched_key]
            for metric in ["ttft_ms", "tpot_ms", "request_throughput", "output_throughput"]:
                cur = row.get(metric)
                old = base.get(metric)
                item[f"{metric}_current"] = cur
                item[f"{metric}_baseline"] = old
                item[f"{metric}_delta"] = None if cur is None or old is None else float(cur) - float(old)
        compared.append(item)
    return compared


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No data collected._"
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(field) is None else str(row.get(field)) for field in fields) + " |")
    return "\n".join(lines)


def render_report(path: Path, perf_rows: list[dict[str, Any]], compare_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failed = [row for row in perf_rows if row.get("sla_pass") == "fail"]
    content = [
        "# DCU LLM 镜像迭代测试报告",
        "",
        "## 结论",
        "",
        f"- 性能结果文件数：{len(perf_rows)}",
        f"- SLA 失败项：{len(failed)}",
        "",
        "## 性能与 SLA",
        "",
        markdown_table(perf_rows),
        "",
        "## 基线对比",
        "",
        markdown_table(compare_rows),
        "",
        "## 说明",
        "",
        "- 精度结果由 OpenCompass work-dir 原始输出保留；不同 OpenCompass 版本输出格式差异较大，建议在 CI 中补充项目内固定解析器。",
        "- SGLang JSONL 输出在不同版本中字段差异较大，本脚本先保留源文件并标记 unknown，可按实际字段扩展解析。",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--output", default="reports/report.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    perf_rows = collect_perf(results_dir)
    baseline = load_baseline(Path(args.baseline)) if args.baseline else {}
    compare_rows = compare_with_baseline(perf_rows, baseline)
    write_csv(results_dir / "perf_summary.csv", perf_rows)
    write_csv(results_dir / "baseline_compare.csv", compare_rows)
    render_report(Path(args.output), perf_rows, compare_rows)
    print(args.output)


if __name__ == "__main__":
    main()

