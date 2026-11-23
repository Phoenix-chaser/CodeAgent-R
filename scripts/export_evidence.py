"""Export a shareable evidence bundle from local experiment artifacts.

README 里的数字此前只能在本机复核：.gitignore 排除了 data/、results/ 与
checkpoints/。这个脚本把其中可以公开的部分整理进 docs/evidence/：

- 逐题结果只保留判定与计数（通过与否、失败类型、token 数、判分证据的摘要）；
- 不带走模型输出、私有测试细节、服务地址、绝对路径、硬件型号和 API key；
- 每个文件同时记录公开版的 sha256 和本地原始文件的 sha256。

导出前按 scripts/privacy_check.py 的规则做一遍检查，命中即中止，不写任何
文件。导出后用 scripts/verify_evidence.py 重算结论；CI 会对已提交的证据包
运行它。

usage:
  python scripts/export_evidence.py --label repair-104-seed42 \
      --results-dir data/experiment/results/<run> --checkpoint-dir checkpoints/sft-<run> \
      [--candidates-manifest data/repair-104/train-candidates.manifest.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import privacy_check  # noqa: E402

# 模型输出、报错原文、任务文件路径、硬件型号：不进公开证据
DROPPED_KEYS = {"raw_response", "generated_code", "error", "tasks_file", "gpu"}
GRADER_COUNTS = ("passed", "exit_status", "expected_tests", "executed_tests", "summary",
                 "duration_ms")
PUBLIC_HOSTS = ("huggingface.co", "arxiv.org", "github.com", "docs.pytest.org")
_URL = re.compile(r"https?://([A-Za-z0-9.\-]+)[^\s\"'<>)\\]*")
_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_./\-])/(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+")
EXTRA_PATTERNS = [("私有 canary", r"CANARY-[0-9a-f]{8}"),
                  ("访问令牌", r"(?:hf|gh|gitlab)_[A-Za-z0-9]{20,}")]
OMITTED = ["model outputs (raw_response, generated_code)",
           "private test details and pytest output tails",
           "absolute paths (last two components kept)",
           "service addresses", "hardware model", "error messages"]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scrub_text(text: str) -> str:
    def host(match):
        return match.group(0) if match.group(1).endswith(PUBLIC_HOSTS) else "<url>"

    def tail(match):
        return "<path>/" + "/".join(match.group(0).split("/")[-2:])

    return _ABSOLUTE_PATH.sub(tail, _URL.sub(host, text))


def grader_summary(details) -> dict:
    """判分证据里可以公开的部分：隔离方式、应当执行与实际执行的测试数。"""
    if not isinstance(details, dict):
        return {}
    layers = details.get("layers", {})
    functional = layers.get("functional", {})
    return {
        "layers": sorted(layers),
        "public_pass": functional.get("public_pass"),
        "private_pass": functional.get("private_pass"),
        "public": {key: functional.get("public", {}).get(key) for key in GRADER_COUNTS},
        "private": {key: functional.get("private", {}).get(key) for key in GRADER_COUNTS},
        "execution": scrub(layers.get("execution")),
        "legality_valid": layers.get("legality", {}).get("valid"),
    }


def scrub(value):
    if isinstance(value, dict):
        return {name: scrub(item) for name, item in value.items() if name not in DROPPED_KEYS}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, str):
        return scrub_text(value)
    return value


def sanitize_report(report: dict) -> dict:
    clean = scrub(report)
    rows = []
    for record in report.get("results", []):
        row = {key: value for key, value in scrub(record).items() if key != "grader_details"}
        row["grader"] = grader_summary(record.get("grader_details"))
        rows.append(row)
    clean["results"] = rows
    return clean


def privacy_violations(text_by_file: dict[str, str]) -> list[str]:
    violations = []
    for relative, text in sorted(text_by_file.items()):
        for label, pattern in [*privacy_check.CONTENT_PATTERNS, *EXTRA_PATTERNS]:
            hits = re.findall(pattern, text, re.IGNORECASE)
            if hits:
                violations.append(f"[{label}] x{len(hits)}: {relative}")
    return violations


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(results_dir: Path | None, checkpoint_dir: Path | None,
            candidates_manifest: Path | None) -> dict[str, tuple[bytes, str | None]]:
    """返回 {包内相对路径: (公开版内容, 本地原始文件的 sha256)}。"""
    files: dict[str, tuple[bytes, str | None]] = {}

    def emit(relative: str, payload, source: Path | None = None) -> None:
        data = payload if isinstance(payload, bytes) else (
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
        files[relative] = (data, _sha256(source.read_bytes()) if source else None)

    def report(relative: str, path: Path) -> None:
        emit(relative, sanitize_report(_read_json(path)), path)

    if results_dir is not None:
        for split in ("dev", "test"):
            for name in ("base", "selected"):
                if (results_dir / split / f"{name}.json").is_file():
                    report(f"{split}/{name}.json", results_dir / split / f"{name}.json")
            comparison = results_dir / split / "comparison.md"
            if comparison.is_file():
                emit(f"{split}/comparison.md",
                     scrub_text(comparison.read_text(encoding="utf-8")), comparison)
        selection = results_dir / "selection" / "selection.json"
        if selection.is_file():
            emit("selection.json", scrub(_read_json(selection)), selection)
        for epoch in sorted(results_dir.glob("selection/epoch-*-dev.json")):
            report(f"dev/{epoch.name}", epoch)
    if checkpoint_dir is not None:
        for name, target in (("training-manifest.json", "training.json"),
                             ("checkpoint-index.json", "checkpoint-index.json")):
            if (checkpoint_dir / name).is_file():
                emit(target, scrub(_read_json(checkpoint_dir / name)), checkpoint_dir / name)
        log = checkpoint_dir / "training-log.jsonl"
        if log.is_file():      # .jsonl 被 .gitignore 排除，转成 JSON 数组
            entries = [json.loads(line) for line in
                       log.read_text(encoding="utf-8").splitlines() if line.strip()]
            emit("training-log.json", scrub(entries), log)
    if candidates_manifest is not None:
        emit("teacher.json", scrub(_read_json(candidates_manifest)), candidates_manifest)
    return files


def build_index(label: str, files: dict[str, tuple[bytes, str | None]]) -> dict:
    models = set()
    for name, (data, _) in files.items():
        if name.endswith(".json"):
            payload = json.loads(data)
            if isinstance(payload, dict) and payload.get("model"):
                models.add(f"{payload['model']}@{payload.get('resolved_revision', '?')}")
    return {
        "schema_version": 1, "label": label, "created_at": time.strftime("%Y-%m-%d"),
        "models": sorted(models), "omitted": OMITTED,
        "files": {name: {"sha256": _sha256(data), "source_sha256": source}
                  for name, (data, source) in sorted(files.items())},
    }


def export(label: str, files: dict[str, tuple[bytes, str | None]], output: Path) -> list[str]:
    """写出证据包，返回隐私违规列表；有违规时不写任何文件。"""
    violations = privacy_violations(
        {name: data.decode("utf-8", errors="replace") for name, (data, _) in files.items()})
    if violations:
        return violations
    index = build_index(label, files)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"evidence directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name, (data, _) in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (output / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="evidence name, e.g. repair-104-seed42")
    parser.add_argument("--results-dir", help="final-eval output directory of one run")
    parser.add_argument("--checkpoint-dir", help="training checkpoint directory")
    parser.add_argument("--candidates-manifest",
                        help="teacher collection manifest (records the teacher model name)")
    parser.add_argument("--output", default=str(ROOT / "docs" / "evidence"))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.label):
        parser.error("--label may only contain letters, digits, '.', '_' and '-'")
    files = collect(*(Path(path) if path else None for path in
                      (args.results_dir, args.checkpoint_dir, args.candidates_manifest)))
    if not files:
        parser.error("nothing to export: no recognized artifacts found")
    output = Path(args.output) / args.label
    violations = export(args.label, files, output)
    if violations:
        print("隐私检查未通过，未写出任何文件：", file=sys.stderr)
        for violation in violations:
            print("  " + violation, file=sys.stderr)
        raise SystemExit(1)
    index = build_index(args.label, files)
    print(f"已导出 {len(files)} 个文件到 {output}")
    print("模型与版本：" + ("；".join(index["models"]) or "（未找到）"))
    print("提交前请通读一遍，再运行 scripts/verify_evidence.py 重算结论。")


if __name__ == "__main__":
    main()
