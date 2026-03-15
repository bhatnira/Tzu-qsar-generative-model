#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REINVENT = ROOT / ".venv" / "bin" / "reinvent"
PYTHON = ROOT / ".venv" / "bin" / "python"
CONFIGS = ROOT / "reinvent_integration" / "configs"
CONFIGS_STRICT = ROOT / "reinvent_integration" / "configs_strict"
RESULTS = ROOT / "reinvent_integration" / "results"
OUTPUTS = ROOT / "outputs" / "generated"
LOG_DIR = ROOT / "reinvent_integration" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "run_all_generation_strict.log"

MODELS = [
    {"name": "reinvent", "folder": "reinvent"},
    {"name": "reinvent_pubchem", "folder": "reinvent_pubchem"},
    {"name": "pepinvent", "folder": "pepinvent"},
    {"name": "mol2mol_similarity", "folder": "mol2mol/similarity"},
    {"name": "mol2mol_high_similarity", "folder": "mol2mol/high_similarity"},
    {"name": "mol2mol_medium_similarity", "folder": "mol2mol/medium_similarity"},
    {"name": "mol2mol_scaffold", "folder": "mol2mol/scaffold"},
    {"name": "mol2mol_scaffold_generic", "folder": "mol2mol/scaffold_generic"},
    {"name": "mol2mol_mmp", "folder": "mol2mol/mmp"},
    {"name": "libinvent", "folder": "libinvent"},
    {"name": "libinvent_transformer_pubchem", "folder": "libinvent_transformer_pubchem"},
    {"name": "linkinvent", "folder": "linkinvent"},
    {"name": "linkinvent_transformer_pubchem", "folder": "linkinvent_transformer_pubchem"},
]

SAMPLE_SMILES = 2000
RL_BATCH_SIZE = 128
CL_BATCH_SIZE = 128
RL_MIN_STEPS = 40
RL_MAX_STEPS = 80
CL_STAGE1_MIN = 30
CL_STAGE1_MAX = 60
CL_STAGE2_MIN = 40
CL_STAGE2_MAX = 80


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_cmd(cmd: list[str], step: str) -> bool:
    log(f"START {step}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.stdout:
        for line in proc.stdout.splitlines():
            log(f"{step} | {line}")
    if proc.stderr:
        for line in proc.stderr.splitlines():
            log(f"{step} | {line}")
    if proc.returncode != 0:
        log(f"FAIL {step} (exit={proc.returncode})")
        return False
    log(f"DONE {step}")
    return True


def ensure_seed_files() -> None:
    data_dir = ROOT / "reinvent_integration" / "data"
    source = data_dir / "series_ce_unique.smi"
    scaffolds = data_dir / "scaffolds.smi"
    warheads = data_dir / "warheads.smi"
    if not source.exists():
        raise FileNotFoundError(f"Missing required file: {source}")
    if not scaffolds.exists():
        scaffolds.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    if not warheads.exists():
        warheads.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def ensure_dirs() -> None:
    for model in MODELS:
        n = model["name"]
        for variant in ("plain", "rl", "cl", "rl_adme", "cl_adme"):
            (RESULTS / f"{n}_{variant}").mkdir(parents=True, exist_ok=True)


def patch_sampling_config(content: str) -> str:
    return re.sub(r"(?m)^\s*num_smiles\s*=\s*\d+\s*$", f"num_smiles = {SAMPLE_SMILES}", content)


def patch_rl_config(content: str) -> str:
    content = re.sub(r"(?m)^\s*batch_size\s*=\s*\d+\s*$", f"batch_size = {RL_BATCH_SIZE}", content, count=1)
    content = re.sub(r"(?m)^\s*min_steps\s*=\s*\d+\s*$", f"min_steps = {RL_MIN_STEPS}", content, count=1)
    content = re.sub(r"(?m)^\s*max_steps\s*=\s*\d+\s*$", f"max_steps = {RL_MAX_STEPS}", content, count=1)
    return content


def patch_cl_config(content: str) -> str:
    content = re.sub(r"(?m)^\s*batch_size\s*=\s*\d+\s*$", f"batch_size = {CL_BATCH_SIZE}", content, count=1)
    content = re.sub(r"(?m)^\s*min_steps\s*=\s*\d+\s*$", f"min_steps = {CL_STAGE1_MIN}", content, count=1)
    content = re.sub(r"(?m)^\s*max_steps\s*=\s*\d+\s*$", f"max_steps = {CL_STAGE1_MAX}", content, count=1)
    content = re.sub(r"(?m)^\s*min_steps\s*=\s*\d+\s*$", f"min_steps = {CL_STAGE2_MIN}", content, count=1)
    content = re.sub(r"(?m)^\s*max_steps\s*=\s*\d+\s*$", f"max_steps = {CL_STAGE2_MAX}", content, count=1)
    return content


def force_cuda(content: str) -> str:
    return re.sub(r'(?m)^\s*device\s*=\s*"[^"]+"\s*$', 'device = "cuda"', content)


def inject_or_replace_line(content: str, pattern: str, line: str, after_pattern: str | None = None) -> str:
    if re.search(pattern, content, flags=re.MULTILINE):
        return re.sub(pattern, line, content, count=1, flags=re.MULTILINE)
    if after_pattern and re.search(after_pattern, content, flags=re.MULTILINE):
        return re.sub(after_pattern, lambda m: m.group(0) + "\n" + line, content, count=1, flags=re.MULTILINE)
    return content


def model_specific_adjust(name: str, content: str) -> str:
    if name in {"pepinvent", "reinvent_pubchem"}:
        content = inject_or_replace_line(
            content,
            r"^\s*smiles_file\s*=\s*.*$",
            'smiles_file = "reinvent_integration/data/series_ce_unique.smi"',
            after_pattern=r"^\s*model_file\s*=\s*.*$|^\s*agent_file\s*=\s*.*$",
        )

    if name == "reinvent_pubchem":
        if "run_type = \"sampling\"" in content:
            content = inject_or_replace_line(
                content,
                r"^\s*sample_strategy\s*=\s*.*$",
                'sample_strategy = "beamsearch"',
                after_pattern=r"^\s*smiles_file\s*=\s*.*$",
            )
        if "run_type = \"staged_learning\"" in content:
            content = inject_or_replace_line(
                content,
                r"^\s*sample_strategy\s*=\s*.*$",
                'sample_strategy = "multinomial"',
                after_pattern=r"^\s*smiles_file\s*=\s*.*$",
            )
    return content


def build_strict_configs() -> None:
    if CONFIGS_STRICT.exists():
        shutil.rmtree(CONFIGS_STRICT)
    CONFIGS_STRICT.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        n = model["name"]
        files = [
            (f"sample_{n}_plain.toml", patch_sampling_config),
            (f"sample_{n}_rl.toml", patch_sampling_config),
            (f"sample_{n}_cl.toml", patch_sampling_config),
            (f"rl_{n}.toml", patch_rl_config),
            (f"cl_{n}.toml", patch_cl_config),
        ]
        for filename, patcher in files:
            src = CONFIGS / filename
            if not src.exists():
                continue
            text = src.read_text(encoding="utf-8")
            text = force_cuda(text)
            text = patcher(text)
            text = model_specific_adjust(n, text)
            (CONFIGS_STRICT / filename).write_text(text, encoding="utf-8")


def score_variant(name: str, folder: str, tag: str, variant_folder: str) -> bool:
    sample_csv = RESULTS / f"{name}_{tag}" / "samples.csv"
    output_dir = OUTPUTS / folder / variant_folder
    label = f"{name}_{tag}"

    if not sample_csv.exists():
        log(f"SKIP score_{label}: missing strict source {sample_csv}")
        return False

    return run_cmd([
        str(PYTHON),
        "reinvent_integration/score_generated_smiles_adme.py",
        "--input-csv", str(sample_csv),
        "--output-dir", str(output_dir),
        "--label", label,
        "--qsar-model", "reinvent_integration/artifacts/qsar_best_model.joblib",
    ], f"score_{label}")


def clear_scored_outputs() -> None:
    for model in MODELS:
        n = model["name"]
        folder = OUTPUTS / model["folder"]
        for tag, vdir in (("plain", "plain"), ("rl", "reinforcement_learning"), ("cl", "curriculum_learning")):
            out = folder / vdir
            if not out.exists():
                out.mkdir(parents=True, exist_ok=True)
            for suffix in ("generated_scored.csv", "generated_scored.xlsx", "generated_optimal.csv", "generated_optimal.xlsx"):
                f = out / f"{n}_{tag}_{suffix}"
                if f.exists():
                    f.unlink()
            m = out / "manifest.csv"
            if m.exists():
                m.unlink()


def main() -> int:
    if not REINVENT.exists():
        print(f"Missing executable: {REINVENT}", file=sys.stderr)
        return 2

    LOG_FILE.write_text("", encoding="utf-8")
    log("=== STRICT GENERATION START (NO FALLBACKS) ===")

    ensure_seed_files()
    ensure_dirs()
    build_strict_configs()
    clear_scored_outputs()

    failed = 0
    total = 0

    for model in MODELS:
        n = model["name"]
        folder = model["folder"]
        log(f"---- MODEL {n} ----")

        total += 1
        if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_plain_strict.log", str(CONFIGS_STRICT / f"sample_{n}_plain.toml")], f"sample_{n}_plain"):
            failed += 1
        total += 1
        if not score_variant(n, folder, "plain", "plain"):
            failed += 1

        total += 1
        if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_rl_train_strict.log", str(CONFIGS_STRICT / f"rl_{n}.toml")], f"train_{n}_rl"):
            failed += 1
        total += 1
        if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_rl_sample_strict.log", str(CONFIGS_STRICT / f"sample_{n}_rl.toml")], f"sample_{n}_rl"):
            failed += 1
        total += 1
        if not score_variant(n, folder, "rl", "reinforcement_learning"):
            failed += 1

        total += 1
        if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_cl_train_strict.log", str(CONFIGS_STRICT / f"cl_{n}.toml")], f"train_{n}_cl"):
            failed += 1
        total += 1
        if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_cl_sample_strict.log", str(CONFIGS_STRICT / f"sample_{n}_cl.toml")], f"sample_{n}_cl"):
            failed += 1
        total += 1
        if not score_variant(n, folder, "cl", "curriculum_learning"):
            failed += 1

    log(f"=== STRICT GENERATION END: total={total} failed={failed} ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
