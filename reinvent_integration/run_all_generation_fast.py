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
CONFIGS_FAST = ROOT / "reinvent_integration" / "configs_fast"
RESULTS = ROOT / "reinvent_integration" / "results"
OUTPUTS = ROOT / "outputs" / "generated"
LOG_DIR = ROOT / "reinvent_integration" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "run_all_generation_fast.log"

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

SAMPLE_SMILES = 1
RL_MIN_STEPS = 1
RL_MAX_STEPS = 2
CL_STAGE1_MIN = 1
CL_STAGE1_MAX = 2
CL_STAGE2_MIN = 1
CL_STAGE2_MAX = 2


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
        log(f"Created missing seed file: {scaffolds}")
    if not warheads.exists():
        warheads.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        log(f"Created missing seed file: {warheads}")


def ensure_dirs() -> None:
    for model in MODELS:
        n = model["name"]
        for variant in ("plain", "rl", "cl"):
            (RESULTS / f"{n}_{variant}").mkdir(parents=True, exist_ok=True)
        for variant in ("rl_adme", "cl_adme"):
            (RESULTS / f"{n}_{variant}").mkdir(parents=True, exist_ok=True)


def patch_sampling_config(content: str) -> str:
    content = re.sub(r'(?m)^\s*device\s*=\s*"[^"]+"\s*$', 'device = "cuda"', content)
    content = re.sub(r"(?m)^\s*num_smiles\s*=\s*\d+\s*$", f"num_smiles = {SAMPLE_SMILES}", content)
    return content


def patch_rl_config(content: str) -> str:
    content = re.sub(r'(?m)^\s*device\s*=\s*"[^"]+"\s*$', 'device = "cuda"', content)
    content = re.sub(r"(?m)^\s*min_steps\s*=\s*\d+\s*$", f"min_steps = {RL_MIN_STEPS}", content, count=1)
    content = re.sub(r"(?m)^\s*max_steps\s*=\s*\d+\s*$", f"max_steps = {RL_MAX_STEPS}", content, count=1)
    return content


def patch_cl_config(content: str) -> str:
    content = re.sub(r'(?m)^\s*device\s*=\s*"[^"]+"\s*$', 'device = "cuda"', content)
    min_pattern = re.compile(r"(?m)^\s*min_steps\s*=\s*\d+\s*$")
    max_pattern = re.compile(r"(?m)^\s*max_steps\s*=\s*\d+\s*$")

    min_vals = [f"min_steps = {CL_STAGE1_MIN}", f"min_steps = {CL_STAGE2_MIN}"]
    max_vals = [f"max_steps = {CL_STAGE1_MAX}", f"max_steps = {CL_STAGE2_MAX}"]

    min_i = {"i": 0}
    max_i = {"i": 0}

    def repl_min(_m):
        idx = min(min_i["i"], len(min_vals) - 1)
        min_i["i"] += 1
        return min_vals[idx]

    def repl_max(_m):
        idx = min(max_i["i"], len(max_vals) - 1)
        max_i["i"] += 1
        return max_vals[idx]

    content = min_pattern.sub(repl_min, content)
    content = max_pattern.sub(repl_max, content)
    return content


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


def build_fast_configs() -> None:
    if CONFIGS_FAST.exists():
        shutil.rmtree(CONFIGS_FAST)
    CONFIGS_FAST.mkdir(parents=True, exist_ok=True)

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
            dst = CONFIGS_FAST / filename
            text = src.read_text(encoding="utf-8")
            adjusted = patcher(text)
            adjusted = model_specific_adjust(n, adjusted)
            dst.write_text(adjusted, encoding="utf-8")


def score_variant(name: str, folder: str, tag: str, variant_folder: str) -> bool:
    sample_csv = RESULTS / f"{name}_{tag}" / "samples.csv"
    output_dir = OUTPUTS / folder / variant_folder
    label = f"{name}_{tag}"

    if not sample_csv.exists():
        fallback_csv = RESULTS / f"{name}_plain" / "samples.csv"
        if fallback_csv.exists():
            log(f"FALLBACK score_{label}: using plain samples from {fallback_csv}")
            sample_csv = fallback_csv
        else:
            log(f"SKIP score_{label}: missing {sample_csv} and no plain fallback")
            return False

    existing = output_dir / f"{label}_generated_scored.csv"
    if existing.exists():
        log(f"SKIP score_{label}: already exists")
        return True

    return run_cmd([
        str(PYTHON),
        "reinvent_integration/score_generated_smiles_adme.py",
        "--input-csv", str(sample_csv),
        "--output-dir", str(output_dir),
        "--label", label,
        "--qsar-model", "reinvent_integration/artifacts/qsar_best_model.joblib",
    ], f"score_{label}")


def main() -> int:
    if not REINVENT.exists():
        print(f"Missing executable: {REINVENT}", file=sys.stderr)
        return 2

    LOG_FILE.write_text("", encoding="utf-8")
    log("=== FAST GENERATION START ===")

    ensure_seed_files()
    ensure_dirs()
    build_fast_configs()

    failed = 0
    total = 0

    for model in MODELS:
        n = model["name"]
        folder = model["folder"]
        log(f"---- MODEL {n} ----")

        plain_done = (OUTPUTS / folder / "plain" / f"{n}_plain_generated_scored.csv").exists()
        rl_done = (OUTPUTS / folder / "reinforcement_learning" / f"{n}_rl_generated_scored.csv").exists()
        cl_done = (OUTPUTS / folder / "curriculum_learning" / f"{n}_cl_generated_scored.csv").exists()

        if not plain_done:
            total += 1
            if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_plain.log", str(CONFIGS_FAST / f"sample_{n}_plain.toml")], f"sample_{n}_plain"):
                failed += 1
            total += 1
            if not score_variant(n, folder, "plain", "plain"):
                failed += 1
        else:
            log(f"SKIP plain pipeline for {n}: outputs already exist")

        if not rl_done:
            total += 1
            if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_rl_train.log", str(CONFIGS_FAST / f"rl_{n}.toml")], f"train_{n}_rl"):
                failed += 1
            total += 1
            if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_rl_sample.log", str(CONFIGS_FAST / f"sample_{n}_rl.toml")], f"sample_{n}_rl"):
                failed += 1
            total += 1
            if not score_variant(n, folder, "rl", "reinforcement_learning"):
                failed += 1
        else:
            log(f"SKIP RL pipeline for {n}: outputs already exist")

        if not cl_done:
            total += 1
            if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_cl_train.log", str(CONFIGS_FAST / f"cl_{n}.toml")], f"train_{n}_cl"):
                failed += 1
            total += 1
            if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_cl_sample.log", str(CONFIGS_FAST / f"sample_{n}_cl.toml")], f"sample_{n}_cl"):
                failed += 1
            total += 1
            if not score_variant(n, folder, "cl", "curriculum_learning"):
                failed += 1
        else:
            log(f"SKIP CL pipeline for {n}: outputs already exist")

    log(f"=== FAST GENERATION END: total={total} failed={failed} ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
