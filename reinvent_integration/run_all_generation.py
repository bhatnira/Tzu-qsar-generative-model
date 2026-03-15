#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
REINVENT = ROOT / ".venv" / "bin" / "reinvent"
PYTHON = ROOT / ".venv" / "bin" / "python"

CONFIGS = ROOT / "reinvent_integration" / "configs"
RESULTS = ROOT / "reinvent_integration" / "results"
OUTPUTS = ROOT / "outputs" / "generated"
HYBRID_STAGE_SCRIPT = ROOT / "reinvent_integration" / "run_hybrid_stage_gates.py"
LOG_DIR = ROOT / "reinvent_integration" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "run_all_generation.log"

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


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(cmd: list[str], step: str) -> bool:
    log(f"START {step}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.stdout:
        for ln in proc.stdout.splitlines():
            log(f"{step} | {ln}")
    if proc.stderr:
        for ln in proc.stderr.splitlines():
            log(f"{step} | {ln}")

    if proc.returncode == 0:
        log(f"DONE {step}")
        return True

    log(f"FAIL {step} (exit={proc.returncode})")
    return False


def ensure_seed_files() -> None:
    data_dir = ROOT / "reinvent_integration" / "data"
    source = data_dir / "series_ce_unique.smi"
    scaff = data_dir / "scaffolds.smi"
    war = data_dir / "warheads.smi"

    if not source.exists():
        raise FileNotFoundError(f"Missing required source seed file: {source}")

    if not scaff.exists():
        scaff.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        log(f"Created missing seed file: {scaff}")
    if not war.exists():
        war.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        log(f"Created missing seed file: {war}")


def score_variant(model_name: str, model_folder: str, variant_tag: str, variant_folder: str) -> bool:
    sample_csv = RESULTS / f"{model_name}_{variant_tag}" / "samples.csv"
    outdir = OUTPUTS / model_folder / variant_folder
    label = f"{model_name}_{variant_tag}"

    if not sample_csv.exists():
        log(f"SKIP scoring {label}: sample file missing: {sample_csv}")
        return False

    return run_cmd([
        str(PYTHON),
        "reinvent_integration/score_generated_smiles_adme.py",
        "--input-csv", str(sample_csv),
        "--output-dir", str(outdir),
        "--label", label,
        "--qsar-model", "reinvent_integration/artifacts/qsar_best_model.joblib",
    ], f"score_{label}")


def run_hybrid_stage_gates(top_k: int, max_per_scaffold: int) -> bool:
    if not HYBRID_STAGE_SCRIPT.exists():
        log(f"SKIP hybrid stage gates: script missing: {HYBRID_STAGE_SCRIPT}")
        return False

    return run_cmd([
        str(PYTHON),
        "reinvent_integration/run_hybrid_stage_gates.py",
        "--input-root", str(OUTPUTS),
        "--output-root", str(ROOT / "outputs" / "hybrid_stage"),
        "--top-k", str(max(1, top_k)),
        "--max-per-scaffold", str(max(1, max_per_scaffold)),
    ], "hybrid_stage_gates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all REINVENT generation modes and post-scoring hybrid stage-gate packaging."
    )
    parser.add_argument(
        "--hybrid-top-k",
        type=int,
        default=200,
        help="Number of candidates to keep in Gate 1 shortlist (default: 200).",
    )
    parser.add_argument(
        "--hybrid-max-per-scaffold",
        type=int,
        default=3,
        help="Maximum shortlist entries per Murcko scaffold bucket (default: 3).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not REINVENT.exists():
        print(f"Missing executable: {REINVENT}", file=sys.stderr)
        return 2

    LOG_FILE.write_text("", encoding="utf-8")
    log("=== RUN ALL GENERATION START ===")

    try:
        ensure_seed_files()
    except Exception as exc:
        log(f"FATAL seed preparation failed: {exc}")
        return 2

    total_steps = 0
    failed_steps = 0

    for m in MODELS:
        name = m["name"]
        folder = m["folder"]
        log(f"---- MODEL {name} ----")

        # 1) plain sampling -> score
        total_steps += 1
        ok_plain_sample = run_cmd([
            str(REINVENT), "-l", f"reinvent_integration/logs/{name}_plain.log",
            str(CONFIGS / f"sample_{name}_plain.toml")
        ], f"sample_{name}_plain")
        if not ok_plain_sample:
            failed_steps += 1

        total_steps += 1
        if not score_variant(name, folder, "plain", "plain"):
            failed_steps += 1

        # 2) RL train -> RL sample -> RL score
        total_steps += 1
        ok_rl_train = run_cmd([
            str(REINVENT), "-l", f"reinvent_integration/logs/{name}_rl_train.log",
            str(CONFIGS / f"rl_{name}.toml")
        ], f"train_{name}_rl")
        if not ok_rl_train:
            failed_steps += 1

        total_steps += 1
        ok_rl_sample = run_cmd([
            str(REINVENT), "-l", f"reinvent_integration/logs/{name}_rl_sample.log",
            str(CONFIGS / f"sample_{name}_rl.toml")
        ], f"sample_{name}_rl")
        if not ok_rl_sample:
            failed_steps += 1

        total_steps += 1
        if not score_variant(name, folder, "rl", "reinforcement_learning"):
            failed_steps += 1

        # 3) CL train -> CL sample -> CL score
        total_steps += 1
        ok_cl_train = run_cmd([
            str(REINVENT), "-l", f"reinvent_integration/logs/{name}_cl_train.log",
            str(CONFIGS / f"cl_{name}.toml")
        ], f"train_{name}_cl")
        if not ok_cl_train:
            failed_steps += 1

        total_steps += 1
        ok_cl_sample = run_cmd([
            str(REINVENT), "-l", f"reinvent_integration/logs/{name}_cl_sample.log",
            str(CONFIGS / f"sample_{name}_cl.toml")
        ], f"sample_{name}_cl")
        if not ok_cl_sample:
            failed_steps += 1

        total_steps += 1
        if not score_variant(name, folder, "cl", "curriculum_learning"):
            failed_steps += 1

    total_steps += 1
    if not run_hybrid_stage_gates(
        top_k=args.hybrid_top_k,
        max_per_scaffold=args.hybrid_max_per_scaffold,
    ):
        failed_steps += 1

    log(f"=== RUN ALL GENERATION END: total_steps={total_steps} failed_steps={failed_steps} ===")
    return 0 if failed_steps == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
