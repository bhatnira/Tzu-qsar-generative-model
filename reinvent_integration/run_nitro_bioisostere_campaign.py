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
CONFIGS_CAMPAIGN = ROOT / "reinvent_integration" / "configs_nitro_campaign"
RESULTS = ROOT / "reinvent_integration" / "results"
OUTPUTS = ROOT / "outputs" / "generated" / "nitro_bioisostere_campaign"
LOG_DIR = ROOT / "reinvent_integration" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "run_nitro_bioisostere_campaign.log"

SEED_SMILES = [
    "CCCCCCC#CC1=CN=CC(N2CCC3(CN(CC4=CC([N+]([O-])=O)=CC([N+]([O-])=O)=C4)C3)C2)=C1",
    "CCCCCCC#CC1=CN=CC(N2CCC3(CN(C(C4=CC([N+]([O-])=O)=CC([N+]([O-])=O)=C4)=O)C3)C2)=C1",
    "CCCCCCC#CC1=CN=CC(N2CCC3(CN(C(C(C=C(OCO4)C4=C5)=C5[N+]([O-])=O)=O)C3)C2)=C1",
]

MODELS = [
    {"name": "mol2mol_mmp", "folder": "mol2mol_mmp"},
    {"name": "mol2mol_similarity", "folder": "mol2mol_similarity"},
    {"name": "mol2mol_scaffold", "folder": "mol2mol_scaffold"},
    {"name": "mol2mol_scaffold_generic", "folder": "mol2mol_scaffold_generic"},
]

NITRO_SMARTS = "[N+](=O)[O-]"
SAMPLE_SMILES = 2000
RL_MIN_STEPS = 20
RL_MAX_STEPS = 40
CL_STAGE1_MIN = 20
CL_STAGE1_MAX = 35
CL_STAGE2_MIN = 25
CL_STAGE2_MAX = 45


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


def ensure_seed_file() -> Path:
    data_dir = ROOT / "reinvent_integration" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    seed_file = data_dir / "nitro_leads.smi"
    seed_file.write_text("\n".join(SEED_SMILES) + "\n", encoding="utf-8")
    return seed_file


def ensure_dirs() -> None:
    for model in MODELS:
        n = model["name"]
        for variant in ("plain", "rl", "cl", "rl_adme", "cl_adme"):
            (RESULTS / f"{n}_{variant}").mkdir(parents=True, exist_ok=True)
        for folder in ("plain", "reinforcement_learning", "curriculum_learning"):
            (OUTPUTS / model["folder"] / folder).mkdir(parents=True, exist_ok=True)


def force_cuda(content: str) -> str:
    return re.sub(r'(?m)^\s*device\s*=\s*"[^"]+"\s*$', 'device = "cuda"', content)


def set_smiles_file(content: str, seed_file: Path) -> str:
    rel = seed_file.relative_to(ROOT).as_posix()
    return re.sub(r'(?m)^\s*smiles_file\s*=\s*"[^"]+"\s*$', f'smiles_file = "{rel}"', content)


def set_num_smiles(content: str) -> str:
    return re.sub(r"(?m)^\s*num_smiles\s*=\s*\d+\s*$", f"num_smiles = {SAMPLE_SMILES}", content)


def set_rl_steps(content: str) -> str:
    content = re.sub(r"(?m)^\s*min_steps\s*=\s*\d+\s*$", f"min_steps = {RL_MIN_STEPS}", content, count=1)
    content = re.sub(r"(?m)^\s*max_steps\s*=\s*\d+\s*$", f"max_steps = {RL_MAX_STEPS}", content, count=1)
    return content


def set_cl_steps(content: str) -> str:
    content = re.sub(r"(?m)^\s*min_steps\s*=\s*\d+\s*$", f"min_steps = {CL_STAGE1_MIN}", content, count=1)
    content = re.sub(r"(?m)^\s*max_steps\s*=\s*\d+\s*$", f"max_steps = {CL_STAGE1_MAX}", content, count=1)
    content = re.sub(r"(?m)^\s*min_steps\s*=\s*\d+\s*$", f"min_steps = {CL_STAGE2_MIN}", content, count=1)
    content = re.sub(r"(?m)^\s*max_steps\s*=\s*\d+\s*$", f"max_steps = {CL_STAGE2_MAX}", content, count=1)
    return content


def add_nitro_penalty_to_scorer(content: str) -> str:
    target = f"--disallow-smarts {NITRO_SMARTS} --disallow-penalty 1.0"
    if target in content:
        return content

    pattern = re.compile(r'(?m)^(\s*params\.args\s*=\s*")([^"]*reinvent_qsar_adme_score\.py[^"]*)("\s*)$')

    def _append_arg(match: re.Match) -> str:
        prefix, body, suffix = match.groups()
        return f"{prefix}{body} {target}{suffix}"

    return pattern.sub(_append_arg, content)


def build_campaign_configs(seed_file: Path) -> None:
    if CONFIGS_CAMPAIGN.exists():
        shutil.rmtree(CONFIGS_CAMPAIGN)
    CONFIGS_CAMPAIGN.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        n = model["name"]
        files = [
            (f"sample_{n}_plain.toml", "sample"),
            (f"sample_{n}_rl.toml", "sample"),
            (f"sample_{n}_cl.toml", "sample"),
            (f"rl_{n}.toml", "rl"),
            (f"cl_{n}.toml", "cl"),
        ]
        for filename, kind in files:
            src = CONFIGS / filename
            if not src.exists():
                continue
            text = src.read_text(encoding="utf-8")
            text = force_cuda(text)
            text = set_smiles_file(text, seed_file)
            if kind == "sample":
                text = set_num_smiles(text)
            elif kind == "rl":
                text = set_rl_steps(text)
                text = add_nitro_penalty_to_scorer(text)
            elif kind == "cl":
                text = set_cl_steps(text)
                text = add_nitro_penalty_to_scorer(text)
            (CONFIGS_CAMPAIGN / filename).write_text(text, encoding="utf-8")


def score_variant(name: str, folder: str, tag: str, variant_folder: str) -> bool:
    sample_csv = RESULTS / f"{name}_{tag}" / "samples.csv"
    output_dir = OUTPUTS / folder / variant_folder
    label = f"{name}_{tag}"

    if not sample_csv.exists():
        log(f"SKIP score_{label}: missing source {sample_csv}")
        return False

    return run_cmd([
        str(PYTHON),
        "reinvent_integration/score_generated_smiles_adme.py",
        "--input-csv", str(sample_csv),
        "--output-dir", str(output_dir),
        "--label", label,
        "--qsar-model", "reinvent_integration/artifacts/qsar_best_model.joblib",
        "--exclude-smarts", NITRO_SMARTS,
    ], f"score_{label}")


def main() -> int:
    if not REINVENT.exists():
        print(f"Missing executable: {REINVENT}", file=sys.stderr)
        return 2

    LOG_FILE.write_text("", encoding="utf-8")
    log("=== NITRO BIOISOSTERE CAMPAIGN START ===")

    seed_file = ensure_seed_file()
    ensure_dirs()
    build_campaign_configs(seed_file)

    failed = 0
    total = 0

    for model in MODELS:
        n = model["name"]
        folder = model["folder"]
        log(f"---- MODEL {n} ----")

        total += 1
        if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_plain_nitro.log", str(CONFIGS_CAMPAIGN / f"sample_{n}_plain.toml")], f"sample_{n}_plain"):
            failed += 1
        total += 1
        if not score_variant(n, folder, "plain", "plain"):
            failed += 1

        total += 1
        if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_rl_train_nitro.log", str(CONFIGS_CAMPAIGN / f"rl_{n}.toml")], f"train_{n}_rl"):
            failed += 1
        total += 1
        if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_rl_sample_nitro.log", str(CONFIGS_CAMPAIGN / f"sample_{n}_rl.toml")], f"sample_{n}_rl"):
            failed += 1
        total += 1
        if not score_variant(n, folder, "rl", "reinforcement_learning"):
            failed += 1

        total += 1
        if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_cl_train_nitro.log", str(CONFIGS_CAMPAIGN / f"cl_{n}.toml")], f"train_{n}_cl"):
            failed += 1
        total += 1
        if not run_cmd([str(REINVENT), "-l", f"reinvent_integration/logs/{n}_cl_sample_nitro.log", str(CONFIGS_CAMPAIGN / f"sample_{n}_cl.toml")], f"sample_{n}_cl"):
            failed += 1
        total += 1
        if not score_variant(n, folder, "cl", "curriculum_learning"):
            failed += 1

    log(f"=== NITRO BIOISOSTERE CAMPAIGN END: total={total} failed={failed} ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
