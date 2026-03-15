#!/usr/bin/env python
"""
Scaffold the full outputs/generated directory tree and all TOML configs for each model
(plain sampling, RL training, CL training, sample-after-RL, sample-after-CL).

Run from project root:
    python reinvent_integration/scaffold_all_configs.py
"""

from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
RI   = ROOT / "reinvent_integration"
VENV_PY = str(ROOT / ".venv/bin/python")

# ---------------------------------------------------------------------------
# Model registry
# Each entry: (name, prior_key, model_type, output_folder)
#   model_type: 'reinvent' | 'mol2mol' | 'libinvent' | 'linkinvent' | 'pepinvent'
#   output_folder: path under outputs/generated/
# ---------------------------------------------------------------------------
MODELS = [
    # ── REINVENT ──────────────────────────────────────────────────────────
    dict(name="reinvent",
         prior="reinvent.prior",
         mtype="reinvent",
         folder="reinvent"),

    dict(name="reinvent_pubchem",
         prior="pubchem_ecfp4_with_count_with_rank_reinvent4_dict_voc.prior",
         mtype="reinvent",
         folder="reinvent_pubchem"),

    # ── PepINVENT ─────────────────────────────────────────────────────────
    dict(name="pepinvent",
         prior="pepinvent.prior",
         mtype="pepinvent",
         folder="pepinvent"),

    # ── Mol2Mol ───────────────────────────────────────────────────────────
    dict(name="mol2mol_similarity",
         prior="mol2mol_similarity.prior",
         mtype="mol2mol",
         folder="mol2mol/similarity"),

    dict(name="mol2mol_high_similarity",
         prior="mol2mol_high_similarity.prior",
         mtype="mol2mol",
         folder="mol2mol/high_similarity"),

    dict(name="mol2mol_medium_similarity",
         prior="mol2mol_medium_similarity.prior",
         mtype="mol2mol",
         folder="mol2mol/medium_similarity"),

    dict(name="mol2mol_scaffold",
         prior="mol2mol_scaffold.prior",
         mtype="mol2mol",
         folder="mol2mol/scaffold"),

    dict(name="mol2mol_scaffold_generic",
         prior="mol2mol_scaffold_generic.prior",
         mtype="mol2mol",
         folder="mol2mol/scaffold_generic"),

    dict(name="mol2mol_mmp",
         prior="mol2mol_mmp.prior",
         mtype="mol2mol",
         folder="mol2mol/mmp"),

    # ── LibINVENT ─────────────────────────────────────────────────────────
    dict(name="libinvent",
         prior="libinvent.prior",
         mtype="libinvent",
         folder="libinvent"),

    dict(name="libinvent_transformer_pubchem",
         prior="libinvent_transformer_pubchem.prior",
         mtype="libinvent",
         folder="libinvent_transformer_pubchem"),

    # ── LinkINVENT ────────────────────────────────────────────────────────
    dict(name="linkinvent",
         prior="linkinvent.prior",
         mtype="linkinvent",
         folder="linkinvent"),

    dict(name="linkinvent_transformer_pubchem",
         prior="linkinvent_transformer_pubchem.prior",
         mtype="linkinvent",
         folder="linkinvent_transformer_pubchem"),
]

# ---------------------------------------------------------------------------
# Helper paths
# ---------------------------------------------------------------------------
SERIES_CE_SMI   = "reinvent_integration/data/series_ce_unique.smi"
SCAFFOLDS_SMI   = "reinvent_integration/data/scaffolds.smi"
WARHEADS_SMI    = "reinvent_integration/data/warheads.smi"
QSAR_MODEL      = "reinvent_integration/artifacts/qsar_best_model.joblib"
SCORE_SCRIPT    = "reinvent_integration/reinvent_qsar_adme_score.py"

def prior_path(prior_file):
    return f"reinvent_integration/priors/{prior_file}"

def results_dir(name, variant):
    """Return path string inside results/ for a given model+variant."""
    return f"reinvent_integration/results/{name}_{variant}"

def outputs_dir(folder, variant):
    return f"outputs/generated/{folder}/{variant}"

# ---------------------------------------------------------------------------
# TOML generators
# ---------------------------------------------------------------------------

def sampling_toml_reinvent(m, variant="plain"):
    """Plain / post-RL / post-CL sampling for REINVENT / PepINVENT."""
    if variant == "plain":
        model_file = prior_path(m["prior"])
    elif variant == "rl":
        model_file = f"{results_dir(m['name'], 'rl_adme')}/{m['name']}_rl.chkpt"
    else:  # cl
        model_file = f"{results_dir(m['name'], 'cl_adme')}/{m['name']}_cl_stage2.chkpt"
    out_csv = f"{results_dir(m['name'], variant)}/samples.csv"
    return textwrap.dedent(f"""\
        run_type = "sampling"
        device = "cpu"

        [parameters]
        model_file = "{model_file}"
        output_file = "{out_csv}"
        num_smiles = 2000
        unique_molecules = false
        randomize_smiles = true
        """)

def sampling_toml_mol2mol(m, variant="plain"):
    if variant == "plain":
        model_file = prior_path(m["prior"])
    elif variant == "rl":
        model_file = f"{results_dir(m['name'], 'rl_adme')}/{m['name']}_rl.chkpt"
    else:
        model_file = f"{results_dir(m['name'], 'cl_adme')}/{m['name']}_cl_stage2.chkpt"
    out_csv = f"{results_dir(m['name'], variant)}/samples.csv"
    return textwrap.dedent(f"""\
        run_type = "sampling"
        device = "cpu"

        [parameters]
        model_file = "{model_file}"
        smiles_file = "{SERIES_CE_SMI}"
        sample_strategy = "beamsearch"
        temperature = 1.0
        output_file = "{out_csv}"
        num_smiles = 500
        unique_molecules = false
        randomize_smiles = true
        """)

def sampling_toml_libinvent(m, variant="plain"):
    if variant == "plain":
        model_file = prior_path(m["prior"])
    elif variant == "rl":
        model_file = f"{results_dir(m['name'], 'rl_adme')}/{m['name']}_rl.chkpt"
    else:
        model_file = f"{results_dir(m['name'], 'cl_adme')}/{m['name']}_cl_stage2.chkpt"
    out_csv = f"{results_dir(m['name'], variant)}/samples.csv"
    return textwrap.dedent(f"""\
        run_type = "sampling"
        device = "cpu"

        [parameters]
        model_file = "{model_file}"
        smiles_file = "{SCAFFOLDS_SMI}"
        output_file = "{out_csv}"
        num_smiles = 500
        unique_molecules = false
        randomize_smiles = true
        """)

def sampling_toml_linkinvent(m, variant="plain"):
    if variant == "plain":
        model_file = prior_path(m["prior"])
    elif variant == "rl":
        model_file = f"{results_dir(m['name'], 'rl_adme')}/{m['name']}_rl.chkpt"
    else:
        model_file = f"{results_dir(m['name'], 'cl_adme')}/{m['name']}_cl_stage2.chkpt"
    out_csv = f"{results_dir(m['name'], variant)}/samples.csv"
    return textwrap.dedent(f"""\
        run_type = "sampling"
        device = "cpu"

        [parameters]
        model_file = "{model_file}"
        smiles_file = "{WARHEADS_SMI}"
        output_file = "{out_csv}"
        num_smiles = 500
        unique_molecules = false
        randomize_smiles = true
        """)

def get_sampling_toml(m, variant):
    dispatch = {
        "reinvent":   sampling_toml_reinvent,
        "pepinvent":  sampling_toml_reinvent,
        "mol2mol":    sampling_toml_mol2mol,
        "libinvent":  sampling_toml_libinvent,
        "linkinvent": sampling_toml_linkinvent,
    }
    return dispatch[m["mtype"]](m, variant)

# ---------------------------------------------------------------------------
# RL Training TOMLs
# ---------------------------------------------------------------------------

def rl_toml_reinvent(m):
    label  = m["name"].upper().replace("_", " ")
    chkpt  = f"{results_dir(m['name'], 'rl_adme')}/{m['name']}_rl.chkpt"
    return textwrap.dedent(f"""\
        run_type = "staged_learning"
        device = "cpu"

        [parameters]
        prior_file = "{prior_path(m['prior'])}"
        agent_file = "{prior_path(m['prior'])}"
        batch_size = 64
        unique_sequences = true
        randomize_smiles = false
        summary_csv_prefix = "{m['name']}_rl_adme"

        [learning_strategy]
        type = "dap"
        sigma = 64
        rate = 0.00005

        [diversity_filter]
        type = "IdenticalMurckoScaffold"
        bucket_size = 25
        minscore = 0.45
        minsimilarity = 0.4
        penalty_multiplier = 0.5

        [[stage]]
        chkpt_file = "{chkpt}"
        termination = "simple"
        min_steps = 20
        max_steps = 40

        [stage.scoring]
        type = "custom_sum"

        [[stage.scoring.component]]
        [stage.scoring.component.ExternalProcess]
        params.executable = "{VENV_PY}"
        params.args = "{SCORE_SCRIPT} --qsar-model {QSAR_MODEL} --target-pic50 5.30103 --clogp-low 1.0 --clogp-high 3.5 --logs-threshold -6.0 --microsome-clearance-max 70 --hepatocyte-clearance-max 80 --half-life-min 40 --qsar-weight 0.40 --clogp-weight 0.15 --solubility-weight 0.15 --microsome-weight 0.10 --hepatocyte-weight 0.10 --half-life-weight 0.10 --property-name qsar_adme_score"

        [[stage.scoring.component.ExternalProcess.endpoint]]
        name = "{label}_RL_ADME"
        weight = 1.0
        [stage.scoring.component.ExternalProcess.endpoint.params]
        property = "qsar_adme_score"
        """)

def rl_toml_mol2mol(m):
    label  = m["name"].upper().replace("_", " ")
    chkpt  = f"{results_dir(m['name'], 'rl_adme')}/{m['name']}_rl.chkpt"
    return textwrap.dedent(f"""\
        run_type = "staged_learning"
        device = "cpu"

        [parameters]
        prior_file = "{prior_path(m['prior'])}"
        agent_file = "{prior_path(m['prior'])}"
        smiles_file = "{SERIES_CE_SMI}"
        batch_size = 32
        unique_sequences = true
        randomize_smiles = false
        sample_strategy = "multinomial"
        summary_csv_prefix = "{m['name']}_rl_adme"

        [learning_strategy]
        type = "dap"
        sigma = 64
        rate = 0.00005

        [diversity_filter]
        type = "IdenticalMurckoScaffold"
        bucket_size = 20
        minscore = 0.45
        minsimilarity = 0.4
        penalty_multiplier = 0.5

        [[stage]]
        chkpt_file = "{chkpt}"
        termination = "simple"
        min_steps = 15
        max_steps = 30

        [stage.scoring]
        type = "custom_sum"

        [[stage.scoring.component]]
        [stage.scoring.component.ExternalProcess]
        params.executable = "{VENV_PY}"
        params.args = "{SCORE_SCRIPT} --qsar-model {QSAR_MODEL} --target-pic50 5.30103 --clogp-low 1.0 --clogp-high 3.5 --logs-threshold -6.0 --microsome-clearance-max 70 --hepatocyte-clearance-max 80 --half-life-min 40 --qsar-weight 0.40 --clogp-weight 0.15 --solubility-weight 0.15 --microsome-weight 0.10 --hepatocyte-weight 0.10 --half-life-weight 0.10 --property-name qsar_adme_score"

        [[stage.scoring.component.ExternalProcess.endpoint]]
        name = "{label}_RL_ADME"
        weight = 1.0
        [stage.scoring.component.ExternalProcess.endpoint.params]
        property = "qsar_adme_score"
        """)

def rl_toml_libinvent(m):
    label  = m["name"].upper().replace("_", " ")
    chkpt  = f"{results_dir(m['name'], 'rl_adme')}/{m['name']}_rl.chkpt"
    return textwrap.dedent(f"""\
        run_type = "staged_learning"
        device = "cpu"

        [parameters]
        prior_file = "{prior_path(m['prior'])}"
        agent_file = "{prior_path(m['prior'])}"
        smiles_file = "{SCAFFOLDS_SMI}"
        batch_size = 32
        unique_sequences = true
        randomize_smiles = false
        summary_csv_prefix = "{m['name']}_rl_adme"

        [learning_strategy]
        type = "dap"
        sigma = 64
        rate = 0.00005

        [diversity_filter]
        type = "IdenticalMurckoScaffold"
        bucket_size = 20
        minscore = 0.45
        minsimilarity = 0.4
        penalty_multiplier = 0.5

        [[stage]]
        chkpt_file = "{chkpt}"
        termination = "simple"
        min_steps = 15
        max_steps = 30

        [stage.scoring]
        type = "custom_sum"

        [[stage.scoring.component]]
        [stage.scoring.component.ExternalProcess]
        params.executable = "{VENV_PY}"
        params.args = "{SCORE_SCRIPT} --qsar-model {QSAR_MODEL} --target-pic50 5.30103 --clogp-low 1.0 --clogp-high 3.5 --logs-threshold -6.0 --microsome-clearance-max 70 --hepatocyte-clearance-max 80 --half-life-min 40 --qsar-weight 0.40 --clogp-weight 0.15 --solubility-weight 0.15 --microsome-weight 0.10 --hepatocyte-weight 0.10 --half-life-weight 0.10 --property-name qsar_adme_score"

        [[stage.scoring.component.ExternalProcess.endpoint]]
        name = "{label}_RL_ADME"
        weight = 1.0
        [stage.scoring.component.ExternalProcess.endpoint.params]
        property = "qsar_adme_score"
        """)

def rl_toml_linkinvent(m):
    label  = m["name"].upper().replace("_", " ")
    chkpt  = f"{results_dir(m['name'], 'rl_adme')}/{m['name']}_rl.chkpt"
    return textwrap.dedent(f"""\
        run_type = "staged_learning"
        device = "cpu"

        [parameters]
        prior_file = "{prior_path(m['prior'])}"
        agent_file = "{prior_path(m['prior'])}"
        smiles_file = "{WARHEADS_SMI}"
        batch_size = 32
        unique_sequences = true
        randomize_smiles = false
        summary_csv_prefix = "{m['name']}_rl_adme"

        [learning_strategy]
        type = "dap"
        sigma = 64
        rate = 0.00005

        [diversity_filter]
        type = "IdenticalMurckoScaffold"
        bucket_size = 20
        minscore = 0.45
        minsimilarity = 0.4
        penalty_multiplier = 0.5

        [[stage]]
        chkpt_file = "{chkpt}"
        termination = "simple"
        min_steps = 15
        max_steps = 30

        [stage.scoring]
        type = "custom_sum"

        [[stage.scoring.component]]
        [stage.scoring.component.ExternalProcess]
        params.executable = "{VENV_PY}"
        params.args = "{SCORE_SCRIPT} --qsar-model {QSAR_MODEL} --target-pic50 5.30103 --clogp-low 1.0 --clogp-high 3.5 --logs-threshold -6.0 --microsome-clearance-max 70 --hepatocyte-clearance-max 80 --half-life-min 40 --qsar-weight 0.40 --clogp-weight 0.15 --solubility-weight 0.15 --microsome-weight 0.10 --hepatocyte-weight 0.10 --half-life-weight 0.10 --property-name qsar_adme_score"

        [[stage.scoring.component.ExternalProcess.endpoint]]
        name = "{label}_RL_ADME"
        weight = 1.0
        [stage.scoring.component.ExternalProcess.endpoint.params]
        property = "qsar_adme_score"
        """)

def get_rl_toml(m):
    dispatch = {
        "reinvent":   rl_toml_reinvent,
        "pepinvent":  rl_toml_reinvent,
        "mol2mol":    rl_toml_mol2mol,
        "libinvent":  rl_toml_libinvent,
        "linkinvent": rl_toml_linkinvent,
    }
    return dispatch[m["mtype"]](m)

# ---------------------------------------------------------------------------
# CL Training TOMLs (two stages)
# ---------------------------------------------------------------------------

def cl_toml_base(m, smiles_extra="", label_prefix=""):
    """Build CL two-stage TOML. smiles_extra adds smiles_file lines."""
    n = m["name"]
    p = prior_path(m["prior"])
    chkpt1 = f"{results_dir(n, 'cl_adme')}/{n}_cl_stage1.chkpt"
    chkpt2 = f"{results_dir(n, 'cl_adme')}/{n}_cl_stage2.chkpt"
    label  = (label_prefix or m["name"].upper().replace("_", " "))
    return textwrap.dedent(f"""\
        run_type = "staged_learning"
        device = "cpu"

        [parameters]
        prior_file = "{p}"
        agent_file = "{p}"
        {smiles_extra}batch_size = 64
        unique_sequences = true
        randomize_smiles = false
        summary_csv_prefix = "{n}_cl_adme"

        [learning_strategy]
        type = "dap"
        sigma = 64
        rate = 0.00005

        [diversity_filter]
        type = "IdenticalMurckoScaffold"
        bucket_size = 25
        minscore = 0.40
        minsimilarity = 0.4
        penalty_multiplier = 0.5

        # ── Stage 1 : Relaxed ADME constraints ──────────────────────────
        [[stage]]
        chkpt_file = "{chkpt1}"
        termination = "simple"
        min_steps = 15
        max_steps = 25

        [stage.scoring]
        type = "custom_sum"

        [[stage.scoring.component]]
        [stage.scoring.component.ExternalProcess]
        params.executable = "{VENV_PY}"
        params.args = "{SCORE_SCRIPT} --qsar-model {QSAR_MODEL} --target-pic50 5.0 --clogp-low 1.0 --clogp-high 4.0 --logs-threshold -6.3 --microsome-clearance-max 85 --hepatocyte-clearance-max 95 --half-life-min 35 --qsar-weight 0.35 --clogp-weight 0.15 --solubility-weight 0.15 --microsome-weight 0.10 --hepatocyte-weight 0.10 --half-life-weight 0.15 --property-name qsar_adme_score"

        [[stage.scoring.component.ExternalProcess.endpoint]]
        name = "{label}_CL_Stage1"
        weight = 1.0
        [stage.scoring.component.ExternalProcess.endpoint.params]
        property = "qsar_adme_score"

        # ── Stage 2 : Tightened ADME constraints ────────────────────────
        [[stage]]
        chkpt_file = "{chkpt2}"
        termination = "simple"
        min_steps = 20
        max_steps = 35

        [stage.scoring]
        type = "custom_sum"

        [[stage.scoring.component]]
        [stage.scoring.component.ExternalProcess]
        params.executable = "{VENV_PY}"
        params.args = "{SCORE_SCRIPT} --qsar-model {QSAR_MODEL} --target-pic50 5.30103 --clogp-low 1.0 --clogp-high 3.5 --logs-threshold -6.0 --microsome-clearance-max 70 --hepatocyte-clearance-max 80 --half-life-min 40 --qsar-weight 0.40 --clogp-weight 0.15 --solubility-weight 0.15 --microsome-weight 0.10 --hepatocyte-weight 0.10 --half-life-weight 0.10 --property-name qsar_adme_score"

        [[stage.scoring.component.ExternalProcess.endpoint]]
        name = "{label}_CL_Stage2"
        weight = 1.0
        [stage.scoring.component.ExternalProcess.endpoint.params]
        property = "qsar_adme_score"
        """)

def get_cl_toml(m):
    extras = {
        "mol2mol":    f'smiles_file = "{SERIES_CE_SMI}"\nsample_strategy = "multinomial"\n',
        "libinvent":  f'smiles_file = "{SCAFFOLDS_SMI}"\n',
        "linkinvent": f'smiles_file = "{WARHEADS_SMI}"\n',
        "reinvent":   "",
        "pepinvent":  "",
    }
    return cl_toml_base(m, smiles_extra=extras[m["mtype"]])

# ---------------------------------------------------------------------------
# Main scaffold routine
# ---------------------------------------------------------------------------

def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  wrote  {path.relative_to(ROOT)}")

def main():
    cfg_dir  = RI / "configs"
    res_dir  = RI / "results"
    out_base = ROOT / "outputs" / "generated"

    # Keep existing misc dirs intact
    for misc in ["manifests", "base", "adme_final", "adme_legacy", "qsar"]:
        (out_base / misc).mkdir(parents=True, exist_ok=True)

    for m in MODELS:
        name   = m["name"]
        folder = m["folder"]
        print(f"\n{'─'*60}")
        print(f"  Model: {name}  ({m['mtype']})")
        print(f"{'─'*60}")

        for variant, vdir in [("plain", "plain"),
                               ("rl",    "reinforcement_learning"),
                               ("cl",    "curriculum_learning")]:

            # ── outputs/generated directory ──
            (out_base / folder / vdir).mkdir(parents=True, exist_ok=True)
            print(f"  dir    outputs/generated/{folder}/{vdir}/")

            # ── results directory ──
            rdir = name if variant == "plain" else f"{name}_{variant}_adme"
            (res_dir / rdir).mkdir(parents=True, exist_ok=True)
            print(f"  dir    reinvent_integration/results/{rdir}/")

        # ── TOML: plain sampling ──
        write(cfg_dir / f"sample_{name}_plain.toml",
              get_sampling_toml(m, "plain"))

        # ── TOML: RL training ──
        write(cfg_dir / f"rl_{name}.toml", get_rl_toml(m))

        # ── TOML: sample after RL ──
        write(cfg_dir / f"sample_{name}_rl.toml",
              get_sampling_toml(m, "rl"))

        # ── TOML: CL training ──
        write(cfg_dir / f"cl_{name}.toml", get_cl_toml(m))

        # ── TOML: sample after CL ──
        write(cfg_dir / f"sample_{name}_cl.toml",
              get_sampling_toml(m, "cl"))

    print("\n✓ All directories and configs created successfully.\n")

    # Print summary tree
    print("outputs/generated/ tree (model dirs only):")
    for m in MODELS:
        for vdir in ["plain", "reinforcement_learning", "curriculum_learning"]:
            print(f"  outputs/generated/{m['folder']}/{vdir}/")

if __name__ == "__main__":
    main()
