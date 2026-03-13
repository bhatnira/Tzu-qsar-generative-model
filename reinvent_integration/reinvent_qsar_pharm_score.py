"""
External scorer for REINVENT4 that combines:
1) QSAR prediction score
2) Pharmacophore score (optional, via Schrödinger Phase)

Input: newline-separated SMILES on stdin.
Output: JSON payload expected by REINVENT external process scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from descriptors import ecfp, maccs, mordred_desc, rdkit_desc


def _safe_sigmoid(value: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        return 0.0 if value < 0 else 1.0


def _fingerprint_from_smiles(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _load_reference_fingerprints(path: str, radius: int = 2, n_bits: int = 2048):
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    fps = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split()
            if not tokens:
                continue
            fp = _fingerprint_from_smiles(tokens[0], radius=radius, n_bits=n_bits)
            if fp is not None:
                fps.append(fp)
    return fps


def _max_tanimoto_scores(smiles_list: List[str], reference_fps, radius: int = 2, n_bits: int = 2048) -> List[float]:
    if not reference_fps:
        return [0.0 for _ in smiles_list]
    scores: List[float] = []
    for smiles in smiles_list:
        fp = _fingerprint_from_smiles(smiles, radius=radius, n_bits=n_bits)
        if fp is None:
            scores.append(0.0)
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
        scores.append(float(max(similarities) if similarities else 0.0))
    return scores


def _required_substructure_scores(smiles_list: List[str], required_smarts: str) -> List[float]:
    if not required_smarts:
        return [0.0 for _ in smiles_list]

    query = Chem.MolFromSmarts(required_smarts)
    if query is None:
        query = Chem.MolFromSmiles(required_smarts)
    if query is None:
        return [0.0 for _ in smiles_list]

    out: List[float] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            out.append(0.0)
        else:
            out.append(1.0 if mol.HasSubstructMatch(query) else 0.0)
    return out


def _forbidden_substructure_scores(smiles_list: List[str], forbidden_smarts: str) -> List[float]:
    if not forbidden_smarts:
        return [0.0 for _ in smiles_list]

    query = Chem.MolFromSmarts(forbidden_smarts)
    if query is None:
        query = Chem.MolFromSmiles(forbidden_smarts)
    if query is None:
        return [0.0 for _ in smiles_list]

    out: List[float] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            out.append(0.0)
        else:
            out.append(0.0 if mol.HasSubstructMatch(query) else 1.0)
    return out


def _compute_descriptor_vector(smiles: str, descriptor_name: str) -> np.ndarray:
    if descriptor_name == "RDKit":
        return np.asarray(rdkit_desc(smiles), dtype=float)
    if descriptor_name == "ECFP_r1":
        return np.asarray(ecfp(smiles, radius=1), dtype=float)
    if descriptor_name == "ECFP_r2":
        return np.asarray(ecfp(smiles, radius=2), dtype=float)
    if descriptor_name == "ECFP_r3":
        return np.asarray(ecfp(smiles, radius=3), dtype=float)
    if descriptor_name == "MACCS":
        return np.asarray(maccs(smiles), dtype=float)
    if descriptor_name == "Mordred":
        return np.asarray(mordred_desc(smiles), dtype=float)
    raise ValueError(f"Unsupported descriptor: {descriptor_name}")


def _predict_pic50(smiles_list: List[str], model_bundle: Dict) -> List[float]:
    descriptor_name = model_bundle["descriptor_name"]
    imputer = model_bundle["imputer"]
    scaler = model_bundle["scaler"]
    model = model_bundle["model"]

    rows = []
    valid_mask = []
    for smiles in smiles_list:
        try:
            rows.append(_compute_descriptor_vector(smiles, descriptor_name))
            valid_mask.append(True)
        except Exception:
            rows.append(None)
            valid_mask.append(False)

    valid_rows = [row for row in rows if row is not None]
    if not valid_rows:
        return [0.0 for _ in smiles_list]

    x_matrix = np.vstack(valid_rows)
    x_matrix = imputer.transform(x_matrix)
    if scaler is not None:
        x_matrix = scaler.transform(x_matrix)

    predicted_log_ic50 = model.predict(x_matrix)
    predicted_pic50 = (6.0 - predicted_log_ic50).astype(float)

    output = []
    idx = 0
    for is_valid in valid_mask:
        if is_valid:
            output.append(float(predicted_pic50[idx]))
            idx += 1
        else:
            output.append(0.0)
    return output


def _extract_first_float(structure, keys: List[str], default: float = 0.0) -> float:
    for key in keys:
        try:
            value = structure.property.get(key)
            if value is not None:
                return float(value)
        except Exception:
            continue
    return float(default)


def _extract_phase_metrics(structure) -> Dict[str, float]:
    fit_score = _extract_first_float(
        structure,
        [
            "r_phase_ScreenScore",
            "r_phase_Score",
            "r_phase_Fitness",
            "r_phase_MatchScore",
        ],
        default=0.0,
    )
    rmsd = _extract_first_float(
        structure,
        [
            "r_phase_RMSD",
            "r_phase_AlignRMSD",
            "r_phase_MapRMSD",
        ],
        default=999.0,
    )
    matched_features = _extract_first_float(
        structure,
        [
            "r_i_phase_MatchedFeatures",
            "i_phase_MatchedFeatures",
            "r_phase_MatchedFeatures",
        ],
        default=0.0,
    )
    return {
        "fit_score": float(fit_score),
        "rmsd": float(rmsd),
        "matched_features": float(matched_features),
    }


def _resolve_confgen_output(temp_dir: Path, input_mae: Path) -> Path | None:
    input_base = input_mae.with_suffix("")
    candidates = [
        Path(str(input_base) + "-out.maegz"),
        Path(str(input_base) + "-out.mae"),
        temp_dir / (input_base.name + "-out.maegz"),
        temp_dir / (input_base.name + "-out.mae"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in sorted(temp_dir.glob("*out*.maegz")):
        return candidate
    for candidate in sorted(temp_dir.glob("*out*.mae")):
        return candidate
    return None


def _phase_scores(
    smiles_list: List[str],
    hypothesis_path: str,
    schrodinger_root: str,
    host_jobs: int,
    confgen_max: int,
    confgen_energy_window: float,
) -> Dict[str, List[float]]:
    if not hypothesis_path or not schrodinger_root:
        zeros = [0.0 for _ in smiles_list]
        return {
            "fit_score": zeros,
            "rmsd": [999.0 for _ in smiles_list],
            "matched_features": zeros,
        }

    hypothesis = Path(hypothesis_path)
    if not hypothesis.exists():
        zeros = [0.0 for _ in smiles_list]
        return {
            "fit_score": zeros,
            "rmsd": [999.0 for _ in smiles_list],
            "matched_features": zeros,
        }

    schrodinger_root = Path(schrodinger_root)
    phase_screen_exe = schrodinger_root / "phase_screen"
    ligprep_exe = schrodinger_root / "ligprep"
    confgen_exe = schrodinger_root / "confgenx"

    if not phase_screen_exe.exists() or not ligprep_exe.exists() or not confgen_exe.exists():
        zeros = [0.0 for _ in smiles_list]
        return {
            "fit_score": zeros,
            "rmsd": [999.0 for _ in smiles_list],
            "matched_features": zeros,
        }

    try:
        from schrodinger.structure import SmilesStructure, StructureReader, StructureWriter
    except Exception:
        zeros = [0.0 for _ in smiles_list]
        return {
            "fit_score": zeros,
            "rmsd": [999.0 for _ in smiles_list],
            "matched_features": zeros,
        }

    with tempfile.TemporaryDirectory(prefix="reinvent_phase_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        ligands_mae = temp_dir / "ligands.mae"
        prepared_mae = temp_dir / "prepared.maegz"
        out_job = "phase_batch"

        with StructureWriter(str(ligands_mae)) as writer:
            for idx, smiles in enumerate(smiles_list):
                title = f"mol_{idx}"
                try:
                    st = SmilesStructure(smiles)
                    st.title = title
                    writer.append(st)
                except Exception:
                    continue

        ligprep_cmd = [
            str(ligprep_exe),
            "-imae",
            str(ligands_mae),
            "-omae",
            str(prepared_mae),
            "-ph",
            "7.0",
            "-pht",
            "2.0",
            "-s",
            "1",
            "-WAIT",
            "-NOJOBID",
        ]
        ligprep_result = subprocess.run(
            ligprep_cmd,
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if ligprep_result.returncode != 0 or not prepared_mae.exists():
            zeros = [0.0 for _ in smiles_list]
            return {
                "fit_score": zeros,
                "rmsd": [999.0 for _ in smiles_list],
                "matched_features": zeros,
            }

        confgen_cmd = [
            str(confgen_exe),
            str(prepared_mae),
            "-m",
            str(max(1, confgen_max)),
            "-ewindow",
            str(max(0.1, confgen_energy_window)),
            "-optimize",
            "-force_field",
            "OPLS_2005",
            "-WAIT",
            "-NOJOBID",
        ]
        confgen_result = subprocess.run(
            confgen_cmd,
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if confgen_result.returncode != 0:
            zeros = [0.0 for _ in smiles_list]
            return {
                "fit_score": zeros,
                "rmsd": [999.0 for _ in smiles_list],
                "matched_features": zeros,
            }

        conformers_mae = _resolve_confgen_output(temp_dir, prepared_mae)
        if conformers_mae is None:
            zeros = [0.0 for _ in smiles_list]
            return {
                "fit_score": zeros,
                "rmsd": [999.0 for _ in smiles_list],
                "matched_features": zeros,
            }

        cmd = [
            str(phase_screen_exe),
            str(conformers_mae),
            str(hypothesis),
            out_job,
            "-distinct",
            "-HOST",
            f"localhost:{host_jobs}",
            "-NJOBS",
            str(host_jobs),
        ]

        run_result = subprocess.run(
            cmd,
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if run_result.returncode != 0:
            zeros = [0.0 for _ in smiles_list]
            return {
                "fit_score": zeros,
                "rmsd": [999.0 for _ in smiles_list],
                "matched_features": zeros,
            }

        potential_outputs = [
            temp_dir / f"{out_job}_hits.maegz",
            temp_dir / f"{out_job}_hits.mae",
            temp_dir / f"{out_job}.maegz",
            temp_dir / f"{out_job}.mae",
            temp_dir / f"{out_job}-out.maegz",
            temp_dir / f"{out_job}-out.mae",
        ]
        result_file = next((path for path in potential_outputs if path.exists()), None)
        if result_file is None:
            zeros = [0.0 for _ in smiles_list]
            return {
                "fit_score": zeros,
                "rmsd": [999.0 for _ in smiles_list],
                "matched_features": zeros,
            }

        best_per_molecule: Dict[int, Dict[str, float]] = {}
        for st in StructureReader(str(result_file)):
            title = st.title or ""
            matched = re.match(r"^mol_(\d+)", title)
            if matched is None:
                continue
            index = int(matched.group(1))
            if index < 0 or index >= len(smiles_list):
                continue

            metrics = _extract_phase_metrics(st)
            current_best = best_per_molecule.get(index)
            if current_best is None or metrics["fit_score"] > current_best["fit_score"]:
                best_per_molecule[index] = metrics

        fit_scores: List[float] = []
        rmsd_scores: List[float] = []
        matched_features_scores: List[float] = []
        for idx in range(len(smiles_list)):
            best = best_per_molecule.get(
                idx,
                {
                    "fit_score": 0.0,
                    "rmsd": 999.0,
                    "matched_features": 0.0,
                },
            )
            fit_scores.append(float(best["fit_score"]))
            rmsd_scores.append(float(best["rmsd"]))
            matched_features_scores.append(float(best["matched_features"]))

        return {
            "fit_score": fit_scores,
            "rmsd": rmsd_scores,
            "matched_features": matched_features_scores,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="REINVENT external QSAR+pharmacophore scorer")
    parser.add_argument("--qsar-model", default="", help="Path to exported QSAR joblib bundle")
    parser.add_argument("--pharm-hypo", default="", help="Path to .phypo pharmacophore model")
    parser.add_argument("--schrodinger", default="", help="Schrodinger installation directory")
    parser.add_argument("--host-jobs", type=int, default=1, help="Phase host parallel jobs")
    parser.add_argument("--confgen-max", type=int, default=50, help="Max conformers per molecule")
    parser.add_argument(
        "--confgen-energy-window",
        type=float,
        default=10.0,
        help="ConfGen energy window (kcal/mol)",
    )
    parser.add_argument("--target-pic50", type=float, default=6.0, help="Target pIC50 midpoint")
    parser.add_argument("--qsar-weight", type=float, default=0.7, help="Weight for QSAR score")
    parser.add_argument("--pharm-weight", type=float, default=0.3, help="Weight for pharmacophore score")
    parser.add_argument("--sim-weight", type=float, default=0.0, help="Weight for Series C/E similarity score")
    parser.add_argument("--ref-smiles-file", default="", help="Reference SMILES file (e.g., Series C/E) for similarity bias")
    parser.add_argument("--sim-radius", type=int, default=2, help="Morgan radius for similarity")
    parser.add_argument("--sim-nbits", type=int, default=2048, help="Morgan bit vector size")
    parser.add_argument("--required-smarts", default="", help="Required substructure SMARTS/SMILES to preserve")
    parser.add_argument("--required-weight", type=float, default=0.0, help="Weight for required substructure match (1 if matched else 0)")
    parser.add_argument("--forbidden-smarts", default="", help="Forbidden substructure SMARTS/SMILES to avoid (1 if absent else 0)")
    parser.add_argument("--forbidden-weight", type=float, default=0.0, help="Weight for forbidden substructure avoidance")
    parser.add_argument("--pharm-center", type=float, default=1.0, help="Phase score midpoint")
    parser.add_argument("--pharm-scale", type=float, default=0.5, help="Phase score scale factor")
    parser.add_argument(
        "--property-name",
        default="qsar_pharm_score",
        help="Property name returned to REINVENT",
    )
    args = parser.parse_args()

    smiles_list = [line.strip() for line in sys.stdin if line.strip()]

    if not smiles_list:
        output = {
            "version": 1,
            "payload": {
                args.property_name: [],
                "qsar_score": [],
                "pharm_score": [],
                "pred_pic50": [],
            },
        }
        print(json.dumps(output))
        return

    predicted_pic50: List[float]
    qsar_scores: List[float]
    qsar_model_path = Path(args.qsar_model) if args.qsar_model else None
    if qsar_model_path is not None and qsar_model_path.exists():
        model_bundle = joblib.load(str(qsar_model_path))
        predicted_pic50 = _predict_pic50(smiles_list, model_bundle)
        qsar_scores = [
            _safe_sigmoid((value - args.target_pic50) / max(1e-6, 1.0))
            for value in predicted_pic50
        ]
    else:
        predicted_pic50 = [0.0 for _ in smiles_list]
        qsar_scores = [0.0 for _ in smiles_list]

    phase_raw_scores = _phase_scores(
        smiles_list=smiles_list,
        hypothesis_path=args.pharm_hypo,
        schrodinger_root=args.schrodinger,
        host_jobs=max(1, args.host_jobs),
        confgen_max=max(1, args.confgen_max),
        confgen_energy_window=max(0.1, args.confgen_energy_window),
    )
    phase_fit_scores = phase_raw_scores["fit_score"]
    phase_rmsd_scores = phase_raw_scores["rmsd"]
    phase_matched_features = phase_raw_scores["matched_features"]
    pharm_scores = [
        _safe_sigmoid((value - args.pharm_center) / max(1e-6, args.pharm_scale))
        for value in phase_fit_scores
    ]

    reference_fps = _load_reference_fingerprints(
        args.ref_smiles_file,
        radius=max(1, args.sim_radius),
        n_bits=max(128, args.sim_nbits),
    )
    sim_scores = _max_tanimoto_scores(
        smiles_list,
        reference_fps,
        radius=max(1, args.sim_radius),
        n_bits=max(128, args.sim_nbits),
    )
    required_scores = _required_substructure_scores(smiles_list, args.required_smarts)
    forbidden_scores = _forbidden_substructure_scores(smiles_list, args.forbidden_smarts)

    total_weight = max(
        1e-6,
        args.qsar_weight + args.pharm_weight + args.sim_weight + args.required_weight + args.forbidden_weight,
    )
    combined_scores = [
        float(
            (
                args.qsar_weight * q
                + args.pharm_weight * p
                + args.sim_weight * s
                + args.required_weight * r
                + args.forbidden_weight * f
            )
            / total_weight
        )
        for q, p, s, r, f in zip(qsar_scores, pharm_scores, sim_scores, required_scores, forbidden_scores)
    ]

    payload = {
        args.property_name: combined_scores,
        "qsar_score": [float(x) for x in qsar_scores],
        "pharm_score": [float(x) for x in pharm_scores],
        "pharm_fit_score": [float(x) for x in phase_fit_scores],
        "pharm_rmsd": [float(x) for x in phase_rmsd_scores],
        "pharm_matched_features": [float(x) for x in phase_matched_features],
        "sim_score": [float(x) for x in sim_scores],
        "required_match": [float(x) for x in required_scores],
        "forbidden_absent": [float(x) for x in forbidden_scores],
        "pred_pic50": [float(x) for x in predicted_pic50],
    }

    print(json.dumps({"version": 1, "payload": payload}))


if __name__ == "__main__":
    main()
