from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "evaluation-data" / "photophysical_records_collected.csv"
OUTPUT_DIR = ROOT / "evaluation-data" / "results"
MODEL_DIR = ROOT / "trained-model"

TASKS = {
    "absorption": "Absorption_Maximum_nm",
    "emission": "Emission_Maximum_nm",
    "quantum_yield": "Fluorescence_Quantum_Yield",
}
MODEL_FILES = {
    task: MODEL_DIR / f"{task}_morgan_onehot_RandomForest.joblib"
    for task in TASKS
}
TRAINING_FILES = {
    task: ROOT / f"data_filtered_{task}.csv" for task in TASKS
}


def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def canonicalize(smiles: str):
    if not smiles:
        return None, ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, ""
    return mol, Chem.MolToSmiles(mol, canonical=True)


def load_training_sets():
    canonical_sets = {}
    fingerprint_sets = {}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for task, path in TRAINING_FILES.items():
        table = pd.read_csv(path, usecols=["Canonical_SMILES"], dtype=str)
        canonicals = set()
        fingerprints = []
        for value in table["Canonical_SMILES"].dropna().unique():
            mol, canonical = canonicalize(clean_text(value))
            if mol is None:
                continue
            canonicals.add(canonical)
            fingerprints.append(generator.GetFingerprint(mol))
        canonical_sets[task] = canonicals
        fingerprint_sets[task] = fingerprints
    return canonical_sets, fingerprint_sets


def make_features(bundle, mol, solvent_name: str) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(bundle["morgan_radius"]), fpSize=int(bundle["morgan_fp_size"])
    )
    substrate = generator.GetFingerprintAsNumPy(mol).astype(np.uint8)
    solvent = bundle["solvent_encoder"].transform(
        pd.DataFrame({"standard_solvent": [solvent_name]})
    )[0]
    solvent = np.asarray(solvent, dtype=float).reshape(-1)
    features = np.concatenate([substrate, solvent]).reshape(1, -1)
    expected = int(getattr(bundle["model"], "n_features_in_", features.shape[1]))
    if features.shape[1] != expected:
        raise RuntimeError(
            f"{bundle['task']}: feature width {features.shape[1]} != {expected}"
        )
    return features


def metric_row(task: str, frame: pd.DataFrame, doi: str | None = None) -> dict:
    observed = frame["Experimental"].to_numpy(float)
    predicted = frame["Predicted"].to_numpy(float)
    errors = predicted - observed
    if frame.empty:
        return {"Task": task, "Eligible_records": 0, "MAE": None, "RMSE": None, "R2": None, **({"DOI": doi} if doi is not None else {})}
    result = {
        "Task": task,
        "Eligible_records": int(len(frame)),
        "MAE": float(mean_absolute_error(observed, predicted)),
        "RMSE": float(mean_squared_error(observed, predicted) ** 0.5),
        "R2": float(r2_score(observed, predicted)) if len(frame) >= 2 else None,
        "Mean_signed_error": float(errors.mean()),
        "Max_absolute_error": float(np.abs(errors).max()),
        "Experimental_min": float(observed.min()),
        "Experimental_max": float(observed.max()),
        "Predicted_min": float(predicted.min()),
        "Predicted_max": float(predicted.max()),
        "Unit": "0-1" if task.startswith("quantum_yield") else "nm",
    }
    if len(frame) < 2:
        result["R2"] = None
    if doi is not None:
        result["DOI"] = doi
    return result


def main() -> None:
    global OUTPUT_DIR, MODEL_FILES
    parser = argparse.ArgumentParser(description="Evaluate the collected BODIPY records.")
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir.resolve()
    MODEL_FILES = {task: args.model_dir / f'{task}_morgan_onehot_RandomForest.joblib' for task in TASKS}
    core = Chem.MolFromSmarts('[*]~[#5-]1(~[*])~[#7]2~[#6,#7]~[#6,#7]~[#6,#7]~[#6,#7]~2~[#6,#7]~[#6,#7]3~[#6,#7]~[#6,#7]~[#6,#7]~[#7]~1~3')
    RDLogger.DisableLog("rdApp.*")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(INPUT, dtype=str, keep_default_na=True)
    source = source[source["Record_ID"].notna()].copy()
    source = source[source["Record_ID"].astype(str).str.strip().ne("")].copy()

    training_sets, training_fps = load_training_sets()
    bundles = {task: joblib.load(path) for task, path in MODEL_FILES.items()}
    union_training_fps = [fp for task in TASKS for fp in training_fps[task]]

    audit_rows = []
    prediction_rows = []
    for _, row in source.iterrows():
        result = {column: clean_text(row.get(column, "")) for column in source.columns}
        input_smiles = clean_text(row.get("Canonical_SMILES", "")) or clean_text(
            row.get("Source_SMILES", "")
        )
        mol, canonical = canonicalize(input_smiles)
        curated_status = clean_text(row.get("Structure_status", ""))

        if not input_smiles:
            structure_check_status = "excluded_missing_source_smiles"
        elif mol is None:
            structure_check_status = "excluded_invalid_smiles"
        elif curated_status != "structure_approved":
            structure_check_status = "excluded_structure_not_approved"
        elif not (sum(atom.GetAtomicNum() == 9 for atom in mol.GetAtoms()) >= 2 and mol.HasSubstructMatch(core)):
            structure_check_status = "excluded_non_bodipy_core"
        else:
            structure_check_status = "structure_pass"

        result["Validated_Canonical_SMILES"] = canonical
        result["Charge_Corrected_SMILES"] = ""
        result["Structure_repair_status"] = "not_needed" if mol else "failed"
        result["Structure_check_status"] = structure_check_status

        max_similarity = ""
        if mol is not None and union_training_fps:
            fp = rdFingerprintGenerator.GetMorganGenerator(
                radius=2, fpSize=2048
            ).GetFingerprint(mol)
            max_similarity = max(DataStructs.BulkTanimotoSimilarity(fp, union_training_fps))
        result["Max_training_Tanimoto"] = max_similarity

        solvent_name = clean_text(row.get("standard_solvent", ""))
        known_any = any(
            solvent_name in set(map(str, bundle["solvent_encoder"].categories_[0]))
            for bundle in bundles.values()
        )
        result["Unknown_solvent_category"] = "no" if known_any else "yes"

        for task, target_column in TASKS.items():
            observed = number(row.get(target_column))
            overlap = canonical in training_sets[task] if canonical else False
            result[f"{task}_training_overlap"] = "yes" if overlap else "no"

            if structure_check_status != "structure_pass":
                eligibility = structure_check_status
            elif not solvent_name:
                eligibility = "excluded_missing_solvent"
            elif observed is not None and (observed < 0 or (task == "quantum_yield" and observed > 1) or (task != "quantum_yield" and observed == 0)):
                eligibility = "excluded_target_out_of_range"
            elif observed is None:
                eligibility = "excluded_missing_target"
            elif overlap:
                eligibility = "excluded_training_overlap"
            else:
                eligibility = "included"

            result[f"{task}_eligibility"] = eligibility
            result[f"{task}_prediction"] = ""
            result[f"{task}_signed_error"] = ""
            result[f"{task}_absolute_error"] = ""

            if eligibility != "included":
                continue

            bundle = bundles[task]
            predicted = float(
                bundle["model"].predict(make_features(bundle, mol, solvent_name))[0]
            )
            signed_error = predicted - observed
            result[f"{task}_prediction"] = predicted
            result[f"{task}_signed_error"] = signed_error
            result[f"{task}_absolute_error"] = abs(signed_error)
            prediction_rows.append(
                {
                    "Record_ID": result["Record_ID"],
                    "Year": result.get("Year", ""),
                    "DOI": result.get("DOI", ""),
                    "Compound_label": result.get("Compound_label", ""),
                    "standard_solvent": solvent_name,
                    "Validated_Canonical_SMILES": canonical,
                    "Task": task,
                    "Features": "morgan_onehot",
                    "Model": "RandomForest",
                    "Experimental": observed,
                    "Predicted": predicted,
                    "Signed_error": signed_error,
                    "Absolute_error": abs(signed_error),
                    "Known_solvent": solvent_name in set(map(str, bundle["solvent_encoder"].categories_[0])),
                }
            )
        audit_rows.append(result)

    audit = pd.DataFrame(audit_rows)
    predictions = pd.DataFrame(prediction_rows, columns=["Record_ID", "Year", "DOI", "Compound_label", "standard_solvent", "Validated_Canonical_SMILES", "Task", "Features", "Model", "Experimental", "Predicted", "Signed_error", "Absolute_error", "Known_solvent"])
    metrics = []
    for task in TASKS:
        task_predictions = predictions[predictions["Task"].eq(task)].copy()
        metrics.append(metric_row(task, task_predictions))
        known = task_predictions[task_predictions["Known_solvent"]].copy()
        metrics.append(metric_row(f"{task}_known_solvent_only", known))
        for doi, doi_frame in task_predictions.groupby("DOI", sort=True):
            metrics.append(metric_row(f"{task}_doi", doi_frame, str(doi)))

    metrics_table = pd.DataFrame(metrics)
    predictions = predictions.drop(columns=["Known_solvent"])
    counts = {
        "Input_nonblank_records": int(len(source)),
        "Records_with_input_smiles": int(
            source.apply(
                lambda row: bool(clean_text(row.get("Canonical_SMILES", "")))
                or bool(clean_text(row.get("Source_SMILES", ""))),
                axis=1,
            ).sum()
        ),
        "Structure_pass_records": int(
            audit["Structure_check_status"].eq("structure_pass").sum()
        ),
        "Invalid_or_missing_structure_records": int(
            audit["Structure_check_status"].ne("structure_pass").sum()
        ),
        "Unique_structure_pass": int(
            audit.loc[
                audit["Structure_check_status"].eq("structure_pass"),
                "Validated_Canonical_SMILES",
            ].nunique()
        ),
        "Training_overlap_records_any_task": int(
            audit[[f"{task}_training_overlap" for task in TASKS]]
            .eq("yes")
            .any(axis=1)
            .sum()
        ),
        "Unknown_solvent_records": int(
            audit["Unknown_solvent_category"].eq("yes").sum()
        ),
    }

    audit_path = OUTPUT_DIR / "cleaning_audit.csv"
    predictions_path = OUTPUT_DIR / "predictions.csv"
    metrics_path = OUTPUT_DIR / "metrics.csv"
    results_path = OUTPUT_DIR / "results.json"
    audit.to_csv(audit_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    metrics_table.to_csv(metrics_path, index=False)
    results_path.write_text(
        json.dumps(
            {
                "source_file": INPUT.relative_to(ROOT).as_posix(),
                "features": "Morgan fingerprint + solvent One-Hot",
                "model": "RandomForest",
                "counts": counts,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(metrics_table[metrics_table["Task"].isin(TASKS)].to_string(index=False))
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
