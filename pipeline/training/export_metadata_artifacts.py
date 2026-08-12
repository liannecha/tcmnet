"""Export lean frontend metadata from frozen IDs and source spreadsheets.

Run from the project root:
    python3 pipeline/training/export_metadata_artifacts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "pipeline" / "artifacts"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_metadata(filename: str, records: list[dict[str, str]]) -> None:
    path = ARTIFACT_DIR / filename
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {path.relative_to(PROJECT_ROOT)} ({len(records)} records)")


def clean_label(value, fallback: str) -> str:
    if pd.isna(value):
        return fallback
    label = str(value).strip()
    return label if label else fallback


def smts_id(value: str | int) -> str:
    text = str(value).strip()
    if text.upper().startswith("SMTS"):
        numeric = text.upper().removeprefix("SMTS").lstrip("0") or "0"
        return f"SMTS{int(numeric):05d}"
    return f"SMTS{int(text):05d}"


def smsy_id(value: str | int) -> str:
    text = str(value).strip()
    if text.upper().startswith("SMSY"):
        numeric = text.upper().removeprefix("SMSY").lstrip("0") or "0"
        return f"SMSY{int(numeric):05d}"
    return f"SMSY{int(text):05d}"


def smhb_id(value: str | int) -> str:
    text = str(value).strip()
    if text.upper().startswith("SMHB"):
        numeric = text.upper().removeprefix("SMHB").lstrip("0") or "0"
        return f"SMHB{int(numeric):05d}"
    return f"SMHB{int(text):05d}"


def build_symptoms_metadata() -> list[dict[str, str]]:
    symptom_columns = read_json(ARTIFACT_DIR / "symptom_columns.json")["columns"]
    source = pd.read_excel(
        PROJECT_ROOT / "pipeline/data/original/Symptoms_Data_SymMap_SMTS.xlsx"
    )
    label_by_id = {
        smts_id(row["TCM_symptom_id"]): clean_label(row["TCM_symptom_name"], "")
        for _, row in source.iterrows()
    }

    records = []
    for symptom in symptom_columns:
        artifact_id = smts_id(symptom)
        records.append(
            {
                "id": artifact_id,
                "label": label_by_id.get(artifact_id) or artifact_id,
            }
        )
    return records


def build_syndromes_metadata() -> list[dict[str, str]]:
    syndrome_ids = read_json(ARTIFACT_DIR / "syndrome_index_to_id.json")["index_to_id"]
    source = pd.read_excel(
        PROJECT_ROOT / "pipeline/data/original/Syndromes_Data_SymMap v2.0, SMSY file.xlsx"
    )
    label_by_id = {}
    for _, row in source.iterrows():
        artifact_id = smsy_id(row["Syndrome_id"])
        label_by_id[artifact_id] = clean_label(
            row.get("Syndrome_English"),
            clean_label(row.get("Syndrome_name"), artifact_id),
        )

    return [
        {
            "id": syndrome_id,
            "label": label_by_id.get(syndrome_id, syndrome_id),
        }
        for syndrome_id in syndrome_ids
    ]


def build_herbs_metadata() -> list[dict[str, str]]:
    herb_ids = read_json(ARTIFACT_DIR / "herb_mapping.json")["herb_ids"]
    source = pd.read_excel(
        PROJECT_ROOT / "pipeline/data/original/SymMap v2.0, SMHB file.xlsx"
    )
    label_by_id = {}
    for _, row in source.iterrows():
        artifact_id = smhb_id(row["Herb_id"])
        label_by_id[artifact_id] = clean_label(
            row.get("English_name"),
            clean_label(
                row.get("Pinyin_name"),
                clean_label(row.get("Chinese_name"), artifact_id),
            ),
        )

    return [
        {
            "id": herb_id,
            "label": label_by_id.get(herb_id, herb_id),
        }
        for herb_id in herb_ids
    ]


def build_concepts_metadata() -> list[dict[str, str]]:
    labels = read_json(ARTIFACT_DIR / "concept_labels.json")["labels"]
    return [{"id": str(label), "label": str(label)} for label in labels]


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_metadata("symptoms_metadata.json", build_symptoms_metadata())
    write_metadata("syndromes_metadata.json", build_syndromes_metadata())
    write_metadata("herbs_metadata.json", build_herbs_metadata())
    write_metadata("concepts_metadata.json", build_concepts_metadata())


if __name__ == "__main__":
    main()
