"""Create English symptom labels for frontend metadata.

This writes a derived table and artifact; it does not edit the raw SymMap file.

Run from the project root:
    python3 pipeline/training/translate_symptom_names.py
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "pipeline" / "artifacts"
OUTPUT_CSV = PROJECT_ROOT / "pipeline" / "data" / "processed" / "Symptom_English_Names.csv"
OUTPUT_JSON = ARTIFACT_DIR / "symptom_english_names.json"
OVERRIDES_JSON = (
    PROJECT_ROOT / "pipeline" / "data" / "processed" / "Symptom_English_Overrides.json"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def smts_id(value: str | int) -> str:
    text = str(value).strip()
    if text.upper().startswith("SMTS"):
        numeric = text.upper().removeprefix("SMTS").lstrip("0") or "0"
        return f"SMTS{int(numeric):05d}"
    return f"SMTS{int(text):05d}"


def clean_translation(value: str, fallback: str) -> str:
    label = re.sub(r"\s+", " ", str(value)).strip()
    label = re.sub(
        r"^(traditional chinese medicine symptoms?|symptoms? of traditional chinese medicine|tcm symptoms?)\s*:\s*",
        "",
        label,
        flags=re.IGNORECASE,
    ).strip()
    return label if label else fallback


def chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def load_existing_translations(force: bool) -> dict[str, str]:
    if force or not OUTPUT_CSV.exists():
        return {}
    df = pd.read_csv(OUTPUT_CSV).fillna("")
    return {
        str(row["id"]): str(row["english_name"]).strip()
        for _, row in df.iterrows()
        if str(row.get("english_name", "")).strip()
    }


def load_overrides() -> dict[str, str]:
    if not OVERRIDES_JSON.exists():
        return {}
    return {
        str(symptom_id): clean_translation(label, str(symptom_id))
        for symptom_id, label in read_json(OVERRIDES_JSON).items()
    }


def translate_names(
    names_by_id: dict[str, str],
    batch_size: int,
    pause: float,
    force: bool,
) -> dict[str, str]:
    translations = load_existing_translations(force=force)
    missing_ids = [
        symptom_id for symptom_id in names_by_id if symptom_id not in translations
    ]
    translator = GoogleTranslator(source="zh-CN", target="en")

    for start, batch_ids in chunks(missing_ids, batch_size):
        # The TCM context avoids some literal mistranslations of short symptom names.
        batch_names = [f"中医症状：{names_by_id[symptom_id]}" for symptom_id in batch_ids]
        translated_text = translator.translate("\n".join(batch_names))
        translated_lines = str(translated_text).splitlines()

        if len(translated_lines) != len(batch_ids):
            raise RuntimeError(
                "Translation line count mismatch. "
                f"Expected {len(batch_ids)}, got {len(translated_lines)}."
            )

        for symptom_id, translated in zip(batch_ids, translated_lines):
            translations[symptom_id] = clean_translation(translated, symptom_id)

        print(f"Translated {min(start + len(batch_ids), len(missing_ids))}/{len(missing_ids)} missing symptoms")
        if pause:
            time.sleep(pause)

    return translations


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate model symptom names to English.")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--pause", type=float, default=0.2)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the existing cached translation table and regenerate all labels.",
    )
    args = parser.parse_args()

    symptom_columns = read_json(ARTIFACT_DIR / "symptom_columns.json")["columns"]
    model_ids = [smts_id(symptom) for symptom in symptom_columns]

    source = pd.read_excel(
        PROJECT_ROOT / "pipeline/data/original/Symptoms_Data_SymMap_SMTS.xlsx"
    ).fillna("")
    source["id"] = source["TCM_symptom_id"].apply(smts_id)
    source = source[source["id"].isin(model_ids)].copy()
    names_by_id = dict(zip(source["id"], source["TCM_symptom_name"].astype(str)))

    translations = translate_names(names_by_id, args.batch_size, args.pause, args.force)
    translations.update(load_overrides())

    source["english_name"] = source["id"].map(translations).fillna(source["id"])
    output = source[
        ["id", "TCM_symptom_id", "TCM_symptom_name", "Symptom_pinYin", "english_name"]
    ].rename(
        columns={
            "TCM_symptom_id": "source_numeric_id",
            "TCM_symptom_name": "chinese_name",
            "Symptom_pinYin": "pinyin",
        }
    )
    output["id"] = pd.Categorical(output["id"], categories=model_ids, ordered=True)
    output = output.sort_values("id").reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(
        json.dumps(
            [
                {"id": row["id"], "english_name": row["english_name"]}
                for _, row in output.iterrows()
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {OUTPUT_CSV.relative_to(PROJECT_ROOT)} ({len(output)} records)")
    print(f"Wrote {OUTPUT_JSON.relative_to(PROJECT_ROOT)} ({len(output)} records)")


if __name__ == "__main__":
    main()
