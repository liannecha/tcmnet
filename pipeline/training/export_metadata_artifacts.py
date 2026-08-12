"""Export lean frontend metadata from frozen IDs and source spreadsheets.

Run from the project root:
    python3 pipeline/training/export_metadata_artifacts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "pipeline" / "artifacts"
SYMPTOM_ENGLISH_FILE = (
    PROJECT_ROOT / "pipeline" / "data" / "processed" / "Symptom_English_Names.csv"
)
SYNDROME_ENGLISH_OVERRIDES = {
    "SMSY00055": "fluid-blood imbalance",
    "SMSY00056": "fluid-blood insufficiency",
    "SMSY00119": "blood stasis swelling and pain",
    "SMSY00132": "cold in the channels",
}


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


ORGAN_MEANINGS = {
    "kidney": "kidney-related warming, fluid, growth, or lower-body function",
    "spleen": "digestion, energy production, and the body's ability to transform fluids",
    "liver": "the smooth movement of qi, blood, and emotions",
    "heart": "circulation, spirit, sleep, and mental-emotional steadiness",
    "cardiac": "circulation, spirit, sleep, and mental-emotional steadiness",
    "lung": "breathing, protective qi, skin, and fluid movement",
    "stomach": "digestion and the downward movement of food and fluids",
    "gallbladder": "the gallbladder channel and the movement of bile or constrained heat",
    "abdomen": "abdominal digestion, pain, fullness, or cold sensations",
}

PATTERN_MEANINGS = {
    "accumulation": "material or functional buildup that is not moving or transforming well",
    "accumulated": "material or functional buildup that is not moving or transforming well",
    "air": "qi strength and movement",
    "qi": "qi movement or qi strength",
    "blood": "blood nourishment, movement, or stagnation",
    "body fluid": "the nourishment and movement of fluids",
    "yin": "cooling, moistening, and nourishing aspects of the body",
    "yang": "warming, activating, and transforming aspects of the body",
    "cold": "cold signs such as chilliness, slow movement, or cold pain",
    "heat": "heat signs such as fever, inflammation, thirst, or irritability",
    "fever": "heat signs such as fever, inflammation, thirst, or irritability",
    "febrile": "heat signs such as fever, inflammation, thirst, or irritability",
    "warm": "heat signs that may enter deeper levels of the body",
    "humidity": "dampness with heat or heavy, turbid symptoms",
    "dampness": "heaviness, swelling, discharge, or sluggish fluid movement",
    "damp": "heaviness, swelling, discharge, or sluggish fluid movement",
    "fluid": "fluid movement, retention, or depletion",
    "phlegm": "thick fluids, mucus, nodules, cloudiness, or obstructed movement",
    "sputum": "phlegm or mucus that is difficult to clear",
    "wind": "moving or changing symptoms such as tremor, dizziness, spasms, or sudden onset",
    "fire": "intense heat signs such as inflammation, agitation, bleeding, or burning pain",
    "dry": "dryness signs such as dry cough, dry stool, thirst, or depleted fluids",
    "dryness": "dryness signs such as dry cough, dry stool, thirst, or depleted fluids",
    "deficiency": "an underlying weakness or lack of nourishment, warmth, or functional energy",
    "weakness": "an underlying weakness or lack of nourishment, warmth, or functional energy",
    "asthenia": "an underlying weakness or lack of nourishment, warmth, or functional energy",
    "debility": "weakness after chronic illness or prolonged depletion",
    "consumptive": "long-running deficiency with fatigue, wasting, or depleted reserves",
    "prostration": "a severe collapse-like deficiency state with loss of functional strength",
    "depression": "constrained or depressed movement of qi, often affecting mood, rib-side tension, or digestion",
    "excess": "a stronger obstructive pattern where something is stuck, accumulated, or overactive",
    "stagnation": "blocked movement, often linked with distension, pain, or emotional constraint",
    "stasis": "slowed or blocked blood movement, often linked with fixed or sharp pain",
    "congestion": "blocked or congested movement of blood, fluids, or qi",
    "retention": "something lingering or collecting rather than clearing normally",
    "indigestion": "poor food transformation with fullness, distension, belching, nausea, or diarrhea",
    "dyspepsia": "poor food transformation with fullness, distension, belching, nausea, or diarrhea",
    "food": "food retention or poor transformation after eating",
    "strain": "overuse or taxation that weakens qi, blood, muscles, or channels",
    "sprain": "traumatic channel injury with pain, swelling, or impaired movement",
    "external": "a surface-level invasion affecting protective qi",
    "externally": "a surface-level invasion affecting protective qi",
    "infected": "an externally contracted pattern affecting protective qi",
    "alcohol": "alcohol-related heat, dampness, toxin, or digestive injury",
    "toxin": "toxic heat or irritant accumulation that can disturb normal function",
    "parasitic": "parasitic accumulation that may cause abdominal pain or digestive disruption",
    "parasite": "parasitic accumulation that may cause abdominal pain or digestive disruption",
    "amenorrhea": "menstrual blockage or lack of menstrual flow",
    "dysentery": "intestinal disturbance with diarrhea, urgency, mucus, or blood",
    "diarrhea": "intestinal disturbance with loose stools or poor transformation",
    "distention": "fullness, pressure, or swelling from stagnation or accumulation",
    "distending": "fullness, pressure, or swelling from stagnation or accumulation",
    "hypochondrium": "rib-side or flank discomfort often linked with constrained flow",
    "camp": "deeper heat affecting nutritive or blood-level function",
}

PATTERN_DETAILS = {
    "retention": (
        "something is lingering or accumulating inside the body rather than moving or "
        "clearing normally"
    ),
    "interior": (
        "the pattern is understood as internal rather than mainly affecting the surface "
        "of the body"
    ),
    "qi": (
        "the pattern involves the body's functional energy, especially movement, lifting, "
        "holding, or transformation"
    ),
    "blood": (
        "the pattern involves nourishment or circulation, and may relate to weakness, "
        "dryness, fixed pain, or poor movement of blood"
    ),
    "yin": (
        "the pattern involves the cooling and nourishing side of the body, so dryness, "
        "heat sensations, or lack of fluids may be relevant"
    ),
    "yang": (
        "the pattern involves the warming and activating side of the body, so coldness, "
        "low energy, or weak transformation may be relevant"
    ),
    "cold": (
        "cold signs may include chilliness, cold pain, pale appearance, slower movement, "
        "or symptoms that feel better with warmth"
    ),
    "heat": (
        "heat signs may include feverishness, thirst, irritability, redness, yellow "
        "secretions, or inflammatory-type symptoms"
    ),
    "fever": (
        "heat signs may include feverishness, thirst, irritability, redness, yellow "
        "secretions, or inflammatory-type symptoms"
    ),
    "dampness": (
        "dampness can show up as heaviness, swelling, loose stools, discharge, nausea, "
        "or a sluggish stuck feeling"
    ),
    "damp": (
        "dampness can show up as heaviness, swelling, loose stools, discharge, nausea, "
        "or a sluggish stuck feeling"
    ),
    "phlegm": (
        "phlegm can mean visible mucus or a broader pattern of thick fluids, "
        "cloudiness, nodules, dizziness, nausea, or obstruction"
    ),
    "sputum": (
        "sputum points to phlegm or mucus accumulation, especially with cough, chest "
        "fullness, throat obstruction, nausea, or heaviness"
    ),
    "wind": (
        "wind often points to symptoms that move, change quickly, or involve shaking, "
        "spasm, dizziness, itching, or sudden onset"
    ),
    "fire": (
        "fire is a stronger heat pattern and may suggest agitation, burning pain, "
        "redness, inflammation, bleeding, or intense thirst"
    ),
    "deficiency": (
        "deficiency means the body lacks enough nourishment, warmth, fluids, blood, or "
        "functional strength to regulate itself well"
    ),
    "excess": (
        "excess means the pattern is driven more by accumulation, obstruction, or an "
        "overactive pathogenic factor"
    ),
    "stagnation": (
        "stagnation means movement is blocked, often creating distension, pressure, "
        "pain, mood constraint, or symptoms that come and go"
    ),
    "stasis": (
        "stasis means blood movement is slowed or blocked, often linked with fixed, "
        "sharp, or persistent pain"
    ),
    "accumulation": (
        "accumulation suggests that food, fluids, blood, or other material is collecting "
        "instead of moving, transforming, or clearing normally"
    ),
    "retention": (
        "retention suggests that food, fluid, phlegm, or another substance is lingering "
        "rather than being transformed and cleared"
    ),
    "food": (
        "food retention often shows up as abdominal fullness, distension, belching, "
        "nausea, sour regurgitation, poor appetite, loose stool, or diarrhea after eating"
    ),
    "indigestion": (
        "indigestion points to poor transformation of food, often with fullness, bloating, "
        "belching, nausea, or irregular stool"
    ),
    "dyspepsia": (
        "dyspepsia points to poor transformation of food, often with fullness, bloating, "
        "belching, nausea, or irregular stool"
    ),
    "febrile": (
        "febrile patterns emphasize heat signs such as fever, thirst, irritability, "
        "restlessness, delirium, or injury to fluids"
    ),
    "warm": (
        "warm disease language points to heat entering more deeply, where it may disturb "
        "fluids, consciousness, or blood-level function"
    ),
    "humidity": (
        "damp-heat or humid-warm patterns often combine heaviness, digestive upset, fever, "
        "sticky discharge, or a turbid stuck feeling"
    ),
    "dryness": (
        "dryness may appear as dry cough, dry throat, dry stool, thirst, or signs that "
        "body fluids are insufficient"
    ),
    "dry": (
        "dryness may appear as dry cough, dry throat, dry stool, thirst, or signs that "
        "body fluids are insufficient"
    ),
    "body fluid": (
        "fluid injury may show up as thirst, dryness, scant fluids, dry stool, or heat "
        "signs that consume fluids"
    ),
    "congestion": (
        "congestion means movement is obstructed, which may cause fixed pain, swelling, "
        "distension, or menstrual blockage"
    ),
    "strain": (
        "strain and overwork can tax qi, blood, muscles, and channels, leading to pain, "
        "weakness, fatigue, or reduced recovery"
    ),
    "sprain": (
        "sprain patterns point to traumatic channel injury with pain, swelling, bruising, "
        "or limited movement"
    ),
    "external": (
        "external patterns affect the body's surface and protective qi, often with acute "
        "onset, chills, fever, body aches, cough, or nasal symptoms"
    ),
    "externally": (
        "external patterns affect the body's surface and protective qi, often with acute "
        "onset, chills, fever, body aches, cough, or nasal symptoms"
    ),
    "infected": (
        "externally contracted patterns affect the body's surface and protective qi, often "
        "with acute onset, chills, fever, body aches, cough, or nasal symptoms"
    ),
    "debil": (
        "long illness can deplete qi, blood, yin, or yang, leaving fatigue, weakness, "
        "poor appetite, low resilience, or slow recovery"
    ),
    "consumptive": (
        "consumptive patterns describe long-running depletion with fatigue, weakness, "
        "weight loss, low reserves, or chronic deficiency signs"
    ),
    "weakness": (
        "weakness may show up as fatigue, low resilience, poor appetite, shortness of "
        "breath, spontaneous sweating, or reduced functional strength"
    ),
    "asthenia": (
        "asthenia describes weakness or depleted functional strength, often with fatigue, "
        "poor resilience, shortness of breath, or reduced organ-system support"
    ),
    "prostration": (
        "prostration points to a severe collapse-like deficiency state, where functional "
        "strength is very low and urgent support is implied in the source terminology"
    ),
    "alcohol": (
        "alcohol-related patterns often involve dampness, heat, toxin, nausea, abdominal "
        "discomfort, thirst, irritability, or injury to the digestive center"
    ),
    "toxin": (
        "toxin language suggests a stronger irritant or heat accumulation that can disturb "
        "digestion, consciousness, skin, or inflammatory signs"
    ),
    "parasitic": (
        "parasitic accumulation may be associated with abdominal pain, digestive upset, "
        "irregular appetite, distension, or stool changes"
    ),
    "amenorrhea": (
        "amenorrhea points to menstrual flow being absent or blocked, often considered "
        "through blood movement, blood nourishment, or stagnation patterns"
    ),
    "dysentery": (
        "dysentery points to intestinal disturbance with diarrhea, urgency, mucus, blood, "
        "or abdominal discomfort"
    ),
    "diarrhea": (
        "diarrhea points to poor intestinal transformation or clearing, often with loose "
        "stools, urgency, abdominal discomfort, dampness, cold, heat, or deficiency"
    ),
    "distention": (
        "distention points to fullness or pressure from stagnation, accumulation, dampness, "
        "or blocked movement"
    ),
    "distending": (
        "distending pain points to pressure from constrained movement, often around the "
        "flank, chest, abdomen, or channel pathway"
    ),
    "hypochondrium": (
        "hypochondriac or flank pain often points to constrained movement around the rib-side "
        "region, commonly framed through liver or gallbladder channel patterns"
    ),
    "depression": (
        "depression in this pattern language usually means constrained qi movement rather "
        "than only a modern mood diagnosis"
    ),
    "camp": (
        "nutritive-level heat can disturb fluids, sleep, restlessness, consciousness, or "
        "blood-level signs"
    ),
}


def matching_meanings(lowered: str, glossary: dict[str, str]) -> list[str]:
    """Return unique readable meanings for terms found in a syndrome name."""
    meanings: list[str] = []
    for term, meaning in glossary.items():
        if term in lowered and meaning not in meanings:
            meanings.append(meaning)
    return meanings


def matching_terms(lowered: str, glossary: dict[str, str]) -> list[str]:
    """Return unique glossary terms found in a syndrome name."""
    return [term for term in glossary if term in lowered]


def syndrome_description(english_name: str, chinese_name: str) -> str:
    """Create a readable TCM description for the UI from available source labels."""
    name = clean_label(english_name, chinese_name).strip()
    lowered = name.lower()
    organ_meanings = matching_meanings(lowered, ORGAN_MEANINGS)
    pattern_meanings = matching_meanings(lowered, PATTERN_MEANINGS)
    detail_terms = matching_terms(lowered, PATTERN_DETAILS)

    if "phlegm" in lowered and "interior" in lowered:
        return (
            f"{name.capitalize()} describes a pattern where phlegm is understood to remain "
            "inside the body and obstruct normal movement of qi and fluids. Phlegm may "
            "refer to visible mucus, but it can also describe heaviness, cloudiness, "
            "nausea, dizziness, fullness, nodules, or a stuck sensation."
        )

    if "middle qi" in lowered:
        return (
            "This pattern means the central qi is weak and cannot hold or lift "
            "the body's functions well. It is often associated with fatigue, dizziness, "
            "abdominal heaviness, loose stools, or prolapse-like symptoms. The name points "
            "to a loss of upward support from the digestive center of the body."
        )

    if "wind-cold" in lowered or ("wind" in lowered and "cold" in lowered):
        return (
            "This pattern describes an external wind-cold invasion, often similar "
            "to an early cold-like presentation. It may involve chills, aversion to cold, "
            "clear nasal discharge, headache, body aches, or cough. The emphasis is on an "
            "external pattern affecting the body's surface and protective qi."
        )

    detail_sentences = [PATTERN_DETAILS[term] for term in detail_terms[:3]]
    detail_text = " ".join(
        sentence.capitalize() + "." for sentence in detail_sentences
    ).strip()

    if organ_meanings and pattern_meanings:
        description = (
            f"{name.capitalize()} points to {organ_meanings[0]} together "
            f"with {', '.join(pattern_meanings[:2])}."
        )
        return f"{description} {detail_text}".strip()

    if pattern_meanings:
        description = f"{name.capitalize()} points to {', '.join(pattern_meanings[:3])}."
        return f"{description} {detail_text}".strip()

    if organ_meanings:
        return (
            f"{name.capitalize()} points to {organ_meanings[0]}. "
            "The syndrome name points to the body system or region most associated "
            "with the symptom pattern."
        )

    return (
        f"{name.capitalize()} is a traditional pattern label associated with {chinese_name}. "
        "The name should be reviewed against the original terminology before using it as "
        "a detailed explanatory label."
    )


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


def concept_display_label(label: str) -> str:
    text = str(label).strip()
    return "Heat" if text.lower() == "hot" else text[:1].upper() + text[1:]


def infer_herb_targets(class_name: str, properties: str, meridians: str) -> list[str]:
    """Infer simple concept targets when a herb has no frozen concept links."""
    text = f"{class_name} {properties} {meridians}".lower()
    targets: list[str] = []
    checks = [
        ("liver", "Wood"),
        ("gallbladder", "Wood"),
        ("heart", "Fire"),
        ("small intestine", "Fire"),
        ("spleen", "Earth"),
        ("stomach", "Earth"),
        ("lung", "Metal"),
        ("large intestine", "Metal"),
        ("kidney", "Water"),
        ("bladder", "Water"),
        ("cold", "Cold"),
        ("cool", "Cold"),
        ("hot", "Heat"),
        ("warm", "Heat"),
        ("calm", "Yin"),
        ("fire", "Heat"),
        ("yin", "Yin"),
        ("yang", "Yang"),
        ("digest", "Earth"),
        ("antitussive", "Metal"),
        ("asthmatic", "Metal"),
        ("cough", "Metal"),
        ("diuretic", "Water"),
        ("dampness", "Earth"),
        ("tonic", "Deficiency"),
        ("deficiency", "Deficiency"),
        ("purging", "Excess"),
        ("clearing", "Excess"),
    ]
    for token, concept in checks:
        if token in text and concept not in targets:
            targets.append(concept)
    return targets


def build_symptoms_metadata() -> list[dict[str, str]]:
    symptom_columns = read_json(ARTIFACT_DIR / "symptom_columns.json")["columns"]
    english_by_id = {}
    if SYMPTOM_ENGLISH_FILE.exists():
        english_df = pd.read_csv(SYMPTOM_ENGLISH_FILE).fillna("")
        english_by_id = {
            str(row["id"]): clean_label(row["english_name"], str(row["id"]))
            for _, row in english_df.iterrows()
        }

    return [
        {
            "id": smts_id(symptom),
            "label": english_by_id.get(smts_id(symptom), smts_id(symptom)),
        }
        for symptom in symptom_columns
    ]


def build_syndromes_metadata() -> list[dict[str, str]]:
    syndrome_ids = read_json(ARTIFACT_DIR / "syndrome_index_to_id.json")["index_to_id"]
    source = pd.read_excel(
        PROJECT_ROOT / "pipeline/data/original/Syndromes_Data_SymMap v2.0, SMSY file.xlsx"
    )
    metadata_by_id = {}
    for _, row in source.iterrows():
        artifact_id = smsy_id(row["Syndrome_id"])
        english_name = SYNDROME_ENGLISH_OVERRIDES.get(
            artifact_id,
            clean_label(
                row.get("Syndrome_English"),
                clean_label(row.get("Syndrome_name"), artifact_id),
            ),
        )
        chinese_name = clean_label(row.get("Syndrome_name"), artifact_id)
        metadata_by_id[artifact_id] = {
            "id": artifact_id,
            "label": english_name,
            "english_name": english_name,
            "chinese_name": chinese_name,
            "description": syndrome_description(english_name, chinese_name),
        }

    return [
        metadata_by_id.get(
            syndrome_id,
            {
                "id": syndrome_id,
                "label": syndrome_id,
                "english_name": syndrome_id,
                "chinese_name": syndrome_id,
                "description": f"A TCM syndrome pattern listed as {syndrome_id}.",
            },
        )
        for syndrome_id in syndrome_ids
    ]


def build_herbs_metadata() -> list[dict[str, str]]:
    herb_mapping = read_json(ARTIFACT_DIR / "herb_mapping.json")
    herb_ids = herb_mapping["herb_ids"]
    concept_labels = read_json(ARTIFACT_DIR / "concept_labels.json")["labels"]
    herb_concept_matrix = np.load(ARTIFACT_DIR / "herb_concept_matrix.npy")
    concept_targets_by_id = {
        herb_id: [
            concept_display_label(str(label))
            for label, value in zip(concept_labels, herb_concept_matrix[index])
            if str(label).lower() != "reproductive" and float(value) > 0
        ]
        for index, herb_id in enumerate(herb_ids)
    }
    source = pd.read_excel(
        PROJECT_ROOT / "pipeline/data/original/SymMap v2.0, SMHB file.xlsx"
    )
    metadata_by_id = {}
    for _, row in source.iterrows():
        artifact_id = smhb_id(row["Herb_id"])
        english_name = clean_label(
            row.get("English_name"),
            clean_label(
                row.get("Pinyin_name"),
                clean_label(row.get("Chinese_name"), artifact_id),
            ),
        )
        chinese_name = clean_label(row.get("Chinese_name"), artifact_id)
        class_name = clean_label(row.get("Class_English"), "")
        properties = clean_label(row.get("Properties_English"), "")
        meridians = clean_label(row.get("Meridians_English"), "")
        description_parts = []
        if class_name:
            description_parts.append(
                f"This herb is commonly categorized in TCM as {class_name.lower()}."
            )
        if properties:
            description_parts.append(f"Its traditional properties are {properties.lower()}.")
        if meridians:
            description_parts.append(f"It is associated with the {meridians} meridian system.")
        description = " ".join(description_parts) or (
            f"This herb is listed in the source data as {english_name}."
        )
        target_concepts = concept_targets_by_id.get(artifact_id, [])
        if not target_concepts:
            target_concepts = infer_herb_targets(class_name, properties, meridians)
        metadata_by_id[artifact_id] = {
            "id": artifact_id,
            "label": english_name,
            "english_name": english_name,
            "chinese_name": chinese_name,
            "description": description,
            "target_concepts": target_concepts,
        }

    return [
        metadata_by_id.get(
            herb_id,
            {
                "id": herb_id,
                "label": herb_id,
                "english_name": herb_id,
                "chinese_name": herb_id,
                "description": f"This herb is listed in the source data as {herb_id}.",
                "target_concepts": concept_targets_by_id.get(herb_id, []),
            },
        )
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
