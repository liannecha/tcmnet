"""Generate broad body-location mappings for every TCMNet symptom.

The matcher uses transparent keyword rules rather than an LLM. These mappings
are intended for ranking/boosting symptom suggestions, not for diagnosis.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMPTOMS_PATH = PROJECT_ROOT / "pipeline" / "artifacts" / "symptoms_metadata.json"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "frontend"
    / "src"
    / "generated"
    / "symptom-body-location-map.json"
)


CATEGORY_REGION_IDS = {
    "head": ["head-front", "head-back"],
    "eye_face": ["head-front"],
    "ear": ["head-front", "head-back"],
    "nose": ["head-front"],
    "mouth_tongue": ["head-front"],
    "throat_neck": ["neck-front", "neck-back"],
    "chest_lung": [
        "chest-right-front",
        "chest-left-front",
        "upperback-left",
        "upperback-right",
    ],
    "heart_chest": ["chest-right-front", "chest-left-front"],
    "abdomen_digestive": [
        "midtorso-right-front",
        "midtorso-left-front",
        "abdomen-right-front",
        "abdomen-left-front",
        "midback-left",
        "midback-right",
        "lowerback-left",
        "lowerback-right",
    ],
    "pelvis_reproductive": [
        "pelvis-right-front",
        "pelvis-left-front",
        "gluteal-left",
        "gluteal-right",
    ],
    "urinary": [
        "pelvis-right-front",
        "pelvis-left-front",
        "lowerback-left",
        "lowerback-right",
    ],
    "back_spine": [
        "upperback-left",
        "upperback-right",
        "midback-left",
        "midback-right",
        "lowerback-left",
        "lowerback-right",
    ],
    "upper_limb": [
        "shoulder-right-front",
        "shoulder-left-front",
        "deltoid-right-front",
        "deltoid-left-front",
        "shoulder-left-back",
        "shoulder-right-back",
        "arm-right-front",
        "arm-left-front",
        "arm-left-back",
        "arm-right-back",
        "elbow-right-front",
        "elbow-left-front",
        "elbow-left-back",
        "elbow-right-back",
        "forearm-right-front",
        "forearm-left-front",
        "forearm-left-back",
        "forearm-right-back",
        "hand-right-palm",
        "hand-left-palm",
        "hand-left-back",
        "hand-right-back",
    ],
    "lower_limb": [
        "thigh-right-front",
        "thigh-left-front",
        "thigh-left-back",
        "thigh-right-back",
        "knee-right-front",
        "knee-left-front",
        "knee-left-back",
        "knee-right-back",
        "calf-right-front",
        "calf-left-front",
        "calf-left-back",
        "calf-right-back",
        "lowercalf-right-front",
        "lowercalf-left-front",
        "lowercalf-left-back",
        "lowercalf-right-back",
        "foot-right-front",
        "foot-left-front",
        "foot-left-back",
        "foot-right-back",
    ],
    "skin_surface": [
        "head-front",
        "neck-front",
        "chest-right-front",
        "chest-left-front",
        "abdomen-right-front",
        "abdomen-left-front",
        "pelvis-right-front",
        "pelvis-left-front",
        "arm-right-front",
        "arm-left-front",
        "hand-right-palm",
        "hand-left-palm",
        "thigh-right-front",
        "thigh-left-front",
        "foot-right-front",
        "foot-left-front",
        "head-back",
        "neck-back",
        "upperback-left",
        "upperback-right",
        "lowerback-left",
        "lowerback-right",
        "gluteal-left",
        "gluteal-right",
        "arm-left-back",
        "arm-right-back",
        "hand-left-back",
        "hand-right-back",
        "thigh-left-back",
        "thigh-right-back",
        "foot-left-back",
        "foot-right-back",
    ],
    "whole_body_systemic": [
        "head-front",
        "head-back",
        "chest-right-front",
        "chest-left-front",
        "abdomen-right-front",
        "abdomen-left-front",
        "upperback-left",
        "upperback-right",
        "arm-right-front",
        "arm-left-front",
        "thigh-right-front",
        "thigh-left-front",
    ],
    "mental_sleep": ["head-front", "head-back"],
}


KEYWORD_RULES = [
    ("eye_face", [r"\beye\b", r"eyes", r"vision", r"conjunct", r"lacrimat", r"tear", r"eyelid", r"look for long", r"white of the eye", r"\bface\b", r"facial", r"cheek", r"complexion", r"flushing"]),
    ("ear", [r"\bear\b", r"otic", r"otorrhea", r"tinnitus", r"deaf", r"hearing"]),
    ("nose", [r"\bnose\b", r"nasal", r"rhino", r"epistaxis", r"nostril"]),
    ("mouth_tongue", [r"mouth", r"tongue", r"tooth", r"teeth", r"gum", r"\blip", r"saliva", r"taste", r"oral"]),
    ("throat_neck", [r"throat", r"pharyn", r"laryn", r"voice", r"hoarse", r"swallow", r"\bneck\b", r"scrofula"]),
    ("head", [r"\bhead\b", r"headache", r"brain", r"cerebral", r"dizziness", r"vertigo", r"hair", r"scalp", r"temple"]),
    ("heart_chest", [r"palpitation", r"\bheart\b", r"cardiac", r"heartbeat"]),
    ("chest_lung", [r"\bchest\b", r"intercostal", r"cough", r"phlegm", r"sputum", r"expectoration", r"hemoptysis", r"tuberculosis", r"asthma", r"breath", r"dyspnea", r"lung", r"wheez"]),
    ("abdomen_digestive", [r"abdomen", r"abdominal", r"ascites", r"belly", r"epigastr", r"stomach", r"gastric", r"reflux", r"retch", r"bowel", r"intestin", r"stool", r"fecal", r"diarrhea", r"constipation", r"defecat", r"tenesmus", r"nausea", r"vomit", r"belch", r"hiccup", r"appetite", r"anorexia", r"hungry", r"\beat", r"eating", r"nagu", r"noisy", r"fullness", r"distension", r"borborygmus", r"rectal", r"\banal\b", r"anus", r"jaundice"]),
    ("pelvis_reproductive", [r"leucorrhea", r"vaginal", r"vagina", r"uter", r"menstr", r"menses", r"dysmenorrhea", r"amenorrhea", r"metrorrhagia", r"pregnan", r"postpartum", r"afterbirth", r"miscarriage", r"infertility", r"semen", r"sperm", r"seminal", r"testic", r"impotence", r"genital", r"pelvis", r"pelvic"]),
    ("urinary", [r"urine", r"urinary", r"urination", r"mictur", r"polyuria", r"stranguria", r"bladder", r"kidney", r"dysuria", r"enuresis"]),
    ("back_spine", [r"\bback\b", r"lumbar", r"spine", r"spinal", r"waist", r"sacral"]),
    ("upper_limb", [r"shoulder", r"\barm\b", r"elbow", r"wrist", r"\bhand\b", r"finger", r"forearm", r"limb", r"joint", r"muscle", r"\bbone", r"tendon", r"arthralgia", r"rheumat", r"stiffness", r"strain"]),
    ("lower_limb", [r"\bleg\b", r"thigh", r"knee", r"calf", r"ankle", r"\bfoot\b", r"\bfeet\b", r"toe", r"heel", r"sciatica", r"limb", r"joint", r"muscle", r"\bbone", r"tendon", r"arthralgia", r"rheumat", r"stiffness", r"strain"]),
    ("skin_surface", [r"skin", r"rash", r"macule", r"papule", r"pimple", r"blister", r"itch", r"urticaria", r"swelling", r"edema", r"sore", r"ulcer", r"acne", r"eczema", r"ecthyma", r"erythema", r"rubella", r"measles", r"bruise", r"bruised", r"pus", r"erosion", r"eruption", r"furuncle", r"carbuncle", r"abscess", r"\bboil"]),
    ("mental_sleep", [r"insomnia", r"\bsleep\b", r"dream", r"irritab", r"anxiety", r"fear", r"panic", r"depression", r"trance", r"cognitive", r"deliri", r"forget", r"memory", r"madness", r"mania", r"epilepsy", r"consciousness", r"coma", r"restless", r"response"]),
    ("whole_body_systemic", [r"fever", r"chill", r"sweat", r"\bcold\b", r"\bheat\b", r"hot flashes", r"body fluid", r"qi", r"jing", r"essence", r"thirst", r"polydipsia", r"fatigue", r"weakness", r"energy", r"listlessness", r"lazy talk", r"burnout", r"thinness", r"tired", r"seizure", r"convulsion", r"tremor", r"spasm", r"faint", r"lethargy", r"pulse", r"hemiplegia", r"numbness", r"tingling", r"bleeding", r"hemorrhage", r"pain", r"emaciation", r"activity", r"movement disorder"]),
]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def map_symptom(label: str) -> dict:
    normalized = normalize(label)
    categories: list[str] = []
    matched_terms: list[str] = []

    for category, patterns in KEYWORD_RULES:
        for pattern in patterns:
            if re.search(pattern, normalized):
                if category not in categories:
                    categories.append(category)
                matched_terms.append(pattern)
                break

    if not categories:
        categories = ["whole_body_systemic"]
        matched_terms = ["fallback_general"]

    region_ids: list[str] = []
    for category in categories:
        for region_id in CATEGORY_REGION_IDS[category]:
            if region_id not in region_ids:
                region_ids.append(region_id)

    return {
        "categories": categories,
        "primaryCategory": categories[0],
        "bodyRegionIds": region_ids,
        "matchedTerms": matched_terms,
    }


def main() -> None:
    symptoms = json.loads(SYMPTOMS_PATH.read_text(encoding="utf-8"))
    mappings = {}
    category_counts = {category: 0 for category in CATEGORY_REGION_IDS}

    for symptom in symptoms:
        mapped = map_symptom(str(symptom["label"]))
        mappings[str(symptom["id"])] = {
            "label": symptom["label"],
            **mapped,
        }
        for category in mapped["categories"]:
            category_counts[category] += 1

    output = {
        "source": str(SYMPTOMS_PATH.relative_to(PROJECT_ROOT)),
        "generatedBy": "tools/export_symptom_body_location_map.py",
        "description": (
            "Broad body-location categories for symptom suggestion boosting. "
            "Every symptom receives at least one category; fallback mappings are "
            "systemic/general and should be reviewed over time."
        ),
        "categoryRegionIds": CATEGORY_REGION_IDS,
        "categoryCounts": category_counts,
        "mappings": mappings,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(mappings)} symptom mappings to {OUTPUT_PATH}")
    print(json.dumps(category_counts, indent=2))


if __name__ == "__main__":
    main()
