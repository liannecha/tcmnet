"""Export frozen backend inference artifacts for browser-side prediction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "pipeline" / "artifacts"
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "src" / "generated" / "tcmnet-artifacts.json"


def read_json(filename: str):
    return json.loads((ARTIFACT_DIR / filename).read_text(encoding="utf-8"))


def read_npy(filename: str):
    return np.load(ARTIFACT_DIR / filename).astype(np.float32).tolist()


def tensor_list(state_dict, key: str):
    return state_dict[key].detach().cpu().numpy().astype(np.float32).tolist()


def main() -> None:
    state_dict = torch.load(ARTIFACT_DIR / "tcmnet.pt", map_location="cpu")
    payload = {
        "model_config": read_json("model_config.json"),
        "symptom_mapping": read_json("symptom_columns.json"),
        "syndrome_mapping": read_json("syndrome_index_to_id.json"),
        "concept_mapping": read_json("concept_labels.json"),
        "herb_mapping": read_json("herb_mapping.json"),
        "symptoms_metadata": read_json("symptoms_metadata.json"),
        "syndromes_metadata": read_json("syndromes_metadata.json"),
        "herbs_metadata": read_json("herbs_metadata.json"),
        "concepts_metadata": read_json("concepts_metadata.json"),
        "herb_concept_matrix": read_npy("herb_concept_matrix.npy"),
        "syndrome_herb_prior": read_npy("syndrome_herb_prior.npy"),
        "weights": {
            "shared_weight": tensor_list(state_dict, "shared_layer.0.weight"),
            "shared_bias": tensor_list(state_dict, "shared_layer.0.bias"),
            "concept_weight": tensor_list(state_dict, "concept_head.weight"),
            "concept_bias": tensor_list(state_dict, "concept_head.bias"),
            "syndrome_hidden_weight": tensor_list(state_dict, "syndrome_head.0.weight"),
            "syndrome_hidden_bias": tensor_list(state_dict, "syndrome_head.0.bias"),
            "syndrome_output_weight": tensor_list(state_dict, "syndrome_head.3.weight"),
            "syndrome_output_bias": tensor_list(state_dict, "syndrome_head.3.bias"),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
