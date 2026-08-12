"""Inference utilities for frozen TCMNet artifacts.

This module intentionally does not train models and does not start an API server.
It only loads exported artifacts and runs local predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "pipeline" / "artifacts"


class TCMNet(nn.Module):
    def __init__(
        self,
        num_symptoms: int,
        num_concepts: int,
        num_syndromes: int,
        shared_hidden: int = 512,
        syndrome_hidden: int = 256,
        shared_dropout: float = 0.3,
        syndrome_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(num_symptoms, shared_hidden),
            nn.ReLU(),
            nn.Dropout(shared_dropout),
        )
        self.concept_head = nn.Linear(shared_hidden, num_concepts)
        self.syndrome_head = nn.Sequential(
            nn.Linear(shared_hidden + num_concepts, syndrome_hidden),
            nn.ReLU(),
            nn.Dropout(syndrome_dropout),
            nn.Linear(syndrome_hidden, num_syndromes),
        )

    def forward(self, x):
        shared_features = self.shared_layer(x)
        concept_preds = torch.sigmoid(self.concept_head(shared_features))
        combined_features = torch.cat((shared_features, concept_preds), dim=1)
        syndrome_preds = self.syndrome_head(combined_features)
        return concept_preds, syndrome_preds


class TCMNetInference:
    def __init__(
        self,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        device: str | torch.device = "cpu",
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        if not self.artifact_dir.is_absolute():
            self.artifact_dir = PROJECT_ROOT / self.artifact_dir
        self.device = torch.device(device)

        self.model_config = self._read_json("model_config.json")
        self.symptom_mapping = self._read_json("symptom_columns.json")
        self.syndrome_mapping = self._read_json("syndrome_index_to_id.json")
        self.concept_mapping = self._read_json("concept_labels.json")
        self.herb_mapping = self._read_json("herb_mapping.json")
        self.symptoms_metadata = self._read_json("symptoms_metadata.json")
        self.syndromes_metadata = self._read_json("syndromes_metadata.json")
        self.herbs_metadata = self._read_json("herbs_metadata.json")
        self.concepts_metadata = self._read_json("concepts_metadata.json")
        self.herb_concept_matrix = self._read_npy("herb_concept_matrix.npy")
        self.syndrome_herb_prior = self._read_npy("syndrome_herb_prior.npy")

        self.symptom_columns = [str(item) for item in self.symptom_mapping["columns"]]
        self.symptom_id_to_index = {
            str(key): int(value)
            for key, value in self.symptom_mapping["symptom_id_to_index"].items()
        }
        self.syndrome_index_to_id = self.syndrome_mapping["index_to_id"]
        self.concept_labels = self.concept_mapping["labels"]
        self.herb_ids = self.herb_mapping["herb_ids"]
        self.recommendation_alpha = float(
            self.herb_mapping.get("recommendation_alpha_default", 0.7)
        )
        self.symptom_label_by_id = self._metadata_label_map(self.symptoms_metadata)
        self.syndrome_label_by_id = self._metadata_label_map(self.syndromes_metadata)
        self.herb_label_by_id = self._metadata_label_map(self.herbs_metadata)

        self._validate_artifacts()
        self.model = self._load_model()

    def predict(
        self,
        symptom_ids: Iterable[str],
        top_syndromes: int = 5,
        top_herbs: int = 5,
        herb_alpha: float | None = None,
    ) -> dict:
        requested_symptoms = [str(symptom_id) for symptom_id in symptom_ids]
        input_vector, known_symptoms, unknown_symptoms = self._vectorize(
            requested_symptoms
        )

        with torch.no_grad():
            x = torch.tensor(input_vector, dtype=torch.float32, device=self.device)
            concept_scores, syndrome_logits = self.model(x.unsqueeze(0))
            concept_array = concept_scores.squeeze(0).cpu().numpy()
            syndrome_probs = torch.softmax(syndrome_logits, dim=1).squeeze(0).cpu().numpy()

        syndrome_predictions = self._top_syndromes(syndrome_probs, top_syndromes)
        pred_syndrome_idx = syndrome_predictions[0]["index"] if syndrome_predictions else 0
        alpha = self.recommendation_alpha if herb_alpha is None else herb_alpha
        herb_recommendations = self._recommend_herbs(
            concept_array,
            pred_syndrome_idx=pred_syndrome_idx,
            top_k=top_herbs,
            alpha=alpha,
        )
        concepts = [
            {"id": label, "label": label, "score": float(score)}
            for label, score in zip(self.concept_labels, concept_array)
        ]

        return {
            "input": {
                "requested_symptom_ids": requested_symptoms,
                "known_symptom_ids": known_symptoms,
                "unknown_symptom_ids": unknown_symptoms,
            },
            "syndromes": syndrome_predictions,
            "concepts": concepts,
            "herbs": herb_recommendations,
            "explanation": self._build_explanation(
                known_symptoms=known_symptoms,
                concepts=concepts,
                pred_syndrome_idx=pred_syndrome_idx,
                herb_recommendations=herb_recommendations,
                alpha=alpha,
            ),
        }

    def _load_model(self) -> TCMNet:
        config = self.model_config
        model = TCMNet(
            num_symptoms=int(config["num_symptoms"]),
            num_concepts=int(config["num_concepts"]),
            num_syndromes=int(config["num_syndromes"]),
            shared_hidden=int(config.get("shared_hidden", 512)),
            syndrome_hidden=int(config.get("syndrome_hidden", 256)),
            shared_dropout=float(config.get("shared_dropout", 0.3)),
            syndrome_dropout=float(config.get("syndrome_dropout", 0.2)),
        )
        weights_path = self.artifact_dir / "tcmnet.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing artifact: {weights_path}")
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _vectorize(self, symptom_ids: Iterable[str]) -> tuple[np.ndarray, list[str], list[str]]:
        vector = np.zeros(len(self.symptom_columns), dtype=np.float32)
        known: list[str] = []
        unknown: list[str] = []

        for symptom_id in symptom_ids:
            lookup_key = self._lookup_symptom_key(symptom_id)
            if lookup_key is None:
                unknown.append(str(symptom_id))
                continue
            vector[self.symptom_id_to_index[lookup_key]] = 1.0
            known.append(str(symptom_id))

        return vector, known, unknown

    def _display_symptom_id(self, symptom_id: str) -> str:
        lookup_key = self._lookup_symptom_key(symptom_id)
        if lookup_key is None:
            return str(symptom_id)
        if str(lookup_key).isdigit():
            return f"SMTS{int(lookup_key):05d}"
        return str(lookup_key)

    def _lookup_symptom_key(self, symptom_id: str) -> str | None:
        raw = str(symptom_id).strip()
        candidates = [raw]

        upper = raw.upper()
        if upper.startswith("SMTS"):
            numeric = upper.removeprefix("SMTS").lstrip("0") or "0"
            candidates.extend([numeric, f"SMTS{int(numeric):05d}"])
        elif raw.isdigit():
            numeric = raw.lstrip("0") or "0"
            candidates.extend([numeric, f"SMTS{int(numeric):05d}"])

        for candidate in candidates:
            if candidate in self.symptom_id_to_index:
                return candidate
        return None

    def _top_syndromes(self, syndrome_probs: np.ndarray, top_k: int) -> list[dict]:
        limit = min(top_k, len(self.syndrome_index_to_id))
        indices = np.argsort(syndrome_probs)[::-1][:limit]
        return [
            {
                "index": int(index),
                "syndrome_id": self.syndrome_index_to_id[int(index)],
                "label": self.syndrome_label_by_id.get(
                    self.syndrome_index_to_id[int(index)],
                    self.syndrome_index_to_id[int(index)],
                ),
                "confidence": float(syndrome_probs[int(index)]),
            }
            for index in indices
        ]

    def _recommend_herbs(
        self,
        concept_scores: np.ndarray,
        pred_syndrome_idx: int,
        top_k: int,
        alpha: float,
    ) -> list[dict]:
        concept_similarity = (self.herb_concept_matrix @ concept_scores) / max(
            len(concept_scores), 1
        )
        prior = self.syndrome_herb_prior[pred_syndrome_idx]
        final_scores = alpha * concept_similarity + (1.0 - alpha) * prior
        used_for_syndrome = prior > 0
        if used_for_syndrome.any():
            candidate_indices = np.where(used_for_syndrome)[0]
        else:
            candidate_indices = np.arange(len(self.herb_ids))

        sorted_candidates = candidate_indices[
            np.argsort(final_scores[candidate_indices])[::-1]
        ][:top_k]
        return [
            {
                "herb_id": self.herb_ids[int(index)],
                "label": self.herb_label_by_id.get(
                    self.herb_ids[int(index)],
                    self.herb_ids[int(index)],
                ),
                "score": float(final_scores[int(index)]),
                "concept_similarity": float(concept_similarity[int(index)]),
                "syndrome_prior": float(prior[int(index)]),
                "known_for_predicted_syndrome": bool(prior[int(index)] > 0),
            }
            for index in sorted_candidates
        ]

    def _build_explanation(
        self,
        known_symptoms: list[str],
        concepts: list[dict],
        pred_syndrome_idx: int,
        herb_recommendations: list[dict],
        alpha: float,
    ) -> dict:
        matched_symptoms = []
        for symptom_id in known_symptoms:
            display_id = self._display_symptom_id(symptom_id)
            matched_symptoms.append(
                {
                    "id": display_id,
                    "label": self.symptom_label_by_id.get(display_id, display_id),
                }
            )

        top_concepts = sorted(
            concepts,
            key=lambda concept: concept["score"],
            reverse=True,
        )[:5]

        syndrome_id = self.syndrome_index_to_id[pred_syndrome_idx]
        prior = self.syndrome_herb_prior[pred_syndrome_idx]
        associated_indices = np.where(prior > 0)[0]
        associated_herbs = [
            {
                "id": self.herb_ids[int(index)],
                "label": self.herb_label_by_id.get(
                    self.herb_ids[int(index)],
                    self.herb_ids[int(index)],
                ),
            }
            for index in associated_indices[:12]
        ]

        return {
            "matching_symptoms": matched_symptoms,
            "concept_alignment": top_concepts,
            "syndrome_herb_associations": {
                "syndrome_id": syndrome_id,
                "label": self.syndrome_label_by_id.get(syndrome_id, syndrome_id),
                "associated_herbs": associated_herbs,
                "total_associated_herbs": int(len(associated_indices)),
            },
            "herb_ranking": {
                "formula": "score = alpha * concept_similarity + (1 - alpha) * syndrome_prior",
                "alpha": float(alpha),
                "items": [
                    {
                        "herb_id": herb["herb_id"],
                        "label": herb["label"],
                        "concept_similarity": herb["concept_similarity"],
                        "syndrome_prior": herb["syndrome_prior"],
                        "score": herb["score"],
                        "known_for_predicted_syndrome": herb[
                            "known_for_predicted_syndrome"
                        ],
                    }
                    for herb in herb_recommendations
                ],
            },
        }

    def _validate_artifacts(self) -> None:
        expected_symptoms = int(self.model_config["num_symptoms"])
        expected_concepts = int(self.model_config["num_concepts"])
        expected_syndromes = int(self.model_config["num_syndromes"])

        if len(self.symptom_columns) != expected_symptoms:
            raise ValueError(
                "Symptom mapping length does not match model_config "
                f"({len(self.symptom_columns)} != {expected_symptoms})."
            )
        if len(self.syndrome_index_to_id) != expected_syndromes:
            raise ValueError(
                "Syndrome mapping length does not match model_config "
                f"({len(self.syndrome_index_to_id)} != {expected_syndromes})."
            )
        if len(self.concept_labels) != expected_concepts:
            raise ValueError(
                "Concept label length does not match model_config "
                f"({len(self.concept_labels)} != {expected_concepts})."
            )
        metadata_lengths = {
            "symptoms_metadata.json": (
                len(self.symptoms_metadata),
                expected_symptoms,
            ),
            "syndromes_metadata.json": (
                len(self.syndromes_metadata),
                expected_syndromes,
            ),
            "herbs_metadata.json": (len(self.herbs_metadata), len(self.herb_ids)),
            "concepts_metadata.json": (
                len(self.concepts_metadata),
                expected_concepts,
            ),
        }
        for filename, (actual, expected) in metadata_lengths.items():
            if actual != expected:
                raise ValueError(
                    f"{filename} length does not match artifacts ({actual} != {expected})."
                )
        if self.herb_concept_matrix.shape != (len(self.herb_ids), expected_concepts):
            raise ValueError(
                "Herb concept matrix shape does not match herb/concept mappings "
                f"({self.herb_concept_matrix.shape} != "
                f"({len(self.herb_ids)}, {expected_concepts}))."
            )
        if self.syndrome_herb_prior.shape != (
            expected_syndromes,
            len(self.herb_ids),
        ):
            raise ValueError(
                "Syndrome-herb prior shape does not match syndrome/herb mappings "
                f"({self.syndrome_herb_prior.shape} != "
                f"({expected_syndromes}, {len(self.herb_ids)}))."
            )

    @staticmethod
    def _metadata_label_map(records: list[dict]) -> dict[str, str]:
        return {
            str(record["id"]): str(record.get("label") or record["id"])
            for record in records
        }

    def _read_json(self, filename: str):
        path = self.artifact_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_npy(self, filename: str) -> np.ndarray:
        path = self.artifact_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")
        return np.load(path)


def _default_smoke_symptoms(inference: TCMNetInference, count: int = 3) -> list[str]:
    symptoms = []
    for symptom_id in inference.symptom_columns[:count]:
        if str(symptom_id).isdigit():
            symptoms.append(f"SMTS{int(symptom_id):05d}")
        else:
            symptoms.append(str(symptom_id))
    return symptoms


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TCMNet inference smoke test.")
    parser.add_argument("symptom_ids", nargs="*", help="Symptom IDs, e.g. SMTS00012")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--top-syndromes", type=int, default=3)
    parser.add_argument("--top-herbs", type=int, default=5)
    args = parser.parse_args()

    inference = TCMNetInference(artifact_dir=args.artifact_dir)
    symptom_ids = args.symptom_ids or _default_smoke_symptoms(inference)
    prediction = inference.predict(
        symptom_ids,
        top_syndromes=args.top_syndromes,
        top_herbs=args.top_herbs,
    )
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
