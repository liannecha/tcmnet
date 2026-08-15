"""Train TCMNet and export app-ready inference artifacts.

Run from the project root:
    python3 pipeline/training/export_tcmnet_artifacts.py
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.baselines.herb_matrix_ranker import (
    DEFAULT_RECOMMENDATION_ALPHA,
    rank_herbs_by_matrix_baseline,
)


DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "pipeline" / "artifacts"


class SyntheticTCMDataset(Dataset):
    def __init__(
        self,
        synthetic_x_file: Path,
        synthetic_y_file: Path,
        concept_file: Path,
        syndrome_herb_file: Path,
    ) -> None:
        self.x_df = pd.read_csv(synthetic_x_file)
        self.y_df = pd.read_csv(synthetic_y_file)
        self.concept_df = pd.read_csv(concept_file, index_col=0)
        self.syndrome_herb_df = pd.read_csv(syndrome_herb_file, index_col=0)
        self.herb_ids = [str(column) for column in self.syndrome_herb_df.columns]

        raw_labels = self.y_df["Syndrome_id"].astype(int).to_numpy()
        self.raw_syndrome_labels = raw_labels
        self.raw_label_order = sorted(np.unique(raw_labels).tolist())
        self.raw_to_model_index = {
            raw_label: model_index
            for model_index, raw_label in enumerate(self.raw_label_order)
        }
        encoded_labels = np.array(
            [self.raw_to_model_index[label] for label in raw_labels],
            dtype=np.int64,
        )

        missing_rows = [
            label for label in self.raw_label_order if label >= len(self.concept_df)
        ]
        if missing_rows:
            raise ValueError(
                "Synthetic labels reference concept rows that do not exist: "
                f"{missing_rows[:10]}"
            )

        concept_targets = self.concept_df.iloc[self.raw_label_order].to_numpy(
            dtype=np.float32
        )[encoded_labels]
        syndrome_ids = [
            str(self.concept_df.index[raw_label]) for raw_label in self.raw_label_order
        ]
        missing_herb_rows = [
            syndrome_id
            for syndrome_id in syndrome_ids
            if syndrome_id not in self.syndrome_herb_df.index
        ]
        if missing_herb_rows:
            raise ValueError(
                "Synthetic labels reference herb target rows that do not exist: "
                f"{missing_herb_rows[:10]}"
            )

        herb_targets = self.syndrome_herb_df.reindex(syndrome_ids).to_numpy(
            dtype=np.float32
        )[encoded_labels]

        self.x = torch.tensor(self.x_df.to_numpy(dtype=np.float32), dtype=torch.float32)
        self.y_syndrome = torch.tensor(encoded_labels, dtype=torch.long)
        self.concept_targets = torch.tensor(concept_targets, dtype=torch.float32)
        self.herb_targets = torch.tensor(herb_targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            self.x[idx],
            self.concept_targets[idx],
            self.y_syndrome[idx],
            self.herb_targets[idx],
        )


class TCMNet(nn.Module):
    def __init__(
        self,
        num_symptoms: int,
        num_concepts: int,
        num_syndromes: int,
        num_herbs: int,
        shared_hidden: int = 512,
        syndrome_hidden: int = 256,
        herb_hidden: int = 256,
        shared_dropout: float = 0.3,
        syndrome_dropout: float = 0.2,
        herb_dropout: float = 0.2,
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
        self.herb_head = nn.Sequential(
            nn.Linear(shared_hidden + num_concepts + num_syndromes, herb_hidden),
            nn.ReLU(),
            nn.Dropout(herb_dropout),
            nn.Linear(herb_hidden, num_herbs),
        )

    def forward(self, x):
        shared_features = self.shared_layer(x)
        concept_preds = torch.sigmoid(self.concept_head(shared_features))
        combined_features = torch.cat((shared_features, concept_preds), dim=1)
        syndrome_logits = self.syndrome_head(combined_features)
        herb_input = torch.cat((shared_features, concept_preds, syndrome_logits), dim=1)
        herb_scores = self.herb_head(herb_input)
        return concept_preds, syndrome_logits, herb_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TCMNet and save inference artifacts."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--lambda-concept", type=float, default=10.0)
    parser.add_argument("--lambda-syndrome", type=float, default=1.0)
    parser.add_argument("--lambda-herb", type=float, default=1.0)
    parser.add_argument("--lambda-herb-bpr", type=float, default=1.0)
    parser.add_argument("--lambda-herb-bce", type=float, default=0.25)
    parser.add_argument("--lambda-herb-distill", type=float, default=0.5)
    parser.add_argument("--herb-pairs-per-sample", type=int, default=8)
    parser.add_argument("--herb-hard-negative-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=229)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args()


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def herb_bpr_loss(
    herb_scores: torch.Tensor,
    true_herbs: torch.Tensor,
    pairs_per_sample: int = 8,
    hard_negative_ratio: float = 0.5,
) -> torch.Tensor:
    """Sample positive/random-hard negative herb pairs and compute BPR loss."""
    true_herbs = true_herbs.to(device=herb_scores.device)
    positive_scores = []
    negative_scores = []
    pairs_per_sample = max(int(pairs_per_sample), 0)
    hard_negative_ratio = min(max(float(hard_negative_ratio), 0.0), 1.0)
    if pairs_per_sample == 0:
        return herb_scores.sum() * 0.0

    for sample_scores, sample_targets in zip(herb_scores, true_herbs):
        positive_indices = torch.nonzero(sample_targets > 0, as_tuple=False).flatten()
        negative_indices = torch.nonzero(sample_targets <= 0, as_tuple=False).flatten()
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue

        pair_count = pairs_per_sample
        hard_count = min(
            negative_indices.numel(),
            int(round(pair_count * hard_negative_ratio)),
        )
        random_count = pair_count - hard_count

        positive_choices = positive_indices[
            torch.randint(
                positive_indices.numel(),
                (pair_count,),
                device=herb_scores.device,
            )
        ]

        negative_choices = []
        if hard_count > 0:
            hard_pool = negative_indices[
                sample_scores[negative_indices].topk(hard_count).indices
            ]
            negative_choices.append(hard_pool)
        if random_count > 0:
            random_choices = negative_indices[
                torch.randint(
                    negative_indices.numel(),
                    (random_count,),
                    device=herb_scores.device,
                )
            ]
            negative_choices.append(random_choices)

        if not negative_choices:
            continue

        negative_choice = torch.cat(negative_choices)
        if negative_choice.numel() < pair_count:
            refill = negative_indices[
                torch.randint(
                    negative_indices.numel(),
                    (pair_count - negative_choice.numel(),),
                    device=herb_scores.device,
                )
            ]
            negative_choice = torch.cat((negative_choice, refill))

        positive_scores.append(sample_scores[positive_choices])
        negative_scores.append(sample_scores[negative_choice[:pair_count]])

    if not positive_scores:
        return herb_scores.sum() * 0.0

    pos = torch.cat(positive_scores)
    neg = torch.cat(negative_scores)
    return -torch.nn.functional.logsigmoid(pos - neg).mean()


def herb_topk_metrics_from_scores(
    herb_scores: torch.Tensor,
    true_herbs: torch.Tensor,
    top_ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Compute top-k metrics for any herb scoring method against multi-hot targets."""
    accum = {
        k: {"precision": 0.0, "recall": 0.0, "hit": 0.0, "count": 0}
        for k in top_ks
    }
    for sample_scores, sample_targets in zip(herb_scores, true_herbs):
        positive_indices = torch.nonzero(sample_targets > 0, as_tuple=False).flatten()
        if positive_indices.numel() == 0:
            continue

        positive_set = set(positive_indices.cpu().numpy().tolist())
        for k, values in accum.items():
            limit = min(k, sample_scores.numel())
            top_indices = sample_scores.topk(limit).indices.cpu().numpy().tolist()
            hits = sum(index in positive_set for index in top_indices)
            values["precision"] += hits / limit
            values["recall"] += hits / positive_indices.numel()
            values["hit"] += 1.0 if hits > 0 else 0.0
            values["count"] += 1

    metrics = {}
    for k, values in accum.items():
        count = values["count"]
        metrics[f"herb_precision_at_{k}"] = (
            0.0 if count == 0 else values["precision"] / count
        )
        metrics[f"herb_recall_at_{k}"] = 0.0 if count == 0 else values["recall"] / count
        metrics[f"herb_hit_at_{k}"] = 0.0 if count == 0 else values["hit"] / count
    return metrics


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def baseline_teacher_probs(
    true_concepts: torch.Tensor,
    true_syndromes: torch.Tensor,
    herb_concept_matrix: torch.Tensor,
    syndrome_herb_prior: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Build stable baseline teacher probabilities from true concepts/syndromes."""
    herb_concept_matrix = herb_concept_matrix.to(device=true_concepts.device)
    syndrome_herb_prior = syndrome_herb_prior.to(device=true_concepts.device)
    concept_similarity = true_concepts @ herb_concept_matrix.T
    concept_similarity = concept_similarity / max(true_concepts.shape[1], 1)
    prior = syndrome_herb_prior[true_syndromes.long()]
    teacher_scores = alpha * concept_similarity + (1.0 - alpha) * prior

    row_min = teacher_scores.min(dim=1, keepdim=True).values
    row_max = teacher_scores.max(dim=1, keepdim=True).values
    score_range = row_max - row_min
    normalized = (teacher_scores - row_min) / score_range.clamp_min(1e-8)
    return torch.where(score_range > 0, normalized, torch.zeros_like(normalized))


def herb_hybrid_loss(
    herb_scores: torch.Tensor,
    true_herbs: torch.Tensor,
    true_concepts: torch.Tensor,
    true_syndromes: torch.Tensor,
    herb_concept_matrix: torch.Tensor,
    syndrome_herb_prior: torch.Tensor,
    alpha: float,
    pairs_per_sample: int,
    hard_negative_ratio: float,
    lambda_bpr: float,
    lambda_bce: float,
    lambda_distill: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine BPR, multi-label BCE, and baseline-score distillation losses."""
    true_herbs = true_herbs.to(device=herb_scores.device).float()
    true_concepts = true_concepts.to(device=herb_scores.device).float()
    true_syndromes = true_syndromes.to(device=herb_scores.device)
    bpr_loss = herb_bpr_loss(
        herb_scores,
        true_herbs,
        pairs_per_sample=pairs_per_sample,
        hard_negative_ratio=hard_negative_ratio,
    )
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        herb_scores,
        true_herbs,
    )
    teacher_probs = baseline_teacher_probs(
        true_concepts=true_concepts,
        true_syndromes=true_syndromes,
        herb_concept_matrix=herb_concept_matrix,
        syndrome_herb_prior=syndrome_herb_prior,
        alpha=alpha,
    )
    distill_loss = torch.nn.functional.mse_loss(
        torch.sigmoid(herb_scores),
        teacher_probs,
    )
    total = (
        lambda_bpr * bpr_loss
        + lambda_bce * bce_loss
        + lambda_distill * distill_loss
    )
    return total, bpr_loss, bce_loss, distill_loss


def train_model(
    model: TCMNet,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    class_weights: torch.Tensor,
    lambda_concept: float,
    lambda_syndrome: float,
    lambda_herb: float,
    lambda_herb_bpr: float,
    lambda_herb_bce: float,
    lambda_herb_distill: float,
    herb_pairs_per_sample: int,
    herb_hard_negative_ratio: float,
    herb_concept_matrix: torch.Tensor,
    syndrome_herb_prior: torch.Tensor,
    recommendation_alpha: float,
) -> dict[str, list[float]]:
    criterion_concept = nn.MSELoss()
    criterion_syndrome = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {
        "total": [],
        "concept": [],
        "syndrome": [],
        "herb_total": [],
        "herb_bpr": [],
        "herb_bce": [],
        "herb_distill": [],
    }

    model.train()
    for epoch in range(epochs):
        epoch_total = 0.0
        epoch_concept = 0.0
        epoch_syndrome = 0.0
        epoch_herb_total = 0.0
        epoch_herb_bpr = 0.0
        epoch_herb_bce = 0.0
        epoch_herb_distill = 0.0

        for symptoms, true_concepts, true_syndromes, true_herbs in dataloader:
            optimizer.zero_grad()
            pred_concepts, pred_syndromes, pred_herbs = model(symptoms.float())

            raw_loss_concept = criterion_concept(pred_concepts, true_concepts.float())
            raw_loss_syndrome = criterion_syndrome(
                pred_syndromes, true_syndromes.long()
            )
            (
                raw_loss_herb,
                raw_loss_herb_bpr,
                raw_loss_herb_bce,
                raw_loss_herb_distill,
            ) = herb_hybrid_loss(
                herb_scores=pred_herbs,
                true_herbs=true_herbs,
                true_concepts=true_concepts,
                true_syndromes=true_syndromes,
                herb_concept_matrix=herb_concept_matrix,
                syndrome_herb_prior=syndrome_herb_prior,
                alpha=recommendation_alpha,
                pairs_per_sample=herb_pairs_per_sample,
                hard_negative_ratio=herb_hard_negative_ratio,
                lambda_bpr=lambda_herb_bpr,
                lambda_bce=lambda_herb_bce,
                lambda_distill=lambda_herb_distill,
            )
            loss = (
                raw_loss_concept * lambda_concept
                + raw_loss_syndrome * lambda_syndrome
                + raw_loss_herb * lambda_herb
            )
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_concept += raw_loss_concept.item()
            epoch_syndrome += raw_loss_syndrome.item()
            epoch_herb_total += raw_loss_herb.item()
            epoch_herb_bpr += raw_loss_herb_bpr.item()
            epoch_herb_bce += raw_loss_herb_bce.item()
            epoch_herb_distill += raw_loss_herb_distill.item()

        history["total"].append(epoch_total / len(dataloader))
        history["concept"].append(epoch_concept / len(dataloader))
        history["syndrome"].append(epoch_syndrome / len(dataloader))
        history["herb_total"].append(epoch_herb_total / len(dataloader))
        history["herb_bpr"].append(epoch_herb_bpr / len(dataloader))
        history["herb_bce"].append(epoch_herb_bce / len(dataloader))
        history["herb_distill"].append(epoch_herb_distill / len(dataloader))
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {history['total'][-1]:.4f}")

    return history


def evaluate_model(model: TCMNet, dataloader: DataLoader) -> dict[str, float]:
    model.eval()
    criterion_concept = nn.MSELoss()
    correct = 0
    top5_correct = 0
    total = 0
    concept_error = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    herb_score_batches = []
    herb_target_batches = []

    with torch.no_grad():
        for symptoms, true_concepts, true_syndromes, true_herbs in dataloader:
            pred_concepts, pred_syndromes, pred_herbs = model(symptoms.float())
            predicted = torch.argmax(pred_syndromes, dim=1)
            top_k = min(5, pred_syndromes.shape[1])
            top5 = pred_syndromes.topk(top_k, dim=1).indices

            correct += (predicted == true_syndromes).sum().item()
            top5_correct += (
                top5 == true_syndromes.unsqueeze(1)
            ).any(dim=1).sum().item()
            total += true_syndromes.size(0)
            concept_error += criterion_concept(pred_concepts, true_concepts).item()
            y_true.extend(true_syndromes.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
            herb_score_batches.append(pred_herbs.cpu())
            herb_target_batches.append(true_herbs.cpu())

    metrics = {
        "accuracy": correct / total,
        "top5_accuracy": top5_correct / total,
        "macro_f1": macro_f1(y_true, y_pred),
        "concept_mse": concept_error / len(dataloader),
    }
    if herb_score_batches:
        metrics.update(
            herb_topk_metrics_from_scores(
                torch.cat(herb_score_batches),
                torch.cat(herb_target_batches),
            )
        )
    return metrics


def evaluate_inference_herb_rankers(
    model: TCMNet,
    dataloader: DataLoader,
    herb_concept_matrix: np.ndarray,
    syndrome_herb_prior: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Compare neural and baseline herb ranking under inference-time inputs."""
    model.eval()
    neural_score_batches = []
    baseline_score_batches = []
    herb_target_batches = []

    with torch.no_grad():
        for symptoms, _true_concepts, _true_syndromes, true_herbs in dataloader:
            concept_scores, syndrome_logits, herb_scores = model(symptoms.float())
            syndrome_probs = torch.softmax(syndrome_logits, dim=1)
            pred_syndromes = torch.argmax(syndrome_probs, dim=1)

            neural_score_batches.append(herb_scores.cpu())
            herb_target_batches.append(true_herbs.cpu())

            baseline_scores = []
            for sample_concepts, pred_syndrome_idx in zip(
                concept_scores.cpu().numpy(),
                pred_syndromes.cpu().numpy(),
            ):
                ranking = rank_herbs_by_matrix_baseline(
                    concept_scores=sample_concepts,
                    pred_syndrome_idx=int(pred_syndrome_idx),
                    herb_concept_matrix=herb_concept_matrix,
                    syndrome_herb_prior=syndrome_herb_prior,
                    top_k=herb_concept_matrix.shape[0],
                    alpha=alpha,
                )
                sample_scores = np.full(
                    herb_concept_matrix.shape[0],
                    -np.inf,
                    dtype=np.float32,
                )
                sample_scores[ranking.indices] = ranking.scores[ranking.indices]
                baseline_scores.append(sample_scores)

            baseline_score_batches.append(torch.tensor(np.stack(baseline_scores)))

    neural_metrics = herb_topk_metrics_from_scores(
        torch.cat(neural_score_batches),
        torch.cat(herb_target_batches),
    )
    baseline_metrics = herb_topk_metrics_from_scores(
        torch.cat(baseline_score_batches),
        torch.cat(herb_target_batches),
    )
    return {
        **prefixed_metrics("neural_inference", neural_metrics),
        **prefixed_metrics("baseline_inference", baseline_metrics),
    }


def balanced_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = labels.size / (present.sum() * counts[present])
    return weights


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    classes = sorted(set(y_true) | set(y_pred))
    if not classes:
        return 0.0

    f1_scores = []
    for class_id in classes:
        true_positive = sum(
            true == class_id and pred == class_id
            for true, pred in zip(y_true, y_pred)
        )
        false_positive = sum(
            true != class_id and pred == class_id
            for true, pred in zip(y_true, y_pred)
        )
        false_negative = sum(
            true == class_id and pred != class_id
            for true, pred in zip(y_true, y_pred)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)

    return float(np.mean(f1_scores))


def normalized_concept_label(column_name: str) -> str:
    return column_name.removeprefix("Is_")


def build_herb_concept_matrix(
    concept_columns: list[str],
    location_file: Path,
    principles_file: Path,
    smhb_excel_file: Path,
    syndrome_herb_file: Path,
) -> tuple[np.ndarray, list[str]]:
    loc_df = pd.read_csv(location_file).set_index("Herb_id")
    loc_df = loc_df[~loc_df.index.duplicated(keep="first")]
    loc_df.columns = [
        "Is_Wood",
        "Is_Fire",
        "Is_Earth",
        "Is_Metal",
        "Is_Water",
        "Is_Reproductive",
    ]

    principles_df = pd.read_csv(principles_file).set_index("HERBDB_ID")
    principles_df = principles_df[
        ["hot", "cold", "internal", "external", "yin", "yang", "excess", "deficiency"]
    ]

    excel_df = pd.read_excel(smhb_excel_file)[["Herb_id", "HERBDB_ID"]].dropna()
    excel_df["HERBDB_ID"] = excel_df["HERBDB_ID"].astype(str).str.split("|")
    excel_df = excel_df.explode("HERBDB_ID")
    excel_df["HERBDB_ID"] = excel_df["HERBDB_ID"].str.strip()
    excel_df["Herb_id"] = excel_df["Herb_id"].astype(int)
    id_map = excel_df.set_index("HERBDB_ID")["Herb_id"]

    principles_df["Herb_id"] = principles_df.index.map(id_map)
    principles_df = principles_df.dropna(subset=["Herb_id"])
    principles_df["Herb_id"] = principles_df["Herb_id"].astype(int)
    principles_df = principles_df.set_index("Herb_id")
    principles_df = principles_df[~principles_df.index.duplicated(keep="first")]

    syndrome_herb_df = pd.read_csv(syndrome_herb_file, index_col=0)
    herb_ids = syndrome_herb_df.columns.tolist()
    herb_ids_numeric = [int(herb_id.replace("SMHB", "")) for herb_id in herb_ids]

    loc_aligned = loc_df.reindex(herb_ids_numeric).fillna(0)
    principles_aligned = principles_df.reindex(herb_ids_numeric).fillna(0)
    herb_features = pd.concat([loc_aligned, principles_aligned], axis=1)
    herb_features = herb_features.reindex(columns=concept_columns).fillna(0)
    return herb_features.to_numpy(dtype=np.float32), herb_ids


def build_syndrome_herb_prior(
    syndrome_herb_file: Path,
    syndrome_index_to_id: list[str],
) -> np.ndarray:
    syndrome_herb_df = pd.read_csv(syndrome_herb_file, index_col=0)
    syndrome_herb_df = syndrome_herb_df.reindex(syndrome_index_to_id).fillna(0)
    return syndrome_herb_df.to_numpy(dtype=np.float32)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def copy_static_metadata_artifacts(artifact_dir: Path) -> None:
    """Carry app metadata into non-default export dirs for inference smoke tests."""
    metadata_files = [
        "symptoms_metadata.json",
        "syndromes_metadata.json",
        "herbs_metadata.json",
        "concepts_metadata.json",
        "symptom_english_names.json",
    ]
    for filename in metadata_files:
        source = DEFAULT_ARTIFACT_DIR / filename
        destination = artifact_dir / filename
        if not source.exists() or source.resolve() == destination.resolve():
            continue
        shutil.copyfile(source, destination)


def export_artifacts(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    synthetic_x_file = project_path("pipeline/data/patient/Synthetic_Patient_Symptoms.csv")
    synthetic_y_file = project_path("pipeline/data/patient/Synthetic_Patient_Labels.csv")
    concept_file = project_path("pipeline/data/processed/Syndrome_Concept_Targets.csv")
    syndrome_herb_file = project_path("pipeline/data/processed/Syndrome_Herb_Targets.csv")

    dataset = SyntheticTCMDataset(
        synthetic_x_file,
        synthetic_y_file,
        concept_file,
        syndrome_herb_file,
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    class_weights = balanced_class_weights(
        labels=dataset.y_syndrome[train_dataset.indices].numpy(),
        num_classes=len(dataset.raw_label_order),
    )
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
    concept_columns = dataset.concept_df.columns.tolist()
    concept_labels = [normalized_concept_label(column) for column in concept_columns]
    syndrome_index_to_id = [
        str(dataset.concept_df.index[raw_label]) for raw_label in dataset.raw_label_order
    ]
    symptom_columns = [str(column) for column in dataset.x_df.columns.tolist()]
    symptom_id_to_index = {
        symptom_id: index for index, symptom_id in enumerate(symptom_columns)
    }
    raw_label_to_syndrome_id = {
        str(raw_label): syndrome_id
        for raw_label, syndrome_id in zip(dataset.raw_label_order, syndrome_index_to_id)
    }
    herb_concept_matrix, herb_ids = build_herb_concept_matrix(
        concept_columns=concept_columns,
        location_file=project_path("pipeline/data/processed/Herb_Location_Features.csv"),
        principles_file=project_path(
            "pipeline/data/processed/Herb_Eight_Principles_Multihot.csv"
        ),
        smhb_excel_file=project_path(
            "pipeline/data/original/SymMap v2.0, SMHB file.xlsx"
        ),
        syndrome_herb_file=syndrome_herb_file,
    )
    syndrome_herb_prior = build_syndrome_herb_prior(
        syndrome_herb_file=syndrome_herb_file,
        syndrome_index_to_id=syndrome_index_to_id,
    )
    herb_concept_tensor = torch.tensor(herb_concept_matrix, dtype=torch.float32)
    syndrome_herb_prior_tensor = torch.tensor(syndrome_herb_prior, dtype=torch.float32)

    model_config = {
        "model_type": "TCMNet",
        "num_symptoms": dataset.x.shape[1],
        "num_concepts": dataset.concept_targets.shape[1],
        "num_syndromes": len(dataset.raw_label_order),
        "num_herbs": dataset.herb_targets.shape[1],
        "shared_hidden": 512,
        "syndrome_hidden": 256,
        "herb_hidden": 256,
        "shared_dropout": 0.3,
        "syndrome_dropout": 0.2,
        "herb_dropout": 0.2,
    }
    model = TCMNet(
        num_symptoms=model_config["num_symptoms"],
        num_concepts=model_config["num_concepts"],
        num_syndromes=model_config["num_syndromes"],
        num_herbs=model_config["num_herbs"],
        shared_hidden=model_config["shared_hidden"],
        syndrome_hidden=model_config["syndrome_hidden"],
        herb_hidden=model_config["herb_hidden"],
        shared_dropout=model_config["shared_dropout"],
        syndrome_dropout=model_config["syndrome_dropout"],
        herb_dropout=model_config["herb_dropout"],
    )

    print(
        "Dataset loaded with "
        f"{model_config['num_syndromes']} syndromes and "
        f"{model_config['num_symptoms']} symptoms, plus "
        f"{model_config['num_herbs']} herb targets."
    )
    history = train_model(
        model=model,
        dataloader=train_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        class_weights=class_weight_tensor,
        lambda_concept=args.lambda_concept,
        lambda_syndrome=args.lambda_syndrome,
        lambda_herb=args.lambda_herb,
        lambda_herb_bpr=args.lambda_herb_bpr,
        lambda_herb_bce=args.lambda_herb_bce,
        lambda_herb_distill=args.lambda_herb_distill,
        herb_pairs_per_sample=args.herb_pairs_per_sample,
        herb_hard_negative_ratio=args.herb_hard_negative_ratio,
        herb_concept_matrix=herb_concept_tensor,
        syndrome_herb_prior=syndrome_herb_prior_tensor,
        recommendation_alpha=DEFAULT_RECOMMENDATION_ALPHA,
    )
    metrics = evaluate_model(model, val_loader)
    metrics.update(
        evaluate_inference_herb_rankers(
            model=model,
            dataloader=val_loader,
            herb_concept_matrix=herb_concept_matrix,
            syndrome_herb_prior=syndrome_herb_prior,
            alpha=DEFAULT_RECOMMENDATION_ALPHA,
        )
    )

    artifact_dir = args.artifact_dir
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_ROOT / artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), artifact_dir / "tcmnet.pt")
    write_json(
        artifact_dir / "model_config.json",
        {
            **model_config,
            "concept_columns": concept_columns,
            "concept_labels": concept_labels,
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "lambda_concept": args.lambda_concept,
                "lambda_syndrome": args.lambda_syndrome,
                "lambda_herb": args.lambda_herb,
                "lambda_herb_bpr": args.lambda_herb_bpr,
                "lambda_herb_bce": args.lambda_herb_bce,
                "lambda_herb_distill": args.lambda_herb_distill,
                "herb_pairs_per_sample": args.herb_pairs_per_sample,
                "herb_hard_negative_ratio": args.herb_hard_negative_ratio,
                "seed": args.seed,
                "train_size": train_size,
                "validation_size": val_size,
            },
            "metrics": metrics,
        },
    )
    write_json(
        artifact_dir / "symptom_columns.json",
        {
            "columns": symptom_columns,
            "symptom_id_to_index": symptom_id_to_index,
        },
    )
    write_json(
        artifact_dir / "syndrome_index_to_id.json",
        {
            "index_to_id": syndrome_index_to_id,
            "raw_label_to_syndrome_id": raw_label_to_syndrome_id,
        },
    )
    write_json(
        artifact_dir / "concept_labels.json",
        {
            "columns": concept_columns,
            "labels": concept_labels,
        },
    )
    write_json(
        artifact_dir / "herb_mapping.json",
        {
            "herb_ids": herb_ids,
            "herb_id_to_index": {
                herb_id: index for index, herb_id in enumerate(herb_ids)
            },
            "recommendation_alpha_default": DEFAULT_RECOMMENDATION_ALPHA,
            "herb_concept_matrix": "herb_concept_matrix.npy",
            "syndrome_herb_prior": "syndrome_herb_prior.npy",
        },
    )
    write_json(artifact_dir / "training_history.json", history)
    np.save(artifact_dir / "herb_concept_matrix.npy", herb_concept_matrix)
    np.save(artifact_dir / "syndrome_herb_prior.npy", syndrome_herb_prior)
    copy_static_metadata_artifacts(artifact_dir)

    print("Artifacts written to:")
    for artifact in sorted(artifact_dir.iterdir()):
        try:
            display_path = artifact.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = artifact
        print(f"  {display_path}")


def main() -> None:
    export_artifacts(parse_args())


if __name__ == "__main__":
    main()
