"""Train TCMNet and export app-ready inference artifacts.

Run from the project root:
    python3 pipeline/training/export_tcmnet_artifacts.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "pipeline" / "artifacts"


class SyntheticTCMDataset(Dataset):
    def __init__(
        self,
        synthetic_x_file: Path,
        synthetic_y_file: Path,
        concept_file: Path,
    ) -> None:
        self.x_df = pd.read_csv(synthetic_x_file)
        self.y_df = pd.read_csv(synthetic_y_file)
        self.concept_df = pd.read_csv(concept_file, index_col=0)

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

        self.x = torch.tensor(self.x_df.to_numpy(dtype=np.float32), dtype=torch.float32)
        self.y_syndrome = torch.tensor(encoded_labels, dtype=torch.long)
        self.concept_targets = torch.tensor(concept_targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.concept_targets[idx], self.y_syndrome[idx]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TCMNet and save inference artifacts."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--lambda-concept", type=float, default=10.0)
    parser.add_argument("--lambda-syndrome", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=229)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args()


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    model: TCMNet,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    class_weights: torch.Tensor,
    lambda_concept: float,
    lambda_syndrome: float,
) -> dict[str, list[float]]:
    criterion_concept = nn.MSELoss()
    criterion_syndrome = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {"total": [], "concept": [], "syndrome": []}

    model.train()
    for epoch in range(epochs):
        epoch_total = 0.0
        epoch_concept = 0.0
        epoch_syndrome = 0.0

        for symptoms, true_concepts, true_syndromes in dataloader:
            optimizer.zero_grad()
            pred_concepts, pred_syndromes = model(symptoms.float())

            raw_loss_concept = criterion_concept(pred_concepts, true_concepts.float())
            raw_loss_syndrome = criterion_syndrome(
                pred_syndromes, true_syndromes.long()
            )
            loss = (
                raw_loss_concept * lambda_concept
                + raw_loss_syndrome * lambda_syndrome
            )
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_concept += raw_loss_concept.item()
            epoch_syndrome += raw_loss_syndrome.item()

        history["total"].append(epoch_total / len(dataloader))
        history["concept"].append(epoch_concept / len(dataloader))
        history["syndrome"].append(epoch_syndrome / len(dataloader))
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

    with torch.no_grad():
        for symptoms, true_concepts, true_syndromes in dataloader:
            pred_concepts, pred_syndromes = model(symptoms.float())
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

    return {
        "accuracy": correct / total,
        "top5_accuracy": top5_correct / total,
        "macro_f1": macro_f1(y_true, y_pred),
        "concept_mse": concept_error / len(dataloader),
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


def export_artifacts(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    synthetic_x_file = project_path("pipeline/data/patient/Synthetic_Patient_Symptoms.csv")
    synthetic_y_file = project_path("pipeline/data/patient/Synthetic_Patient_Labels.csv")
    concept_file = project_path("pipeline/data/processed/Syndrome_Concept_Targets.csv")
    syndrome_herb_file = project_path("pipeline/data/processed/Syndrome_Herb_Targets.csv")

    dataset = SyntheticTCMDataset(synthetic_x_file, synthetic_y_file, concept_file)
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

    model_config = {
        "model_type": "TCMNet",
        "num_symptoms": dataset.x.shape[1],
        "num_concepts": dataset.concept_targets.shape[1],
        "num_syndromes": len(dataset.raw_label_order),
        "shared_hidden": 512,
        "syndrome_hidden": 256,
        "shared_dropout": 0.3,
        "syndrome_dropout": 0.2,
    }
    model = TCMNet(
        num_symptoms=model_config["num_symptoms"],
        num_concepts=model_config["num_concepts"],
        num_syndromes=model_config["num_syndromes"],
        shared_hidden=model_config["shared_hidden"],
        syndrome_hidden=model_config["syndrome_hidden"],
        shared_dropout=model_config["shared_dropout"],
        syndrome_dropout=model_config["syndrome_dropout"],
    )

    print(
        "Dataset loaded with "
        f"{model_config['num_syndromes']} syndromes and "
        f"{model_config['num_symptoms']} symptoms."
    )
    history = train_model(
        model=model,
        dataloader=train_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        class_weights=class_weight_tensor,
        lambda_concept=args.lambda_concept,
        lambda_syndrome=args.lambda_syndrome,
    )
    metrics = evaluate_model(model, val_loader)

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
            "recommendation_alpha_default": 0.7,
            "herb_concept_matrix": "herb_concept_matrix.npy",
            "syndrome_herb_prior": "syndrome_herb_prior.npy",
        },
    )
    write_json(artifact_dir / "training_history.json", history)
    np.save(artifact_dir / "herb_concept_matrix.npy", herb_concept_matrix)
    np.save(artifact_dir / "syndrome_herb_prior.npy", syndrome_herb_prior)

    print("Artifacts written to:")
    for artifact in sorted(artifact_dir.iterdir()):
        print(f"  {artifact.relative_to(PROJECT_ROOT)}")


def main() -> None:
    export_artifacts(parse_args())


if __name__ == "__main__":
    main()
