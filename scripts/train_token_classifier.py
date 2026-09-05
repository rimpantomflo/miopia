"""Ajuste real de NER Hugging Face sobre JSONL local con tokens y BIO."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from clinical_nlp_course.transformers import (
    align_word_labels,
    read_token_classification_jsonl,
)


def train(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        from datasets import Dataset, DatasetDict
        from seqeval.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError(
            "instala dependencias con: uv sync --extra transformers"
        ) from error

    rows = read_token_classification_jsonl(args.data)
    labels = sorted({tag for row in rows for tag in row["ner_tags"]})
    if "O" in labels:
        labels.remove("O")
        labels.insert(0, "O")
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    grouped = {
        split: Dataset.from_list([row for row in rows if row["split"] == split])
        for split in ("train", "development", "test")
    }
    if any(len(dataset) == 0 for dataset in grouped.values()):
        raise ValueError("se necesitan filas train, development y test")
    dataset = DatasetDict(grouped)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=args.max_length,
        )
        encoded["labels"] = align_word_labels(
            encoded.word_ids(),
            [label_to_id[label] for label in batch["ner_tags"]],
        )
        return encoded

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    def metrics(evaluation_prediction):
        logits, gold = evaluation_prediction
        predicted = np.argmax(logits, axis=-1)
        true_predictions = []
        true_labels = []
        for prediction_row, gold_row in zip(predicted, gold, strict=True):
            mask = gold_row != -100
            true_predictions.append(
                [id_to_label[int(value)] for value in prediction_row[mask]]
            )
            true_labels.append([id_to_label[int(value)] for value in gold_row[mask]])
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "output_dir": str(output),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "seed": args.seed,
        "data_seed": args.seed,
        "report_to": "none",
    }
    evaluation_name = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    kwargs[evaluation_name] = "epoch"
    set_seed(args.seed)
    trainer_kwargs = {
        "model": model,
        "args": TrainingArguments(**kwargs),
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["development"],
        "data_collator": DataCollatorForTokenClassification(tokenizer),
        "compute_metrics": metrics,
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(
        **trainer_kwargs,
    )
    trainer.train()
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    trainer.save_model(str(output / "best_model"))
    tokenizer.save_pretrained(str(output / "best_model"))
    (output / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/transformer_ner")
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
