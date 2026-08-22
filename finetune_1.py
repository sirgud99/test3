from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import shutil
import sys
from itertools import chain
from pathlib import Path
from typing import Any


ARTIFACTS = Path("artifacts")
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
BLOCK_SIZE = 10
WANDB_PROJECT = "qwen2.5-3b-wikitext103"

PROFILES = {
    "smoke": {
        "model_id": "Qwen/Qwen2.5-0.5B",
        "model_dir": "smoke",
        "train_blocks": 256,
        "validation_blocks": 64,
        "test_blocks": 64,
        "source_rows": 2_000,
    },
    "mac": {
        "model_id": "Qwen/Qwen2.5-0.5B",
        "model_dir": "smoke",
        "train_blocks": 10_000,
        "validation_blocks": 1_000,
        "test_blocks": 1_000,
        "source_rows": 20_000,
    },
    "full": {
        "model_id": "Qwen/Qwen2.5-3B",
        "model_dir": "full",
        "train_blocks": None,
        "validation_blocks": None,
        "test_blocks": None,
        "source_rows": None,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse the small command-line interface."""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 on WikiText-103.")
    commands = parser.add_subparsers(dest="command", required=True)

    profile_options = argparse.ArgumentParser(add_help=False)
    profile_options.add_argument("--profile", choices=PROFILES, default="full")

    train_options = argparse.ArgumentParser(add_help=False, parents=[profile_options])
    train_options.add_argument("--epochs", type=float, default=None)
    train_options.add_argument("--max-steps", type=int, default=None)
    train_options.add_argument("--batch-size", type=int, default=None)
    train_options.add_argument("--gradient-accumulation", type=int, default=None)
    train_options.add_argument("--learning-rate", type=float, default=None)
    train_options.add_argument("--wandb-project", default=WANDB_PROJECT)
    train_options.add_argument("--run-name", default=None)

    commands.add_parser("download", parents=[profile_options], help="Download model and data.")
    commands.add_parser("prepare", parents=[profile_options], help="Tokenize and group the data.")
    commands.add_parser("train", parents=[train_options], help="Train and evaluate a model.")
    commands.add_parser("all", parents=[train_options], help="Download, prepare, and train.")

    evaluate_parser = commands.add_parser(
        "evaluate", parents=[profile_options], help="Evaluate a saved checkpoint."
    )
    evaluate_parser.add_argument("--checkpoint", type=Path, default=None)
    commands.add_parser("smoke", help="Run the fixed, quick MPS smoke experiment.")
    return parser.parse_args()


def download(args: argparse.Namespace) -> None:
    """Download the selected Qwen model and the raw WikiText dataset."""
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    profile = PROFILES[args.profile]
    model_dir = ARTIFACTS / "models" / profile["model_dir"]
    raw_dir = ARTIFACTS / "data" / "wikitext-103-raw"

    if model_dir.exists():
        print(f"Using existing model at {model_dir}")
    else:
        snapshot_download(repo_id=profile["model_id"], local_dir=model_dir)

    if raw_dir.exists():
        print(f"Using existing dataset at {raw_dir}")
    else:
        load_dataset(DATASET_ID, DATASET_CONFIG).save_to_disk(raw_dir)


def tokenize_batch(examples: dict[str, list[str]], tokenizer: Any) -> dict[str, Any]:
    """Tokenize a batch of raw WikiText strings with the selected Qwen tokenizer."""
    return tokenizer(examples["text"])


def group_tokens(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
    """Concatenate tokens and split them into non-overlapping causal-LM blocks."""
    joined = {name: list(chain.from_iterable(values)) for name, values in examples.items()}
    usable_length = len(joined["input_ids"]) // BLOCK_SIZE * BLOCK_SIZE
    blocks = {
        name: [values[i : i + BLOCK_SIZE] for i in range(0, usable_length, BLOCK_SIZE)]
        for name, values in joined.items()
    }
    blocks["labels"] = [values.copy() for values in blocks["input_ids"]]
    return blocks


def prepare(args: argparse.Namespace) -> None:
    """Tokenize, group, limit, verify, and save all dataset splits."""
    from datasets import DatasetDict, load_from_disk
    from transformers import AutoTokenizer

    profile = PROFILES[args.profile]
    model_dir = ARTIFACTS / "models" / profile["model_dir"]
    raw_dir = ARTIFACTS / "data" / "wikitext-103-raw"
    output_dir = ARTIFACTS / "data" / f"processed-{args.profile}"

    if not model_dir.exists() or not raw_dir.exists():
        raise FileNotFoundError("Run the download command before prepare.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    raw = load_from_disk(raw_dir)
    prepared = {}

    for split in ("train", "validation", "test"):
        source = raw[split]
        if profile["source_rows"] is not None:
            source = source.select(range(min(profile["source_rows"], len(source))))

        tokenized = source.map(
            lambda batch: tokenize_batch(batch, tokenizer),
            batched=True,
            remove_columns=source.column_names,
            desc=f"Tokenizing {split}",
        )
        grouped = tokenized.map(
            group_tokens,
            batched=True,
            batch_size=1_000,
            remove_columns=tokenized.column_names,
            desc=f"Creating {BLOCK_SIZE}-token {split} blocks",
        )

        limit = profile[f"{split}_blocks"]
        if limit is not None:
            if len(grouped) < limit:
                raise RuntimeError(f"Only created {len(grouped)} {split} blocks; need {limit}.")
            grouped = grouped.select(range(limit))

        sample = grouped[0]
        if len(sample["input_ids"]) != BLOCK_SIZE or sample["labels"] != sample["input_ids"]:
            raise RuntimeError(f"Invalid prepared {split} sample.")
        prepared[split] = grouped

    if output_dir.exists():
        shutil.rmtree(output_dir)
    DatasetDict(prepared).save_to_disk(output_dir)
    print(f"Saved prepared {args.profile} dataset to {output_dir}")


def load_assets(profile_name: str, model_path: Path | None = None) -> tuple[Any, Any, Any]:
    """Load a model, its exact tokenizer, and the prepared dataset."""
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mps_run = profile_name in ("smoke", "mac")
    if mps_run and not torch.backends.mps.is_available():
        raise RuntimeError("The smoke and mac profiles require an available Apple MPS device.")
    if profile_name == "full":
        if not torch.cuda.is_available():
            raise RuntimeError("The full profile requires a CUDA GPU.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The full profile requires a CUDA GPU with BF16 support.")

    source = model_path or ARTIFACTS / "models" / PROFILES[profile_name]["model_dir"]
    data_path = ARTIFACTS / "data" / f"processed-{profile_name}"
    if not Path(source).exists() or not data_path.exists():
        raise FileNotFoundError("Model or prepared data is missing. Run download and prepare first.")

    dtype = torch.float32 if mps_run else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype)
    dataset = load_from_disk(data_path)
    return model, tokenizer, dataset


def compute_metrics(evaluation: Any) -> dict[str, float]:
    """Compute accuracy after applying the model's one-token causal shift."""
    predictions = evaluation.predictions[:, :-1]
    labels = evaluation.label_ids[:, 1:]
    valid = labels != -100
    accuracy = (predictions[valid] == labels[valid]).mean()
    return {"accuracy": float(accuracy)}


def evaluate_model(trainer: Any, dataset: Any, prefix: str) -> dict[str, float]:
    """Evaluate one model and add perplexity derived from its mean loss."""
    metrics = trainer.evaluate(dataset, metric_key_prefix=prefix)
    loss = float(metrics[f"{prefix}_loss"])
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    metrics[f"{prefix}_perplexity"] = perplexity
    trainer.log({f"{prefix}_perplexity": perplexity})
    trainer.log_metrics(prefix, metrics)
    return {name: float(value) for name, value in metrics.items()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the baseline, train, reload the saved model, and evaluate again."""
    import torch
    import wandb
    from transformers import Trainer, TrainingArguments, default_data_collator

    profile_name = args.profile
    smoke_run = profile_name == "smoke"
    mps_run = profile_name in ("smoke", "mac")
    output_root = ARTIFACTS / profile_name
    checkpoint_dir = output_root / "checkpoints"
    final_dir = output_root / "final"
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_MODE"] = "online"
    if not wandb.login():
        raise RuntimeError("W&B login is required before training.")

    run_name = args.run_name or f"{profile_name}-{PROFILES[profile_name]['model_id'].split('/')[-1]}"
    run_config = {
        "profile": profile_name,
        "model": PROFILES[profile_name]["model_id"],
        "dataset": f"{DATASET_ID}/{DATASET_CONFIG}",
        "block_size": BLOCK_SIZE,
        "stride": BLOCK_SIZE,
        "hardware": platform.platform(),
    }

    with wandb.init(
        project=args.wandb_project,
        name=run_name,
        tags=[profile_name, "full-finetune"],
        config=run_config,
    ) as run:
        model, tokenizer, dataset = load_assets(profile_name)

        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            run_name=run_name,
            report_to=["wandb"],
            num_train_epochs=args.epochs or 1.0,
            max_steps=args.max_steps if args.max_steps is not None else (10 if smoke_run else -1),
            per_device_train_batch_size=args.batch_size or (1 if smoke_run else 8 if mps_run else 16),
            per_device_eval_batch_size=4 if smoke_run else 8 if mps_run else 16,
            gradient_accumulation_steps=args.gradient_accumulation or (1 if mps_run else 16),
            learning_rate=args.learning_rate or (5e-5 if mps_run else 2e-5),
            lr_scheduler_type="linear" if mps_run else "cosine",
            warmup_steps=0.0 if smoke_run else 0.03,
            weight_decay=0.0 if smoke_run else 0.01,
            max_grad_norm=1.0,
            optim="adamw_torch" if mps_run else "adamw_torch_fused",
            bf16=not mps_run,
            tf32=not mps_run,
            gradient_checkpointing=not mps_run,
            eval_strategy="no",
            save_strategy="no" if mps_run else "steps",
            save_steps=1_000,
            save_total_limit=2,
            logging_steps=1 if smoke_run else 10,
            dataloader_num_workers=0,
            dataloader_pin_memory=not mps_run,
            seed=42,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=lambda logits, _: (
                logits[0] if isinstance(logits, tuple) else logits
            ).argmax(dim=-1),
        )

        baseline = evaluate_model(trainer, dataset["test"], "baseline")
        trainer.train()
        validation = evaluate_model(trainer, dataset["validation"], "validation")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        del trainer, model
        gc.collect()
        if mps_run:
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()

        saved_model, tokenizer, dataset = load_assets(profile_name, final_dir)
        trainer = Trainer(
            model=saved_model,
            args=training_args,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=lambda logits, _: (
                logits[0] if isinstance(logits, tuple) else logits
            ).argmax(dim=-1),
        )
        fine_tuned = evaluate_model(trainer, dataset["test"], "fine_tuned")

        comparison = {
            "loss_change": fine_tuned["fine_tuned_loss"] - baseline["baseline_loss"],
            "perplexity_change": fine_tuned["fine_tuned_perplexity"] - baseline["baseline_perplexity"],
            "accuracy_change": fine_tuned["fine_tuned_accuracy"] - baseline["baseline_accuracy"],
        }
        trainer.log(comparison)
        run.summary.update(comparison)

        required = [
            baseline["baseline_loss"],
            validation["validation_loss"],
            fine_tuned["fine_tuned_loss"],
            fine_tuned["fine_tuned_accuracy"],
        ]
        if not all(math.isfinite(value) for value in required):
            raise RuntimeError("Training completed but produced non-finite metrics.")

        results = {
            "profile": profile_name,
            "baseline": baseline,
            "validation": validation,
            "fine_tuned": fine_tuned,
            "comparison": comparison,
            "checkpoint": str(final_dir),
        }
        (output_root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))
        return results


def smoke(args: argparse.Namespace) -> None:
    """Run the fixed 0.5B, 10-step MPS sign-of-life experiment."""
    args.profile = "smoke"
    args.epochs = 1.0
    args.max_steps = 10
    args.batch_size = 1
    args.gradient_accumulation = 1
    args.learning_rate = 5e-5
    args.wandb_project = WANDB_PROJECT
    args.run_name = "smoke-mps-qwen2.5-0.5b"
    download(args)
    prepare(args)
    train(args)


def main() -> None:
    """Dispatch the requested pipeline command."""
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11 or newer is required. See README.md for setup.")

    args = parse_args()
    if args.command == "download":
        download(args)
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        train(args)
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "all":
        download(args)
        prepare(args)
        train(args)
    elif args.command == "evaluate":
        from transformers import Trainer, TrainingArguments, default_data_collator

        checkpoint = args.checkpoint or ARTIFACTS / args.profile / "final"
        model, _, dataset = load_assets(args.profile, checkpoint)
        evaluation_args = TrainingArguments(
            output_dir=str(ARTIFACTS / args.profile / "evaluation"),
            report_to=[],
            per_device_eval_batch_size=4 if args.profile == "smoke" else 16,
            bf16=args.profile == "full",
            tf32=args.profile == "full",
        )
        trainer = Trainer(
            model=model,
            args=evaluation_args,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=lambda logits, _: (
                logits[0] if isinstance(logits, tuple) else logits
            ).argmax(dim=-1),
        )
        metrics = evaluate_model(trainer, dataset["test"], "test")
        result_path = ARTIFACTS / args.profile / "evaluation.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
