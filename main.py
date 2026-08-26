from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import shutil
import sys
import zipfile
from contextlib import nullcontext
from itertools import chain
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve


ARTIFACTS = Path("artifacts")
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
BLOCK_SIZE = 10
WANDB_PROJECT = "qwen2.5-3b-wikitext103"
NORARE_BASE_URL = "https://raw.githubusercontent.com/concepticon/norare-cldf/main/cldf"
NORARE_VARIABLES = {
    "frequency": "Brysbaert-2009-Frequency-ENGLISH_FREQUENCY",
    "concreteness": "Brysbaert-2014-Concreteness-ENGLISH_CONCRETENESS_MEAN",
    "iconicity": "Winter-2024-Iconicity-ENGLISH_ICONICITY_MEAN",
    "age_of_acquisition": "Kuperman-2012-AoA-ENGLISH_AOA_MEAN",
    "valence": "Warriner-2013-AffectiveRatings-ENGLISH_VALENCE_MEAN",
    "arousal": "Warriner-2013-AffectiveRatings-ENGLISH_AROUSAL_MEAN",
    "dominance": "Warriner-2013-AffectiveRatings-ENGLISH_DOMINANCE_MEAN",
}
LAYERS = {"last": -1, "penultimate": -2, "antepenultimate": -3}
TOP_EIGENMODES = 20

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
    """Parse the command-line interface."""
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

    analysis_options = argparse.ArgumentParser(add_help=False, parents=[profile_options])
    analysis_options.add_argument("--checkpoint", type=Path, default=None)
    analysis_options.add_argument(
        "--analysis-samples",
        type=int,
        default=None,
        help="Limit transitions for a deliberate test run (default: use all).",
    )
    analysis_options.add_argument("--analysis-batch-size", type=int, default=8)
    analysis_options.add_argument("--pair-stride", type=int, default=1)
    analysis_options.add_argument(
        "--analysis-split",
        choices=("train", "test"),
        default="test",
        help="Split used by pairs/annotate; analyze automatically runs train and test.",
    )
    analysis_options.add_argument(
        "--koopman-rank",
        type=int,
        default=None,
        help="Use a reduced POD rank for testing (default: full hidden dimension).",
    )
    analysis_options.add_argument("--ridge-alpha", type=float, default=1e-3)
    analysis_options.add_argument("--top-utterances", type=int, default=10)

    commands.add_parser("download", parents=[profile_options], help="Download model and data.")
    commands.add_parser("prepare", parents=[profile_options], help="Tokenize and group the data.")
    commands.add_parser("train", parents=[train_options], help="Train and evaluate a model.")
    commands.add_parser("all", parents=[train_options], help="Download, prepare, and train.")

    evaluate_parser = commands.add_parser(
        "evaluate", parents=[profile_options], help="Evaluate a saved checkpoint."
    )
    evaluate_parser.add_argument("--checkpoint", type=Path, default=None)
    commands.add_parser("smoke", help="Run the fixed, quick MPS smoke experiment.")
    commands.add_parser(
        "pairs", parents=[analysis_options], help="Build true/predicted utterance pairs."
    )
    commands.add_parser(
        "annotate", parents=[analysis_options], help="Annotate utterances with lexical features."
    )
    commands.add_parser(
        "koopman", parents=[analysis_options], help="Fit and analyze six Koopman operators."
    )
    commands.add_parser(
        "analyze", parents=[analysis_options], help="Run pairs, annotations, and Koopman analysis."
    )
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
    if trainer.is_world_process_zero():
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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    primary = rank == 0
    if profile_name == "full" and world_size != 2:
        raise RuntimeError(
            "Full training requires exactly two processes. Launch it with "
            "torchrun --standalone --nproc_per_node=2 main.py train --profile full."
        )

    per_device_batch = args.batch_size if args.batch_size is not None else (
        1 if smoke_run else 8 if mps_run else 32
    )
    accumulation = args.gradient_accumulation if args.gradient_accumulation is not None else (
        1 if mps_run else 16
    )
    output_root = ARTIFACTS / profile_name
    checkpoint_dir = output_root / "checkpoints"
    final_dir = output_root / "final"
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_MODE"] = "online"
    if primary and not wandb.login():
        raise RuntimeError("W&B login is required before training.")

    run_name = args.run_name or f"{profile_name}-{PROFILES[profile_name]['model_id'].split('/')[-1]}"
    run_config = {
        "profile": profile_name,
        "model": PROFILES[profile_name]["model_id"],
        "dataset": f"{DATASET_ID}/{DATASET_CONFIG}",
        "block_size": BLOCK_SIZE,
        "stride": BLOCK_SIZE,
        "hardware": platform.platform(),
        "gpus": world_size,
        "per_device_batch_size": per_device_batch,
        "gradient_accumulation": accumulation,
        "effective_batch_size": per_device_batch * accumulation * world_size,
    }

    run_context = (
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=[profile_name, "full-finetune", f"{world_size}-gpu"],
            config=run_config,
        )
        if primary
        else nullcontext(None)
    )
    with run_context as run:
        model, tokenizer, dataset = load_assets(profile_name)

        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            run_name=run_name,
            report_to=["wandb"] if primary else [],
            num_train_epochs=args.epochs or 1.0,
            max_steps=args.max_steps if args.max_steps is not None else (10 if smoke_run else -1),
            per_device_train_batch_size=per_device_batch,
            per_device_eval_batch_size=4 if smoke_run else 8 if mps_run else 16,
            gradient_accumulation_steps=accumulation,
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
            ddp_find_unused_parameters=False if not mps_run else None,
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
        if primary:
            tokenizer.save_pretrained(final_dir)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

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
        if trainer.is_world_process_zero():
            trainer.log(comparison)
        if run is not None:
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
        if primary:
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


def download_norare() -> Path:
    """Download the NoRaRe CLDF tables used for English lexical annotations."""
    output_dir = ARTIFACTS / "norare"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("Wordlist-metadata.json", "variables.csv", "glosses.csv"):
        destination = output_dir / name
        if destination.exists():
            continue
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        print(f"Downloading NoRaRe {name}")
        urlretrieve(f"{NORARE_BASE_URL}/{name}", temporary)
        temporary.replace(destination)
    value_path = output_dir / "norare.csv"
    if not value_path.exists():
        archive = output_dir / "norare.csv.zip"
        if not archive.exists():
            print("Downloading NoRaRe norare.csv.zip")
            urlretrieve(f"{NORARE_BASE_URL}/norare.csv.zip", archive)
        temporary = value_path.with_suffix(".csv.tmp")
        with zipfile.ZipFile(archive) as zipped, zipped.open("norare.csv") as source:
            with temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        temporary.replace(value_path)
    return output_dir


def construct_pairs(args: argparse.Namespace) -> None:
    """Create one-token-shifted ground-truth and predicted utterance pairs."""
    import numpy as np
    import pandas as pd
    import torch
    from tqdm.auto import tqdm

    if (
        (args.analysis_samples is not None and args.analysis_samples < 1)
        or args.analysis_batch_size < 1
        or args.pair_stride < 1
    ):
        raise ValueError("Any sample limit, batch size, and pair stride must be positive.")

    checkpoint = args.checkpoint or ARTIFACTS / args.profile / "final"
    model, tokenizer, dataset = load_assets(args.profile, checkpoint)
    device = torch.device("mps" if args.profile in ("smoke", "mac") else "cuda")
    model.to(device).eval()
    split = args.analysis_split

    token_stream = np.asarray(
        list(chain.from_iterable(dataset[split]["input_ids"])), dtype=np.int64
    )
    possible = (len(token_stream) - BLOCK_SIZE - 1) // args.pair_stride + 1
    sample_count = (
        possible if args.analysis_samples is None else min(args.analysis_samples, possible)
    )
    if sample_count < 1:
        raise RuntimeError(f"The {split} split is too short to form a shifted utterance pair.")
    if args.analysis_samples is not None and sample_count < args.analysis_samples:
        print(f"Using all {sample_count} available pairs instead of {args.analysis_samples}.")

    starts = np.arange(sample_count) * args.pair_stride
    rows: list[dict[str, Any]] = []
    correct_predictions = 0
    representations: dict[str, list[Any]] = {
        f"{kind}_{layer}": []
        for kind in ("source", "ground_truth", "predicted")
        for layer in LAYERS
    }

    batches = range(0, sample_count, args.analysis_batch_size)
    for batch_start in tqdm(batches, desc="Constructing utterance pairs", unit="batch"):
        batch_starts = starts[batch_start : batch_start + args.analysis_batch_size]
        current = np.stack([token_stream[i : i + BLOCK_SIZE] for i in batch_starts])
        true_ids = np.asarray([token_stream[i + BLOCK_SIZE] for i in batch_starts])
        current_tensor = torch.as_tensor(current, device=device)
        true_tensor = torch.as_tensor(true_ids, device=device)

        with torch.inference_mode():
            current_output = model(
                current_tensor, output_hidden_states=True, use_cache=False
            )
            log_probs = torch.log_softmax(current_output.logits[:, -1].float(), dim=-1)
            predicted_tensor = log_probs.argmax(dim=-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            surprisal = -log_probs.gather(1, true_tensor[:, None]).squeeze(1)
            ground_truth_tensor = torch.cat(
                (current_tensor[:, 1:], true_tensor[:, None]), dim=1
            )
            predicted_window_tensor = torch.cat(
                (current_tensor[:, 1:], predicted_tensor[:, None]), dim=1
            )
            ground_truth_output = model(
                ground_truth_tensor, output_hidden_states=True, use_cache=False
            )
            predicted_output = model(
                predicted_window_tensor, output_hidden_states=True, use_cache=False
            )

        predicted_ids = predicted_tensor.cpu().numpy()
        correct_predictions += int((predicted_tensor == true_tensor).sum().item())
        ground_truth = ground_truth_tensor.cpu().numpy()
        predicted_windows = predicted_window_tensor.cpu().numpy()
        for layer_name, layer_index in LAYERS.items():
            for kind, output in (
                ("source", current_output),
                ("ground_truth", ground_truth_output),
                ("predicted", predicted_output),
            ):
                hidden = output.hidden_states[layer_index][:, -1].float().cpu().numpy()
                representations[f"{kind}_{layer_name}"].append(hidden)

        entropy_values = entropy.cpu().numpy()
        surprisal_values = surprisal.cpu().numpy()
        for offset, pair_id in enumerate(range(batch_start, batch_start + len(batch_starts))):
            source_text = tokenizer.decode(current[offset], skip_special_tokens=True)
            common = {
                "pair_id": pair_id,
                "source_utterance": source_text,
                "true_next_token_id": int(true_ids[offset]),
                "true_next_token": tokenizer.decode([int(true_ids[offset])]),
                "argmax_next_token_id": int(predicted_ids[offset]),
                "argmax_next_token": tokenizer.decode([int(predicted_ids[offset])]),
                "next_token_entropy": float(entropy_values[offset]),
                "true_next_token_surprisal": float(surprisal_values[offset]),
                "prediction_correct": bool(true_ids[offset] == predicted_ids[offset]),
            }
            rows.append(
                common
                | {
                    "target_type": "ground_truth",
                    "target_utterance": tokenizer.decode(
                        ground_truth[offset], skip_special_tokens=True
                    ),
                }
            )
            rows.append(
                common
                | {
                    "target_type": "predicted",
                    "target_utterance": tokenizer.decode(
                        predicted_windows[offset], skip_special_tokens=True
                    ),
                }
            )

        del current_output, ground_truth_output, predicted_output

    output_dir = ARTIFACTS / args.profile / "analysis" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "utterance_pairs.csv", index=False)
    arrays = {name: np.concatenate(parts) for name, parts in representations.items()}
    np.savez_compressed(output_dir / "representations.npz", **arrays)
    accuracy = correct_predictions / sample_count
    accuracy_report = {
        "checkpoint": str(checkpoint),
        "dataset_split": split,
        "evaluated_next_tokens": sample_count,
        "correct_predictions": correct_predictions,
        "top_1_accuracy": accuracy,
        "top_1_accuracy_percent": 100.0 * accuracy,
        "window_size": BLOCK_SIZE,
        "pair_stride": args.pair_stride,
    }
    (output_dir / "model_accuracy.json").write_text(
        json.dumps(accuracy_report, indent=2) + "\n"
    )
    metadata = {
        "checkpoint": str(checkpoint),
        "dataset_split": split,
        "samples": sample_count,
        "window_size": BLOCK_SIZE,
        "pair_shift": 1,
        "pair_stride": args.pair_stride,
        "layers": LAYERS,
        "representation": "final-token hidden state",
        "top_1_accuracy": accuracy,
    }
    (output_dir / "pairs_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Analysis top-1 accuracy on {sample_count} {split} transitions: "
        f"{accuracy:.2%} ({correct_predictions}/{sample_count})"
    )
    print(f"Saved {sample_count} true/predicted pairs to {output_dir}")

    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()


def load_norare() -> Any:
    """Load selected English NoRaRe variables into a word-indexed table."""
    import pandas as pd

    norare_dir = download_norare()
    variable_ids = set(NORARE_VARIABLES.values())
    value_parts = []
    for chunk in pd.read_csv(
        norare_dir / "norare.csv",
        usecols=["Unit_ID", "Variable_ID", "Value"],
        chunksize=200_000,
        low_memory=False,
    ):
        selected = chunk[chunk["Variable_ID"].isin(variable_ids)].copy()
        selected["Value"] = pd.to_numeric(selected["Value"], errors="coerce")
        value_parts.append(selected.dropna(subset=["Value"]))
    values = pd.concat(value_parts, ignore_index=True)
    units = set(values["Unit_ID"])
    glosses = pd.read_csv(
        norare_dir / "glosses.csv",
        usecols=["ID", "Language_ID", "Form"],
        low_memory=False,
    )
    glosses = glosses[(glosses["Language_ID"] == "eng") & glosses["ID"].isin(units)]
    wide = values.pivot_table(
        index="Unit_ID", columns="Variable_ID", values="Value", aggfunc="mean"
    )
    lexicon = glosses.join(wide, on="ID")
    lexicon["word"] = lexicon["Form"].astype(str).str.lower().str.strip()
    lexicon = lexicon.rename(columns={value: key for key, value in NORARE_VARIABLES.items()})
    lexicon = lexicon.groupby("word")[list(NORARE_VARIABLES)].mean()
    lexicon.to_csv(norare_dir / "english_lexicon.csv")
    return lexicon


def annotate(args: argparse.Namespace) -> None:
    """Annotate each target utterance with contextual and NoRaRe features."""
    import numpy as np
    import pandas as pd
    import spacy
    from tqdm.auto import tqdm

    output_dir = ARTIFACTS / args.profile / "analysis" / args.analysis_split
    pair_path = output_dir / "utterance_pairs.csv"
    if not pair_path.exists():
        raise FileNotFoundError("Run the pairs command before annotate.")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as error:
        raise RuntimeError(
            "spaCy model en_core_web_sm is missing; install requirements.txt again."
        ) from error

    pairs = pd.read_csv(pair_path)
    lexicon = load_norare()
    content_pos = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
    annotations = []
    documents = nlp.pipe(pairs["target_utterance"].fillna(""), batch_size=128)

    for document in tqdm(documents, total=len(pairs), desc="Annotating utterances"):
        tokens = [token for token in document if any(char.isalnum() for char in token.text)]
        last = tokens[-1] if tokens else None

        def key(token: Any) -> str:
            lemma = token.lemma_ if token.lemma_ and token.lemma_ != "-PRON-" else token.text
            return "".join(
                char for char in lemma.lower().strip() if char.isalnum() or char in "-'"
            )

        word = key(last) if last is not None else ""
        last_values = lexicon.loc[word] if word in lexicon.index else None
        row: dict[str, Any] = {
            "last_word": last.text if last is not None else "",
            "last_lemma": word,
            "last_word_length": len(word),
            "last_is_numeric": bool(last.like_num) if last is not None else False,
            "last_is_alphabetic": word.isalpha(),
            "last_pos": last.pos_ if last is not None else "MISSING",
            "last_lexical_class": (
                "content" if last is not None and last.pos_ in content_pos else "function"
            ),
            "last_is_proper_noun": bool(last and last.pos_ == "PROPN"),
            "last_is_common_noun": bool(last and last.pos_ == "NOUN"),
            "last_noun_type": (
                "proper"
                if last is not None and last.pos_ == "PROPN"
                else "common"
                if last is not None and last.pos_ == "NOUN"
                else "not_noun"
            ),
            "max_dependency_length": max(
                (abs(token.i - token.head.i) for token in document), default=0
            ),
        }
        matched_last = 0
        token_keys = [key(token) for token in tokens]
        for feature in NORARE_VARIABLES:
            value = float(last_values[feature]) if last_values is not None else np.nan
            row[f"last_{feature}"] = value
            matched_last += int(np.isfinite(value))
            sentence_values = [
                float(lexicon.at[token_key, feature])
                for token_key in token_keys
                if token_key in lexicon.index
                and np.isfinite(float(lexicon.at[token_key, feature]))
            ]
            row[f"utterance_mean_{feature}"] = (
                float(np.mean(sentence_values)) if sentence_values else np.nan
            )
            row[f"utterance_{feature}_coverage"] = (
                len(sentence_values) / len(token_keys) if token_keys else 0.0
            )
        row["last_norare_coverage"] = matched_last / len(NORARE_VARIABLES)
        annotations.append(row)

    features = pd.concat((pairs, pd.DataFrame(annotations)), axis=1)
    features.to_csv(output_dir / "utterance_features.csv", index=False)
    coverage = {
        feature: float(features[f"last_{feature}"].notna().mean())
        for feature in NORARE_VARIABLES
    }
    (output_dir / "feature_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    print(f"Saved {len(features)} annotated utterances to {output_dir}")


def feature_matrix(frame: Any, statistics: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """Create a regression matrix, fitting or applying normalization statistics."""
    import numpy as np
    import pandas as pd

    numeric = [
        "next_token_entropy",
        "true_next_token_surprisal",
        "prediction_correct",
        "last_word_length",
        "last_is_numeric",
        "last_is_alphabetic",
        "last_is_proper_noun",
        "last_is_common_noun",
        "max_dependency_length",
        "last_norare_coverage",
    ]
    for feature in NORARE_VARIABLES:
        numeric.extend(
            (
                f"last_{feature}",
                f"utterance_mean_{feature}",
                f"utterance_{feature}_coverage",
            )
        )
    categories = ["last_pos", "last_lexical_class", "last_noun_type"]
    numeric_values = frame[numeric].apply(pd.to_numeric, errors="coerce")
    missing_values = numeric_values.isna().astype(float).rename(
        columns={name: f"{name}_missing" for name in numeric}
    )
    if statistics is None:
        medians = numeric_values.median().fillna(0.0).to_numpy(dtype=np.float64)
    else:
        medians = statistics["numeric_medians"]
    numeric_values = numeric_values.fillna(dict(zip(numeric, medians)))
    values = pd.concat(
        (
            numeric_values,
            missing_values,
            pd.get_dummies(frame[categories], prefix=categories, dummy_na=True),
        ),
        axis=1,
    ).astype(float)
    if statistics is not None:
        values = values.reindex(columns=statistics["feature_names"], fill_value=0.0)
    matrix = values.to_numpy(dtype=np.float64)
    if statistics is None:
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales[scales < 1e-12] = 1.0
        statistics = {
            "numeric_names": numeric,
            "numeric_medians": medians,
            "feature_names": list(values.columns),
            "means": means,
            "scales": scales,
        }
    return (matrix - statistics["means"]) / statistics["scales"], statistics


def load_unembedding(checkpoint: Path) -> Any:
    """Load only the saved output-projection weight, including tied embeddings."""
    from safetensors import safe_open

    index_path = checkpoint / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else {}
    for key in ("lm_head.weight", "model.embed_tokens.weight"):
        if weight_map:
            filename = weight_map.get(key)
        else:
            filename = "model.safetensors"
        if filename and (checkpoint / filename).exists():
            with safe_open(checkpoint / filename, framework="pt", device="cpu") as tensors:
                if key in tensors.keys():
                    return tensors.get_tensor(key)
    raise FileNotFoundError(f"Could not find Qwen's unembedding weight in {checkpoint}.")


def largest_singular_value(matrix: Any, device: Any, chunk_size: int = 2_048) -> float:
    """Compute a tall matrix's largest singular value through its Gram matrix."""
    import torch

    hidden_size = matrix.shape[1]
    gram = torch.zeros((hidden_size, hidden_size), device=device, dtype=torch.float32)
    for start in range(0, matrix.shape[0], chunk_size):
        rows = matrix[start : start + chunk_size].to(device=device, dtype=torch.float32)
        gram.addmm_(rows.T, rows)
    eigenvalues = torch.linalg.eigvalsh(gram.cpu())
    return float(eigenvalues[-1].clamp_min(0.0).sqrt())


def softmax_bound_metrics(
    left_hidden: Any,
    right_hidden: Any,
    deltas: Any,
    labels: Any,
    unembedding: Any,
    device: Any,
    batch_size: int,
) -> dict[str, Any]:
    """Measure cross-entropy, KL divergence, and proposed/corrected bounds."""
    import numpy as np
    import torch
    import torch.nn.functional as functional

    collected = {
        name: []
        for name in (
            "cross_entropy",
            "entropy",
            "kl_divergence",
            "same_label_ce_difference",
            "probability_l2_distance",
            "hidden_distance",
            "logit_distance",
            "delta",
            "proposed_bound",
            "probability_logit_bound",
            "same_label_ce_logit_bound",
            "kl_logit_bound",
        )
    }
    weight = unembedding.to(device)
    for start in range(0, len(left_hidden), batch_size):
        stop = start + batch_size
        left = torch.as_tensor(left_hidden[start:stop], device=device, dtype=weight.dtype)
        right = torch.as_tensor(right_hidden[start:stop], device=device, dtype=weight.dtype)
        label = torch.tensor(labels[start:stop].copy(), device=device, dtype=torch.long)
        delta = torch.as_tensor(
            deltas[start:stop], device=device, dtype=torch.float32
        )
        with torch.inference_mode():
            left_logits = functional.linear(left, weight).float()
            right_logits = functional.linear(right, weight).float()
            left_log_probabilities = torch.log_softmax(left_logits, dim=-1)
            right_log_probabilities = torch.log_softmax(right_logits, dim=-1)
            probabilities = left_log_probabilities.exp()
            entropy = -(probabilities * left_log_probabilities).sum(dim=-1)
            cross_entropy = -(probabilities * right_log_probabilities).sum(dim=-1)
            kl_divergence = (
                probabilities * (left_log_probabilities - right_log_probabilities)
            ).sum(dim=-1).clamp_min(0.0)
            left_label_loss = -left_log_probabilities.gather(1, label[:, None]).squeeze(1)
            right_label_loss = -right_log_probabilities.gather(1, label[:, None]).squeeze(1)
            same_label_ce_difference = (left_label_loss - right_label_loss).abs()
            right_probabilities = right_log_probabilities.exp()
            probability_l2_distance = torch.linalg.vector_norm(
                probabilities - right_probabilities, dim=-1
            )
            hidden_distance = torch.linalg.vector_norm(left.float() - right.float(), dim=-1)
            logit_distance = torch.linalg.vector_norm(left_logits - right_logits, dim=-1)
        values = {
            "cross_entropy": cross_entropy,
            "entropy": entropy,
            "kl_divergence": kl_divergence,
            "same_label_ce_difference": same_label_ce_difference,
            "probability_l2_distance": probability_l2_distance,
            "hidden_distance": hidden_distance,
            "logit_distance": logit_distance,
            "delta": delta,
            "proposed_bound": math.sqrt(2.0) * delta,
            "probability_logit_bound": 0.5 * logit_distance,
            "same_label_ce_logit_bound": math.sqrt(2.0) * logit_distance,
            "kl_logit_bound": math.sqrt(2.0) * logit_distance,
        }
        for name, value in values.items():
            collected[name].append(value.float().cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in collected.items()}


def fit_koopman(source: Any, target: Any, rank: int | None, ridge: float) -> dict[str, Any]:
    """Fit a full-space or explicitly rank-reduced Koopman matrix."""
    import numpy as np

    center = source.mean(axis=0)
    source_centered = source - center
    target_centered = target - center
    if rank is None:
        fitted_rank = source.shape[1]
        basis = np.eye(fitted_rank)
    else:
        _, singular_values, right_vectors = np.linalg.svd(source_centered, full_matrices=False)
        usable = int(np.sum(singular_values > singular_values[0] * 1e-8))
        fitted_rank = min(rank, usable, source.shape[0] - 1, source.shape[1])
        if fitted_rank < 1:
            raise RuntimeError("Hidden representations have no usable variation.")
        basis = right_vectors[:fitted_rank].T
    source_coordinates = source_centered @ basis
    target_coordinates = target_centered @ basis
    scale = source_coordinates.std(axis=0)
    scale[scale < 1e-8] = 1.0
    source_coordinates /= scale
    target_coordinates /= scale
    gram = source_coordinates.T @ source_coordinates + ridge * np.eye(fitted_rank)
    operator = np.linalg.solve(gram, source_coordinates.T @ target_coordinates)
    prediction = source_coordinates @ operator
    residual = np.sum((target_coordinates - prediction) ** 2)
    total = np.sum((target_coordinates - target_coordinates.mean(axis=0)) ** 2)
    eigenvalues, eigenvectors = np.linalg.eig(operator)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    raw_activations = np.abs(source_coordinates @ eigenvectors)
    activation_means = raw_activations.mean(axis=0)
    activation_scales = raw_activations.std(axis=0)
    activation_scales[activation_scales < 1e-12] = 1.0
    zscored_activations = (raw_activations - activation_means) / activation_scales
    operator_singular_values = np.linalg.svd(operator, compute_uv=False)
    participation = operator_singular_values.sum() ** 2 / np.sum(operator_singular_values**2)
    stable_rank = np.sum(operator_singular_values**2) / operator_singular_values[0] ** 2
    return {
        "operator": operator,
        "basis": basis,
        "center": center,
        "scale": scale,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "raw_activations": raw_activations,
        "zscored_activations": zscored_activations,
        "activation_means": activation_means,
        "activation_scales": activation_scales,
        "rank": fitted_rank,
        "r2": float(1.0 - residual / total) if total > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean((target_coordinates - prediction) ** 2))),
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "effective_dimension": float(participation),
        "stable_rank": float(stable_rank),
        "smallest_singular_value": float(operator_singular_values[-1]),
        "numerical_rank": int(np.linalg.matrix_rank(operator)),
    }


def plot_koopman(result: dict[str, Any], coefficients: Any, path: Path, title: str) -> None:
    """Plot the eigenvalue spectrum, mode alignments, and feature coefficients."""
    matplotlib_cache = ARTIFACTS / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib.pyplot as plt
    import numpy as np

    eigenvalues = result["eigenvalues"]
    activations = result["zscored_activations"]
    coefficient_values = coefficients.to_numpy(dtype=float)
    plotted_modes = min(TOP_EIGENMODES, result["rank"])
    visible = np.asarray(
        [
            index
            for index, name in enumerate(coefficients.columns)
            if "coverage" not in name.lower() and "missing" not in name.lower()
        ]
    )
    displayed_activations = activations[:, :plotted_modes]
    displayed_coefficients = coefficient_values[:plotted_modes, visible].T

    figure_height = max(7.0, 0.27 * len(visible))
    figure, axes = plt.subplots(1, 3, figsize=(22, figure_height))
    angle = np.linspace(0, 2 * np.pi, 300)
    axes[0].plot(np.cos(angle), np.sin(angle), color="lightgray", linewidth=1)
    axes[0].scatter(eigenvalues.real, eigenvalues.imag, c=np.arange(len(eigenvalues)))
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].axvline(0, color="gray", linewidth=0.5)
    axes[0].set(xlabel="Real", ylabel="Imaginary", title="Eigenvalue spectrum")
    axes[0].set_aspect("equal", adjustable="box")
    activation_limit = max(float(np.max(np.abs(displayed_activations))), 1e-12)
    activation_image = axes[1].imshow(
        displayed_activations,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-activation_limit,
        vmax=activation_limit,
    )
    axes[1].set(
        xlabel="Eigenmode",
        ylabel="Utterance pair",
        title=f"Z-scored activation: top {plotted_modes} modes",
    )
    axes[1].set_xticks(range(plotted_modes))
    figure.colorbar(activation_image, ax=axes[1], shrink=0.8)
    coefficient_limit = max(float(np.max(np.abs(displayed_coefficients))), 1e-12)
    image = axes[2].imshow(
        displayed_coefficients,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-coefficient_limit,
        vmax=coefficient_limit,
    )
    axes[2].set_xticks(range(plotted_modes))
    axes[2].set_yticks(range(len(visible)))
    axes[2].set_yticklabels([coefficients.columns[index] for index in visible], fontsize=7)
    axes[2].set(
        xlabel="Eigenmode",
        ylabel="Feature",
        title="Z-scored activation regression coefficients",
    )
    figure.colorbar(image, ax=axes[2], shrink=0.8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def koopman(args: argparse.Namespace) -> None:
    """Fit six training-set Koopman matrices and evaluate them on test data."""
    import numpy as np
    import pandas as pd
    import torch

    if (
        (args.koopman_rank is not None and args.koopman_rank < 1)
        or args.ridge_alpha < 0
        or not 10 <= args.top_utterances <= 20
    ):
        raise ValueError(
            "Any Koopman rank must be positive, ridge must be nonnegative, and top "
            "utterances must be between 10 and 20."
        )
    output_dir = ARTIFACTS / args.profile / "analysis"
    split_data = {}
    for split in ("train", "test"):
        split_dir = output_dir / split
        feature_path = split_dir / "utterance_features.csv"
        representation_path = split_dir / "representations.npz"
        if not feature_path.exists() or not representation_path.exists():
            raise FileNotFoundError(f"Run pairs and annotate for the {split} split first.")
        split_data[split] = (
            pd.read_csv(feature_path),
            np.load(representation_path),
        )

    train_features, train_representations = split_data["train"]
    test_features, test_representations = split_data["test"]
    checkpoint = args.checkpoint or ARTIFACTS / args.profile / "final"
    device = torch.device("mps" if args.profile in ("smoke", "mac") else "cuda")
    unembedding = load_unembedding(checkpoint).to(device)
    unembedding_sigma_max = largest_singular_value(unembedding, device)
    summary_rows = []
    mode_rows = []
    coefficient_rows = []
    aligned_rows = []
    theory_rows = []
    surprisal_bound_rows = []

    for layer in LAYERS:
        train_source = train_representations[f"source_{layer}"].astype(np.float64)
        test_source = test_representations[f"source_{layer}"].astype(np.float64)
        for target_type in ("ground_truth", "predicted"):
            train_target = train_representations[f"{target_type}_{layer}"].astype(np.float64)
            test_target = test_representations[f"{target_type}_{layer}"].astype(np.float64)
            train_branch = train_features[
                train_features["target_type"] == target_type
            ].sort_values("pair_id")
            test_branch = test_features[
                test_features["target_type"] == target_type
            ].sort_values("pair_id")
            if len(train_branch) != len(train_source) or len(test_branch) != len(test_source):
                raise RuntimeError("Feature rows and representation rows do not align.")

            train_matrix, feature_statistics = feature_matrix(train_branch)
            test_matrix, _ = feature_matrix(test_branch, feature_statistics)
            feature_names = feature_statistics["feature_names"]
            result = fit_koopman(
                train_source, train_target, args.koopman_rank, args.ridge_alpha
            )
            test_source_coordinates = (
                (test_source - result["center"]) @ result["basis"] / result["scale"]
            )
            test_target_coordinates = (
                (test_target - result["center"]) @ result["basis"] / result["scale"]
            )
            test_prediction = test_source_coordinates @ result["operator"]
            test_residual = np.sum((test_target_coordinates - test_prediction) ** 2)
            test_total = np.sum(
                (test_target_coordinates - test_target_coordinates.mean(axis=0)) ** 2
            )
            test_transition_r2 = (
                float(1.0 - test_residual / test_total) if test_total > 0 else float("nan")
            )
            test_transition_rmse = float(
                np.sqrt(np.mean((test_target_coordinates - test_prediction) ** 2))
            )
            test_raw_activations = np.abs(
                test_source_coordinates @ result["eigenvectors"]
            )
            test_zscored_activations = (
                test_raw_activations - result["activation_means"]
            ) / result["activation_scales"]

            if layer == "last":
                true_token_ids = test_branch["true_next_token_id"].to_numpy()
                predicted_target_hidden = (
                    (test_prediction * result["scale"]) @ result["basis"].T
                    + result["center"]
                )
                direct_deltas = np.linalg.norm(
                    test_target - predicted_target_hidden, axis=1
                )
                direct_metrics = softmax_bound_metrics(
                    test_target,
                    predicted_target_hidden,
                    direct_deltas,
                    true_token_ids,
                    unembedding,
                    device,
                    args.analysis_batch_size,
                )
                if target_type == "ground_truth":
                    coordinate_operator = (
                        result["operator"] / result["scale"][:, None]
                    ) * result["scale"][None, :]
                    hidden_operator = (
                        result["basis"] @ coordinate_operator @ result["basis"].T
                    )
                    hidden_operator_sigma_min = float(
                        np.linalg.svd(hidden_operator, compute_uv=False)[-1]
                    )
                    per_utterance_mse = np.mean(
                        (test_target - predicted_target_hidden) ** 2, axis=1
                    )
                    model_surprisal = test_branch[
                        "true_next_token_surprisal"
                    ].to_numpy(dtype=np.float64)
                    k_sigma_min = hidden_operator_sigma_min
                    multiplier = (
                        math.sqrt(2.0) * unembedding_sigma_max / k_sigma_min
                        if k_sigma_min > 0
                        else float("inf")
                    )
                    requested_bound = multiplier * per_utterance_mse
                    l2_error = np.sqrt(test_target.shape[1] * per_utterance_mse)
                    l2_bound = multiplier * l2_error
                    for index, pair_id in enumerate(test_branch["pair_id"].to_numpy()):
                        surprisal_bound_rows.append(
                            {
                                "pair_id": int(pair_id),
                                "true_next_token_id": int(true_token_ids[index]),
                                "true_next_token": test_branch.iloc[index]["true_next_token"],
                                "surprisal": model_surprisal[index],
                                "k_hidden_mse": per_utterance_mse[index],
                                "k_hidden_l2_error": l2_error[index],
                                "unembedding_sigma_max": unembedding_sigma_max,
                                "k_hidden_space_sigma_min": k_sigma_min,
                                "requested_mse_bound": requested_bound[index],
                                "requested_mse_bound_holds": bool(
                                    model_surprisal[index] <= requested_bound[index]
                                ),
                                "l2_error_bound": l2_bound[index],
                                "l2_error_bound_holds": bool(
                                    model_surprisal[index] <= l2_bound[index]
                                ),
                            }
                        )
                inverse = np.linalg.pinv(result["operator"])
                preimage_coordinates = test_target_coordinates @ inverse
                preimage_hidden = (
                    (preimage_coordinates * result["scale"]) @ result["basis"].T
                    + result["center"]
                )
                coordinate_deltas = np.linalg.norm(
                    test_target_coordinates - test_prediction, axis=1
                )
                inverse_distances = np.linalg.norm(
                    test_source_coordinates - preimage_coordinates, axis=1
                )
                smallest_singular_value = result["smallest_singular_value"]
                inverse_bounds = (
                    coordinate_deltas / smallest_singular_value
                    if smallest_singular_value > 0
                    else np.full_like(coordinate_deltas, np.inf)
                )
                preimage_metrics = softmax_bound_metrics(
                    test_source,
                    preimage_hidden,
                    coordinate_deltas,
                    true_token_ids,
                    unembedding,
                    device,
                    args.analysis_batch_size,
                )
                pair_ids = test_branch["pair_id"].to_numpy()
                for comparison, metrics in (
                    ("direct_target", direct_metrics),
                    ("preimage", preimage_metrics),
                ):
                    for index, pair_id in enumerate(pair_ids):
                        row = {
                            "comparison": comparison,
                            "target_type": target_type,
                            "pair_id": int(pair_id),
                            **{name: value[index] for name, value in metrics.items()},
                        }
                        row["proposed_cross_entropy_holds"] = bool(
                            row["cross_entropy"] <= row["proposed_bound"] + 1e-6
                        )
                        row["proposed_same_label_ce_difference_holds"] = bool(
                            row["same_label_ce_difference"]
                            <= row["proposed_bound"] + 1e-6
                        )
                        row["same_label_ce_logit_bound_holds"] = bool(
                            row["same_label_ce_difference"]
                            <= row["same_label_ce_logit_bound"] + 1e-6
                        )
                        row["proposed_probability_distance_holds"] = bool(
                            row["probability_l2_distance"]
                            <= row["proposed_bound"] + 1e-6
                        )
                        row["softmax_lipschitz_probability_bound_holds"] = bool(
                            row["probability_l2_distance"]
                            <= row["probability_logit_bound"] + 1e-6
                        )
                        row["proposed_kl_holds"] = bool(
                            row["kl_divergence"] <= row["proposed_bound"] + 1e-6
                        )
                        row["corrected_kl_holds"] = bool(
                            row["kl_divergence"] <= row["kl_logit_bound"] + 1e-6
                        )
                        if comparison == "preimage":
                            row["coordinate_delta"] = coordinate_deltas[index]
                            row["preimage_coordinate_distance"] = inverse_distances[index]
                            row["inverse_distance_bound"] = inverse_bounds[index]
                            row["smallest_singular_value"] = smallest_singular_value
                        theory_rows.append(row)

            stem = f"{layer}_{target_type}"
            np.save(output_dir / f"K_{stem}.npy", result["operator"])
            np.savez_compressed(
                output_dir / f"koopman_state_{stem}.npz",
                basis=result["basis"],
                center=result["center"],
                scale=result["scale"],
                eigenvalues=result["eigenvalues"],
                eigenvectors=result["eigenvectors"],
                train_raw_mode_activations=result["raw_activations"].astype(np.float32),
                train_zscored_mode_activations=result["zscored_activations"].astype(np.float32),
                test_raw_mode_activations=test_raw_activations.astype(np.float32),
                test_zscored_mode_activations=test_zscored_activations.astype(np.float32),
                activation_means=result["activation_means"],
                activation_scales=result["activation_scales"],
                feature_names=np.asarray(feature_names),
                numeric_feature_names=np.asarray(feature_statistics["numeric_names"]),
                feature_medians=feature_statistics["numeric_medians"],
                feature_means=feature_statistics["means"],
                feature_scales=feature_statistics["scales"],
            )

            coefficients = []
            gram = train_matrix.T @ train_matrix
            regularized_gram = gram + args.ridge_alpha * np.eye(gram.shape[0])
            for mode in range(result["rank"]):
                raw_activation = result["raw_activations"][:, mode]
                zscored_activation = result["zscored_activations"][:, mode]
                test_raw_activation = test_raw_activations[:, mode]
                test_zscored_activation = test_zscored_activations[:, mode]
                regression = np.linalg.solve(
                    regularized_gram,
                    train_matrix.T @ zscored_activation,
                )
                train_fitted = train_matrix @ regression
                test_fitted = test_matrix @ regression
                train_total = np.sum(zscored_activation**2)
                test_mode_total = np.sum(
                    (test_zscored_activation - test_zscored_activation.mean()) ** 2
                )
                train_mode_r2 = (
                    1.0 - np.sum((zscored_activation - train_fitted) ** 2) / train_total
                    if train_total > 0
                    else np.nan
                )
                test_mode_r2 = (
                    1.0
                    - np.sum((test_zscored_activation - test_fitted) ** 2) / test_mode_total
                    if test_mode_total > 0
                    else np.nan
                )
                test_mode_rmse = float(
                    np.sqrt(np.mean((test_zscored_activation - test_fitted) ** 2))
                )
                coefficients.append(regression)
                eigenvalue = result["eigenvalues"][mode]
                top_features = np.argsort(np.abs(regression))[::-1][:10]
                mode_rows.append(
                    {
                        "layer": layer,
                        "target_type": target_type,
                        "mode": mode,
                        "eigenvalue_real": eigenvalue.real,
                        "eigenvalue_imag": eigenvalue.imag,
                        "eigenvalue_magnitude": abs(eigenvalue),
                        "raw_activation_mean": result["activation_means"][mode],
                        "raw_activation_std": result["activation_scales"][mode],
                        "train_feature_regression_r2": train_mode_r2,
                        "test_feature_regression_r2": test_mode_r2,
                        "test_feature_regression_rmse": test_mode_rmse,
                        "top_features": "; ".join(
                            f"{feature_names[index]}={regression[index]:.5g}"
                            for index in top_features
                        ),
                    }
                )
                for name, value in zip(feature_names, regression):
                    coefficient_rows.append(
                        {
                            "layer": layer,
                            "target_type": target_type,
                            "mode": mode,
                            "feature": name,
                            "coefficient": value,
                        }
                    )
                if mode < TOP_EIGENMODES:
                    for data_split, branch, raw_values, zscored_values in (
                        ("train", train_branch, raw_activation, zscored_activation),
                        ("test", test_branch, test_raw_activation, test_zscored_activation),
                    ):
                        for rank_index, row_index in enumerate(
                            np.argsort(raw_values)[::-1][: args.top_utterances], start=1
                        ):
                            utterance = branch.iloc[row_index]
                            aligned_rows.append(
                                {
                                    "data_split": data_split,
                                    "layer": layer,
                                    "target_type": target_type,
                                    "mode": mode,
                                    "eigenvalue_magnitude": abs(eigenvalue),
                                    "rank": rank_index,
                                    "raw_activation": raw_values[row_index],
                                    "zscored_activation": zscored_values[row_index],
                                    "pair_id": int(utterance["pair_id"]),
                                    "utterance": utterance["target_utterance"],
                                }
                            )

            coefficient_frame = pd.DataFrame(coefficients, columns=feature_names)
            plot_result = result | {"zscored_activations": test_zscored_activations}
            plot_koopman(
                plot_result,
                coefficient_frame,
                output_dir / f"eigenmodes_{stem}.png",
                f"{layer.replace('_', ' ').title()} layer — "
                f"{target_type.replace('_', ' ')} — test activations",
            )
            summary_rows.append(
                {
                    "layer": layer,
                    "target_type": target_type,
                    "koopman_rank": result["rank"],
                    "train_transition_r2": result["r2"],
                    "train_transition_rmse": result["rmse"],
                    "test_transition_r2": test_transition_r2,
                    "test_transition_rmse": test_transition_rmse,
                    "spectral_radius": result["spectral_radius"],
                    "effective_dimension": result["effective_dimension"],
                    "stable_rank": result["stable_rank"],
                    "smallest_singular_value": result["smallest_singular_value"],
                    "numerical_rank": result["numerical_rank"],
                }
            )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_dir / "koopman_summary.csv", index=False)
    pd.DataFrame(mode_rows).to_csv(output_dir / "eigenmode_regressions.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(output_dir / "eigenmode_coefficients.csv", index=False)
    pd.DataFrame(aligned_rows).to_csv(output_dir / "eigenmode_utterances.csv", index=False)
    theory_frame = pd.DataFrame(theory_rows)
    theory_frame.to_csv(output_dir / "theory_bound_samples.csv", index=False)
    theory_summary = (
        theory_frame.groupby(["comparison", "target_type"], as_index=False)
        .agg(
            samples=("pair_id", "size"),
            mean_delta=("delta", "mean"),
            mean_hidden_distance=("hidden_distance", "mean"),
            mean_cross_entropy=("cross_entropy", "mean"),
            mean_same_label_ce_difference=("same_label_ce_difference", "mean"),
            mean_kl_divergence=("kl_divergence", "mean"),
            mean_probability_l2_distance=("probability_l2_distance", "mean"),
            proposed_probability_distance_pass_rate=(
                "proposed_probability_distance_holds",
                "mean",
            ),
            softmax_lipschitz_probability_pass_rate=(
                "softmax_lipschitz_probability_bound_holds",
                "mean",
            ),
            proposed_cross_entropy_pass_rate=("proposed_cross_entropy_holds", "mean"),
            proposed_same_label_ce_difference_pass_rate=(
                "proposed_same_label_ce_difference_holds",
                "mean",
            ),
            same_label_ce_logit_bound_pass_rate=(
                "same_label_ce_logit_bound_holds",
                "mean",
            ),
            proposed_kl_pass_rate=("proposed_kl_holds", "mean"),
            corrected_kl_pass_rate=("corrected_kl_holds", "mean"),
        )
    )
    theory_summary.to_csv(output_dir / "theory_bound_summary.csv", index=False)
    surprisal_bound_frame = pd.DataFrame(surprisal_bound_rows)
    satisfying = int(surprisal_bound_frame["requested_mse_bound_holds"].sum())
    percentage = 100.0 * satisfying / len(surprisal_bound_frame)
    raw_bounds = surprisal_bound_frame["requested_mse_bound"].to_numpy(dtype=float)
    raw_surprisals = surprisal_bound_frame["surprisal"].to_numpy(dtype=float)
    ratios = pd.Series(
        np.divide(
            raw_surprisals,
            raw_bounds,
            out=np.full_like(raw_surprisals, np.inf),
            where=raw_bounds > 0,
        )
    )
    finite_ratios = ratios[np.isfinite(ratios)]
    surprisal_bound_summary = {
        "samples": len(surprisal_bound_frame),
        "satisfying_utterances": satisfying,
        "percentage_satisfying_bound": percentage,
        "mean_surprisal": float(surprisal_bound_frame["surprisal"].mean()),
        "mean_k_hidden_mse": float(surprisal_bound_frame["k_hidden_mse"].mean()),
        "unembedding_sigma_max": unembedding_sigma_max,
        "k_hidden_space_sigma_min": float(
            surprisal_bound_frame["k_hidden_space_sigma_min"].iloc[0]
        ),
        "requested_mse_bound_pass_rate": float(
            surprisal_bound_frame["requested_mse_bound_holds"].mean()
        ),
        "median_surprisal_to_bound_ratio": float(finite_ratios.median()),
        "l2_error_bound_pass_rate": float(
            surprisal_bound_frame["l2_error_bound_holds"].mean()
        ),
    }
    (output_dir / "surprisal_bound_summary.json").write_text(
        json.dumps(surprisal_bound_summary, indent=2) + "\n"
    )
    summary = {
        "operators": len(summary_rows),
        "layers": list(LAYERS),
        "targets": ["ground_truth", "predicted"],
        "operator_fit_split": "train",
        "operator_evaluation_split": "test",
        "lifting_function": "final-token hidden representation produced by each learned layer",
        "fit": (
            "full hidden-space ridge regression Y = X K"
            if args.koopman_rank is None
            else f"POD-reduced rank-{args.koopman_rank} ridge regression Y = X K"
        ),
        "effective_dimension": "singular-value participation ratio of K",
        "mode_regressions": (
            "all eigenmodes; fit on training z-scored activations and features, "
            "then evaluated on test data"
        ),
        "feature_normalization": (
            "fit separately on ground_truth and predicted training utterances; "
            "the corresponding training statistics transform test utterances"
        ),
        "raw_values": "raw features in train/test utterance_features.csv; train/test raw "
        "and z-scored mode activations in each koopman_state_*.npz",
        "heatmaps": (
            f"display the {TOP_EIGENMODES} eigenmodes with largest eigenvalue magnitude; "
            "coefficient heatmaps hide feature names containing coverage or missing"
        ),
        "saved_utterances": (
            f"top {args.top_utterances} train and test utterances for the "
            f"{TOP_EIGENMODES} eigenmodes with largest eigenvalue magnitude"
        ),
        "theory_test": (
            "last-layer pointwise test on held-out utterances; the primary quantity "
            "is the same-fixed-label cross-entropy loss difference from the cited "
            "sqrt(2)-Lipschitz result; probability distance, raw cross-entropy, and "
            "KL are additional diagnostics"
        ),
        "requested_surprisal_test": (
            "per-test-utterance surprisal <= hidden-state MSE * sqrt(2) * "
            "sigma_max(U) / sigma_min(K), plus the norm-consistent version "
            "using sqrt(hidden_dimension * MSE)"
        ),
    }
    (output_dir / "koopman_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = np.where(
        surprisal_bound_frame["requested_mse_bound_holds"], "tab:green", "tab:red"
    )
    positive_bounds = raw_bounds[raw_bounds > 0]
    positive_surprisals = raw_surprisals[raw_surprisals > 0]
    bound_floor = float(positive_bounds.min() / 2) if len(positive_bounds) else 1e-12
    surprisal_floor = (
        float(positive_surprisals.min() / 2) if len(positive_surprisals) else 1e-12
    )
    x_values = np.maximum(raw_bounds, bound_floor)
    y_values = np.maximum(raw_surprisals, surprisal_floor)
    lower = min(float(x_values.min()), float(y_values.min()))
    upper = max(float(x_values.max()), float(y_values.max()))
    axes[0].scatter(x_values, y_values, c=colors, s=5, alpha=0.25, rasterized=True)
    axes[0].plot([lower, upper], [lower, upper], color="black", linestyle="--")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set(
        xlabel="Proposed upper bound",
        ylabel="True-token surprisal",
        title=f"{percentage:.2f}% satisfy surprisal ≤ bound",
    )
    finite_log_ratios = np.log10(finite_ratios.clip(lower=1e-30))
    axes[1].hist(finite_log_ratios, bins=60, color="steelblue")
    axes[1].axvline(0.0, color="black", linestyle="--")
    axes[1].set(
        xlabel="log10(surprisal / bound)",
        ylabel="Utterances",
        title="Distance from the bound (≤ 0 satisfies)",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "surprisal_bound.png", dpi=180)
    plt.close(figure)

    labels = [f"{row['layer']}\n{row['target_type']}" for row in summary_rows]
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(labels, summary_frame["effective_dimension"])
    axis.set(ylabel="Participation-ratio dimension", title="Effective dimension of K")
    axis.tick_params(axis="x", labelrotation=25)
    figure.tight_layout()
    figure.savefig(output_dir / "effective_dimensions.png", dpi=180)
    plt.close(figure)
    print(summary_frame.to_string(index=False))
    print(theory_summary.to_string(index=False))
    print(json.dumps(surprisal_bound_summary, indent=2))


def analyze(args: argparse.Namespace) -> None:
    """Fit the analysis on training utterances and evaluate it on test utterances."""
    accuracy = {}
    for split in ("train", "test"):
        args.analysis_split = split
        construct_pairs(args)
        annotate(args)
        report_path = ARTIFACTS / args.profile / "analysis" / split / "model_accuracy.json"
        accuracy[split] = json.loads(report_path.read_text())
    accuracy_path = ARTIFACTS / args.profile / "analysis" / "model_accuracy.json"
    accuracy_path.write_text(json.dumps(accuracy, indent=2) + "\n")
    koopman(args)


def main() -> None:
    """Dispatch the requested pipeline command."""
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11 or newer is required. See README.md for setup.")

    args = parse_args()
    if args.command == "all" and args.profile == "full":
        raise SystemExit(
            "For two-GPU training, run download and prepare once, then launch train with torchrun."
        )
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
    elif args.command == "pairs":
        construct_pairs(args)
    elif args.command == "annotate":
        annotate(args)
    elif args.command == "koopman":
        koopman(args)
    elif args.command == "analyze":
        analyze(args)
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
