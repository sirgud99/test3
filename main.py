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
        "annotate", parents=[profile_options], help="Annotate utterances with lexical features."
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

    token_stream = np.asarray(
        list(chain.from_iterable(dataset["test"]["input_ids"])), dtype=np.int64
    )
    possible = (len(token_stream) - BLOCK_SIZE - 1) // args.pair_stride + 1
    sample_count = (
        possible if args.analysis_samples is None else min(args.analysis_samples, possible)
    )
    if sample_count < 1:
        raise RuntimeError("The test split is too short to form a shifted utterance pair.")
    if args.analysis_samples is not None and sample_count < args.analysis_samples:
        print(f"Using all {sample_count} available pairs instead of {args.analysis_samples}.")

    starts = np.arange(sample_count) * args.pair_stride
    rows: list[dict[str, Any]] = []
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

    output_dir = ARTIFACTS / args.profile / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "utterance_pairs.csv", index=False)
    arrays = {name: np.concatenate(parts) for name, parts in representations.items()}
    np.savez_compressed(output_dir / "representations.npz", **arrays)
    metadata = {
        "checkpoint": str(checkpoint),
        "samples": sample_count,
        "window_size": BLOCK_SIZE,
        "pair_shift": 1,
        "pair_stride": args.pair_stride,
        "layers": LAYERS,
        "representation": "final-token hidden state",
    }
    (output_dir / "pairs_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
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

    output_dir = ARTIFACTS / args.profile / "analysis"
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


def feature_matrix(frame: Any) -> tuple[Any, list[str]]:
    """Create an imputed, standardized regression matrix from utterance features."""
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
    values = frame[numeric].apply(pd.to_numeric, errors="coerce")
    for name in list(values.columns):
        if values[name].isna().any():
            values[f"{name}_missing"] = values[name].isna().astype(float)
            median = values[name].median()
            values[name] = values[name].fillna(0.0 if pd.isna(median) else median)
    values = pd.concat(
        (values, pd.get_dummies(frame[categories], prefix=categories, dummy_na=True)), axis=1
    ).astype(float)
    matrix = values.to_numpy(dtype=np.float64)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-12] = 1.0
    return (matrix - means) / scales, list(values.columns)


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
    alignments = np.abs(source_coordinates @ eigenvectors)
    operator_singular_values = np.linalg.svd(operator, compute_uv=False)
    participation = operator_singular_values.sum() ** 2 / np.sum(operator_singular_values**2)
    stable_rank = np.sum(operator_singular_values**2) / operator_singular_values[0] ** 2
    return {
        "operator": operator,
        "basis": basis,
        "center": center,
        "scale": scale,
        "eigenvalues": eigenvalues,
        "alignments": alignments,
        "rank": fitted_rank,
        "r2": float(1.0 - residual / total) if total > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean((target_coordinates - prediction) ** 2))),
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "effective_dimension": float(participation),
        "stable_rank": float(stable_rank),
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
    alignments = result["alignments"]
    normalized = alignments / np.maximum(alignments.max(axis=0, keepdims=True), 1e-12)
    coefficient_values = coefficients.to_numpy(dtype=float)
    feature_strength = np.max(np.abs(coefficient_values), axis=0)
    selected = np.argsort(feature_strength)[-20:]

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    angle = np.linspace(0, 2 * np.pi, 300)
    axes[0].plot(np.cos(angle), np.sin(angle), color="lightgray", linewidth=1)
    axes[0].scatter(eigenvalues.real, eigenvalues.imag, c=np.arange(len(eigenvalues)))
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].axvline(0, color="gray", linewidth=0.5)
    axes[0].set(xlabel="Real", ylabel="Imaginary", title="Eigenvalue spectrum")
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].imshow(normalized.T, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[1].set(xlabel="Utterance pair", ylabel="Eigenmode", title="Mode alignment")
    image = axes[2].imshow(
        coefficient_values[:, selected], aspect="auto", interpolation="nearest", cmap="coolwarm"
    )
    axes[2].set_xticks(range(len(selected)))
    axes[2].set_xticklabels(
        [coefficients.columns[index] for index in selected], rotation=90, fontsize=7
    )
    axes[2].set(xlabel="Feature", ylabel="Eigenmode", title="Mode regression coefficients")
    figure.colorbar(image, ax=axes[2], shrink=0.8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def koopman(args: argparse.Namespace) -> None:
    """Fit six Koopman matrices and regress every eigenmode on annotations."""
    import numpy as np
    import pandas as pd

    if (
        (args.koopman_rank is not None and args.koopman_rank < 1)
        or args.ridge_alpha < 0
        or args.top_utterances < 1
    ):
        raise ValueError(
            "Any Koopman rank and top utterances must be positive; ridge must be nonnegative."
        )
    output_dir = ARTIFACTS / args.profile / "analysis"
    feature_path = output_dir / "utterance_features.csv"
    representation_path = output_dir / "representations.npz"
    if not feature_path.exists() or not representation_path.exists():
        raise FileNotFoundError("Run pairs and annotate before koopman.")

    features = pd.read_csv(feature_path)
    representations = np.load(representation_path)
    summary_rows = []
    mode_rows = []
    coefficient_rows = []
    aligned_rows = []

    for layer in LAYERS:
        source = representations[f"source_{layer}"].astype(np.float64)
        for target_type in ("ground_truth", "predicted"):
            target = representations[f"{target_type}_{layer}"].astype(np.float64)
            branch = features[features["target_type"] == target_type].sort_values("pair_id")
            if len(branch) != len(source):
                raise RuntimeError("Feature rows and representation rows do not align.")
            regression_matrix, feature_names = feature_matrix(branch)
            result = fit_koopman(source, target, args.koopman_rank, args.ridge_alpha)
            stem = f"{layer}_{target_type}"
            np.save(output_dir / f"K_{stem}.npy", result["operator"])
            np.savez_compressed(
                output_dir / f"koopman_state_{stem}.npz",
                basis=result["basis"],
                center=result["center"],
                scale=result["scale"],
                eigenvalues=result["eigenvalues"],
            )

            coefficients = []
            for mode in range(result["rank"]):
                alignment = result["alignments"][:, mode]
                centered = alignment - alignment.mean()
                gram = regression_matrix.T @ regression_matrix
                regression = np.linalg.solve(
                    gram + args.ridge_alpha * np.eye(gram.shape[0]),
                    regression_matrix.T @ centered,
                )
                fitted = regression_matrix @ regression + alignment.mean()
                total = np.sum((alignment - alignment.mean()) ** 2)
                mode_r2 = 1.0 - np.sum((alignment - fitted) ** 2) / total if total > 0 else np.nan
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
                        "feature_regression_r2": mode_r2,
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
                for rank_index, row_index in enumerate(
                    np.argsort(alignment)[::-1][: args.top_utterances], start=1
                ):
                    utterance = branch.iloc[row_index]
                    aligned_rows.append(
                        {
                            "layer": layer,
                            "target_type": target_type,
                            "mode": mode,
                            "rank": rank_index,
                            "alignment": alignment[row_index],
                            "pair_id": int(utterance["pair_id"]),
                            "utterance": utterance["target_utterance"],
                        }
                    )

            coefficient_frame = pd.DataFrame(coefficients, columns=feature_names)
            plot_koopman(
                result,
                coefficient_frame,
                output_dir / f"eigenmodes_{stem}.png",
                f"{layer.replace('_', ' ').title()} layer — {target_type.replace('_', ' ')}",
            )
            summary_rows.append(
                {
                    "layer": layer,
                    "target_type": target_type,
                    "koopman_rank": result["rank"],
                    "transition_r2": result["r2"],
                    "transition_rmse": result["rmse"],
                    "spectral_radius": result["spectral_radius"],
                    "effective_dimension": result["effective_dimension"],
                    "stable_rank": result["stable_rank"],
                    "numerical_rank": result["numerical_rank"],
                }
            )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_dir / "koopman_summary.csv", index=False)
    pd.DataFrame(mode_rows).to_csv(output_dir / "eigenmode_regressions.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(output_dir / "eigenmode_coefficients.csv", index=False)
    pd.DataFrame(aligned_rows).to_csv(output_dir / "eigenmode_utterances.csv", index=False)
    summary = {
        "operators": len(summary_rows),
        "layers": list(LAYERS),
        "targets": ["ground_truth", "predicted"],
        "lifting_function": "final-token hidden representation produced by each learned layer",
        "fit": (
            "full hidden-space ridge regression Y = X K"
            if args.koopman_rank is None
            else f"POD-reduced rank-{args.koopman_rank} ridge regression Y = X K"
        ),
        "effective_dimension": "singular-value participation ratio of K",
    }
    (output_dir / "koopman_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib.pyplot as plt

    labels = [f"{row['layer']}\n{row['target_type']}" for row in summary_rows]
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(labels, summary_frame["effective_dimension"])
    axis.set(ylabel="Participation-ratio dimension", title="Effective dimension of K")
    axis.tick_params(axis="x", labelrotation=25)
    figure.tight_layout()
    figure.savefig(output_dir / "effective_dimensions.png", dpi=180)
    plt.close(figure)
    print(summary_frame.to_string(index=False))


def analyze(args: argparse.Namespace) -> None:
    """Run utterance construction, feature annotation, and Koopman analysis."""
    construct_pairs(args)
    annotate(args)
    koopman(args)


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
