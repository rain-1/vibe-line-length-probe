import argparse
import os
from typing import Dict, Optional

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency at runtime
    wandb = None


NUM_CLASSES = 70  # line lengths 0..69 inclusive; rows with >=70 are dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train logistic probes for line length.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/line_length_dataset_sample.jsonl",
        help="Path to JSONL with fields: text, line_length.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-3-270m",
        help="Base model checkpoint for features.",
    )
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Where to save probe weights.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256, help="Truncate/pad sequences to this length.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=50, help="Log train loss every N steps.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on samples for quick experiments.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="line-length-probes")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g., cpu or cuda.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_prepare_dataset(
    data_path: str,
    tokenizer,
    max_length: int,
    val_split: float,
    seed: int,
    limit: Optional[int],
):
    dataset = load_dataset("json", data_files={"train": data_path})["train"]
    dataset = dataset.filter(lambda ex: ex["line_length"] < NUM_CLASSES)

    if limit:
        dataset = dataset.select(range(limit))

    dataset = dataset.shuffle(seed=seed)
    split = dataset.train_test_split(test_size=val_split, seed=seed)

    def preprocess(batch: Dict[str, list]):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = batch["line_length"]
        return tokenized

    split = split.map(preprocess, batched=True, remove_columns=split["train"].column_names)
    return split["train"], split["test"]


class ProbeModel(nn.Module):
    def __init__(self, lm: AutoModelForCausalLM):
        super().__init__()
        self.lm = lm
        for param in self.lm.parameters():
            param.requires_grad = False
        hidden_size = self.lm.config.hidden_size
        self.classifier = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)
        lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        last_tokens = hidden[batch_indices, lengths]
        logits = self.classifier(last_tokens)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return logits, loss


def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, loss = model(**batch, labels=labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
    return {
        "loss": total_loss / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    lm = AutoModelForCausalLM.from_pretrained(args.model_id)
    probe_model = ProbeModel(lm).to(device)

    train_ds, val_ds = load_and_prepare_dataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_split=args.val_split,
        seed=args.seed,
        limit=args.limit,
    )

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        probe_model.classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed; pip install wandb.")
        wandb_mode = os.environ.get("WANDB_MODE", None)
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            mode=wandb_mode,
        )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        probe_model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, loss = probe_model(**batch, labels=labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            global_step += 1

            if global_step % args.log_every == 0:
                step_log = {"train/loss": loss.item(), "step": global_step, "epoch": epoch}
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                if run:
                    run.log(step_log)

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(probe_model, val_loader, device)
        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        print(log)
        if run:
            run.log(log)

    os.makedirs(args.output_dir, exist_ok=True)
    probe_path = os.path.join(args.output_dir, "probe_head.pt")
    torch.save(probe_model.classifier.state_dict(), probe_path)
    print(f"Saved probe head to {probe_path}")

    if run:
        artifact = wandb.Artifact("probe_head", type="model")
        artifact.add_file(probe_path)
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
