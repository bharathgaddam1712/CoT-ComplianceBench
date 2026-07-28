"""
train_modernbert.py — Fine-tune ModernBERT-base as a CoT compliance classifier.

Input  : data/{batch}/labeled/labeled_{model}.jsonl (all models merged), where
         --batch selects which labeled/ dir(s) to pool:
           combined        -> both with_attacks/labeled/ and without_attacks/labeled/
           with_attacks    -> only with_attacks/labeled/
           without_attacks -> only without_attacks/labeled/
Output : data/modernbert/{batch}/  (saved model + tokenizer + eval results)

Labels : compliance_quadrant  Q1→0  Q2→1  Q3→2  Q4→3
Feature: cot_trace (truncated to MAX_SEQ_LEN tokens)

  python src/train_modernbert.py                             # --batch combined (default)
  python src/train_modernbert.py --batch without_attacks
  python src/train_modernbert.py --epochs 5 --seq-len 1024    # faster run
  python src/train_modernbert.py --dry-run                    # 50 rows, 1 epoch
"""

import os
import json
import argparse
import random
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score

from config import DATA_DIR, MODELS, SEED, batch_dirs

# ----------------------------------------------------------------------- #
# Paths + constants
# ----------------------------------------------------------------------- #
def labels_dirs_for(batch: str) -> list:
    """Resolve which labeled/ dir(s) to pool training rows from."""
    if batch == "combined":
        return [batch_dirs("without_attacks")["labeled"], batch_dirs("with_attacks")["labeled"]]
    return [batch_dirs(batch)["labeled"]]


BASE_MODEL  = "answerdotai/ModernBERT-base"
MAX_SEQ_LEN = 2048       # CoT traces are long; 2048 balances coverage vs speed
BATCH_SIZE  = 4          # T4 fits 4 at seq_len=2048
GRAD_ACCUM  = 4          # effective batch = 16
EPOCHS      = 3
LR          = 2e-5
WARMUP_FRAC = 0.1

QUAD2ID = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
ID2QUAD = {v: k for k, v in QUAD2ID.items()}
NUM_LABELS = 4


# ----------------------------------------------------------------------- #
# Data loading
# ----------------------------------------------------------------------- #
def load_all_labeled(dry_run: bool, labels_dirs: list) -> list:
    rows = []
    for labels_dir in labels_dirs:
        for mkey in MODELS:
            path = labels_dir / f"labeled_{mkey}.jsonl"
            if not path.exists():
                print(f"[warn] {path} not found — skipping")
                continue
            for line in path.open():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                quad = r.get("compliance_quadrant")
                cot  = (r.get("cot_trace") or "").strip()
                if quad not in QUAD2ID or not cot:
                    continue
                rows.append({"text": cot, "label": QUAD2ID[quad]})
    random.shuffle(rows)
    if dry_run:
        rows = rows[:50]
    return rows


def split(rows, train_frac=0.8, val_frac=0.1):
    n = len(rows)
    t = int(n * train_frac)
    v = int(n * (train_frac + val_frac))
    return rows[:t], rows[t:v], rows[v:]


# ----------------------------------------------------------------------- #
# Dataset
# ----------------------------------------------------------------------- #
class CotDataset(Dataset):
    def __init__(self, rows, tok):
        self.rows = rows
        self.tok  = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(
            r["text"],
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(r["label"], dtype=torch.long),
        }


# ----------------------------------------------------------------------- #
# Class weights (Q3 is rare — upweight it)
# ----------------------------------------------------------------------- #
def compute_class_weights(rows, device):
    counts = Counter(r["label"] for r in rows)
    total  = sum(counts.values())
    weights = [total / (NUM_LABELS * counts.get(i, 1)) for i in range(NUM_LABELS)]
    return torch.tensor(weights, dtype=torch.float).to(device)


# ----------------------------------------------------------------------- #
# Train / eval loops
# ----------------------------------------------------------------------- #
def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss   = loss_fn(logits, labels) / grad_accum
        loss.backward()
        total_loss += loss.item() * grad_accum

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 50 == 0:
            print(f"    step {step}/{len(loader)}  loss={loss.item()*grad_accum:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss   = loss_fn(logits, labels)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    q3_f1    = f1_score(all_labels, all_preds, labels=[2], average="micro", zero_division=0)
    print(classification_report(
        all_labels, all_preds,
        target_names=[ID2QUAD[i] for i in range(NUM_LABELS)],
        zero_division=0,
    ))
    return total_loss / len(loader), macro_f1, q3_f1


# ----------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------- #
def main():
    global MAX_SEQ_LEN, BATCH_SIZE, EPOCHS   # must precede any read of these names below

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=EPOCHS)
    ap.add_argument("--seq-len",    type=int,   default=MAX_SEQ_LEN)
    ap.add_argument("--batch-size", type=int,   default=BATCH_SIZE,
                    help="training batch size (renamed from --batch to avoid clashing "
                         "with the new --batch data-selection flag below)")
    ap.add_argument("--batch",      default="combined",
                    choices=["combined", "with_attacks", "without_attacks"],
                    help="which labeled data to train on: both batches pooled (default), "
                         "or a single batch for a with/without-attacks ablation")
    ap.add_argument("--lr",         type=float, default=LR)
    ap.add_argument("--dry-run",    action="store_true",
                    help="50 rows, 1 epoch — sanity check only")
    args = ap.parse_args()

    MAX_SEQ_LEN = args.seq_len
    BATCH_SIZE  = args.batch_size
    EPOCHS      = 1 if args.dry_run else args.epochs

    output_dir  = DATA_DIR / "modernbert" / args.batch
    labels_dirs = labels_dirs_for(args.batch)

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- load data ---------------------------------------------------- #
    print(f"Loading labeled data (batch={args.batch})...")
    rows = load_all_labeled(args.dry_run, labels_dirs)
    if not rows:
        raise RuntimeError(f"No rows found in {labels_dirs}. Run labeling pipeline first.")

    train_rows, val_rows, test_rows = split(rows)
    print(f"  train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")
    print(f"  train label dist: {Counter(r['label'] for r in train_rows)}")

    # ---- tokenizer + model -------------------------------------------- #
    print(f"Loading {BASE_MODEL}...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    ).to(device)

    # ---- dataloaders -------------------------------------------------- #
    train_ds = CotDataset(train_rows, tok)
    val_ds   = CotDataset(val_rows,   tok)
    test_ds  = CotDataset(test_rows,  tok)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ---- class-weighted loss (Q3 upweighted) -------------------------- #
    class_weights = compute_class_weights(train_rows, device)
    print(f"  class weights: {class_weights.tolist()}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # ---- optimizer + scheduler ---------------------------------------- #
    total_steps   = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_FRAC)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ---- training loop ------------------------------------------------ #
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 loss_fn, device, GRAD_ACCUM)
        print(f"  train loss: {train_loss:.4f}")

        print("  -- validation --")
        val_loss, val_macro_f1, val_q3_f1 = evaluate(model, val_loader, loss_fn, device)
        print(f"  val loss={val_loss:.4f}  macro_f1={val_macro_f1:.4f}  Q3_f1={val_q3_f1:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            model.save_pretrained(output_dir / "best")
            tok.save_pretrained(output_dir / "best")
            print(f"  ✓ saved best model (macro_f1={best_val_f1:.4f})")

    # ---- test eval on best checkpoint --------------------------------- #
    print(f"\n{'='*60}\nTest evaluation (best checkpoint)")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        output_dir / "best"
    ).to(device)
    test_loss, test_macro_f1, test_q3_f1 = evaluate(best_model, test_loader, loss_fn, device)
    print(f"Test  loss={test_loss:.4f}  macro_f1={test_macro_f1:.4f}  Q3_f1={test_q3_f1:.4f}")

    # save results
    results = {
        "batch":         args.batch,
        "base_model":    BASE_MODEL,
        "max_seq_len":   MAX_SEQ_LEN,
        "epochs":        EPOCHS,
        "best_val_f1":   best_val_f1,
        "test_macro_f1": test_macro_f1,
        "test_q3_f1":    test_q3_f1,
        "train_size":    len(train_rows),
        "val_size":      len(val_rows),
        "test_size":     len(test_rows),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {output_dir / 'results.json'}")
    print(f"Model saved   -> {output_dir / 'best'}")


if __name__ == "__main__":
    main()
