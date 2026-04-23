"""
train.py — ScienceQA Vision Challenge
DDP training on 2x T4 GPUs via Accelerate

Usage:
    !accelerate launch --multi_gpu --num_processes=2 /kaggle/working/train.py
"""

import json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator

# ── CONFIG — edit this between runs ──────────────────────────────────────────
CONFIG = {
    # Identity
    "run_id": "run_01",
    "model_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
    # Paths
    "csv_dir": "/kaggle/input/competitions/pixels-to-predictions",
    "image_dir": "/kaggle/input/competitions/pixels-to-predictions/images/images",
    "output_dir": "/kaggle/working/output",
    # LoRA
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # Training
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size": 4,
    "grad_accum": 4,  # effective batch per GPU = 16, total = 32
    "img_size": 224,
    "seed": 42,
    # Eval
    "quick_val_n": 150,
}
# ─────────────────────────────────────────────────────────────────────────────

# ── Accelerator ───────────────────────────────────────────────────────────────
accelerator = Accelerator(mixed_precision="fp16")

OUTPUT_DIR = Path(CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])

if accelerator.is_main_process:
    print(f"\n{'=' * 55}")
    print(f"  Run       : {CONFIG['run_id']}")
    print(f"  GPUs      : {accelerator.num_processes}")
    print(
        f"  r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']}, "
        f"lr={CONFIG['learning_rate']}, epochs={CONFIG['num_epochs']}"
    )
    print(f"  Targets   : {CONFIG['lora_targets']}")
    print(f"{'=' * 55}\n")

# ── Stage 1: Data ─────────────────────────────────────────────────────────────
CSV_DIR = Path(CONFIG["csv_dir"])
IMAGE_DIR = Path(CONFIG["image_dir"])

train_df = pd.read_csv(CSV_DIR / "train.csv")
train_df["choices"] = train_df["choices"].apply(
    lambda x: json.loads(x) if isinstance(x, str) else x
)

if accelerator.is_main_process:
    print(f"Train: {len(train_df):,} examples")


def build_prompt(row, include_answer=False):
    parts = ["<image>"]
    ctx = []
    if pd.notna(row.get("hint")) and str(row["hint"]).strip():
        ctx.append(str(row["hint"]).strip())
    if pd.notna(row.get("lecture")) and str(row["lecture"]).strip():
        ctx.append(str(row["lecture"]).strip())
    if ctx:
        parts.append("Context:\n" + "\n".join(ctx))
    parts.append(f"Question: {row['question']}")
    lines = ["Choices:"]
    for i, c in enumerate(row["choices"]):
        lines.append(f"  {chr(65 + i)}. {c}")
    parts.append("\n".join(lines))
    parts.append("Answer:")
    if include_answer:
        parts.append(chr(65 + int(row["answer"])))
    return "\n".join(parts)


class ScienceQADataset(Dataset):
    def __init__(self, df, split):
        self.df = df.reset_index(drop=True)
        self.img_dir = IMAGE_DIR / split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / f"{row['id']}.png").convert("RGB")
        img = img.resize((CONFIG["img_size"], CONFIG["img_size"]), Image.BICUBIC)
        return {
            "image": img,
            "text": build_prompt(row, include_answer=True),
            "answer": int(row["answer"]),
        }


train_ds = ScienceQADataset(train_df, "train")

# ── Stage 2: Model ────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(CONFIG["model_id"])
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

# process_index maps each DDP process to its own GPU (0 or 1)
model = AutoModelForVision2Seq.from_pretrained(
    CONFIG["model_id"],
    quantization_config=bnb_config,
    device_map={"": accelerator.process_index},
    low_cpu_mem_usage=True,
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(
    model,
    LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_targets"],
        bias="none",
    ),
)

if accelerator.is_main_process:
    model.print_trainable_parameters()


# ── Stage 3: DataLoader ───────────────────────────────────────────────────────
def collate_fn(batch):
    inputs = processor(
        text=[b["text"] for b in batch],
        images=[b["image"] for b in batch],
        return_tensors="pt",
        padding=True,
    )
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs, torch.tensor([b["answer"] for b in batch])


train_loader = DataLoader(
    train_ds,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)

# ── Stage 4: Optimizer + Scheduler ───────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)
total_steps = (len(train_loader) // CONFIG["grad_accum"]) * CONFIG["num_epochs"]
warmup_steps = max(50, int(0.05 * total_steps))


def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))


scheduler = LambdaLR(optimizer, lr_lambda)

if accelerator.is_main_process:
    print(f"Steps/epoch : {len(train_loader)}")
    print(f"Total steps : {total_steps} | Warmup: {warmup_steps}")
    print(
        f"Eff. batch  : {CONFIG['batch_size'] * CONFIG['grad_accum']} per GPU "
        f"({CONFIG['batch_size'] * CONFIG['grad_accum'] * accelerator.num_processes} total)\n"
    )

# ── Stage 5: DDP Prepare ──────────────────────────────────────────────────────
model, optimizer, train_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, scheduler
)

# ── Stage 6: Training Loop ────────────────────────────────────────────────────
run_start = time.time()
model.train()
global_step = 0

for epoch in range(CONFIG["num_epochs"]):
    epoch_loss = 0.0
    t0 = time.time()
    optimizer.zero_grad()

    for step, (inputs, _) in enumerate(train_loader):
        loss = model(**inputs).loss / CONFIG["grad_accum"]
        accelerator.backward(loss)
        epoch_loss += loss.item() * CONFIG["grad_accum"]

        if (step + 1) % CONFIG["grad_accum"] == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        if step % 100 == 0 and accelerator.is_main_process:
            print(
                f"Epoch {epoch + 1}/{CONFIG['num_epochs']} | "
                f"Step {step}/{len(train_loader)} | "
                f"Loss: {epoch_loss / max(1, step + 1):.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"{(time.time() - t0) / 60:.1f}min"
            )

    if accelerator.is_main_process:
        print(
            f"\n✓ Epoch {epoch + 1} | "
            f"Avg loss: {epoch_loss / len(train_loader):.4f} | "
            f"{(time.time() - t0) / 60:.1f} min\n"
        )

total_min = (time.time() - run_start) / 60

# ── Stage 7: Save ─────────────────────────────────────────────────────────────
if accelerator.is_main_process:
    adapter_dir = OUTPUT_DIR / f"adapter_{CONFIG['run_id']}"
    accelerator.unwrap_model(model).save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    (OUTPUT_DIR / "train_results.json").write_text(
        json.dumps(
            {
                "run_id": CONFIG["run_id"],
                "adapter_dir": str(adapter_dir),
                "total_time_min": round(total_min, 1),
                "status": "done",
            },
            indent=2,
        )
    )

    print(f"✅ Done | adapter → {adapter_dir} | {total_min:.1f} min")

accelerator.wait_for_everyone()
