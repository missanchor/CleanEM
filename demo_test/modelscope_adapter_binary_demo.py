#!/usr/bin/env python3
"""Binary classification demo using a custom adapter head on top of a ModelScope LLM.

The script downloads the Qwen 0.5B instruct model from ModelScope, freezes it,
and trains a lightweight adapter head for a toy two-class intent classification
task. It is designed to be quick to run (no gradient updates to the LLM) while
showing how a custom adapter can be attached to an LLM backbone.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope.hub.snapshot_download import snapshot_download
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


LOG_FORMAT = "[%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger("binary-adapter-demo")


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class Sample:
    text: str
    label: int


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        tokenizer,
        max_length: int = 256,
        device: torch.device | None = None,
    ) -> None:
        texts = [sample.text for sample in samples]
        self.labels = torch.tensor([sample.label for sample in samples], dtype=torch.long)
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        if device is not None:
            self.input_ids = self.input_ids.to(device)
            self.attention_mask = self.attention_mask.to(device)
            self.labels = self.labels.to(device)

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


class BinaryAdapterHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        target_dtype = next(self.parameters()).dtype
        hidden_states = hidden_states.to(target_dtype)
        # pooled mean over non-masked tokens
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / lengths
        return self.net(pooled)


class FrozenBackboneBinaryClassifier(nn.Module):
    def __init__(self, backbone: AutoModelForCausalLM, adapter_head: BinaryAdapterHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter_head = adapter_head
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
            )
            if outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = outputs[0]
        logits = self.adapter_head(hidden_states, attention_mask)
        return logits


def prepare_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    if not force_cpu and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def build_toy_dataset() -> tuple[List[Sample], List[Sample]]:
    train_samples = [
        Sample("这个产品非常好用，我会推荐给朋友。", 1),
        Sample("服务态度很差，体验糟糕。", 0),
        Sample("问题解决得很及时，五星好评！", 1),
        Sample("再也不会购买了，质量太差。", 0),
        Sample("客服沟通顺畅，问题顺利解决。", 1),
        Sample("包裹破损严重，投诉无门。", 0),
    ]

    eval_samples = [
        Sample("体验不错，功能符合预期。", 1),
        Sample("物流太慢，等得人心烦。", 0),
    ]
    return train_samples, eval_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a binary adapter head on a frozen LLM backbone")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="ModelScope repo id for the base LLM")
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "modelscope"))
    parser.add_argument("--local-dir", default=None, help="Optional directory to download the LLM into")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    LOGGER.info("Using device auto-detection (force_cpu=%s)", args.force_cpu)
    device = prepare_device(force_cpu=args.force_cpu)
    LOGGER.info("Device: %s", device)

    LOGGER.info("Downloading base model %s", args.model_id)
    model_dir = snapshot_download(
        args.model_id,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
    )

    LOGGER.info("Loading tokenizer from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading backbone model from %s", model_dir)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    backbone = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=dtype,
    )
    backbone.to(device)

    adapter_head = BinaryAdapterHead(hidden_size=backbone.config.hidden_size).to(device)

    classifier = FrozenBackboneBinaryClassifier(backbone, adapter_head).to(device)

    train_samples, eval_samples = build_toy_dataset()
    train_dataset = TextClassificationDataset(train_samples, tokenizer, args.max_length, device=device)
    eval_dataset = TextClassificationDataset(eval_samples, tokenizer, args.max_length, device=device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    optimizer = torch.optim.AdamW(classifier.adapter_head.parameters(), lr=args.lr)

    LOGGER.info("Starting training for %d epochs", args.epochs)
    classifier.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = classifier(batch["input_ids"], batch["attention_mask"])
            loss = F.cross_entropy(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        LOGGER.info("Epoch %d/%d - loss: %.4f", epoch, args.epochs, avg_loss)

    LOGGER.info("Evaluating on validation samples")
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_loader:
            logits = classifier(batch["input_ids"], batch["attention_mask"])
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
            LOGGER.info("Input: %s", tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))
            LOGGER.info("Pred probs: %s", torch.softmax(logits, dim=-1).cpu().numpy())
            LOGGER.info("Pred label: %d", preds.item())

    accuracy = correct / total if total else 0.0
    LOGGER.info("Validation accuracy: %.2f%% (%d/%d)", accuracy * 100, correct, total)


if __name__ == "__main__":
    main()

