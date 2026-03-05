#!/usr/bin/env python3
"""Export SPECTER2 text encoder + tokenizer for Rust/tch runtime.

Outputs:
- static/models/specter2_traced.pt
- static/models/tokenizer/tokenizer.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trace SPECTER2 encoder (with adapter) and export tokenizer.json."
    )
    p.add_argument(
        "--model",
        default="allenai/specter2_aug2023refresh",
        help="Base SPECTER2 family id (default: allenai/specter2_aug2023refresh)",
    )
    p.add_argument(
        "--adapter",
        default="adhoc_query",
        help="Adapter name or full HF id (default: adhoc_query)",
    )
    p.add_argument(
        "--pooling",
        choices=["mean", "cls"],
        default="mean",
        help="Token pooling for sentence embedding (default: mean)",
    )
    p.add_argument(
        "--orthogonalize",
        action="store_true",
        default=True,
        help="Enable reference-vector orthogonalization (default: enabled)",
    )
    p.add_argument(
        "--no-orthogonalize",
        dest="orthogonalize",
        action="store_false",
        help="Disable reference-vector orthogonalization",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="torch device for export (default: cpu)",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max length (default: 512)",
    )
    p.add_argument(
        "--out-model",
        default="static/models/specter2_traced.pt",
        help="Output path for traced TorchScript model",
    )
    p.add_argument(
        "--tokenizer-dir",
        default="static/models/tokenizer",
        help="Output directory for tokenizer files",
    )
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load only from local HF cache (no downloads)",
    )
    return p


class Specter2Encoder(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        pooling: str = "mean",
        ref: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        if ref is None:
            self.register_buffer("ref", torch.empty(0), persistent=False)
        else:
            self.register_buffer("ref", ref, persistent=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # return_dict=False gives tuple output that traces reliably.
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        hidden = outputs[0]  # [batch, seq, hidden]

        if self.pooling == "cls":
            emb = hidden[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1e-9)
            emb = pooled / denom

        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        if self.ref.numel() > 0:
            proj = (emb * self.ref).sum(dim=-1, keepdim=True)
            emb = emb - proj * self.ref
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb


def resolve_base_model(model: str) -> str:
    return model if model.endswith("_base") else f"{model}_base"


def resolve_adapter_id(model: str, adapter: str | None) -> str | None:
    if not adapter:
        return None
    if "/" in adapter:
        return adapter
    if adapter == "proximity":
        return "allenai/specter2"
    return f"{model}_{adapter}"


def main() -> None:
    args = build_parser().parse_args()

    # Avoid transformers pulling TF/Flax stacks.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")

    from adapters import AutoAdapterModel
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(args.device)
    base_model = resolve_base_model(args.model)
    adapter_id = resolve_adapter_id(args.model, args.adapter)

    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )

    print(f"Loading model backbone: {base_model}")
    if adapter_id is None:
        backbone = AutoModel.from_pretrained(
            base_model,
            local_files_only=args.local_files_only,
        )
    else:
        backbone = AutoAdapterModel.from_pretrained(
            base_model,
            local_files_only=args.local_files_only,
        )
        print(f"Loading adapter: {adapter_id}")
        backbone.load_adapter(
            adapter_id,
            source="hf",
            load_as="specter2",
            set_active=True,
            local_files_only=args.local_files_only,
        )

    backbone = backbone.to(device).eval()

    def tokenize(text: str) -> dict[str, torch.Tensor]:
        toks = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        return {k: v.to(device) for k, v in toks.items()}

    ref = None
    if args.orthogonalize:
        with torch.inference_mode():
            tmp = Specter2Encoder(backbone, pooling=args.pooling, ref=None).to(device).eval()
            empty = tokenize("")
            ref = tmp(empty["input_ids"], empty["attention_mask"])
        print("Computed orthogonalization reference vector from empty input.")

    wrapper = Specter2Encoder(backbone, pooling=args.pooling, ref=ref).to(device).eval()

    sample = tokenize("test input for tracing")
    with torch.inference_mode():
        traced = torch.jit.trace(
            wrapper,
            (sample["input_ids"], sample["attention_mask"]),
            strict=False,
        )

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_model))

    tokenizer_dir = Path(args.tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    tok_json = tokenizer_dir / "tokenizer.json"
    if not tok_json.exists() and hasattr(tokenizer, "backend_tokenizer"):
        tokenizer.backend_tokenizer.save(str(tok_json))

    if not tok_json.exists():
        raise RuntimeError(f"Failed to export tokenizer.json at {tok_json}")

    print(f"Saved traced encoder: {out_model}")
    print(f"Saved tokenizer files: {tokenizer_dir}")
    print(f"Tokenizer JSON: {tok_json}")


if __name__ == "__main__":
    main()
