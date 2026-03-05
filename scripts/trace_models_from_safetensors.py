#!/usr/bin/env python3
"""Trace decoder and aligner TorchScript modules from safetensors checkpoints.

The safetensors files provide only linear layer weights, so activation choice must be
provided explicitly. Default is ReLU to match prior NeuroVLM assumptions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file


def build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def build_aligner(state: dict[str, torch.Tensor], activation: str) -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(768, 512),
        build_activation(activation),
        nn.Linear(512, 384),
    )
    with torch.no_grad():
        model[0].weight.copy_(state["aligner.0.weight"])
        model[0].bias.copy_(state["aligner.0.bias"])
        model[2].weight.copy_(state["aligner.2.weight"])
        model[2].bias.copy_(state["aligner.2.bias"])
    model.eval()
    return model


def build_decoder(state: dict[str, torch.Tensor], activation: str) -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(384, 512),
        build_activation(activation),
        nn.Linear(512, 1024),
        build_activation(activation),
        nn.Linear(1024, 28542),
    )
    with torch.no_grad():
        model[0].weight.copy_(state["decoder.0.weight"])
        model[0].bias.copy_(state["decoder.0.bias"])
        model[2].weight.copy_(state["decoder.2.weight"])
        model[2].bias.copy_(state["decoder.2.bias"])
        model[4].weight.copy_(state["decoder.4.weight"])
        model[4].bias.copy_(state["decoder.4.bias"])
    model.eval()
    return model


def trace_and_save(model: nn.Module, example_input: torch.Tensor, out_path: Path) -> None:
    traced = torch.jit.trace(model, example_input)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create aligner_traced.pt and decoder_traced.pt from safetensors."
    )
    parser.add_argument(
        "--autoencoder",
        default="static/models/autoencoder.safetensors",
        help="Path to autoencoder.safetensors",
    )
    parser.add_argument(
        "--aligner-safetensors",
        default="static/models/proj_head_text_mse.safetensors",
        help="Path to proj_head_text_mse.safetensors",
    )
    parser.add_argument(
        "--decoder-out",
        default="static/models/decoder_traced.pt",
        help="Output path for traced decoder",
    )
    parser.add_argument(
        "--aligner-out",
        default="static/models/aligner_traced.pt",
        help="Output path for traced aligner",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "gelu", "none"],
        default="relu",
        help="Hidden activation for MLPs (default: relu)",
    )
    args = parser.parse_args()

    decoder_state = load_file(args.autoencoder)
    aligner_state = load_file(args.aligner_safetensors)

    decoder = build_decoder(decoder_state, args.activation)
    aligner = build_aligner(aligner_state, args.activation)

    trace_and_save(decoder, torch.randn(1, 384), Path(args.decoder_out))
    trace_and_save(aligner, torch.randn(1, 768), Path(args.aligner_out))

    print(f"Saved decoder: {args.decoder_out}")
    print(f"Saved aligner: {args.aligner_out}")


if __name__ == "__main__":
    main()
