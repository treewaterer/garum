#!/usr/bin/env python3
"""garum as a Decision Index engine (the kit's `Engine` interface: engine(state, questions) -> (response, raw)), for the kit's runner
(github.com/apolinario/decision-index; tested at commit 9eb2dbe2).

    python -m decision_index run --engine garum_engine:GarumEngine \\
        --option base=google/gemma-4-31B-it --option adapter=plantwaterer/garum \\
        --rows rows.jsonl.gz --out runs/garum

The engine runs modeling_garum.GarumForDecision.predict_proba on each request; the inference procedure is described there. In short:
one prompt per question (a choice question's options in one fixed non-identity order derived from a hash of the request; yes / no
definitions shown when the request provides them, unless noul_defs=false), one forward pass per prompt, the answer is the
calibrated Bayesian last-layer head's output, more than 52 options in two rounds, nothing truncated (a request that does not fit
`max_len` tokens is reported as unsupported).

Options (all optional): base, adapter (a Hub repo id or a local directory with the adapter and head files), revision, device
(default cuda), max_len (32768), tokens_per_batch (24576), batching (exact | pad; default exact), noul_defs (true | false; default
true), attn_implementation (default: transformers' own, sdpa).
Dependencies: torch, transformers, peft, safetensors, numpy, and the kit on sys.path.
"""
from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import garum_core as core                                          # noqa: E402
import garum_head                                                  # noqa: E402
import modeling_garum as mg                                        # noqa: E402

from decision_index.engines.base import Engine, Unsupported      # noqa: E402  (the kit must be on sys.path)


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


class GarumEngine(Engine):
    name = "garum"
    latency = ("CUDA-synchronized in-process request wall time: conversion, one prompt per question (equal-length prompts of a "
               "request together, unpadded; K > 52 round 2 after round 1), the head's output; excludes model loading.")

    def __init__(self, base="google/gemma-4-31B-it", adapter="plantwaterer/garum", revision=None, device="cuda", max_len=32768,
                 tokens_per_batch=24576, batching="exact", noul_defs=True, attn_implementation=None, **options):
        super().__init__(**options)
        import torch
        kw = {"attn_implementation": str(attn_implementation)} if attn_implementation else {}
        self.model = mg.GarumForDecision.from_pretrained(adapter=adapter, base=base, revision=revision, device_map=device,
                                                         max_len=int(max_len), tokens_per_batch=int(tokens_per_batch),
                                                         batching=str(batching), noul_defs=mg.as_bool(noul_defs), **kw)
        self.device = device
        self._sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else (lambda: None)
        import peft
        import transformers
        here = os.path.dirname(os.path.abspath(__file__))
        self.provenance = {"kind": "in-process", "base": base, "adapter": str(adapter), "revision": revision, "dtype": "bfloat16",
                           "batching": self.model.config.batching, "tokens_per_batch": self.model.config.tokens_per_batch,
                           "noul_defs": self.model.config.noul_defs, "max_len": self.model.config.max_len,
                           "attn_implementation": getattr(self.model.model.config, "_attn_implementation", None),
                           "readout": "calibrated Bayesian last-layer head (garum_head.py)",
                           "peft": peft.__version__, "transformers": transformers.__version__, "torch": torch.__version__,
                           "files_md5": {f: _md5(os.path.join(here, f)) for f in
                                         ("garum_engine.py", "modeling_garum.py", "garum_core.py", "garum_head.py")}}

    def synchronize(self):
        self._sync()

    def runtime(self):
        try:
            import torch
            return {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        except Exception:                          # noqa: BLE001
            return {}

    def __call__(self, state, questions):
        try:
            answers = self.model.predict_proba(state, questions)
        except mg.Unsupported as e:
            raise Unsupported(str(e)) from None
        return {"model": self.name, "answers": answers}, None


__all__ = ["GarumEngine", "core", "garum_head"]
