"""garum's calibrated Bayesian last-layer head (inference only).

garum's answer to a prompt is this head's posterior predictive, computed from one forward pass of the adapted model: the final
hidden state h at the answer position and the model's own next-token logits ell of the option letters. For the option shown under
letter j of a K-option prompt:

    z_j = tau * ell_j + phi_j^T beta,      beta ~ N(mu, L L^T)
    phi_j = u * (c_j - mean_{k<K} c_k) / sqrt(F),   u = rms(P rms(h)),   c_j = a fixed code per letter
    p = mean over the stored posterior draws of softmax(z)

The draws are stored in the head file (`eps`, standard normal), so the answer is deterministic.

Files: vbll_head.safetensors (P, mu, L_diag, L_off, log_tau, codes, eps; every tensor has a leading head axis of size 1) and
vbll_head.json (format, dimensions, the letter token ids, the tensor file's sha256).
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import torch

HEAD_FILE, HEAD_META = "vbll_head.safetensors", "vbll_head.json"
FORMAT = "garum-head-1"
KEYS = ("P", "mu", "L_diag", "L_off", "log_tau", "codes", "eps")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class BayesianHead(torch.nn.Module):
    """The head, with its stored posterior draws. predict(h, ell, ks) -> p [B, K] (zeros beyond each row's K)."""

    def __init__(self, state, meta):
        super().__init__()
        missing = [k for k in KEYS if k not in state]
        if missing:
            raise ValueError(f"head tensors missing: {missing}")
        self.F = int(meta["dim"])
        self.label_ids = [int(x) for x in meta["label_ids"]]
        for k in ("P", "mu", "L_diag", "L_off", "log_tau"):
            setattr(self, k, torch.nn.Parameter(state[k].float(), requires_grad=False))
        self.register_buffer("codes", state["codes"].float())
        self.register_buffer("eps", state["eps"].float())
        G, F, H = self.P.shape
        if G != 1 or F != self.F or tuple(self.codes.shape) != (1, len(self.label_ids), F) or self.eps.shape[0] != 1 \
                or self.eps.shape[2] != F or tuple(self.L_off.shape) != (1, F, F):
            raise ValueError("head tensor shapes do not match vbll_head.json")
        self.hidden = H

    @classmethod
    def from_dir(cls, d):
        """Load and sha256-check the head files in directory `d`."""
        from safetensors.torch import load_file
        p, j = os.path.join(d, HEAD_FILE), os.path.join(d, HEAD_META)
        if not (os.path.exists(p) and os.path.exists(j)):
            raise FileNotFoundError(f"{d}: no {HEAD_FILE} + {HEAD_META}. garum's answers are this head's output; the model must "
                                    "not be served from the letter logits alone.")
        with open(j) as f:
            meta = json.load(f)
        if meta.get("format") != FORMAT:
            raise ValueError(f"{j}: format {meta.get('format')!r}, expected {FORMAT!r}")
        if _sha256(p) != meta["tensor_sha256"]:
            raise ValueError(f"{p}: sha256 differs from {HEAD_META}")
        return cls(load_file(p), meta), meta

    def sigma(self):
        return torch.nn.functional.softplus(self.L_diag) + 1e-6

    def chol(self):
        return torch.tril(self.L_off, -1) + torch.diag_embed(self.sigma())

    @torch.no_grad()
    def predict(self, h, ell, ks):
        """h [B, H] (the hidden state the letter logits are read from), ell [B, >= max K] (the letter logits, letter order),
        ks [B] (options shown per prompt) -> p [B, kmax] float32, the posterior predictive (0 beyond each row's K)."""
        ks = torch.as_tensor(ks, device=ell.device)
        kmax = int(ks.max())
        ell = ell[:, :kmax]
        mask = torch.arange(kmax, device=ell.device)[None, :] < ks[:, None]
        m = mask.float()
        h = h.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        u = torch.einsum("gfh,bh->gbf", self.P, h)
        u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6)
        w = self.codes[:, :kmax]
        wbar = torch.einsum("bk,gkf->gbf", m, w) / ks.to(w.dtype).clamp_min(1)[None, :, None]
        phi = u[:, :, None, :] * (w[:, None, :, :] - wbar[:, :, None, :]) * (m[None, :, :, None] / math.sqrt(self.F))
        beta = self.mu[:, None] + torch.einsum("gef,gsf->gse", self.chol(), self.eps)
        z = self.log_tau.exp()[:, None, None, None] * ell.float()[None, None] + torch.einsum("gbkf,gsf->gsbk", phi, beta)
        ps = torch.softmax(z.masked_fill(~mask[None, None], -1e9), -1)          # [1, S, B, K]
        return ps.mean(1)[0]
