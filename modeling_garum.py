"""garum as a PyTorch model: `GarumForDecision`, a torch.nn.Module that answers multiple-choice decision questions with a probability
for every option.

    from modeling_garum import GarumForDecision
    model = GarumForDecision.from_pretrained()            # google/gemma-4-31B-it + plantwaterer/garum, bf16, on "cuda"
    answers = model.predict_proba(
        "The forecast gives a 70% chance of heavy rain this afternoon. The match is outdoors, on grass.",
        {"plan": {"type": "choice", "instructions": "What should the organisers do?",
                  "criteria": {"play": "Play as planned", "indoors": "Move it indoors", "postpone": "Postpone it"}}})
    # {"plan": {"type": "choice", "choice": ..., "probabilities": {"play": ..., "indoors": ..., "postpone": ...}}}

garum is a LoRA adapter for google/gemma-4-31B-it plus a calibrated Bayesian last-layer head (garum_head.py). Its answer is the
head's output, computed from one forward pass per prompt: the final hidden state at the answer position and the option letters'
next-token logits go through the head. The letter logits alone are not garum's answer.

Three levels of use:
  * `predict_proba(state, questions)`: a request (a state plus one or more named questions) -> an answer per question. This is
    the computation garum_engine.GarumEngine runs for the Decision Index.
  * `encode` / `encode_request` + `collate` + `forward`: build the exact prompts, batch them, and run the module yourself.
  * `forward(input_ids, attention_mask, option_token_ids, ...)`: the head's option probabilities (and the letter logits they
    were computed from) at each row's last real token, and an optional negative log-likelihood over the options.

Questions follow the Decision Index request format: {"type": "choice", "instructions": ..., "criteria": {key: description}} or
{"type": "noul", "instructions": ...} (a yes / no question; the answer is p(yes)).

Inference procedure:
  * each question becomes one prompt (the model's chat template; options lettered A-Z a-z, at most 52 per prompt). A choice
    question's options are shown in one fixed non-identity order derived from a hash of (state, question key, option keys);
    yes / no keeps the order Yes, No. With `noul_defs` (default True) a yes / no question's definitions (its `criteria`, keyed
    yes / no or true / false), when the request provides them, are shown as "Yes: <definition>" / "No: <definition>";
  * one forward pass per prompt; the head's posterior predictive over the shown options -> mapped back to the options;
  * a choice with more than 52 options is split into ceil(K / 52) chunks (round 1); round 2 asks again over the top 52 // C options
    of every chunk; the answer puts (1 - 0.001) on the round-2 options and 0.001 on the rest, in proportion to round 1;
  * no truncation: a prompt that does not fit `max_len` tokens without cutting the state raises `Unsupported`;
  * batching (`batching`): "exact" (default) runs prompts of equal token length together, unpadded, up to `tokens_per_batch`
    tokens, in a stable order (request order among equal lengths), and every other prompt alone; "pad" runs consecutive prompts
    in right-padded batches of at most `tokens_per_batch` padded tokens. Batch composition (which prompts run together, and in
    which order) can move bf16 outputs slightly: a letter logit by a bf16 step.
The adapter is loaded with autocast_adapter_dtype=False, so the LoRA update is computed in the base model's dtype (bf16); peft's
default would upcast it to fp32 and shift some letter logits by a bf16 step. Attention is the transformers default (sdpa).

nn.Module behaviour: the wrapped transformers model (`.model`) and the head (`.head`) are registered submodules, so `.to()`,
`.eval()`, `parameters()`, `state_dict()` behave as usual. `from_pretrained` returns the module in eval mode with every parameter
frozen. Use eager mode (no torch.compile) to reproduce the reported results.

Dependencies: torch, transformers, peft, safetensors, numpy. Tested with torch 2.13, transformers 5.12.1, peft 0.21.2.
"""
from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers.utils import ModelOutput

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import garum_core as core                                          # noqa: E402
import garum_head                                                  # noqa: E402

__all__ = ["GarumConfig", "GarumOutput", "EncodedQuestion", "GarumForDecision", "Unsupported"]

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", garum_head.HEAD_FILE, garum_head.HEAD_META)


class Unsupported(ValueError):
    """A request the model does not answer: an unsupported question type, or a prompt that would not fit `max_len` tokens
    without cutting the state (nothing is truncated)."""


@dataclass
class GarumConfig:
    base: str = "google/gemma-4-31B-it"
    adapter: str = "plantwaterer/garum"
    max_len: int = 32768
    tokens_per_batch: int = 24576
    batching: str = "exact"                       # "exact" | "pad"
    noul_defs: bool = True
    letters: str = core.LABELS                     # A-Z a-z: at most 52 options per prompt


@dataclass
class GarumOutput(ModelOutput):
    """`probs` [B, K]: the head's probabilities over the shown options (0 where `option_mask` is False); `log_probs` [B, K]: their
    log; `option_logits` [B, K]: the next-token logits at the option letters the head read (float32; -inf where `option_mask` is
    False); `loss`: mean negative log-likelihood of the gold option when `labels` (its letter index, -100 to ignore) are given."""
    loss: Optional[torch.FloatTensor] = None
    probs: Optional[torch.FloatTensor] = None
    log_probs: Optional[torch.FloatTensor] = None
    option_logits: Optional[torch.FloatTensor] = None


@dataclass
class EncodedQuestion:
    """One prompt. `order[j]` is the index (into `option_keys`) of the option shown under letter j."""
    qid: str
    text: str
    input_ids: list
    order: list
    option_keys: list
    kind: str
    key: str = field(default="")


def request_id(state, questions):
    return "eng-" + hashlib.sha1(json.dumps({"state": state, "questions": questions}, sort_keys=True, ensure_ascii=False,
                                            default=str).encode("utf-8")).hexdigest()[:16]


def question_seed(state_text, qid_key, keys):
    h = hashlib.sha256(json.dumps([state_text, qid_key, list(keys)], ensure_ascii=False).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def option_order(qtype, k, seed):
    """choice: one non-identity permutation drawn from a generator seeded by `seed`; any other type: the stored order."""
    if qtype != "choice" or k < 2:
        return list(range(k))
    rng = np.random.default_rng(seed)
    while True:
        o = [int(x) for x in rng.permutation(k)]
        if o != list(range(k)):
            return o


def as_bool(x):
    if isinstance(x, str):
        if x.strip().lower() in ("1", "true", "yes", "on"):
            return True
        if x.strip().lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a boolean: {x!r}")
    return bool(x)


def _split_request(request_or_state, questions):
    if questions is None:
        if not isinstance(request_or_state, dict) or "questions" not in request_or_state:
            raise TypeError("pass (state, questions) or a request dict {'state': ..., 'questions': {...}}")
        return request_or_state.get("state"), request_or_state["questions"]
    return request_or_state, questions


def resolve_adapter(adapter: str, revision: Optional[str] = None) -> str:
    """A local directory with the adapter and head files, or a Hub repo id (downloaded to the Hub cache)."""
    if os.path.isdir(adapter):
        return adapter
    from huggingface_hub import snapshot_download
    return snapshot_download(adapter, revision=revision, allow_patterns=list(ADAPTER_FILES))


class GarumForDecision(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, tokenizer, head: garum_head.BayesianHead, config: Optional[GarumConfig] = None,
                 chat: Optional[dict] = None):
        super().__init__()
        self.model = model
        self.head = head
        self.tokenizer = tokenizer
        self.config = config or GarumConfig()
        if self.config.batching not in ("exact", "pad"):
            raise ValueError(f"batching {self.config.batching!r}: 'exact' or 'pad'")
        self.chat = chat
        ids = [tokenizer(c, add_special_tokens=False)["input_ids"] for c in self.config.letters]
        if any(len(x) != 1 for x in ids):
            raise ValueError("every option letter must be a single token")
        if [x[0] for x in ids] != list(head.label_ids):
            raise ValueError("the head was fitted with other letter tokens than this tokenizer's")
        if tokenizer.pad_token_id is None:
            raise ValueError("the tokenizer needs a pad token")
        self.register_buffer("letter_token_ids", torch.tensor([x[0] for x in ids], dtype=torch.long), persistent=False)

    # ------------------------------------------------------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, adapter: str = "plantwaterer/garum", base: str = "google/gemma-4-31B-it", dtype=torch.bfloat16,
                        device_map="cuda", max_len: int = 32768, tokens_per_batch: int = 24576, batching: str = "exact",
                        noul_defs: bool = True, revision: Optional[str] = None, **model_kwargs):
        """Base model (`dtype`, `device_map`, extra `model_kwargs` go to the transformers loader) + the LoRA adapter, loaded with
        peft's PeftModel.from_pretrained(..., autocast_adapter_dtype=False), + the head. Eval mode, all parameters frozen."""
        from peft import PeftModel
        from transformers import AutoConfig, AutoTokenizer
        adir = resolve_adapter(adapter, revision)
        head, _ = garum_head.BayesianHead.from_dir(adir)
        mcfg = AutoConfig.from_pretrained(base)
        _, model_cls = core.model_loader(mcfg, "auto")
        tok = AutoTokenizer.from_pretrained(base)
        tok.padding_side = "right"
        m = model_cls.from_pretrained(base, dtype=dtype, device_map=device_map, **model_kwargs).eval()
        pm = PeftModel.from_pretrained(m, adir, is_trainable=False, autocast_adapter_dtype=False)
        inner = pm.get_base_model().eval()          # the same module tree, LoRA layers injected in place
        for p in inner.parameters():
            p.requires_grad_(False)
        if inner.get_output_embeddings().weight.shape[1] != head.hidden:
            raise ValueError("the head reads hidden states of another width than this model's")
        head = head.to(inner.get_output_embeddings().weight.device).eval()
        cfg = GarumConfig(base=base, adapter=str(adapter), max_len=int(max_len), tokens_per_batch=int(tokens_per_batch),
                          batching=str(batching), noul_defs=as_bool(noul_defs))
        return cls(inner, tok, head, cfg, chat=core.chat_preset(mcfg.model_type, False)).eval()

    @property
    def device(self) -> torch.device:
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, NotImplementedError):
            return next(self.model.parameters()).device

    # ------------------------------------------------------------------------------------------------------------- forward
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, option_token_ids: torch.LongTensor,
                option_mask: Optional[torch.BoolTensor] = None, labels: Optional[torch.LongTensor] = None,
                return_dict: bool = True):
        """input_ids / attention_mask [B, T], RIGHT-padded; option_token_ids [B, K]: the token ids of the letters shown in each
        prompt, in letter order (`collate` builds all four); option_mask [B, K]: True for real options; labels [B]: gold letter
        index or -100. A batch without padding runs as one unpadded forward (logits_to_keep=1)."""
        dev = self.device
        input_ids, attention_mask = input_ids.to(dev), attention_mask.to(dev)
        lens = attention_mask.sum(1)
        pos = torch.arange(attention_mask.shape[1], device=dev)
        if not torch.equal(attention_mask, (pos[None, :] < lens[:, None]).to(attention_mask.dtype)) or bool((lens < 1).any()):
            raise ValueError("attention_mask must be right-padded (ones then zeros), with at least one token per row")
        keep = []
        if bool((lens == attention_mask.shape[1]).all()):      # no padding: the unpadded forward, last position only
            with core.head_input(self.model, keep, lambda x: x[:, -1]):
                z = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=1).logits[:, -1]
        else:
            last = lens - 1
            rows = torch.arange(input_ids.shape[0], device=dev)
            with core.head_input(self.model, keep, lambda x: x[rows, rows]):
                z = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=last).logits
            z = z[rows, rows]
        z = z.float()
        h = torch.cat(keep, 0)
        if h.shape[0] != input_ids.shape[0]:
            raise RuntimeError(f"{h.shape[0]} hidden states for {input_ids.shape[0]} prompts")
        k_max = option_token_ids.shape[1]
        if option_mask is None:
            option_mask = torch.ones_like(option_token_ids, dtype=torch.bool)
        option_mask = option_mask.to(dev).bool()
        ks = option_mask.sum(1)
        if not torch.equal(option_mask, torch.arange(k_max, device=dev)[None, :] < ks[:, None]):
            raise ValueError("option_mask must be a prefix mask (the first K letters of each row)")
        ell = z.gather(1, option_token_ids.to(dev))
        probs = self.head.to(dev).predict(h, ell, ks)
        if probs.shape[1] < k_max:
            probs = torch.nn.functional.pad(probs, (0, k_max - probs.shape[1]))
        probs = probs.masked_fill(~option_mask, 0.0)
        log_probs = torch.log(probs)
        option_logits = ell.masked_fill(~option_mask, float("-inf"))
        loss = None
        if labels is not None:
            lab = labels.to(dev)
            ok = lab != -100
            if bool(ok.any()):
                loss = -log_probs[ok].gather(1, lab[ok][:, None]).mean()
            else:
                loss = log_probs.new_zeros(())
        res = GarumOutput(loss=loss, probs=probs, log_probs=log_probs, option_logits=option_logits)
        return res if return_dict else res.to_tuple()

    # ------------------------------------------------------------------------------------------------------------- prompts
    def _unit(self, rec, qid):
        q = rec["questions"][qid]
        keys, texts, _ = core.question_options(q, rec["labels"][qid])
        return {"state": rec["state"], "instructions": q["instructions"], "qtype": q["type"], "option_keys": keys,
                "option_texts": texts, "noul_criteria": q.get("criteria") if q["type"] == "noul" else None}

    def _prompt(self, u, qid, max_len):
        order = option_order(u["qtype"], len(u["option_keys"]), question_seed(u["state"], qid.split("|", 1)[1], u["option_keys"]))
        text = core.build_prompt(self.tokenizer, u, order, max_len, chat=self.chat, noul_defs=self.config.noul_defs)
        if text is None:
            raise Unsupported(f"maximum context length: question {qid.split('|', 1)[1]!r} does not fit {max_len} tokens "
                              f"without cutting the state (no truncation)")
        if u["state"].strip() not in text:
            # The chat template trims the user message, so a state's leading whitespace never reaches the prompt: compare the
            # stripped state. Before 2026-10-20 the raw state was compared, so every state starting with whitespace was refused
            # here as "maximum context length" (Decision Index 0.3 public: MMLU-Pro 304 + iSarcasmEval 5 rows).
            raise Unsupported(f"the prompt for question {qid.split('|', 1)[1]!r} does not contain the full state")
        return order, text

    def _encoded(self, qid, u, order, text):
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return EncodedQuestion(qid=qid, text=text, input_ids=list(ids), order=list(order), option_keys=list(u["option_keys"]),
                               kind=u["qtype"], key=qid.split("|", 1)[1])

    def _prepare(self, state, questions, max_len):
        for k, q in questions.items():
            if q.get("type") not in ("choice", "noul"):
                raise Unsupported(f"question type {q.get('type')!r}")
            if q["type"] == "choice" and len(q.get("criteria") or {}) < 2:
                raise Unsupported("a choice needs at least two options")
        rid = request_id(state, questions)
        stats = {"k_hist": collections.Counter(), "k52_questions": 0, "units": 0}
        row = {"state": state, "questions": questions, "expected": {}, "_evaluation": {"catalog_id": 57, "run_id": rid}}
        rec = core.convert_row(row, None, stats, "text")
        items = []
        for qid in rec["questions"]:
            u = self._unit(rec, qid)
            order, text = self._prompt(u, qid, max_len)
            items.append(self._encoded(qid, u, order, text))
        return rec, items

    def encode_request(self, request_or_state, questions=None, max_len: Optional[int] = None) -> list:
        """The round-1 prompts of a request (a choice with more than 52 options gives one prompt per chunk; its round-2 prompt
        depends on round-1 results and is built inside predict_proba)."""
        state, questions = _split_request(request_or_state, questions)
        return self._prepare(state, questions, int(max_len or self.config.max_len))[1]

    def encode(self, state, question, options=None, kind: str = "choice", key: str = "answer",
               max_len: Optional[int] = None) -> EncodedQuestion:
        """One question -> one prompt. kind 'choice': `options` = {key: description} (or a list of option texts, each its own
        key); 'noul': a yes / no question (`options`: its definitions {yes: ..., no: ...}, optional). `key` is the question's
        name; it enters the option-order hash."""
        max_len = int(max_len or self.config.max_len)
        q = {"type": kind, "instructions": question}
        if kind == "choice":
            if isinstance(options, dict):
                q["criteria"] = options
            else:
                opts = [str(o) for o in options]
                if len(set(opts)) != len(opts):
                    raise ValueError("duplicate option texts: pass {key: description} instead")
                q["criteria"] = {o: o for o in opts}
        elif kind == "noul":
            if options is not None:
                q["criteria"] = options
        else:
            raise Unsupported(f"question type {kind!r}")
        items = self.encode_request(state, {key: q}, max_len=max_len)
        if len(items) != 1:
            raise ValueError(f"{len(q.get('criteria') or {})} options need the two-round procedure: use predict_proba")
        return items[0]

    def collate(self, encoded: list) -> dict:
        """Right-padded batch: input_ids / attention_mask [B, T], option_token_ids / option_mask [B, K] (K = the most options)."""
        lens = [len(e.input_ids) for e in encoded]
        k_max = max(len(e.order) for e in encoded)
        input_ids = torch.full((len(encoded), max(lens)), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max(lens)), dtype=torch.long)
        option_token_ids = torch.zeros((len(encoded), k_max), dtype=torch.long)
        option_mask = torch.zeros((len(encoded), k_max), dtype=torch.bool)
        letters = self.letter_token_ids.cpu()
        for b, e in enumerate(encoded):
            input_ids[b, : lens[b]] = torch.tensor(e.input_ids)
            attention_mask[b, : lens[b]] = 1
            k = len(e.order)
            option_token_ids[b, :k] = letters[:k]
            option_token_ids[b, k:] = letters[0]
            option_mask[b, :k] = True
        return {"input_ids": input_ids, "attention_mask": attention_mask, "option_token_ids": option_token_ids,
                "option_mask": option_mask}

    # ------------------------------------------------------------------------------------------------------------- inference
    def batches(self, items: list, tokens_per_batch: Optional[int] = None, batching: Optional[str] = None) -> list:
        """'exact': prompts sorted by token length (a stable sort, so equal-length prompts keep their request order on every
        machine); prompts of equal length run together (unpadded) up to `tokens_per_batch` tokens, every other prompt alone. 'pad': consecutive prompts whose padded size (longest x count) stays within
        `tokens_per_batch` (a single prompt longer than that runs alone)."""
        tpb = int(tokens_per_batch or self.config.tokens_per_batch)
        mode = batching or self.config.batching
        lens = [len(e.input_ids) for e in items]
        out, i = [], 0
        if mode == "exact":
            idx = [int(x) for x in np.argsort(lens, kind="stable")]   # stable: equal-length prompts keep request order on every machine
            while i < len(idx):
                L, j = lens[idx[i]], i + 1
                while j < len(idx) and lens[idx[j]] == L and (j - i + 1) * L <= tpb:
                    j += 1
                out.append([items[t] for t in idx[i:j]])
                i = j
            return out
        if mode != "pad":
            raise ValueError(f"batching {mode!r}")
        while i < len(items):
            j, mx = i, 0
            while j < len(items) and max(mx, lens[j]) * (j - i + 1) <= max(tpb, lens[i]):
                mx = max(mx, lens[j])
                j += 1
            out.append(items[i:j])
            i = j
        return out

    @torch.no_grad()
    def probs_for(self, group: list) -> list:
        """One forward pass over `group` (EncodedQuestion objects, run together as given) -> the head's probabilities per prompt,
        float64, mapped back to each question's option order."""
        p = self.forward(**self.collate(group)).probs.float().cpu().numpy()
        out = []
        for b, e in enumerate(group):
            k = len(e.order)
            q = np.zeros(k)
            q[np.asarray(e.order)] = p[b, :k].astype(np.float64)
            out.append(q)
        return out

    @torch.no_grad()
    def _run(self, items, tokens_per_batch, batching, details):
        out = {}
        for group in self.batches(items, tokens_per_batch, batching):
            for e, q in zip(group, self.probs_for(group)):
                out[e.qid] = q.tolist()
            if details is not None:
                details["batches"].append([e.qid for e in group])
                for e in group:
                    details["order"][e.qid] = list(e.order)
        return out

    @torch.no_grad()
    def predict_proba(self, request_or_state, questions=None, *, tokens_per_batch: Optional[int] = None,
                      batching: Optional[str] = None, max_len: Optional[int] = None, return_details: bool = False):
        """{question key: {"type": "choice", "choice": key, "probabilities": {key: p}} | {"type": "noul", "noul": p(yes)}}.
        With return_details=True also returns {"order": {prompt id: letter order}, "batches": [[prompt ids], ...]}."""
        state, questions = _split_request(request_or_state, questions)
        max_len = int(max_len or self.config.max_len)
        details = {"order": {}, "batches": []} if return_details else None
        rec, items = self._prepare(state, questions, max_len)
        probs = self._run(items, tokens_per_batch, batching, details)
        finals = []
        for qkey in (rec["meta"].get("k52") or {}):
            keys = core.finalists(rec, qkey, probs)
            fq = core.qid_of(rec["id"], qkey, "final")
            q2, lab, kind = core.final_question(rec, qkey, keys)
            r2 = {**rec, "questions": {fq: q2}, "labels": {fq: lab}}
            u = self._unit(r2, fq)
            order, text = self._prompt(u, fq, max_len)
            finals.append(self._encoded(fq, u, order, text))
        if finals:
            probs.update(self._run(finals, tokens_per_batch, batching, details))
        st = collections.Counter()
        answers = {}
        for qkey, q in questions.items():
            a = core.answer_for(rec, qkey, q, probs, st)
            if a is None:
                raise RuntimeError(f"no prediction for question {qkey!r}")
            answers[qkey] = a
        return (answers, details) if return_details else answers
