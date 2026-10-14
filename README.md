---
license: apache-2.0
base_model: google/gemma-4-31B-it
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- peft
- decision-making
- multiple-choice
---

# garum

garum is a LoRA adapter for [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) that answers multiple-choice
decision questions with a probability for each option. It includes a calibrated Bayesian last-layer head (`vbll_head.safetensors`,
`garum_head.py`): garum's probabilities are the head's output, computed from the adapted model's hidden state and option-letter
logits at the answer position. The letter logits on their own are not garum's answer, so load it through the code below.

The weights (`adapter_model.safetensors` and `vbll_head.safetensors`) are hosted on the Hugging Face Hub at https://huggingface.co/plantwaterer/garum; the GitHub mirror holds the code, with Git LFS pointers in place of the weight files.

## Usage

```python
import sys
from huggingface_hub import snapshot_download

repo = snapshot_download("plantwaterer/garum")
sys.path.insert(0, repo)
from modeling_garum import GarumForDecision

model = GarumForDecision.from_pretrained(repo, base="google/gemma-4-31B-it", device_map="cuda")

# a request: a state plus named questions; each answer has a probability for every option
answers = model.predict_proba(
    "The forecast gives a 70% chance of heavy rain this afternoon. The match is outdoors, on grass.",
    {"plan": {"type": "choice", "instructions": "What should the organisers do?",
              "criteria": {"play": "Play as planned", "indoors": "Move it indoors", "postpone": "Postpone it"}},
     "rain": {"type": "noul", "instructions": "Is heavy rain likely?"}})
print(answers["plan"]["probabilities"], answers["rain"]["noul"])
```

Yes / no questions use the options `Yes` and `No` in that order. When a request gives a yes / no question's definitions (its
`criteria`, keyed `yes` / `no` or `true` / `false`), they are shown with the options, as `Yes: <definition>` / `No: <definition>`
(`noul_defs=False` turns this off).

### PyTorch model class

`modeling_garum.py` (with `garum_core.py` and `garum_head.py`) defines `GarumForDecision`, a `torch.nn.Module` that wraps the base
model, the adapter and the head. It loads the adapter with `autocast_adapter_dtype=False`, so the LoRA update stays in bf16.
`predict_proba` is the computation the Decision Index engine below runs. The prompts can also be built, batched and run directly:

```python
import torch

enc = [model.encode("The match is outdoors.", "What should the organisers do?",
                    {"play": "Play", "postpone": "Postpone"}, key="plan"),
       model.encode("The invoice total is $1,240; the purchase order says $1,420.", "Is the invoice consistent?",
                    kind="noul", key="check")]
batch = model.collate(enc)                        # right-padded input_ids / attention_mask, option_token_ids / option_mask
with torch.no_grad():
    out = model(**batch)                          # probs, log_probs: [batch, options] in letter order (the head's output)
for e, p in zip(enc, out.probs):
    print({e.option_keys[o]: float(p[j]) for j, o in enumerate(e.order)})   # letter j shows option e.order[j]
```

`forward(..., labels=...)` also returns the negative log-likelihood of the gold option (its letter index per row, -100 to skip).
Batch composition (which prompts run together, and in which order) can move bf16 outputs slightly, by about a bf16 step in a
letter logit. `predict_proba` runs a request's prompts of equal token length together, unpadded, in a stable order, and every other
prompt alone, so the same request gives the same answer on every machine with the same GPU and software. Use eager mode (no
`torch.compile`) to keep that arithmetic.

### Decision Index engine

`garum_engine.py` runs garum through the [Decision Index kit](https://github.com/apolinario/decision-index)'s runner:

```bash
python -m decision_index run --engine garum_engine:GarumEngine \
  --option base=google/gemma-4-31B-it --option adapter=plantwaterer/garum --compact --out runs/garum
```

The inference procedure is as follows:
- **Prompts:** one prompt per question. A choice question's options are shown in one fixed non-identity order derived from a hash of the request.
- **Yes / no definitions:** when the request provides them, they are shown with the options (`--option noul_defs=false` turns this off).
- **Read-out:** one forward pass per prompt. The head turns the hidden state and the option letters' next-token logits into the probabilities.
- **More than 52 options:** handled in two rounds.
- **No truncation:** a request that does not fit 32,768 tokens is reported as unsupported.
- **Batching:** the prompts of one request that have the same token length run together, unpadded, up to 24,576 tokens, in a stable order; every other prompt runs alone (`--option batching=pad` uses right-padded batches instead). Batch composition can move bf16 outputs slightly.

Dependencies are torch, transformers, peft, safetensors and numpy. The code was tested with torch 2.13, transformers 5.12.1 and
peft 0.21.2. Attention is the transformers default (SDPA).

## Evaluation

**Decision Index 0.3, public panel (37 benchmarks).** Measured by us on the kit's public suite (edition 0.3, kit commit 9eb2dbe2,
scored with the kit's own scorer), with our own scoring pipeline rather than the kit's runner and this engine. That pipeline used
the same weights, head and prompt format. It differs from the engine in a few ways:
- a choice question's option order was drawn from a seeded generator, not from a hash of the request;
- for questions with more than 52 options, the first-round split of the options was seeded by the kit's request id;
- yes / no definitions were not shown;
- prompts were batched across each benchmark, not within a request;
- 319 prompts longer than 32,768 tokens (MMLU-Pro 304, iSarcasmEval 15) were scored with the state cut to fit. The engine reports such
  requests as unsupported.

The 19 test items listed below are counted as wrong. This is our own measurement, not a leaderboard entry.

| | public index | knowledge | language | retrieval | tools | arts |
|---|---|---|---|---|---|---|
| garum | 58.29 | 45.01 | 61.55 | 62.88 | 75.30 | 43.92 |
| [garas](https://huggingface.co/plantwaterer/garas), same pipeline | 56.68 | 43.67 | 58.62 | 61.48 | 74.26 | 43.58 |

garas's own card reports 56.57 from a full run of its engine through the kit's runner; the procedures differ as listed above.

### Overlap with evaluation data

Training used the train splits of BANKING77, CLINC150 and GSM8K. It also used WinoGrande train, Amazon ESCI train queries, Lichess
puzzles (a different dataset from ChessBench's), and ANLI training-split pairs that are part of DocNLI. No test split of any
benchmark was used.

An audit of the training text against the Decision Index public suite found 19 test items that are exact or near-exact
duplicates of utterances in the BANKING77 and CLINC150 train splits. They are counted as wrong in the numbers above, as the board
does for train-split duplicates of test items (see the Jet v6.2 and lev entries under Contamination in the
[Decision Index methodology](https://multimodalart-jev-decision-index.static.hf.space/methodology.html)):
`4:BANKING77:BANKING77:test:` 554, 677, 727, 742, 747, 751, 754, 1432, 1735, 1754, 1993, 2081, 2211, 2440, 2644, 2651, 2987,
3070; `5:CLINC150+OOS:CLINC150+OOS:test:1591`.
The data garum adds to garas's was screened against the public suite (exact, near-exact and 8-word-shingle matches), and every
matching item was removed before training. No ANLI test pair occurs in it.

## Licence

The adapter and the head are released under Apache-2.0, and the base model, google/gemma-4-31B-it, is Apache-2.0. The training data
come from the sources below, under the licences and terms listed. Some of these sources carry use conditions that apply to the data
itself; they are repeated here: CourtListener (no Fair Credit Reporting Act uses), US Census PUMS (never use the data to identify any
person or household), NVD / US Department of Labor (not endorsed or certified), California DMHC (modified; not official government
data), MITRE CVE / CWE (copyright notices below) and ANLI (CC BY-NC 4.0, non-commercial; its training pairs are part of DocNLI).

## Data attribution

garum was trained on decision questions built from the public data sources below, plus synthetic questions generated by the author
(no third-party data). Thanks to everyone who builds and maintains them. Records were reformatted, and many were modified (fields
selected, text clipped, identifiers masked), so none of the material should be read as an official copy of its source.

### Open data and public records

- **GDELT Project**: event data. Data from The GDELT Project, https://www.gdeltproject.org/.
- **Wikipedia**: article text from the English Wikipedia dump of 2023-11-01, via
  [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.en`, by Wikipedia contributors.
  Licensed CC BY-SA 3.0/4.0 and GFDL. https://en.wikipedia.org/
- **Wikipedia Articles for Deletion**: deletion debates by English Wikipedia contributors (CC BY-SA 3.0/4.0, GFDL), via the AfD
  corpus of Mayfield & Black (2019), https://github.com/emayfield/AFD_Decision_Corpus (the repository's code is GPL-3.0; the
  corpus files are distributed separately).
- **Wikidata**: structured data (CC0 1.0), and Requests-for-deletion discussions by the Wikidata contributors (CC BY-SA 4.0).
  https://www.wikidata.org/
- **Swiss Judgment Prediction XL**: [rcds/swiss_judgment_prediction_xl](https://huggingface.co/datasets/rcds/swiss_judgment_prediction_xl)
  (Rasiah et al., 2023), CC BY-SA 4.0. Decisions of the Swiss Federal Supreme Court, anonymised by the court.
- **US federal courts**: the Federal Judicial Center Integrated Database (public domain), and CourtListener bulk data from Free Law
  Project (Public Domain Mark 1.0), https://www.courtlistener.com/. Not for uses covered by the US Fair Credit Reporting Act.
- **State Supreme Court Data Project**: doi:10.7910/DVN/Z80F7P, CC0 1.0.
- **OSHA** Severe Injury Reports and enforcement data, US Department of Labor (public domain), https://www.osha.gov/severe-injury-reports
  and https://data.dol.gov/. Not endorsed or certified by the US Department of Labor.
- **NTSB** aviation accident database (public domain), https://data.ntsb.gov/avdata
- **NHTSA** Office of Defects Investigation investigations and complaints (public domain), https://static.nhtsa.gov/odi/ffdd/
- **openFDA** enforcement reports and device recalls, CC0 1.0, https://open.fda.gov/
- **ClinicalTrials.gov** (US National Library of Medicine), via the ClinicalTrials.gov API v2; data processed by ClinicalTrials.gov on
  2026-10-01. Modified: selected registration fields only, free text clipped, trial ids masked, study-status sentences removed. Some
  ClinicalTrials.gov data may be subject to third-party copyright outside the United States.
- **ClinVar** (NCBI): Landrum et al., Nucleic Acids Research 2018, PMID 29165669.
- **Orphanet** inheritance data, CC BY 4.0, https://www.orpha.net/
- **CIViC** (civicdb.org), CC0 1.0: Griffith et al., Nature Genetics 2017, PMID 28138153.
- **California Department of Managed Health Care**, "DMHC IMR Data, 2001 - Current", CC BY,
  https://data.ca.gov/dataset/independent-medical-review-imr-determinations-trend. Modified; not official government data.
- **NOAA / NWS Storm Prediction Center** text products (public domain), via the Iowa Environmental Mesonet, Iowa State University.
- **LIGO / Virgo / KAGRA** public alerts and the GWOSC event catalogue, CC BY 4.0, by the LIGO Scientific, Virgo and KAGRA
  Collaborations. This model uses data obtained from the Gravitational Wave Open Science Center (gwosc.org), a service of the LIGO
  Scientific Collaboration, the Virgo Collaboration and KAGRA.
- **USGS** PAGER and ComCat (public domain). Population exposure is from ORNL LandScan (CC BY 4.0).
- **National Vulnerability Database** (NIST, public domain) and **CVE** records (CVE Terms of Use; Copyright The MITRE Corporation).
  **CWE**: Copyright 2006-2026, The MITRE Corporation; CWE is a trademark of The MITRE Corporation. This product uses the NVD API
  but is not endorsed or certified by the NVD.
- **American Community Survey PUMS**, US Census Bureau (CC0 1.0), with task definitions from
  [folktables](https://github.com/socialfoundations/folktables) (Ding et al., 2021; MIT). Never to be used to identify any person or
  household.
- **OpenReview** reviews and scores, CC BY 4.0 (titles and abstracts: CC0 1.0), https://openreview.net/
- **Lichess** open database, CC0 1.0, https://database.lichess.org/
- **Amazon ESCI** shopping queries, [amazon-science/esci-data](https://github.com/amazon-science/esci-data) (Reddy et al., 2022),
  Apache-2.0. Copyright Amazon.com, Inc. or its affiliates.
- **AIDev**: [hao-li/AIDev](https://huggingface.co/datasets/hao-li/AIDev) (Li et al., 2025), CC BY 4.0; records from permissively
  licensed repositories only.
- **SWE-rebench**: [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
  (Trofimova et al., 2025), CC BY 4.0.
- **Open-Jev**: [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev), CC0 1.0.

### Research datasets

| dataset | reference | licence |
|---|---|---|
| BANKING77 | Casanueva et al., 2020 | CC BY 4.0 |
| CLINC150 | Larson et al., 2019 | CC BY 3.0 |
| GoEmotions | Demszky et al., 2020 | Apache-2.0 |
| GSM8K | Cobbe et al., 2021 | MIT |
| HelpSteer2 | Wang et al., 2024 | CC BY 4.0 |
| Measuring Hate Speech | Kennedy et al., 2020 | CC BY 4.0 |
| IBM Debater ArgQ-Rank-30kArgs | Gretz et al., 2019 | CC BY 3.0 |
| BBQ | Parrish et al., 2022 | CC BY 4.0 |
| CaSiNo | Chawla et al., 2021 | CC BY 4.0 |
| Circa | Louis et al., 2020 | CC BY 4.0 (dataset card; its README links CC BY-SA 4.0) |
| CodeContests | Li et al., 2022 | CC BY 4.0 |
| CosmosQA | Huang et al., 2019 | CC BY 4.0 |
| CommonsenseQA | Talmor et al., 2019 | MIT |
| e-CARE | Du et al., 2022 | MIT |
| Social IQa | Sap et al., 2019 | CC BY 4.0 |
| WinoGrande | Sakaguchi et al., 2019 | CC BY (code Apache-2.0) |
| MedMCQA | Pal et al., 2022 | Apache-2.0 |
| PubMedQA | Jin et al., 2019 | MIT |
| Stanford Politeness Corpus, via ConvoKit | Danescu-Niculescu-Mizil et al., 2013 | CC BY 4.0 |
| Scruples | Lourie et al., 2021 | Apache-2.0 |
| TabFact | Chen et al., 2020 | CC BY 4.0 |
| WANLI | Liu et al., 2022 | CC BY 4.0 |
| DocNLI | Yin et al., 2021 | BSD-3-Clause |
| ANLI (training pairs, part of DocNLI) | Nie et al., 2020 | CC BY-NC 4.0 |
| Fool Me Twice | Eisenschlos et al., 2021 | Apache-2.0 |
| PUBHEALTH | Kotonya & Toni, 2020 | MIT |
| VitaminC | Schuster et al., 2021 | CC BY-SA 3.0 |
| WiCE | Kamoi et al., 2023 | CC BY-SA 4.0 |
| WikiHop (QAngaroo) | Welbl et al., 2018 | CC BY-SA 3.0 |
| HotpotQA | Yang et al., 2018 | CC BY-SA 4.0 |
| SQuAD 2.0 | Rajpurkar et al., 2018 | CC BY-SA 4.0 |
| QuAC | Choi et al., 2018 | MIT (dataset card; quac.ai states CC BY-SA 4.0) |
| NevIR | Weller et al., 2024 | MIT |
| WANDS | Chen et al., 2022 | MIT |

The twelve datasets from WANLI to WANDS (and the ANLI pairs inside DocNLI) were used through
[tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions)
(tasksource, Sileo), each under its own licence. DocNLI: Copyright (c) 2021, Salesforce.com, Inc. WANDS: Copyright (c) 2021 ecir2022.

DocNLI package (BSD-3-Clause, Yin et al., 2021): 64 of the 400 DocNLI training units in this release's data contain a complete
premise-hypothesis pair from upstream ANLI (CC BY-NC 4.0, Nie et al., 2020), and up to 82 contain ANLI text; all of it comes from
ANLI's training splits. No dataset text is redistributed.

### Community synthetic datasets

- [tasksource/procedural-typed-decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions)
  (tasksource, Sileo, 2024), Apache-2.0
- [frontier-infra/jebadiah-synth-v2](https://huggingface.co/datasets/frontier-infra/jebadiah-synth-v2), Apache-2.0
- [kaivoss/system-one-270m-data](https://huggingface.co/datasets/kaivoss/system-one-270m-data), Apache-2.0
- [vagmi/jevlite_dataset](https://huggingface.co/datasets/vagmi/jevlite_dataset), CC BY-SA 4.0
- [mghafiri/decision-model-scenarios](https://huggingface.co/datasets/mghafiri/decision-model-scenarios), MIT
