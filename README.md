# FITTER: Vocabulary-Agnostic Cross-Domain Inference on Temporal Knowledge Graphs

This repository contains the reference implementation of **FITTER**, a fully-inductive
structural model for temporal knowledge graph (TKG) link prediction that supports
**cross-domain transfer**: a model trained on one TKG can be applied directly to a
different target TKG whose entities, relation names, and timestamps are entirely unseen.

FITTER represents relations by their structural interaction patterns and time through
sinusoidal encodings of *relative* (rather than absolute) ordering, so the learned
representations are vocabulary-agnostic and transfer across graphs with different domains,
granularities, and time spans.

FITTER builds on [ULTRA](https://github.com/DeepGraphLearning/ULTRA) (Galkin et al.,
*Towards Foundation Models for Knowledge Graph Reasoning*, ICLR 2024): it reuses ULTRA's
vocabulary-agnostic relation-interaction backbone and NBFNet-style conditional message
passing, and adds temporal encodings, temporal-aware entity propagation, and local/global
context fusion for the temporal setting.

---

## Repository structure

```
.
├── fitter/                 # core model package
│   ├── models.py           # FITTER model (relation + entity modules)
│   ├── base_nbfnet.py      # NBFNet-style conditional message passing backbone
│   ├── layers.py           # temporal message/aggregation layers
│   ├── datasets.py         # TKG dataset loaders (transductive + inductive)
│   ├── tasks.py            # negative sampling, ranking, metrics
│   ├── util.py             # config parsing, logging, distributed helpers
│   └── rspmm/              # JIT-compiled CUDA/C++ sparse message-passing op
├── script/run.py           # single entry point for training and inference
├── config/transductive/    # inference configs, one per dataset
├── ckpts/                  # pre-trained checkpoints (ICEWS14, ICEWS05-15, GDELT)
└── kg-datasets/            # benchmark datasets (raw splits)
```

---

## Installation

Tested with Python 3.9 and CUDA 11.8.

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-geometric==2.4.0 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install ninja easydict pyyaml
```

(Or `pip install -r requirements.txt` for loosely-pinned versions.)

> **Note on the `rspmm` extension.** FITTER's message passing uses a custom CUDA/C++
> operator under `fitter/rspmm/` (inherited from ULTRA/NBFNet). It is **compiled
> just-in-time on first run**, which requires a working CUDA toolkit and `ninja`. The
> first invocation therefore takes a few extra minutes while the extension builds.

---

## Datasets

Six benchmarks are included under `kg-datasets/`:

| Dataset       | `--dataset` value | Setting                      | Granularity |
|---------------|-------------------|------------------------------|-------------|
| ICEWS14       | `ICEWS14`         | interpolation (transductive) | daily       |
| ICEWS05-15    | `ICEWS0515`       | interpolation (transductive) | daily       |
| GDELT         | `GDELT`           | interpolation (transductive) | 15 minutes  |
| ICEWS18       | `ICEWS18Ind`      | extrapolation                | daily       |
| YAGO          | `YAGOInd`         | extrapolation                | yearly      |
| WIKI          | `WIKIInd`         | extrapolation                | yearly      |

Each dataset provides `raw/{train,valid,test}.txt` as `(head, relation, tail, timestamp)`
quadruples. Before running, set the dataset location by editing the `root:` field of the
config file (default `~/git/kg-datasets/`) to point at this repository's `kg-datasets/`
directory, and set `output_dir:` to where logs/checkpoints should be written.

---

## Pre-trained checkpoints

Three checkpoints are provided in `ckpts/`: `ICEWS14.pth`, `ICEWS0515.pth`, and
`GDELT.pth`. Because FITTER is vocabulary-agnostic, any checkpoint can be applied to any
target dataset — training and inference graphs need not share entities, relations, or
timestamps.

---

## Running inference (fully-inductive cross-domain transfer)

`script/run.py` is the single entry point. Setting `--epochs 0` runs **inference only**
with the provided checkpoint (no gradient updates); the only inference-time
hyperparameters are the local window size `k` and the fusion weight `alpha`, set in the
config file.

The arguments are:

| Argument      | Meaning |
|---------------|---------|
| `-c <yaml>`   | path to the config file |
| `--dataset`   | target dataset class (see table above) |
| `--epochs`    | number of training epochs; `0` = inference only |
| `--bpe`       | batches per epoch; `null` uses the full dataloader |
| `--gpus`      | `null` for CPU, `[0]` for one GPU, `[0,1,...]` for multi-GPU |
| `--ckpt`      | path to the checkpoint to load |
| `-s/--seed`   | random seed (default `1024`) |

**Cross-domain transfer** = a target-dataset config paired with a checkpoint trained on a
*different* source dataset. For example, transfer a GDELT-trained model to ICEWS14:

```bash
python script/run.py -c config/transductive/inf_ICEWS14.yaml \
    --dataset ICEWS14 --epochs 0 --bpe null --gpus [0] \
    --ckpt ckpts/GDELT.pth
```

In-domain evaluation simply uses the matching checkpoint (e.g. `inf_ICEWS14.yaml` with
`ckpts/ICEWS14.pth`). To run on CPU, use `--gpus null`.

Extrapolation targets use their own configs, e.g.:

```bash
python script/run.py -c config/transductive/inf_YAGO.yaml \
    --dataset YAGOInd --epochs 0 --bpe null --gpus [0] \
    --ckpt ckpts/ICEWS14.pth
```

---

## Training

To train from scratch (or fine-tune a checkpoint), set `--epochs` greater than `0`:

```bash
python script/run.py -c config/transductive/inf_ICEWS14.yaml \
    --dataset ICEWS14 --epochs 10 --bpe null --gpus [0] --ckpt null
```

The best checkpoint (selected on the validation split) is saved to `output_dir` as
`model_epoch_<n>.pth`. For multi-GPU training, pass e.g. `--gpus [0,1,2,3]`; the script
uses PyTorch `DistributedDataParallel`.

---

## Hardware

Experiments in the paper were run on NVIDIA A100 GPUs. The NBFNet-style message passing
loads the full inference graph into GPU memory, so memory and runtime scale with graph
size and density; dense, fine-grained graphs such as GDELT are the most demanding.

---

## Acknowledgements

This codebase is built on top of [ULTRA](https://github.com/DeepGraphLearning/ULTRA), and
reuses its relation-interaction backbone, NBFNet-style message passing, and the `rspmm`
CUDA/C++ operator. We thank the ULTRA authors for releasing their code.
