# Ship Re-Identification in the Wild: ShipReID-2400 and D2InterNet++

This repository accompanies the manuscript **“Ship Re-Identification in the Wild: A Large-Scale Dataset and a Strong Baseline.”** It contains the ShipReID-2400 data interface and an implementation snapshot of the Feature-Decoupled Dual-Branch Interaction Network (**D2InterNet++**).

## Overview

Ship re-identification associates the same vessel across disjoint cameras and time periods. Compared with pedestrian and vehicle ReID, real-world ship ReID is strongly affected by large scale changes, partial observations, cargo and water-level variations, and severe spatial misalignment.

ShipReID-2400 was collected from a real-world waterway monitoring system over 53 months. It contains **17,241 unique images of 2,400 ship identities** captured by eight cameras. Query folders contain evaluation copies and are not counted as additional unique images.

D2InterNet++ contains:

- a global visual branch for holistic ship semantics;
- a local branch with dual-axis part tokens for horizontal and vertical ship regions;
- a Part-Aware Feature Perception Module (PFPM);
- a Feature Interaction Module (FIM);
- a Collaborative Interaction Learning Module (CILM);
- a part-feature alignment objective for partially overlapping views;
- an auxiliary frozen CLIP text encoder and bidirectional image-text contrastive loss during training.

The text branch is intended for training only and is discarded during visual retrieval inference.

## Repository Status

This directory is a research snapshot, not yet a complete paper-reproduction package.

| Capability | Current status |
|---|---|
| Global/local visual branches, PFPM, FIM and CILM | Implemented |
| Frozen CLIP text encoder and image-text contrastive loss | Implemented |
| JSONL annotation loading and prompt serialization utility | Implemented |
| Generated per-image captions used by the data loader | **Not enabled in the current snapshot**; the loader currently returns one fixed caption |
| Separate validation/test gallery construction | **Not implemented**; both loaders currently use the shared `bounding_box_test` gallery |
| Standalone `test.py` with the text-aware batch format | Requires an inference-loop update |
| CUB-200-2011, Stanford Cars, NABirds and VesselReID reproduction configs | Not included |

Consequently, the manuscript's generated-caption and cross-domain tables cannot yet be reproduced directly from this checkout.

## Installation

The original experiments used Python 3.8, PyTorch 1.9.1 and CUDA 11.1.

```bash
conda create -n shipreid python=3.8
conda activate shipreid
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

The code currently assumes an NVIDIA CUDA device. The CLIP ViT-B/16 weights are downloaded automatically on first use and verified by SHA-256.

## Dataset

The dataset download currently referenced by the project is available from [Google Drive](https://drive.google.com/file/d/1OTGK09zo5RKuoGSL-s58e7UzJzIWy7nP/view?usp=sharing).

The current data factory key is `ShipReID2400`, and the loader also uses this value as the directory name. Place or link the extracted dataset at:

```text
<DATASETS.ROOT_DIR>/ShipReID2400/
├── bounding_box_train/       # 12,988 unique training images
├── bounding_box_test/        # 4,253 unique evaluation images
├── val_query/                # 691 query copies
├── test_query/               # 706 query copies
├── bounding_box_train.jsonl
├── val_query.jsonl
├── val_gallery.jsonl
├── test_query.jsonl
└── test_gallery.jsonl
```

The 17,241 unique images are counted as:

```text
12,988 training images + 2,149 validation images + 2,104 test images = 17,241
```

The manuscript also reports 691 validation-query and 706 test-query entries. If those entries are added again to the figures above, the directory-entry count becomes `18,638` (`17,241 + 691 + 706`), but the manuscript never states that 18,638 is the number of unique images. This repository interprets the query directories as evaluation copies/subsets that do not increase the unique-image total.

That interpretation exposes an ambiguity in the manuscript: Section 3.3 says that the query images are selected first and that "all remaining images" form the gallery, which would make query and gallery disjoint and would instead imply 18,638 unique images. The paper should explicitly state whether query images are duplicated in/included within the reported gallery counts, and revise either the wording or the split counts accordingly.

### Image naming convention

The ship loader expects JPEG filenames containing an identity and camera identifier:

```text
<ship_id>_c<camera_id>_...jpg
```

Camera IDs must currently be between 1 and 8.

### Text annotation format

Each JSONL line should contain an `image_path` and an `annotation` object. The current mapping performs an exact string match between `image_path` and the image path produced by the dataset loader, so absolute/relative path conventions must be consistent.

```json
{
  "image_path": "/absolute/path/to/ShipReID2400/bounding_box_train/0000_c002_01_T_20220105_11_04_07_484171.jpg",
  "annotation": {
    "hull_color": "rusty blue",
    "fenders": {
      "density": "dense",
      "arrangement": "single row"
    },
    "cabin_details": {
      "structure_description": "a single-level blue cabin with a white railing",
      "windows_doors": "multiple rectangular windows"
    },
    "deck_info": {
      "deck_surface": "open cargo deck",
      "discriminative_objects": [
        {
          "object": "air-conditioning unit",
          "description": "a white outdoor unit mounted on the cabin side",
          "location": "cabin side"
        }
      ]
    }
  }
}
```

`utils/ship_text.py` can convert this structure into a global caption, visual-cue description and attribute tags. In the current snapshot, `datasets/bases.py` does not yet call this serializer and instead supplies a fixed caption. This must be corrected before reproducing the generated-caption experiment.

## Code Structure

```text
├── config/                    # Default configuration definitions
├── configs/ship/              # ShipReID experiment configuration
├── datasets/                  # Dataset adapters, transforms and samplers
├── exp/                       # Legacy experiment shell scripts
├── fig/                       # Repository figures
├── loss/                      # Classification, metric and alignment losses
├── model/                     # D2InterNet++ and CLIP implementation
├── processor/                 # Training and inference loops
├── solver/                    # Optimizers and learning-rate schedulers
├── utils/ship_text.py         # JSONL loading and prompt serialization
├── train.py                   # Training entry point
└── test.py                    # Evaluation entry point
```

`model/make_model copy.py` is a legacy development copy and is not imported by the training entry point.

## Training

Run commands from the repository root. Replace the dataset and output paths with absolute paths on your machine.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config_file configs/ship/vit_base.yml \
  DATASETS.ROOT_DIR /absolute/path/to/datasets \
  DATASETS.NAMES ShipReID2400 \
  OUTPUT_DIR outputs/d2internetpp \
  SOLVER.IMS_PER_BATCH 48 \
  SOLVER.MAX_EPOCHS 300 \
  MODEL.PART_H 16 \
  MODEL.PART_W 4 \
  MODEL.ID_LOSS_WEIGHT 0.5 \
  MODEL.TRIPLET_LOSS_WEIGHT 5.0 \
  MODEL.PART_ID_LOSS_WEIGHT 0.5 \
  MODEL.PART_TRIPLET_LOSS_WEIGHT 5.0 \
  MODEL.TOKEN_CONTRAST_LOSS_WEIGHT 10.0 \
  MODEL.PART_RATIO 0.6 \
  MODEL.TOP_K 1.0 \
  MODEL.TOKEN_CONTRAST_TYPE triplet \
  MODEL.NO_TOKEN_MARGIN False \
  MODEL.text 0.4 \
  SOLVER.TOKEN_MARGIN 0.3
```

The current implementation uses `256 x 256` inputs because the dual-axis part coordinates are hard-coded for a `16 x 16` patch grid. The manuscript reports `256 x 128` inputs and 60 training epochs; those settings are not currently consistent with this code snapshot and should not be claimed as directly reproducible until the implementation or manuscript is corrected.

During training, validation and test metrics are logged every `SOLVER.EVAL_PERIOD` epochs and checkpoints are saved every `SOLVER.CHECKPOINT_PERIOD` epochs.

## Evaluation

The intended evaluation command is:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/ship/vit_base.yml \
  DATASETS.ROOT_DIR /absolute/path/to/datasets \
  DATASETS.NAMES ShipReID2400 \
  OUTPUT_DIR outputs/d2internetpp_eval \
  MODEL.PART_H 16 \
  MODEL.PART_W 4 \
  TEST.WEIGHT /absolute/path/to/checkpoint.pth
```

Before using this command, update `processor.do_inference` to unpack the text-aware validation batches returned by `val_collate_fn`. The periodic evaluation path inside `do_train` already accepts this batch format.

Evaluation reports mean Average Precision (mAP) and CMC Rank-1/5/10 after removing gallery samples with the same identity and camera as each query.

## Results Reported in the Manuscript

These values are copied from the manuscript and have **not** been independently reproduced from this repository snapshot.

| Configuration | ShipReID-2400 mAP | Rank-1 |
|---|---:|---:|
| D2InterNet++ visual model | 44.0 | 51.0 |
| D2InterNet++ with generated-caption supervision | 44.9 | 52.4 |

## Reproducibility Checklist

Before treating the repository as the artifact for the manuscript, the following items should be completed:

- connect `annotation_to_texts()` to `ImageDataset.__getitem__`;
- validate JSONL path coverage and fail loudly on missing annotations;
- construct separate validation and test galleries matching the paper protocol;
- align the input size, epoch count and loss weights with the manuscript;
- align PFAM's mathematical description with its implemented assignment algorithm;
- align the inference embedding with the feature concatenation defined in the manuscript;
- repair and test the standalone evaluation entry point;
- publish the exact checkpoints, logs, random seeds and baseline configurations used for every table;
- add the fine-grained retrieval and VesselReID configurations claimed in the manuscript;
- add a software license, dataset terms of use and a citation entry.

## License and Citation

No software license or dataset terms of use are included in this snapshot. Add them before public release. A final BibTeX entry should be added after the manuscript metadata is finalized.
