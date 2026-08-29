# PixCell

**PixCell** is a medical deep learning framework developed for the **IAAA 2026 Brain CT Triage Challenge**.

The framework is designed to process non-contrast brain CT scans, reconstruct clinical targets from DICOM data and annotations, prepare training datasets, train deep learning models, and evaluate their predictions.

## Overview

PixCell supports a modular pipeline for several brain CT tasks, including:

* **Intracranial Hemorrhage (ICH) segmentation**

  * IVH
  * IPH
  * SDH
  * EDH
  * SAH
* **Midline Shift (MLS) regression**
* **Skull fracture detection/classification**
* **Clinical triage prediction**

The final triage category is derived from intermediate clinical predictions using the competition's predefined triage rule.

PixCell uses a **configuration-driven and registry-based architecture**. Dataset paths, preprocessing options, targets, training parameters, optimizers, schedulers, models, losses, metrics, transforms, and data readers can be configured through YAML files.

This allows experiments to be reproduced and modified without changing the core source code.

## Current Status

> **Development status: In Progress**

The `models/` module currently provides the model interface and integration points, while concrete neural network implementations and their trained weights are intentionally **not provided**.

The repository is currently intended primarily as the **framework and pipeline implementation**.

## Data

The project expects the competition data to be available locally in the following general structure:

```text
data/
├── training/
├── annotations/
└── training_df.pkl
```

Raw competition data is **not included** in this repository.
