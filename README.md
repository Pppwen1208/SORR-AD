# SORR-AD Public Release

This repository is the official public source release for **SORR-AD** research.

It provides a lightweight and extensible codebase for dataset preparation,
feature extraction, feature caching, anomaly-metric computation, and public
research examples. The repository is designed to support inspection,
extension, and further research based on the SORR-AD project structure.

## Features

* MVTec AD and VisA dataset readers
* WideResNet-based feature extraction utilities
* Local feature-cache support
* Common image-level and pixel-level anomaly-detection metrics
* Dataset download helpers
* Public examples for dataset inspection, feature extraction, and metric usage
* Minimal Python package metadata for editable installation
* Clean repository structure for downstream research extension

## Installation

Create a Python environment and install the package:

```bash
pip install -r requirements.txt
pip install -e .
```

Install a platform-specific PyTorch build separately when CUDA support is
required.

## Data Preparation

Download MVTec AD:

```bash
python scripts/download_mvtec.py --root data/mvtec --categories all --source hf-full
```

Download VisA:

```bash
python scripts/download_visa.py --root data/visa --raw-root data/visa_raw --categories all
```

Downloaded datasets and generated artifacts are stored locally and excluded by
`.gitignore`.

## Public Examples

Inspect a downloaded dataset:

```bash
python scripts/inspect_dataset.py --dataset mvtec --root data/mvtec --categories bottle
```

Extract support-image features. Backbone weights are obtained by PyTorch at
runtime and are not bundled with the repository:

```bash
python scripts/extract_support_features.py --dataset mvtec --root data/mvtec --category bottle --shots 1
```

Run a synthetic metric calculation without downloading a dataset:

```bash
python scripts/demo_metrics.py
```

## Repository Layout

```text
scripts/          Dataset helpers and public examples
src/sorrad/       Public reusable infrastructure
```

## Research Usage

This release focuses on reusable infrastructure for few-shot industrial anomaly
detection research. The provided components can be used to inspect datasets,
extract local visual features, cache intermediate representations, and evaluate
anomaly-detection outputs with standard image-level and pixel-level metrics.

The codebase is intentionally lightweight, making it suitable as a clean
foundation for independent research, method extension, and reproducibility
studies involving MVTec AD, VisA, and related industrial anomaly-detection
benchmarks.

## License

The files in this repository are released under the MIT License. See `LICENSE`.
