from __future__ import annotations

import argparse
from pathlib import Path

from sorrad.datasets import MVTecCategory, VisaCategory, parse_categories


def category_type(dataset: str):
    return MVTecCategory if dataset == "mvtec" else VisaCategory


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the public dataset interface.")
    parser.add_argument("--dataset", choices=["mvtec", "visa"], default="mvtec")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--categories", default="all")
    args = parser.parse_args()

    root = args.root or Path("data") / args.dataset
    dataset_cls = category_type(args.dataset)
    for category in parse_categories(args.categories, dataset=args.dataset):
        dataset = dataset_cls(root, category)
        samples = dataset.test_samples()
        anomalies = sum(sample.label for sample in samples)
        masks = sum(sample.mask_path is not None for sample in samples)
        print(
            f"{category}: train_good={len(dataset.train_good_paths)} "
            f"test={len(samples)} anomalies={anomalies} masks={masks}"
        )


if __name__ == "__main__":
    main()
