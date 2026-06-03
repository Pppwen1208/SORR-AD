from __future__ import annotations

import argparse
from pathlib import Path

from sorrad.datasets import MVTecCategory, VisaCategory, relative_key
from sorrad.features import FeatureCache, WideResNetFeatures, save_feature_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract public backbone features for support images.")
    parser.add_argument("--dataset", choices=["mvtec", "visa"], default="mvtec")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--category", required=True)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--proj-dim", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-out", type=Path)
    args = parser.parse_args()

    root = args.root or Path("data") / args.dataset
    dataset_cls = MVTecCategory if args.dataset == "mvtec" else VisaCategory
    dataset = dataset_cls(root, args.category)
    paths = dataset.support_paths(args.shots, args.seed)
    extractor = WideResNetFeatures(
        image_size=args.image_size,
        proj_dim=args.proj_dim,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    features = extractor.extract(paths, desc=f"extract {args.category}")
    if not features:
        raise RuntimeError("No support images were found.")

    shape = tuple(features[0].shape)
    print(f"extracted={len(features)} feature_shape={shape}")

    if args.cache_out is not None:
        save_feature_cache(
            args.cache_out,
            FeatureCache(
                paths=[relative_key(path, dataset.root) for path in paths],
                features=features,
                grid_shape=shape[:2],
                image_size=args.image_size,
                proj_dim=args.proj_dim,
            ),
        )
        print(f"cache_written={args.cache_out}")


if __name__ == "__main__":
    main()
