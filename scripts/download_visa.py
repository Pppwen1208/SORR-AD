from __future__ import annotations

import argparse
import csv
import tarfile
from pathlib import Path
from shutil import copyfile

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


VISA_ARCHIVE_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
SPLIT_CSV_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"
VISA_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 SORR-AD downloader"}
    with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with target.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract(archive: Path, raw_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:") as tar:
        tar.extractall(raw_root)


def raw_base(raw_root: Path) -> Path:
    if (raw_root / "candle").exists():
        return raw_root
    nested = raw_root / "VisA_20220922"
    if nested.exists():
        return nested
    candidates = [p for p in raw_root.iterdir() if p.is_dir() and (p / "candle").exists()]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not find extracted VisA category folders under {raw_root}")


def binarize_mask(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.open(src).convert("L")
    arr = np.asarray(mask)
    arr = np.where(arr != 0, 255, 0).astype(np.uint8)
    Image.fromarray(arr).save(dst)


def prepare(raw_root: Path, prepared_root: Path, split_csv: Path, categories: list[str]) -> None:
    base = raw_base(raw_root)
    for category in categories:
        (prepared_root / category / "train" / "good").mkdir(parents=True, exist_ok=True)
        (prepared_root / category / "test" / "good").mkdir(parents=True, exist_ok=True)
        (prepared_root / category / "test" / "bad").mkdir(parents=True, exist_ok=True)
        (prepared_root / category / "ground_truth" / "bad").mkdir(parents=True, exist_ok=True)

    with split_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["object"]
            if category not in categories:
                continue
            split = row["split"]
            label = "good" if row["label"] == "normal" else "bad"
            image_rel = Path(row["image"])
            image_src = base / image_rel
            image_dst = prepared_root / category / split / label / image_rel.name
            image_dst.parent.mkdir(parents=True, exist_ok=True)
            if not image_dst.exists():
                copyfile(image_src, image_dst)

            mask_rel = row.get("mask", "")
            if split == "test" and label == "bad" and mask_rel:
                mask_src = base / mask_rel
                mask_dst = prepared_root / category / "ground_truth" / "bad" / f"{image_rel.stem}.png"
                if not mask_dst.exists():
                    binarize_mask(mask_src, mask_dst)


def parse_categories(value: str) -> list[str]:
    if value == "all":
        return list(VISA_CATEGORIES)
    out = [v.strip() for v in value.split(",") if v.strip()]
    unknown = sorted(set(out) - set(VISA_CATEGORIES))
    if unknown:
        raise SystemExit(f"Unknown VisA categories: {unknown}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/visa")
    parser.add_argument("--raw-root", default="data/visa_raw")
    parser.add_argument("--categories", default="all")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    prepared_root = Path(args.root)
    raw_root = Path(args.raw_root)
    categories = parse_categories(args.categories)
    archive = raw_root / "_archives" / "VisA_20220922.tar"
    split_csv = raw_root / "split_csv" / "1cls.csv"

    missing = [c for c in categories if not (prepared_root / c / "test").exists()]
    if missing:
        if not archive.exists() and not raw_root.exists():
            print("[download] VisA archive from AWS Open Data")
            download(VISA_ARCHIVE_URL, archive)
        elif not archive.exists() and not any((raw_root / c).exists() for c in VISA_CATEGORIES):
            print("[download] VisA archive from AWS Open Data")
            download(VISA_ARCHIVE_URL, archive)
        if archive.exists() and not any((raw_root / c).exists() for c in VISA_CATEGORIES):
            print("[extract] VisA archive")
            extract(archive, raw_root)
        if not split_csv.exists():
            print("[download] VisA 1cls split CSV")
            download(SPLIT_CSV_URL, split_csv)
        print("[prepare] VisA 1cls pytorch layout")
        prepare(raw_root, prepared_root, split_csv, categories)
        if archive.exists() and not args.keep_archive:
            archive.unlink()
    else:
        print("[skip] requested VisA categories already prepared")


if __name__ == "__main__":
    main()
