from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


HF_FULL_ARCHIVE_URL = (
    "https://huggingface.co/datasets/ProgrammerGnome/MVTecAD/resolve/main/"
    "mvtec_anomaly_detection.tar.xz?download=true"
)

CATEGORY_URLS = {
    "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz",
}


def parse_categories(value: str) -> list[str]:
    if value == "all":
        return list(CATEGORY_URLS)
    out = [v.strip() for v in value.split(",") if v.strip()]
    unknown = sorted(set(out) - set(CATEGORY_URLS))
    if unknown:
        raise SystemExit(f"Unknown categories: {unknown}")
    return out


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


def extract(archive: Path, root: Path) -> None:
    with tarfile.open(archive, mode="r:xz") as tar:
        tar.extractall(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/mvtec")
    parser.add_argument("--categories", default="all")
    parser.add_argument(
        "--source",
        choices=["official", "hf-full"],
        default="official",
        help="Use official per-category links or a full Hugging Face mirror archive.",
    )
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    archives = root / "_archives"
    if args.source == "hf-full":
        archive = archives / "mvtec_anomaly_detection.tar.xz"
        if not archive.exists():
            print("[download] MVTec full archive from Hugging Face mirror")
            download(HF_FULL_ARCHIVE_URL, archive)
        missing = [c for c in parse_categories(args.categories) if not (root / c / "test").exists()]
        if missing:
            print("[extract] MVTec full archive")
            extract(archive, root)
        if not args.keep_archives:
            archive.unlink(missing_ok=True)
        return

    for category in parse_categories(args.categories):
        done = root / category / "test"
        if done.exists():
            print(f"[skip] {category}: already extracted")
            continue
        archive = archives / f"{category}.tar.xz"
        if not archive.exists():
            print(f"[download] {category}")
            download(CATEGORY_URLS[category], archive)
        print(f"[extract] {category}")
        extract(archive, root)
        if not args.keep_archives:
            archive.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
