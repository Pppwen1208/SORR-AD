from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image


MVTec_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class TestSample:
    image_path: Path
    label: int
    defect_type: str
    mask_path: Path | None


def parse_categories(value: str | Sequence[str], dataset: str = "mvtec") -> List[str]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    else:
        parts = list(value)
    known = MVTec_CATEGORIES if dataset == "mvtec" else VISA_CATEGORIES
    if not parts or parts == ["all"] or "all" in parts:
        return list(known)
    unknown = sorted(set(parts) - set(known))
    if unknown:
        raise ValueError(f"Unknown {dataset} categories: {unknown}")
    return parts


def image_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


class MVTecCategory:
    def __init__(self, root: str | Path, category: str):
        self.root = Path(root)
        self.category = category
        self.category_root = self.root / category
        if not self.category_root.exists():
            raise FileNotFoundError(
                f"Missing category folder: {self.category_root}. "
                "Run scripts/download_mvtec.py first."
            )

    @property
    def train_good_paths(self) -> List[Path]:
        return image_files(self.category_root / "train" / "good")

    def support_paths(self, shots: int, seed: int) -> List[Path]:
        paths = self.train_good_paths
        if shots <= 0:
            return paths
        if len(paths) < shots:
            raise ValueError(
                f"{self.category} has only {len(paths)} train/good images, "
                f"cannot sample K={shots}."
            )
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(paths), size=shots, replace=False)
        return [paths[int(i)] for i in idx]

    def test_samples(self) -> List[TestSample]:
        test_root = self.category_root / "test"
        samples: List[TestSample] = []
        for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            for image_path in image_files(defect_dir):
                mask_path = None
                if label == 1:
                    mask_path = (
                        self.category_root
                        / "ground_truth"
                        / defect_type
                        / f"{image_path.stem}_mask.png"
                    )
                    if not mask_path.exists():
                        mask_path = None
                samples.append(TestSample(image_path, label, defect_type, mask_path))
        return samples


class VisaCategory:
    cache_namespace = "visa"

    def __init__(self, root: str | Path, category: str):
        self.root = Path(root)
        self.category = category
        self.category_root = self.root / category
        if not self.category_root.exists():
            raise FileNotFoundError(
                f"Missing category folder: {self.category_root}. "
                "Run scripts/download_visa.py first."
            )

    @property
    def train_good_paths(self) -> List[Path]:
        return image_files(self.category_root / "train" / "good")

    def support_paths(self, shots: int, seed: int) -> List[Path]:
        paths = self.train_good_paths
        if shots <= 0:
            return paths
        if len(paths) < shots:
            raise ValueError(
                f"{self.category} has only {len(paths)} train/good images, "
                f"cannot sample K={shots}."
            )
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(paths), size=shots, replace=False)
        return [paths[int(i)] for i in idx]

    def test_samples(self) -> List[TestSample]:
        test_root = self.category_root / "test"
        samples: List[TestSample] = []
        for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            for image_path in image_files(defect_dir):
                mask_path = None
                if label == 1:
                    gt_dir = self.category_root / "ground_truth" / defect_type
                    candidates = [
                        gt_dir / image_path.name,
                        gt_dir / f"{image_path.stem}.png",
                        gt_dir / f"{image_path.stem}_mask.png",
                    ]
                    mask_path = next((p for p in candidates if p.exists()), None)
                samples.append(TestSample(image_path, label, defect_type, mask_path))
        return samples


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(sample: TestSample, size: int) -> np.ndarray:
    if sample.label == 0 or sample.mask_path is None:
        return np.zeros((size, size), dtype=np.uint8)
    mask = Image.open(sample.mask_path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def relative_key(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
