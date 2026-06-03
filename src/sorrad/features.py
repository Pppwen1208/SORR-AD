from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


@dataclass
class FeatureCache:
    paths: List[str]
    features: List[torch.Tensor]
    grid_shape: tuple[int, int]
    image_size: int
    proj_dim: int


class ImageJobDataset(Dataset):
    def __init__(self, jobs: Sequence[tuple[str | Path, str]], transform) -> None:
        self.jobs = [(str(path), mode) for path, mode in jobs]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, index: int) -> torch.Tensor:
        path, mode = self.jobs[index]
        image = Image.open(path).convert("RGB")
        if mode == "hflip":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif mode == "vflip":
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif mode == "rot90":
            image = image.transpose(Image.ROTATE_90)
        elif mode == "rot270":
            image = image.transpose(Image.ROTATE_270)
        elif mode != "orig":
            raise ValueError(f"Unknown augmentation mode: {mode}")
        return self.transform(image)


class WideResNetFeatures:
    def __init__(
        self,
        image_size: int = 448,
        proj_dim: int = 384,
        batch_size: int = 96,
        num_workers: int = 4,
        amp: bool = False,
        device: str = "cuda",
        seed: int = 123,
    ) -> None:
        self.image_size = image_size
        self.proj_dim = proj_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.amp = amp
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.outputs: Dict[str, torch.Tensor] = {}
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.model = models.wide_resnet50_2(pretrained=True).to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        self.model.layer2.register_forward_hook(self._hook("layer2"))
        self.model.layer3.register_forward_hook(self._hook("layer3"))
        self.projector: torch.Tensor | None = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def _hook(self, name: str):
        def fn(_module, _inputs, output):
            self.outputs[name] = output

        return fn

    def _ensure_projector(self, channels: int) -> torch.Tensor:
        if self.projector is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
            proj = torch.randn(channels, self.proj_dim, generator=generator)
            proj = proj / (self.proj_dim ** 0.5)
            self.projector = proj.to(self.device)
        return self.projector

    @torch.no_grad()
    def _embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        self.outputs.clear()
        batch = batch.to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            batch = batch.contiguous(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(enabled=self.amp and self.device.type == "cuda"):
            _ = self.model(batch)
        f2 = self.outputs["layer2"]
        f3 = F.interpolate(self.outputs["layer3"], size=f2.shape[-2:], mode="bilinear", align_corners=False)
        feat = torch.cat([f2, f3], dim=1).float()
        feat = feat.permute(0, 2, 3, 1).contiguous()
        b, h, w, c = feat.shape
        feat = feat.view(b * h * w, c)
        proj = self._ensure_projector(c)
        feat = feat @ proj
        feat = F.normalize(feat, dim=1)
        return feat.view(b, h, w, self.proj_dim).cpu()

    def _loader(self, jobs: Sequence[tuple[str | Path, str]]) -> DataLoader:
        dataset = ImageJobDataset(jobs, self.transform)
        workers = self.num_workers if len(jobs) >= max(self.batch_size, 16) else 0
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": workers,
            "pin_memory": self.device.type == "cuda",
        }
        if workers > 0:
            kwargs["prefetch_factor"] = 2
        return DataLoader(dataset, **kwargs)

    def _extract_jobs(self, jobs: Sequence[tuple[str | Path, str]], desc: str) -> List[torch.Tensor]:
        if not jobs:
            return []
        features: List[torch.Tensor] = []
        loader = self._loader(jobs)
        for batch in tqdm(loader, desc=desc):
            emb = self._embed_batch(batch)
            features.extend([emb[i].contiguous() for i in range(emb.shape[0])])
        return features

    def extract(self, paths: Sequence[str | Path], desc: str = "extract") -> List[torch.Tensor]:
        return self._extract_jobs([(p, "orig") for p in paths], desc=desc)

    def extract_augmented(
        self,
        paths: Sequence[str | Path],
        modes: Sequence[str],
        desc: str = "augment",
    ) -> List[torch.Tensor]:
        jobs = [(p, m) for m in modes for p in paths if m != "orig"]
        return self._extract_jobs(jobs, desc=desc)


def load_feature_cache(cache_path: Path) -> FeatureCache | None:
    if not cache_path.exists():
        return None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        payload = torch.load(cache_path, map_location="cpu")
    except RuntimeError as exc:
        message = str(exc).lower()
        if "not enough memory" not in message and "defaultcpuallocator" not in message:
            raise
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        payload = torch.load(cache_path, map_location="cpu")
    return FeatureCache(
        paths=payload["paths"],
        features=payload["features"],
        grid_shape=tuple(payload["grid_shape"]),
        image_size=int(payload["image_size"]),
        proj_dim=int(payload["proj_dim"]),
    )


def save_feature_cache(cache_path: Path, cache: FeatureCache) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "paths": cache.paths,
            "features": cache.features,
            "grid_shape": cache.grid_shape,
            "image_size": cache.image_size,
            "proj_dim": cache.proj_dim,
        },
        cache_path,
    )
