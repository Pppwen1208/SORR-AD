from __future__ import annotations

import numpy as np

from sorrad.metrics import pixel_metrics, safe_aupr, safe_auroc


def main() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    image_scores = np.array([0.05, 0.20, 0.72, 0.91], dtype=np.float32)

    normal_mask = np.zeros((8, 8), dtype=np.uint8)
    anomaly_mask = normal_mask.copy()
    anomaly_mask[2:6, 3:7] = 1
    normal_map = np.linspace(0.0, 0.2, 64, dtype=np.float32).reshape(8, 8)
    anomaly_map = normal_map.copy()
    anomaly_map[2:6, 3:7] += 0.8

    metrics = pixel_metrics([normal_mask, anomaly_mask], [normal_map, anomaly_map], device="cpu")
    print(f"image_auroc={safe_auroc(labels, image_scores):.4f}")
    print(f"image_aupr={safe_aupr(labels, image_scores):.4f}")
    print(f"pixel_auroc={metrics['pixel_auroc']:.4f}")
    print(f"pro={metrics['pro']:.4f}")


if __name__ == "__main__":
    main()
