from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _gaussian_kernel1d(sigma_px: float, radius: Optional[int] = None) -> np.ndarray:
    if sigma_px <= 0:
        return np.array([1.0])
    if radius is None:
        radius = max(1, int(math.ceil(3 * sigma_px)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kern = np.exp(-(x ** 2) / (2 * sigma_px ** 2))
    kern /= kern.sum()
    return kern


def _gaussian_blur2d(image: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return image
    k = _gaussian_kernel1d(sigma_px)
    # Separable conv: horizontal then vertical
    pad = len(k) // 2
    # Horizontal
    tmp = np.pad(image, ((0, 0), (pad, pad)), mode="edge")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis=1, arr=tmp)
    # Vertical
    tmp = np.pad(tmp, ((pad, pad), (0, 0)), mode="edge")
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis=0, arr=tmp)
    return out


def save_heatmap(fixations: pd.DataFrame, out_path: str, sigma_norm: float = 0.03, grid: int = 256) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if fixations is None or fixations.empty:
        # Create empty white heatmap
        plt.figure(figsize=(4, 4), dpi=150)
        plt.axis('off')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return

    xs = np.clip(fixations["x"].to_numpy(), 0.0, 1.0)
    ys = np.clip(fixations["y"].to_numpy(), 0.0, 1.0)
    w = np.maximum(1.0, fixations["duration_ms"].to_numpy())

    # Weighted histogram on grid
    H, xedges, yedges = np.histogram2d(xs, ys, bins=grid, range=[[0, 1], [0, 1]], weights=w)
    # Note: histogram2d gives shape (grid, grid) with axes [x,y]; we transpose for plotting with origin upper
    H = H.T
    sigma_px = sigma_norm * grid
    H_blur = _gaussian_blur2d(H, sigma_px=sigma_px)
    H_blur /= (H_blur.max() + 1e-9)

    plt.figure(figsize=(4, 4), dpi=150)
    plt.imshow(H_blur, origin='upper', extent=[0, 1, 1, 0], cmap='hot')
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_scanpath(fixations: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(4, 4), dpi=150)
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # invert y to match display coords
    ax.set_aspect('equal')
    ax.axis('off')

    if fixations is None or fixations.empty:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return

    xs = fixations["x"].to_numpy()
    ys = fixations["y"].to_numpy()
    durs = fixations["duration_ms"].to_numpy()
    # Scale circle sizes (radius in pixels ~ sqrt(duration))
    sizes = 100 * np.sqrt(np.maximum(1.0, durs) / np.nanmax(durs))

    # Draw arrows between fixations
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.5, alpha=0.8))

    # Draw circles and numbers
    ax.scatter(xs, ys, s=sizes, facecolors='none', edgecolors='tab:red', linewidths=1.5)
    for i, (x, y) in enumerate(zip(xs, ys), start=1):
        ax.text(x, y, str(i), color='white', fontsize=8, ha='center', va='center')

    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


