from pathlib import Path as _Path
import torch as _torch
from torch.utils.data import Dataset as _Dataset
from typing import Literal, Dict
import h5py
import numpy as np


def _download(name: str) -> _Path:
    """
    downloads a file from github and puts it under downloads
    """
    to = _Path(__file__).parent  / "downloads" / name
    if not to.exists():
        _torch.hub.download_url_to_file(
            f"https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/{name}",
            to
        )
    return to

# precomputed results


class _PrecomputedDataset(_Dataset):
    def __init__(self, img_gt_root: _Path, pred_hints_root: _Path, in_memory: bool = False):
        img_gt = h5py.File(img_gt_root)
        self._img = img_gt["img"]
        self._gt = img_gt["gt"]
        pred_hints = h5py.File(pred_hints_root)
        self._preds = pred_hints["preds"]
        self._hints = pred_hints["hints"]
        if in_memory:
            self._img = np.array(self._img)
            self._gt = np.array(self._gt)
            self._preds = np.array(self._preds)
            self._hints = np.array(self._hints)

    def __len__(self):
        return self._img.shape[0]

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        return {
            "img": self._img[index],
            "gt": self._gt[index],
            "hints": self._hints[index],
            "pred": self._preds[index],
        }


def kitti_official_precomputed(
    hints_density: Literal["lines4", "lines8", "lines16", "lines32", "lines64"], 
    in_memory: bool = False,
) -> _Dataset:
    assert hints_density in ["lines4", "lines8", "lines16", "lines32", "lines64"], f"{hints_density} not available"
    img_gt = _download("kitti_img_gt.h5")
    preds = _download(f"kitti_pred_with_{hints_density}.h5")
    return _PrecomputedDataset(img_gt, preds, in_memory)


def nyu_depth_v2_ma_downsampled_precomputed(
    hints_density: Literal[5, 50, 100, 200, 500, "livox", "grid-shift"],
    in_memory: bool = False
) -> _Dataset:
    assert hints_density in [5, 50, 100, 200, 500, "livox", "grid-shift"], f"{hints_density} not available"
    img_gt = _download("nyu_img_gt.h5")
    preds = _download(f"nyu_pred_with_{hints_density}.h5")
    return _PrecomputedDataset(img_gt, preds, in_memory)