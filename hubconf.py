from pathlib import Path as _Path
import torch as _torch
from torch.utils.data import Dataset as _Dataset
import tarfile as _tarfile
from typing import Literal, Dict
import h5py
import numpy as np


def _download(name: str, ext: str = "") -> _Path:
    """
    Downloads from github the required resource, and if it is a tar dir it extract it in a
    folder with the same name without extension
    """
    if not (out_path := (_Path(__file__).parent / f"downloads/{name}{ext}")).exists():
        _torch.hub.download_url_to_file(
            f"https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/{name}{ext}",
            str(_Path(__file__).parent / f"downloads/{name}{ext}"),
        )
    if ext == ".tar":
        if not (out_dir := out_path.parent / name).exists():
            with _tarfile.open(out_path) as tar:
                out_dir = out_path.parent / name
                out_dir.mkdir(exist_ok=True, parents=True)
                tar.extractall(out_dir)
        return out_dir
    else:
        return out_path


# precomputed results


class _PrecomputedDataset(_Dataset):
    def __init__(self, img_gt_root: _Path, pred_hints_root: _Path):
        img_gt = h5py.File(img_gt_root)
        self._img = np.array(img_gt["img"])
        self._gt = np.array(img_gt["gt"])
        pred_hints = h5py.File(pred_hints_root)
        self._preds = np.array(pred_hints["preds"])
        self._hints = np.array(pred_hints["hints"])

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
    hints_density: Literal["lines4", "lines8", "lines16", "lines32", "lines64"]
) -> _Dataset:
    root = _download("kitti-official", ".tar")
    return _PrecomputedDataset(
        root / "img_gt.h5", root / f"pred_with_{hints_density}.h5"
    )


def nyu_depth_v2_ma_downsampled_precomputed(
    hints_density: Literal[5, 50, 100, 200, 500, "livox", "grid-shift"]
) -> _Dataset:
    root = _download("nyu-depth-v2-ma-downsampled", ".tar")
    return _PrecomputedDataset(
        root / "img_gt.h5", root / f"pred_with_{hints_density}.h5"
    )


# models

# TODO: wip
