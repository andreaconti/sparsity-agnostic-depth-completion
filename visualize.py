"""
Streamlit Application to visualize depth maps
"""

import streamlit as st
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import torch

## Utilities

# simple introduction and
# a button for the dataset, a button for the number of hints
# and finally a button for the example to visualize

_DEBUG_LOCAL = False


def sidebar():
    hints_dilate = st.sidebar.number_input("Hints dilate", min_value=0, max_value=5)
    hints_cmap = st.sidebar.selectbox("Hints color map", options=["gray", "magma_r"])
    crop_sky = st.sidebar.number_input("Crop Sky", 0, 128, 100)
    gt_dilate = st.sidebar.number_input("Groundtruth dilate", min_value=0, max_value=3)
    return {
        "dilate": hints_dilate,
        "hints_cmap": hints_cmap,
        "crop_sky": crop_sky,
        "gt_dilate": gt_dilate,
    }


def main():
    cfg = sidebar()

    st.markdown(
        """
    # Sparsity Agnostic Depth Completion

    Visualize predictions from KITTI and NYU Depth V2 with different input sparsity patterns
    """
    )

    left, center, right = st.columns(3)
    with left:
        ds_name = st.selectbox(
            "Dataset", options=["kitti-official", "nyu-depth-v2-ma-downsampled"]
        )
    with center:
        if ds_name == "kitti-official":
            options = ["lines4", "lines8", "lines16", "lines32", "lines64"]
        else:
            options = [5, 50, 100, 200, 500, "grid-shift", "livox"]
        density = st.selectbox("Hints Density", options=options)

    data = load_data(ds_name, density)

    with right:
        idx = st.number_input("Example Index", min_value=0, max_value=len(data))

    ex = data[idx]
    img, hints, pred, gt = ex["img"], ex["hints"], ex["pred"], ex["gt"]
    if (d := cfg["dilate"]) > 0:
        hints = morphology.dilation(hints[..., 0], np.ones([d, d]))[..., None]
    if (cr := cfg["crop_sky"]) and ds_name == "kitti-official":
        img, hints, pred, gt = img[cr:], hints[cr:], pred[cr:], gt[cr:]

    if ds_name == "nyu-depth-v2-ma-downsampled":
        left, center, right, gt_col = st.columns(4)
        with left:
            show_img("Image", img, cmap=None)
        with center:
            if cfg["hints_cmap"] == "gray":
                show_img("Hints", hints > 0, cmap="gray")
            else:
                show_img("Hints", hints, cmap="magma_r")
        with right:
            show_img("Prediction", pred, cmap="magma_r")
        with gt_col:
            show_img("Groundtruth", gt, cmap="magma_r")
    else:
        show_img("Image", img, cmap=None)
        if cfg["hints_cmap"] == "gray":
            show_img("Hints", hints > 0, cmap="gray")
        else:
            show_img("Hints", hints, cmap="magma_r")
        show_img("Prediction", pred, cmap="magma_r")
        if (d := cfg["gt_dilate"]) > 0:
            gt = morphology.dilation(gt[..., 0], np.ones([d, d]))[..., None]
        show_img("Groundtruth", gt, cmap="magma_r")


def show_img(label: str, dmap: np.ndarray, cmap=None):
    st.caption(
        f'<div style="text-align: center;">{label}</div>', unsafe_allow_html=True
    )
    fig, ax = plt.subplots()
    ax.set_axis_off()
    fig.frameon = False
    ax.imshow(dmap, cmap=cmap)
    st.pyplot(fig)


@st.cache
def load_data(ds_name, density):
    ds = torch.hub.load(
        "andreaconti/sparsity_agnostic_depth_completion"
        if not _DEBUG_LOCAL
        else str(Path(__file__).parent),
        ds_name.replace("-", "_") + "_precomputed",
        density,
        source="github" if not _DEBUG_LOCAL else "local",
    )
    return ds


if __name__ == "__main__":
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    main()
