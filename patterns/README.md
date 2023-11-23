# Sparsity Patterns

This folder contains the sparsity patterns used in [Sparsity Agnostic Depth Completion](https://openaccess.thecvf.com/content/WACV2023/html/Conti_Sparsity_Agnostic_Depth_Completion_WACV_2023_paper.html).

## Grid Pattern

About the grid pattern please refer to [Low Memory Footprint Quantized Depth Completion](https://github.com/sony/ai-research-code/tree/master/quantized-depth-completion). Be aware that the grid pattern is distributed under the following [LICENSE](https://github.com/andreaconti/sparsity-agnostic-depth-completion/blob/master/patterns/LICENSE.md). If you use the grid pattern, please cite:


```
@inproceedings{jiang_low_2022,
	title = {A {Low} {Memory} {Footprint} {Quantized} {Neural} {Network} for {Depth} {Completion} of {Very} {Sparse} {Time}-of-{Flight} {Depth} {Maps}},
	booktitle = {Proceedings of the {IEEE}/{CVF} {Conference} on {Computer} {Vision} and {Pattern} {Recognition} ({CVPR}) {Workshops}},
	author = {Jiang, Xiaowen and Cambareri, Valerio and Agresti, Gianluca and Ugwu, Cynthia I and Simonetto, Adriano and Zanuttigh, Pietro and Cardinaux, Fabien},
	month = jun,
	year = {2022},
}
```

## Livox Pattern

The Livox pattern has been synthetically generated to roughly resemble the pattern provided by [Livox sensors](https://www.livoxtech.com/). Since the real pattern rotates in time, we provide a mask of shape $H \times W \times N$ where the pattern rotates along the $N$ dimension. At the $320 \times 240$ resolution it provides about 150 sparse points per frame. If you use this pattern, please cite:

```
@InProceedings{Conti_2023_WACV,
    author    = {Conti, Andrea and Poggi, Matteo and Mattoccia, Stefano},
    title     = {Sparsity Agnostic Depth Completion},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {5871-5880}
}
```

## Fair Comparison

To perform a fair comparison with this work we encourage to use the precomputed bundles on the test set containing sparse points, groundtruth and image, which can be downloaded here:

### Nyu Depth V2

- [RGB+GT](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_img_gt.h5)
- [5 Points](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_pred_with_5.h5)
- [50 Points](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_pred_with_50.h5)
- [100 Points](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_pred_with_100.h5)
- [200 Points](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_pred_with_200.h5)
- [500 Points](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_pred_with_500.h5)
- [Livox](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_pred_with_livox.h5)
- [Grid Shift](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_pred_with_grid-shift.h5)

### KITTI

- [RGB+GT](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/kitti_img_gt.h5)
- [4 Lines](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/kitti_pred_with_lines4.h5)
- [8 Lines](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/kitti_pred_with_lines8.h5)
- [16 Lines](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/kitti_pred_with_lines16.h5)
- [32 Lines](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/kitti_pred_with_lines32.h5)
- [64 Lines](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/kitti_pred_with_lines64.h5)