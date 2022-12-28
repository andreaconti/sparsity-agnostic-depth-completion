# [Sparsity Agnostic Depth Completion](https://arxiv-export1.library.cornell.edu/pdf/2212.00790)

<p>
<div align="center">
    <a href="https://andreaconti.github.io">Andrea Conti</a>
    &middot;
    <a href="https://mattpoggi.github.io">Matteo Poggi</a>
    &middot;
    <a href="http://vision.deis.unibo.it/~smatt/Site/Home.html">Stefano Mattoccia</a>
</div>
<div align="center">
    <a href="https://arxiv.org/pdf/2212.00790.pdf">[Arxiv]</a>
    <a href="https://andreaconti.github.io/projects/sparsity_agnostic_depth_completion/">[Project Page]</a>
</div>
</p>

This repository provides the evaluation code for our WACV 2023 [paper](https://arxiv.org/pdf/2212.00790.pdf).

We present a novel depth completion approach agnostic to the sparsity of depth points, that is very likely to vary in many practical applications. State-of-the-art approaches yield accurate results only when processing a specific density and distribution of input points, i.e. the one observed during training, narrowing their deployment in real use cases. On the contrary, our solution is robust to uneven distributions and extremely low densities never witnessed during training. Experimental results on standard indoor and outdoor benchmarks highlight the robustness of our framework, achieving accuracy comparable to state-of-the-art methods when tested with density and distribution equal to the training one while being much more accurate in the other cases.

## Citation

```
@inproceedings{aconti2023spagnet,
    title={Sparsity Agnostic Depth Completion},
    author={Conti, Andrea and Poggi, Matteo and Mattoccia, Stefano},
    booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision},
    note={WACV},
    year={2023},
}
```

## Qualitative Results

To better visualize the performance of our proposal we provide a simple [streamlit](https://streamlit.io) application, which can be executed in the following way:

```bash
$ git clone https://github.com/andreaconti/sparsity-agnostic-depth-Completion
$ cd sparsity-agnostic-depth-Completion
$ mamba env create -f environment.yml
$ mamba activate sparsity-agnostic-depth-Completion
$ streamlit run visualize.py
```

![](https://github.com/andreaconti/sparsity-agnostic-depth-completion/blob/master/readme_assets/visualize-demo.gif)

It may take a while when you change dataset or hints density to display since it have to download and unpack the data.

## Quantitative Results

We provide precomputed depth maps for [KITTI Depth Completion](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/kitti-official.tar) and [NYU Depth V2](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu-depth-v2-ma-downsampled.tar), with different sparsity patterns.

Moreover we provide a simple evaluation script to compute metrics:

```bash
$ git clone https://github.com/andreaconti/sparsity-agnostic-depth-Completion
$ cd sparsity-agnostic-depth-Completion
$ mamba env create -f environment.yml
$ mamba activate sparsity-agnostic-depth-Completion
$ python evaluate.py <kitti-official | nyu-depth-v2-ma-downsampled> <hints density>
```

For instance:

```bash
# KITTI evaluation
$ python evaluate.py kitti-official lines64
# NYU Depth V2 evaluation
$ python evaluate.py nyu-depth-v2-ma-downsampled 500
```