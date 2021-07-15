# [MM'21] Constrained Graphic Layout Generation via Latent Optimization

This repository provides the official code for the paper "Constrained Graphic Layout Generation via Latent Optimization", especially the following code:

-   [TODO] The training code for our generative adversarial networks for layout, **LayoutGAN++**.
-   [TODO] The code to generate layouts that satisfy the given constraints by our framework, **CLG-LO**.
-   [TODO] The evaluation code to compute the metric of **Layout FID** from a set of bounding boxes.

## Installation

1. Clone this repository

    ```bash
    git clone https://github.com/ktrk115/const_layout.git
    cd const_layout
    ```

2. Create a new [conda](https://docs.conda.io/en/latest/miniconda.html) environment (Python 3.8)

    ```bash
    conda create -n const_layout python=3.8
    conda activate const_layout
    ```

3. Install the dependent Python libraries

    ```bash
    pip install -r requirements.txt
    ```

4. Install `torch-geometric` for PyTorch 1.8 following the [documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Development environment

-   Ubuntu 18.04, CUDA 11.1

## Citation

If this repository helps your research, please consider citing our paper.

```
@inproceedings{Kikuchi2021,
    title = {Constrained Graphic Layout Generation via Latent Optimization},
    author = {Kotaro Kikuchi and Edgar Simo-Serra and Mayu Otani and Kota Yamaguchi},
    booktitle = {Proceedings of the ACM International Conference on Multimedia},
    series = {MM '21},
    year = 2021,
}
```

[TODO] add volume, pages, and doi

## Licence

[TODO] add licence

## Related repositories

-   https://github.com/JiananLi2016/LayoutGAN-Tensorflow
-   https://github.com/alexandre01/deepsvg
