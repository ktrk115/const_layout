# [MM'21] Constrained Graphic Layout Generation via Latent Optimization

This repository provides the official code for the paper "Constrained Graphic Layout Generation via Latent Optimization", especially the code for:

-   **LayoutGAN++**: generative adversarial networks for layout generation
-   **CLG-LO**: a framework for generating layouts that satisfy constraints
-   **Layout evaluation**: measuring the quantitative metrics of _Layout FID_, _Maximum IoU_, _Alignment_, and _Overlap_ for generated layouts

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

3. Install [PyTorch 1.8.\*](https://pytorch.org/get-started/locally/) and the corresponding versoin of [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

4. Install the other dependent libraries

    ```bash
    pip install -r requirements.txt
    ```

5. Prepare data (see [this instruction](data/))

6. Download pre-trained models

    ```bash
    ./download_model.sh
    ```

## Development environment

-   Ubuntu 18.04, CUDA 11.1

## Generate layouts with LayoutGAN++

```bash
python generate.py pretrained/layoutganpp_rico.pth.tar --out_path output/generated_layouts.pkl --num_save 5
```

## Evaluate generate layouts

```bash
python eval.py rico output/generated_layouts.pkl
```

## Train LayoutGAN++ by yourself

```bash
python train.py --dataset rico --batch_size 64 --iteration 200000 --latent_size 4 --lr 1e-05 --G_d_model 256 --G_nhead 4 --G_num_layers 8 --D_d_model 256 --D_nhead 4 --D_num_layers 8
```

## Citation

If this repository helps your research, please consider citing our [paper](https://doi.org/10.1145/3474085.3475497).

```
@inproceedings{Kikuchi2021,
    title = {Constrained Graphic Layout Generation via Latent Optimization},
    author = {Kotaro Kikuchi and Edgar Simo-Serra and Mayu Otani and Kota Yamaguchi},
    booktitle = {Proceedings of the ACM International Conference on Multimedia},
    series = {MM '21},
    volume = {},
    year = {2021},
    pages = {},
    doi = {10.1145/3474085.3475497}
}
```

## Licence

GNU AGPLv3

## Related repositories

-   https://github.com/JiananLi2016/LayoutGAN-Tensorflow
-   https://github.com/alexandre01/deepsvg
