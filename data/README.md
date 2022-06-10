# Data preparation

## Create a folder for the datasets

```bash
TOPDIR=$(git rev-parse --show-toplevel)
DATASET=$TOPDIR/data/dataset
mkdir $DATASET
```

## Download and place the datasets

### [Rico](https://interactionmining.org/rico)

1.  Download `rico_dataset_v0.1_semantic_annotations.zip` from "UI Screenshots and Hierarchies with Semantic Annotations" and decompress it.
2.  Create the new directory `$DATASET/rico/raw/` and move the contents into it as shown below:

    ```dircolors
    $DATASET/rico/raw/
    └── semantic_annotations
        ├── 0.json
        ├── 0.png
        ├── 10000.json
        ├── 10000.png
        ├── 10002.json
        ├── ...
    ```

### [PubLayNet](https://developer.ibm.com/exchanges/data/all/publaynet/)

1.  Download `labels.tar.gz` and decompress it.
2.  Create the new directory `$DATASET/publaynet/raw/` and move the contents into it as shown below:

    ```dircolors
    $DATASET/publaynet/raw/
    └── publaynet
        ├── LICENSE.txt
        ├── README.txt
        ├── train.json
        └── val.json
    ```

### [Magazine](https://xtqiao.com/projects/content_aware_layout/)

1.  Download `MagLayout.zip` and decompress it.
2.  Create the new directory `$DATASET/magazine/raw/` and move the contents into it as shown below:

    ```dircolors
    $DATASET/magazine/raw/
    └── layoutdata
        ├── annotations
        │   ├── fashion_0001.xml
        │   ├── fashion_0002.xml
        │   ├── fashion_0003.xml
        │   ├── fashion_0004.xml
        │   ├── fashion_0005.xml
        │   ├── ...
    ```

# Statistics

| Name      | # labels | max. # elements | # train layouts | # val layouts | # test layouts |
| :-------- | -------: | --------------: | --------------: | ------------: | -------------: |
| rico      |       13 |               9 |          17,515 |         1,030 |          2,061 |
| publaynet |        5 |               9 |         160,549 |         8,450 |          4,226 |
| magazine  |        5 |              33 |           3,331 |           196 |            392 |
