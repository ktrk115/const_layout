#!/bin/bash -e

mkdir -p pretrained

function download {
    ckpt_name=${1}_${2}.pth.tar
    url=https://esslab.jp/~kotaro/files/const_layout/${ckpt_name}
    wget ${url} -O pretrained/${ckpt_name} -q || wget ${url} -O pretrained/${ckpt_name}
}

for name in rico publaynet magazine;
do
    # LayoutGAN++
    download layoutganpp $name
    # LayoutNet
    download layoutnet $name
done

echo "Successfully downloaded the pretrained models"