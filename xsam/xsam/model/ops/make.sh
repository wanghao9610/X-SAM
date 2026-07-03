#!/usr/bin/env bash
# refer to https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/make.sh
if [ -d build ]; then
    rm -rf build
fi

python setup.py build install
