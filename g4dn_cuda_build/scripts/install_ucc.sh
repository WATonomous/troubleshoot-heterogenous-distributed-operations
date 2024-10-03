#!/bin/bash

set -ex

git clone --recursive https://github.com/openucx/ucc.git
pushd ucc
git checkout ${UCC_COMMIT}
git submodule update --init --recursive

./autogen.sh
NVCC_GENCODE="-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_86,code=compute_86"
./configure --prefix=/usr                 \
  --with-ucx=/usr                         \
  --with-cuda=/usr/local/cuda             \
  --with-nvcc-gencode="${NVCC_GENCODE}"
time make -j
sudo make install

popd
rm -rf ucc