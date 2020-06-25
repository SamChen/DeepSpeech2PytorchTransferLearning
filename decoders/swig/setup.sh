#!/usr/bin/env bash

rm kenlm openfst-1.6.3 ThreadPool
rm build

if [ ! -d kenlm ]; then
    git clone https://github.com/kpu/kenlm.git
    echo -e "\n"
fi

if [ ! -d openfst-1.6.3 ]; then
    echo "Download and extract openfst ..."
    wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.3.tar.gz
    tar -xzvf openfst-1.6.3.tar.gz
    echo -e "\n"
fi

if [ ! -d ThreadPool ]; then
    git clone https://github.com/progschj/ThreadPool.git
    echo -e "\n"
fi

# echo "Install openfst"
# cd openfst-1.6.3
# make clean
# ./confugration
# make
# make install
# 
# echo "Install openfst"
# cd ../kenlm
# mkdir -p build
# cd build
# cmake ..
# make -j 4
pip install https://github.com/kpu/kenlm/archive/master.zip

echo "Install decoders ..."
cd ../..
python setup.py install --num_processes 8
