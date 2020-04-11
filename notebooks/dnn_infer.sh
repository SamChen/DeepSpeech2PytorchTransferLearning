#!/bin/bash

for manifest_path in iconect/csvs_orig/*
do
    manifest="$(basename "$manifest_path")"
    echo ${manifest}
    python ./dnn_infer.py ${manifest} || exit 0;
    break
done
