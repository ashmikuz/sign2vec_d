#!/bin/bash

./vectors_from_models.py --arch resnet18 --pretrained models/deepcluster --data_path data/cyprominoan/dataset_deepcluster_corrections/ --context_path data/contexts/context_deepcluster_corrections.csv --vectors_filename "pretrained-vectors/vectors_deepcluster.h5" --names_filename "pretrained-vectors/names_deepcluster.txt"
