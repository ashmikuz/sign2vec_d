#!/bin/bash

./vectors_from_models.py --arch resnet18 --pretrained ./models/sign2vec --data_path ./data/cyprominoan/dataset_sign2vec_corrections/ --context_path ./data/contexts/context_sign2vec_corrections.csv --vectors_filename "pretrained-vectors/vectors_sign2vec.h5" --names_filename "pretrained-vectors/names_sign2vec.txt"
