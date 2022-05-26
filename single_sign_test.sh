#!/bin/bash

./single_sign_test.py --arch resnet18 --pretrained $1 --context_path $2  --data_path data/cyprominoan/dataset/ --img_names "${@:4}" --correct_assignment $3
