#!/bin/bash

./paleo_vector_test.py --num_models 20 --list_for_dir ./data/paleo-vector-info/validation_signs.txt --cm2only_file ./data/paleo-vector-info/cm2only_sign.txt --tests_file ./data/paleo-vector-info/tests_type1.txt --vectors_filename ./pretrained-vectors/vectors_sign2vec.h5 --names_filename ./pretrained-vectors/names_sign2vec.txt
