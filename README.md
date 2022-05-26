# Sign2Vec<sub>d</sub>

This directory contains all the code needed to reproduce the results in the paper "Unsupervised Deep Learning Supports Reclassification of Bronze Age Cypriot Writing System". The Cypro-Minoan dataset is unfortunately unavailable, as it is covered by copyright. The code is based on [DeepClusterv2](https://github.com/facebookresearch/swav).

# Requirements
The following packages were used (we list packages versions as we can't be sure that no incompatibilities were introduced with later versions):
- pytorch 1.5<sup>1</sup>
- torchvision 0.6<sup>1</sup>
- bridson 0.1<sup>1</sup>
- pillow 7.1<sup>1</sup>
- opencv 4.3.0<sup>1</sup>
- pingouin 0.3<sup>2</sup>
- numpy 1.15
- pandas 1.2
- scipy 1.4.1
- h5py 2.10

1: these packages are needed for retraining the models, for the single sign relabelings and for obtaining the vector files from models.<br />
2: these packages are needed for the tests of single sign relabelings only.


# Training the models

We provide two scripts that can be used to train the two models: DeepClusterv2 and Sign2Vec<sub>d</sub>. To run them, use: <br />
`./train_deepcluster.sh` <br />
and <br />
`./train_sign2vec.sh` <br />
respectively. <br />
These two scripts invoke the same python code, but the DeepClusterv2 version sets the lambda constant used to weigh the Sign2Vec component of the loss to 0. <br />
:warning: **The scripts are set up to train 20 models in sequence**: this will take very long (approximately 17 days for DeepClusterv2 and Sign2Vec respectively) and we recommend editing the script to launch them in parallel on different machines instead.

# Reproducing the paper's results

Since the pretrained models used in the paper are very large in terms of storage space, we provide them separately. They are foud [here](https://drive.google.com/file/d/1xpo4DleWYVoChhggpp9IjVDzrXsHos3K/view?usp=sharing): and need to be extracted in the `models` directory. <br />
Since obtaining the vectors from the pretrained models requires a cuda-capable GPU and the Cypro Minoan signs (not included), the main results of the paper can be reproduced by using only the vectors provided for convenience (see "Validation and tests using the paleographic vector").

## Single sign relabeling

In order to perform the preliminary tests for the relabeling of signs, use the following scripts: <br />
`./single_sign_tests_deepcluster.sh` <br />
`./single_sign_tests_sign2vec.sh` <br />
for DeepClusterv2 and Sign2Vec<sub>d</sub> respectively.

:warning: **This operation is computationally expensive and it will take some time (approximately 10 minutes for each signle sign correction)** <br/>
:warning: **Some tests require temporairly renaming of signs in the dataset**, see the comments in the two scripts.

## Vector files from models

In order to obtain the required "vector.h5" and "names.txt" files from both pretrained models, use: <br />
`./get_vectors_deepcluster.sh` <br />
`./get_vectors_sign2vec.sh` <br />
for DeepCluster and Sign2Vec<sub>d</sub> respectively.

## Validation and tests using the paleographic vector

For the validation of the paleographic vector on the 32 "safe" signs, use:<br />
`./validation_paleo_vector_deepcluster.sh`  <br />
`./validation_paleo_vector_sign2vec.sh` <br />
for DeepClusterv2 and Sign2Vec<sub>d</sub> respectively.

In order to test allography using the paleographic vector, use: <br />
`./tests_type1.sh` <br />
`./tests_type2.sh` <br />
for type 1 and type 2 tests respectively.<br />
