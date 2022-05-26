#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import base64
import csv
import glob
import os
from logging import getLogger

import h5py
import numpy as np
import scipy
import torch
import torch.nn.parallel
import torch.optim
from torch.nn.utils import rnn as rnn_utils

from src import resnet50 as resnet_models
from src.dataset import FlatDataset
from src.utils import bool_flag, initialize_exp

logger = getLogger()


parser = argparse.ArgumentParser(
    description="Evaluate vectors and save them to .h5 file"
)

##########################
#### input parameters ####
##########################

parser.add_argument(
    "--data_path",
    type=str,
    default="../../data/mnist",
    help="path to dataset repository",
)
parser.add_argument("--context_path", action="store")

###########################
#### output parameters ####
###########################

parser.add_argument("--vectors_filename", action="store")
parser.add_argument("--names_filename", action="store")

#########################
#### model parameters ###
#########################
parser.add_argument(
    "--arch", default="resnet50", type=str, help="convnet architecture"
)
parser.add_argument(
    "--pretrained",
    default="./models-10runs/",
    type=str,
    help="path to pretrained weights",
)
parser.add_argument(
    "--global_pooling",
    default=True,
    type=bool_flag,
    help="if True, we use the resnet50 global average pooling",
)
parser.add_argument(
    "--use_bn",
    default=False,
    type=bool_flag,
    help="optionally add a batchnorm layer before the linear classifier",
)

parser.add_argument(
    "--output_dim",
    default=128,
    type=int,
    help="optionally add a batchnorm layer before the linear classifier",
)

parser.add_argument(
    "--hidden_mlp",
    default=2048,
    type=int,
    help="hidden layer dimension in projection head",
)


def collate_strokes(tensor_list):

    stroke_list = [tensor[0] for tensor in tensor_list]
    image_list = [tensor[1] for tensor in tensor_list]
    label_list = [tensor[2] for tensor in tensor_list]

    strokes = rnn_utils.pack_sequence(stroke_list, enforce_sorted=False)

    images = torch.stack(image_list, 0)
    labels = torch.stack(label_list, 0)

    return strokes, images, labels


def get_olivier_code(fname):
    if fname == "None" or fname == "0":
        return ""

    return fname.split(".")[-2].split("_")[-1]


def main():
    global args, best_acc
    args = parser.parse_args()

    train_dataset = FlatDataset(args.data_path, context_path=args.context_path)

    logger.info("Building data done")

    # build model
    model = resnet_models.__dict__[args.arch](
        output_dim=args.output_dim,
        eval_mode=False,
        hidden_mlp=args.hidden_mlp,
        normalize=True,
    )

    # convert batch norm layers (if any)
    # model to gpu
    model = model.cuda()
    model.eval()

    num_images = len(train_dataset)

    old_names = None
    num_models = len(os.listdir(args.pretrained))

    h5py_vectors = h5py.File(args.vectors_filename, "w")
    vectors_dataset = h5py_vectors.create_dataset(
        "vec", (num_images, num_models, 128)
    )

    for model_idx, pretrained_dir in enumerate(os.listdir(args.pretrained)):

        pretrained = "{}/{}/checkpoint.pth.tar".format(
            args.pretrained, pretrained_dir
        )

        if os.path.isfile(pretrained):
            state_dict = torch.load(pretrained, map_location="cuda")

            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            # remove prefixe "module."
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }

            for k, v in model.state_dict().items():
                if k not in list(state_dict):
                    logger.info(
                        'key "{}" could not be found in provided state dict'.format(
                            k
                        )
                    )
                elif state_dict[k].shape != v.shape:
                    logger.info(
                        'key "{}" is of different shape in model and provided state dict'.format(
                            k
                        )
                    )
                    state_dict[k] = v
            msg = model.load_state_dict(state_dict, strict=False)
            logger.info("Load pretrained model with msg: {}".format(msg))
        else:
            logger.info(
                "No pretrained weights found => training with random weights"
            )

        # train the network for one epoch
        logger.info("============ Starting eval ... ============")

        names = []

        for idx in range(len(train_dataset)):

            image, _ = train_dataset[idx]

            imgname = train_dataset.get_imgname(idx)

            # forward
            with torch.no_grad():
                output = model(image.unsqueeze(0).cuda())
            output = output.cpu().numpy()

            names.append(imgname)
            vectors_dataset[idx, model_idx, :] = output[0]

        names = np.asarray(names)

        assert old_names is None or (names == old_names).all()

        if model_idx == 0:
            names_file = open(args.names_filename, "w")
            names_file.write("\n".join(names))
            names_file.close()

    h5py_vectors.close()


if __name__ == "__main__":
    main()
