#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import csv
import os
from logging import getLogger

import numpy as np
import pandas as pd
import pingouin
import torch
import torch.nn.parallel
import torch.optim
from scipy.spatial.distance import cdist
from torch.nn.utils import rnn as rnn_utils

from src import resnet50 as resnet_models
from src.dataset import FlatDataset
from src.utils import bool_flag

logger = getLogger()


parser = argparse.ArgumentParser(
    description="Evaluate models: Linear classification on ImageNet"
)

#########################
#### main parameters ####
#########################
parser.add_argument(
    "--dump_path",
    type=str,
    default=".",
    help="experiment dump path for checkpoints and log",
)
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument(
    "--data_path",
    type=str,
    default="../../data/mnist",
    help="path to dataset repository",
)
parser.add_argument(
    "--workers", default=10, type=int, help="number of data loading workers"
)

parser.add_argument("--context_path", default="context.csv")

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

#########################
#### test parameters ##
#########################

parser.add_argument("--img_names", action="store", type=str, nargs="+")

parser.add_argument("--correct_assignment", action="store", type=str)


def collate_strokes(tensor_list):

    stroke_list = [tensor[0] for tensor in tensor_list]
    image_list = [tensor[1] for tensor in tensor_list]
    label_list = [tensor[2] for tensor in tensor_list]

    strokes = rnn_utils.pack_sequence(stroke_list, enforce_sorted=False)

    images = torch.stack(image_list, 0)
    labels = torch.stack(label_list, 0)

    return strokes, images, labels


def get_olivier_code(fname):
    if fname == "None" or fname == "0" or fname == "False":
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

    # load weights

    label_idx = 0
    label2idx_dict = {}
    idx2label_dict = {}
    label_count = {}

    olivier_dists = []
    correct_dists = []

    # we need to exclude hapaxes

    for idx in range(len(train_dataset)):
        imgname = train_dataset.get_imgname(idx)

        label = get_olivier_code(imgname)

        if imgname.endswith("_.png") or label == "boh":
            continue

        if label not in label_count:
            label_count[label] = 1
        else:
            label_count[label] += 1

    hapaxes_labels = set(
        [label for label, count in label_count.items() if count == 1]
    )

    for idx in range(len(train_dataset)):
        imgname = train_dataset.get_imgname(idx)

        label = get_olivier_code(imgname)

        if (
            imgname.endswith("_.png")
            or label == "boh"
            or label in hapaxes_labels
        ):
            continue

        if label not in label2idx_dict:
            label2idx_dict[label] = label_idx
            idx2label_dict[label_idx] = label
            label_count[label] = 1
            label_idx += 1
        else:
            label_count[label] += 1

    num_models = os.listdir(args.pretrained)

    for model_idx, pretrained_dir in enumerate(os.listdir(args.pretrained)):

        logger.info(
            "computing sign distances for model {} of {}".format(
                model_idx, num_models
            )
        )

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

        outputs = []
        names = []
        label_names = []
        label_indices = []

        target_img_names = []
        target_img_outputs = []
        target_img_labels = []

        vectors_by_labelname = {}

        for idx in range(len(train_dataset)):
            image, _ = train_dataset[idx]

            imgname = train_dataset.get_imgname(idx)

            label_name = get_olivier_code(imgname)

            if (
                imgname.endswith("_.png")
                or label_name == "boh"
                or label_name in hapaxes_labels
            ):
                continue

            # forward
            image = image.unsqueeze(0).cuda()
            with torch.no_grad():
                output = model(image)
            output = output.cpu().numpy()

            outputs.append(output)
            names.append(imgname)

            if imgname in args.img_names:
                target_img_names.append(imgname)
                target_img_outputs.append(output[0])
                target_img_labels.append(label_name)
            else:
                if label_name not in vectors_by_labelname:
                    vectors_by_labelname[label_name] = [output[0]]
                else:
                    vectors_by_labelname[label_name].append(output[0])

                label_names.append(label_name)
                label_indices.append(label2idx_dict[label_name])

        outputs = np.asarray(outputs)
        label_indices = np.asarray(label_indices)

        target_img_outputs = np.asarray(target_img_outputs)

        for lname, lvectors in vectors_by_labelname.items():
            if lname == target_img_labels[0]:
                lvec = np.asarray(lvectors)

                olivier_dists += list(
                    cdist(target_img_outputs, lvec, metric="cosine").flatten()
                )

            elif lname == args.correct_assignment:

                lvec = np.asarray(lvectors)

                correct_dists += list(
                    cdist(target_img_outputs, lvec, metric="cosine").flatten()
                )

    pvalue_correction_better = pingouin.mwu(
        olivier_dists, correct_dists, tail="greater"
    )

    pd.set_option("display.max_columns", None)

    print("testing distances for the following files")
    print(" ".join(target_img_names))

    print("proposed correction is")
    print(args.correct_assignment)

    print("testing whether correction is better")
    print(pvalue_correction_better)

    if pvalue_correction_better["p-val"][0] > 0.05:
        print("null hypothesis not rejected, correction is not better")
    else:
        print("null hypothesis rejected, correction is better")

    pvalue_olivier_better = pingouin.mwu(
        olivier_dists, correct_dists, tail="less"
    )

    print("testing whether published reading is better")
    print(pvalue_olivier_better)

    if pvalue_olivier_better["p-val"][0] > 0.05:
        print("null hypothesis not rejected, published reading is not better")
    else:
        print("null hypothesis rejected, published reading is better")


if __name__ == "__main__":
    main()
