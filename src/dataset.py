#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import csv
import os
import random
from abc import ABC, abstractmethod
from glob import glob
from logging import getLogger

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from bridson import poisson_disc_samples
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, IterableDataset

logger = getLogger()


# create an image composed of random points using posson disc sampling
def random_point_image():
    im = Image.new("RGB", (1000, 1000), color="white")
    draw = ImageDraw.Draw(im)

    points = poisson_disc_samples(width=980, height=980, r=70, k=30)

    if len(points) < 10:
        logger.debug(
            "created random point image with less than 10 points, this should never happen"
        )

    for point in points:

        random_size = random.randint(10, 15)
        x, y = point

        draw.ellipse(
            (
                x - random_size,
                y - random_size,
                x + random_size,
                y + random_size,
            ),
            fill=(0, 0, 0),
            outline=(0, 0, 0),
        )

    im = im.resize((100, 100), Image.LANCZOS)

    return im


class GenericDataset(Dataset):
    def __init__(
        self,
        data_path,
        context_path,
        damaged_path=None,
        debug=False,
        text_path=None,
        size_dataset=None,
        skip_artificial_dividers=False,
    ):
        super()

        self.data_path = data_path
        self.damaged_path = damaged_path
        self.debug = debug

        if text_path:
            text_file = open(text_path, "r")
            self.text = text_file.read()
            text_file.close()

            self.text = " " + self.text
            self.text = self.text.lower()
        else:
            self.text = None

        if self.debug:
            logger.debug("populating samples from context file")
        self.populate_samples_from_context(context_path)

        self.means, self.stds = self.get_means_stds()

        if size_dataset:
            self.size_dataset = size_dataset
        else:
            self.size_dataset = len(self.samples)

    def populate_samples_from_context(
        self, context_path, skip_artificial_dividers=True
    ):

        self.context_dict = {}
        context_file = open(context_path, "r")
        reader = csv.reader(context_file)

        self.samples = []

        for line in reader:
            if self.text:
                sign = line[1].split("/")[-1]
                sign_num = int(sign.split(".")[0])
                category = self.text[sign_num]
            elif "/" in line:
                category = line[1].split("/")[0]
            else:
                category = 0

            if skip_artificial_dividers and line[1] == "None":
                continue

            self.samples.append((line[1], category))
            self.context_dict[line[1]] = [line[0], line[2]]

        if self.debug:
            logger.debug("found {} files".format(len(self.samples)))

    # get means and stds from the dataset
    def get_means_stds(self):

        totensor = transforms.ToTensor()
        images = []

        all_images = []

        for key, value in self.context_dict.items():
            all_images += [key] + value

        all_images = set(all_images)

        for path in all_images:
            if path == "0" or path == "None" or path == "False":
                continue

            img = self.loader(path)
            img = totensor(img)
            img = img.permute(1, 2, 0).view(-1, 3)

            images.append(img)

        images = torch.cat(images)
        means = torch.mean(images, dim=(0), dtype=torch.float64)
        stds = torch.std(images, dim=(0))

        if self.debug:
            logger.debug(
                "found {} images".format(images.size()[0] / (100 * 100))
            )
            logger.debug("means: {}".format(means))
            logger.debug("stds: {}".format(stds))

        return means, stds

    # our version of the loader function from datasets.ImageFolder
    def loader(self, path):

        complete_path = "{}/{}".format(self.data_path, path)

        if self.debug:
            logger.debug("attempting to load {}".format(complete_path))

        # if the file does not exist, try the damaged path

        if not os.path.isfile(complete_path):
            complete_path = "{}/{}".format(self.damaged_path, path)

        with open(complete_path, "rb") as f:
            img = Image.open(f)

            return img.convert("RGB")
        # return img

    def get_imgname(self, index):
        return self.samples[index][0].split("/")[-1]

    def __len__(self):
        return self.size_dataset


class FlatDataset(GenericDataset):
    def __init__(self, data_path, context_path, debug=False, text_path=None):
        super().__init__(
            data_path,
            context_path,
            debug=debug,
            text_path=text_path,
            skip_artificial_dividers=True,
        )
        self.data_path = data_path

        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds),
            ]
        )

    def __getitem__(self, index):
        path, label = self.samples[index]

        img = self.loader(path)

        img = self.trans(img)

        return img, label


class MultiCropCbowDataset(GenericDataset):
    def __init__(
        self,
        data_path,
        nmb_crops,
        size_crops,
        min_scale_crops,
        max_scale_crops,
        context_path,
        damaged_path=None,
        return_index=False,
        bootstrap_fraction=None,
        dump_path=None,
        separator_names=None,
        debug=False,
        size_dataset=None,
    ):
        super().__init__(
            data_path,
            context_path,
            damaged_path=damaged_path,
            debug=debug,
            size_dataset=size_dataset,
        )
        self.return_index = return_index
        self.epoch = 0

        self.trans = []
        color_transform = transforms.Compose(
            [get_color_distortion(), RandomGaussianBlur()]
        )

        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size=size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            self.trans.extend(
                [
                    transforms.Compose(
                        [
                            randomresizedcrop,
                            transforms.RandomHorizontalFlip(p=0.5),
                        ]
                    )
                ]
                * nmb_crops[i]
            )

        self.normalize = transforms.Compose(
            [
                color_transform,
                transforms.ToTensor(),
                transforms.Normalize(mean=self.means, std=self.stds),
            ]
        )

        # populate separators, we need to temporairly change dir

        if separator_names:
            cwd = os.getcwd()
            os.chdir(self.data_path)
            self.separator_files = []
            for separator_name in separator_names:
                self.separator_files += glob(
                    "1/*_{}.png".format(separator_name)
                )
            os.chdir(cwd)
        else:
            self.separator_files = []

        logger.info(
            "found {} word separators".format(len(self.separator_files))
        )

        # apply bootstrap

        if bootstrap_fraction is not None:
            total_dataset_size = len(self.samples)
            num_samples = round(bootstrap_fraction * total_dataset_size)
            self.samples = [
                self.samples[i] for i in torch.randperm(total_dataset_size)
            ]
            self.samples = [
                self.samples[random.randint(0, num_samples)]
                for _ in range(total_dataset_size)
            ]

            # save bootstrap to resume
            bootstrap_sample_file = open(dump_path + "/bootstrap_samples", "w")

            sample_paths = "\n".join([path for path, _ in self.samples])
            bootstrap_sample_file.write(sample_paths)

            bootstrap_sample_file.close()

    def load_context_img(self, path):

        if path == "None":
            # INITIAL or FINAL

            assert self.separator_files

            random_separator_idx = random.randrange(
                0, len(self.separator_files)
            )

            if self.debug:
                logger.debug("loading random separators")

            return self.loader(self.separator_files[random_separator_idx])

        elif path == "0":
            # DAMAGED

            return random_point_image()
        elif path == "False":
            return Image.new("RGB", (100, 100), color="black")
        else:
            return self.loader(path)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):

        if index >= len(self.samples):
            oversample_size = len(self)
            data_size = len(self.samples)
            new_index = index - len(self.samples)
            index = (
                ((oversample_size - data_size) * self.epoch) + new_index
            ) % data_size

        path, _ = self.samples[index]
        path_left, path_right = self.context_dict[path]

        div_nodiv_bool = [
            float("I.png" in path or "SPACE.png" in path or path == "None")
        ] * len(self.trans)

        image_center = self.load_context_img(path)
        image_left = self.load_context_img(path_left)
        image_right = self.load_context_img(path_right)

        multi_crops_center = list(
            map(lambda trans: trans(image_center), self.trans)
        )
        multi_crops_left = list(
            map(lambda trans: trans(image_left), self.trans)
        )

        multi_crops_right = list(
            map(lambda trans: trans(image_right), self.trans)
        )

        images = (image_center, image_left, image_right)
        paths = (path, path_left, path_right)

        for img_idx, crops in enumerate(
            [multi_crops_center, multi_crops_left, multi_crops_right]
        ):
            for idx, crop in enumerate(crops):

                count_nonwhite_pixels = torch.sum(
                    transforms.ToTensor()(images[img_idx])[0] < 1.0
                )

                pixel_threshold = min(0.5 * count_nonwhite_pixels, 50)

                count_redo = 0

                while (
                    torch.sum(transforms.ToTensor()(crop)[0] < 1.0)
                    < pixel_threshold
                ):
                    count_redo += 1

                    if count_redo >= 10:

                        logger.info(
                            "redoing too many times for image {}".format(
                                paths[img_idx]
                            )
                        )
                        logger.info(
                            "sum is {}".format(
                                torch.sum(transforms.ToTensor()(crop)[0] < 1.0)
                            )
                        )

                        logger.info("threshold is {}".format(pixel_threshold))

                        if self.debug:
                            fig, ax = plt.subplots(nrows=2, ncols=1)
                            ax[0].imshow(images[img_idx])
                            ax[1].imshow(crop)

                            plt.show()

                    crop = self.trans[idx](images[img_idx])
                crops[idx] = self.normalize(crop)

        # DEBUG: show the images and their distorted counterparts
        # if self.debug:

        # fig, ax = plt.subplots(nrows=2, ncols=3)

        # ax[0][0].imshow(image_left)
        # ax[0][1].imshow(image_center)
        # ax[0][2].imshow(image_right)

        # ax[1][0].imshow(multi_crops_left[-1].permute(1, 2, 0))
        # ax[1][1].imshow(multi_crops_center[-1].permute(1, 2, 0))
        # ax[1][2].imshow(multi_crops_right[-1].permute(1, 2, 0))

        # plt.show()

        if self.return_index:
            return (
                index,
                multi_crops_center,
                multi_crops_left,
                multi_crops_right,
                div_nodiv_bool,
            )

        return (
            multi_crops_center,
            multi_crops_left,
            multi_crops_right,
            div_nodiv_bool,
        )


class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5

        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1

        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)


def get_color_distortion(s=0.1):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])

    return color_distort


def test():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("damaged_path")
    parser.add_argument("context_path")
    parser.add_argument("separator_name")
    args = parser.parse_args()

    dset = MultiCropCbowDataset(
        args.data_path,
        nmb_crops=[6, 10],
        size_crops=[80, 60],
        min_scale_crops=[0.6, 0.4],
        max_scale_crops=[0.8, 0.6],
        context_path=args.context_path,
        debug=True,
        separator_name=args.separator_name,
    )
    #
    # dset = FlatDataset(
    # args.data_path,
    # args.context_path,
    # "../../data/englishchars/alicewonderland.txt",
    # )

    for cbow_center, cbow_left, cbow_right in dset:
        print(cbow_center[0].shape)

        continue


if __name__ == "__main__":
    test()
