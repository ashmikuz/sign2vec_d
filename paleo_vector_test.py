#!/usr/bin/env python3

import argparse
import sys

import h5py
import numpy as np
from scipy.spatial.distance import cdist


def get_subcorpus(fname):
    if "CM_ENKO.Atab.001" in fname:
        print("ERROR")
    elif "CM_ENKO.Atab" in fname:
        label = "CM2"
    elif "CM_RASH.Atab" in fname:
        label = "CM2"
    elif (
        "CM_RASH.Aéti" in fname
        or "CMADD_RASH" in fname
        or "CM_RASH.Mvas" in fname
        or "CM_SYRI.Psce" in fname
        or "CMADD_RS-1963.Avas.003" in fname
    ):
        label = "CM1"
    else:
        label = "CM1"

    return label


def get_classes(names):
    classes = []

    for name in names:
        subcorpus = get_subcorpus(name)

        if subcorpus == "CM1":
            classes.append(0)
        else:
            classes.append(1)

    return np.asarray(classes)


def get_sign_code(fname):
    if fname == "None" or fname == "0" or "_.png" in fname:
        return ""

    sign_name = fname.split(".")[-2].split("_")[-1]

    # in the original data, 008 and 013 are swapped
    if sign_name == "008":
        sign_name = "013"
    elif sign_name == "013":
        sign_name = "008"

    return sign_name


def main(args):

    sign_subcorpora_dict = {"CM1": {}, "CM2": {}, "CM3": {}}

    sign_centroid_list_cm23 = []
    sign_centroid_names_cm23 = []

    ignore = set(
        [
            "VII",
            "CCC",
            "IIII",
            "202",
            "XXX",
            "066",
            "III",
            "IIIIII",
            "084",
            "X",
            "201",
            "I",
            "P",
            "et",
            "punto",
            "105",
            "058",
            "12bis",
            "066",
            "094",
            "098",
            "108",
            "114",
        ]
    )
    names_file = open(args.names_filename, "r")
    names_list = names_file.read().split("\n")

    vectors_h5_file = h5py.File(args.vectors_filename, "r")
    vectors_dset = vectors_h5_file["vec"]
    num_images = vectors_dset.shape[0]
    vectors = np.zeros((num_images, args.num_models, 128))
    vectors[:] = vectors_dset[:]
    vectors = np.reshape(vectors, (num_images, 128 * args.num_models))

    # populate signs and vectors in subcorpora dict

    for name_idx, name in enumerate(names_list):

        ocode = get_sign_code(name)

        if name.endswith("_.png") or ocode == "boh" or ocode in ignore:
            continue

        subcorpus = get_subcorpus(name)

        subcorpora_dict = sign_subcorpora_dict[subcorpus]

        if ocode in subcorpora_dict:
            subcorpora_dict[ocode].append((name, vectors[name_idx]))
        else:
            subcorpora_dict[ocode] = [(name, vectors[name_idx])]

    # calculate centroids for cm23

    for sign_code, values in sign_subcorpora_dict["CM2"].items():
        if len(values) > 2:
            sign_centroid_names_cm23.append(sign_code)

            centroid = np.asarray([s[1] for s in values])

            sign_centroid_list_cm23.append(np.mean(centroid, axis=0))

    sign_centroid_names_cm23only = []
    sign_centroid_list_cm23only = []

    if args.cm2only_file is not None:
        cm2only_file = open(args.cm2only_file, "r")
        cm2only = [s[:-1] for s in cm2only_file]
        cm2only_file.close()

        for sign_code in cm2only:
            values = sign_subcorpora_dict["CM2"][sign_code]

            if len(values) > 2:
                sign_centroid_names_cm23only.append(sign_code)
                centroid = np.asarray([s[1] for s in values])
                sign_centroid_list_cm23only.append(np.mean(centroid, axis=0))

    else:
        sign_centroid_names_cm23only = sign_centroid_names_cm23
        sign_centroid_list_cm23only = sign_centroid_list_cm23
        cm2only = sign_centroid_names_cm23only

    list_signs_for_dir_file = open(args.list_for_dir, "r")
    list_signs_for_dir = [line[:-1] for line in list_signs_for_dir_file]
    list_signs_for_dir_file.close()

    differences_cm23_1 = []

    for s_idx, sign_name in enumerate(list_signs_for_dir):

        sys.stderr.write(
            "sign {} of {}\n".format(s_idx, len(list_signs_for_dir))
        )

        cm1_instances = sign_subcorpora_dict["CM1"][sign_name]
        cm2_instances = sign_subcorpora_dict["CM2"][sign_name]

        # calculate difference between centroids
        cm1_centroid = np.mean(
            np.asarray([sign[1] for sign in cm1_instances]), axis=0
        )
        cm2_centroid = np.mean(
            np.asarray([sign[1] for sign in cm2_instances]), axis=0
        )

        differences_cm23_1.append(cm2_centroid - cm1_centroid)

    mean_difference = np.mean(np.asarray(differences_cm23_1), axis=0)

    # signs to be tested

    tests_file = open(args.tests_file, "r")
    tests_signs = [s[:-1] for s in tests_file]
    tests_file.close()

    for test_sign in tests_signs:

        test_sign_vectors = [
            v[1] for v in sign_subcorpora_dict["CM1"][test_sign]
        ]

        cm23only_centroids = np.asarray(sign_centroid_list_cm23only)

        # calculate the sum between a centroid for a cm1 sign and the mean cm23/1 distance
        cm1_centroid = np.mean(test_sign_vectors, axis=0)

        cm2_point = np.asarray(cm1_centroid + mean_difference)

        cm2_point = np.expand_dims(cm2_point, 0)

        # get the closest cm2only signs
        cm2point_centroids_distances = cdist(
            cm2_point, cm23only_centroids, metric="cosine"
        )
        closest_sign_idx = np.argsort(cm2point_centroids_distances[0], axis=0)
        closest_sign = np.asarray(sign_centroid_names_cm23only)[
            closest_sign_idx
        ]

        distances = cm2point_centroids_distances[0, closest_sign_idx]

        print("{} is closest to {}".format(test_sign, closest_sign[:10]))
        print("distances are {}".format(distances[:10]))
        print("")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vectors_filename", action="store", default="vectors.h5"
    )
    parser.add_argument(
        "--names_filename", action="store", default="names.txt"
    )
    parser.add_argument("--num_models", action="store", type=int, default=10)
    parser.add_argument("--list_for_dir", action="store", required=True)
    parser.add_argument("--cm2only_file", action="store", default=None)
    parser.add_argument("--tests_file", action="store", required=True)
    args = parser.parse_args()

    main(args)
