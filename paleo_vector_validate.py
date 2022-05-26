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

    # in the dataset, 008 and 013 are swapped
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
            "063",
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

    validation_signs_file = open(
        "./data/paleo-vector-info/validation_signs.txt", "r"
    )
    validation_signs = [line[:-1] for line in validation_signs_file]
    validation_signs_file.close()

    # zero metrics
    accuracy_at_one_correct = 0
    accuracy_at_two_correct = 0
    accuracy_at_three_correct = 0
    accuracy_at_five_correct = 0

    for validation_sign in validation_signs:

        # exclude current sign from estimating the vector
        list_signs_for_dir = list(
            set(validation_signs) - set([validation_sign])
        )

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

        validation_sign_vectors = [
            v[1] for v in sign_subcorpora_dict["CM1"][validation_sign]
        ]

        cm23_centroids = np.asarray(sign_centroid_list_cm23)

        # calculate the sum between a centroid for a cm1 sign and the mean cm23/1 distance
        cm1_centroid = np.mean(validation_sign_vectors, axis=0)

        cm2_point = np.asarray(cm1_centroid + mean_difference)

        cm2_point = np.expand_dims(cm2_point, 0)

        # get the closest cm2only signs
        cm2point_centroids_distances = cdist(
            cm2_point, cm23_centroids, metric="cosine"
        )
        closest_sign_idx = np.argsort(cm2point_centroids_distances[0], axis=0)
        closest_sign = np.asarray(sign_centroid_names_cm23)[closest_sign_idx]

        distances = cm2point_centroids_distances[0, closest_sign_idx]

        sign_pos = np.where(closest_sign == validation_sign)[0][0]

        if sign_pos == 0:
            accuracy_at_one_correct += 1

        if sign_pos <= 1:
            accuracy_at_two_correct += 1

        if sign_pos <= 2:
            accuracy_at_three_correct += 1

        if sign_pos <= 4:
            accuracy_at_five_correct += 1

        print("{} is closest to {}".format(validation_sign, closest_sign[:50]))
        print("numerical distances: {}".format(distances[:10]))
        print("{} is number {}".format(validation_sign, sign_pos + 1))
        print("")

    print(
        "correct@1 {} @2 {} @3 {} @5 {} total {}".format(
            accuracy_at_one_correct,
            accuracy_at_two_correct,
            accuracy_at_three_correct,
            accuracy_at_five_correct,
            len(validation_signs),
        )
    )

    print(
        "top-one accuracy is {}".format(
            float(accuracy_at_one_correct) / len(validation_signs)
        )
    )
    print(
        "top-two accuracy is {}".format(
            float(accuracy_at_two_correct) / len(validation_signs)
        )
    )
    print(
        "top-three accuracy is {}".format(
            float(accuracy_at_three_correct) / len(validation_signs)
        )
    )

    print(
        "top-five accuracy is {}".format(
            float(accuracy_at_five_correct) / len(validation_signs)
        )
    )

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
    args = parser.parse_args()

    main(args)
