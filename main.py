import os
import random
import json
import argparse
import numpy as np
import pickle as p
import matplotlib.pyplot as plt

list_ops = np.array(['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3'])


def encoded_arch_2_api_encoding(encoded_arch):
    list_int_ops = np.array(list(map(int, list(encoded_arch))))
    list_str_ops = list_ops[list_int_ops]
    return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*list_str_ops)


def draw_box_plot(data):
    fig = plt.figure(figsize=(10, 7))
    plt.grid(linestyle="--")
    plt.ylabel("accuracy")
    plt.boxplot(data)
    plt.show()


def main(kwargs):
    database_path = 'data/NASBench201'
    database = p.load(open(database_path + f'/[CIFAR-10]_data.p', 'rb'))

    list_encoded_arch = list(database['200'].keys())
    dict_acc = dict()
    list_acc = []
    for i, encoded_arch in enumerate(list_encoded_arch):
        val_acc = database['200'][encoded_arch]['val_acc']
        dict_acc.update({encoded_arch: val_acc})
        list_acc.append(val_acc)
        # print(f"{encoded_arch_2_api_encoding(encoded_arch)} - {val_acc}")
    # for key in dict_acc.keys():
    #    print(key)

    d = 'cifar10'
    k = 'pgd@Linf'
    m = 'accuracy'
    file = os.path.join('robustness-data/cifar10/', f"{k}_{m}.json")

    pgd_acc = []

    with open(file, "r") as f:
        r = json.load(f)
        ja_acc = r[d][k][m]
        for item in ja_acc:
            pgd_acc.append(ja_acc[item][1])

    draw_box_plot([list_acc, pgd_acc])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
