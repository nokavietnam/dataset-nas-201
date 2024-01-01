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


# def decode_arch(arch):
#     split_result = arch.split('|')
#     filtered_result = [item for item in split_result if item.strip() != "" and item != '+']
#     a = []
#     for item in filtered_result:
#         a.append(item.split('~')[0])
#
#     converted_list = [str(np.where(list_ops == item)[0][0]) for item in a]
#     result = ''.join(converted_list)
#     return result

# print(result)

def decode_arch(arch):
    return ''.join(
        str(np.where(list_ops == item.split('~')[0])[0][0]) for item in filter(None, arch.split('|')) if item != '+')


def main():
    d = 'cifar10'
    k = 'pgd@Linf'
    m = 'accuracy'
    file = os.path.join('robustness-data/cifar10/', f"{k}_{m}.json")
    file_meta = os.path.join('robustness-data', 'meta.json')

    pgd_acc = []

    f_meta = open(file_meta, "r")
    meta = json.load(f_meta)
    f_meta.close()

    arch_dict = dict()

    for item in meta['ids']:
        arch_dict.update({decode_arch(meta["ids"][item]["nb201-string"]): item})

    #print(arch_dict.keys())

    # string_test = '|nor_conv_3x3~0|+|nor_conv_3x3~0|skip_connect~1|+|skip_connect~0|nor_conv_3x3~1|none~2|'
    # print(decode_arch_1(string_test))

    with open(file, "r") as f:
        r = json.load(f)
        ja_acc = r[d][k][m]
        for item in ja_acc:
            pgd_acc.append(ja_acc[item][1])
            print(item)


if __name__ == '__main__':
    main()
