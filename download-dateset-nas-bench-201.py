from nas_201_api import NASBench201API as API


def main():
    path_to_meta_nas_bench_file = "nas-bench-201/NAS-Bench-201-v1_1-096897.pth"
    api = API(path_to_meta_nas_bench_file, verbose=False)
    num = len(api)

    for i, arch_str in enumerate(api):
        print('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))


if __name__ == "__main__":
    main()
