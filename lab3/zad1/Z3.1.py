import z3_1
import time
import matplotlib.pyplot as plt
import numpy as np


def normalize_pad(vector):
    nor = vector - np.min(vector)
    nor /= np.max(nor)
    nor = (nor * 255).astype(np.int32)
    nor = np.pad(nor, 1, 'edge')
    return nor


def main():
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    lena = plt.imread('lenna.png')
    img = normalize_pad(lena)

    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]], dtype='int32')

    start = time.time()
    nc_result = z3_1.naive_convolve(img, kernel)
    interval_n = (time.time() - start)
    print("Naive convolve: %.6f sec" % interval_n)

    start = time.time()
    sc_result = z3_1.speed_convolve(img, kernel)
    interval_s = (time.time() - start)
    print("Speed convolve: %.6f sec" % interval_s)

    ax[0].imshow(img, cmap='binary_r')
    ax[0].set_title('Original')

    ax[1].imshow(sc_result, cmap='binary_r')
    ax[1].set_title('Speed convolve')

    ax[2].imshow(nc_result, cmap='binary_r')
    ax[2].set_title('Naive convolve')

    print(f'Speed convolve is {round((interval_n / interval_s), 2)} times faster than naive convolve')

    plt.tight_layout()
    plt.savefig('lab3.1.png')


if __name__ == '__main__':
    main()
