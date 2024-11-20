import z3_0
import matplotlib.pyplot as plt
import numpy as np


def main():
    lena = plt.imread('lenna.png')
    img = lena - np.min(lena)
    img /= np.max(img)
    img = (img * 255).astype(np.int32)
    img = np.pad(img, 1, 'edge')

    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]], dtype='int32')

    nc_result = z3_0.naive_convolve(img, kernel)

    plt.imshow(nc_result, cmap='binary_r')
    plt.tight_layout()
    plt.savefig('lab3.0.png')


if __name__ == "__main__":
    main()
