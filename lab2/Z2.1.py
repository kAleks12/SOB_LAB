from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import matplotlib.image as img


def process_row(row, img_pad, kernel,):
    kernel_x, kernel_y = np.shape(kernel)
    org_y, _ = np.shape(img_pad)
    arr = [np.sum(img_pad[row:row + kernel_x, y: y + kernel_y] * kernel) for y in range(org_y - kernel_y + 1)]
    return np.array(arr)


def calc_sum(params):
    process_row(*params)


def main():
    lena = img.imread("lenna.png")

    start = datetime.utcnow()
    lena_pad = np.pad(lena, 1, 'edge')
    kernel = np.array([(-1, -1, -1), (-1, 8, -1), (-1, -1, -1)])

    lena_x, lena_y = lena_pad.shape
    kernel_x, kernel_y = kernel.shape

    new_lena = np.zeros(lena.shape)
    raw_params = [(i, lena_pad, kernel) for i in range(lena_y - kernel_y + 1)]
    with ProcessPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(calc_sum, raw_params))
    end = datetime.utcnow()

    for i, row_result in enumerate(result):
        new_lena[i, :] = row_result

    print(f'Processing took {end-start}')
    img.imsave("filtered_lena_mp.png", new_lena, cmap='gray')


if __name__ == '__main__':
    main()
