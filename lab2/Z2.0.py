from datetime import datetime
import numpy as np
import matplotlib.image as img

lena = img.imread("lenna.png")

start = datetime.utcnow()
lena_pad = np.pad(lena, 1, 'edge')
kernel = np.array([(-1, -1, -1), (-1, 8, -1), (-1, -1, -1)])

lena_x, lena_y = lena_pad.shape
kernel_x, kernel_y = kernel.shape

new_lena = np.zeros(lena.shape)

for x in range(lena_x - kernel_x + 1):
    for y in range(lena_y - kernel_y + 1):
        new_lena[x][y] = np.sum(lena_pad[x:x + kernel_x, y: y + kernel_y] * kernel)
end = datetime.utcnow()
print(f'Processing took {end-start}')
img.imsave("filtered_lena.png", new_lena, cmap='gray')
