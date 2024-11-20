import numpy as np
cimport numpy as np
cimport cython

np.import_array()


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def naive_convolve(np.ndarray[np.int32_t, ndim=2] image,
                  np.ndarray[np.int32_t, ndim=2] kernel):
    cdef int image_height = image.shape[0]
    cdef int image_width = image.shape[1]
    cdef int kernel_height = kernel.shape[0]
    cdef int kernel_width = kernel.shape[1]
    cdef int output_height = image_height - kernel_height + 1
    cdef int output_width = image_width - kernel_width + 1

    cdef np.ndarray output = np.zeros((output_height, output_width), dtype=np.int32)

    cdef int i, j, m, n
    cdef int sum_value
    for i in range(output_height):
        for j in range(output_width):
            sum_value = 0
            for m in range(kernel_height):
                for n in range(kernel_width):
                    sum_value += image[i + m, j + n] * kernel[m, n]
            output[i, j] = sum_value

    return output
