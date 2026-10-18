import numpy as np
from PIL import Image

def load_image_as_matrix(image_path, grayscale = True):
    
    img = Image.open(image_path)

    if grayscale:
        img = img.convert("L")

    matrix = np.array(img, dtype=np.float64)

    return matrix


def manual_matrix_norm(A):
    m, n = A.shape
    total = 0

    for i in range(m):
        for j in range(n):
            total += A[i, j] ** 2

    return np.sqrt(total)


def frobenius_error(A, A_k):
    m, n = A.shape
    E = np.zeros((m, n), dtype=np.float64)

    for i in range(m):
        for j in range(n):
            E[i, j] = A[i, j] - A_k[i, j]

    norm_E = manual_matrix_norm(E)
    norm_A = manual_matrix_norm(A)

    if norm_A < 1e-12:
        return 0

    return norm_E / norm_A


def compression_ratio(A, A_k):
    m, n = A.shape
    original_elements = m * n

    k = np.linalg.matrix_rank(A_k)

    compressed_elements = k * m + k + k * n

    if compressed_elements == 0:
        return 0

    return original_elements / compressed_elements