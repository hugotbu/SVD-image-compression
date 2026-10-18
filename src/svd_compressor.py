import numpy as np


def matrix_multiply(A, B):
    m, n = A.shape
    q, p = B.shape
    C = np.zeros((m, p), dtype=np.float64)

    for i in range(m):
        for j in range(p):
            total = 0
            for k in range(n):
                total += A[i, k] * B[k, j]
            C[i, j] = total
    return C

def matrix_vector_multiply(A, v):
    m, n = A.shape
    y = np.zeros(m, dtype=np.float64)

    for i in range(m):
        total = 0
        for j in range(n):
            total += A[i, j] * v[j]
        y[i] = total
    return y