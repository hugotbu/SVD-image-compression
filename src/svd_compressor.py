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

def qr_decomposition(A):
    n = A.shape[0]
    Q = np.zeros((n, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        v = A[:, j].copy()

        for i in range(j):
            dot = 0
            for k in range(n):
                dot += Q[k, i] * A[k, j]
            R[i, j] = dot

            for k in range(n):
                v[k] -= R[i, j] * Q[k, i]

        norm_v = np.sqrt(sum(v_k ** 2 for v_k in v))
        R[j, j] = norm_v

        if norm_v > 1e-12:
            for k in range(n):
                Q[k, j] = v[k] / norm_v

    return Q, R

def eigenvalues(A_T_multiply_A, max_iterations=1000):
    A_k = A_T_multiply_A.copy()

    for _ in range(max_iterations):
        Q, R = qr_decomposition(A_k)
        A_k = matrix_multiply(R, Q)

    n = A_k.shape[0]
    eigenvalues = np.array([A_k[i, i] for i in range(n)])

    eigenvalues = np.sort(eigenvalues)[::-1]

    return eigenvalues

def manual_norm(v):
    total = 0
    for value in v:
        total += value**2
    return np.sqrt(total)


def compute_eigenvectors_power_iteration(A_T_multiply_A, eigenvalues, max_iterations = 100):
    n = A_T_multiply_A.shape[0]
    V = np.zeros((n, n), dtype=np.float64)
    A_T_muliply_A_current = A_T_multiply_A.copy().astype(np.float64)

    for i in range(n):
        lambda = eigenvalues[i]

        if lambda < 1e-10:
            v = np.zeros(n, dtype=np.float64)
            v[i] = 1
            V[:, i] = v
            continue

        v = np.ones(n, dtype=np.float64)
        v = v / manual_norm(v)

        for _ in range(max_iterations):
            v_next = matrix_vector_multiply(A_T_muliply_A_current, v)
            norm = manual_norm(v_next)

            if norm < 1e-12:
                break

            v = v_next / norm

        V[:, i] = v

        for r in range(n):
            for c in range(n):
                A_T_muliply_A_current[r, c] -= lambda * v[r] * v[c]

    return V

