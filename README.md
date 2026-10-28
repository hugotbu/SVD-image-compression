# Manual SVD Image Compression

This is a personal project buildt from scratch using singular value decomposition, low-rank matrix approximation, and power iteration, with error analysis, to compress images.

The project is not build for optimized image compression, but for implementation of linear algebra and computational mathmatics theory.

## Overview

Singular value decomposition factorizes a matrix $A$ into:

$A = U \Sigma V^T$

Where $U$ contains a orthonormal basis for column space of $A$, $V$ contains a orthonormal basis for row space of $A$, and $\Sigma$ is a diagonal matrix with the singularvalues of $A$. 

SVD is a powerful tool for low rank approximation of large matrixes. The largest k sigularvalues and corresponding orthonormal basis vectors in $U$ and $V$ contain most of the information stored in the matrix. Therefore it is possible to compress a matrix corresponding to an image without loosing significant detail in the photo, and draticaly redusing the amount of data stored.

## A visual representation of SVD image compression

![SVD Summary Result](data/svd_summary_result.png)

## Theory
### SVD
Any real $m \times n$ matrix $A$ can be factorized using Singular Value Decomposition:
$$A = U \Sigma V^T$$

$U$ is an $m \times m$ orthogonal matrix containing the left-singular vectors, the eigenvectors of $A A^T$.
$\Sigma$ is an $m \times n$ diagonal matrix containing the singular values $\sigma_i$ arranged in descending order. 
$V^T$ is the transpose of an $n \times n$ orthogonal matrix containing the right-singular vectors, the eigenvectors of $A^T A$.

### Power iterations

To compute the decomposition completely from scratch without relying on built-in solvers like numpy.linalg svd, this project implements a power iteration algorithm combined with deflation.

To find the dominant eigenvalue and eigenvector of the symmetric matrix $A^T A$, an initial vector $v_0$ is iteratively updated:
$v_{k+1} = \frac{(A^T A) v_k}{\|(A^T A) v_k\|}$

Once a dominant eigenvalue $\lambda$ and its normalized eigenvector $v_i$ are found, their contribution is removed from the matrix through deflation:

$A^T A - \lambda_i v_i v_i^T$
This process is repeated iteratively to extract the subsequent eigenvalues and right-singular vectors.

### Low rank approximation

By only using the chosen rank k amount of eigenvectors and eigenvalues, we obtain a low-rank approximation $A_k$ of the original image matrix:
$$A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T$$

The image can then be stored using fewer parameters while retaining the visual structures.

### Error analysis

To evaluate the quality of the compressed image, I measure the relative error using the matrix norm. For k, the error relative to the original image matrix $A$ is defined as:

Relative Error $= \frac{\|A - A_k\|}{\|A\|}$

### The Eckart-Young-Mirsky Theorem
According to the Eckart-Young-Mirsky Theorem the optimal low rank error for a given rank k is bounded by the magnitude of the discarded singular values:

$\|A - A_k\| = \sqrt{\sum_{i=k+1}^{\min(m,n)} \sigma_i^2}$

## Practical observation
In an implementation of power iterations, numerical behavior deviate from pure theory at higher ranks. For example each deflation step introduces minor inaccuracies. These inaccuracies propagate and end up disturbing lower value eigenvectors. 
However this does not infulence the efficacy of the model since any real compression made whould have a k value lower then any significant numerical errors. 

## Project Structure


data/
- test_image(1, 2, 3).jpg
- svd_summary_result.png  # Generated summary plot

src/
-  svd_compressor.py       # Manual SVD & power iteration logic
- utils.py                # Matrix operations and helper functions

generate_svd_image.py       # Script to run compression

requirements.txt            # Project dependencies