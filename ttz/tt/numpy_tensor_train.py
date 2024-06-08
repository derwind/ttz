from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def TT_SVD_NumPy(
    C: np.ndarray, r: Sequence[int] | None = None, check_r: bool = False
) -> list[np.ndarray]:
    """TT_SVD algorithm

    I. V. Oseledets, Tensor-Train Decomposition, https://epubs.siam.org/doi/10.1137/090752286

    Args:
        C (np.ndarray): n-dimensional input tensor
        r (Sequence[int]): a list of bond dimensions.
                           If `r` is None, `r` will be automatically calculated
        check_r (bool): check if `r` is valid

    Returns:
        list[np.ndarray]: a list of core tensors of TT-decomposition
    """

    dims = C.shape
    n = len(dims)  # n-dimensional tensor

    if r is None or check_r:
        # Theorem 2.1
        r_ = []
        for sep in range(1, n):
            row_dim = np.prod(dims[:sep])
            col_dim = np.prod(dims[sep:])
            rank = np.linalg.matrix_rank(C.reshape(row_dim, col_dim))
            r_.append(rank)
        if r is None:
            r = r_

    if len(r) != n - 1:
        raise ValueError(f"{len(r)=} must be {n - 1}.")
    if check_r:
        for i, (r1, r2) in enumerate(zip(r, r_, strict=True)):
            if r1 > r2:
                raise ValueError(f"{i}th dim {r1} must not be larger than {r2}.")

    # Algorithm 1
    tt_cores = []
    for i in range(n - 1):
        if i == 0:
            ri_1 = 1
        else:
            ri_1 = r[i - 1]
        ri = r[i]
        C = C.reshape(ri_1 * dims[i], np.prod(dims[i + 1 :]))
        U, S, Vh = np.linalg.svd(C, full_matrices=False)
        # approximation
        U = U[:, :ri]
        S = S[:ri]
        Vh = Vh[:ri, :]
        tt_cores.append(U.reshape(ri_1, dims[i], ri))
        C = np.diag(S) @ Vh
    tt_cores.append(C)
    tt_cores[0] = tt_cores[0].reshape(dims[0], r[0])
    return tt_cores
