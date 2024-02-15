from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def TT_SVD(
    C: torch.Tensor,
    bond_dims: Sequence[int] | None = None,
    check_bond_dims: bool = False,
    return_sv: bool = False,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
    """TT_SVD algorithm

    I. V. Oseledets, Tensor-Train Decomposition, https://epubs.siam.org/doi/10.1137/090752286, Vol. 33, Iss. 5 (2011)

    Args:
        C (torch.Tensor): n-dimensional input tensor
        bond_dims (Sequence[int]): a list of bond dimensions.
                                   If `bond_dims` is None,
                                   `bond_dims` will be automatically calculated
        check_bond_dims (bool): check if `bond_dims` is valid
        return_sv (bool): return singular values

    Returns:
        list[torch.Tensor]: a list of core tensors of TT-decomposition
    """

    dims = C.shape
    n = len(dims)  # n-dimensional tensor

    if bond_dims is None or check_bond_dims:
        # Theorem 2.1
        bond_dims_ = []
        for sep in range(1, n):
            row_dim = dims[:sep].numel()
            col_dim = dims[sep:].numel()
            rank = torch.linalg.matrix_rank(C.reshape(row_dim, col_dim))
            bond_dims_.append(rank)
        if bond_dims is None:
            bond_dims = bond_dims_

    if len(bond_dims) != n - 1:
        raise ValueError(f"{len(bond_dims)=} must be {n - 1}.")
    if check_bond_dims:
        for i, (dim1, dim2) in enumerate(zip(bond_dims, bond_dims_)):
            if dim1 > dim2:
                raise ValueError(f"{i}th dim {dim1} must not be larger than {dim2}.")

    tt_cores = []
    SVs = []
    for i in range(n - 1):
        if i == 0:
            ri_1 = 1
        else:
            ri_1 = bond_dims[i - 1]
        ri = bond_dims[i]
        C = C.reshape(ri_1 * dims[i], dims[i + 1 :].numel())
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        # approximation
        U = U[:, :ri]
        S = S[:ri]
        if return_sv:
            SVs.append(S.detach().clone())
        Vh = Vh[:ri, :]
        tt_cores.append(U.reshape(ri_1, dims[i], ri))
        C = torch.diag(S) @ Vh
    tt_cores.append(C)
    tt_cores[0] = tt_cores[0].reshape(dims[0], bond_dims[0])
    if return_sv:
        return tt_cores, SVs
    return tt_cores


class TTLayer(nn.Module):
    """Tensor-Train Layer

    Alexander Novikov, Dmitrii Podoprikhin, Anton Osokin, Dmitry P. Vetrov, Tensorizing Neural Networks, https://papers.nips.cc/paper_files/paper/2015/hash/6855456e2fe46a9d49d3d3af4f57443d-Abstract.html, (NIPS 2015)

    Args:
        in_shape (Sequence[int]): input shape
        out_shape (Sequence[int]): output shape
        w (torch.Tensor | None): weight tensor
        b (torch.Tensor | None): bias tensor
        tt_shapes (Sequence[Sequence[int]] | None): a list of shapes of core tensors
        bond_dims (Sequence[int] | None): a list of bond dimensions
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        w: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        tt_shapes: Sequence[Sequence[int]] | None = None,
        bond_dims: Sequence[int] | None = None,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

        if w is not None:
            tt_W, B = self._tt_decompose(w, b, in_shape, out_shape, bond_dims)
            self.n_comps = len(tt_W)
            for i, comp in enumerate(tt_W):
                setattr(self, f"tt_W{i}", nn.parameter.Parameter(comp))
            self.B = None
            if B is not None:
                self.B = nn.parameter.Parameter(B)
        elif tt_shapes is not None:
            self.n_comps = len(tt_shapes)
            for i, shape in enumerate(tt_shapes):
                setattr(self, f"tt_W{i}", nn.parameter.Parameter(torch.zeros(shape)))
            self.B = nn.parameter.Parameter(torch.zeros(out_shape))
        else:
            raise ValueError("Both `w` and `` must not be None.")

    def forward(self, x):
        x = x.reshape(x.shape[0], *self.in_shape)
        # The `equation` string specifies the subscripts (letters in [a-zA-Z])
        # "NCD,Aa,aBb,bCc,cD->NAB"
        equation = self._make_equation()
        output = torch.einsum(equation, x, *self.tt_W)
        if self.B is not None:
            output += self.B

        return torch.flatten(output, start_dim=1)

    @property
    def tt_W(self):
        return [getattr(self, f"tt_W{i}") for i in range(self.n_comps)]

    @staticmethod
    def from_linear_layer(
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        layer: nn.Linear,
        bond_dims: Sequence[int] | None = None,
    ):
        return TTLayer(in_shape, out_shape, layer.weight, layer.bias, bond_dims=bond_dims)

    def _tt_decompose(self, w, b, in_shape, out_shape, bond_dims):
        # [1000, 4096] -> [o1, o2, ..., oM, i1, i1, ..., iN]
        W = w.reshape(out_shape + in_shape)
        # [1000] -> [o1, o2, ..., oM]
        B = None
        if b is not None:
            B = b.reshape(out_shape)

        return TT_SVD(W, bond_dims), B

    def _make_equation(self):
        in_shape = self.in_shape
        out_shape = self.out_shape

        small_letters = [chr(uni) for uni in range(ord("a"), ord("z") + 1)]
        capital_letters = [chr(uni) for uni in range(ord("A"), ord("Z") + 1) if uni != ord("N")]
        froms = []
        tos = []
        tts = []

        last_small_letter = None

        # "NCD,Aa,aBb,bCc,cD->NAB"
        for i in range(len(out_shape)):
            L = capital_letters[0]
            l = small_letters[0]
            capital_letters = capital_letters[1:]
            small_letters = small_letters[1:]

            tos.append(L)

            if i == 0:
                tts.append(f"{L}{l}")
            else:
                tts.append(f"{last_small_letter}{L}{l}")
            last_small_letter = l
        for i in range(len(in_shape)):
            L = capital_letters[0]
            l = small_letters[0]
            capital_letters = capital_letters[1:]
            small_letters = small_letters[1:]

            froms.append(L)

            if i >= len(in_shape) - 1:
                tts.append(f"{last_small_letter}{L}")
            else:
                tts.append(f"{last_small_letter}{L}{l}")
            last_small_letter = l
        return f'N{"".join(froms)},{",".join(tts)}->N{"".join(tos)}'
