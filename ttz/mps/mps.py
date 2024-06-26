from __future__ import annotations

import numpy as np
from opt_einsum import contract


class MPS:
    def __init__(
        self,
        num_qubits: int,
        gammas: list[np.ndarray] | None = None,
        lambdas: list[np.ndarray] | None = None,
    ):
        self.num_qubits = num_qubits
        if gammas is not None and lambdas is not None:
            self._gammas = [gamma.copy() for gamma in gammas]
            self._lambdas = [lambda_.copy() for lambda_ in gammas]
        else:
            self._gammas, self._lambdas = zeros_mps(num_qubits)

    def __str__(self):
        return f"MPS(num_qubits={self.num_qubits})"

    def __repr__(self):
        return f"MPS(num_qubits={self.num_qubits})"

    def amplitude(self, bitstring: str) -> complex:
        if len(bitstring) != len(self._gammas):
            raise ValueError(f"bitstring must have length {len(self._gammas)}.")

        bits = [int(bit) for bit in bitstring]
        operands = [self._gammas[0][bits[0], :]]
        for lambda_, gamma, bit in zip(self._lambdas[:-1], self._gammas[1:-1], bits[1:-1], strict=True):
            operands.append(lambda_)
            operands.append(gamma[:, bit, :])
        operands.append(self._lambdas[-1])
        operands.append(self._gammas[-1][:, bits[-1]])

        expr = remove_outer_indices(make_expr(self.num_qubits))

        return contract(expr, *operands)

    def state_vector(self) -> np.ndarray:
        return np.array(
            [self.amplitude(f"{i:0{self.num_qubits}b}"[::-1]) for i in range(2**self.num_qubits)]
        )

    def x(self, qubit: int) -> None:
        apply_X(self._gammas, qubit)

    def y(self, qubit: int) -> None:
        apply_Y(self._gammas, qubit)

    def z(self, qubit: int) -> None:
        apply_Z(self._gammas, qubit)

    def h(self, qubit: int) -> None:
        apply_H(self._gammas, qubit)

    def s(self, qubit: int) -> None:
        apply_S(self._gammas, qubit)

    def sdg(self, qubit: int) -> None:
        apply_Sdag(self._gammas, qubit)

    def t(self, qubit: int) -> None:
        apply_T(self._gammas, qubit)

    def tdg(self, qubit: int) -> None:
        apply_Tdag(self._gammas, qubit)

    def rx(self, theta: float, qubit: int) -> None:
        apply_Rx(self._gammas, theta, qubit)

    def ry(self, theta: float, qubit: int) -> None:
        apply_Ry(self._gammas, theta, qubit)

    def rz(self, theta: float, qubit: int) -> None:
        apply_Rz(self._gammas, theta, qubit)

    def cx(self, control: int, target: int, auto_swap: bool = True) -> None:
        apply_CX(self._gammas, self._lambdas, control, target, auto_swap)

    def cy(self, control: int, target: int, auto_swap: bool = True) -> None:
        apply_CY(self._gammas, self._lambdas, control, target, auto_swap)

    def cz(self, control: int, target: int, auto_swap: bool = True) -> None:
        apply_CZ(self._gammas, self._lambdas, control, target, auto_swap)

    def swap(self, qubit1: int, qubit2: int) -> None:
        apply_SWAP(self._gammas, self._lambdas, qubit1, qubit2)

    def ccx(self, i: int, j: int, k: int, auto_swap: bool = True) -> None:
        apply_CCX(self._gammas, self._lambdas, i, j, k, auto_swap)


# https://github.com/Qiskit/qiskit-aer/blob/0.13.1/src/simulators/matrix_product_state/matrix_product_state_internal.cpp#L1754-L1819
def TT_SVD_Vidal(
    C: np.ndarray, num_qubits: int | None = None, dims: tuple[int] = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """TT_SVD Vidal algorithm

    Guifré Vidal, Efficient Classical Simulation of Slightly Entangled Quantum Computations,
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.91.147902, Phys. Rev. Lett. 91, 147902 – Published 1 October 2003

    Args:
        C (np.ndarray): n-dimensional input tensor
        num_qubits (int | None): number of qubits

    Returns:
        list[np.ndarray]: Γs
        list[np.ndarray]: Λs
    """

    gammas = []
    lambdas = []

    if num_qubits is None:
        num_qubits = int(np.log2(np.prod(C.shape)))
    if num_qubits < 2:
        raise ValueError(f"num_qubits ({num_qubits}) must be larger than one.")

    if dims is None:
        dims = (2,) * num_qubits
    C = C.reshape(dims)

    r = []
    for sep in range(1, num_qubits):
        row_dim = np.prod(dims[:sep])
        col_dim = np.prod(dims[sep:])
        rank = np.linalg.matrix_rank(C.reshape(row_dim, col_dim))
        r.append(rank)

    for i in range(num_qubits - 1):
        if i == 0:
            ri_1 = 1
        else:
            ri_1 = r[i - 1]
        ri = r[i]
        C = C.reshape(ri_1 * dims[i], np.prod(dims[i + 1 :]))
        U, S, Vh = np.linalg.svd(C, full_matrices=False)
        U = U[:, :ri]
        S = S[:ri]
        Vh = Vh[:ri, :]
        U = U.reshape(ri_1, dims[i], ri)
        if i > 0:
            for a in range(U.shape[0]):
                U[a, :, :] /= lambdas[-1][a]
        gammas.append(U)
        lambdas.append(S)
        C = np.diag(S) @ Vh
    gammas.append(Vh)
    gammas[0] = gammas[0].reshape(dims[0], r[0])
    return gammas, lambdas


def zeros_mps(num_qubits: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    gammas = [np.array([[1.0], [0.0]], dtype=complex)]
    lambdas = [np.array([1.0], dtype=complex)]
    for _ in range(1, num_qubits - 1):
        gammas.append(np.array([[[1.0], [0.0]]], dtype=complex))
        lambdas.append(np.array([1.0], dtype=complex))
    gammas.append(np.array([[1.0, 0.0]], dtype=complex))
    return gammas, lambdas


# can allow up to 50 qubits
alphabets = [
    chr(uni) for uni in list(range(ord("A"), ord("Z") + 1)) + list(range(ord("a"), ord("z") + 1))
]
greek_alphabets = [
    chr(uni) for uni in list(range(ord("Α"), ord("Ω") + 1)) + list(range(ord("α"), ord("ω") + 1))
]


def make_expr(n_qubits: int) -> str:
    outer_indices = alphabets
    inner_indices = greek_alphabets

    expr = []
    prev_inner = ""
    for i, (outer_i, inner_i) in enumerate(zip(outer_indices, inner_indices, strict=True)):
        if i + 1 < n_qubits:
            expr.extend([f"{prev_inner}{outer_i}{inner_i}", inner_i])
            prev_inner = inner_i
        else:
            expr.extend([f"{prev_inner}{outer_i}"])
            break
    return ",".join(expr) + "->" + "".join(outer_indices[:n_qubits])


def remove_outer_indices(expr: str) -> str:
    new_expr = []
    for v in expr.split("->")[0].split(","):
        for c in alphabets:
            v = v.replace(c, "")  # noqa: PLW2901
        new_expr.append(v)
    return ",".join(new_expr) + "->"


def apply_one_qubit_gate(gammas: list[np.ndarray], U: np.ndarray, qubit: int) -> None:
    gamma = gammas[qubit]
    if qubit == 0:  # expr: ia
        gamma = np.einsum("ij,ja->ia", U, gamma)
    elif qubit + 1 >= len(gammas):  # expr: ai
        gamma = np.einsum("ij,aj->ai", U, gamma)
    else:  # expr: aib
        gamma = np.einsum("ij,ajb->aib", U, gamma)
    gammas[qubit][:] = gamma


PauliX = np.array([[0, 1], [1, 0]], dtype=complex)
PauliY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PauliZ = np.array([[1, 0], [0, -1]], dtype=complex)
Hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
Sdag = np.array([[1, 0], [0, -1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
Tdag = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)


def Rx(theta: float):
    return np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2) * 1j],
            [-np.sin(theta / 2) * 1j, np.cos(theta / 2)],
        ],
        dtype=complex,
    )


def Ry(theta: float):
    return np.array(
        [[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]],
        dtype=complex,
    )


def Rz(theta: float):
    return np.array(
        [
            [np.cos(theta / 2) - np.sin(theta / 2) * 1j, 0],
            [0, np.cos(theta / 2) + np.sin(theta / 2) * 1j],
        ],
        dtype=complex,
    )


def apply_X(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, PauliX, qubit)


def apply_Y(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, PauliY, qubit)


def apply_Z(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, PauliZ, qubit)


def apply_H(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, Hadamard, qubit)


def apply_S(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, S, qubit)


def apply_Sdag(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, Sdag, qubit)


def apply_T(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, T, qubit)


def apply_Tdag(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, Tdag, qubit)


def apply_Rx(gammas: list[np.ndarray], theta: float, qubit: int) -> None:
    apply_one_qubit_gate(gammas, Rx(theta), qubit)


def apply_Ry(gammas: list[np.ndarray], theta: float, qubit: int) -> None:
    apply_one_qubit_gate(gammas, Ry(theta), qubit)


def apply_Rz(gammas: list[np.ndarray], theta: float, qubit: int) -> None:
    apply_one_qubit_gate(gammas, Rz(theta), qubit)


def swap_ij_of_controlled_gate(mat: np.ndarray) -> np.ndarray:
    mat2 = np.zeros_like(mat)
    mat2[0, 0] = mat2[2, 2] = 1
    mat2[1, 1] = mat[2, 2]
    mat2[1, 3] = mat[2, 3]
    mat2[3, 1] = mat[3, 2]
    mat2[3, 3] = mat[3, 3]
    return mat2


def apply_two_qubits_gate(
    gammas: list[np.ndarray],
    lambdas: list[np.ndarray],
    U: np.ndarray,
    control: int,
    target: int,
    auto_swap: bool = False,
) -> None:
    i, j = control, target

    if i == j:
        raise ValueError("control and target must be different.")

    auto_swap_list: list[tuple[int, int]] = []
    if i + 1 != j and j + 1 != i:
        if not auto_swap:
            raise ValueError("only adjuscent qubits are supported.")

        if i > j:  # move control i -> i - 1
            for k in range(i, j + 1, -1):
                apply_two_qubits_gate(gammas, lambdas, SWAP, k - 1, k)
                auto_swap_list.append((k - 1, k))
            i = j + 1
        else:  # move control i -> i + 1
            for k in range(i, j - 1):
                apply_two_qubits_gate(gammas, lambdas, SWAP, k, k + 1)
                auto_swap_list.append((k, k + 1))
            i = j - 1

    reverse = False
    if j < i:
        i, j = j, i
        reverse = True

    if i == 0:
        if len(gammas) == 2:
            expr = "IJAB,Aa,a,aB->IJ"
        else:
            expr = "IJAB,Aa,a,aBb->IJb"
        left_dim = gammas[i].shape[0]
    elif j + 1 < len(gammas):
        expr = "IJAB,aAb,b,bBc->aIJc"
        left_dim = gammas[i].shape[0] * gammas[i].shape[1]
    else:
        expr = "IJAB,aAb,b,bB->aIJ"
        left_dim = gammas[i].shape[0] * gammas[i].shape[1]

    if not reverse:
        U = U.reshape(2, 2, 2, 2)
        C = np.einsum(expr, U, gammas[i], lambdas[i], gammas[j])
    else:
        U = swap_ij_of_controlled_gate(U).reshape(2, 2, 2, 2)
        C = np.einsum(expr, U, gammas[i], lambdas[i], gammas[j])

    updated_gammas, updated_lambdas = TT_SVD_Vidal(C, num_qubits=2, dims=(left_dim, -1))
    if i > 0:
        updated_gammas[0] = updated_gammas[0].reshape(-1, 2, len(updated_lambdas[0]))
    if j + 1 < len(gammas):
        updated_gammas[1] = updated_gammas[1].reshape(len(updated_lambdas[0]), 2, -1)
    else:
        updated_gammas[1] = updated_gammas[1].reshape(-1, 2)
    gammas[i] = updated_gammas[0]
    lambdas[i] = updated_lambdas[0]
    gammas[i + 1] = updated_gammas[1]

    if auto_swap_list is not None:
        for k, m in auto_swap_list[::-1]:
            apply_two_qubits_gate(gammas, lambdas, SWAP, k, m)


CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


CY = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex)


CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)


SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def apply_CX(
    gammas: list[np.ndarray], lambdas: list[np.ndarray], i: int, j: int, auto_swap: bool = True
) -> None:
    apply_two_qubits_gate(gammas, lambdas, CX, i, j, auto_swap)


def apply_CY(
    gammas: list[np.ndarray], lambdas: list[np.ndarray], i: int, j: int, auto_swap: bool = True
) -> None:
    apply_two_qubits_gate(gammas, lambdas, CY, i, j, auto_swap)


def apply_CZ(
    gammas: list[np.ndarray], lambdas: list[np.ndarray], i: int, j: int, auto_swap: bool = True
) -> None:
    apply_two_qubits_gate(gammas, lambdas, CZ, i, j, auto_swap)


def apply_SWAP(
    gammas: list[np.ndarray], lambdas: list[np.ndarray], i: int, j: int, auto_swap: bool = True
) -> None:
    apply_two_qubits_gate(gammas, lambdas, SWAP, i, j)


def apply_CCX(
    gammas: list[np.ndarray],
    lambdas: list[np.ndarray],
    i: int,
    j: int,
    k: int,
    auto_swap: bool = True,
) -> None:
    apply_H(gammas, k)
    apply_CX(gammas, lambdas, j, k, auto_swap)
    apply_Tdag(gammas, k)
    apply_CX(gammas, lambdas, i, k, auto_swap)
    apply_T(gammas, k)
    apply_CX(gammas, lambdas, j, k, auto_swap)
    apply_Tdag(gammas, j)
    apply_Tdag(gammas, k)
    apply_CX(gammas, lambdas, i, k, auto_swap)
    apply_CX(gammas, lambdas, i, j, auto_swap)
    apply_T(gammas, i)
    apply_Tdag(gammas, j)
    apply_T(gammas, k)
    apply_CX(gammas, lambdas, i, j, auto_swap)
    apply_S(gammas, j)
    apply_H(gammas, k)
