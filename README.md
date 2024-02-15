# Tensor-Trains

A Toolbox for Tensor-Train and MPS (Matrix Product State).

- `TT_SVD` and `TTLayer` which allows low rank approximation of `nn.Linear`.
- `TT_SVD_Vidal` and `MPS` which allows MPS simulation of quantum circuits.

## Installation

```bash
pip install .
```

## References

- [Guifré Vidal, Efficient Classical Simulation of Slightly Entangled Quantum Computations](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.91.147902), Phys. Rev. Lett. 91, 147902 – Published 1 October 2003
- [I. V. Oseledets, Tensor-Train Decomposition](https://epubs.siam.org/doi/10.1137/090752286), Vol. 33, Iss. 5 (2011)
- [Alexander Novikov, Dmitrii Podoprikhin, Anton Osokin, Dmitry P. Vetrov, Tensorizing Neural Networks](https://papers.nips.cc/paper_files/paper/2015/hash/6855456e2fe46a9d49d3d3af4f57443d-Abstract.html), (NIPS 2015)
