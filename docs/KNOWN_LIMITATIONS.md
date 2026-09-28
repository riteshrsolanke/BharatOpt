# KNOWN LIMITATIONS

## Architectural
- GNN Inference may timeout if the external Python Sidecar (`ml/sidecar.py`) is unavailable. The fallback mechanism (classical branching) is robust but loses the ML speedup.
- CUDA backend requires an external NVIDIA Toolkit and `USE_CUDA` defined in CMake. CPU fallback is currently enforced.

## Mathematical
- NLP models without explicit gradients rely on structural automatic differentiation; highly degenerate paths (e.g. nested non-smooth functions) are flagged as `NUMERICAL_FAILURE`.
- Does not claim global optimality for non-convex NLPs. The status strictly enforces `LOCAL_SOLUTION`.
