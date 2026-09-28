# BharatOpt-X Planned Module Interfaces

These C++ header files in `engine/include/bharatopt/` define planned Phase-2 and Phase-3 architectural interfaces for upcoming modular decompositions (including generalized nonlinear formulations, sensitivity, and parameter tuning).

For the current sovereign release, all active, certified mathematical solvers (Two-Phase Revised Simplex, Branch-and-Bound with GNN Oracle, and GPU-accelerated PDLP) are fully integrated and tested in:
- `engine/main.cpp`
- `engine/gpu_kernels.cu` / `engine/gpu_kernels.h`

Both `CMakeLists.txt` and `run_demo.ps1` compile `engine/main.cpp` as the authoritative `bharatopt_engine.exe` binary.
