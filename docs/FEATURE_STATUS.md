# FEATURE STATUS

## STABLE
* **LP Solver**: Two-Phase Simplex, Revised Simplex, Interior Point Method
* **MILP Engine**: Branch-and-Bound, LP Relaxation, Cuts Interfaces
* **Model Router**: Automatic feature detection and solver fallback
* **Parameter Tuner**: Auto-tuning with historical random seeds
* **Certificates**: Primal/Dual Verification for LPs
* **GPU Backend**: Basic SpMV architecture and CPU fallbacks

## EXPERIMENTAL
* **NLP Differentiable Layer**: Explicit expression DAG and automatic derivatives
* **Convex QP & ADMM**: KKT system generation and dual splitting updates
* **GNN Guided Branching**: External Python-sidecar connectivity with PyTorch Geometric
* **Uncertainty Optimization**: Scenario trees and interval robust optimization

## UNSUPPORTED
* **MIQP**: Mixed-Integer Quadratic Programming is structurally identified but fully certified solvers are unavailable without fallback.
* **Non-Convex Global Optimization**: The system strictly prevents calling local optimal solutions as globally optimal without certificates.
