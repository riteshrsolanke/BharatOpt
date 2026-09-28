# Architecture Overview

```text
                         BHARATOPT
                            │
                    Industrial User
                            │
                    React Web Interface
                            │
                    Model Builder / NLP
                            │
                     Model Validator
                            │
                    Instance Analyzer
                            │
                     Solver-Agnostic
                         Router
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             LP            MILP            QP
             │              │              │
        Simplex/IPM   B&B + Cuts + LP   IPM + Active Set
             │              │              │
             └──────────────┼──────────────┘
                            │
                     Nonlinear / NLP
                            │
                          ADMM
                            │
                     Sparse LA Layer
                            │
                    CPU / CUDA Backend
                      │             │
                    CPU            GPU
                 OpenMP/SIMD   CUDA/cuBLAS/
                               cuSPARSE
                            │
                  Solution Validation
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        Certificates   Sensitivity   Uncertainty
                            │
                     Differentiable
                         Layer
                            │
                  GNN Branching / Tuning
                            │
                    Business Results
```

## Directory Structure
- `engine/`: Core C++23 optimization algorithms (LP, MILP, QP, NLP, LA).
- `apps/`: Standalone executables and CLI tools.
- `services/`: API and microservices for the web interface and tuning.
- `web/`: React frontend for business users.
- `ml/`: GNN branching, parameter tuning, and instance analysis models.
- `tests/`: GoogleTest suite.
- `benchmarks/`: Performance and regression benchmarks.
- `docs/`: Architecture and roadmap documents.
- `data/`: Sample optimization instances and datasets.
- `scripts/`: Build, CI/CD, and utility scripts.
