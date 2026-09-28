# BENCHMARK RESULTS

## Phase 14 Ablation
- **Classical Branching:** 124.5s average solve time, 4,500 nodes explored
- **GNN-Guided Branching:** 98.2s average solve time, 2,100 nodes explored
*Improvement is structurally captured in internal diagnostics; note that GNN requires graph build overhead.*

## CPU vs GPU SpMV Parity
- Density > 5%: GPU provides up to 4.2x speedup on matrices exceeding 10,000x10,000.
- Density < 1%: CPU retains 1.5x advantage due to PCIe H2D/D2H transfer bounds.
