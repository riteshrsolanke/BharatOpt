# BharatOpt-X: Sovereign GPU-Accelerated Optimization Engine
**Smart India Hackathon 2026 | Problem Statement: SIH26119 | Team: NEXORA**

![BharatOpt-X](web/assets/logo.png) <!-- Optional if logo exists -->

BharatOpt-X is a production-grade, mathematically robust, and GPU-accelerated optimization engine built entirely from scratch in C++23. It serves as a 100% sovereign, on-premises alternative to commercial foreign solvers like IBM CPLEX, Gurobi, and FICO Xpress for India's strategic industrial sectors (refining, logistics, and power grids).

## 🚀 One-Click Quickstart (For Evaluators)

We have provided a fully automated startup script for Windows that will install dependencies, launch the AI sidecar, and open the interactive Web Dashboard.

1. Double-click **`START_SIH_DEMO.bat`**
2. The script will open your default browser to **http://localhost:8000**
3. Select a benchmark model from the sidebar (e.g., *1M Megascale Logistics* or *LP Crude Blending*) and click **Solve**.

*(Alternatively, you can run `python services/api/main.py` directly).*

---

## 🌟 Key Innovations

1. **Massive GPU Acceleration (cuSPARSE)**
   Unlike traditional CPU-bound solvers, BharatOpt-X streams Compressed Sparse Row (CSR) matrices directly to NVIDIA GPUs, effortlessly solving models with **1,000,000+ constraints** without memory crashes.

2. **AI-Guided Branching (PyTorch GraphSAGE)**
   Solving Mixed-Integer (MILP) problems is NP-Hard. We pioneered a decoupled "sidecar" architecture. The C++ math engine passes the problem state to a PyTorch GNN, which predicts the absolute best variable to branch on, speeding up solve times by ~44%.

3. **Sahayak: Agentic NLP Dashboard**
   Planners do not read raw `.mps` math files. Our interactive dashboard features "Sahayak", an AI assistant that translates complex shadow prices and dual variables into plain English and Indian Rupees (₹).

4. **100% Data Sovereignty**
   Completely air-gapped and container-ready. Critical petroleum and supply chain data never leaves the facility. Zero recurring licensing fees.

---

## 📂 Architecture & Directory Structure
- `engine/` - Core C++23 Mathematical solvers (LP, MILP, Sparse, Routing, NLP).
- `services/api/main.py` - FastAPI Backend (handles AI models, routes C++ requests, serves Web UI).
- `web/` - React/Tailwind/HTML Dashboard (Interactive Polytope Explorer).
- `data/` & `benchmarks/` - Canonical test datasets (Netlib SC50B, MIPLIB Stein9).
- `third_party/` - Eigen 3.4.0 (Exact LU Factorization and sparse math).

## 📊 Proof of Work & Benchmarks
Please read the included **`BharatOpt-X_Final_Report.pdf`** (or `.html` / `.md`) located in the root folder for a comprehensive breakdown of our technical benchmarks, exactness proofs (KKT Residuals), and financial viability.

---
**Made with ❤️ in India by Team NEXORA**
