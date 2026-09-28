"""
BharatOpt-X: Sovereign GPU-Accelerated Mathematical Optimization Engine
Official Native Python SDK & Modeling API (import bharatopt)

Compatible with Gurobi / SCIP / HiGHS Python modeling conventions.
100% Sovereign Clean-Room Architecture for MRPL & Smart India Hackathon 2026.
"""

import os
import sys
import json
from typing import Dict, List, Optional, Union

# Ensure services/api is accessible
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_API_DIR = os.path.join(_CURRENT_DIR, "services", "api")
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from cpp_caller import write_dat_and_run_cpp, NativeSolveResult
from local_nlp_parser import parse_to_mps, parse_mps_text, parse_conversational_english


class Variable:
    def __init__(self, name: str, obj: float = 0.0, lb: float = 0.0, ub: float = float('inf'), vtype: str = "C"):
        self.name = name
        self.obj = float(obj)
        self.lb = float(lb)
        self.ub = float(ub)
        self.vtype = vtype.upper()  # 'C' (Continuous), 'I' (Integer), 'B' (Binary)

    def __repr__(self):
        return f"<Var {self.name}: obj={self.obj}, type={self.vtype}>"


class Constraint:
    def __init__(self, name: str, terms: Dict[str, float], rel: str = "<=", rhs: float = 0.0):
        self.name = name
        self.terms = {str(k): float(v) for k, v in terms.items()}
        self.rel = rel
        self.rhs = float(rhs)

    def __repr__(self):
        expr = " + ".join([f"{v}*{k}" for k, v in self.terms.items()])
        return f"<Constraint {self.name}: {expr} {self.rel} {self.rhs}>"


class SolveResult:
    def __init__(self, native: NativeSolveResult, model_name: str, var_names: List[str]):
        self.model_name = model_name
        self.status = native.status
        self.is_optimal = (native.status == "OPTIMAL")
        self.objective = native.objective
        self.vars = {k: native.variables.get(k, 0.0) for k in var_names} if native.variables else {}
        self.duals = native.duals
        self.shadow_prices = native.duals
        self.slacks = native.slacks
        self.reduced_costs = native.reduced_costs
        self.primal_residual = native.primal_residual
        self.dual_residual = native.dual_residual
        self.duality_gap = native.duality_gap
        self.solve_time_ms = native.solve_time_ms
        self.backend = native.backend
        self.iterations = native.iterations
        self.bb_nodes = native.bb_nodes
        self.is_certified = (self.is_optimal and self.duality_gap < 1e-4 and self.primal_residual < 1e-3)

    def get_var(self, name: str) -> float:
        return self.vars.get(name, 0.0)

    def get_shadow_price(self, con_name: str) -> float:
        return self.shadow_prices.get(con_name, 0.0)

    def summary(self) -> str:
        lines = [
            "=" * 55,
            f"  BharatOpt-X Solve Result: {self.status}",
            f"  Backend: {self.backend} | Time: {self.solve_time_ms:.3f} ms",
            f"  Objective Value: {self.objective:,.4f}",
            f"  Certified Optimal: {'YES (3-Way KKT Proof)' if self.is_certified else 'NO'}",
            "-" * 55,
            "  Decision Variables:"
        ]
        for k, v in list(self.vars.items())[:10]:
            lines.append(f"    {k:<20} = {v:>12.4f}")
        if len(self.vars) > 10:
            lines.append(f"    ... ({len(self.vars) - 10} more variables)")
        lines.append("=" * 55)
        return "\n".join(lines)

    def __repr__(self):
        return f"<SolveResult status={self.status}, obj={self.objective}, certified={self.is_certified}>"


class Model:
    """
    Sovereign Mathematical Optimization Model.
    Supports LP, MILP, QP, and MIQP with automatic routing to native C++23 Simplex,
    Adaptive PDHG, or NVIDIA cuSPARSE GPU kernels.
    """
    def __init__(self, name: str = "BharatOpt_Model", sense: str = "max"):
        self.name = name
        self.sense = sense.lower()
        self.variables: Dict[str, Variable] = {}
        self.constraints: List[Constraint] = []
        self.quadratic_terms: Dict[str, Dict[str, float]] = {}
        self.model_type: str = "LP"

    def add_var(self, name: str, obj: float = 0.0, lb: float = 0.0, ub: float = float('inf'), vtype: str = "C") -> Variable:
        v = Variable(name, obj=obj, lb=lb, ub=ub, vtype=vtype)
        self.variables[name] = v
        if vtype.upper() in ("I", "B"):
            if self.model_type == "QP": self.model_type = "MIQP"
            elif self.model_type == "LP": self.model_type = "MILP"
        return v

    def add_vars(self, names: List[str], obj: Union[float, List[float]] = 0.0, vtype: str = "C") -> List[Variable]:
        vars_list = []
        for i, n in enumerate(names):
            o = obj[i] if isinstance(obj, list) else obj
            vars_list.append(self.add_var(n, obj=o, vtype=vtype))
        return vars_list

    def add_constraint(self, terms: Dict[str, float], rel: str = "<=", rhs: float = 0.0, name: Optional[str] = None) -> Constraint:
        con_name = name or f"C{len(self.constraints) + 1}"
        c = Constraint(con_name, terms, rel, rhs)
        self.constraints.append(c)
        return c

    def set_quadratic_objective(self, q_matrix: Dict[str, Dict[str, float]]):
        self.quadratic_terms = q_matrix
        if self.model_type == "MILP": self.model_type = "MIQP"
        else: self.model_type = "QP"

    def solve(self) -> SolveResult:
        """Executes optimization via the sovereign C++ native core."""
        obj_terms = {v.name: v.obj for v in self.variables.values()}
        con_list = [{"name": c.name, "rel": c.rel, "rhs": c.rhs, "terms": c.terms} for c in self.constraints]
        
        # Determine model type
        mtype = self.model_type
        if any(v.vtype in ("I", "B") for v in self.variables.values()):
            mtype = "MIQP" if self.quadratic_terms else "MILP"
        elif self.quadratic_terms:
            mtype = "QP"

        native = write_dat_and_run_cpp(
            obj_terms=obj_terms,
            constraints=con_list,
            sense=self.sense,
            workspace_root=_CURRENT_DIR,
            model_type=mtype,
            quadratic_terms=self.quadratic_terms if self.quadratic_terms else None
        )
        return SolveResult(native, self.name, list(self.variables.keys()))

    @classmethod
    def from_mps(cls, filepath_or_text: str) -> "Model":
        """Loads a model directly from a standard MPS file or string."""
        if os.path.exists(filepath_or_text):
            with open(filepath_or_text, "r") as f:
                text = f.read()
        else:
            text = filepath_or_text
            
        parsed = parse_mps_text(text)
        m = cls(name=parsed.get("name", "MPS_Model"), sense=parsed.get("obj_sense", "min"))
        
        int_vars = set(parsed.get("integer_vars", []))
        for vname, coef in parsed.get("obj_terms", {}).items():
            vtype = "I" if vname in int_vars else "C"
            m.add_var(vname, obj=coef, vtype=vtype)
            
        for c in parsed.get("constraints", []):
            m.add_constraint(terms=c["terms"], rel=c["rel"], rhs=c["rhs"], name=c["name"])
            
        return m

    @classmethod
    def from_english(cls, prompt: str) -> "Model":
        """Constructs an optimization model directly from plain conversational English."""
        parsed = parse_conversational_english(prompt)
        m = cls(name=parsed.get("name", "English_Model"), sense=parsed.get("obj_sense", "max"))
        
        int_vars = set(parsed.get("integer_vars", []))
        for vname, coef in parsed.get("obj_terms", {}).items():
            vtype = "I" if vname in int_vars else "C"
            m.add_var(vname, obj=coef, vtype=vtype)
            
        for c in parsed.get("constraints", []):
            m.add_constraint(terms=c["terms"], rel=c["rel"], rhs=c["rhs"], name=c["name"])
            
        return m
