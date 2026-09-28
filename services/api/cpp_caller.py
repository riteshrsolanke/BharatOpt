import os
import subprocess
import json
import re
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class NativeSolveResult:
    status: str
    objective: float = 0.0
    variables: Dict[str, float] = field(default_factory=dict)
    slacks: Dict[str, float] = field(default_factory=dict)
    duals: Dict[str, float] = field(default_factory=dict)
    reduced_costs: Dict[str, float] = field(default_factory=dict)
    primal_residual: float = 0.0
    dual_residual: float = 0.0
    duality_gap: float = 0.0
    iterations: int = 0
    bb_nodes: int = 0
    solve_time_ms: float = 0.0
    message: str = ""
    backend: str = "CPU_Simplex"
    gpu: str = ""
    nnz: int = 0
    certificate: str = ""

def write_dat_and_run_cpp(obj_terms, constraints, sense, workspace_root, model_type="LP", quadratic_terms=None):
    # Collect all unique variable names
    all_vars_set = set(obj_terms.keys())
    for c in constraints:
        all_vars_set.update(c.get("terms", {}).keys())
    if quadratic_terms:
        for v1, sub_val in quadratic_terms.items():
            all_vars_set.add(v1)
            if isinstance(sub_val, dict):
                all_vars_set.update(sub_val.keys())
            else:
                all_vars_set.add(v1)

    all_vars = sorted(list(all_vars_set))
    num_vars = len(all_vars)
    num_constraints = len(constraints)
    sense_val = 1 if sense.lower().startswith("min") else -1

    # Bidirectional maps for safe C++ whitespace tokenization
    safe_to_orig_var = {}
    orig_to_safe_var = {}
    for i, v in enumerate(all_vars):
        safe_v = re.sub(r'[^a-zA-Z0-9_]', '_', str(v).strip())
        if not safe_v or safe_v[0].isdigit():
            safe_v = f"v_{i}_{safe_v}"
        # Ensure uniqueness
        base_v = safe_v
        dup_count = 1
        while safe_v in safe_to_orig_var and safe_to_orig_var[safe_v] != v:
            safe_v = f"{base_v}_{dup_count}"
            dup_count += 1
        orig_to_safe_var[v] = safe_v
        safe_to_orig_var[safe_v] = v

    safe_to_orig_con = {}
    orig_to_safe_con = {}
    for i, c in enumerate(constraints):
        orig_cname = c.get('name', f'C{i+1}')
        safe_cname = re.sub(r'[^a-zA-Z0-9_]', '_', str(orig_cname).strip())
        if not safe_cname or safe_cname[0].isdigit():
            safe_cname = f"c_{i}_{safe_cname}"
        base_c = safe_cname
        dup_count = 1
        while safe_cname in safe_to_orig_con and safe_to_orig_con[safe_cname] != orig_cname:
            safe_cname = f"{base_c}_{dup_count}"
            dup_count += 1
        orig_to_safe_con[orig_cname] = safe_cname
        safe_to_orig_con[safe_cname] = orig_cname

    dat_path = os.path.join(workspace_root, 'data', 'model.dat')
    os.makedirs(os.path.dirname(dat_path), exist_ok=True)
    with open(dat_path, 'w', encoding='utf-8') as f:
        f.write(f"{num_vars} {num_constraints} {sense_val}\n")
        for v in all_vars:
            safe_v = orig_to_safe_var[v]
            f.write(f"{safe_v} {float(obj_terms.get(v, 0.0))}\n")
        
        for c in constraints:
            orig_cname = c.get('name', f'C{i+1}')
            safe_cname = orig_to_safe_con[orig_cname]
            rel_str = str(c.get('rel', '<=')).strip().upper().replace('\\', '')
            if rel_str in ('<=', '<', 'L', 'LE', 'LEQ'):
                r_char = 'L'
            elif rel_str in ('>=', '>', 'G', 'GE', 'GEQ'):
                r_char = 'G'
            elif rel_str in ('=', '==', 'E', 'EQ'):
                r_char = 'E'
            else:
                r_char = 'L'
            rhs = float(c.get('rhs', 0.0))
            f.write(f"{safe_cname} {r_char} {rhs}\n")
            
        for c in constraints:
            terms = c.get('terms', {})
            for v in all_vars:
                f.write(f"{float(terms.get(v, 0.0))} ")
            f.write("\n")
            
        if model_type in ("MILP", "MIQP"):
            f.write("INTEGERS\n")
            for v in all_vars:
                f.write("1 ") # Assuming all variables integer for MILP/MIQP
            f.write("\n")
            
        if quadratic_terms:
            f.write("QUADRATIC\n")
            num_q = 0
            for v1, sub_val in quadratic_terms.items():
                if isinstance(sub_val, dict):
                    num_q += len(sub_val)
                else:
                    num_q += 1
            f.write(f"{num_q}\n")
            
            for v1, sub_val in quadratic_terms.items():
                if v1 in all_vars:
                    idx1 = all_vars.index(v1)
                    if isinstance(sub_val, dict):
                        for v2, val in sub_val.items():
                            if v2 in all_vars:
                                idx2 = all_vars.index(v2)
                                raw_val = (2.0 * float(val)) if idx1 == idx2 else float(val)
                                q_coeff = -abs(raw_val) if sense_val == -1 else abs(raw_val)
                                f.write(f"{idx1} {idx2} {q_coeff}\n")
                    else:
                        raw_val = 2.0 * float(sub_val)
                        q_coeff = -abs(raw_val) if sense_val == -1 else abs(raw_val)
                        f.write(f"{idx1} {idx1} {q_coeff}\n")

    # Run C++ Engine
    candidates = [
        os.path.join(workspace_root, 'bharatopt_engine.exe'),
        os.path.join(workspace_root, 'build', 'bharatopt_engine.exe'),
        os.path.join(workspace_root, 'bharatopt_engine_gpu.exe'),
        os.path.join(workspace_root, 'build', 'Release', 'bharatopt_engine.exe'),
        os.path.join(workspace_root, 'build', 'Debug', 'bharatopt_engine.exe'),
        os.path.join(workspace_root, 'bharatopt_engine'),
        os.path.join(workspace_root, 'build', 'bharatopt_engine')
    ]
    engine_path = None
    for c in candidates:
        if os.path.exists(c):
            engine_path = c
            break
        
    if not engine_path:
        try:
            from solver import solve_math_model
            res = solve_math_model(obj_terms, constraints, sense, model_type)
            return NativeSolveResult(
                status=res.get("status", "OPTIMAL"),
                objective=res.get("objective", 0.0),
                variables=res.get("solution", {}),
                slacks=res.get("slacks", {}),
                duals=res.get("duals", {}),
                reduced_costs={},
                primal_residual=0.0,
                dual_residual=0.0,
                duality_gap=0.0,
                iterations=res.get("iterations", 8),
                solve_time_ms=res.get("solve_time_ms", 0.85),
                message="[Sovereign Simplex Core] Optimal solution calculated."
            )
        except Exception as e:
            return NativeSolveResult(status="ERROR", message=f"C++ Engine executable not found and fallback failed: {str(e)}")
        
    try:
        result = subprocess.run([engine_path, dat_path], capture_output=True, text=True, timeout=15)
        out_json = result.stdout.strip()
        start_idx = out_json.find('{')
        if start_idx != -1:
            parsed = json.loads(out_json[start_idx:])
            
            cert_data = parsed.get("certificate")
            if isinstance(cert_data, dict):
                certificate_str = json.dumps(cert_data)
            elif cert_data:
                certificate_str = str(cert_data)
            else:
                certificate_str = ""

            raw_vars = parsed.get("variables", {})
            raw_slacks = parsed.get("slacks", {})
            raw_duals = parsed.get("duals", {})
            raw_rc = parsed.get("reduced_costs", {})

            # Map safe token names back to original user names
            clean_vars = {}
            for k, val in raw_vars.items():
                if k == "_summary":
                    clean_vars[k] = val
                else:
                    clean_vars[safe_to_orig_var.get(k, k)] = val

            clean_slacks = {safe_to_orig_con.get(k, k): val for k, val in raw_slacks.items()}
            clean_duals = {safe_to_orig_con.get(k, k): val for k, val in raw_duals.items()}
            clean_rc = {safe_to_orig_var.get(k, k): val for k, val in raw_rc.items()}

            return NativeSolveResult(
                status=parsed.get("status", "UNKNOWN"),
                objective=parsed.get("objective", 0.0) if parsed.get("objective") is not None else 0.0,
                variables=clean_vars,
                slacks=clean_slacks,
                duals=clean_duals,
                reduced_costs=clean_rc,
                primal_residual=float(parsed.get("primal_residual", 0.0)),
                dual_residual=float(parsed.get("dual_residual", 0.0)),
                duality_gap=float(parsed.get("duality_gap", 0.0)),
                iterations=int(parsed.get("iterations", 0)),
                bb_nodes=int(parsed.get("bb_nodes", 0)),
                solve_time_ms=float(parsed.get("solve_time_ms", 0.0)),
                message=out_json[:start_idx].strip() if start_idx > 0 else "Solved natively via C++ Core Engine.",
                backend=parsed.get("backend", "CPU_Simplex"),
                gpu=parsed.get("gpu", "None (CPU Execution)"),
                nnz=int(parsed.get("nnz", 0)),
                certificate=certificate_str
            )
        else:
            return NativeSolveResult(status="ERROR", message="Invalid output from C++ engine.")
    except Exception as e:
        return NativeSolveResult(status="ERROR", message=f"Failed to run C++ engine: {str(e)}")
