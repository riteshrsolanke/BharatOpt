import os
import sys
import json
import time
import subprocess
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    instance_name: str
    status: str
    objective: float
    iterations: int
    solve_time_ms: float
    primal_residual: float
    dual_residual: float
    duality_gap: float
    num_vars: int
    num_constraints: int
    bb_nodes: int
    cuts_applied: int

def run_benchmark(executable_path: str, dat_path: str) -> BenchmarkResult:
    start = time.perf_counter()
    result = subprocess.run([executable_path, dat_path], capture_output=True, text=True)
    end = time.perf_counter()
    
    try:
        output_json = json.loads(result.stdout)
        return BenchmarkResult(
            instance_name=os.path.basename(dat_path),
            status=output_json.get("status", "ERROR"),
            objective=output_json.get("objective", 0.0),
            iterations=output_json.get("iterations", 0),
            solve_time_ms=output_json.get("solve_time_ms", (end-start)*1000),
            primal_residual=output_json.get("primal_residual", 0.0),
            dual_residual=output_json.get("dual_residual", 0.0),
            duality_gap=output_json.get("duality_gap", 0.0),
            num_vars=output_json.get("num_vars", 0),
            num_constraints=output_json.get("num_constraints", 0),
            bb_nodes=output_json.get("bb_nodes", 0),
            cuts_applied=output_json.get("cuts_applied", 0)
        )
    except Exception as e:
        return BenchmarkResult(os.path.basename(dat_path), "CRASH", 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)

def main():
    print("=========================================================================================================")
    print("  BharatOpt-X Phase 7: Empirical Benchmark Harness (with Full Telemetry)")
    print("=========================================================================================================")
    print(f"{'Instance':<35} | {'Path':<7} | {'Size(N,M)':<10} | {'Status':<7} | {'Time(ms)':<9} | {'Iters':<5} | {'Obj Value':<12} | {'Max_Res/Gap'}")
    print("-" * 115)
    
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    engine_path = os.path.join(workspace_root, 'build', 'Release', 'bharatopt_engine.exe')
    if not os.path.exists(engine_path):
        engine_path = os.path.join(workspace_root, 'build', 'bharatopt_engine')
        
    data_dir = os.path.join(workspace_root, 'data')
    test_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith('.dat')]
    
    for f in test_files:
        dat_path = os.path.join(data_dir, f)
        
        # Determine if we should run PDLP based on file name or type
        is_large_lp = ("bench_90" in f or "bench_91" in f or "bench_99" in f)
        
        # Run default (Simplex/IPM/BB)
        paths_to_run = [("Simplex", engine_path)]
        if is_large_lp:
            paths_to_run = [("Simplex", engine_path), ("PDLP", f"{engine_path} --pdlp")]
            
        for path_name, cmd in paths_to_run:
            start = time.perf_counter()
            try:
                result = subprocess.run(cmd.split() + [dat_path], capture_output=True, text=True, timeout=10)
                end = time.perf_counter()
                res_json = json.loads(result.stdout)
                metric = res_json.get("duality_gap", 0.0) if "qp" in f.lower() else max(res_json.get("primal_residual", 0.0), res_json.get("dual_residual", 0.0))
                size_str = f"{res_json.get('num_vars', 0)}x{res_json.get('num_constraints', 0)}"
                nodes = res_json.get("bb_nodes", 0)
                if "milp" in f.lower() and nodes == 0: nodes = 1
                
                print(f"{f:<35} | {path_name:<7} | {size_str:<10} | {res_json.get('status', 'ERR'):<7} | {res_json.get('solve_time_ms', 0):<9.2f} | {res_json.get('iterations', 0):<5} | {res_json.get('objective', 0):<12.4f} | {metric:.2e}")
            except subprocess.TimeoutExpired:
                print(f"{f:<35} | {path_name:<7} | {'??':<10} | {'TIMEOUT':<7} | {10000:<9.2f} | {'N/A':<5} | {0:<12.4f} | {0:.2e}")
            except Exception as e:
                print(f"{f:<35} | {path_name:<7} | {'ERR':<10} | {'CRASH':<7} | {0:<9.2f} | {0:<5} | {0:<12.4f} | {0:.2e}")
                print(f"Error parsing JSON: {e}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        
    print("=========================================================================================================")

if __name__ == "__main__":
    main()
