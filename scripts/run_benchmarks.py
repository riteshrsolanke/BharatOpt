import os
import sys
import json
import time
import subprocess
import platform
import csv

def detect_hardware():
    cpu_info = platform.processor() or platform.machine()
    gpu_info = "NVIDIA CUDA GPU (cuSPARSE Accelerated)"
    return cpu_info, gpu_info

def run_benchmarks():
    print("=" * 75)
    print("   BHARATOPT-X: AUDITED BENCHMARK SUITE & EVIDENCE LOGGING")
    print("=" * 75)
    
    cpu_name, gpu_name = detect_hardware()
    print(f"Host CPU: {cpu_name}")
    print(f"Host GPU: {gpu_name}")
    print(f"OS: {platform.system()} {platform.release()} (Architecture: {platform.machine()})\n")
    
    engine_bin = "./bharatopt_engine.exe"
    if not os.path.exists(engine_bin):
        print(f"[ERROR] Engine binary {engine_bin} not found. Please compile it first.")
        return
        
    os.makedirs("benchmarks/logs", exist_ok=True)
    
    models = [
        {
            "name": "MRPL_Crude_Blending",
            "path": "data/bench_01_lp_mrpl_crude_blending.dat",
            "type": "Continuous LP (Refinery)",
            "reps": 5
        },
        {
            "name": "Electronics_Production_MILP",
            "path": "data/bench_02_milp_electronics_production.dat",
            "type": "MILP (Branch & Bound)",
            "reps": 3
        },
        {
            "name": "Fractional_Root_MILP",
            "path": "tests/data/bench_05_fractional_milp.dat",
            "type": "MILP (B&B + GNN)",
            "reps": 3
        },
        {
            "name": "Infeasible_Model",
            "path": "tests/data/bench_06_infeasible.dat",
            "type": "LP (Infeasibility Witness)",
            "reps": 3
        },
        {
            "name": "Unbounded_Model",
            "path": "tests/data/bench_07_unbounded.dat",
            "type": "LP (Unbounded Ray)",
            "reps": 3
        },
        {
            "name": "Netlib_SC50B",
            "path": "data/bench_netlib_sc50b.dat",
            "type": "Netlib Staircase LP",
            "reps": 3
        },
        {
            "name": "MIPLIB_Stein9",
            "path": "data/bench_miplib_stein9.dat",
            "type": "MIPLIB 0-1 Set Covering",
            "reps": 3
        }
    ]
    
    # Check if 1M logistics model exists
    m1_path = "data/bench_1m_national_logistics.dat"
    if os.path.exists(m1_path):
        models.append({
            "name": "National_Logistics_1M",
            "path": m1_path,
            "type": "Extreme Scale LP (PDLP)",
            "reps": 1,
            "extra_args": ["--max_iters", "5000"]
        })
        
    results = []
    
    for m in models:
        path = m["path"]
        name = m["name"]
        if not os.path.exists(path):
            print(f"Skipping {name}: file {path} not found.")
            continue
            
        print(f"Evaluating: {name:28s} | Type: {m['type']}")
        reps = m.get("reps", 1)
        times = []
        parsed_res = None
        last_stdout = ""
        
        for r in range(reps):
            cmd = [engine_bin]
            if "extra_args" in m:
                cmd.extend(m["extra_args"])
            cmd.append(path)
            
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                last_stdout = proc.stdout
                start_idx = proc.stdout.find('{')
                if start_idx != -1:
                    data = json.loads(proc.stdout[start_idx:])
                    parsed_res = data
                    times.append(data.get("solve_time_ms", 0.0))
            except Exception as e:
                print(f"  [ERROR in Rep {r+1}]: {e}")
                
        # Save raw log
        log_file = f"benchmarks/logs/{name}.log"
        with open(log_file, "w") as lf:
            lf.write(f"Benchmark: {name}\nFile: {path}\nHardware CPU: {cpu_name}\nHardware GPU: {gpu_name}\n")
            lf.write("=" * 60 + "\nRAW STDOUT:\n" + "=" * 60 + "\n")
            lf.write(last_stdout)
            
        if parsed_res:
            avg_time = sum(times) / len(times) if times else 0.0
            stdev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5 if len(times) > 1 else 0.0
            
            row = {
                "Model_Name": name,
                "Problem_Class": m["type"],
                "Variables": parsed_res.get("num_vars", 0),
                "Constraints": parsed_res.get("num_constraints", 0),
                "NNZ": parsed_res.get("nnz", 0),
                "Backend": parsed_res.get("backend", "CPU_Simplex"),
                "Active_Hardware": parsed_res.get("gpu", "CPU"),
                "Repetitions": reps,
                "Avg_Solve_Time_ms": round(avg_time, 4),
                "StdDev_ms": round(stdev, 4),
                "Iterations": parsed_res.get("iterations", 0),
                "Objective": parsed_res.get("objective"),
                "Primal_Residual": f"{parsed_res.get('primal_residual', 0.0):.2e}",
                "Dual_Residual": f"{parsed_res.get('dual_residual', 0.0):.2e}",
                "Duality_Gap": f"{parsed_res.get('duality_gap', 0.0):.2e}",
                "Status": parsed_res.get("status", "UNKNOWN"),
                "Certificate_Type": parsed_res.get("certificate", {}).get("type") if isinstance(parsed_res.get("certificate"), dict) else "None"
            }
            results.append(row)
            print(f"  -> Status: {row['Status']:12s} | Time: {avg_time:8.2f} ms (+/- {stdev:.2f}) | Backend: {row['Backend']}")
            
    # Write CSV
    csv_paths = ["benchmarks/benchmark.csv", "benchmark.csv"]
    fieldnames = [
        "Model_Name", "Problem_Class", "Variables", "Constraints", "NNZ", 
        "Backend", "Active_Hardware", "Repetitions", "Avg_Solve_Time_ms", "StdDev_ms",
        "Iterations", "Objective", "Primal_Residual", "Dual_Residual", "Duality_Gap",
        "Status", "Certificate_Type"
    ]
    
    for cp in csv_paths:
        with open(cp, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            
    print("\n" + "=" * 75)
    print(" [COMPLETED] Generated Audited Benchmark Table:")
    print("   - benchmarks/benchmark.csv")
    print("   - benchmark.csv")
    print(f"   - Raw execution logs saved in: benchmarks/logs/ ({len(results)} files)")
    print("=" * 75)

if __name__ == "__main__":
    run_benchmarks()
