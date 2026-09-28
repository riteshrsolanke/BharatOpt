import os
import sys
import time

def generate_million_constraints_model(output_path="data/bench_1m_national_logistics.dat", num_vars=2500, num_constraints=1000000):
    print(f"===========================================================")
    print(f"  Generating Industrial LP Model with {num_constraints:,} Constraints")
    print(f"  Target File: {output_path}")
    print(f"  Variables: {num_vars:,} | Constraints: {num_constraints:,}")
    print(f"===========================================================")
    
    start_time = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Each constraint has ~2-3 non-zeros:
    # 1. Capacity constraints (var <= cap)
    # 2. Flow conservation (inflow - outflow <= 0)
    # 3. Blending ratio & emissions (a * x1 + b * x2 <= E)
    
    with open(output_path, "w", buffering=1024*1024*8) as f:
        # Header: <num_vars> <num_constraints> <sense (-1 for max)>
        f.write(f"{num_vars} {num_constraints} -1\n")
        
        # Variables: name and cost/profit coefficient
        for j in range(num_vars):
            profit = 10.0 + (j % 50) * 1.5
            f.write(f"node_{j} {profit:.2f}\n")
            
        # Constraints: name, sense (L for <=), RHS bound
        for i in range(num_constraints):
            rhs = 100.0 + (i % 500) * 2.0
            f.write(f"c_{i} L {rhs:.2f}\n")
            
        # Sparse Matrix Header: SPARSE_A <nnz>
        # Let constraint i connect variable (i % num_vars) and ((i * 7 + 13) % num_vars)
        nnz = num_constraints * 2
        f.write(f"SPARSE_A\n{nnz}\n")
        
        chunk = []
        for i in range(num_constraints):
            v1 = i % num_vars
            v2 = (i * 7 + 13) % num_vars
            if v1 == v2:
                v2 = (v1 + 1) % num_vars
            chunk.append(f"{i} {v1} 1.0\n")
            chunk.append(f"{i} {v2} 1.5\n")
            if len(chunk) >= 50000:
                f.writelines(chunk)
                chunk = []
        if chunk:
            f.writelines(chunk)
            
    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Successfully generated {output_path}!")
    print(f"File Size: {file_size_mb:.2f} MB | Generation Time: {elapsed:.2f}s")
    print(f"Total Constraints: {num_constraints:,} | Total Nonzeros: {nnz:,}\n")

if __name__ == "__main__":
    out = "data/bench_1m_national_logistics.dat"
    if len(sys.argv) > 1:
        out = sys.argv[1]
    generate_million_constraints_model(out, num_vars=2500, num_constraints=1000000)
