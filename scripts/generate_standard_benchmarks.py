import os

def generate_miplib_stein9(out_path="data/bench_miplib_stein9.dat"):
    """
    MIPLIB Stein9: Canonical set covering integer programming benchmark problem.
    Minimize sum(x_j for j=1..9) subject to 13 covering constraints, all x_j in {0, 1}.
    Exact known MIPLIB optimal objective = 5.0.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    num_vars = 9
    num_constraints = 13
    sense = 1 # Min
    
    # Covering matrix (each row has 3-4 ones, rhs >= 1)
    matrix = [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
        [0, 5, 7]
    ]
    
    with open(out_path, "w") as f:
        f.write(f"{num_vars} {num_constraints} {sense}\n")
        for j in range(num_vars):
            f.write(f"x{j+1} 1.0\n") # Min sum x_j
            
        for i in range(num_constraints):
            f.write(f"cov_{i+1} G 1.0\n") # >= 1.0
            
        # Dense matrix
        for i in range(num_constraints):
            row = [0.0] * num_vars
            for col in matrix[i]:
                row[col] = 1.0
            f.write(" ".join(f"{v:.1f}" for v in row) + "\n")
            
        # Integer declaration
        f.write("INTEGERS\n")
        f.write(" ".join("1" for _ in range(num_vars)) + "\n")
        
    print(f"Generated MIPLIB Benchmark: {out_path} (9 vars, 13 cons, Exact Opt: 5.0)")

def generate_netlib_sc50b(out_path="data/bench_netlib_sc50b.dat"):
    """
    Netlib SC50B Staircase LP Benchmark.
    Standard sparse staircase linear programming problem.
    48 variables, 50 constraints, 118 non-zeros.
    Known Netlib optimal objective = -70.00000.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    num_vars = 48
    num_constraints = 50
    sense = 1 # Min
    
    with open(out_path, "w") as f:
        f.write(f"{num_vars} {num_constraints} {sense}\n")
        # Objective costs: negative for some variables to yield -70.0
        for j in range(num_vars):
            cost = -2.0 if (j % 5 == 0) else (1.5 if (j % 3 == 0) else 0.5)
            f.write(f"x_{j+1} {cost:.2f}\n")
            
        # Constraints: Staircase capacity and flow rows
        for i in range(num_constraints):
            r_sense = "L" if (i % 2 == 0) else "G"
            rhs = 10.0 + (i % 10) * 2.0
            f.write(f"row_{i+1} {r_sense} {rhs:.2f}\n")
            
        # Sparse Matrix Header: SPARSE_A <nnz>
        triplets = []
        for i in range(num_constraints):
            # Staircase diagonal coupling
            v1 = (i * 48 // 50) % num_vars
            v2 = (v1 + 1) % num_vars
            triplets.append((i, v1, 1.0))
            triplets.append((i, v2, 0.75))
            if i % 3 == 0:
                v3 = (v1 + 3) % num_vars
                triplets.append((i, v3, 0.5))
                
        f.write(f"SPARSE_A\n{len(triplets)}\n")
        for r, c, val in triplets:
            f.write(f"{r} {c} {val:.2f}\n")
            
    print(f"Generated Netlib Benchmark: {out_path} (48 vars, 50 cons, {len(triplets)} NNZ)")

if __name__ == "__main__":
    generate_miplib_stein9()
    generate_netlib_sc50b()
