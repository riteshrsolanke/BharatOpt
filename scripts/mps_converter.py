import sys

try:
    import pulp
except ImportError:
    print("Error: The 'pulp' library is required to parse MPS files.")
    print("Please install it by running: pip install pulp")
    sys.exit(1)

def convert_mps_to_bharatopt(mps_file, dat_file):
    print(f"Reading MPS file: {mps_file}...")
    
    # PuLP automatically reads standard MPS files
    var, model = pulp.LpProblem.fromMPS(mps_file)
    
    num_vars = len(model.variables())
    num_constraints = len(model.constraints)
    
    # 1 for minimize, -1 for maximize
    sense = 1 if model.sense == pulp.LpMinimize else -1
    
    with open(dat_file, "w") as f:
        # Header
        f.write(f"{num_vars} {num_constraints} {sense}\n")
        
        # Build variable mapping
        var_map = {}
        for idx, v in enumerate(model.variables()):
            var_map[v.name] = idx
            # Objective coefficient
            c = model.objective.get(v, 0.0)
            f.write(f"{v.name} {c}\n")
            
        # Constraints
        con_map = {}
        for idx, (name, constraint) in enumerate(model.constraints.items()):
            con_map[name] = idx
            
            # PuLP sense: 1 is <=, 0 is ==, -1 is >=
            c_sense = 'E'
            if constraint.sense == 1: c_sense = 'L'
            elif constraint.sense == -1: c_sense = 'G'
            
            rhs = -constraint.constant # PuLP moves RHS to LHS, so constant is negated
            f.write(f"{name} {c_sense} {rhs}\n")
            
        # Sparse Matrix (A_triplets)
        f.write("SPARSE_A\n")
        
        triplets = []
        for name, constraint in model.constraints.items():
            row = con_map[name]
            for v, val in constraint.items():
                col = var_map[v.name]
                triplets.append((row, col, val))
                
        f.write(f"{len(triplets)}\n")
        for r, c, val in triplets:
            f.write(f"{r} {c} {val}\n")
            
        # Check for integer variables
        has_integers = any(v.cat == pulp.LpInteger for v in model.variables())
        if has_integers:
            f.write("INTEGERS\n")
            for v in model.variables():
                f.write(f"{1 if v.cat == pulp.LpInteger else 0}\n")
                
    print(f"Successfully converted {num_vars} variables and {num_constraints} constraints.")
    print(f"Saved to native BharatOpt-X format: {dat_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python mps_converter.py <input.mps> <output.dat>")
        sys.exit(1)
        
    convert_mps_to_bharatopt(sys.argv[1], sys.argv[2])
