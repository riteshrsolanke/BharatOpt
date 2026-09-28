import sys
import random

def generate_dataset(num_vars, num_constraints, density, output_file):
    print(f"Generating large-scale dataset: {num_vars} vars, {num_constraints} constraints, density {density}")
    
    with open(output_file, "w") as f:
        # Header: num_vars num_constraints sense (1 for min, -1 for max)
        f.write(f"{num_vars} {num_constraints} 1\n")
        
        # Variables and objective coefficients
        for j in range(num_vars):
            c = round(random.uniform(-10, 10), 4)
            f.write(f"v{j} {c}\n")
            
        # Constraints and RHS
        for i in range(num_constraints):
            b = round(random.uniform(10, 100), 4)
            # Mix of L (less than), G (greater than), E (equal)
            sense = random.choice(['L', 'G', 'E']) 
            f.write(f"c{i} {sense} {b}\n")
            
        # Sparse Matrix Generator
        f.write("SPARSE_A\n")
        
        # To avoid massive memory footprint, we write directly
        # and estimate NNZ. For exactness, we'll write the triplets to a temp file, count them, then write the header.
        # But this is Python, we can just generate the triplets in memory if it's small,
        # or we write it to a buffer. For 1 million vars * 1 million cons * 0.00001 density = 10 million NNZ.
        # 10M tuples is about 100MB, easily fits in RAM.
        
        nnz = int(num_vars * num_constraints * density)
        print(f"Targeting ~{nnz} non-zeros...")
        
        triplets = []
        for i in range(num_constraints):
            # Guarantee at least one entry per row to avoid empty constraints
            col = random.randint(0, num_vars - 1)
            val = round(random.uniform(-5, 5), 4)
            if val == 0: val = 1.0
            triplets.append((i, col, val))
            
        # Add random entries
        remaining = nnz - num_constraints
        for _ in range(remaining):
            r = random.randint(0, num_constraints - 1)
            c = random.randint(0, num_vars - 1)
            val = round(random.uniform(-5, 5), 4)
            if val != 0:
                triplets.append((r, c, val))
                
        # Remove exact duplicates (optional, just to be safe)
        triplets = list(set(triplets))
        
        f.write(f"{len(triplets)}\n")
        for r, c, val in triplets:
            f.write(f"{r} {c} {val}\n")

    print(f"Saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python generate_large_scale.py <num_vars> <num_cons> <density> <output.dat>")
        sys.exit(1)
    
    nv = int(sys.argv[1])
    nc = int(sys.argv[2])
    den = float(sys.argv[3])
    out = sys.argv[4]
    generate_dataset(nv, nc, den, out)
