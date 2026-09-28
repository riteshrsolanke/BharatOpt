import random
import os

def generate_large_lp(n, m, filename):
    with open(filename, 'w') as f:
        f.write(f"{n} {m} 1\n") 
        for i in range(n): f.write(f"v{i} {random.uniform(-10, 10):.2f}\n")
        for i in range(m): f.write(f"C{i} L {random.uniform(50, 100):.2f}\n")
        for i in range(m):
            row = []
            for j in range(n):
                if random.random() < 0.02: row.append(f"{random.uniform(0, 5):.2f}")
                else: row.append("0.0")
            f.write(" ".join(row) + "\n")

out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bench_99_large_scale.dat')
generate_large_lp(500, 500, out_path)
print(f"Generated {out_path}")
