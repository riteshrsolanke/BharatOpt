import os
import sys
import json
import subprocess

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(workspace_root)
from services.api.cpp_caller import write_dat_and_run_cpp

def generate_dats():
    data_dir = os.path.join(workspace_root, 'data')
    samples_dir = os.path.join(data_dir, 'samples')
    
    for filename in os.listdir(samples_dir):
        if filename.endswith('.json'):
            with open(os.path.join(samples_dir, filename), 'r') as f:
                data = json.load(f)
            
            obj_terms = data.get("objective", {}).get("terms", {})
            sense = data.get("objective", {}).get("sense", "max")
            constraints = data.get("constraints", [])
            model_type = data.get("type", "LP").upper()
            
            all_vars = sorted(list(set(list(obj_terms.keys()) + [v for c in constraints for v in c.get("terms", {}).keys()])))
            num_vars = len(all_vars)
            num_constraints = len(constraints)
            
            dat_path = os.path.join(data_dir, f"bench_{filename.replace('.json', '.dat')}")
            with open(dat_path, "w") as f_out:
                f_out.write(f"{num_vars} {num_constraints} {1 if sense.lower() == 'min' else -1}\n")
                for v in all_vars:
                    f_out.write(f"{v} {obj_terms.get(v, 0.0)}\n")
                
                for c in constraints:
                    rel = c.get("rel", "<=")
                    char_rel = "L" if rel in ["<=", "L"] else "G" if rel in [">=", "G"] else "E"
                    f_out.write(f"{c.get('name', 'C')} {char_rel} {c.get('rhs', 0.0)}\n")
                
                for c in constraints:
                    terms = c.get("terms", {})
                    for v in all_vars:
                        f_out.write(f"{terms.get(v, 0.0)} ")
                    f_out.write("\n")
                    
                if model_type == "MILP":
                    f_out.write("INTEGERS\n")
                    for v in all_vars:
                        f_out.write("1 ")
                    f_out.write("\n")
            print(f"Generated {dat_path}")

if __name__ == "__main__":
    generate_dats()
