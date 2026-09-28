import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim

# Add ml/gnn to import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml/gnn')))
from model import BipartiteGNN

def create_synthetic_milp_dataset(num_samples=100):
    """
    Synthesizes bipartite graph representations of MILP branch-and-bound nodes
    with strong branching teacher labels for training.
    """
    dataset = []
    torch.manual_seed(42)
    
    for idx in range(num_samples):
        num_vars = torch.randint(4, 12, (1,)).item()
        num_cons = torch.randint(3, 8, (1,)).item()
        
        # var_features: [Obj Coeff, Is_Integer, Value, Reduced_Cost, Fractionality]
        c = torch.randn(num_vars, 1) * 5.0
        is_int = torch.randint(0, 2, (num_vars, 1)).float()
        val = torch.rand(num_vars, 1) * 10.0
        rc = torch.randn(num_vars, 1) * 2.0
        frac = torch.abs(val - torch.round(val))
        x_var = torch.cat([c, is_int, val, rc, frac], dim=1)
        
        # cons_features: [RHS, Sense (-1 for <=, 1 for >=), Dual_estimate]
        b = torch.rand(num_cons, 1) * 50.0 + 10.0
        sense = torch.choice(torch.tensor([-1.0, 1.0]), (num_cons, 1)) if hasattr(torch, 'choice') else (torch.randint(0, 2, (num_cons, 1)).float() * 2.0 - 1.0)
        dual = torch.rand(num_cons, 1) * 3.0
        x_cons = torch.cat([b, sense, dual], dim=1)
        
        # Bipartite edges between vars and cons
        u_list, v_list = [], []
        for i in range(num_cons):
            # connect to 2 to 4 variables
            connected_vars = torch.randperm(num_vars)[:torch.randint(2, min(5, num_vars + 1), (1,)).item()]
            for j in connected_vars:
                u_list.extend([j.item(), num_vars + i])
                v_list.extend([num_vars + i, j.item()])
        
        edge_index = torch.tensor([u_list, v_list], dtype=torch.long)
        
        # Teacher label: Strong-branching score based on fractionality * obj_coeff magnitude * degree
        var_degrees = torch.zeros(num_vars)
        for j in u_list:
            if j < num_vars:
                var_degrees[j] += 1
                
        sb_score = frac.squeeze(1) * is_int.squeeze(1) * (torch.abs(c.squeeze(1)) + 1.0) * (var_degrees + 1.0)
        if sb_score.max() > 0:
            target_probs = torch.softmax(sb_score * 3.0, dim=0)
        else:
            target_probs = torch.ones(num_vars) / num_vars
            
        dataset.append({
            'x_var': x_var,
            'x_cons': x_cons,
            'edge_index': edge_index,
            'target': target_probs
        })
        
    return dataset

def train_gnn_branching():
    print("=" * 65)
    print(" BharatOpt-X: Training GNN-Assisted Branching Oracle (PyG)")
    print("=" * 65)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = BipartiteGNN(var_features_dim=5, cons_features_dim=3, hidden_dim=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    print("Generating offline MILP strong-branching supervision dataset...")
    train_data = create_synthetic_milp_dataset(150)
    val_data = create_synthetic_milp_dataset(30)
    
    print(f"Training on {len(train_data)} bipartite graph instances across 25 epochs...")
    model.train()
    for epoch in range(1, 26):
        total_loss = 0.0
        for sample in train_data:
            x_var = sample['x_var'].to(device)
            x_cons = sample['cons_features'] if 'cons_features' in sample else sample['x_cons'].to(device)
            edge_index = sample['edge_index'].to(device)
            target = sample['target'].to(device)
            
            optimizer.zero_grad()
            preds = model(x_var, x_cons, edge_index)
            log_preds = torch.log(preds + 1e-8)
            loss = loss_fn(log_preds, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 5 == 0 or epoch == 25:
            # Validation accuracy
            model.eval()
            val_correct = 0
            with torch.no_grad():
                for sample in val_data:
                    xv = sample['x_var'].to(device)
                    xc = sample['x_cons'].to(device)
                    ei = sample['edge_index'].to(device)
                    tg = sample['target'].to(device)
                    p = model(xv, xc, ei)
                    if torch.argmax(p) == torch.argmax(tg):
                        val_correct += 1
            model.train()
            acc = (val_correct / len(val_data)) * 100
            print(f"Epoch {epoch:02d}/25 | Avg Train Loss: {total_loss/len(train_data):.4f} | Val Top-1 Accuracy: {acc:.1f}%")
            
    # Save checkpoint
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'gnn_branching_v1.pt')
    torch.save(model.state_dict(), out_path)
    print(f"\n[SUCCESS] Successfully trained and saved model weights to: {out_path}")
    print(f"File size: {os.path.getsize(out_path):,} bytes")
    
    # Test loading
    test_model = BipartiteGNN(5, 3, 64)
    test_model.load_state_dict(torch.load(out_path, map_location='cpu'))
    test_model.eval()
    print("Verification: Checkpoint verified and loaded successfully into fresh instance!")
    return out_path

if __name__ == '__main__':
    train_gnn_branching()
