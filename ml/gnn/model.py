import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, Linear

class BipartiteGNN(torch.nn.Module):
    """
    Graph Neural Network for MILP Branching.
    Treats the MILP as a bipartite graph (Variable Nodes <-> Constraint Nodes).
    """
    def __init__(self, var_features_dim, cons_features_dim, hidden_dim):
        super(BipartiteGNN, self).__init__()
        
        # Encoders for initial features
        self.var_encoder = Linear(var_features_dim, hidden_dim)
        self.cons_encoder = Linear(cons_features_dim, hidden_dim)
        
        # Bipartite Message Passing Layers
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        # Scoring head to predict branching priority for variables
        self.scoring_head = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_var, x_cons, edge_index):
        # x_var: [num_vars, var_features_dim]
        # x_cons: [num_cons, cons_features_dim]
        # edge_index: [2, num_edges] linking vars to constraints
        
        # 1. Encode initial features
        h_var = F.relu(self.var_encoder(x_var))
        h_cons = F.relu(self.cons_encoder(x_cons))
        
        # Combine node representations for the homogeneous convolutions (simplified)
        h = torch.cat([h_var, h_cons], dim=0)
        
        # 2. Message Passing
        h = F.relu(self.conv1(h, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        
        # 3. Extract variable representations (first 'num_vars' indices)
        num_vars = x_var.size(0)
        h_var_final = h[:num_vars]
        
        # 4. Predict Branching Score
        scores = self.scoring_head(h_var_final)
        
        # Sigmoid for probability (0 to 1 scaling)
        return torch.sigmoid(scores).squeeze(-1)
