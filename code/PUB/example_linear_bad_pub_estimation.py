import torch
import torch.nn as nn

# Define a simple linear network
class LinearNetwork(nn.Module):
    def __init__(self, depth, input_dim, output_dim):
        super(LinearNetwork, self).__init__()
        layers = [nn.Linear(input_dim, input_dim) for _ in range(depth - 1)]
        layers.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Function to compute PUB
def compute_pub_linear(net):
    pub = 1.0
    for layer in net.model:
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data
            spectral_norm = torch.linalg.norm(weight, ord=2).item()
            pub *= spectral_norm
    return pub

# Function to compute resulting matrix and its spectral norm
def compute_resulting_matrix_and_true_lipschitz(net):
    with torch.no_grad():
        # Start with identity matrix
        resulting_matrix = torch.eye(net.model[0].weight.size(1))
        for layer in net.model:
            if isinstance(layer, nn.Linear):
                resulting_matrix = layer.weight @ resulting_matrix
        true_lipschitz = torch.linalg.norm(resulting_matrix, ord=2).item()
        return resulting_matrix, true_lipschitz

# Example: Linear network with 110 layers
depth = 110
input_dim = 100
output_dim = 10  # Keep dimensions manageable
linear_net = LinearNetwork(depth, input_dim, output_dim)

# Initialize weights for reproducibility
torch.manual_seed(0)
for layer in linear_net.model:
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, a=5**0.5)  # He initialization

# Compute PUB
pub = compute_pub_linear(linear_net)

# Compute resulting matrix and true Lipschitz constant
resulting_matrix, true_lipschitz = compute_resulting_matrix_and_true_lipschitz(linear_net)

# Output results
print(f"Estimated PUB for Linear Network: {pub:.2e}")
print(f"True Lipschitz constant for Linear Network: {true_lipschitz:.2e}")