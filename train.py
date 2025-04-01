import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from gnn import GCN, GraphSAGE, GAT
import numpy as np

dataset = Planetoid(root='./data', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.edge_index.shape[1]}')
print(f'Unique node labels: {torch.unique(data.y)}')
print(f'Number of unique labels (classes): {len(torch.unique(data.y))}')

models = {
    'GCN': GCN(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes),
    'GraphSAGE': GraphSAGE(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes),
    'GAT': GAT(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes)
}

optimizers = {name: Adam(model.parameters(), lr=0.01, weight_decay=5e-4) for name, model in models.items()}

def add_gaussian_noise(data, noise_level=0.1):
    noise = torch.randn_like(data.x) * noise_level
    selected_nodes = torch.randint(0, data.num_nodes, (data.num_nodes // 10,))
    data.x[selected_nodes] += noise[selected_nodes]
    return data

def remove_random_edges(data, removal_ratio=0.05):
    num_edges = data.edge_index.shape[1]
    num_to_remove = int(removal_ratio * num_edges)
    edge_idx_to_remove = torch.randperm(num_edges)[:num_to_remove]
    edge_mask = torch.ones(num_edges, dtype=torch.bool)
    edge_mask[edge_idx_to_remove] = False
    data.edge_index = data.edge_index[:, edge_mask]
    return data

def train_and_test(model, data, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            out = model(data)
            pred = out.argmax(dim=1)
            correct = pred[data.test_mask] == data.y[data.test_mask]
            acc = correct.sum().item() / data.test_mask.sum().item()
            print(f"[{model.__class__.__name__}] Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = correct.sum().item() / data.test_mask.sum().item()
    return acc

results = {scenario: {} for scenario in ['no_attack', 'gaussian_noise', 'edge_removal']}

scenarios = {
    'no_attack': lambda data: data,
    'gaussian_noise': lambda data: add_gaussian_noise(data.clone(), noise_level=0.1),
    'edge_removal': lambda data: remove_random_edges(add_gaussian_noise(data.clone(), noise_level=0.1), removal_ratio=0.05)
}

num_epochs = 200

for scenario, attack_function in scenarios.items():
    print(f"\nTraining with {scenario.replace('_', ' ').capitalize()}:")
    for name, model in models.items():
        optimizer = optimizers[name]
        attacked_data = attack_function(data)
        final_acc = train_and_test(model, attacked_data, optimizer, num_epochs)
        results[scenario][name] = final_acc

print("\nFinal Test Accuracy after training:")
for scenario, scenario_results in results.items():
    print(f"\nScenario: {scenario.replace('_', ' ').capitalize()}")
    for model_name, acc in scenario_results.items():
        print(f"{model_name}: {acc:.4f}")
