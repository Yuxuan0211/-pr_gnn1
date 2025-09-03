# src/pr_gnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.autograd import grad
import numpy as np

class PRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=10):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels)
        self.decoder = MLPDecoder(hidden_channels, out_channels)
        self.gamma = 1.4

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        q_pred = self.decoder(h)
        return q_pred, h

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

class MLPDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

    def forward(self, x):
        return self.mlp(x)

def assign_regions(data, config):
    device = data.x.device
    P = data.y[:, 6].cpu().numpy()
    rho = data.y[:, 7].cpu().numpy()
    V = torch.norm(data.y[:, :3], dim=1).cpu().numpy()
    mu_t = data.y[:, 8].cpu().numpy()
    h = data.y[:, 9].cpu().numpy()

    P_inf, rho_inf, V_inf = config['P_inf'], config['rho_inf'], config['V_inf']
    mask = np.zeros(len(data.y), dtype=int)

    mask[(P > 1.5 * P_inf) | (rho > 1.5 * rho_inf)] = 0
    mask[(mu_t > 0.1) & (V < 0.3 * V_inf)] = 1
    mask[(mu_t > 0.05) & (data.y[:, 0].cpu().numpy() < 0.2 * V_inf)] = 2
    mask[(mu_t < 1e-5) & (mask == 0)] = 3
    close_to_inf = np.isclose(P, P_inf, rtol=1e-2) & np.isclose(rho, rho_inf, rtol=1e-2) & (V > 0.95 * V_inf)
    mask[close_to_inf] = 4

    return torch.tensor(mask, device=device)

class PhysicsLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gamma = 1.4

    def forward(self, pred, data, region_mask):
        loss_sup = F.mse_loss(pred, data.y)

        P_pred, rho_pred, h_pred = pred[:, 6], pred[:, 7], pred[:, 9]
        thermo_residual = P_pred - ((self.gamma - 1) / self.gamma) * rho_pred * h_pred
        loss_thermo = torch.mean(thermo_residual ** 2)

        coords = data.x[:, :3].requires_grad_(True)
        V_pred = pred[:, :3]
        jacobian = []
        for i in range(3):
            grads = grad(V_pred[:, i], coords, grad_outputs=torch.ones_like(V_pred[:, i]),
                         create_graph=True, retain_graph=True)[0]
            jacobian.append(grads)
        jacobian = torch.stack(jacobian, dim=1)
        curl_x = jacobian[:, 2, 1] - jacobian[:, 1, 2]
        curl_y = jacobian[:, 0, 2] - jacobian[:, 2, 0]
        curl_z = jacobian[:, 1, 0] - jacobian[:, 0, 1]
        omega_pred = pred[:, 3:6]
        omega_true = torch.stack([curl_x, curl_y, curl_z], dim=1)
        loss_vorticity = F.mse_loss(omega_pred, omega_true)

        total_h_pred = h_pred + 0.5 * torch.sum(pred[:, :3] ** 2, dim=1)
        h_total_inf = self.config['h_inf'] + 0.5 * self.config['V_inf'] ** 2
        mask_34 = (region_mask == 3) | (region_mask == 4)
        loss_energy = F.mse_loss(total_h_pred[mask_34], torch.full_like(total_h_pred[mask_34], h_total_inf))

        loss_noslip = F.mse_loss(pred[region_mask == 1, :3], torch.zeros_like(pred[region_mask == 1, :3]))
        loss_wake = torch.mean(pred[region_mask == 2, :3] ** 2) + torch.mean(F.relu(1e-4 - pred[region_mask == 2, 8]))
        loss_inviscid = torch.mean(pred[region_mask == 3, 8] ** 2)
        loss_freestream = F.mse_loss(pred[region_mask == 4], data.y[region_mask == 4])

        λ = self.config['loss_weights']
        physics_loss = (
            λ['thermo'] * loss_thermo +
            λ['vorticity'] * loss_vorticity +
            λ['energy'] * loss_energy +
            λ['noslip'] * loss_noslip +
            λ['wake'] * loss_wake +
            λ['inviscid'] * loss_inviscid +
            λ['freestream'] * loss_freestream
        )
        total_loss = loss_sup + physics_loss
        return total_loss, {
            'sup': loss_sup.item(),
            'thermo': loss_thermo.item(),
            'vorticity': loss_vorticity.item(),
            'energy': loss_energy.item()
        }
