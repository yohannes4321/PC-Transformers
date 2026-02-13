import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block used within the transformer architecture.
    Includes two linear layers and two predictive coding layers for local learning.
    """

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.fc2 = nn.Linear(4 * config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

        # Connector aliases for latent transitions
        self.attnoutput_fc1 = self.fc1
        self.fc1_fc2 = self.fc2

        self.pc_layer2 = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=getattr(config, 'update_bias', True),
            energy_fn_name=getattr(config, 'internal_energy_fn_name', 'pc_e'),
            optimizer_name=getattr(config, 'optimizer_name', 'adam'),
            optimizer_beta1=getattr(config, 'optimizer_beta1', 0.9),
            optimizer_beta2=getattr(config, 'optimizer_beta2', 0.999),
            optimizer_eps=getattr(config, 'optimizer_eps', 1e-8),
            optimizer_sign_value=getattr(config, 'optimizer_sign_value', -1.0),
            optimizer_weight_bound=getattr(config, 'optimizer_weight_bound', 0.0),
        )

        self.pc_layer1 = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=getattr(config, 'update_bias', True),
            energy_fn_name=getattr(config, 'internal_energy_fn_name', 'pc_e'),
            optimizer_name=getattr(config, 'optimizer_name', 'adam'),
            optimizer_beta1=getattr(config, 'optimizer_beta1', 0.9),
            optimizer_beta2=getattr(config, 'optimizer_beta2', 0.999),
            optimizer_eps=getattr(config, 'optimizer_eps', 1e-8),
            optimizer_sign_value=getattr(config, 'optimizer_sign_value', -1.0),
            optimizer_weight_bound=getattr(config, 'optimizer_weight_bound', 0.0),
        )
