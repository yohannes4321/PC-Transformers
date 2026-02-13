import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class OutputLayer(nn.Module):
    """
    Output layer for the transformer model, consisting of a linear projection and a predictive coding layer.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output = nn.Linear(config.n_embed, config.vocab_size)
        self.fc2_linear_output = self.output
        
        self.pc_layer = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=getattr(config, 'update_bias', True),
            energy_fn_name=getattr(config, 'output_energy_fn_name', 'pc_e'),
            optimizer_name=getattr(config, 'optimizer_name', 'adam'),
            optimizer_beta1=getattr(config, 'optimizer_beta1', 0.9),
            optimizer_beta2=getattr(config, 'optimizer_beta2', 0.999),
            optimizer_eps=getattr(config, 'optimizer_eps', 1e-8),
            optimizer_sign_value=getattr(config, 'optimizer_sign_value', -1.0),
            optimizer_weight_bound=getattr(config, 'optimizer_weight_bound', 0.0),
        )
