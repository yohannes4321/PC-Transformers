import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class Attention(nn.Module):
    """
    Multi-head self-attention module with predictive coding layers for use in transformer architectures.
    Computes attention scores, applies masking, and outputs context vectors.
    Includes KV caching for efficient generation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.num_heads
        self.dropout = nn.Dropout(config.dropout)

        self.q = nn.Linear(config.n_embed, config.n_embed)
        self.k = nn.Linear(config.n_embed, config.n_embed)
        self.v = nn.Linear(config.n_embed, config.n_embed)
        self.output = nn.Linear(config.n_embed, config.n_embed)

        # Latent-to-latent connector weights
        self.q_and_score = nn.Linear(self.head_dim, config.block_size)
        self.k_and_score = nn.Linear(self.head_dim, config.block_size)
        self.v_and_attenout = nn.Linear(self.head_dim, self.head_dim)
        self.score_X_A = nn.Linear(config.block_size, config.block_size)
        self.X_A_and_Attenout = nn.Linear(config.block_size, self.head_dim)

        self.pc_X_Q = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=getattr(config, 'update_bias', True),
            energy_fn_name=getattr(config, 'internal_energy_fn_name', 'pc_e'),
            num_heads=config.num_heads,
            n_embed=config.n_embed,
            optimizer_name=getattr(config, 'optimizer_name', 'adam'),
            optimizer_beta1=getattr(config, 'optimizer_beta1', 0.9),
            optimizer_beta2=getattr(config, 'optimizer_beta2', 0.999),
            optimizer_eps=getattr(config, 'optimizer_eps', 1e-8),
            optimizer_sign_value=getattr(config, 'optimizer_sign_value', -1.0),
            optimizer_weight_bound=getattr(config, 'optimizer_weight_bound', 0.0),
        )

        self.pc_X_K = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=getattr(config, 'update_bias', True),
            energy_fn_name=getattr(config, 'internal_energy_fn_name', 'pc_e'),
            num_heads=config.num_heads,
            n_embed=config.n_embed,
            optimizer_name=getattr(config, 'optimizer_name', 'adam'),
            optimizer_beta1=getattr(config, 'optimizer_beta1', 0.9),
            optimizer_beta2=getattr(config, 'optimizer_beta2', 0.999),
            optimizer_eps=getattr(config, 'optimizer_eps', 1e-8),
            optimizer_sign_value=getattr(config, 'optimizer_sign_value', -1.0),
            optimizer_weight_bound=getattr(config, 'optimizer_weight_bound', 0.0),
        )

        self.pc_X_V = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=getattr(config, 'update_bias', True),
            energy_fn_name=getattr(config, 'internal_energy_fn_name', 'pc_e'),
            num_heads=config.num_heads,
            n_embed=config.n_embed,
            optimizer_name=getattr(config, 'optimizer_name', 'adam'),
            optimizer_beta1=getattr(config, 'optimizer_beta1', 0.9),
            optimizer_beta2=getattr(config, 'optimizer_beta2', 0.999),
            optimizer_eps=getattr(config, 'optimizer_eps', 1e-8),
            optimizer_sign_value=getattr(config, 'optimizer_sign_value', -1.0),
            optimizer_weight_bound=getattr(config, 'optimizer_weight_bound', 0.0),
        )

        self.pc_X_score = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=getattr(config, 'update_bias', True),
            energy_fn_name=getattr(config, 'internal_energy_fn_name', 'pc_e'),
            num_heads=config.num_heads,
            n_embed=config.n_embed,
            optimizer_name=getattr(config, 'optimizer_name', 'adam'),
            optimizer_beta1=getattr(config, 'optimizer_beta1', 0.9),
            optimizer_beta2=getattr(config, 'optimizer_beta2', 0.999),
            optimizer_eps=getattr(config, 'optimizer_eps', 1e-8),
            optimizer_sign_value=getattr(config, 'optimizer_sign_value', -1.0),
            optimizer_weight_bound=getattr(config, 'optimizer_weight_bound', 0.0),
        )

        self.pc_X_A = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=getattr(config, 'update_bias', True),
            energy_fn_name=getattr(config, 'internal_energy_fn_name', 'pc_e'),
            num_heads=config.num_heads,
            n_embed=config.n_embed,
            optimizer_name=getattr(config, 'optimizer_name', 'adam'),
            optimizer_beta1=getattr(config, 'optimizer_beta1', 0.9),
            optimizer_beta2=getattr(config, 'optimizer_beta2', 0.999),
            optimizer_eps=getattr(config, 'optimizer_eps', 1e-8),
            optimizer_sign_value=getattr(config, 'optimizer_sign_value', -1.0),
            optimizer_weight_bound=getattr(config, 'optimizer_weight_bound', 0.0),
        )

        self.pc_output = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=getattr(config, 'update_bias', True),
            energy_fn_name=getattr(config, 'internal_energy_fn_name', 'pc_e'),
            num_heads=config.num_heads,
            n_embed=config.n_embed,
            optimizer_name=getattr(config, 'optimizer_name', 'adam'),
            optimizer_beta1=getattr(config, 'optimizer_beta1', 0.9),
            optimizer_beta2=getattr(config, 'optimizer_beta2', 0.999),
            optimizer_eps=getattr(config, 'optimizer_eps', 1e-8),
            optimizer_sign_value=getattr(config, 'optimizer_sign_value', -1.0),
            optimizer_weight_bound=getattr(config, 'optimizer_weight_bound', 0.0),
        )
        
        # KV cache for generation: stores (K, V) tensors
        self.kv_cache = None
        
    def clear_kv_cache(self):
        """Clear the KV cache"""
        self.kv_cache = None
