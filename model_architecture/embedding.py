import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class Embedding_Layer(nn.Module):
    """
    Embedding layer with word and positional embeddings, layer normalization, dropout, and a predictive coding layer.
    """
    def __init__(self, config):
        super(Embedding_Layer, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embed)
        self.rms_norm = nn.RMSNorm(config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        
        self.pc_layer= PCLayer(
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
