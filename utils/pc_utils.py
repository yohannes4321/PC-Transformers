import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from typing import Optional, Tuple, Any
from utils.attention_utils import apply_flash_attention, apply_standard_attention
    
def x_init(batch_size: int, seq_len: int, embedding_size: int, device: torch.device = None) -> torch.Tensor:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return torch.randn(batch_size, seq_len, embedding_size, device = device)


def x_init_iavg(
    prev_hidden_states: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    layer_idx: int,
    hybrid_m: int,
) -> torch.Tensor:
    """
    Average initialization (Iavg) for stream-aligned training.
    
    h^(0,b+1)_l,i = (C/n) * sum_{j:yj=yi} h^(T,b)_l,j
    
    For layers l <= m (hybrid_m), use random initialization (Ifw-like behavior).
    For layers l > m, use average initialization based on class labels.
    
    Args:
        prev_hidden_states: Hidden states from previous batch (batch_size, seq_len, hidden_dim)
        labels: Class labels for each sample (batch_size,)
        num_classes: Number of unique classes
        layer_idx: Current layer index
        hybrid_m: Number of layers to use random init (hybrid approach)
    
    Returns:
        Initialized tensor for current batch
    """
    if layer_idx <= hybrid_m:
        return x_init(prev_hidden_states.size(0), prev_hidden_states.size(1), 
                      prev_hidden_states.size(2), prev_hidden_states.device)
    
    batch_size, seq_len, hidden_dim = prev_hidden_states.shape
    device = prev_hidden_states.device
    
    initialized = torch.zeros_like(prev_hidden_states)
    
    for c in range(num_classes):
        class_mask = (labels == c)
        if class_mask.sum() > 0:
            class_states = prev_hidden_states[class_mask]
            class_avg = class_states.mean(dim=0, keepdim=True)
            initialized[class_mask] = class_avg
    
    return initialized


class HopfieldMemory(nn.Module):
    """
    Hopfield Network as Associative Memory for initialization (Imem).
    Used for unsupervised learning where labels are not available.
    
    h^(0)_l,i = sigma(delta_l * (o_i * Q_l + b_l) * K_l^T) * V_l
    """
    def __init__(self, input_dim: int, memory_dim: int, delta: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.delta = nn.Parameter(torch.tensor(delta))
        self.bias = nn.Parameter(torch.zeros(memory_dim))
        
        self.query = nn.Linear(input_dim, memory_dim)
        self.key = nn.Linear(input_dim, memory_dim)
        self.value = nn.Linear(memory_dim, input_dim)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Retrieve memory states based on observations.
        
        Args:
            observations: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        
        Returns:
            Retrieved memory states
        """
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)
        
        batch_size, seq_len, _ = observations.shape
        
        Q = self.query(observations)
        K = self.key(observations)
        V = self.value.weight.T 
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.delta
        attention = F.softmax(scores, dim=-1)
        
        retrieved = torch.matmul(attention, V)
        
        return retrieved
    
    def init_value_matrix(self, ifw_states: torch.Tensor, observations: torch.Tensor):
        """
        Initialize V matrix to minimize difference between Imem and Ifw.
        V = A^+ * h^(0)_Ifw
        
        Args:
            ifw_states: Hidden states from forward pass initialization
            observations: Input observations
        """
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)
        
        Q = self.query(observations)
        A = F.softmax(self.delta * torch.matmul(Q, self.key(observations).transpose(-2, -1)), dim=-1)
        
        A_pseudo_inv = torch.pinverse(A)
        
        if ifw_states.dim() == 3:
            ifw_states = ifw_states.mean(dim=1)
        
        V_init = torch.matmul(A_pseudo_inv, ifw_states)
        self.value.weight.data = V_init.T


def x_init_imem(
    observations: torch.Tensor,
    hopfield_memory: HopfieldMemory,
) -> torch.Tensor:
    """
    Memory-based initialization (Imem) using Hopfield networks.
    
    Args:
        observations: Input observations (batch_size, seq_len, input_dim)
        hopfield_memory: HopfieldMemory module
    
    Returns:
        Initialized hidden states from memory
    """
    return hopfield_memory(observations)

def step_embed(
    t: int,
    T: int,
    target: torch.Tensor,
    layer: dict,
    layer_type: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    requires_update: bool,
    layer_norm: Optional[nn.Module] = None,
    )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predictive coding step for embedding layer.
    Returns (mu, mu_word, mu_pos, error)
    """
    word_layer: nn.Embedding = layer["word"]
    pos_layer: nn.Embedding = layer["pos"]
    
    # clip ids
    vocab_size = word_layer.weight.size(0)
    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size-1)
    max_pos = pos_layer.weight.size(0)
    if position_ids.max() >= max_pos:
        position_ids = torch.clamp(position_ids, max=max_pos-1)
         
    mu_word = word_layer(input_ids)
    mu_pos = pos_layer(position_ids)
        
    mu = mu_word + mu_pos
    mu_norm=layer_norm(mu) if layer_norm is not None else mu

    error = target - mu_norm
        
    if requires_update: 
        with torch.no_grad():
            flat_input_ids = input_ids.reshape(-1)
            flat_update = error.reshape(-1, error.size(-1))
            flat_position_ids = position_ids.reshape(-1)
            
            delta = local_lr * flat_update
            delta = torch.clamp(delta, -0.01, 0.01)
            
            word_layer.weight.data.index_add_(0, flat_input_ids, delta)
            pos_layer.weight.data.index_add_(0, flat_position_ids, delta)
            
    if t == T - 1:
           finalize_step(mu, target, error, t, layer_type, energy_fn_name)
  
    return mu, mu_word, mu_pos, error
    
def step_linear(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    layer: nn.Module,
    lateral_conn: Optional[Any], 
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module], 
   ):
    """
    Predictive coding step for linear-like layers.
    Returns: (updated_x, mu, bu_err)
    """
    if layer_norm is not None and layer_type == "fc1":
        x_input = layer_norm(x)
    elif layer_type == "fc2":
        x_input = F.gelu(x)
    else:
        x_input = x
        
    mu = layer(x_input)
        
    if layer_type == "fc1":
        mu = F.gelu(mu)
    elif layer_norm is not None and layer_type in ["linear_attn", "fc2"]:
        mu = layer_norm(mu)
            
    if layer_type=="linear_output":
        bu_err= target - F.softmax(mu, dim=-1) 
    else:    
        bu_err = target - mu 
        
    # project bottom-up error through weights
    error_proj= bu_err @ layer.weight      
    error = error_proj- td_err if td_err is not None else error_proj  
    
    if lateral_conn is not None:
        delta_x = lateral_conn.forward(x, error)
        x = x + local_lr * delta_x

        if requires_update:
            lateral_conn.update_weights(x.detach())
    else:
        x= x + local_lr * error 

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))
    
    # parameter updates for the layer
    if requires_update:
        delta_W = local_lr * torch.einsum("bsv, bsh -> vh", bu_err, x_input.detach())
        delta_W = torch.clamp(delta_W, -0.01, 0.01)
        layer.weight.data.add_(delta_W)
        if layer.bias is not None and update_bias:
            delta_b = local_lr * bu_err.mean(dim=(0, 1))
            delta_b = torch.clamp(delta_b, -0.01, 0.01)
            layer.bias.data.add_(delta_b)

    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type,energy_fn_name)

    return x, mu, bu_err

def step_attn(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    lateral_conn: Optional[Any],
    proj_layers: dict,
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    num_heads: int,
    n_embed: int,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module],
    flash: bool = False,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    ):
    """
    Predictive coding step for attention with KV caching support.
    Returns (updated_x, mu, bu_err).
    - proj_layers must contain 'q_proj','k_proj','v_proj' modules
    """
    assert proj_layers is not None, "proj_layers dict is required for attention"

    device = x.device
    
    x_norm=layer_norm(x) if layer_norm is not None else x
        
    q_proj = proj_layers["q_proj"]
    k_proj = proj_layers["k_proj"]
    v_proj = proj_layers["v_proj"]
    assert q_proj is not None and k_proj is not None and v_proj is not None, "Missing Q/K/V projections"  
        
    batch_size, seq_len, embed_dim = target.shape
    head_dim = n_embed // num_heads
   
    Q= q_proj(x_norm)
    
    # KV Cache logic: only compute K,V for new tokens if cache exists
    if use_cache and kv_cache is not None:
        K_new = k_proj(x_norm)
        V_new = v_proj(x_norm)
        
        K_cached, V_cached = kv_cache
        K = torch.cat([K_cached, K_new], dim=1)
        V = torch.cat([V_cached, V_new], dim=1)
    else:
        # Compute full K, V
        K = k_proj(x_norm)
        V = v_proj(x_norm)
    
    new_kv_cache = (K.detach(), V.detach()) if use_cache else None
    Q = Q.view(batch_size, num_heads, seq_len, head_dim)
    K = K.view(batch_size, num_heads, -1, head_dim)
    V = V.view(batch_size, num_heads, -1, head_dim)
        
    #create causal mask (1=keep, 0=mask)
    kv_len = K.size(2)
    causal_mask = torch.tril(torch.ones(seq_len, kv_len, device=device)).unsqueeze(0).unsqueeze(0)

    # !! Causal Mask
    if flash:
        mu_heads = apply_flash_attention(Q, K, V, mask=causal_mask)
    else:
        mu_heads = apply_standard_attention(Q, K, V, mask=causal_mask)
    
    mu = mu_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
    
    bu_err = target - mu  # B, T, D
    error = bu_err - td_err if td_err is not None else bu_err  
                
    if lateral_conn is not None:
        delta_x = lateral_conn.forward(x, error)
        x = x + local_lr * delta_x
        
        if requires_update:
            lateral_conn.update_weights(x.detach())
    else:
        x = x + local_lr * error

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    # PC update W_latent
    if requires_update:
        with torch.no_grad():
            B, S = batch_size, seq_len

            K_update = K[:, :, -seq_len:, :] 
            V_update = V[:, :, -seq_len:, :]
            
            # Multi-head Q, K, V updates
            for h in range(num_heads):
                q_slice = Q[:, h, :, :]  # [B, S, head_dim]
                k_slice = K_update[:, h, :, :]
                v_slice = V_update[:, h, :, :]
                
                dW_q_h = torch.einsum("bsd,bse->de", q_slice, x_norm) / (B * S)
                dW_k_h = torch.einsum("bsd,bse->de", k_slice, x_norm) / (B * S)
                dW_v_h = torch.einsum("bsd,bse->de", v_slice, x_norm) / (B * S)

                start = h * head_dim
                end = (h + 1) * head_dim
                
                q_proj.weight.data[start:end, :] += torch.clamp(local_lr * dW_q_h, -clamp_value, clamp_value)
                k_proj.weight.data[start:end, :] += torch.clamp(local_lr * dW_k_h, -clamp_value, clamp_value)
                v_proj.weight.data[start:end, :] += torch.clamp(local_lr * dW_v_h, -clamp_value, clamp_value)
                
                if update_bias:
                    if q_proj.bias is not None:
                        delta_b_q = (q_slice.mean(dim=(0, 1)) / (B * S))
                        q_proj.bias.data[start:end] += torch.clamp(local_lr * delta_b_q, -clamp_value, clamp_value)
                    if k_proj.bias is not None:
                        delta_b_k = (k_slice.mean(dim=(0, 1)) / (B * S))
                        k_proj.bias.data[start:end] += torch.clamp(local_lr * delta_b_k, -clamp_value, clamp_value)
                    if v_proj.bias is not None:
                        delta_b_v = (v_slice.mean(dim=(0, 1)) / (B * S))
                        v_proj.bias.data[start:end] += torch.clamp(local_lr * delta_b_v, -clamp_value, clamp_value)
 
    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type,energy_fn_name)
     
    return x, mu, bu_err, new_kv_cache
    
ENERGY_FUNCTIONS = {
    "pc_e": lambda mu, x: ((mu - x) ** 2) * 0.5,    
    "kld": lambda mu, x: torch.clamp(
        F.kl_div(mu.log_softmax(dim=-1), x, reduction="batchmean"), min=0.0, max=100.0
    ),
}

def energy_fn(mu: torch.Tensor, x: torch.Tensor,energy_fn_name: str) -> torch.Tensor:
    if energy_fn_name not in ENERGY_FUNCTIONS:
        raise ValueError(f"Unknown energy function: {energy_fn_name}. Choose from {list(ENERGY_FUNCTIONS.keys())}")
    return ENERGY_FUNCTIONS[energy_fn_name](mu, x)

def finalize_step(mu: torch.Tensor, target: torch.Tensor, error: torch.Tensor, t: int, layer_type: str, energy_fn_name: str):
    device = mu.device
    target = target.to(device)
    error = error.to(device)
    energy = float(energy_fn(mu, target, energy_fn_name).mean().item())
    errors = [{"step": t, "type": layer_type, "error": error.mean().item()}]
    return energy, errors
    
def ids_to_one_hot(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    device = input_ids.device
    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size-1)
    return F.one_hot(input_ids, num_classes=vocab_size).float().to(device)

def cleanup_memory():
    """Comprehensive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
