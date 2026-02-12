import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from typing import Optional, Tuple, Any
from utils.attention_utils import apply_flash_attention, apply_standard_attention
from utils.optim.optim_utils import PCOptimizer
    
def x_init(batch_size: int, seq_len: int, embedding_size: int, device: torch.device = None) -> torch.Tensor:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return torch.randn(batch_size, seq_len, embedding_size, device = device)

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
    optimizer: Optional[PCOptimizer] = None,
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

            if optimizer is not None:
                update_word = torch.zeros_like(word_layer.weight)
                update_pos = torch.zeros_like(pos_layer.weight)
                update_word.index_add_(0, flat_input_ids, flat_update)
                update_pos.index_add_(0, flat_position_ids, flat_update)

                optimizer.step_param(word_layer.weight, update_word, local_lr, clamp_value=0.01)
                optimizer.step_param(pos_layer.weight, update_pos, local_lr, clamp_value=0.01)
            else:
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
    optimizer: Optional[PCOptimizer] = None,
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
            lateral_conn.update_weights(x.detach(), optimizer=optimizer, clamp_value=0.01)
    else:
        x= x + local_lr * error 

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))
    
    # parameter updates for the layer
    if requires_update:
        update_w = torch.einsum("bsv, bsh -> vh", bu_err, x_input.detach())
        if optimizer is not None:
            optimizer.step_param(layer.weight, update_w, local_lr, clamp_value=0.01)
        else:
            delta_W = torch.clamp(local_lr * update_w, -0.01, 0.01)
            layer.weight.data.add_(delta_W)

        if layer.bias is not None and update_bias:
            update_b = bu_err.mean(dim=(0, 1))
            if optimizer is not None:
                optimizer.step_param(layer.bias, update_b, local_lr, clamp_value=0.01)
            else:
                delta_b = torch.clamp(local_lr * update_b, -0.01, 0.01)
                layer.bias.data.add_(delta_b)

    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type,energy_fn_name)

    return x, mu, bu_err


def _reshape_to_heads(tensor: Optional[torch.Tensor], num_heads: int) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.dim() == 4:
        batch_size, dim1, seq_len, last_dim = tensor.shape
        # If this is already in the correct format (B, H, S, D), return it
        if dim1 == num_heads and last_dim * num_heads == tensor.numel() // (batch_size * seq_len):
            return tensor
        # Otherwise, assume it's (B, S, H*D) or similar and reshape it
        total_embed_dim = dim1 * last_dim
        head_dim = total_embed_dim // num_heads
        return tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    batch_size, seq_len, embed_dim = tensor.shape
    head_dim = embed_dim // num_heads
    return tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def _score_target_to_q(score_target: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Map attention-score targets to Q space using K."""
    return torch.matmul(score_target, k)


def _score_target_to_k(score_target: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Map attention-score targets to K space using Q."""
    return torch.matmul(score_target.transpose(-1, -2), q)


def _merge_heads(tensor: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, seq_len, head_dim = tensor.shape
    return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)


def step_Q(
    t: int,
    T: int,
    target: Optional[torch.Tensor],
    x: torch.Tensor,
    lateral_conn: Optional[Any],
    proj_layers: dict,
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    n_embed: int,
    num_heads: int,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module],
    optimizer: Optional[PCOptimizer] = None,
):
    if x.dim() == 4:
        x_input = _merge_heads(x)
    else:
        x_input = x
    x_input = layer_norm(x_input) if layer_norm is not None else x_input
    q_proj = proj_layers["q_proj"]
    q = q_proj(x_input)

    q = _reshape_to_heads(q, num_heads)
    if target is not None and target.dim() == 4 and target.shape[-1] != q.shape[-1]:
        k_proj = proj_layers["k_proj"]
        k = _reshape_to_heads(k_proj(x_input), num_heads)
        target_q = _score_target_to_q(target, k)
    else:
        target_q = _reshape_to_heads(target, num_heads) if target is not None else q
    bu_err = target_q - q
    bu_err_flat = _merge_heads(bu_err)
    error_proj = bu_err_flat @ q_proj.weight
    error_proj = torch.clamp(error_proj, -clamp_value, clamp_value)

    if td_err is not None:
        td_err_heads = _reshape_to_heads(td_err, num_heads)
        td_err_flat = _merge_heads(td_err_heads)
        td_err_proj = td_err_flat @ q_proj.weight
        td_err_proj = torch.clamp(td_err_proj, -clamp_value, clamp_value)
        error = error_proj - td_err_proj
    else:
        error = error_proj

    error_for_update = _reshape_to_heads(error, num_heads) if x.dim() == 4 else error
    if lateral_conn is not None:
        delta_x = lateral_conn.forward(x, error_for_update)
        x = x + local_lr * delta_x
        if requires_update:
            lateral_conn.update_weights(x.detach(), optimizer=optimizer, clamp_value=clamp_value)
    else:
        x = x + local_lr * error_for_update

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    if requires_update:
        with torch.no_grad():
            batch_size, seq_len = q.size(0), q.size(2)
            head_dim = n_embed // num_heads
            update_q = torch.zeros_like(q_proj.weight)
            update_b_q = torch.zeros_like(q_proj.bias) if q_proj.bias is not None else None
            for h in range(num_heads):
                q_slice = q[:, h, :, :]
                dW_q_h = torch.einsum("bsd,bse->de", q_slice, x_input) / (batch_size * seq_len)
                start = h * head_dim
                end = (h + 1) * head_dim

                update_q[start:end, :] = dW_q_h
                if update_bias and update_b_q is not None:
                    update_b_q[start:end] = q_slice.mean(dim=(0, 1)) / (batch_size * seq_len)

            if optimizer is not None:
                optimizer.step_param(q_proj.weight, update_q, local_lr, clamp_value=clamp_value)
                if update_bias and update_b_q is not None:
                    optimizer.step_param(q_proj.bias, update_b_q, local_lr, clamp_value=clamp_value)
            else:
                q_proj.weight.data.add_(torch.clamp(local_lr * update_q, -clamp_value, clamp_value))
                if update_bias and update_b_q is not None:
                    q_proj.bias.data.add_(torch.clamp(local_lr * update_b_q, -clamp_value, clamp_value))

    if t == T - 1:
        finalize_step(q, target_q, bu_err, t, layer_type, energy_fn_name)

    return x, q, bu_err


def step_K(
    t: int,
    T: int,
    target: Optional[torch.Tensor],
    x: torch.Tensor,
    lateral_conn: Optional[Any],
    proj_layers: dict,
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    n_embed: int,
    num_heads: int,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module],
    optimizer: Optional[PCOptimizer] = None,
):
    if x.dim() == 4:
        x_input = _merge_heads(x)
    else:
        x_input = x
    x_input = layer_norm(x_input) if layer_norm is not None else x_input
    k_proj = proj_layers["k_proj"]
    k = k_proj(x_input)

    k = _reshape_to_heads(k, num_heads)
    if target is not None and target.dim() == 4 and target.shape[-1] != k.shape[-1]:
        q_proj = proj_layers["q_proj"]
        q = _reshape_to_heads(q_proj(x_input), num_heads)
        target_k = _score_target_to_k(target, q)
    else:
        target_k = _reshape_to_heads(target, num_heads) if target is not None else k

    bu_err = target_k - k
    bu_err_flat = _merge_heads(bu_err)
    error_proj = bu_err_flat @ k_proj.weight
    error_proj = torch.clamp(error_proj, -clamp_value, clamp_value)

    if td_err is not None:
        td_err_heads = _reshape_to_heads(td_err, num_heads)
        td_err_flat = _merge_heads(td_err_heads)
        td_err_proj = td_err_flat @ k_proj.weight
        td_err_proj = torch.clamp(td_err_proj, -clamp_value, clamp_value)
        error = error_proj - td_err_proj
    else:
        error = error_proj

    error_for_update = _reshape_to_heads(error, num_heads) if x.dim() == 4 else error
    if lateral_conn is not None:
        delta_x = lateral_conn.forward(x, error_for_update)
        x = x + local_lr * delta_x
        if requires_update:
            lateral_conn.update_weights(x.detach(), optimizer=optimizer, clamp_value=clamp_value)
    else:
        x = x + local_lr * error_for_update

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    if requires_update:
        with torch.no_grad():
            batch_size, seq_len = k.size(0), k.size(2)
            head_dim = n_embed // num_heads
            update_k = torch.zeros_like(k_proj.weight)
            update_b_k = torch.zeros_like(k_proj.bias) if k_proj.bias is not None else None
            for h in range(num_heads):
                k_slice = k[:, h, :, :]
                dW_k_h = torch.einsum("bsd,bse->de", k_slice, x_input) / (batch_size * seq_len)
                start = h * head_dim
                end = (h + 1) * head_dim

                update_k[start:end, :] = dW_k_h
                if update_bias and update_b_k is not None:
                    update_b_k[start:end] = k_slice.mean(dim=(0, 1)) / (batch_size * seq_len)

            if optimizer is not None:
                optimizer.step_param(k_proj.weight, update_k, local_lr, clamp_value=clamp_value)
                if update_bias and update_b_k is not None:
                    optimizer.step_param(k_proj.bias, update_b_k, local_lr, clamp_value=clamp_value)
            else:
                k_proj.weight.data.add_(torch.clamp(local_lr * update_k, -clamp_value, clamp_value))
                if update_bias and update_b_k is not None:
                    k_proj.bias.data.add_(torch.clamp(local_lr * update_b_k, -clamp_value, clamp_value))

    if t == T - 1:
        finalize_step(k, target_k, bu_err, t, layer_type, energy_fn_name)

    return x, k, bu_err




def step_V(
    t: int,
    T: int,
    target: Optional[torch.Tensor],
    x: torch.Tensor,
    lateral_conn: Optional[Any],
    proj_layers: dict,
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    n_embed: int,
    num_heads: int,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module],
    optimizer: Optional[PCOptimizer] = None,
):
    if x.dim() == 4:
        x_input = _merge_heads(x)
    else:
        x_input = x
    x_input = layer_norm(x_input) if layer_norm is not None else x_input
    v_proj = proj_layers["v_proj"]
    v = v_proj(x_input)

    v = _reshape_to_heads(v, num_heads)
    target_v = _reshape_to_heads(target, num_heads) if target is not None else v

    bu_err = target_v - v
    bu_err_flat = _merge_heads(bu_err)
    error_proj = bu_err_flat @ v_proj.weight

    if td_err is not None:
        td_err_heads = _reshape_to_heads(td_err, num_heads)
        td_err_flat = _merge_heads(td_err_heads)
        td_err_proj = td_err_flat @ v_proj.weight
        error = error_proj - td_err_proj
    else:
        error = error_proj

    error_for_update = _reshape_to_heads(error, num_heads) if x.dim() == 4 else error
    if lateral_conn is not None:
        delta_x = lateral_conn.forward(x, error_for_update)
        x = x + local_lr * delta_x
        if requires_update:
            lateral_conn.update_weights(x.detach(), optimizer=optimizer, clamp_value=clamp_value)
    else:
        x = x + local_lr * error_for_update

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    if requires_update:
        with torch.no_grad():
            batch_size, seq_len = v.size(0), v.size(2)
            head_dim = n_embed // num_heads
            update_v = torch.zeros_like(v_proj.weight)
            update_b_v = torch.zeros_like(v_proj.bias) if v_proj.bias is not None else None
            for h in range(num_heads):
                v_slice = v[:, h, :, :]
                dW_v_h = torch.einsum("bsd,bse->de", v_slice, x_input) / (batch_size * seq_len)
                start = h * head_dim
                end = (h + 1) * head_dim

                update_v[start:end, :] = dW_v_h
                if update_bias and update_b_v is not None:
                    update_b_v[start:end] = v_slice.mean(dim=(0, 1)) / (batch_size * seq_len)

            if optimizer is not None:
                optimizer.step_param(v_proj.weight, update_v, local_lr, clamp_value=clamp_value)
                if update_bias and update_b_v is not None:
                    optimizer.step_param(v_proj.bias, update_b_v, local_lr, clamp_value=clamp_value)
            else:
                v_proj.weight.data.add_(torch.clamp(local_lr * update_v, -clamp_value, clamp_value))
                if update_bias and update_b_v is not None:
                    v_proj.bias.data.add_(torch.clamp(local_lr * update_b_v, -clamp_value, clamp_value))

    if t == T - 1:
        finalize_step(v, target_v, bu_err, t, layer_type, energy_fn_name)

    return x, v, bu_err


def step_X_score(
    t: int,
    T: int,
    target: Optional[torch.Tensor],
    x: torch.Tensor,
    lateral_conn: Optional[Any],
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    requires_update: bool,
    num_heads: int,
    n_embed: int,
    layer:nn.Module,
    td_err: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
):
    if q is None or k is None:
        raise ValueError("X_score requires both Q and K tensors")
    if use_cache and kv_cache is not None:
        k_cached, _ = kv_cache
        k = torch.cat([k_cached, k], dim=2)

    head_dim = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

    seq_len = q.size(2)
    kv_len = k.size(2)
    device = scores.device
    causal_mask = torch.tril(torch.ones(seq_len, kv_len, device=device)).unsqueeze(0).unsqueeze(0)
    # Avoid -inf which can blow up energy calculations downstream.
    scores = scores.masked_fill(causal_mask == 0, -1e4)
    mu=layer(scores)

    
    target_scores = target if target is not None else mu
    bu_err = target_scores - mu
    error = bu_err - td_err if td_err is not None else bu_err

    if lateral_conn is not None:
        x_lat = x.permute(0, 2, 3, 1)
        error_lat = error.permute(0, 2, 3, 1)
        if lateral_conn.size != x_lat.size(-1):
            lateral_conn = None
        else:
            delta_x = lateral_conn.forward(x_lat, error_lat)
            x = x + local_lr * delta_x.permute(0, 3, 1, 2)
            if requires_update:
                lateral_conn.update_weights(x_lat.detach(), optimizer=None, clamp_value=clamp_value)
    else:
        x = x + local_lr * error

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    if t == T - 1:
        finalize_step(mu, target_scores, bu_err, t, layer_type, energy_fn_name)

    new_kv_cache = (k.detach(), None) if use_cache else None
    return x, mu, bu_err, new_kv_cache


def step_X_A(
    t: int,
    T: int,
    target: Optional[torch.Tensor],
    x: torch.Tensor,
    lateral_conn: Optional[Any],
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    requires_update: bool,
    td_err: Optional[torch.Tensor],
    score: torch.Tensor,
):
    if score is None:
        raise ValueError("X_A requires a score tensor")
    mu_logits = x if x is not None else score
    mu = torch.softmax(mu_logits, dim=-1)
    target_probs = torch.softmax(score, dim=-1)
    bu_err = target_probs - mu
    error = bu_err - td_err if td_err is not None else bu_err

    if lateral_conn is not None:
        x_lat = x.permute(0, 2, 3, 1)
        error_lat = error.permute(0, 2, 3, 1)
        if lateral_conn.size != x_lat.size(-1):
            lateral_conn = None
        else:
            delta_x = lateral_conn.forward(x_lat, error_lat)
            x = x + local_lr * delta_x.permute(0, 3, 1, 2)
            if requires_update:
                lateral_conn.update_weights(x_lat.detach(), optimizer=None, clamp_value=clamp_value)
    else:
        x = x + local_lr * error

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    if t == T - 1:
        finalize_step(mu_logits, target_probs, bu_err, t, layer_type, energy_fn_name)

    return x, mu, bu_err


def step_X_attnOut(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    lateral_conn: Optional[Any],
    layer: nn.Module,
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module],
    a_weights: torch.Tensor,
    v: torch.Tensor,
    optimizer: Optional[PCOptimizer] = None,
):
    if a_weights is None or v is None:
        raise ValueError("attn_output requires both attention weights and V tensors")
    context_heads = torch.matmul(a_weights, v)
    context = _merge_heads(context_heads)

    x_input = context
    mu = layer(x_input)
    if layer_norm is not None:
        mu = layer_norm(mu)

    bu_err = target - mu
    error_proj = bu_err @ layer.weight
    if td_err is not None and td_err.dim() == 4:
        td_err = _merge_heads(td_err)
    error = error_proj - td_err if td_err is not None else error_proj

    num_heads = v.shape[1]
    error_for_update = _reshape_to_heads(error, num_heads) if x.dim() == 4 else error
    if lateral_conn is not None:
        if lateral_conn.size != x.size(-1):
            lateral_conn = None
        else:
            delta_x = lateral_conn.forward(x, error_for_update)
            x = x + local_lr * delta_x
            if requires_update:
                lateral_conn.update_weights(x.detach(), optimizer=optimizer, clamp_value=clamp_value)
    else:
        x = x + local_lr * error_for_update

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    if requires_update:
        update_w = torch.einsum("bsv,bsh->vh", bu_err, x_input.detach())
        if optimizer is not None:
            optimizer.step_param(layer.weight, update_w, local_lr, clamp_value=0.01)
        else:
            delta_w = torch.clamp(local_lr * update_w, -0.01, 0.01)
            layer.weight.data.add_(delta_w)

        if layer.bias is not None and update_bias:
            update_b = bu_err.mean(dim=(0, 1))
            if optimizer is not None:
                optimizer.step_param(layer.bias, update_b, local_lr, clamp_value=0.01)
            else:
                delta_b = torch.clamp(local_lr * update_b, -0.01, 0.01)
                layer.bias.data.add_(delta_b)

    if t == T - 1:
        finalize_step(mu, target, bu_err, t, layer_type, energy_fn_name)

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
    optimizer: Optional[PCOptimizer] = None,
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
            lateral_conn.update_weights(x.detach(), optimizer=optimizer, clamp_value=clamp_value)
    else:
        x = x + local_lr * error

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    # PC update W_latent using back-projected errors
    if requires_update:
        with torch.no_grad():
            B, S = batch_size, seq_len

            K_update = K[:, :, -seq_len:, :]
            V_update = V[:, :, -seq_len:, :]

            scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
            scores = scores.masked_fill(causal_mask == 0, -1e4)
            attn = torch.softmax(scores, dim=-1)

            error_heads = _reshape_to_heads(error, num_heads)
            err_v = torch.matmul(attn.transpose(-1, -2), error_heads)

            err_scores = torch.matmul(error_heads, V.transpose(-1, -2))
            err_scores = (err_scores - (err_scores * attn).sum(dim=-1, keepdim=True)) * attn

            err_q = torch.matmul(err_scores, K) / (head_dim ** 0.5)
            err_k = torch.matmul(err_scores.transpose(-1, -2), Q) / (head_dim ** 0.5)

            update_q = torch.zeros_like(q_proj.weight)
            update_k = torch.zeros_like(k_proj.weight)
            update_v = torch.zeros_like(v_proj.weight)

            update_b_q = torch.zeros_like(q_proj.bias) if q_proj.bias is not None else None
            update_b_k = torch.zeros_like(k_proj.bias) if k_proj.bias is not None else None
            update_b_v = torch.zeros_like(v_proj.bias) if v_proj.bias is not None else None

            # Multi-head Q, K, V updates
            for h in range(num_heads):
                err_q_h = err_q[:, h, :, :]
                err_k_h = err_k[:, h, :, :]
                err_v_h = err_v[:, h, :, :]

                dW_q_h = torch.einsum("bsd,bse->de", err_q_h, x_norm) / (B * S)
                dW_k_h = torch.einsum("bsd,bse->de", err_k_h, x_norm) / (B * S)
                dW_v_h = torch.einsum("bsd,bse->de", err_v_h, x_norm) / (B * S)

                start = h * head_dim
                end = (h + 1) * head_dim

                update_q[start:end, :] = dW_q_h
                update_k[start:end, :] = dW_k_h
                update_v[start:end, :] = dW_v_h

                if update_bias:
                    if update_b_q is not None:
                        update_b_q[start:end] = err_q_h.mean(dim=(0, 1)) / (B * S)
                    if update_b_k is not None:
                        update_b_k[start:end] = err_k_h.mean(dim=(0, 1)) / (B * S)
                    if update_b_v is not None:
                        update_b_v[start:end] = err_v_h.mean(dim=(0, 1)) / (B * S)

            if optimizer is not None:
                optimizer.step_param(q_proj.weight, update_q, local_lr, clamp_value=clamp_value)
                optimizer.step_param(k_proj.weight, update_k, local_lr, clamp_value=clamp_value)
                optimizer.step_param(v_proj.weight, update_v, local_lr, clamp_value=clamp_value)

                if update_bias:
                    if update_b_q is not None:
                        optimizer.step_param(q_proj.bias, update_b_q, local_lr, clamp_value=clamp_value)
                    if update_b_k is not None:
                        optimizer.step_param(k_proj.bias, update_b_k, local_lr, clamp_value=clamp_value)
                    if update_b_v is not None:
                        optimizer.step_param(v_proj.bias, update_b_v, local_lr, clamp_value=clamp_value)
            else:
                q_proj.weight.data.add_(torch.clamp(local_lr * update_q, -clamp_value, clamp_value))
                k_proj.weight.data.add_(torch.clamp(local_lr * update_k, -clamp_value, clamp_value))
                v_proj.weight.data.add_(torch.clamp(local_lr * update_v, -clamp_value, clamp_value))

                if update_bias:
                    if update_b_q is not None:
                        q_proj.bias.data.add_(torch.clamp(local_lr * update_b_q, -clamp_value, clamp_value))
                    if update_b_k is not None:
                        k_proj.bias.data.add_(torch.clamp(local_lr * update_b_k, -clamp_value, clamp_value))
                    if update_b_v is not None:
                        v_proj.bias.data.add_(torch.clamp(local_lr * update_b_v, -clamp_value, clamp_value))
 
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