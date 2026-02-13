import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from typing import Optional, Tuple, Any
from utils.attention_utils import apply_flash_attention, apply_standard_attention
from utils.optim.optim_utils import PCOptimizer

def _apply_linear(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return None
    if x.dim() == 4:
        b, h, s, d = x.shape
        y = layer(x.reshape(b * h * s, d))
        return y.reshape(b, h, s, -1)
    if x.dim() == 3:
        b, s, d = x.shape
        y = layer(x.reshape(b * s, d))
        return y.reshape(b, s, -1)
    return layer(x)

def _flatten_for_update(error: torch.Tensor, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if error.dim() == 4:
        b, h, s, d = error.shape
        err = error.reshape(b * h * s, d)
        inp_flat = inp.reshape(b * h * s, inp.shape[-1])
        return err, inp_flat
    if error.dim() == 3:
        b, s, d = error.shape
        err = error.reshape(b * s, d)
        inp_flat = inp.reshape(b * s, inp.shape[-1])
        return err, inp_flat
    return error, inp

def _backproject_error(layer: nn.Module, error: torch.Tensor) -> torch.Tensor:
    if error.dim() == 4:
        b, h, s, d = error.shape
        err = error.reshape(b * h * s, d)
        proj = err @ layer.weight
        return proj.reshape(b, h, s, -1)
    if error.dim() == 3:
        b, s, d = error.shape
        err = error.reshape(b * s, d)
        proj = err @ layer.weight
        return proj.reshape(b, s, -1)
    return error @ layer.weight

def step_bi(
    t: int,
    T: int,
    target: Optional[torch.Tensor],
    current: torch.Tensor,
    previous: Optional[torch.Tensor],
    bottom_layer: Optional[nn.Module],
    top_layer: Optional[nn.Module],
    lateral_conn: Optional[Any],
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    layer_norm: Optional[nn.Module],
    optimizer: Optional[PCOptimizer] = None,
    previous_2: Optional[torch.Tensor] = None,
    bottom_layer_2: Optional[nn.Module] = None,
    bottom_chain: bool = False,
):
    if current is None:
        raise ValueError(f"Current state is required for layer_type={layer_type}")

    prev_input = previous
    if prev_input is not None and prev_input.dim() == 4 and bottom_layer is not None:
        merged = _merge_heads(prev_input)
        if merged.shape[-1] == bottom_layer.weight.shape[1]:
            prev_input = merged
    if layer_norm is not None and prev_input is not None and prev_input.dim() == 3:
        prev_input = layer_norm(prev_input)

    bottom_pred_1 = _apply_linear(bottom_layer, prev_input) if bottom_layer is not None else None

    bottom_pred_2 = None
    if bottom_layer_2 is not None and previous_2 is not None:
        if bottom_chain and bottom_layer is not None:
            inter = _apply_linear(bottom_layer_2, previous_2)
            bottom_pred_2 = _apply_linear(bottom_layer, inter)
        else:
            bottom_pred_2 = _apply_linear(bottom_layer_2, previous_2)

    if bottom_pred_1 is not None and bottom_pred_2 is not None:
        bottom_pred_raw = 0.5 * (bottom_pred_1 + bottom_pred_2)
    else:
        bottom_pred_raw = bottom_pred_1 if bottom_pred_1 is not None else bottom_pred_2

    current_for_bottom = current
    if current_for_bottom is not None and current_for_bottom.dim() == 4 and bottom_pred_raw is not None and bottom_pred_raw.dim() == 3:
        current_for_bottom = _merge_heads(current_for_bottom)

    if bottom_pred_raw is not None:
        bottom_error_raw = current_for_bottom - bottom_pred_raw
    else:
        bottom_error_raw = torch.zeros_like(current_for_bottom if current_for_bottom is not None else current)

    bottom_pred = bottom_pred_raw
    if current.dim() == 4 and bottom_pred is not None and bottom_pred.dim() == 3:
        b, h, s, d = current.shape
        if bottom_pred.shape[-1] == h * d:
            bottom_pred = _reshape_to_heads(bottom_pred, h)

    if bottom_pred is not None:
        bottom_error = current - bottom_pred
    else:
        bottom_error = torch.zeros_like(current)

    current_input = current
    if current_input is not None and current_input.dim() == 4 and top_layer is not None:
        merged = _merge_heads(current_input)
        if merged.shape[-1] == top_layer.weight.shape[1]:
            current_input = merged
    if layer_norm is not None and current_input.dim() == 3:
        current_input = layer_norm(current_input)

    if top_layer is not None:
        top_pred = _apply_linear(top_layer, current_input)
    else:
        top_pred = current_input if target is not None else None

    top_error_raw = None
    if target is not None and top_pred is not None:
        top_error_raw = target - top_pred
        if top_layer is not None:
            top_error = _backproject_error(top_layer, top_error_raw)
        else:
            top_error = top_error_raw
    else:
        top_error = torch.zeros_like(current)

    if current.dim() == 4 and top_error.dim() == 3:
        b, h, s, d = current.shape
        if top_error.shape[-1] == h * d:
            top_error = _reshape_to_heads(top_error, h)

    total_error = -bottom_error + top_error

    if lateral_conn is not None:
        delta_x = lateral_conn.forward(current, total_error)
        current = current + local_lr * delta_x
        if requires_update:
            lateral_conn.update_weights(current.detach(), optimizer=optimizer, clamp_value=clamp_value)
    else:
        current = current + local_lr * total_error

    current = torch.clamp(current, -abs(clamp_value), abs(clamp_value))

    if requires_update and bottom_layer is not None and prev_input is not None:
        err, inp = _flatten_for_update(bottom_error_raw, prev_input)
        update_w = err.t() @ inp / max(err.size(0), 1)
        if optimizer is not None:
            optimizer.step_param(bottom_layer.weight, update_w, local_lr, clamp_value=0.01)
        else:
            bottom_layer.weight.data.add_(torch.clamp(local_lr * update_w, -0.01, 0.01))
        if bottom_layer.bias is not None and update_bias:
            update_b = err.mean(dim=0)
            if optimizer is not None:
                optimizer.step_param(bottom_layer.bias, update_b, local_lr, clamp_value=0.01)
            else:
                bottom_layer.bias.data.add_(torch.clamp(local_lr * update_b, -0.01, 0.01))

    if requires_update and top_layer is not None and top_error_raw is not None:
        err, inp = _flatten_for_update(top_error_raw, current_input)
        update_w = err.t() @ inp / max(err.size(0), 1)
        if optimizer is not None:
            optimizer.step_param(top_layer.weight, update_w, local_lr, clamp_value=0.01)
        else:
            top_layer.weight.data.add_(torch.clamp(local_lr * update_w, -0.01, 0.01))
        if top_layer.bias is not None and update_bias:
            update_b = err.mean(dim=0)
            if optimizer is not None:
                optimizer.step_param(top_layer.bias, update_b, local_lr, clamp_value=0.01)
            else:
                top_layer.bias.data.add_(torch.clamp(local_lr * update_b, -0.01, 0.01))

    if t == T - 1 and target is not None:
        energy_mu = top_pred if top_pred is not None else current
        if current.dim() == 4 and energy_mu is not None and energy_mu.dim() == 3:
            b, h, s, d = current.shape
            if energy_mu.shape[-1] == h * d:
                energy_mu = _reshape_to_heads(energy_mu, h)
        error = target - energy_mu
        finalize_step(energy_mu, target, error, t, layer_type, energy_fn_name)

    return current, current, top_error
    
def x_init(batch_size: int, seq_len: int, embedding_size: int, device: torch.device = None) -> torch.Tensor:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return torch.randn(batch_size, seq_len, embedding_size, device = device)

def step_embed(
    t: int,
    T: int,
    target: torch.Tensor,
    current_state:torch.Tensor,
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
    top_layers: Optional[dict] = None,
    target_q: Optional[torch.Tensor] = None,
    target_k: Optional[torch.Tensor] = None,
    target_v: Optional[torch.Tensor] = None,
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

    bottum_error = current_state - mu_norm
    X

    # TODO UPDATE X AND UPDATE WIEGHT

    UPDATE  X dtd​xil​=−ϵil​+j∑​ϵjl+1​wijl​   WHERE −ϵil​=−(xil​−x^il​)   Second Term summation ej l+1 wij This is top-down pressure.

Upper layer has error:
    # UPDATE WEIGHT

    # UPDATE WEIGHT dtd​wikl​=ϵkl+1​xil​   Δw∝error×input

# This is Hebbian-like:

    if top_layers is not None and (target_q is not None or target_k is not None or target_v is not None):
        top_errors = []
        if target_q is not None:
            target_q_3d = _merge_heads(target_q) if target_q.dim() == 4 else target_q
            q_pred = top_layers["q"](current_state)
            top_errors.append((target_q_3d - q_pred) @ top_layers["q"].weight)
        if target_k is not None:
            target_k_3d = _merge_heads(target_k) if target_k.dim() == 4 else target_k
            k_pred = top_layers["k"](mu_norm)
            top_errors.append((target_k_3d - k_pred)@ top_layers["k"].weight)
        if target_v is not None:
            target_v_3d = _merge_heads(target_v) if target_v.dim() == 4 else target_v
            v_pred = top_layers["v"](mu_norm)
            top_errors.append((target_v_3d - v_pred) @ top_layers["v"].weight)
        if top_errors:
            error = sum(top_errors) / len(top_errors)
        
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
    current: Optional[torch.Tensor],
    lateral_conn: Optional[Any],
    proj_layers: Optional[dict],
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    layer_bottom: Optional[nn.Module],
    layer_top: Optional[nn.Module],
    n_embed: int,
    num_heads: int,
    previous: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module],
    optimizer: Optional[PCOptimizer] = None,
):
    

    mu=layer_bottom(previous)
    bottum_up_error=current-mu
    top_error=target-layer_top(current)
    total_error = - bottum_up_error+top_error
    return x, mu, bu_err


def step_K(
    t: int,
    T: int,
    target: Optional[torch.Tensor],
    current_state:Optional[torch.Tensor]
    previous:Optional[torch.Tensor],
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
   
    layer_norm: Optional[nn.Module],
    optimizer: Optional[PCOptimizer] = None,
):
    
    mu=layer_bottom(previous)
    bottum_up_error=current-mu
    top_error=target-layer_top(current)
    total_error = - bottum_up_error+top_error




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
    current:Optional[torch.Tensor],
    previous:Optional[torch.Tensor],
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
    
    layer_norm: Optional[nn.Module],
    optimizer: Optional[PCOptimizer] = None,
):
    
    mu=layer_bottom(previous)
    bottum_up_error=current-mu
    top_error=target-layer_top(current)
    total_error = - bottum_up_error+top_error

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
    
    previous_score_q: torch.Tensor,
    previous_score_k: torch.Tensor,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
):
    
    mu_q_score=bottom_layer_q_score(previous_score_q)
    mu_k_score=bottom_layer_k_score(previous_score_k)
    mu_score=(mu_q_score@mu_k_score.T )/sqrt(n_embed)

    bottum_up_error=current-mu_score
   
    top_error=target-layer_top(current)
    total_error = - bottum_up_error+top_error


    
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
    scores = scores.masked_fill(causal_mask == 0, -1e9)
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
    return  mu_score, bu_err, new_kv_cache


def step_X_A(
    t: int,
    T: int,
    target: Optional[torch.Tensor],
    previous:
    current: 
  
    lateral_conn: Optional[Any],
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    requires_update: bool,
    layer:nn.Module,
    td_err: Optional[torch.Tensor],
    score: torch.Tensor,
):
    mu=layer_bottom(previous)
    mu=nn.softamx(mu)
    bottum_up_error=current-mu
    top_error=target-layer_top(current)
    total_error = - bottum_up_error+top_error

    if score is None:
        raise ValueError("X_A requires a score tensor")
    mu_logits = x if x is not None else score
    mu_logits = torch.softmax(mu_logits, dim=-1)
    mu=layer(mu)
    target_probs = target
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
    previous_attenout_X_A,
    previous_attenout_xscore,
    bottom_layer_attenout_X_A,
    bottom_layer_attenout_xscore,
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
    mu_attnout_X_A=bottom_layer_attenout_X_A(previous_attenout_X_A)
    mu_attnout_X_score=bottom_layer_attenout_xscore(previous_attenout_xscore)
    mu_to_attenout=mu_attnout_X_A@mu_attnout_X_score

    bottum_up_error=current-mu_to_attenout
    top_error=target-layer_top(current)
    total_error = - bottum_up_error+top_error

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

    


def step_fc1(
    t: int,
    T: int,
    target: Optional[torch.Tensor],
    previous:
    current: 
  
    lateral_conn: Optional[Any],
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    requires_update: bool,
    layer:nn.Module,
    td_err: Optional[torch.Tensor],
    score: torch.Tensor,
):
    mu=layer_bottom(previous)
                                                                                    
    bottum_up_error=current-mu
    mufc1_fc2=F.gelu(layer_top(current))
    top_error=target-mufc1_fc2
    total_error = - bottum_up_error+top_error
def step_fc2(
    t: int,
    T: int,
    target: Optional[torch.Tensor],
    previous:
    current: 
  
    lateral_conn: Optional[Any],
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    requires_update: bool,
    layer:nn.Module,
    td_err: Optional[torch.Tensor],
    score: torch.Tensor,
):
    
    
    mu=layer_bottom(previous)
    mu=F.gelu(mu)
                                                                                    
    bottum_up_error=current-mu
    prediction_fc2toouput=target-layer_top(current)
    top_error=target-prediction_fc2toouput
    total_error = - bottum_up_error+top_error





    if score is None:
        raise ValueError("X_A requires a score tensor")
    mu_logits = x if x is not None else score
    mu_logits = torch.softmax(mu_logits, dim=-1)
    mu=layer(mu)
    target_probs = target
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
def step_output(
    t: int,
    T: int,
    target: Optional[torch.Tensor],
    previous:
    current: 
  
    lateral_conn: Optional[Any],
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    requires_update: bool,
    layer:nn.Module,
    td_err: Optional[torch.Tensor],
    score: torch.Tensor,
):
    
    
    mu=layer_bottom(previous)
    mu=F.softmax(mu)
                                                                                    
    bottum_up_error=current-mu
    predict_f=layer_top(current)
 
    total_error = - bottum_up_error





    if score is None:
        raise ValueError("X_A requires a score tensor")
    mu_logits = x if x is not None else score
    mu_logits = torch.softmax(mu_logits, dim=-1)
    mu=layer(mu)
    target_probs = target
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