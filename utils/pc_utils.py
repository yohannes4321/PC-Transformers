"""
Predictive Coding Utilities

This module implements the core predictive coding equations:

Activity Update (Inference):
    dx_i^l/dt = -ε_i^l + Σ_j ε_j^(l+1) * w_ij^l
    
    where ε_i^l = x_i^l - μ_i^l (prediction error)
    First term: bottom-up error (mismatch with lower layer prediction)
    Second term: top-down pressure (error from upper layer weighted by connections)

Weight Update (Learning):
    dw_ik^l/dt = ε_k^(l+1) * x_i^l
    
    Hebbian-like update proportional to error × input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Any


def activity_update(
    x_current: torch.Tensor,
    bottom_error: torch.Tensor,
    top_error: Optional[torch.Tensor],
    top_weights: Optional[torch.Tensor],
    lateral_conn: Optional[Any],
    local_lr: float,
    clamp_value: float
) -> torch.Tensor:
    """
    Activity Update (Inference Step):
    dx_i^l/dt = -ε_i^l + Σ_j ε_j^(l+1) * w_ij^l
    
    Args:
        x_current: Current activity x_i^l
        bottom_error: Bottom-up error ε_i^l = x_i^l - μ_i^l
        top_error: Top-down error ε_j^(l+1) from layer above
        top_weights: Weights w_ij^l connecting current to upper layer
        lateral_conn: Lateral connections for local interactions
        local_lr: Learning rate for activity update
        clamp_value: Value to clamp activity
    
    Returns:
        Updated activity x_i^l(t+dt)
    """
    # Bottom-up term: -ε_i^l (negative because we want to reduce error)
    delta_x = -bottom_error
    
    # Top-down term: Σ_j ε_j^(l+1) * w_ij^l
    if top_error is not None and top_weights is not None:
        # Project top-down error through weights
        if top_error.dim() == 4:  # (B, H, S, D)
            B, H, S, D = top_error.shape
            top_error_flat = top_error.reshape(B * H * S, D)
            td_contribution = top_error_flat @ top_weights
            td_contribution = td_contribution.reshape(B, H, S, -1)
        elif top_error.dim() == 3:  # (B, S, D)
            B, S, D = top_error.shape
            top_error_flat = top_error.reshape(B * S, D)
            td_contribution = top_error_flat @ top_weights
            td_contribution = td_contribution.reshape(B, S, -1)
        else:
            td_contribution = top_error @ top_weights
        
        delta_x = delta_x + td_contribution
    
    # Apply lateral connections if available
    if lateral_conn is not None:
        delta_x = lateral_conn.forward(x_current, delta_x)
    
    # Update activity: x_new = x + lr * delta_x
    x_new = x_current + local_lr * delta_x
    
    # Clamp activity to prevent explosion
    x_new = torch.clamp(x_new, -abs(clamp_value), abs(clamp_value))
    
    return x_new


def weight_update(
    weight: nn.Parameter,
    error: torch.Tensor,
    input_activity: torch.Tensor,
    local_lr: float,
    optimizer: Optional[Any],
    clamp_value: float = 0.01,
    update_bias: bool = False,
    bias: Optional[nn.Parameter] = None
) -> None:
    """
    Weight Update (Learning Step):
    dw_ik^l/dt = ε_k^(l+1) * x_i^l
    
    Hebbian-like update proportional to error × input
    
    Args:
        weight: Weight parameter to update (w_ik^l)
        error: Error from upper layer ε_k^(l+1)
        input_activity: Input activity from current layer x_i^l
        local_lr: Learning rate
        optimizer: Optional optimizer for weight update
        clamp_value: Gradient clamping value
        update_bias: Whether to update bias
        bias: Optional bias parameter
    """
    with torch.no_grad():
        # Flatten for batch processing
        if error.dim() == 4:  # (B, H, S, D)
            B, H, S, D = error.shape
            error_flat = error.reshape(B * H * S, D)
            input_flat = input_activity.reshape(B * H * S, -1)
        elif error.dim() == 3:  # (B, S, D)
            B, S, D = error.shape
            error_flat = error.reshape(B * S, D)
            input_flat = input_activity.reshape(B * S, -1)
        else:
            error_flat = error
            input_flat = input_activity
        
        # Compute weight update: Δw = error^T @ input / batch_size
        delta_w = torch.matmul(error_flat.t(), input_flat) / max(error_flat.size(0), 1)
        
        # Apply update
        if optimizer is not None:
            optimizer.step_param(weight, delta_w, local_lr, clamp_value=clamp_value)
        else:
            weight.data.add_(torch.clamp(local_lr * delta_w, -clamp_value, clamp_value))
        
        # Update bias if requested
        if update_bias and bias is not None:
            delta_b = error_flat.mean(dim=0)
            if optimizer is not None:
                optimizer.step_param(bias, delta_b, local_lr, clamp_value=clamp_value)
            else:
                bias.data.add_(torch.clamp(local_lr * delta_b, -clamp_value, clamp_value))


def compute_prediction_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    energy_fn_name: str = "pc_e"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute prediction error and energy.
    
    Args:
        target: Target activity (ground truth or from above)
        prediction: Predicted activity (from below)
        energy_fn_name: Energy function name
    
    Returns:
        error: Prediction error (target - prediction)
        energy: Scalar energy value
    """
    error = target - prediction
    
    if energy_fn_name == "pc_e":
        # Squared error energy: E = 0.5 * ||target - prediction||^2
        energy = 0.5 * (error ** 2).mean()
    elif energy_fn_name == "kld":
        # KL divergence energy
        energy = torch.clamp(
            F.kl_div(prediction.log_softmax(dim=-1), target, reduction="batchmean"),
            min=0.0, max=100.0
        )
    else:
        energy = 0.5 * (error ** 2).mean()
    
    return error, energy


def apply_layer_with_norm(
    layer: nn.Module,
    x: torch.Tensor,
    layer_norm: Optional[nn.Module] = None
) -> torch.Tensor:
    """
    Apply linear layer with optional layer normalization.
    
    Args:
        layer: Linear layer
        x: Input tensor
        layer_norm: Optional layer normalization
    
    Returns:
        Output tensor
    """
    # Apply layer normalization first (Pre-LN architecture)
    if layer_norm is not None:
        x = layer_norm(x)
    
    # Handle multi-head attention shapes
    if x.dim() == 4:  # (B, H, S, D)
        B, H, S, D = x.shape
        x_flat = x.reshape(B * H * S, D)
        out = layer(x_flat)
        out = out.reshape(B, H, S, -1)
    elif x.dim() == 3:  # (B, S, D)
        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)
        out = layer(x_flat)
        out = out.reshape(B, S, -1)
    else:
        out = layer(x)
    
    return out


def reshape_for_attention(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int
) -> torch.Tensor:
    """
    Reshape tensor for multi-head attention: (B, S, D) -> (B, H, S, D/H)
    """
    B, S, D = x.shape
    return x.view(B, S, num_heads, head_dim).transpose(1, 2)


def merge_attention_heads(
    x: torch.Tensor
) -> torch.Tensor:
    """
    Merge attention heads: (B, H, S, D) -> (B, S, H*D)
    """
    B, H, S, D = x.shape
    return x.transpose(1, 2).contiguous().view(B, S, H * D)


def softmax_with_temperature(
    logits: torch.Tensor,
    dim: int = -1,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Softmax with temperature scaling.
    """
    return F.softmax(logits / temperature, dim=dim)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal mask for autoregressive attention.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)


def ids_to_one_hot(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Convert token IDs to one-hot encoding.
    """
    device = input_ids.device
    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size - 1)
    return F.one_hot(input_ids, num_classes=vocab_size).float().to(device)


# ============================================================================
# Layer-Specific Step Functions
# ============================================================================

def step_embedding(
    x: torch.Tensor,
    target_q: Optional[torch.Tensor],
    target_k: Optional[torch.Tensor],
    target_v: Optional[torch.Tensor],
    word_embeddings: nn.Embedding,
    pos_embeddings: nn.Embedding,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    q_proj: nn.Module,
    k_proj: nn.Module,
    v_proj: nn.Module,
    layer_norm: Optional[nn.Module],
    local_lr: float,
    clamp_value: float,
    requires_update: bool,
    optimizer: Optional[Any] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predictive coding step for embedding layer.
    
    Architecture:
        Input -> Word Embed + Pos Embed -> LayerNorm -> Q/K/V projections
    
    Returns:
        x_embed: Updated embedding activity
        mu_word: Word embedding prediction
        mu_pos: Position embedding prediction
    """
    # Get embeddings
    mu_word = word_embeddings(input_ids)
    mu_pos = pos_embeddings(position_ids)
    mu_embed = mu_word + mu_pos
    
    # Apply layer normalization (Pre-LN)
    if layer_norm is not None:
        mu_embed_norm = layer_norm(mu_embed)
    else:
        mu_embed_norm = mu_embed
    
    # Bottom-up error: ε = x - μ
    bottom_error = x - mu_embed_norm
    
    # Top-down error from Q, K, V layers
    top_error = None
    td_weights = None
    
    if target_q is not None or target_k is not None or target_v is not None:
        td_errors = []
        
        # Q projection error
        if target_q is not None:
            target_q_merged = merge_attention_heads(target_q) if target_q.dim() == 4 else target_q
            q_pred = q_proj(mu_embed_norm)
            q_error = target_q_merged - q_pred
            td_errors.append((q_error, q_proj.weight))
        
        # K projection error
        if target_k is not None:
            target_k_merged = merge_attention_heads(target_k) if target_k.dim() == 4 else target_k
            k_pred = k_proj(mu_embed_norm)
            k_error = target_k_merged - k_pred
            td_errors.append((k_error, k_proj.weight))
        
        # V projection error
        if target_v is not None:
            target_v_merged = merge_attention_heads(target_v) if target_v.dim() == 4 else target_v
            v_pred = v_proj(mu_embed_norm)
            v_error = target_v_merged - v_pred
            td_errors.append((v_error, v_proj.weight))
        
        # Combine top-down errors
        if td_errors:
            top_error = sum(err for err, _ in td_errors) / len(td_errors)
            # Average weights for top-down projection
            td_weights = sum(w for _, w in td_errors) / len(td_errors)
    
    # Activity update: dx/dt = -ε + Σ(ε_j * w_ij)
    x_new = activity_update(
        x_current=x,
        bottom_error=bottom_error,
        top_error=top_error,
        top_weights=td_weights,
        lateral_conn=None,  # No lateral for embedding
        local_lr=local_lr,
        clamp_value=clamp_value
    )
    
    # Weight update: dw/dt = ε * x
    if requires_update:
        # Update word embeddings
        with torch.no_grad():
            flat_input_ids = input_ids.reshape(-1)
            flat_error = bottom_error.reshape(-1, bottom_error.size(-1))
            
            # Word embedding update
            delta_word = torch.zeros_like(word_embeddings.weight)
            delta_word.index_add_(0, flat_input_ids, flat_error)
            
            if optimizer is not None:
                optimizer.step_param(word_embeddings.weight, delta_word, local_lr, clamp_value=0.01)
                if pos_embeddings.weight.requires_grad:
                    delta_pos = torch.zeros_like(pos_embeddings.weight)
                    flat_pos_ids = position_ids.reshape(-1)
                    delta_pos.index_add_(0, flat_pos_ids, flat_error)
                    optimizer.step_param(pos_embeddings.weight, delta_pos, local_lr, clamp_value=0.01)
            else:
                word_embeddings.weight.data.add_(
                    torch.clamp(local_lr * delta_word, -0.01, 0.01)
                )
                if pos_embeddings.weight.requires_grad:
                    delta_pos = torch.zeros_like(pos_embeddings.weight)
                    flat_pos_ids = position_ids.reshape(-1)
                    delta_pos.index_add_(0, flat_pos_ids, flat_error)
                    pos_embeddings.weight.data.add_(
                        torch.clamp(local_lr * delta_pos, -0.01, 0.01)
                    )
    
    return x_new, mu_word, mu_pos


def step_qkv_projection(
    x: torch.Tensor,
    target: Optional[torch.Tensor],
    previous: torch.Tensor,
    proj_layer: nn.Module,
    layer_norm: Optional[nn.Module],
    local_lr: float,
    clamp_value: float,
    requires_update: bool,
    lateral_conn: Optional[Any] = None,
    optimizer: Optional[Any] = None,
    num_heads: int = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predictive coding step for Q, K, or V projection.
    
    Args:
        x: Current activity (4D: B, H, S, D)
        target: Target from above (4D: B, H, S, D)
        previous: Activity from below (3D: B, S, D)
        proj_layer: Projection linear layer
        layer_norm: Layer normalization
        local_lr: Learning rate
        clamp_value: Clamping value
        requires_update: Whether to update weights
        lateral_conn: Lateral connections
        optimizer: Optimizer
        num_heads: Number of attention heads
    
    Returns:
        x_new: Updated activity
        projection: Projected values
        error: Prediction error
    """
    # Apply layer norm to previous
    if layer_norm is not None:
        previous_norm = layer_norm(previous)
    else:
        previous_norm = previous
    
    # Compute projection (prediction from below)
    proj_flat = proj_layer(previous_norm.reshape(-1, previous_norm.size(-1)))
    
    if num_heads is not None and x.dim() == 4:
        B, H, S, D = x.shape
        head_dim = D
        projection = proj_flat.reshape(B, S, H, head_dim).transpose(1, 2)
    else:
        projection = proj_flat.reshape_as(x)
    
    # Bottom-up error
    bottom_error = x - projection
    
    # Top-down error
    top_error = None
    td_weights = None
    if target is not None:
        top_error = target - x
        td_weights = proj_layer.weight
    
    # Activity update
    x_new = activity_update(
        x_current=x,
        bottom_error=bottom_error,
        top_error=top_error,
        top_weights=td_weights,
        lateral_conn=lateral_conn,
        local_lr=local_lr,
        clamp_value=clamp_value
    )
    
    # Weight update: dw/dt = ε * x
    if requires_update:
        weight_update(
            weight=proj_layer.weight,
            error=bottom_error if target is None else top_error,
            input_activity=previous_norm,
            local_lr=local_lr,
            optimizer=optimizer,
            clamp_value=0.01,
            update_bias=True if hasattr(proj_layer, 'bias') and proj_layer.bias is not None else False,
            bias=proj_layer.bias if hasattr(proj_layer, 'bias') else None
        )
    
    return x_new, projection, bottom_error


def step_attention_scores(
    x: torch.Tensor,
    target: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    local_lr: float,
    clamp_value: float,
    requires_update: bool,
    lateral_conn: Optional[Any] = None,
    causal_mask: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predictive coding step for attention scores.
    
    Args:
        x: Current score activity (B, H, S_q, S_k)
        target: Target from above (B, H, S_q, S_k)
        q: Query tensor (B, H, S_q, D)
        k: Key tensor (B, H, S_k, D)
        local_lr: Learning rate
        clamp_value: Clamping value
        requires_update: Whether to update
        lateral_conn: Lateral connections
        causal_mask: Whether to apply causal masking
    
    Returns:
        x_new: Updated activity
        scores: Computed attention scores
        error: Prediction error
    """
    # Compute attention scores: Q @ K^T / sqrt(d)
    head_dim = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # Apply causal mask if needed
    if causal_mask:
        seq_len = scores.size(-1)
        device = scores.device
        mask = create_causal_mask(seq_len, device)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Bottom-up error
    bottom_error = x - scores
    
    # Top-down error
    top_error = None
    if target is not None:
        top_error = target - x
    
    # Activity update
    x_new = activity_update(
        x_current=x,
        bottom_error=bottom_error,
        top_error=top_error,
        top_weights=None,  # Scores don't have direct weight connections up
        lateral_conn=lateral_conn,
        local_lr=local_lr,
        clamp_value=clamp_value
    )
    
    return x_new, scores, bottom_error


def step_attention_weights(
    x: torch.Tensor,
    target: Optional[torch.Tensor],
    scores: torch.Tensor,
    local_lr: float,
    clamp_value: float,
    requires_update: bool,
    lateral_conn: Optional[Any] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predictive coding step for attention weights (softmax of scores).
    
    Args:
        x: Current weight activity (B, H, S, S)
        target: Target from above (B, H, S, S)
        scores: Attention scores (B, H, S, S)
        local_lr: Learning rate
        clamp_value: Clamping value
        requires_update: Whether to update
        lateral_conn: Lateral connections
    
    Returns:
        x_new: Updated activity
        weights: Softmax weights
        error: Prediction error
    """
    # Compute softmax weights
    weights = F.softmax(scores, dim=-1)
    
    # Bottom-up error
    bottom_error = x - weights
    
    # Top-down error
    top_error = None
    if target is not None:
        top_error = target - x
    
    # Activity update
    x_new = activity_update(
        x_current=x,
        bottom_error=bottom_error,
        top_error=top_error,
        top_weights=None,
        lateral_conn=lateral_conn,
        local_lr=local_lr,
        clamp_value=clamp_value
    )
    
    return x_new, weights, bottom_error


def step_attention_output(
    x: torch.Tensor,
    target: Optional[torch.Tensor],
    weights: torch.Tensor,
    v: torch.Tensor,
    output_proj: nn.Module,
    layer_norm: Optional[nn.Module],
    local_lr: float,
    clamp_value: float,
    requires_update: bool,
    lateral_conn: Optional[Any] = None,
    optimizer: Optional[Any] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predictive coding step for attention output.
    
    Args:
        x: Current output activity (B, H, S, D)
        target: Target from above (3D: B, S, D)
        weights: Attention weights (B, H, S, S)
        v: Value tensor (B, H, S, D)
        output_proj: Output projection layer
        layer_norm: Layer normalization
        local_lr: Learning rate
        clamp_value: Clamping value
        requires_update: Whether to update
        lateral_conn: Lateral connections
        optimizer: Optimizer
    
    Returns:
        x_new: Updated activity
        output: Attention output
        error: Prediction error
    """
    # Compute attention output: weights @ V
    context = torch.matmul(weights, v)  # (B, H, S, D)
    
    # Merge heads and apply output projection
    B, H, S, D = context.shape
    context_merged = context.transpose(1, 2).contiguous().view(B, S, H * D)
    
    # Apply layer norm before projection (Pre-LN)
    if layer_norm is not None:
        context_norm = layer_norm(context_merged)
    else:
        context_norm = context_merged
    
    output = output_proj(context_norm)  # (B, S, n_embed)
    
    # Reshape for comparison
    if x.dim() == 4:
        head_dim = D
        output_reshaped = output.view(B, S, H, head_dim).transpose(1, 2)
    else:
        output_reshaped = output
    
    # Bottom-up error
    bottom_error = x - output_reshaped
    
    # Top-down error
    top_error = None
    td_weights = None
    if target is not None:
        top_error = target - output
        td_weights = output_proj.weight
    
    # Activity update
    x_new = activity_update(
        x_current=x,
        bottom_error=bottom_error,
        top_error=top_error if x.dim() == 3 else None,
        top_weights=td_weights if x.dim() == 3 else None,
        lateral_conn=lateral_conn,
        local_lr=local_lr,
        clamp_value=clamp_value
    )
    
    # Weight update
    if requires_update:
        weight_update(
            weight=output_proj.weight,
            error=bottom_error.reshape(-1, bottom_error.size(-1)),
            input_activity=context_norm,
            local_lr=local_lr,
            optimizer=optimizer,
            clamp_value=0.01,
            update_bias=True if output_proj.bias is not None else False,
            bias=output_proj.bias
        )
    
    return x_new, output, bottom_error


def step_mlp(
    x: torch.Tensor,
    target: Optional[torch.Tensor],
    previous: torch.Tensor,
    fc_layer: nn.Module,
    activation: str = "gelu",
    layer_norm: Optional[nn.Module] = None,
    local_lr: float = 0.01,
    clamp_value: float = 3.0,
    requires_update: bool = False,
    lateral_conn: Optional[Any] = None,
    optimizer: Optional[Any] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predictive coding step for MLP layer.
    
    Args:
        x: Current activity (3D: B, S, D)
        target: Target from above (3D: B, S, D)
        previous: Activity from below (3D: B, S, D)
        fc_layer: FC layer
        activation: Activation function
        layer_norm: Layer normalization
        local_lr: Learning rate
        clamp_value: Clamping value
        requires_update: Whether to update weights
        lateral_conn: Lateral connections
        optimizer: Optimizer
    
    Returns:
        x_new: Updated activity
        output: Layer output
        error: Prediction error
    """
    # Apply layer norm first (Pre-LN)
    if layer_norm is not None:
        previous_norm = layer_norm(previous)
    else:
        previous_norm = previous
    
    # Compute prediction
    output = fc_layer(previous_norm)
    
    # Apply activation if specified
    if activation == "gelu":
        output = F.gelu(output)
    elif activation == "relu":
        output = F.relu(output)
    
    # Bottom-up error
    bottom_error = x - output
    
    # Top-down error
    top_error = None
    td_weights = None
    if target is not None:
        top_error = target - x
        td_weights = fc_layer.weight
    
    # Activity update
    x_new = activity_update(
        x_current=x,
        bottom_error=bottom_error,
        top_error=top_error,
        top_weights=td_weights,
        lateral_conn=lateral_conn,
        local_lr=local_lr,
        clamp_value=clamp_value
    )
    
    # Weight update: dw/dt = ε * x
    if requires_update:
        weight_update(
            weight=fc_layer.weight,
            error=bottom_error if target is None else top_error,
            input_activity=previous_norm,
            local_lr=local_lr,
            optimizer=optimizer,
            clamp_value=0.01,
            update_bias=True if fc_layer.bias is not None else False,
            bias=fc_layer.bias
        )
    
    return x_new, output, bottom_error


def step_output_layer(
    x: torch.Tensor,
    target: torch.Tensor,
    previous: torch.Tensor,
    output_layer: nn.Module,
    layer_norm: Optional[nn.Module] = None,
    local_lr: float = 0.01,
    clamp_value: float = 3.0,
    requires_update: bool = False,
    lateral_conn: Optional[Any] = None,
    optimizer: Optional[Any] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predictive coding step for final output layer.
    
    Args:
        x: Current activity (B, S, vocab_size)
        target: Target logits/one-hot (B, S, vocab_size)
        previous: Activity from below (B, S, D)
        output_layer: Output linear layer
        layer_norm: Layer normalization
        local_lr: Learning rate
        clamp_value: Clamping value
        requires_update: Whether to update
        lateral_conn: Lateral connections
        optimizer: Optimizer
    
    Returns:
        x_new: Updated activity
        logits: Output logits
        error: Prediction error
    """
    # Apply layer norm first (Pre-LN)
    if layer_norm is not None:
        previous_norm = layer_norm(previous)
    else:
        previous_norm = previous
    
    # Compute logits
    logits = output_layer(previous_norm)
    
    # Bottom-up error
    bottom_error = x - logits
    
    # Top-down error (target is the ground truth)
    top_error = target - x
    
    # Activity update (no top-down projection needed for output)
    x_new = activity_update(
        x_current=x,
        bottom_error=bottom_error,
        top_error=top_error,
        top_weights=None,
        lateral_conn=lateral_conn,
        local_lr=local_lr,
        clamp_value=clamp_value
    )
    
    # Weight update
    if requires_update:
        weight_update(
            weight=output_layer.weight,
            error=top_error,
            input_activity=previous_norm,
            local_lr=local_lr,
            optimizer=optimizer,
            clamp_value=0.01,
            update_bias=True if output_layer.bias is not None else False,
            bias=output_layer.bias
        )
    
    return x_new, logits, top_error


def cleanup_memory():
    """Comprehensive memory cleanup for garbage collection and CUDA cache."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
