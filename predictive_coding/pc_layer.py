"""
Predictive Coding Layer

This module implements a complete predictive coding layer that manages:
1. Activity state x (latent representations)
2. Prediction μ (from bottom-up)
3. Error ε = x - μ
4. Activity updates: dx/dt = -ε + Σ(ε_j^(l+1) * w_ij)
5. Weight updates: dw/dt = ε * x

The layer supports lateral connections and integrates with the transformer architecture.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Any

from utils.pc_utils import (
    activity_update,
    weight_update,
    compute_prediction_error,
    step_embedding,
    step_qkv_projection,
    step_attention_scores,
    step_attention_weights,
    step_attention_output,
    step_mlp,
    step_output_layer,
    reshape_for_attention,
    merge_attention_heads,
)
from utils.optim.optim_utils import PCOptimizer
from predictive_coding.lateral_connc import LateralConnections


class PCLayer(nn.Module):
    """
    Predictive Coding Layer with complete activity and weight update equations.
    
    Activity Update (Inference):
        dx_i^l/dt = -ε_i^l + Σ_j ε_j^(l+1) * w_ij^l
        
    Weight Update (Learning):
        dw_ik^l/dt = ε_k^(l+1) * x_i^l
    
    Attributes:
        T: Number of inference iterations
        local_lr: Learning rate for updates
        lateral_connections: Dictionary of lateral connection modules
        _x_cache: Cache for activity states
        _mu_cache: Cache for predictions
        _error_cache: Cache for errors
    """
    
    def __init__(
        self,
        T: int,
        lr: float,
        update_bias: bool = True,
        energy_fn_name: str = "pc_e",
        num_heads: Optional[int] = None,
        n_embed: Optional[int] = None,
        optimizer_name: str = "adam",
        optimizer_beta1: float = 0.9,
        optimizer_beta2: float = 0.999,
        optimizer_eps: float = 1e-8,
        optimizer_sign_value: float = -1.0,
        optimizer_weight_bound: float = 0.0,
    ):
        super().__init__()
        self.T = T
        self.local_lr = lr
        self.update_bias = update_bias
        self.clamp_value = 3.0
        self.energy_fn_name = energy_fn_name
        self.num_heads = num_heads
        self.n_embed = n_embed
        
        # Optimizer for weight updates
        self.optimizer = PCOptimizer(
            opt_name=optimizer_name,
            beta1=optimizer_beta1,
            beta2=optimizer_beta2,
            eps=optimizer_eps,
            sign_value=optimizer_sign_value,
            weight_bound=optimizer_weight_bound,
        )
        
        # Lateral connections for each layer type
        self.lateral_connections: Dict[str, LateralConnections] = nn.ModuleDict()
        
        # State caches
        self._x_cache: Dict[str, torch.Tensor] = {}
        self._mu_cache: Dict[str, torch.Tensor] = {}
        self._error_cache: Dict[str, torch.Tensor] = {}
        self._energy_cache: Dict[str, float] = {}
        self._energy = 0.0
        self._errors = []
    
    def register_lateral(self, layer_type: str, size: int):
        """Register lateral connections for a layer type."""
        if layer_type not in self.lateral_connections:
            self.lateral_connections[layer_type] = LateralConnections(size, self.local_lr)
    
    def get_lateral(self, layer_type: str) -> Optional[LateralConnections]:
        """Get lateral connections for a layer type."""
        return self.lateral_connections.get(layer_type, None)
    
    def forward(
        self,
        layer_type: str,
        t: int,
        T: int,
        requires_update: bool = True,
        current_state: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        # Embedding specific
        word_embeddings: Optional[nn.Embedding] = None,
        pos_embeddings: Optional[nn.Embedding] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        q_proj: Optional[nn.Module] = None,
        k_proj: Optional[nn.Module] = None,
        v_proj: Optional[nn.Module] = None,
        # QKV specific
        previous: Optional[torch.Tensor] = None,
        proj_layer: Optional[nn.Module] = None,
        # Attention scores/weights specific
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
        # Attention output specific
        weights: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        output_proj: Optional[nn.Module] = None,
        # MLP specific
        fc_layer: Optional[nn.Module] = None,
        activation: str = "gelu",
        # Output specific
        output_layer: Optional[nn.Module] = None,
        # Common
        layer_norm: Optional[nn.Module] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for one inference step.
        
        Returns:
            x_new: Updated activity
            mu: Prediction
            error: Prediction error
        """
        # Get current state from cache if not provided
        if current_state is None:
            current_state = self._x_cache.get(layer_type)
        
        if current_state is None:
            raise ValueError(f"No current state provided or cached for layer_type={layer_type}")
        
        # Get lateral connections
        lateral_conn = self.get_lateral(layer_type)
        
        # Route to appropriate step function
        if layer_type == "embed":
            x_new, mu_word, mu_pos = step_embedding(
                x=current_state,
                target_q=target,
                target_k=kwargs.get("target_k"),
                target_v=kwargs.get("target_v"),
                word_embeddings=word_embeddings,
                pos_embeddings=pos_embeddings,
                input_ids=input_ids,
                position_ids=position_ids,
                q_proj=q_proj,
                k_proj=k_proj,
                v_proj=v_proj,
                layer_norm=layer_norm,
                local_lr=self.local_lr,
                clamp_value=self.clamp_value,
                requires_update=requires_update,
                optimizer=self.optimizer
            )
            self._x_cache[layer_type] = x_new
            self._mu_cache[layer_type] = mu_word + mu_pos
            return x_new, mu_word + mu_pos, x_new - (mu_word + mu_pos)
        
        elif layer_type in ["X_Q", "X_K", "X_V"]:
            x_new, projection, error = step_qkv_projection(
                x=current_state,
                target=target,
                previous=previous,
                proj_layer=proj_layer,
                layer_norm=layer_norm,
                local_lr=self.local_lr,
                clamp_value=self.clamp_value,
                requires_update=requires_update,
                lateral_conn=lateral_conn,
                optimizer=self.optimizer,
                num_heads=self.num_heads
            )
            self._x_cache[layer_type] = x_new
            self._mu_cache[layer_type] = projection
            self._error_cache[layer_type] = error
            return x_new, projection, error
        
        elif layer_type == "X_score":
            x_new, computed_scores, error = step_attention_scores(
                x=current_state,
                target=target,
                q=q,
                k=k,
                local_lr=self.local_lr,
                clamp_value=self.clamp_value,
                requires_update=requires_update,
                lateral_conn=lateral_conn,
                causal_mask=kwargs.get("causal_mask", True)
            )
            self._x_cache[layer_type] = x_new
            self._mu_cache[layer_type] = computed_scores
            self._error_cache[layer_type] = error
            return x_new, computed_scores, error
        
        elif layer_type == "X_A":
            x_new, computed_weights, error = step_attention_weights(
                x=current_state,
                target=target,
                scores=scores,
                local_lr=self.local_lr,
                clamp_value=self.clamp_value,
                requires_update=requires_update,
                lateral_conn=lateral_conn
            )
            self._x_cache[layer_type] = x_new
            self._mu_cache[layer_type] = computed_weights
            self._error_cache[layer_type] = error
            return x_new, computed_weights, error
        
        elif layer_type == "attn_output":
            x_new, output, error = step_attention_output(
                x=current_state,
                target=target,
                weights=weights,
                v=v,
                output_proj=output_proj,
                layer_norm=layer_norm,
                local_lr=self.local_lr,
                clamp_value=self.clamp_value,
                requires_update=requires_update,
                lateral_conn=lateral_conn,
                optimizer=self.optimizer
            )
            self._x_cache[layer_type] = x_new
            self._mu_cache[layer_type] = output
            self._error_cache[layer_type] = error
            return x_new, output, error
        
        elif layer_type in ["fc1", "fc2"]:
            x_new, output, error = step_mlp(
                x=current_state,
                target=target,
                previous=previous,
                fc_layer=fc_layer,
                activation=activation,
                layer_norm=layer_norm,
                local_lr=self.local_lr,
                clamp_value=self.clamp_value,
                requires_update=requires_update,
                lateral_conn=lateral_conn,
                optimizer=self.optimizer
            )
            self._x_cache[layer_type] = x_new
            self._mu_cache[layer_type] = output
            self._error_cache[layer_type] = error
            return x_new, output, error
        
        elif layer_type == "output":
            x_new, logits, error = step_output_layer(
                x=current_state,
                target=target,
                previous=previous,
                output_layer=output_layer,
                layer_norm=layer_norm,
                local_lr=self.local_lr,
                clamp_value=self.clamp_value,
                requires_update=requires_update,
                lateral_conn=lateral_conn,
                optimizer=self.optimizer
            )
            self._x_cache[layer_type] = x_new
            self._mu_cache[layer_type] = logits
            self._error_cache[layer_type] = error
            
            # Compute and store energy for output layer
            if t == T - 1:
                _, energy = compute_prediction_error(target, logits, self.energy_fn_name)
                self._energy += energy.item()
                self._energy_cache[layer_type] = energy.item()
            
            return x_new, logits, error
        
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}")
    
    def init_x(
        self,
        batch_size: int,
        seq_len: int,
        layer_type: str,
        device: torch.device,
        word_embeddings: Optional[nn.Embedding] = None,
        pos_embeddings: Optional[nn.Embedding] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        bottom_layer: Optional[nn.Module] = None,
        top_layer: Optional[nn.Module] = None,
    ):
        """
        Initialize activity state x for a layer type.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            layer_type: Type of layer (embed, X_Q, X_K, X_V, X_score, X_A, attn_output, fc1, fc2, output)
            device: Device to create tensors on
            word_embeddings: Word embedding layer (for embed)
            pos_embeddings: Position embedding layer (for embed)
            input_ids: Input token IDs (for embed)
            position_ids: Position IDs (for embed)
            bottom_layer: Bottom layer for shape inference
            top_layer: Top layer for shape inference
        """
        if layer_type == "embed":
            assert input_ids is not None and position_ids is not None
            assert word_embeddings is not None and pos_embeddings is not None
            
            # Initialize with embeddings
            vocab_size = word_embeddings.weight.size(0)
            max_pos = pos_embeddings.weight.size(0)
            input_ids = torch.clamp(input_ids, max=vocab_size - 1)
            position_ids = torch.clamp(position_ids, max=max_pos - 1)
            
            x_word = word_embeddings(input_ids)
            x_pos = pos_embeddings(position_ids)
            x_embed = x_word + x_pos
            
            self._x_cache[layer_type] = x_embed
            self._mu_cache[layer_type] = x_embed.clone()
        
        elif layer_type in ["X_Q", "X_K", "X_V"]:
            assert self.n_embed is not None and self.num_heads is not None
            head_dim = self.n_embed // self.num_heads
            
            # Initialize with small random values
            x = torch.randn(batch_size, self.num_heads, seq_len, head_dim, device=device) * 0.1
            self._x_cache[layer_type] = x
            self._mu_cache[layer_type] = torch.zeros_like(x)
            
            # Register lateral connections
            self.register_lateral(layer_type, head_dim)
        
        elif layer_type in ["X_score", "X_A"]:
            assert self.num_heads is not None
            
            # Initialize attention scores/weights (B, H, S, S)
            x = torch.randn(batch_size, self.num_heads, seq_len, seq_len, device=device) * 0.1
            self._x_cache[layer_type] = x
            self._mu_cache[layer_type] = torch.zeros_like(x)
            
            # Register lateral connections across heads
            self.register_lateral(layer_type, self.num_heads)
        
        elif layer_type == "attn_output":
            assert self.n_embed is not None and self.num_heads is not None
            head_dim = self.n_embed // self.num_heads
            
            # Initialize attention output (B, H, S, D)
            x = torch.randn(batch_size, self.num_heads, seq_len, head_dim, device=device) * 0.1
            self._x_cache[layer_type] = x
            self._mu_cache[layer_type] = torch.zeros_like(x)
            
            # Register lateral connections
            self.register_lateral(layer_type, head_dim)
        
        elif layer_type in ["fc1", "fc2"]:
            ref_layer = bottom_layer or top_layer
            assert ref_layer is not None
            
            input_dim = ref_layer.weight.shape[1]
            
            # Initialize MLP layer (B, S, D)
            x = torch.randn(batch_size, seq_len, input_dim, device=device) * 0.1
            self._x_cache[layer_type] = x
            self._mu_cache[layer_type] = torch.zeros_like(x)
            
            # Register lateral connections
            self.register_lateral(layer_type, input_dim)
        
        elif layer_type == "output":
            ref_layer = bottom_layer or top_layer
            assert ref_layer is not None
            
            output_dim = ref_layer.weight.shape[0]
            
            # Initialize output layer (B, S, vocab_size)
            x = torch.randn(batch_size, seq_len, output_dim, device=device) * 0.1
            self._x_cache[layer_type] = x
            self._mu_cache[layer_type] = torch.zeros_like(x)
            
            # Register lateral connections
            self.register_lateral(layer_type, output_dim)
        
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}")
    
    def get_x(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get cached activity for a layer type."""
        return self._x_cache.get(layer_type)
    
    def get_mu(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get cached prediction for a layer type."""
        return self._mu_cache.get(layer_type)
    
    def get_error(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get cached error for a layer type."""
        return self._error_cache.get(layer_type)
    
    def get_energy(self) -> float:
        """Get total accumulated energy."""
        return self._energy
    
    def get_energy_by_layer(self, layer_type: str) -> Optional[float]:
        """Get energy for a specific layer type."""
        return self._energy_cache.get(layer_type)
    
    def clear_energy(self):
        """Clear all cached energy values."""
        self._energy = 0.0
        self._energy_cache.clear()
    
    def clear_errors(self):
        """Clear all cached errors."""
        self._errors = []
        self._error_cache.clear()
    
    def clear_all(self):
        """Clear all caches."""
        self._x_cache.clear()
        self._mu_cache.clear()
        self._error_cache.clear()
        self._energy_cache.clear()
        self._energy = 0.0
        self._errors = []
    
    def set_learning_rate(self, lr: float):
        """Set learning rate for this layer."""
        self.local_lr = lr
        for conn in self.lateral_connections.values():
            conn.set_learning_rate(lr)
