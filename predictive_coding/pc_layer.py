import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from utils.pc_utils import (
    x_init,
    step_embed,
    step_linear,
    step_attn,
    step_Q,
    step_K,
    step_V,
    step_X_score,
    step_X_A,
    step_X_attnOut,
    
    _reshape_to_heads,
    _merge_heads,
    finalize_step,
)
from utils.optim.optim_utils import PCOptimizer
from predictive_coding.lateral_connc import LateralConnections

class PCLayer(nn.Module):
    """
    Predictive Coding Layer wrapper that manages iterative inference state and
    delegates computation to helper functions (step_embed, step_attn, step_linear).
    """
    def __init__(
        self,
        T: int,
        lr: float,
        update_bias: bool,
        energy_fn_name: str,
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

        self.optimizer = PCOptimizer(
            opt_name=optimizer_name,
            beta1=optimizer_beta1,
            beta2=optimizer_beta2,
            eps=optimizer_eps,
            sign_value=optimizer_sign_value,
            weight_bound=optimizer_weight_bound,
        )
        
        self.lateral_connections: Dict[str, LateralConnections] = {}
        
        self._x_cache: Dict[str, torch.Tensor] = {}
        self._mu_cache: Dict[str, torch.Tensor] = {}
        self._error_cache: Dict[str, torch.Tensor] = {}
        self._td_err_cache: Dict[str, Optional[torch.Tensor]] = {}
        self._energy_cache: Dict[str, float] = {}
        self._energy = 0.0
        self._errors = []
    
    def register_lateral(self, layer_type: str, size: int):
        """Create and register lateral connections for layer_type."""
        existing = self.lateral_connections.get(layer_type)
        if existing is None or getattr(existing, "size", None) != size:
            self.lateral_connections[layer_type] = LateralConnections(size, self.local_lr)
            self.add_module(f"lateral_{layer_type}", self.lateral_connections[layer_type])

    def _reset_step_state(self) -> None:
        """Reset step-local accumulators, kept for future extension."""
        return
    
    def _get_cached_state(self, layer_type: str):
        return self._x_cache.get(layer_type, None)

    
    def forward(
        self,
        target_activity: torch.Tensor,
        layer_type: str,
        t: int,
        T: int,
        requires_update: bool,
        td_err:  Optional[torch.Tensor] = None,
        current_state: Optional[torch.Tensor] = None,
        previous: Optional[torch.Tensor] = None,
        previous_2: Optional[torch.Tensor] = None,
        bottom_layer: Optional[nn.Module] = None,
        bottom_layer_2: Optional[nn.Module] = None,
        top_layer: Optional[nn.Module] = None,
        embed_layers: Optional[dict] = None,
        layer: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        a_weights: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        layer_norm: Optional[nn.Module] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        flash: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # ADD THIS
        use_cache: bool = False, 
        top_layers: Optional[dict] = None,
        target_q_for_embed: Optional[torch.Tensor] = None,
        target_k_for_embed: Optional[torch.Tensor] = None,
        target_v_for_embed: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Perform one predictive coding inference step."""
        self._reset_step_state()
        self._td_err_cache[layer_type] = td_err
        x = current_state if current_state is not None else self._get_cached_state(layer_type)

        if layer_type == "embed":
            if embed_layers is None and isinstance(layer, dict):
                embed_layers = layer
            

            mu, mu_word, mu_pos, bu_err = step_embed(
                t,
                T,
                 target_activity,
                embed_layers,
                layer_type,
                input_ids,
                position_ids,
                self.local_lr,
                self.clamp_value,
                self.energy_fn_name,
                requires_update,
                layer_norm=layer_norm,
                optimizer=self.optimizer,
                top_layers=top_layers,
                target_q=target_q_for_embed,
                target_k=target_k_for_embed,
                target_v=target_v_for_embed,
            )            
            # store for later retrieval
            self._x_cache["embed"] = (mu_word, mu_pos)
            self._mu_cache["embed"] = mu.detach().clone()
            if bu_err is not None:
                self._error_cache["embed"] = bu_err.detach().clone()

            # compute energy
            energy_target = target_activity if target_activity is not None else target_embed
            if energy_target is not None:
                error = energy_target - mu
                energy, step_errors = finalize_step(mu, energy_target, error, t, layer_type, self.energy_fn_name)
                self._energy += energy
                self._energy_cache[layer_type] = energy
                self._errors.extend(step_errors)
            return mu_word, mu_pos
        
        like embed call calucaute energey input in the x_cache and mu_cache
        elif layer type =="X_Q": step_X_Q 
        elif layer type =="X_K": step x_k
        elif layer type =="X_V": step_X_V
        elif layer type =="X_SCORE": step x_score
        elif layer type=="X_A": step_x_A
        elif layer type =="x_attnout" step_x_attnout
        elif layer type =="fc1" step_fc1

        elif layer type =="fc2" step_fc2
        elif layer type =="output" step_output

    def init_x(
        self,
        batch_size: int,
        seq_len: int,
        layer_type: str,
        device: torch.device,
        embed_layers: Optional[dict] = None,
        bottom_layer: Optional[nn.Module] = None,
        top_layer: Optional[nn.Module] = None,
        layer: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Initialize cached activity `x` and mean `mu` for the layer type.
        Ensures 4D shapes (B, H, S, D) for Attention internals and 3D (B, S, D) for others.
        """
        # --- 1. Embedding Layer (3D: Batch, Seq, Embed) ---
        if layer_type == "embed":
            assert input_ids is not None and position_ids is not None
            if embed_layers is None and isinstance(layer, dict):
                embed_layers = layer
            vocab_size = embed_layers["word"].weight.size(0)
            max_pos = embed_layers["pos"].weight.size(0)
            
            # Safety clamp
            input_ids = torch.clamp(input_ids, max=vocab_size-1)
            position_ids = torch.clamp(position_ids, max=max_pos-1)
            
            x_word = embed_layers["word"].weight[input_ids] 
            x_pos = embed_layers["pos"].weight[position_ids] 
            self._x_cache["embed"] = (x_word, x_pos)
            # Embed doesn't usually use a stored '_mu' in the same way, 
            # but if needed, it would be 3D.

        # --- 2. Attention Projections (4D: Batch, Head, Seq, Head_Dim) ---
        elif layer_type in {"X_Q", "X_K", "X_V"}:
            assert self.n_embed is not None and self.num_heads is not None
            head_dim = self.n_embed // self.num_heads

            # Initialize X as 4D with smaller values to reduce initial energy
            self._x_cache[layer_type] = torch.randn(
                batch_size, self.num_heads, seq_len, head_dim, device=device
            ) * 0.1
            # Initialize Mu as 4D (zeros)
            self._mu_cache[layer_type] = torch.zeros(
                batch_size, self.num_heads, seq_len, head_dim, device=device
            )

            # Lateral connections act on the last dimension (head_dim)
            self.register_lateral(layer_type, head_dim)
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device)

        # --- 3. Attention Scores & Weights (4D: Batch, Head, Seq, Seq) ---
        elif layer_type in {"X_score", "X_A"}:
            assert self.num_heads is not None
            # These are N x N matrices per head, so last dim is seq_len, NOT head_dim
            # Initialize with smaller values to reduce initial energy
            self._x_cache[layer_type] = torch.randn(
                batch_size, self.num_heads, seq_len, seq_len, device=device
            ) * 0.1
            self._mu_cache[layer_type] = torch.zeros(
                batch_size, self.num_heads, seq_len, seq_len, device=device
            )

            # Lateral connections act across heads (last dim after permute)
            self.register_lateral(layer_type, self.num_heads)
            if layer_type in self.lateral_connections:
                 self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device)

        # --- 4. Attention Output (4D: Batch, Head, Seq, Head_Dim) ---
        # You listed this as "running in n_head", so it is the result of (A @ V)
        elif layer_type == "attn_output":
             assert self.n_embed is not None
             head_dim = self.n_embed // self.num_heads
             
             self._x_cache[layer_type] = torch.randn(
                batch_size, self.num_heads, seq_len, head_dim, device=device
            )
             self._mu_cache[layer_type] = torch.zeros(
                batch_size, self.num_heads, seq_len, head_dim, device=device
            )
             
             self.register_lateral(layer_type, head_dim)
             if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device)

        # --- 5. Linear / MLP Layers (3D: Batch, Seq, Embed) ---
        else:  
            ref_layer = bottom_layer or top_layer or layer
            assert ref_layer is not None
            input_dim = ref_layer.weight.shape[1]
            
            self._x_cache[layer_type] = x_init(batch_size, seq_len, input_dim, device) * 0.1
            # Ensure mu matches x shape (3D)
            self._mu_cache[layer_type] = torch.zeros(
                 batch_size, seq_len, input_dim, device=device
            )
            
            self.register_lateral(layer_type, input_dim)  
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device)
    
    def get_x(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached activity tensor for a given layer type."""
        return self._x_cache.get(layer_type, None)
    
    def get_mu(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached mu (prediction) tensor for a given layer type."""
        return self._mu_cache.get(layer_type, None)
    
    def get_td_err(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached top-down error tensor for a given layer type."""
        return self._error_cache.get(layer_type, None)

    def get_td_err_input(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached input top-down error tensor for a given layer type."""
        return self._td_err_cache.get(layer_type, None)

    def get_energy_by_layer(self, layer_type: str) -> Optional[float]:
        """Get the latest energy value for a given layer type."""
        return self._energy_cache.get(layer_type, None)

    def get_energy(self) -> Optional[float]:
        """Get the accumulated energy for the layer."""
        return float(self._energy)

    def clear_energy(self):
        """Clear the stored energy and cached states for the layer."""
        self._energy = 0.0
        self._x_cache.clear()
        self._mu_cache.clear()
        self._error_cache.clear()
        self._td_err_cache.clear()
        self._energy_cache.clear()
        
    def get_errors(self) -> list:
        """Get the list of error values accumulated during inference."""
        return self._errors

    def clear_errors(self):
        """Clear the stored errors for the layer."""
        self._errors = []
        self._error_cache.clear()
        self._td_err_cache.clear()
        
    def set_learning_rate(self, lr: float):
        """Set the local learning rate for the layer."""
        self.local_lr = float(lr)
        for lateral in self.lateral_connections.values():
            lateral.set_learning_rate(lr)
        
    def get_learning_rate(self) -> float:
        """Get the current local learning rate for the layer."""
        return float(self.local_lr)
