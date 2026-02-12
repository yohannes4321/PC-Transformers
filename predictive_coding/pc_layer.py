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
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        a_weights: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        layer: Optional[nn.Module] = None,
        layer_norm: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        flash: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # ADD THIS
        use_cache: bool = False, 
    ):
        """Perform one predictive coding inference step."""
        self._reset_step_state()
        self._td_err_cache[layer_type] = td_err
        x = self._get_cached_state(layer_type)

        if layer_type == "embed":
            mu, mu_word, mu_pos, bu_err = step_embed(
                t,
                T,
                target_activity,
                layer,
                layer_type,
                input_ids,
                position_ids,
                self.local_lr,
                self.clamp_value,
                self.energy_fn_name,
                requires_update,
                layer_norm=layer_norm,
                optimizer=self.optimizer,
            )            
            # store for later retrieval
            self._x_cache["embed"] = (mu_word, mu_pos)
            self._mu_cache["embed"] = mu.detach().clone()
            if bu_err is not None:
                self._error_cache["embed"] = bu_err.detach().clone()

            # compute energy
            error = target_activity - mu
            energy, step_errors = finalize_step(mu, target_activity, error, t, layer_type, self.energy_fn_name)
            self._energy += energy
            self._energy_cache[layer_type] = energy
            self._errors.extend(step_errors)
            return mu_word, mu_pos
        
        
        elif layer_type == "X_Q":
            lateral_conn = self.lateral_connections.get(layer_type, None)
            x, mu_Q, bu_err = step_Q(
                t,
                T,
                target_activity,
                x,
                lateral_conn,
                proj_layers,
                layer_type,
                self.local_lr,
                self.clamp_value,
                self.energy_fn_name,
                self.update_bias,
                requires_update,
                self.n_embed,
                self.num_heads,
                td_err=td_err, 
                layer_norm=layer_norm,
                optimizer=self.optimizer,
            )
            self._x_cache["X_Q"] = x
            self._mu_cache["X_Q"] = mu_Q.detach().clone()
            if bu_err is not None:
                self._error_cache["X_Q"] = bu_err.detach().clone()

            # compute energy
            if target_activity is None:
                # When target is None, use mu_Q + bu_err
                fallback = mu_Q + bu_err if bu_err is not None else mu_Q
                target_for_energy = _reshape_to_heads(fallback, self.num_heads) if fallback.dim() == 3 else fallback
            else:
                # Handle different target shapes
                if target_activity.dim() == 3:
                    # This could be a merged 3D tensor, reshape to 4D
                    target_for_energy = _reshape_to_heads(target_activity, self.num_heads)
                elif target_activity.dim() == 4:
                    # Check if this is already in the correct format (B, H, S, D)
                    if target_activity.shape[1] == self.num_heads and target_activity.shape[-1] == mu_Q.shape[-1]:
                        target_for_energy = target_activity
                    else:
                        # Reshape if dimensions don't match
                        target_for_energy = _reshape_to_heads(target_activity, self.num_heads)
                else:
                    # Fallback: try to reshape anyway
                    target_for_energy = _reshape_to_heads(target_activity, self.num_heads)
            error = target_for_energy - mu_Q
            energy, step_errors = finalize_step(mu_Q, target_for_energy, error, t, layer_type, self.energy_fn_name)
            self._energy += energy
            self._energy_cache[layer_type] = energy
            self._errors.extend(step_errors)
            return mu_Q
            
        elif layer_type == "X_K":
            lateral_conn = self.lateral_connections.get(layer_type, None)
            x, mu_K, bu_err = step_K(
                t,
                T,
                target_activity,
                x,
                lateral_conn,
                proj_layers,
                layer_type,
                self.local_lr,
                self.clamp_value,
                self.energy_fn_name,
                self.update_bias,
                requires_update,
                self.n_embed,
                self.num_heads,
                td_err=td_err, 
                layer_norm=layer_norm,
                optimizer=self.optimizer,
            )
            self._x_cache["X_K"] = x
            self._mu_cache["X_K"] = mu_K.detach().clone()
            if bu_err is not None:
                self._error_cache["X_K"] = bu_err.detach().clone()

            # compute energy
            if target_activity is None:
                # When target is None, use mu_K + bu_err
                fallback = mu_K + bu_err if bu_err is not None else mu_K
                target_for_energy = _reshape_to_heads(fallback, self.num_heads) if fallback.dim() == 3 else fallback
            else:
                # Handle different target shapes
                if target_activity.dim() == 3:
                    # This could be a merged 3D tensor, reshape to 4D
                    target_for_energy = _reshape_to_heads(target_activity, self.num_heads)
                elif target_activity.dim() == 4:
                    # Check if this is already in the correct format (B, H, S, D)
                    if target_activity.shape[1] == self.num_heads and target_activity.shape[-1] == mu_K.shape[-1]:
                        target_for_energy = target_activity
                    else:
                        # Reshape if dimensions don't match
                        target_for_energy = _reshape_to_heads(target_activity, self.num_heads)
                else:
                    # Fallback: try to reshape anyway
                    target_for_energy = _reshape_to_heads(target_activity, self.num_heads)
            error = target_for_energy - mu_K
            energy, step_errors = finalize_step(mu_K, target_for_energy, error, t, layer_type, self.energy_fn_name)
            self._energy += energy
            self._energy_cache[layer_type] = energy
            self._errors.extend(step_errors)
            return mu_K
        elif layer_type == "X_V":
            lateral_conn = self.lateral_connections.get(layer_type, None)
            x, mu_V, bu_err = step_V(
                t,
                T,
                target_activity,
                x,
                lateral_conn,
                proj_layers,
                layer_type,
                self.local_lr,
                self.clamp_value,
                self.energy_fn_name,
                self.update_bias,
                requires_update,
                self.n_embed,
                self.num_heads,
                td_err=td_err, 
                layer_norm=layer_norm,
                optimizer=self.optimizer,
            )
            self._x_cache["X_V"] = x
            self._mu_cache["X_V"] = mu_V.detach().clone()
            if bu_err is not None:
                self._error_cache["X_V"] = bu_err.detach().clone()

            # compute energy
            if target_activity is None:
                # When target is None, use mu_V + bu_err
                fallback = mu_V + bu_err if bu_err is not None else mu_V
                target_for_energy = _reshape_to_heads(fallback, self.num_heads) if fallback.dim() == 3 else fallback
            else:
                # Handle different target shapes
                if target_activity.dim() == 3:
                    # This could be a merged 3D tensor, reshape to 4D
                    target_for_energy = _reshape_to_heads(target_activity, self.num_heads)
                elif target_activity.dim() == 4:
                    # Check if this is already in the correct format (B, H, S, D)
                    if target_activity.shape[1] == self.num_heads and target_activity.shape[-1] == mu_V.shape[-1]:
                        target_for_energy = target_activity
                    else:
                        # Reshape if dimensions don't match
                        target_for_energy = _reshape_to_heads(target_activity, self.num_heads)
                else:
                    # Fallback: try to reshape anyway
                    target_for_energy = _reshape_to_heads(target_activity, self.num_heads)
            error = target_for_energy - mu_V
            energy, step_errors = finalize_step(mu_V, target_for_energy, error, t, layer_type, self.energy_fn_name)
            self._energy += energy
            self._energy_cache[layer_type] = energy
            self._errors.extend(step_errors)
            return mu_V
        elif layer_type == "X_score":
            lateral_conn = self.lateral_connections.get(layer_type, None)
            q_source = q if q is not None else self._mu_cache.get("X_Q")
            k_source = k if k is not None else self._mu_cache.get("X_K")
            x, mu_score, bu_err, new_kv_cache = step_X_score(
                t,
                T,
                target_activity,
                x,
                lateral_conn,
                layer_type,
                self.local_lr,
                self.clamp_value,
                self.energy_fn_name,
                requires_update,
                self.num_heads,
                self.n_embed,
                td_err=td_err, 
                q=q_source,
                k=k_source,
                kv_cache=kv_cache,
                use_cache=use_cache,
            )
            self._x_cache["X_score"] = mu_score
            self._mu_cache["X_score"] = mu_score.detach().clone()
            if bu_err is not None:
                self._error_cache["X_score"] = bu_err.detach().clone()
            target_for_energy = target_activity if target_activity is not None else mu_score
            error = target_for_energy - mu_score
            energy, step_errors = finalize_step(mu_score, target_for_energy, error, t, layer_type, self.energy_fn_name)
            self._energy += energy
            self._energy_cache[layer_type] = energy
            self._errors.extend(step_errors)
            return mu_score

        elif layer_type == "X_A":
            lateral_conn = self.lateral_connections.get(layer_type, None)
            score_source = score if score is not None else self._mu_cache.get("X_score")
            energy_fn_name_xa = "kld"
            x, mu_X_A, bu_err = step_X_A(
                t,
                T,
                target_activity,
                x,
                lateral_conn,
                layer_type,
                self.local_lr,
                self.clamp_value,
                energy_fn_name_xa,
                requires_update,
                td_err=td_err, 
                score=score_source,
            )
            self._x_cache["X_A"] = mu_X_A
            self._mu_cache["X_A"] = mu_X_A.detach().clone()
            if bu_err is not None:
                self._error_cache["X_A"] = bu_err.detach().clone()
            mu_logits = x if x is not None else mu_X_A
            target_for_energy = F.softmax(score_source, dim=-1) if score_source is not None else mu_X_A
            error = target_for_energy - mu_X_A
            energy, step_errors = finalize_step(mu_logits, target_for_energy, error, t, layer_type, energy_fn_name_xa)
            self._energy += energy
            self._energy_cache[layer_type] = energy
            self._errors.extend(step_errors)
            return mu_X_A
            # Store cache for retrieval
        elif layer_type == "attn_output":
            lateral_conn = self.lateral_connections.get(layer_type, None)
            a_source = a_weights if a_weights is not None else self._mu_cache.get("X_A")
            v_source = v if v is not None else self._mu_cache.get("X_V")
            x, mu_attnOut, bu_err = step_X_attnOut(
                t,
                T,
                target_activity,
                x,
                lateral_conn,
                layer,
                layer_type,
                self.local_lr,
                self.clamp_value,
                self.energy_fn_name,
                self.update_bias,
                requires_update,
                td_err=td_err, 
                layer_norm=layer_norm,
                a_weights=a_source,
                v=v_source,
                optimizer=self.optimizer,
            )
            # Store cache for retrieval
            self._x_cache["attn_output"] = mu_attnOut
            self._mu_cache["attn_output"] = mu_attnOut.detach().clone()
            if bu_err is not None:
                self._error_cache["attn_output"] = bu_err.detach().clone()
            error = target_activity - mu_attnOut
            energy, step_errors = finalize_step(mu_attnOut, target_activity, error, t, layer_type, self.energy_fn_name)
            self._energy += energy
            self._energy_cache[layer_type] = energy
            self._errors.extend(step_errors)
            return mu_attnOut
           
        
        else:
            lateral_conn = self.lateral_connections.get(layer_type, None)
            x, mu, bu_err = step_linear(
                t,
                T,
                target_activity,
                x,
                layer, 
                lateral_conn,  
                layer_type,
                self.local_lr, 
                self.clamp_value, 
                self.energy_fn_name, 
                self.update_bias, 
                requires_update,
                td_err=td_err, 
                layer_norm=layer_norm,
                optimizer=self.optimizer,
            )
            
        # cache and stats
        self._mu_cache[layer_type] = mu.detach().clone()  
        if bu_err is not None:
            self._error_cache[layer_type] = bu_err.detach().clone()
        
        error = target_activity - mu
        energy, step_errors = finalize_step(mu, target_activity, error, t, layer_type, self.energy_fn_name)
        self._energy += energy
        self._energy_cache[layer_type] = energy
        self._errors.extend(step_errors)
        # update x cache
        self._x_cache[layer_type] = x
        return x, mu

    def init_x(
        self,
        batch_size: int,
        seq_len: int,
        layer_type: str,
        device: torch.device,
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
            vocab_size = layer["word"].weight.size(0)
            max_pos = layer["pos"].weight.size(0)
            
            # Safety clamp
            input_ids = torch.clamp(input_ids, max=vocab_size-1)
            position_ids = torch.clamp(position_ids, max=max_pos-1)
            
            x_word = layer["word"].weight[input_ids] 
            x_pos = layer["pos"].weight[position_ids] 
            self._x_cache["embed"] = (x_word, x_pos)
            # Embed doesn't usually use a stored '_mu' in the same way, 
            # but if needed, it would be 3D.

        # --- 2. Attention Projections (4D: Batch, Head, Seq, Head_Dim) ---
        elif layer_type in {"X_Q", "X_K", "X_V"}:
            assert proj_layers is not None
            # Total embedding dim
            d_model = proj_layers["q_proj"].weight.shape[1] 
            head_dim = d_model // self.num_heads

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
            assert layer is not None
            input_dim = layer.weight.shape[1]
            
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
