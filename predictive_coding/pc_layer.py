import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from utils.pc_utils import (
    x_init,
    step_embed,
    step_linear,
    step_attn,
    finalize_step,
)
from predictive_coding.lateral_connc import LateralConnections
from predictive_coding.init_memory import StreamAverageMemory, HopfieldInitMemory

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
    ):
        super().__init__()
        self.T = T
        self.local_lr = lr
        self.update_bias = update_bias
        self.clamp_value = 3.0
        self.energy_fn_name = energy_fn_name 
        self.num_heads = num_heads
        self.n_embed = n_embed
        
        self.lateral_connections: Dict[str, LateralConnections] = {}
        
        self._x_cache: Dict[str, torch.Tensor] = {}
        self._mu_cache: Dict[str, torch.Tensor] = {}
        self._error_cache: Dict[str, torch.Tensor] = {}
        self._energy = 0.0
        self._errors = []
        self.stream_memories = nn.ModuleDict()
        self.hopfield_memories = nn.ModuleDict()
    
    def register_lateral(self, layer_type: str, size: int):
        """Create and register lateral connections for layer_type."""
        if layer_type not in self.lateral_connections:
            self.lateral_connections[layer_type] = LateralConnections(size, self.local_lr)
            self.add_module(f"lateral_{layer_type}", self.lateral_connections[layer_type])

    def _reset_step_state(self) -> None:
        """Reset step-local accumulators, kept for future extension."""
        return
    
    def _get_cached_state(self, layer_type: str):
        return self._x_cache.get(layer_type, None)

    def _init_stream_memory(
        self,
        layer_type: str,
        hidden_dim: int,
        num_classes: int,
        momentum: float,
        device: Optional[torch.device] = None,
    ):
        module_name = f"stream_{layer_type}"
        if module_name not in self.stream_memories:
            self.stream_memories[module_name] = StreamAverageMemory(num_classes, hidden_dim, momentum)
        if device is not None:
            self.stream_memories[module_name] = self.stream_memories[module_name].to(device)

    def _init_hopfield_memory(
        self,
        layer_type: str,
        hidden_dim: int,
        obs_dim: int,
        memory_slots: int,
        memory_temperature: float,
        memory_lr: float,
        device: Optional[torch.device] = None,
    ):
        module_name = f"hopfield_{layer_type}"
        if module_name not in self.hopfield_memories:
            self.hopfield_memories[module_name] = HopfieldInitMemory(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim,
                num_slots=memory_slots,
                temperature=memory_temperature,
                memory_lr=memory_lr,
            )
        if device is not None:
            self.hopfield_memories[module_name] = self.hopfield_memories[module_name].to(device)

    def update_initialization_memory(
        self,
        layer_type: str,
        settled_x: torch.Tensor,
        stream_labels: Optional[torch.Tensor] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> None:
        if settled_x is None:
            return

        stream_name = f"stream_{layer_type}"
        if stream_labels is not None and stream_name in self.stream_memories:
            self.stream_memories[stream_name] = self.stream_memories[stream_name].to(settled_x.device)
            self.stream_memories[stream_name].update(settled_x.detach(), stream_labels.detach())

        hopfield_name = f"hopfield_{layer_type}"
        if observation is not None and hopfield_name in self.hopfield_memories:
            self.hopfield_memories[hopfield_name] = self.hopfield_memories[hopfield_name].to(settled_x.device)
            self.hopfield_memories[hopfield_name].update(observation.detach(), settled_x.detach())

    def set_x(self, layer_type: str, x_value: torch.Tensor) -> None:
        self._x_cache[layer_type] = x_value
    
    def forward(
        self,
        target_activity: torch.Tensor,
        layer_type: str,
        t: int,
        T: int,
        requires_update: bool,
        td_err:  Optional[torch.Tensor] = None,
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
            self._errors.extend(step_errors)
            return mu_word, mu_pos
        
        elif layer_type == "attn":
            lateral_conn = self.lateral_connections.get(layer_type, None)
            x, mu, bu_err, new_kv_cache = step_attn(
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
                self.num_heads,
                self.n_embed,
                td_err=td_err, 
                layer_norm=layer_norm,
                flash=flash, 
                kv_cache=kv_cache,  
                use_cache=use_cache,
            )
            # Store cache for retrieval
            if use_cache:
                self._last_kv_cache = new_kv_cache
        
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
                layer_norm=layer_norm
            )
            
        # cache and stats
        self._mu_cache[layer_type] = mu.detach().clone()  
        if bu_err is not None: 
         self._error_cache[layer_type] = bu_err.detach().clone()   
        
        error = target_activity - mu
        energy, step_errors = finalize_step(mu, target_activity, error, t, layer_type, self.energy_fn_name)
        self._energy += energy
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
        init_strategy: str = "random",
        stream_labels: Optional[torch.Tensor] = None,
        observation: Optional[torch.Tensor] = None,
        num_classes: int = 64,
        stream_momentum: float = 0.1,
        obs_dim: int = 8,
        memory_slots: int = 128,
        memory_temperature: float = 1.0,
        memory_lr: float = 0.05,
    ):
        """
        Initialize cached activity `x` for the layer type.
        - embed: stores (x_word, x_pos) from embedding weights
        - attn: creates random initialization shaped (B, S, H_out)
        - linear/others: random init sized to layer input dimension
        """
        if layer_type == "embed":
            assert input_ids is not None and position_ids is not None, "Embedding layer requires input_ids and position_ids"
            vocab_size = layer["word"].weight.size(0)
            if input_ids.max() >= vocab_size:
                input_ids = torch.clamp(input_ids, max=vocab_size-1)
            
            max_pos = layer["pos"].weight.size(0)
            if position_ids.max() >= max_pos:
                position_ids = torch.clamp(position_ids, max=max_pos-1)
            
            x_word = layer["word"].weight[input_ids] 
            x_pos = layer["pos"].weight[position_ids] 
            self._x_cache["embed"] = (x_word, x_pos)
            
        elif layer_type == "attn":
            assert proj_layers is not None, "Attention layer requires proj_layers"
            H_in = proj_layers["q_proj"].weight.shape[1]
            H_out = proj_layers["v_proj"].weight.shape[0] 
            x_tensor = None

            if init_strategy in {"stream_avg", "hybrid"} and stream_labels is not None:
                self._init_stream_memory(layer_type, H_out, num_classes, stream_momentum, device=device)
                stream_memory = self.stream_memories[f"stream_{layer_type}"]
                x_tensor = stream_memory.retrieve(stream_labels).unsqueeze(1).expand(-1, seq_len, -1).clone()

            if x_tensor is None and init_strategy in {"memory", "hybrid"} and observation is not None:
                self._init_hopfield_memory(
                    layer_type,
                    H_out,
                    obs_dim,
                    memory_slots,
                    memory_temperature,
                    memory_lr,
                    device=device,
                )
                hopfield_memory = self.hopfield_memories[f"hopfield_{layer_type}"]
                retrieved, _ = hopfield_memory.retrieve(observation)
                x_tensor = retrieved.unsqueeze(1).expand(-1, seq_len, -1).clone()

            self._x_cache["attn"] = x_tensor if x_tensor is not None else x_init(batch_size, seq_len, H_out, device)
            
            self.register_lateral(layer_type, H_in)
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device) 
        
        else:  
            assert layer is not None, "Linear layer requires layer parameter"
            input_dim = layer.weight.shape[1]
            x_tensor = None

            if init_strategy in {"stream_avg", "hybrid"} and stream_labels is not None:
                self._init_stream_memory(layer_type, input_dim, num_classes, stream_momentum, device=device)
                stream_memory = self.stream_memories[f"stream_{layer_type}"]
                x_tensor = stream_memory.retrieve(stream_labels).unsqueeze(1).expand(-1, seq_len, -1).clone()

            if x_tensor is None and init_strategy in {"memory", "hybrid"} and observation is not None:
                self._init_hopfield_memory(
                    layer_type,
                    input_dim,
                    obs_dim,
                    memory_slots,
                    memory_temperature,
                    memory_lr,
                    device=device,
                )
                hopfield_memory = self.hopfield_memories[f"hopfield_{layer_type}"]
                retrieved, _ = hopfield_memory.retrieve(observation)
                x_tensor = retrieved.unsqueeze(1).expand(-1, seq_len, -1).clone()

            self._x_cache[layer_type] = x_tensor if x_tensor is not None else x_init(batch_size, seq_len, input_dim, device)
            
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

    def get_energy(self) -> Optional[float]:
        """Get the accumulated energy for the layer."""
        return float(self._energy)

    def clear_energy(self):
        """Clear the stored energy and cached states for the layer."""
        self._energy = 0.0
        self._x_cache.clear()
        self._mu_cache.clear()
        
    def get_errors(self) -> list:
        """Get the list of error values accumulated during inference."""
        return self._errors

    def clear_errors(self):
        """Clear the stored errors for the layer."""
        self._errors = []
        
    def set_learning_rate(self, lr: float):
        """Set the local learning rate for the layer."""
        self.local_lr = float(lr)
        
    def get_learning_rate(self) -> float:
        """Get the current local learning rate for the layer."""
        return float(self.local_lr)
