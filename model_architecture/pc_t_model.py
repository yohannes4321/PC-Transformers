import torch
import torch.nn as nn
from .embedding import Embedding_Layer
from .transformer_block import TransformerBlock
from utils.pc_utils import ids_to_one_hot, HopfieldMemory
from .output import OutputLayer
from utils.device_utils import create_streams_or_futures, execute_parallel, synchronize_execution
from typing import Optional, Dict, List

class PCTransformer(nn.Module):
    """
    Top-down Predictive Coding Transformer model.

    This model integrates predictive coding principles into a transformer architecture.
    It consists of an embedding layer, multiple transformer blocks, and an output layer,
    each equipped with predictive coding layers for iterative inference and local learning.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = Embedding_Layer(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])
        self.output = OutputLayer(config)
        
        self.init_method = getattr(config, 'init_method', 'random')
        self.hybrid_m = getattr(config, 'hybrid_m', (config.n_blocks * 4 + 2) // 2 + 1)
        
        self.prev_hidden_states: Dict[str, torch.Tensor] = {}
        self.prev_labels: Optional[torch.Tensor] = None
        self.num_classes: int = getattr(config, 'num_classes', 0)
        
        self.hopfield_memories: Dict[str, HopfieldMemory] = {}
        if self.init_method == 'imem':
            self._setup_hopfield_memories()
    
    def _setup_hopfield_memories(self):
        """Setup Hopfield memory modules for each layer for Imem initialization."""
        n_layers = self.config.n_blocks * 4 + 2
        for layer_idx in range(n_layers):
            hidden_dim = self.config.n_embed
            self.hopfield_memories[f"layer_{layer_idx}"] = HopfieldMemory(
                input_dim=hidden_dim,
                memory_dim=hidden_dim,
                delta=1.0
            )
        self.hopfield_memories = nn.ModuleDict(self.hopfield_memories)
    
    def _init_hopfield_with_first_batch(self, input_ids, target_ids):
        """Initialize Hopfield memory with first batch (Ifw-like) to set starting point."""
        if self.init_method != 'imem':
            return
        
        with torch.no_grad():
            self._forward_for_hopfield_init(input_ids, target_ids)
            
            for layer_idx in range(self.config.n_blocks * 4 + 2):
                hopfield_mem = self.hopfield_memories.get(f"layer_{layer_idx}")
                if hopfield_mem is not None:
                    layer_key = self._get_layer_key(layer_idx)
                    hidden_state = self.prev_hidden_states.get(layer_key)
                    if hidden_state is not None:
                        obs = input_ids.float().unsqueeze(1) if layer_idx == 0 else hidden_state
                        hopfield_mem.init_value_matrix(hidden_state, obs)
    
    def _get_layer_key(self, layer_idx: int) -> str:
        """Map layer_idx to hidden state key."""
        if layer_idx == 0:
            return "embed"
        elif layer_idx == self.config.n_blocks * 4 + 1:
            return "output"
        else:
            block_idx = (layer_idx - 1) // 4
            layer_type = (layer_idx - 1) % 4
            if layer_type == 0:
                return f"attn_{block_idx}"
            elif layer_type == 1:
                return f"attn_output_{block_idx}"
            elif layer_type == 2:
                return f"mlp1_{block_idx}"
            else:
                return f"mlp2_{block_idx}"
    
    def _forward_for_hopfield_init(self, input_ids, target_ids):
        """Run forward pass to get hidden states for Hopfield initialization."""
        B, S = input_ids.shape
        device = input_ids.device
        vocab_size = self.output.config.vocab_size
        
        if input_ids.max() >= vocab_size:
            input_ids = torch.clamp(input_ids, max=vocab_size-1)
        if target_ids.max() >= vocab_size:
            target_ids = torch.clamp(target_ids, max=vocab_size-1)
        
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        
        x_word = self.embedding.word_embeddings(input_ids)
        x_pos = self.embedding.position_embeddings(position_ids)
        x = x_word + x_pos
        self.prev_hidden_states["embed"] = x
        
        for block_idx, block in enumerate(self.blocks):
            x = block.attn(block.ln2(x))
            self.prev_hidden_states[f"attn_{block_idx}"] = x
            
            x = block.attn.pc_output.get_mu("linear_attn") if hasattr(block.attn, 'pc_output') and block.attn.pc_output.get_mu("linear_attn") is not None else x
            x = block.mlp(block.ln1(x))
            self.prev_hidden_states[f"mlp1_{block_idx}"] = x
            
            x = block.mlp.pc_layer2.get_mu("fc2") if hasattr(block.mlp, 'pc_layer2') and block.mlp.pc_layer2.get_mu("fc2") is not None else x
            self.prev_hidden_states[f"mlp2_{block_idx}"] = x
    
    def get_hopfield_memory(self, layer_idx: int) -> Optional[HopfieldMemory]:
        """Get Hopfield memory for a specific layer."""
        return self.hopfield_memories.get(f"layer_{layer_idx}", None)
    
    def store_hidden_states(self):
        """Store hidden states from current batch for Iavg initialization in next batch."""
        self.prev_hidden_states["embed"] = self.embedding.pc_layer.get_mu("embed")
        
        for idx, block in enumerate(self.blocks):
            base_idx = idx * 4 + 1
            self.prev_hidden_states[f"attn_{idx}"] = block.attn.pc_qkv.get_mu("attn")
            self.prev_hidden_states[f"attn_output_{idx}"] = block.attn.pc_output.get_mu("linear_attn")
            self.prev_hidden_states[f"mlp1_{idx}"] = block.mlp.pc_layer1.get_mu("fc1")
            self.prev_hidden_states[f"mlp2_{idx}"] = block.mlp.pc_layer2.get_mu("fc2")
        
        self.prev_hidden_states["output"] = self.output.pc_layer.get_mu("linear_output")
    
    def get_prev_hidden_state(self, layer_key: str) -> Optional[torch.Tensor]:
        """Get previous hidden state for a specific layer."""
        return self.prev_hidden_states.get(layer_key, None)
    
    def clear_prev_hidden_states(self):
        """Clear stored hidden states."""
        self.prev_hidden_states.clear()
    
    def set_labels(self, labels: torch.Tensor):
        """Set labels for stream-aligned training."""
        self.prev_labels = labels
    
    def set_num_classes(self, num_classes: int):
        """Set number of classes for stream-aligned training."""
        self.num_classes = num_classes

    def register_all_lateral_weights(self):
        """
        Register lateral weights for all predictive coding layers in the model.
        This enables lateral connections for local learning in each layer.
        """
        for block in self.blocks:
            block.attn.pc_qkv.register_lateral("attn", block.attn.q.in_features)
            block.attn.pc_output.register_lateral("linear", block.attn.output.in_features)
            block.mlp.pc_layer1.register_lateral("fc1", block.mlp.fc1.in_features)
            block.mlp.pc_layer2.register_lateral("linear", block.mlp.fc2.in_features)
        self.output.pc_layer.register_lateral("linear", self.output.output.in_features)

        for module in self.modules():
            if hasattr(module, 'W_latents'):
                for key in module.W_latents:
                    if module.W_latents[key] is not None:
                        module.W_latents[key] = module.W_latents[key].to(next(self.parameters()).device)

    def forward(self, target_ids, input_ids, use_kv_cache=False, labels=None):
        """
        Forward pass of the PCTransformer model, using device-specific parallelism (CUDA streams or torch.jit.fork).

        Args:
            target_ids (torch.Tensor): Target token IDs of shape (B, T).
            input_ids (torch.Tensor): Input token IDs of shape (B, T).
            labels (torch.Tensor): Labels for stream-aligned training (B,). For language modeling, can use target_ids.

        Returns:
            logits (torch.Tensor): Tensor of shape (B, T, vocab_size), the model's output logits for each token position.
        """
        for module in self.modules():
            if hasattr(module, "clear_energy"):
                module.clear_energy()
            
            if hasattr(module, "clear_errors"):
                module.clear_errors()

        B, S = input_ids.shape
        device = input_ids.device
        vocab_size = self.output.config.vocab_size
        
        if labels is None:
            labels = target_ids
        
        num_classes = self.num_classes if self.num_classes > 0 else vocab_size
        
        init_method = self.init_method
        
        if init_method == "iavg" and (self.prev_labels is None or self.prev_hidden_states == {}):
            init_method = "random"
        
        if init_method == "imem" and len(self.hopfield_memories) > 0 and self.prev_hidden_states == {}:
            self._init_hopfield_with_first_batch(input_ids, target_ids)
        
        observations = input_ids.float().unsqueeze(-1) if init_method == "imem" else None
        
        if input_ids.max() >= vocab_size:
            input_ids = torch.clamp(input_ids, max=vocab_size-1)
        
        if target_ids.max() >= vocab_size:
            target_ids = torch.clamp(target_ids, max=vocab_size-1)
        
        target_logits = ids_to_one_hot(target_ids, vocab_size).to(device)
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)

        prev_h_embed = self.get_prev_hidden_state("embed")
        hopfield_mem_0 = self.hopfield_memories.get("layer_0")
        self.embedding.pc_layer.init_x(
            batch_size=B,
            seq_len=S,
            layer_type="embed",
            device = device,
            layer={"word": self.embedding.word_embeddings, "pos": self.embedding.position_embeddings},
            proj_layers=None,
            input_ids=input_ids,
            position_ids=position_ids,
            init_method=init_method,
            prev_hidden_states=prev_h_embed,
            labels=labels,
            num_classes=num_classes,
            layer_idx=0,
            hybrid_m=self.hybrid_m,
            hopfield_memory=hopfield_mem_0,
            observations=observations,
        )

        for block_idx, block in enumerate(self.blocks):
            base_layer_idx = block_idx * 4 + 1
            
            prev_h_attn = self.get_prev_hidden_state(f"attn_{block_idx}")
            hopfield_mem_attn = self.hopfield_memories.get(f"layer_{base_layer_idx}")
            block.attn.pc_qkv.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="attn",
                device = device,
                layer = None,
                proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v},
                input_ids = None,
                position_ids = None,
                init_method=init_method,
                prev_hidden_states=prev_h_attn,
                labels=labels,
                num_classes=num_classes,
                layer_idx=base_layer_idx,
                hybrid_m=self.hybrid_m,
                hopfield_memory=hopfield_mem_attn,
                observations=observations,
            )
            
            prev_h_attn_out = self.get_prev_hidden_state(f"attn_output_{block_idx}")
            hopfield_mem_attn_out = self.hopfield_memories.get(f"layer_{base_layer_idx + 1}")
            block.attn.pc_output.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="linear_attn",
                device=device,
                layer=block.attn.output,
                proj_layers= None, 
                input_ids = None,
                position_ids = None,
                init_method=init_method,
                prev_hidden_states=prev_h_attn_out,
                labels=labels,
                num_classes=num_classes,
                layer_idx=base_layer_idx + 1,
                hybrid_m=self.hybrid_m,
                hopfield_memory=hopfield_mem_attn_out,
                observations=observations,
            )
            
            prev_h_mlp1 = self.get_prev_hidden_state(f"mlp1_{block_idx}")
            hopfield_mem_mlp1 = self.hopfield_memories.get(f"layer_{base_layer_idx + 2}")
            block.mlp.pc_layer1.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="fc1",
                device=device,
                layer=block.mlp.fc1,
                proj_layers= None, 
                input_ids = None,
                position_ids = None,
                init_method=init_method,
                prev_hidden_states=prev_h_mlp1,
                labels=labels,
                num_classes=num_classes,
                layer_idx=base_layer_idx + 2,
                hybrid_m=self.hybrid_m,
                hopfield_memory=hopfield_mem_mlp1,
                observations=observations,
            )
            
            prev_h_mlp2 = self.get_prev_hidden_state(f"mlp2_{block_idx}")
            hopfield_mem_mlp2 = self.hopfield_memories.get(f"layer_{base_layer_idx + 3}")
            block.mlp.pc_layer2.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="fc2",
                device=device,
                layer=block.mlp.fc2,
                proj_layers= None, 
                input_ids = None,
                position_ids = None,
                init_method=init_method,
                prev_hidden_states=prev_h_mlp2,
                labels=labels,
                num_classes=num_classes,
                layer_idx=base_layer_idx + 3,
                hybrid_m=self.hybrid_m,
                hopfield_memory=hopfield_mem_mlp2,
                observations=observations,
            )
        
        prev_h_output = self.get_prev_hidden_state("output")
        output_layer_idx = self.config.n_blocks * 4 + 1
        hopfield_mem_output = self.hopfield_memories.get(f"layer_{output_layer_idx}")
        self.output.pc_layer.init_x(
            batch_size=B,
            seq_len=S,
            layer_type="linear_output",
            device=device,
            layer=self.output.output,
            proj_layers= None, 
            input_ids = None,
            position_ids = None,
            init_method=init_method,
            prev_hidden_states=prev_h_output,
            labels=labels,
            num_classes=num_classes,
            layer_idx=output_layer_idx,
            hybrid_m=self.hybrid_m,
            hopfield_memory=hopfield_mem_output,
            observations=observations,
        )

        # Initialize streams or futures for parallel execution
        use_cuda, streams_or_futures = create_streams_or_futures(device, len(self.blocks) * 4 + 2)

        for t in range(self.config.T):
            # Execute output layer
            td_mlp2 = self.blocks[-1].mlp.pc_layer2.get_td_err("fc2") if t > 0 else None
            execute_parallel(
                use_cuda,
                streams_or_futures,
                self.output.pc_layer.forward,
                target_activity=target_logits,
                layer_type="linear_output",
                t=t,
                T=self.config.T,
                requires_update=True,
                td_err= td_mlp2,
                layer=self.output.output,
                layer_norm=None,
                proj_layers=None,
                input_ids=None,
                position_ids=None,
                flash=False

            )

            # Iterate through blocks in reverse order for parallel execution
            for idx in range(len(self.blocks) - 1, -1, -1):
                block = self.blocks[idx]
                next_target = (
                    self.blocks[idx + 1].attn.pc_qkv.get_x("attn")
                    if idx < len(self.blocks) - 1
                    else self.output.pc_layer.get_x("linear_output")
                )
                
                layer_norm2 = (block.ln2
                   if idx < len(self.blocks) - 1
                    else None)
                td_mlp1 = block.mlp.pc_layer1.get_td_err("fc1") if t > 0 else None

                # Execute MLP layer 2
                execute_parallel(
                    use_cuda,
                    streams_or_futures,
                    block.mlp.pc_layer2.forward,
                    target_activity=next_target,
                    layer_type="fc2",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err= td_mlp1,
                    layer=block.mlp.fc2,
                    layer_norm=layer_norm2,
                    proj_layers=None,
                    input_ids=None,
                    position_ids=None,
                    flash=False

                )
                td_attn_op = block.attn.pc_output.get_td_err("linear_attn") if t > 0 else None

                # Execute MLP layer 1
                execute_parallel(
                    use_cuda,
                    streams_or_futures,
                    block.mlp.pc_layer1.forward,
                    target_activity=block.mlp.pc_layer2.get_x("fc2"),
                    layer_type="fc1",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err= td_attn_op,
                    layer=block.mlp.fc1,
                    layer_norm=block.ln1, 
                    proj_layers=None,
                    input_ids=None,
                    position_ids=None,
                    flash=False

                )
                
                if idx == 0:
                   td_embed = self.embedding.pc_layer.get_td_err("embed") if t > 0 else None
                else:
                   td_embed = self.blocks[idx - 1].mlp.pc_layer2.get_td_err("fc2") if t > 0 else None
                
                td_attn_qkv = block.attn.pc_qkv.get_td_err("attn") if t > 0 else None

    
                # Execute attention output
                execute_parallel(
                    use_cuda,
                    streams_or_futures,
                    block.attn.pc_output.forward,
                    target_activity=block.mlp.pc_layer1.get_x("fc1"),
                    layer_type="linear_attn",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err= td_attn_qkv,
                    layer=block.attn.output, 
                    layer_norm=block.ln1,
                    proj_layers=None,
                    input_ids=None,
                    position_ids=None,
                    flash=False

                )

                # Execute attention QKV
                execute_parallel(
                    use_cuda,
                    streams_or_futures,
                    block.attn.pc_qkv.forward,
                    target_activity=block.attn.pc_output.get_x("linear_attn"),
                    layer_type="attn",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err= td_embed,
                    layer = None,
                    layer_norm=block.ln2,
                    proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v},
                    input_ids=None,
                    position_ids=None,
                    flash=getattr(self.config, 'use_flash_attention', False),
                    use_cache=use_kv_cache,  
                    kv_cache=block.attn.kv_cache if use_kv_cache else None, 
                )

                # Update cache after last iteration
                if use_kv_cache and t == self.config.T - 1:
                    block.attn.kv_cache = block.attn.pc_qkv._last_kv_cache
    
            # Execute embedding layer
            execute_parallel(
                use_cuda,
                streams_or_futures,
                self.embedding.pc_layer.forward,
                target_activity=self.blocks[0].attn.pc_qkv.get_x("attn"),
                layer_type="embed",
                t=t,
                T=self.config.T,
                requires_update=True,
                td_err = None,
                layer={"word": self.embedding.word_embeddings, "pos": self.embedding.position_embeddings},
                layer_norm= block.ln2,
                proj_layers=None,
                input_ids=input_ids,
                position_ids=position_ids,
                flash=False
            )

            # Synchronize all parallel tasks
            synchronize_execution(use_cuda, streams_or_futures)
        
        if self.init_method == "iavg":
            self.store_hidden_states()
            self.set_labels(labels)
        
        logits = self.output.pc_layer.get_mu("linear_output")
        return logits
    