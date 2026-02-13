"""
Predictive Coding Transformer (PCT) Model

This module implements a complete Predictive Coding Transformer with:
1. Self-Attention Blocks with LayerNorm and skip connections
2. MLP Blocks with LayerNorm and skip connections
3. Activity updates: dx/dt = -ε + Σ(ε_j^(l+1) * w_ij)
4. Weight updates: dw/dt = ε * x
5. Lateral connections for local learning

Architecture:
Input
  ↓
Embedding
  ↓
[For each block]:
  LayerNorm → Attention → +skip
  LayerNorm → MLP → +skip
  ↓
Final LayerNorm (optional)
  ↓
Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from model_architecture.embedding import Embedding_Layer
from model_architecture.transformer_block import TransformerBlock
from model_architecture.output import OutputLayer
from utils.pc_utils import ids_to_one_hot
from utils.device_utils import create_streams_or_futures, execute_parallel, synchronize_execution
from predictive_coding.pc_layer import PCLayer


class PCTransformer(nn.Module):
    """
    Complete Predictive Coding Transformer model.
    
    Implements the equations:
    - Activity Update: dx_i^l/dt = -ε_i^l + Σ_j ε_j^(l+1) * w_ij^l
    - Weight Update: dw_ik^l/dt = ε_k^(l+1) * x_i^l
    
    With self-attention blocks containing:
    - LayerNorm before QKV projections
    - Skip connection after attention
    - LayerNorm before MLP
    - Skip connection after MLP
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = Embedding_Layer(config)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_blocks)
        ])
        
        # Output layer
        self.output = OutputLayer(config)
        
        # Optional final layer norm
        self.final_ln = nn.LayerNorm(config.n_embed) if hasattr(config, 'use_final_ln') and config.use_final_ln else None
        
        # PC Layers for each component
        self._init_pc_layers(config)
    
    def _init_pc_layers(self, config):
        """Initialize predictive coding layers for all components."""
        # Embedding PC layer
        self.embedding.pc_layer = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=config.update_bias,
            energy_fn_name=config.energy_fn_name,
            optimizer_name=getattr(config, 'optimizer_name', 'adam'),
            optimizer_beta1=getattr(config, 'optimizer_beta1', 0.9),
            optimizer_beta2=getattr(config, 'optimizer_beta2', 0.999),
        )
        
        # PC layers for each block
        for block in self.blocks:
            head_dim = config.n_embed // config.num_heads
            
            # Attention Q, K, V projections
            block.attn.pc_X_Q = PCLayer(
                T=config.T,
                lr=config.lr,
                update_bias=config.update_bias,
                energy_fn_name=config.internal_energy_fn_name,
                num_heads=config.num_heads,
                n_embed=config.n_embed,
            )
            block.attn.pc_X_K = PCLayer(
                T=config.T,
                lr=config.lr,
                update_bias=config.update_bias,
                energy_fn_name=config.internal_energy_fn_name,
                num_heads=config.num_heads,
                n_embed=config.n_embed,
            )
            block.attn.pc_X_V = PCLayer(
                T=config.T,
                lr=config.lr,
                update_bias=config.update_bias,
                energy_fn_name=config.internal_energy_fn_name,
                num_heads=config.num_heads,
                n_embed=config.n_embed,
            )
            
            # Attention scores and weights
            block.attn.pc_X_score = PCLayer(
                T=config.T,
                lr=config.lr,
                update_bias=config.update_bias,
                energy_fn_name=config.internal_energy_fn_name,
                num_heads=config.num_heads,
                n_embed=config.n_embed,
            )
            block.attn.pc_X_A = PCLayer(
                T=config.T,
                lr=config.lr,
                update_bias=config.update_bias,
                energy_fn_name=config.internal_energy_fn_name,
                num_heads=config.num_heads,
                n_embed=config.n_embed,
            )
            
            # Attention output
            block.attn.pc_output = PCLayer(
                T=config.T,
                lr=config.lr,
                update_bias=config.update_bias,
                energy_fn_name=config.internal_energy_fn_name,
                num_heads=config.num_heads,
                n_embed=config.n_embed,
            )
            
            # MLP layers
            block.mlp.pc_layer1 = PCLayer(
                T=config.T,
                lr=config.lr,
                update_bias=config.update_bias,
                energy_fn_name=config.internal_energy_fn_name,
            )
            block.mlp.pc_layer2 = PCLayer(
                T=config.T,
                lr=config.lr,
                update_bias=config.update_bias,
                energy_fn_name=config.internal_energy_fn_name,
            )
            
            # LayerNorm PC layers (for latent state tracking)
            block.pc_ln1 = PCLayer(
                T=config.T,
                lr=config.lr,
                update_bias=False,
                energy_fn_name=config.internal_energy_fn_name,
            )
            block.pc_ln2 = PCLayer(
                T=config.T,
                lr=config.lr,
                update_bias=False,
                energy_fn_name=config.internal_energy_fn_name,
            )
        
        # Output PC layer
        self.output.pc_layer = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=config.update_bias,
            energy_fn_name=config.energy_fn_name,
        )
    
    def register_all_lateral_weights(self):
        """Register lateral connections for all layers."""
        for block in self.blocks:
            head_dim = self.config.n_embed // self.config.num_heads
            
            # Register lateral connections for attention
            block.attn.pc_X_Q.register_lateral("X_Q", head_dim)
            block.attn.pc_X_K.register_lateral("X_K", head_dim)
            block.attn.pc_X_V.register_lateral("X_V", head_dim)
            block.attn.pc_output.register_lateral("attn_output", head_dim)
            
            # Register lateral connections for MLP
            fc1_dim = block.mlp.fc1.weight.shape[0]
            fc2_dim = block.mlp.fc2.weight.shape[0]
            block.mlp.pc_layer1.register_lateral("fc1", fc1_dim)
            block.mlp.pc_layer2.register_lateral("fc2", fc2_dim)
        
        # Output lateral connections
        output_dim = self.output.output.weight.shape[0]
        self.output.pc_layer.register_lateral("output", output_dim)
        
        # Move all lateral weights to correct device
        for module in self.modules():
            if hasattr(module, 'lateral_connections'):
                for key, conn in module.lateral_connections.items():
                    if conn is not None:
                        conn.to(next(self.parameters()).device)
    
    def forward(self, input_ids: torch.Tensor, target_ids: Optional[torch.Tensor] = None):
        """
        Forward pass with predictive coding inference.
        
        Args:
            input_ids: Input token IDs (B, S)
            target_ids: Target token IDs for training (B, S)
        
        Returns:
            logits: Output logits (B, S, vocab_size)
        """
        B, S = input_ids.shape
        device = input_ids.device
        
        # Clear all caches
        for module in self.modules():
            if hasattr(module, 'clear_energy'):
                module.clear_energy()
            if hasattr(module, 'clear_errors'):
                module.clear_errors()
        
        # Convert target to one-hot if provided
        if target_ids is not None:
            vocab_size = self.config.vocab_size
            target_logits = ids_to_one_hot(target_ids, vocab_size).to(device)
        else:
            target_logits = None
        
        # Position IDs
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        
        # Initialize all PC layers
        self.embedding.pc_layer.init_x(
            batch_size=B,
            seq_len=S,
            layer_type="embed",
            device=device,
            word_embeddings=self.embedding.word_embeddings,
            pos_embeddings=self.embedding.position_embeddings,
            input_ids=input_ids,
            position_ids=position_ids
        )
        
        for block in self.blocks:
            # Initialize attention PC layers
            block.attn.pc_X_Q.init_x(B, S, "X_Q", device)
            block.attn.pc_X_K.init_x(B, S, "X_K", device)
            block.attn.pc_X_V.init_x(B, S, "X_V", device)
            block.attn.pc_X_score.init_x(B, S, "X_score", device)
            block.attn.pc_X_A.init_x(B, S, "X_A", device)
            block.attn.pc_output.init_x(B, S, "attn_output", device)
            
            # Initialize MLP PC layers
            block.mlp.pc_layer1.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="fc1",
                device=device,
                bottom_layer=block.mlp.fc1
            )
            block.mlp.pc_layer2.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="fc2",
                device=device,
                bottom_layer=block.mlp.fc2
            )
            
            # Initialize LayerNorm PC layers
            block.pc_ln1.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="ln1",
                device=device,
                bottom_layer=block.mlp.fc1
            )
            block.pc_ln2.init_x(
                batch_size=B,
                seq_len=S,
                layer_type="ln2",
                device=device,
                bottom_layer=block.mlp.fc1
            )
        
        self.output.pc_layer.init_x(
            batch_size=B,
            seq_len=S,
            layer_type="output",
            device=device,
            bottom_layer=self.output.output
        )
        
        # Iterative inference over T steps
        for t in range(self.config.T):
            # Process embedding layer
            self._step_embedding(t, B, S, device, input_ids, position_ids)
            
            # Process transformer blocks
            for block_idx, block in enumerate(self.blocks):
                self._step_transformer_block(t, block_idx, block, B, S, device)
            
            # Process output layer
            self._step_output(t, target_logits)
        
        # Return final logits
        return self.output.pc_layer.get_x("output")
    
    def _step_embedding(self, t: int, B: int, S: int, device: torch.device, 
                       input_ids: torch.Tensor, position_ids: torch.Tensor):
        """Process embedding layer for one inference step."""
        embed_x = self.embedding.pc_layer.get_x("embed")
        first_block = self.blocks[0]
        
        # Get targets from Q, K, V layers of first block
        target_q = first_block.attn.pc_X_Q.get_x("X_Q")
        target_k = first_block.attn.pc_X_K.get_x("X_K")
        target_v = first_block.attn.pc_X_V.get_x("X_V")
        
        # Forward pass
        x_new, mu, error = self.embedding.pc_layer.forward(
            layer_type="embed",
            t=t,
            T=self.config.T,
            requires_update=True,
            current_state=embed_x,
            target=target_q,
            target_k=target_k,
            target_v=target_v,
            word_embeddings=self.embedding.word_embeddings,
            pos_embeddings=self.embedding.position_embeddings,
            input_ids=input_ids,
            position_ids=position_ids,
            q_proj=first_block.attn.q,
            k_proj=first_block.attn.k,
            v_proj=first_block.attn.v,
            layer_norm=None  # No LayerNorm before embedding
        )
    
    def _step_transformer_block(self, t: int, block_idx: int, block, B: int, S: int, device: torch.device):
        """Process one transformer block for one inference step."""
        
        # Get previous layer output (embedding or previous block's MLP output)
        if block_idx == 0:
            prev_output = self.embedding.pc_layer.get_x("embed")
        else:
            prev_output = self.blocks[block_idx - 1].mlp.pc_layer2.get_x("fc2")
        
        # ===== SELF-ATTENTION BLOCK =====
        # Skip connection 1
        skip1 = prev_output
        
        # LayerNorm 1 (before attention)
        ln1_output = block.ln1(prev_output) if hasattr(block, 'ln1') else prev_output
        block.pc_ln1._x_cache["ln1"] = ln1_output
        
        # Q, K, V projections with predictive coding
        self._step_qkv_projections(t, block, ln1_output)
        
        # Attention scores
        self._step_attention_scores(t, block)
        
        # Attention weights (softmax)
        self._step_attention_weights(t, block)
        
        # Attention output
        self._step_attention_output(t, block)
        
        # Add skip connection 1: x = attn_output + skip1
        attn_output = block.attn.pc_output.get_x("attn_output")
        if attn_output.dim() == 4:  # (B, H, S, D)
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)
        block.pc_ln1._x_cache["ln1"] = attn_output + skip1  # Store for MLP
        
        # ===== MLP BLOCK =====
        # Skip connection 2
        skip2 = block.pc_ln1.get_x("ln1")
        
        # LayerNorm 2 (before MLP)
        ln2_output = block.ln2(skip2) if hasattr(block, 'ln2') else skip2
        block.pc_ln2._x_cache["ln2"] = ln2_output
        
        # MLP Layer 1
        self._step_mlp1(t, block, ln2_output)
        
        # MLP Layer 2
        self._step_mlp2(t, block)
        
        # Add skip connection 2: x = fc2_output + skip2
        fc2_output = block.mlp.pc_layer2.get_x("fc2")
        final_output = fc2_output + skip2
        block.mlp.pc_layer2._x_cache["fc2"] = final_output
    
    def _step_qkv_projections(self, t: int, block, previous: torch.Tensor):
        """Process Q, K, V projections."""
        # Q projection
        x_q = block.attn.pc_X_Q.get_x("X_Q")
        x_q_new, mu_q, error_q = block.attn.pc_X_Q.forward(
            layer_type="X_Q",
            t=t,
            T=self.config.T,
            requires_update=True,
            current_state=x_q,
            target=block.attn.pc_X_score.get_x("X_score"),
            previous=previous,
            proj_layer=block.attn.q,
            layer_norm=None,  # Already applied
            num_heads=self.config.num_heads
        )
        
        # K projection
        x_k = block.attn.pc_X_K.get_x("X_K")
        x_k_new, mu_k, error_k = block.attn.pc_X_K.forward(
            layer_type="X_K",
            t=t,
            T=self.config.T,
            requires_update=True,
            current_state=x_k,
            target=block.attn.pc_X_score.get_x("X_score"),
            previous=previous,
            proj_layer=block.attn.k,
            layer_norm=None,
            num_heads=self.config.num_heads
        )
        
        # V projection
        x_v = block.attn.pc_X_V.get_x("X_V")
        x_v_new, mu_v, error_v = block.attn.pc_X_V.forward(
            layer_type="X_V",
            t=t,
            T=self.config.T,
            requires_update=True,
            current_state=x_v,
            target=block.attn.pc_output.get_x("attn_output"),
            previous=previous,
            proj_layer=block.attn.v,
            layer_norm=None,
            num_heads=self.config.num_heads
        )
    
    def _step_attention_scores(self, t: int, block):
        """Process attention scores (Q @ K^T / sqrt(d))."""
        q = block.attn.pc_X_Q.get_x("X_Q")
        k = block.attn.pc_X_K.get_x("X_K")
        x_score = block.attn.pc_X_score.get_x("X_score")
        
        x_score_new, mu_score, error_score = block.attn.pc_X_score.forward(
            layer_type="X_score",
            t=t,
            T=self.config.T,
            requires_update=True,
            current_state=x_score,
            target=block.attn.pc_X_A.get_x("X_A"),
            q=q,
            k=k,
            causal_mask=True
        )
    
    def _step_attention_weights(self, t: int, block):
        """Process attention weights (softmax of scores)."""
        scores = block.attn.pc_X_score.get_x("X_score")
        x_a = block.attn.pc_X_A.get_x("X_A")
        
        x_a_new, mu_a, error_a = block.attn.pc_X_A.forward(
            layer_type="X_A",
            t=t,
            T=self.config.T,
            requires_update=True,
            current_state=x_a,
            target=block.attn.pc_output.get_x("attn_output"),
            scores=scores
        )
    
    def _step_attention_output(self, t: int, block):
        """Process attention output (weights @ V @ Wo)."""
        weights = block.attn.pc_X_A.get_x("X_A")
        v = block.attn.pc_X_V.get_x("X_V")
        x_out = block.attn.pc_output.get_x("attn_output")
        
        # Get target from MLP layer 1
        target = block.mlp.pc_layer1.get_x("fc1")
        
        x_out_new, mu_out, error_out = block.attn.pc_output.forward(
            layer_type="attn_output",
            t=t,
            T=self.config.T,
            requires_update=True,
            current_state=x_out,
            target=target,
            weights=weights,
            v=v,
            output_proj=block.attn.output,
            layer_norm=None  # Skip connection will handle this
        )
    
    def _step_mlp1(self, t: int, block, previous: torch.Tensor):
        """Process MLP layer 1."""
        x_fc1 = block.mlp.pc_layer1.get_x("fc1")
        target_fc2 = block.mlp.pc_layer2.get_x("fc2")
        
        x_fc1_new, mu_fc1, error_fc1 = block.mlp.pc_layer1.forward(
            layer_type="fc1",
            t=t,
            T=self.config.T,
            requires_update=True,
            current_state=x_fc1,
            target=target_fc2,
            previous=previous,
            fc_layer=block.mlp.fc1,
            activation="gelu",
            layer_norm=None  # Already applied
        )
    
    def _step_mlp2(self, t: int, block):
        """Process MLP layer 2."""
        x_fc2 = block.mlp.pc_layer2.get_x("fc2")
        x_fc1 = block.mlp.pc_layer1.get_x("fc1")
        
        # Get target from next block or output
        block_idx = self.blocks.index(block)
        if block_idx < len(self.blocks) - 1:
            target = None  # Next block will compute
        else:
            target = self.output.pc_layer.get_x("output")
        
        x_fc2_new, mu_fc2, error_fc2 = block.mlp.pc_layer2.forward(
            layer_type="fc2",
            t=t,
            T=self.config.T,
            requires_update=True,
            current_state=x_fc2,
            target=target,
            previous=x_fc1,
            fc_layer=block.mlp.fc2,
            activation=None,  # No activation after fc2
            layer_norm=None
        )
    
    def _step_output(self, t: int, target_logits: Optional[torch.Tensor]):
        """Process output layer."""
        x_out = self.output.pc_layer.get_x("output")
        
        # Get input from last block's MLP
        last_block = self.blocks[-1]
        previous = last_block.mlp.pc_layer2.get_x("fc2")
        
        x_out_new, logits, error = self.output.pc_layer.forward(
            layer_type="output",
            t=t,
            T=self.config.T,
            requires_update=True,
            current_state=x_out,
            target=target_logits,
            previous=previous,
            output_layer=self.output.output,
            layer_norm=self.final_ln  # Optional final layer norm
        )
    
    def get_energy(self) -> float:
        """Get total energy from all PC layers."""
        total_energy = 0.0
        
        # Embedding energy
        total_energy += self.embedding.pc_layer.get_energy()
        
        # Block energies
        for block in self.blocks:
            total_energy += block.attn.pc_X_Q.get_energy()
            total_energy += block.attn.pc_X_K.get_energy()
            total_energy += block.attn.pc_X_V.get_energy()
            total_energy += block.attn.pc_X_score.get_energy()
            total_energy += block.attn.pc_X_A.get_energy()
            total_energy += block.attn.pc_output.get_energy()
            total_energy += block.mlp.pc_layer1.get_energy()
            total_energy += block.mlp.pc_layer2.get_energy()
        
        # Output energy
        total_energy += self.output.pc_layer.get_energy()
        
        return total_energy
    
    def clear_caches(self):
        """Clear all PC layer caches."""
        self.embedding.pc_layer.clear_all()
        for block in self.blocks:
            block.attn.pc_X_Q.clear_all()
            block.attn.pc_X_K.clear_all()
            block.attn.pc_X_V.clear_all()
            block.attn.pc_X_score.clear_all()
            block.attn.pc_X_A.clear_all()
            block.attn.pc_output.clear_all()
            block.mlp.pc_layer1.clear_all()
            block.mlp.pc_layer2.clear_all()
            block.pc_ln1.clear_all()
            block.pc_ln2.clear_all()
        self.output.pc_layer.clear_all()
