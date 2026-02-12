import torch
import torch.nn as nn
from .embedding import Embedding_Layer
from .transformer_block import TransformerBlock
from utils.pc_utils import ids_to_one_hot, _merge_heads
from .output import OutputLayer
from utils.device_utils import create_streams_or_futures, execute_parallel, synchronize_execution
import math
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

    def register_all_lateral_weights(self):
        """
        Register lateral weights for all predictive coding layers in the model.
        This enables lateral connections for local learning in each layer.
        """
        for block in self.blocks:
            head_dim = block.attn.n_embed // block.attn.num_heads
            block.attn.pc_X_Q.register_lateral("X_Q", head_dim)
            block.attn.pc_X_K.register_lateral("X_K", head_dim)
            block.attn.pc_X_V.register_lateral("X_V", head_dim)
            block.mlp.pc_layer1.register_lateral("fc1", block.mlp.fc1.in_features)
            block.attn.pc_output.register_lateral("attn_output", head_dim)
            block.mlp.pc_layer2.register_lateral("fc2", block.mlp.fc2.in_features)
        self.output.pc_layer.register_lateral("linear_output", self.output.output.in_features)



        for module in self.modules():
            if hasattr(module, 'W_latents'):
                for key in module.W_latents:
                    if module.W_latents[key] is not None:
                        module.W_latents[key] = module.W_latents[key].to(next(self.parameters()).device)

    def forward(self, target_ids, input_ids, use_kv_cache=False):
        """
        Forward pass of the PCTransformer model.
        """
        for module in self.modules():
            if hasattr(module, "clear_energy"):
                module.clear_energy()
            if hasattr(module, "clear_errors"):
                module.clear_errors()

        B, S = input_ids.shape
        device = input_ids.device
        vocab_size = self.output.config.vocab_size
        
        # Clip input_ids and target_ids to valid range
        input_ids = torch.clamp(input_ids, max=vocab_size-1)
        target_ids = torch.clamp(target_ids, max=vocab_size-1)
        
        target_logits = ids_to_one_hot(target_ids, vocab_size).to(device)
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)

        # Initialize all PC layers
        self.embedding.pc_layer.init_x(
            batch_size=B, seq_len=S, layer_type="embed", device=device,
            layer={"word": self.embedding.word_embeddings, "pos": self.embedding.position_embeddings},
            input_ids=input_ids, position_ids=position_ids
        )

        for block in self.blocks:
            # Initialize Attention Latents
            block.attn.pc_X_Q.init_x(batch_size=B, seq_len=S, layer_type="X_Q", device=device,
                                    proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v})
            block.attn.pc_X_K.init_x(batch_size=B, seq_len=S, layer_type="X_K", device=device,
                                    proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v})
            block.attn.pc_X_V.init_x(batch_size=B, seq_len=S, layer_type="X_V", device=device,
                                    proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v})
            block.attn.pc_X_score.init_x(batch_size=B, seq_len=S, layer_type="X_score", device=device)
            block.attn.pc_X_A.init_x(batch_size=B, seq_len=S, layer_type="X_A", device=device)
            
            # Initialize Output and MLP Latents
            block.attn.pc_output.init_x(batch_size=B, seq_len=S, layer_type="attn_output", device=device, layer=block.attn.output)
            block.mlp.pc_layer1.init_x(batch_size=B, seq_len=S, layer_type="fc1", device=device, layer=block.mlp.fc1)
            block.mlp.pc_layer2.init_x(batch_size=B, seq_len=S, layer_type="fc2", device=device, layer=block.mlp.fc2)

        self.output.pc_layer.init_x(batch_size=B, seq_len=S, layer_type="linear_output", device=device, layer=self.output.output)

        use_cuda, streams_or_futures = create_streams_or_futures(device, len(self.blocks) * 8 + 2)

        def ensure_3d(tensor: torch.Tensor) -> torch.Tensor:
            if tensor is not None and tensor.dim() == 4:
                return _merge_heads(tensor)
            return tensor
        # 3. Embedding Layer Forward
               
        for t in range(self.config.T):
            first_block = self.blocks[0]
            embed_target = (ensure_3d(first_block.attn.pc_X_Q.get_x("X_Q")) + 
                            ensure_3d(first_block.attn.pc_X_K.get_x("X_K")) + 
                                ensure_3d(first_block.attn.pc_X_V.get_x("X_V"))) / 3.0
                
            execute_parallel(
                    use_cuda, streams_or_futures, self.embedding.pc_layer.forward,
                    target_activity=embed_target, layer_type="embed", t=t, T=self.config.T,
                    requires_update=True, td_err=None,
                    layer={"word": self.embedding.word_embeddings, "pos": self.embedding.position_embeddings},
                    input_ids=input_ids, position_ids=position_ids
                ) 
            

            # 2. Reverse Block Iteration
            for idx in range(len(self.blocks) - 1, -1, -1):
                block = self.blocks[idx]
                
                

     
                block_td_source = self.embedding.pc_layer.get_td_err("embed") if idx == 0 else self.blocks[idx-1].mlp.pc_layer2.get_td_err("fc2")
                ####

                
                target_score=block.attn.pc_X_score.get_x("X_score")
                
                execute_parallel(
                    use_cuda,
                    streams_or_futures,
                    block.attn.pc_X_Q.forward,
                    target_activity=target_score,
                    layer_type="X_Q",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err=block_td_source,
                    layer=None,
                    layer_norm=block.ln2,
                    proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v},
                    input_ids=None,
                    position_ids=None,
                    flash=False,
                )

                execute_parallel(
                    use_cuda,
                    streams_or_futures,
                    block.attn.pc_X_K.forward,
                    target_activity=target_score,
                    layer_type="X_K",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err=block_td_source,
                    layer=None,
                    layer_norm=block.ln2,
                    proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v},
                    input_ids=None,
                    position_ids=None,
                    flash=False,
                )
                target_attn_output = block.attn.pc_output.get_x("attn_output")


                
                q_mu = block.attn.pc_X_Q.get_mu("X_Q")
                k_mu = block.attn.pc_X_K.get_mu("X_K")
                

                # Compute Score TD error properly - use actual attention scores
                head_dim = q_mu.shape[-1]
                mu_score = torch.matmul(q_mu, k_mu.transpose(-2, -1)) / (head_dim ** 0.5)
                
                # Get the actual score prediction
                x_score = block.attn.pc_X_score.get_x("X_score")
                if x_score is not None:
                    td_score = x_score - mu_score
                else:
                    td_score = None
      
               
                # Execute Attention mechanism PC nodes
                execute_parallel(
                    use_cuda, streams_or_futures, block.attn.pc_X_score.forward,
                    target_activity=block.attn.pc_X_A.get_x("X_A"), layer_type="X_score",
                    t=t, T=self.config.T, requires_update=True, td_err=td_score,
                    q=q_mu, k=k_mu, use_cache=use_kv_cache
                )
                execute_parallel(
                    use_cuda, streams_or_futures, block.attn.pc_X_A.forward,
                    target_activity=target_attn_output, layer_type="X_A",
                    t=t, T=self.config.T, requires_update=True,
                    td_err=block.attn.pc_X_score.get_td_err("X_score") ,
                    score=block.attn.pc_X_score.get_mu("X_score"),
            
                )

                

                execute_parallel(
                    use_cuda,
                    streams_or_futures,
                    block.attn.pc_X_V.forward,
                    target_activity=target_attn_output,
                    layer_type="X_V",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    td_err=block_td_source,
                    layer=None,
                    layer_norm=block.ln2,
                    proj_layers={"q_proj": block.attn.q, "k_proj": block.attn.k, "v_proj": block.attn.v},
                    input_ids=None,
                    position_ids=None,
                    flash=False,
                )
                 # Compute Attention Output TD error from Softmax(A) and V errors
                v_mu = block.attn.pc_X_V.get_mu("X_V")
                a_mu = block.attn.pc_X_A.get_mu("X_A")
                td_err_xa = block.attn.pc_X_A.get_td_err("X_A")
                td_err_xv = block.attn.pc_X_V.get_td_err("X_V")
                td_X_A_X_V = td_err_xv
                if td_err_xa is not None and td_err_xv is not None:
                    td_a = td_err_xa.mean(dim=-1, keepdim=True)
                    td_a = td_a.expand_as(td_err_xv)
                    td_X_A_X_V = td_err_xv + td_a

                

                execute_parallel(
                    use_cuda, streams_or_futures, block.attn.pc_output.forward,
                    target_activity=block.mlp.pc_layer1.get_x("fc1"), layer_type="attn_output",
                    t=t, T=self.config.T, requires_update=True, td_err=td_X_A_X_V,
                    a_weights=a_mu, v=v_mu, layer=block.attn.output, layer_norm=block.ln1
                )
                td_mlp1 = block.mlp.pc_layer1.get_td_err("fc1") 
                td_attn_op = block.attn.pc_output.get_td_err("attn_output")
                # Determine target for the end of the block (MLP)
                if idx < len(self.blocks) - 1:
                    next_block = self.blocks[idx + 1]
                    # Averaging Q, K, V targets as top-down pressure
                    next_q = ensure_3d(next_block.attn.pc_X_Q.get_x("X_Q"))
                    next_k = ensure_3d(next_block.attn.pc_X_K.get_x("X_K"))
                    next_v = ensure_3d(next_block.attn.pc_X_V.get_x("X_V"))
                    next_target = (next_q + next_k + next_v) / 3.0
                    layer_norm_mlp2 = block.ln2
                else:
                    next_target = self.output.pc_layer.get_x("linear_output")
                    layer_norm_mlp2 = None
                # --- MLP Layer 2 ---
                execute_parallel(
                    use_cuda, streams_or_futures, block.mlp.pc_layer2.forward,
                    target_activity=next_target, layer_type="fc2", t=t, T=self.config.T,
                    requires_update=True, td_err=td_mlp1, layer=block.mlp.fc2,
                    layer_norm=layer_norm_mlp2
                )

                # --- MLP Layer 1 ---
                execute_parallel(
                    use_cuda, streams_or_futures, block.mlp.pc_layer1.forward,
                    target_activity=block.mlp.pc_layer2.get_x("fc2"), layer_type="fc1",
                    t=t, T=self.config.T, requires_update=True, td_err=td_attn_op,
                    layer=block.mlp.fc1, layer_norm=block.ln1
                )
            # 1. Output Layer Forward
            td_mlp2_last = self.blocks[-1].mlp.pc_layer2.get_td_err("fc2") 
            execute_parallel(
                use_cuda, streams_or_futures, self.output.pc_layer.forward,
                target_activity=target_logits, layer_type="linear_output",
                t=t, T=self.config.T, requires_update=True, td_err=td_mlp2_last,
                layer=self.output.output, flash=False
            )    

            

            synchronize_execution(use_cuda, streams_or_futures)

        return self.output.pc_layer.get_mu("linear_output")