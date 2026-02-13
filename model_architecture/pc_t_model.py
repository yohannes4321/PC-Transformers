import torch
import torch.nn as nn
from .embedding import Embedding_Layer
from .transformer_block import TransformerBlock
from utils.pc_utils import ids_to_one_hot
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
            embed_layers={"word": self.embedding.word_embeddings, "pos": self.embedding.position_embeddings},
            input_ids=input_ids, position_ids=position_ids
        )

        for block in self.blocks:
            # Initialize Attention Latents
            block.attn.pc_X_Q.init_x(batch_size=B, seq_len=S, layer_type="X_Q", device=device)
            block.attn.pc_X_K.init_x(batch_size=B, seq_len=S, layer_type="X_K", device=device)
            block.attn.pc_X_V.init_x(batch_size=B, seq_len=S, layer_type="X_V", device=device)
            block.attn.pc_X_score.init_x(batch_size=B, seq_len=S, layer_type="X_score", device=device)
            block.attn.pc_X_A.init_x(batch_size=B, seq_len=S, layer_type="X_A", device=device)
            
            # Initialize Output and MLP Latents
            block.attn.pc_output.init_x(batch_size=B, seq_len=S, layer_type="attn_output", device=device)
            block.mlp.pc_layer1.init_x(batch_size=B, seq_len=S, layer_type="fc1", device=device, bottom_layer=block.mlp.attnoutput_fc1)
            block.mlp.pc_layer2.init_x(batch_size=B, seq_len=S, layer_type="fc2", device=device, bottom_layer=block.mlp.fc1_fc2)

        self.output.pc_layer.init_x(batch_size=B, seq_len=S, layer_type="linear_output", device=device, bottom_layer=self.output.fc2_linear_output)

        use_cuda, streams_or_futures = create_streams_or_futures(device, len(self.blocks) * 8 + 2)

        # 3. Embedding Layer Forward
               
        for t in range(self.config.T):
            first_block = self.blocks[0]
            execute_parallel(
                    use_cuda, streams_or_futures, self.embedding.pc_layer.forward,
                    target=None,
                    current_state=self.embedding.pc_layer.get_x("embed"),
                    target_q_for_embed=first_block.attn.pc_X_Q.get_x("X_Q"),
                    target_k_for_embed=first_block.attn.pc_X_K.get_x("X_K"),
                    target_v_for_embed=first_block.attn.pc_X_V.get_x("X_V"),
                    top_layers={"q": first_block.attn.q, "k": first_block.attn.k, "v": first_block.attn.v},
                    layer_type="embed", t=t, T=self.config.T,
                    requires_update=True, previous=None,
                    embed_layers={"word": self.embedding.word_embeddings, "pos": self.embedding.position_embeddings},
                    input_ids=input_ids, position_ids=position_ids
                ) 
            

            # 2. Reverse Block Iteration
            for idx in range(len(self.blocks) - 1, -1, -1):
                block = self.blocks[idx]
                
                

     
                previous_embedorfc2 = self.embedding.pc_layer.get_x("embed") if idx == 0 else self.blocks[idx-1].mlp.pc_layer2.get_x("fc2")
                ####

                
                target_score=block.attn.pc_X_score.get_x("X_score")
                
                execute_parallel(
                    use_cuda,
                    streams_or_futures,
                    block.attn.pc_X_Q.forward,
                    target=target_score,
                    current_state=block.attn.pc_X_Q.get_x("X_Q"),
                    layer_type="X_Q",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    previous=previous_embedorfc2,
                    bottom_layer=block.attn.q,
                    top_layer=block.attn.q_and_score,
                    layer_norm=block.ln2,
                )

                execute_parallel(
                    use_cuda,
                    streams_or_futures,
                    block.attn.pc_X_K.forward,
                    target=target_score,
                    current_state=block.attn.pc_X_K.get_x("X_K"),
                    layer_type="X_K",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    previous=previous_embedorfc2,
                    bottom_layer=block.attn.k,
                    top_layer=block.attn.k_and_score,
                    layer_norm=block.ln2,
                )
               


                
                
                
      
               
                
                execute_parallel(
                    use_cuda, streams_or_futures, block.attn.pc_X_score.forward,
                    target=block.attn.pc_X_A.get_x("X_A"),
                    current_state=block.attn.pc_X_score.get_x("X_score") ,
                    layer_type="X_score",
                    t=t, T=self.config.T, requires_update=True,
                    previous_score_q=block.attn.pc_X_Q.get_x("X_Q"),
                    previous_score_k=block.attn.pc_X_K.get_x("X_K"),
                    bottom_layer_q_score=block.attn.q_and_score,
                    bottom_layer_k_score=block.attn.k_and_score,
                    top_layer=block.attn.score_X_A,
                )
            
                execute_parallel(
                    use_cuda, streams_or_futures, block.attn.pc_X_A.forward,
                    target=block.attn.pc_output.get_x("attn_output"),
                    current_state=block.attn.pc_X_A.get_x("X_A") ,

                    layer_type="X_A",
                    t=t, T=self.config.T, requires_update=True,
                    previous=block.attn.pc_X_score.get_x("X_score") ,
                    bottom_layer=block.attn.score_X_A,
                    top_layer=block.attn.X_A_and_Attenout,
           
            
                )

                

                execute_parallel(
                    use_cuda,
                    streams_or_futures,
                    block.attn.pc_X_V.forward,
                    target=block.attn.pc_output.get_x("attn_output"),
                    current_state=block.attn.pc_X_V.get_x("X_V"),
                    layer_type="X_V",
                    t=t,
                    T=self.config.T,
                    requires_update=True,
                    previous=previous_embedorfc2,
                    bottom_layer=block.attn.v,
                    top_layer=block.attn.v_and_attenout,
                    layer_norm=block.ln2,
                )
                

                

                execute_parallel(
                    use_cuda, streams_or_futures, block.attn.pc_output.forward,
                    target=block.mlp.pc_layer1.get_x("fc1"), 
                    current_state=block.attn.pc_output.get_x("attn_output"),
                    layer_type="attn_output",
                    t=t, T=self.config.T, requires_update=True, 
                    previous_attenout_X_A=block.attn.pc_X_A.get_x("X_A"),
                    previous_attenout_xscore=block.attn.pc_X_score.get_x("X_score"),
                    bottom_layer_attenout_X_A=block.attn.X_A_and_Attenout,
                    bottom_layer_attenout_xscore=block.attn.score_X_A,
                    top_layer=block.mlp.attnoutput_fc1,
                    layer_norm=block.ln1
                )
              
                # Determine target for the end of the block (MLP)
                if idx < len(self.blocks) - 1:
                    next_target = None
                else:
                    next_target = self.output.pc_layer.get_x("linear_output")
                   
                
                # --- MLP Layer 1 ---
                execute_parallel(
                    use_cuda, streams_or_futures, block.mlp.pc_layer1.forward,
                    target=block.mlp.pc_layer2.get_x("fc2"),
                    current_state= block.mlp.pc_layer1.get_x("fc1") ,
                    layer_type="fc1",
                    previous=block.attn.pc_output.get_x("attn_output"),
                    t=t, T=self.config.T, requires_update=True,
                    bottom_layer=block.mlp.attnoutput_fc1,
                    top_layer=block.mlp.fc1_fc2,
                    layer_norm=block.ln1
                )
                # --- MLP Layer 2 ---
                execute_parallel(
                    use_cuda, streams_or_futures, block.mlp.pc_layer2.forward,
                    target= next_target if idx >=len(self.blocks) else None,
                    layer_type="fc2", 
                    current_state=block.mlp.pc_layer2.get_x("fc2"),
                    t=t, T=self.config.T,
                    requires_update=True,
                    previous=block.mlp.pc_layer1.get_x("fc1"),
                    bottom_layer=block.mlp.fc1_fc2,
                    top_layer=self.output.fc2_linear_output,
                    layer_norm=block.ln2,
                )

            # 1. Output Layer Forward
           
            execute_parallel(
                use_cuda, streams_or_futures, self.output.pc_layer.forward,
                current_state= self.output.pc_layer.get_x("linear_output"),
                target=target_logits, layer_type="linear_output",
                t=t, T=self.config.T, requires_update=True,
                previous=self.blocks[-1].mlp.pc_layer2.get_x("fc2") ,
                bottom_layer=self.output.fc2_linear_output,
            )    

            

            synchronize_execution(use_cuda, streams_or_futures)

        return self.output.pc_layer.get_mu("linear_output")