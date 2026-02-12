import torch
import torch.nn as nn
import torch.nn.functional as F

class LateralConnections(nn.Module):
    """
    Manages lateral connections for a layer.
    Implements anti-Hebbian learning for decorrelation.
    """
    def __init__(self, size: int, local_lr: float):
        super().__init__()
        self.size = size
        self.local_lr = local_lr
        
        # Initialize lateral weight matrix
        W = torch.empty(size, size)
        nn.init.xavier_uniform_(W)
        self.W_lateral = nn.Parameter(W)
    
    def forward(self, x: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        """
        Apply lateral connections to combine error with lateral influence.
        
        Args:
            x: Current layer activity (B, S, H)
            error: Prediction error (B, S, H)
            
        Returns:
            delta_x: Combined error + lateral influence (B, S, H)
        """
        if self.W_lateral.device != x.device:
            self.W_lateral.data = self.W_lateral.data.to(x.device)
        if x.shape[-1] != self.size:
            raise ValueError("LateralConnections expects last dim size to match W_lateral")
        x_flat = x.reshape(-1, self.size)
        error_flat = error.reshape(-1, self.size)
        x_latent_flat = x_flat @ self.W_lateral
        delta_flat = error_flat + x_latent_flat
        return delta_flat.view_as(error)
    
    def update_weights(self, x: torch.Tensor, optimizer=None, clamp_value: float = None):
        """Anti-Hebbian weight update for decorrelation."""
        with torch.no_grad():
            if self.W_lateral.device != x.device:
                self.W_lateral.data = self.W_lateral.data.to(x.device)
            if x.shape[-1] != self.size:
                raise ValueError("LateralConnections expects last dim size to match W_lateral")
            x_flat = x.reshape(-1, self.size)
            anti_hebbian = -x_flat.t() @ x_flat
            if optimizer is not None:
                optimizer.step_param(self.W_lateral, anti_hebbian, self.local_lr, clamp_value=clamp_value)
            else:
                self.W_lateral.data.add_(self.local_lr * anti_hebbian)

            self.W_lateral.data = F.normalize(self.W_lateral.data, p=2, dim=1)
            
    def set_learning_rate(self, lr: float):
        """Set the local learning rate for the layer."""
        self.local_lr = float(lr)