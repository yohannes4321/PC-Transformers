import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamAverageMemory(nn.Module):
    """Class-conditional running averages used for stream-aligned initialization."""

    def __init__(self, num_classes: int, hidden_dim: int, momentum: float = 0.1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.momentum = float(momentum)

        self.register_buffer("class_memory", torch.zeros(self.num_classes, self.hidden_dim))
        self.register_buffer("class_counts", torch.zeros(self.num_classes))

    def retrieve(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long().clamp(min=0, max=self.num_classes - 1)
        return self.class_memory[labels]

    def update(self, settled_x: torch.Tensor, labels: torch.Tensor) -> None:
        labels = labels.long().clamp(min=0, max=self.num_classes - 1)
        x_summary = settled_x.mean(dim=1) if settled_x.dim() == 3 else settled_x

        with torch.no_grad():
            for class_idx in labels.unique().tolist():
                class_mask = labels == class_idx
                if not class_mask.any():
                    continue

                batch_mean = x_summary[class_mask].mean(dim=0)
                if self.class_counts[class_idx] == 0:
                    self.class_memory[class_idx].copy_(batch_mean)
                else:
                    self.class_memory[class_idx].mul_(1.0 - self.momentum).add_(self.momentum * batch_mean)

                self.class_counts[class_idx] += class_mask.sum().float()


class HopfieldInitMemory(nn.Module):
    """Hopfield-like associative memory for unsupervised initialization."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_slots: int = 128,
        temperature: float = 1.0,
        memory_lr: float = 0.05,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.memory_lr = float(memory_lr)

        self.query = nn.Linear(self.obs_dim, self.hidden_dim, bias=True)
        self.keys = nn.Parameter(torch.randn(self.num_slots, self.hidden_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(self.num_slots, self.hidden_dim) * 0.02)
        self.log_temp = nn.Parameter(torch.tensor(float(temperature)).log())

    def _temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp).clamp(min=1e-3, max=100.0)

    def retrieve(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(observation)
        logits = (query @ self.keys.t()) * self._temperature()
        attention = F.softmax(logits, dim=-1)
        retrieved = attention @ self.values
        return retrieved, attention

    def update(self, observation: torch.Tensor, settled_x: torch.Tensor) -> None:
        target = settled_x.mean(dim=1) if settled_x.dim() == 3 else settled_x
        retrieved, attention = self.retrieve(observation)
        error = target - retrieved

        with torch.no_grad():
            delta_v = attention.t() @ error / max(observation.shape[0], 1)
            self.values.add_(self.memory_lr * torch.clamp(delta_v, -0.05, 0.05))

            delta_k = attention.t() @ self.query(observation) / max(observation.shape[0], 1)
            self.keys.add_(self.memory_lr * 0.1 * torch.tanh(delta_k))
