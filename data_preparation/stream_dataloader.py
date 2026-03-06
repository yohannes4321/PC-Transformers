import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset
from data_preparation.config import encoded_dir, max_len, batch_size
from data_preparation.dataset import EncodedDataset


class StreamAlignedDataset(Dataset):
    """
    Dataset that organizes data by streams (groups of samples with same label).
    For language modeling, the "label" is the next token.
    
    This implements stream-aligned training from the research paper:
    - Group samples by their target (next token)
    - Each stream contains samples with the same target
    """
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        self.streams = {}
        self._create_streams()
        
    def _create_streams(self):
        """Group samples by their target (next token)."""
        for i in range(len(self.data) - self.seq_len):
            input_seq = self.data[i:i + self.seq_len]
            target = self.data[i + self.seq_len].item()
            
            if target not in self.streams:
                self.streams[target] = []
            self.streams[target].append((input_seq, target))
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq_len], self.data[idx + self.seq_len]


def collate_stream_aligned(batch, num_classes=1000):
    """
    Collate function that creates stream-aligned batches.
    Groups samples by their target and arranges them sequentially.
    
    Args:
        batch: List of (input, target) tuples
        num_classes: Number of unique targets to consider
    
    Returns:
        Stream-aligned input_ids and target_ids
    """
    input_seqs = torch.stack([item[0] for item in batch])
    targets = torch.tensor([item[1] for item in batch])
    
    target_counts = {}
    for t in targets:
        t_val = t.item() if t.dim() > 0 else t
        target_counts[t_val] = target_counts.get(t_val, 0) + 1
    
    sorted_indices = sorted(range(len(targets)), key=lambda i: (
        -target_counts[targets[i].item() if targets[i].dim() > 0 else targets[i]]
    ))
    
    sorted_inputs = input_seqs[sorted_indices]
    sorted_targets = targets[sorted_indices]
    
    return sorted_inputs, sorted_targets


def get_stream_aligned_loaders(distributed=False, num_classes=1000):
    """
    Get stream-aligned dataloaders for training.
    This implements the stream-aligned training from the research paper.
    """
    train_dataset = EncodedDataset(encoded_dir/"train.pt", max_len)
    valid_dataset = EncodedDataset(encoded_dir/"valid.pt", max_len)
    test_dataset = EncodedDataset(encoded_dir/"test.pt", max_len)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=True,
        collate_fn=lambda x: collate_stream_aligned(x, num_classes)
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader


def get_loaders(distributed: bool = False):
    """Original dataloader."""
    return get_stream_aligned_loaders(distributed)
