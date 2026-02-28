import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader, DistributedSampler, Subset
from data_preparation.config import encoded_dir, max_len, batch_size
from data_preparation.dataset import EncodedDataset

def get_datasets(max_batches: int = 30):
    """Load train, validation, and test datasets, truncated to save time."""
    base_dir = Path(encoded_dir)
    train_dataset = EncodedDataset(base_dir/"train.pt", max_len)
    valid_dataset = EncodedDataset(base_dir/"valid.pt", max_len)
    test_dataset = EncodedDataset(base_dir/"test.pt", max_len)
    
    # Calculate the exact number of samples needed for 30 batches
    max_samples = max_batches * batch_size
    
    # Restrict the datasets using Subset
    train_dataset = Subset(train_dataset, range(min(max_samples, len(train_dataset))))
    valid_dataset = Subset(valid_dataset, range(min(max_samples, len(valid_dataset))))
    test_dataset = Subset(test_dataset, range(min(max_samples, len(test_dataset))))
    
    return train_dataset, valid_dataset, test_dataset

def get_loaders(distributed: bool = False):
    """Wrap datasets into PyTorch DataLoaders with batching and shuffling."""
    train_dataset, valid_dataset, test_dataset = get_datasets(max_batches=30)
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = valid_sampler = test_sampler = None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        shuffle=(train_sampler is None), 
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size,
        sampler=valid_sampler,
        shuffle=False,  
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader