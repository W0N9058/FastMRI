import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
        
    def __call__(self, kspace, target, fname, slice):
        # Convert data to tensor
        kspace = to_tensor(kspace)
        target = to_tensor(target) if not self.isforward else -1

        # Check if the input is complex (has two dimensions for real and imaginary parts)
        if kspace.ndim == 4 and kspace.shape[-1] == 2:
            kspace = torch.stack((kspace[..., 0], kspace[..., 1]), dim=-1)
        else:
            # If the input is not complex, we assume it is in a suitable format
            kspace = kspace.unsqueeze(-1)  # Add a channel dimension for consistency

        # Handle maximum value key
        maximum = self.max_key if not self.isforward else -1
        
        return kspace, target, maximum, fname, slice