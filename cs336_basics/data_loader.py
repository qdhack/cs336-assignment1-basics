import torch
import numpy as np

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    """
    Randomly samples a batch of sequences from the training data.

    Args:
        x: np.ndarray — The full dataset as a flat array of token IDs.
        batch_size: int — How many independent sequences to process in parallel.
        context_length: int — The size of the time window (sequence length) for each sample.
        device: str — The target device (e.g., 'cuda', 'cpu', 'mps').

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - x_batch: Inputs of shape (batch_size, context_length)
            - y_batch: Targets of shape (batch_size, context_length) (inputs shifted right by 1)
    """
    
    # Calculate the last valid starting index.
    # We need room for the input sequence (context_length) PLUS one extra token 
    # because the target 'y' is shifted one step to the right.
    max_start_idx = len(x) - context_length - 1

    if max_start_idx < 0:
        raise ValueError(f"Input array length {len(x)} is too short for context_length {context_length}")

    # Generate random starting positions for the entire batch in parallel.
    # We use (max_start_idx + 1) because randint is exclusive on the upper bound.
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)

    # Pre-allocate arrays for inputs (x) and targets (y) to avoid dynamic resizing.
    x_sequences = np.zeros((batch_size, context_length), dtype=np.int64)
    y_sequences = np.zeros((batch_size, context_length), dtype=np.int64)

    # Slice the data for each batch index.
    # If x is [0, 1, 2, 3, 4] and context_length is 3:
    # Input becomes [0, 1, 2] and Target becomes [1, 2, 3].
    for i, start_idx in enumerate(start_indices):
        x_sequences[i] = x[start_idx : start_idx + context_length]
        y_sequences[i] = x[start_idx + 1 : start_idx + context_length + 1]

    # Convert numpy arrays to PyTorch tensors.
    x_batch = torch.from_numpy(x_sequences)
    y_batch = torch.from_numpy(y_sequences)

    # Move data to the target device.
    if device.startswith("cuda"):
        # Optimization: Pin memory (page-lock) to enable faster, direct memory access (DMA) 
        # transfers to the GPU. 'non_blocking=True' allows the CPU to proceed 
        # asynchronously without waiting for the transfer to complete.
        x_batch = x_batch.pin_memory().to(device, non_blocking=True)
        y_batch = y_batch.pin_memory().to(device, non_blocking=True)
    else:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

    return x_batch, y_batch
