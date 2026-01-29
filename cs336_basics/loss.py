import torch


def cross_entropy_loss_naive(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross entropy loss between logits and target indices.

    Args:
        logits (torch.Tensor): Logits with shape (Batch_Size, Vocab_Size)
        targets (torch.Tensor): Target indices with shape (Batch_Size)

    Returns:
        torch.Tensor: Average cross entropy loss across the batch
    """
    # --- 1. Numerical Stability (The "Max Trick") ---
    # We find the maximum value in each row (dim=-1). 
    # keepdim=True preserves shape (Batch, 1) so we can broadcast subtraction later.
    max_logits = logits.max(dim=-1, keepdim=True).values 
    
    # Subtracting max(x) prevents huge numbers. e^1000 causes overflow (Infinity), 
    # but e^0 is safe. This shift doesn't change the resulting probabilities.
    logits_shifted = logits - max_logits

    # --- 2. Log-Softmax Calculation ---
    # Formula: log( exp(x_i) / sum(exp(x_j)) ) 
    #        = x_i - log( sum(exp(x_j)) )
    
    # Calculate the denominator: sum(exp(x_j))
    sum_exp = torch.exp(logits_shifted).sum(dim=-1, keepdim=True)
    
    # Take the log of the denominator
    log_sum_exp = torch.log(sum_exp)
    
    # Perform the subtraction: Numerator (x_i) - Denominator Term
    # This gives us the Log Probability for *every* class.
    log_probs = logits_shifted - log_sum_exp

    # --- 3. Select Targets (The Gather) ---
    # We have log_probs for all classes, but we only want the one corresponding 
    # to the correct target index.
    
    # unsqueeze(-1): Changes targets from shape (Batch) to (Batch, 1) 
    # to match the dimensions of log_probs for gather.
    target_indices = targets.unsqueeze(-1)
    
    # gather: Looks up the value at 'target_indices' for each row.
    # squeeze(-1): Removes the extra dimension, returning shape back to (Batch).
    target_log_probs = log_probs.gather(dim=-1, index=target_indices).squeeze(-1)

    # --- 4. Negative Log Likelihood (NLL) ---
    # Cross Entropy minimizes the negative log probability of the true class.
    # We negate the values (making them positive loss) and average across the batch.
    return -target_log_probs.mean()
