import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import Transformer


def decode(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 52,
    temperature: float = 0.7,
    top_p: float = 0.8,
):
    """
    Generates text from a prompt using a pre-trained Transformer model.
    Uses temperature scaling and top-p (nucleus) sampling.
    """
    
    # 1. PREPARATION
    # Get the ID for the special "End of Text" token to know when to stop early.
    end_id = tokenizer.encode("<|endoftext|>")[0]
    
    # Convert the input string prompt into a list of integer token IDs.
    input_ids = tokenizer.encode(prompt)
    
    # Determine which hardware device (CPU or GPU) the model is on to move tensors there.
    device = next(model.parameters()).device
    
    # Retrieve the model's maximum allowed input size (e.g., 1024 or 2048 tokens).
    context_length = model.context_length

    # Disable gradient calculation because we are only inferencing (generating), not training.
    # This saves memory and computation.
    with torch.no_grad():
        
        # 2. GENERATION LOOP
        # Loop up to the maximum number of new tokens we want to generate.
        for _ in range(max_new_tokens):
            
            # 2a. CONTEXT WINDOWING
            # If the current sequence is longer than the model can handle, 
            # slice it to keep only the most recent 'context_length' tokens.
            window_input_ids = input_ids[-context_length:] if len(input_ids) >= context_length else input_ids
            
            # Create a PyTorch tensor from the list and move it to the correct device (GPU/CPU).
            # Shape: [1, sequence_length] (Batch size of 1)
            x = torch.tensor([window_input_ids], dtype=torch.long, device=device)

            # 2b. MODEL FORWARD PASS
            # Run the model. Output 'logits' usually has shape [Batch, Seq_Len, Vocab_Size].
            logits = model(x)
            
            # We only care about the logits for the *last* token in the sequence, 
            # because that determines the prediction for the next upcoming token.
            # Shape: [Vocab_Size]
            next_logits = logits[0, -1, :]

            # 2c. TEMPERATURE SCALING
            # Divide logits by temperature. 
            # temp < 1.0 makes peaks pointier (more confident).
            # temp > 1.0 makes distribution flatter (more random).
            scaled = next_logits / temperature
            
            # Numerical stability: subtract max value to prevent overflow when doing .exp() later.
            # (Math property: softmax(x) == softmax(x - c))
            stable = scaled - scaled.max()
            
            # Convert logits (unnormalized scores) to probabilities (0.0 to 1.0).
            exp_vals = stable.exp()
            probs = exp_vals / exp_vals.sum()

            # 2d. TOP-P (NUCLEUS) SAMPLING
            # Sort probabilities from highest to lowest.
            sorted_probs, sorted_idxs = torch.sort(probs, descending=True)
            
            # Calculate cumulative sum (e.g., [0.5, 0.8, 0.9, ...]).
            cumsum = torch.cumsum(sorted_probs, dim=0)
            
            # Find the index where the cumulative sum reaches 'top_p' (e.g., 0.8).
            # This determines the "nucleus" of tokens we want to sample from.
            cutoff_idx = torch.searchsorted(cumsum, top_p)
             
            # Slice to keep only the top tokens (the "nucleus").
            # trimmed_probs: The actual probability scores (e.g., [0.5, 0.25, 0.05])
            # trimmed_idxs: The REAL Token IDs owning those scores (e.g., [450, 12, 99])
            trimmed_probs = sorted_probs[: cutoff_idx + 1]
            trimmed_idxs = sorted_idxs[: cutoff_idx + 1]
            
            # Renormalize the trimmed probabilities so they sum to 1.0 again.
            trimmed_probs /= trimmed_probs.sum()

            # 2e. SAMPLING AND UPDATE
            # torch.multinomial returns a RELATIVE index (0, 1, 2...) pointing 
            # to a position inside 'trimmed_probs'.
            relative_sample_idx = torch.multinomial(trimmed_probs, 1)
            # Extract the raw Python integer from the tensor using .item()
            # We use this relative index to look up the REAL Token ID in 'trimmed_idxs'.
            index_int = relative_sample_idx.item() 
            next_token = trimmed_idxs[index_int]
            
            # Check for the stopping condition (End of Text token).
            if next_token.item() == end_id:
                break
            
            # Append the new token to our growing list of inputs.
            input_ids.append(next_token.item())

    # 3. DECODING
    # Convert the list of token IDs back into a human-readable string.
    return tokenizer.decode(input_ids)
