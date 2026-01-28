import torch
import math
from einops import einsum, rearrange, reduce



class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()

        mean = 0
        std = math.sqrt(2 / (out_features + in_features))
        lower = -3 * std
        upper = 3 * std
        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, mean=mean, std=std, a=lower, b=upper)

        self.weight = torch.nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation using weight matrix. 
        """
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
    

class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int, 
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        """
        Args:
            num_embeddings: int - Size of the vocabulary
            embedding_dim: int - Dimension of the embedding vectors, i.e., dmodel
        """
        super().__init__()

        mean = 0
        std = 1
        lower = -3
        upper = 3

        if kwargs.get("embedding_std", None) is not None:
            std = kwargs.get("embedding_std")

        w = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, mean=mean, std=std, a=lower, b=upper)

        self.weight = torch.nn.Parameter(w)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embedding vectors for given token IDs.
        
        Args:
            token_ids: Tensor of token indices of shape (...,)
            
        Returns:
            Embedding vectors of shape (..., embedding_dim)
        """

        # PyTorch's "Advanced Indexing" https://gemini.google.com/app/c0055d454423ef32
        return self.weight[token_ids]
    

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(reduce(x**2, "... d -> ... 1", "mean") + self.eps)
        result = x * self.weight / rms

        return result.to(in_dtype)
    

def silu_activation(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)  # element-wise


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1 = self.w1(x)
        silu = silu_activation(a1)
        return self.w2(silu * self.w3(x))


# class SiLU(torch.nn.Module):
#     def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
#         super().__init__()

#         self.w1 = Linear(d_model, d_ff, device, dtype)
#         self.w2 = Linear(d_ff, d_model, device, dtype)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         a1 = self.w1(x)
#         silu = silu_activation(a1)
#         return self.w2(silu)
    

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)
        freqs = torch.arange(0, d_k, 2, device=device) / d_k
        inv_freq = 1.0 / (theta**freqs)
        angles = positions * inv_freq

        self.register_buffer("cos", angles.cos().to(dtype), persistent=False)
        self.register_buffer("sin", angles.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos

        x_rot = rearrange([x_rot_even, x_rot_odd], "two ... -> ... two")
        x_out = rearrange(x_rot, "... d1 d2 -> ... (d1 d2)")

        return x_out
    

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute softmax along specified dimension.

    Args:
        x: Input tensor.
        dim: Dimension along which to compute softmax.

    Returns:
        Tensor with softmax applied along specified dimension.
    """
    # Without keepdim=True, the code will fail with a RuntimeError (shape mismatch) 
    # whenever the reduction dimension (dim) is not the last dimension, due to how PyTorch handles broadcasting.
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    d_k = Q.shape[-1]

    attention_scores = einsum(Q, K, "... seq_q d, ... seq_k d -> ... seq_q seq_k")
    attention_scores = attention_scores / math.sqrt(d_k)
    # Masking converts Control Flow (jumping code paths) into Data Flow (math operations), 
    # no matter it's eager vs graph mode.
    attention_scores = torch.where(mask, attention_scores, float("-inf"))

    # dim=-1 : 
    # For one specific query (one row), we want to calculate a probability distribution over all available keys
    # "For this specific Query, how much do I care about Key A vs Key B vs Key C?"
    attention_weights = softmax(attention_scores, dim=-1)
    output = einsum(attention_weights, V, "... seq_q seq_k, ... seq_k d -> ... seq_q d")

    return output


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, **kwargs):
        super().__init__()

        self.wqkv = Linear(d_model, 3 * d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        # qkv shape: (Batch , Seq_Len, 3 * d_model)
        qkv = self.wqkv(x)

        # Split into separate q, k, v tensors. 
        # dim=2: slice along the last dimension (the feature dimension)
        # self.d_model: the size of each chunk
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape from (batch, seq_len, dim) to (batch, heads, seq_len, head_dim)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = rope(q, token_positions)
            k = rope(k, token_positions)

        # Create causal mask for self-attention.  
        # torch.triu creates Triangular Upper
        mask = ~torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)

        y = scaled_dot_product_attention(q, k, v, mask)
        y = rearrange(y, "b h s d -> b s (h d)")
        return self.output_proj(y)


class Block(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        """
        Transformer Block with Pre-Norm architecture.
        
        Args:
            d_model: Hidden dimension size.
            num_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension size.
            rope: Rotary Positional Embedding module.
            device: Torch device.
            dtype: Torch data type.
            **kwargs: 
                - ffn_type (str): 'swiglu' (default) or 'silu'.
                - dropout (float): Dropout probability for residual connections (default 0.0).
                - remaining kwargs are passed to CausalMultiHeadSelfAttention.
        """
        super().__init__()

        self.rope = rope

        # 1. Clean extraction of Block-specific args.
        # using .pop() ensures these don't accidentally get passed to self.attn below,
        # which would cause a TypeError if the attention class is strict.
        ffn_type = kwargs.pop("ffn_type", "swiglu")
        dropout_p = kwargs.pop("dropout", 0.0)

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        
        # 2. Pass only the remaining relevant kwargs to Attention
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, device, dtype, **kwargs)

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        # 3. Add Residual Dropout (Crucial for training deep networks)
        # Use Identity if dropout is 0 to save compute/memory overhead
        self.resid_dropout = torch.nn.Dropout(dropout_p) if dropout_p > 0 else torch.nn.Identity()

        if ffn_type == "silu":
            self.ffn = SiLU(d_model, d_ff, device, dtype)
        elif ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        else:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard Pre-Norm Residual connection: x = x + Drop(Sublayer(Norm(x)))
        
        # Attention Block
        x = x + self.resid_dropout(self.attn(self.ln1(x), self.rope))
        
        # Feed-Forward Block
        x = x + self.resid_dropout(self.ffn(self.ln2(x)))
        
        return x
    

class Transformer(torch.nn.Module):
    """
    A Decoder-only Transformer model (similar to LLaMA or GPT).

    Architecture:
    - Input: Token indices [batch, seq_len]
    - Output: Logits [batch, seq_len, vocab_size] (No Softmax applied)
    - Components: RMSNorm, SwiGLU (implied in blocks), and RoPE.
    """
    def __init__(
        self,
        d_model: int,         # Embedding dimension (e.g., 4096)
        num_heads: int,       # Number of attention heads
        d_ff: int,            # Hidden dimension of the Feed-Forward Network
        vocab_size: int,      # Total size of vocabulary (e.g., 50257)
        context_length: int,  # Maximum sequence length model can handle
        num_layers: int,      # Number of Transformer Blocks to stack
        rope_theta: float = 10000.0, # Base frequency for Rotary Embeddings
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()

        self.context_length = context_length

        # 1. Token Embeddings
        # Converts integer token indices into dense vectors.
        # Note: We do NOT add absolute positional embeddings here (like in GPT-2).
        # Positional info is injected via RoPE inside the attention layers.
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype, **kwargs)

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        d_head = d_model // num_heads

        # 2. Rotary Positional Embeddings (RoPE)
        # Q: Does this include trainable parameters?
        # A: NO. RoPE is completely static. It pre-computes Sine and Cosine tables
        #    (buffers) based on 'rope_theta'. These do not change during training.
        rope = RotaryPositionalEmbedding(rope_theta, d_head, context_length, device=device, dtype=dtype)

        # 3. Transformer Blocks
        # Q: Why is the same 'rope' object passed to every block?
        # A: Since RoPE math depends only on position (index 1, 2, 3...) and is identical
        #    for every layer, we share the single 'rope' instance to save memory and compute.
        #    It acts as a shared lookup table for frequency rotations.
        self.layers = torch.nn.ModuleList(
            [Block(d_model, num_heads, d_ff, rope, device, dtype, **kwargs) for _ in range(num_layers)]
        )

        # 4. Final Normalization (RMSNorm)
        # Normalizes the features before the final projection.
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        # 5. Language Model Head
        # Projects internal state [d_model] back to vocabulary size [vocab_size].
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

        # Weight Tying (Optional) 
        # Links the output head weights to the input embedding weights.
        # if kwargs.get("weight_tying", False):
        #     self.lm_head.weight = self.token_embeddings.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input 'x' shape: [batch_size, seq_len]
        batch_size, seq_len = x.shape

        # Validation: Ensure input doesn't exceed the pre-computed RoPE tables.
        if seq_len > self.context_length:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds model context length ({self.context_length})")

        # Step A: Embed Tokens
        # Shape: [batch, seq_len] -> [batch, seq_len, d_model]
        x = self.token_embeddings(x)

        # Step B: Transformer Layers
        # Shape remains [batch, seq_len, d_model] throughout.
        # The shared 'rope' object is used inside every layer's Attention mechanism.
        for layer in self.layers:
            x = layer(x)

        # Step C: Final Norm
        x = self.ln_final(x)

        # Step D: Output Projection
        # Shape: [batch, seq_len, d_model] -> [batch, seq_len, vocab_size]
        # NOTE: This returns raw LOGITS.
        # - The Softmax layer (shown in diagrams) is technically missing here.
        # - This is intentional. We apply Softmax externally (e.g., inside CrossEntropyLoss)
        #   for numerical stability during training.
        x = self.lm_head(x)

        return x