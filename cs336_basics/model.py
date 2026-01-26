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