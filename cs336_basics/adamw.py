import torch
import math

# AdamW: Apply adaptive update => Speed up in flat directions, slow down in steep/volatile directions.
# Update = Signal / Noise: Divide the smoothed gradient (m) by its volatility (sqrt(v)) to adaptively scale the step.
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        **kwargs,
    ):
        # Pack defaults into a dictionary to pass to the parent class
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        
        # Super init handles the creation of 'self.param_groups' 
        # This organizes parameters into groups (useful for different LRs per layer)
        super().__init__(params, defaults)

        # Store these mainly for reference/external access
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    # @torch.no_grad() is critical: Optimization steps should not be tracked 
    # by the autograd engine, as they don't need gradients themselves.
    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        # Iterate over all parameter groups (handles the "differential learning rates" logic)
        for group in self.param_groups:
            # Unpack hyperparameters for this specific group
            b1, b2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                # Skip parameters that didn't receive a gradient (e.g., frozen layers)
                if p.grad is None:
                    continue

                # 'self.state' is a dictionary where the optimizer remembers 
                # previous values (momentum/variance) for every specific parameter.
                state = self.state[p]

                # Lazy Initialization:
                # If this is the first step, create zero-tensors for momentum (m) and variance (v).
                # 't' tracks the number of steps taken (used for bias correction).
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 1)

                grad = p.grad

                # --- 1. Update Momentum (First Moment) ---
                # Blending previous momentum with current gradient
                # state["m"] = beta1 * m + (1 - beta1) * gradient
                state["m"] = b1 * m + (1 - b1) * grad

                # --- 2. Update Variance (Second Moment) ---
                # Blending previous variance with current squared gradient
                # state["v"] = beta2 * v + (1 - beta2) * (gradient^2)
                state["v"] = b2 * v + (1 - b2) * grad.pow(2)

                # --- 3. Compute Bias-Corrected Step Size ---
                # Because m and v start at 0, they are biased towards 0 early on.
                # This formula boosts the learning rate in early steps to correct that.
                step_size = lr * (math.sqrt(1 - b2**t) / (1 - b1**t))

                # --- 4. Apply Adaptive Weight Update ---
                # p = p - step_size * (m / (sqrt(v) + eps))
                # addcdiv_ performs: tensor + value * (tensor1 / tensor2)
                p.data.addcdiv_(state["m"], torch.sqrt(state["v"]) + eps, value=-step_size)

                # --- 5. Apply Decoupled Weight Decay (The "W" in AdamW) ---
                # Standard Adam adds decay to the gradient. AdamW applies it directly 
                # to the weights separately. This results in better generalization.
                if wd != 0:
                    p.data.add_(p.data, alpha=-lr * wd)

                # Increment step counter
                state["t"] = t + 1

        return loss
