"""
Optimizer Module.

Implements specialized optimizers for efficient training of deep networks.
"""

import torch
import torch.optim as optim

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz Iteration for Matrix Orthogonalization.
    
    Computes the zeroth power of matrix G (approximating UV^T where G = USV^T).
    Used in Muon optimization to maintain orthogonality of updates.
    
    Args:
        G (torch.Tensor): Input matrix (2D).
        steps (int): Number of iterations.
        eps (float): Epsilon for numerical stability.
        
    Returns:
        torch.Tensor: Orthogonalized matrix.
    """
    assert len(G.shape) == 2, "Newton-Schulz requires 2D matrix."
    
    # Coefficients for 5th-order approximation
    a, b, c = (3.4445, -4.7750,  2.0315)
    
    # Use bfloat16 for speed if available, else float32
    X = G.bfloat16() if G.dtype != torch.float32 else G
    
    # Pre-conditioning: Normalize by spectral norm estimate
    X /= (X.norm() + eps)
    
    if G.size(0) > G.size(1):
        X = X.T

    # Iterative Update
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
        
    if G.size(0) > G.size(1):
        X = X.T

    return X.to(G.dtype)

class Muon(optim.Optimizer):
    """
    Muon (MomentUm Orthogonalized) Optimizer.
    
    A specialized optimizer for massive parameter updates (like embeddings or 
    transformers) that orthgonalizes the update steps using Newton-Schulz iteration,
    allowing for efficient training of deep backbones.
    
    Attributes:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate.
        momentum (float): Momentum factor.
        nesterov (bool): Enables Nesterov momentum.
        ns_steps (int): Newton-Schulz iteration steps.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                
                # Handle >2D tensors (e.g., Conv2d weights) by flattening
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                
                state = self.state[p]

                # State initialization
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                
                # Momentum Update
                buf.mul_(momentum).add_(g)
                
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                # Orthogonalization (Muon Core)
                if update.ndim == 2 and update.size(0) > 1 and update.size(1) > 1:
                     ortho_update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                else:
                     # Fallback for 1D vectors (biases, normalization params)
                     ortho_update = update

                # Apply Update
                if p.grad.ndim > 2:
                    ortho_update = ortho_update.view_as(p)

                p.data.add_(ortho_update, alpha=-lr)
                
        return loss

