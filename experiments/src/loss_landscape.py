"""Loss landscape analysis: Hessian spectrum for a priori Lipschitz computation.

For a quadratic L(θ) = 0.5 θᵀAθ, the Hessian IS A — eigenvalues are exact.
For neural nets, we use stochastic Lanczos quadrature (top-k eigenvalues).

The Hessian spectrum feeds into lipschitz_apriori.py to compute L_pred
BEFORE running any rewritten trajectory.
"""
import torch
import torch.nn as nn
from typing import Tuple, Callable, Optional


def make_quadratic_hessian(n: int, condition_number: float,
                           seed: int = 42) -> torch.Tensor:
    """Generate a random PSD matrix A with prescribed condition number.
    Returns A such that L(θ) = 0.5 θᵀAθ has eigenvalues in [1, κ].
    """
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    eigs = torch.linspace(1.0, condition_number, n)
    A = Q @ torch.diag(eigs) @ Q.T
    return A


def make_quadratic_loss(A: torch.Tensor) -> Callable:
    """L(θ) = 0.5 θᵀAθ, ∇L = Aθ"""
    def loss_fn(theta):
        return (0.5 * theta @ A @ theta).item(), A @ theta
    return loss_fn


def hessian_eigenvalues_exact(A: torch.Tensor) -> torch.Tensor:
    """Exact eigenvalues of symmetric A. For quadratic losses."""
    return torch.linalg.eigvalsh(A)


def hessian_eigenvalues_lanczos(
    loss_fn_module: nn.Module,
    data_batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    k: int = 20,
    num_iterations: int = 100,
    seed: int = 42,
) -> torch.Tensor:
    """Top-k Hessian eigenvalues via stochastic Lanczos quadrature.

    Uses Hessian-vector products (no explicit Hessian storage).
    Suitable for neural networks up to ~10M parameters.

    Args:
        loss_fn_module: nn.Module with parameters
        data_batch: (inputs, targets) tuple
        criterion: loss function (e.g. nn.CrossEntropyLoss())
        k: number of eigenvalues to estimate
        num_iterations: Lanczos iterations
    """
    torch.manual_seed(seed)
    params = [p for p in loss_fn_module.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    def _flatten(tensors):
        return torch.cat([t.reshape(-1) for t in tensors])

    def _unflatten(flat, shapes):
        out, idx = [], 0
        for s in shapes:
            n = 1
            for d in s:
                n *= d
            out.append(flat[idx:idx+n].reshape(s))
            idx += n
        return out

    def hvp(v_flat):
        """Hessian-vector product via double backprop."""
        shapes = [p.shape for p in params]
        v_parts = _unflatten(v_flat, shapes)

        inputs, targets = data_batch
        loss_fn_module.zero_grad()
        out = loss_fn_module(inputs)
        loss = criterion(out, targets)
        grads = torch.autograd.grad(loss, params, create_graph=True)

        # grad · v
        gv = sum((g * v).sum() for g, v in zip(grads, v_parts))
        hvp_parts = torch.autograd.grad(gv, params)
        return _flatten([h.detach() for h in hvp_parts])

    # Lanczos tridiagonalization
    alphas = torch.zeros(num_iterations)
    betas = torch.zeros(num_iterations)

    q = torch.randn(n_params)
    q = q / q.norm()

    q_prev = torch.zeros_like(q)
    for j in range(min(num_iterations, k * 3)):
        w = hvp(q)
        alphas[j] = w.dot(q)
        w = w - alphas[j] * q - betas[max(0, j-1)] * q_prev
        betas[j] = w.norm()
        if betas[j] < 1e-10:
            break
        q_prev = q
        q = w / betas[j]

    # extract eigenvalues from tridiagonal matrix
    used = min(j + 1, num_iterations)
    T_mat = torch.zeros(used, used)
    for i in range(used):
        T_mat[i, i] = alphas[i]
        if i > 0:
            T_mat[i, i-1] = betas[i-1]
            T_mat[i-1, i] = betas[i-1]

    eigs = torch.linalg.eigvalsh(T_mat)
    return eigs[-k:]  # top-k