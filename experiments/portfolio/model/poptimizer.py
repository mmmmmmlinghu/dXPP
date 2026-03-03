import torch
import torch.nn as nn
import sys
from pathlib import Path

# Mapping model names to their classes
def _get_model_class(model_name):
    if model_name == 'CNNLSTM':
        from model.CNNLSTM import Model
        return Model
    elif model_name == 'Linear':
        from model.Linear import Model
        return Model
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

from model.covupdater import CovUpdater, update_cov, repeat_cov
import sys
from pathlib import Path

# Add the project root directory to sys.path to ensure src and layer can be found
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import dXPP

# Import baseline methods
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer as CvxpyLayer_fun
from deps.qpth_dual.qp import QPFunction
from deps.scqpth_dual.control import scqpth_control
from deps.scqpth_dual.scqpth import SCQPTHNet
from deps.dQP import dQP


def _getattr(configs, name: str, default):
    return getattr(configs, name, default)

def _make_qp_layer(configs):
    opt = str(configs.optimizer)
    eps = float(_getattr(configs, "eps", 1e-6))
    
    if opt == "dQP":
        settings = dQP.build_settings(
            solve_type="dense",
            verbose=False,
            time=False,
            normalize_constraints=False,
            warm_start_from_previous=False,
            eps_active=eps,
            eps_abs=eps,
            eps_rel=0.0,
            qp_solver="gurobi",
        )
        return dQP.dQP_layer(settings), "dQP"
    
    if opt == "dXPP":
        layer = dXPP.dXPPLayer(
            beta=float(_getattr(configs, "lam", 1e-4)),
            penalty_coeff=float(_getattr(configs, "penalty_coeff", 10.0)),
            eps_abs=eps,
            eps_rel=0.0,
            warm_start=bool(_getattr(configs, "warm_start", True)),
            verbose=bool(_getattr(configs, "verbose_qp", False)),
            solve_type="dense",
            qp_solver="gurobi",
        )
        return layer, "dXPP"
    
    if opt == "OptNet":
        layer = QPFunction(verbose=False, eps=eps)
        return layer, "OptNet"
    
    if opt == "Cvxpy":
        # Cvxpylayer needs to be created per-problem (dim, nIneq, nEq vary)
        # We return a placeholder; actual layer is created in _solve_qp
        return None, "Cvxpy"
    
    if opt == "SCQPTH":
        # SCQPTH also needs per-problem setup
        return None, "SCQPTH"
    
    raise ValueError(f"Unsupported optimizer '{opt}', expected one of: dQP, dXPP, OptNet, Cvxpy, SCQPTH")


def _build_multiperiod_qp(mu: torch.Tensor,
                          cov: torch.Tensor,
                          z0: torch.Tensor,
                          tv_threshold: float,
                          delta: float):
    # z = [w_0..w_{H-1}, u_0..u_{H-1}] with u_t linearizing ||w_t-w_{t-1}||_1, w_-1 := z0
    if mu.ndim == 1:
        mu = mu.unsqueeze(0)
    if cov.ndim == 2:
        cov = cov.unsqueeze(0)
    H, N = mu.shape[:2]

    dtype, device = torch.float64, mu.device
    mu, cov, z0 = (t.to(device=device, dtype=dtype) for t in (mu, cov, z0))

    dim_w = H * N
    dim_u = H * N
    dim = dim_w + dim_u

    # Objective: sum_t (delta/2 w_t^T Omega_t w_t - r_t^T w_t) => Q_w = delta*Omega_t, q_w = -r_t
    Q = torch.zeros((dim, dim), dtype=dtype, device=device)
    Q_w = torch.block_diag(*[float(delta) * cov[t] for t in range(H)]) if H > 1 else float(delta) * cov[0]
    Q[:dim_w, :dim_w] = Q_w
    # Add small regularization to u variables to make Q positive definite (required by OptNet PDIPM)
    reg = 1e-8
    Q[dim_w:, dim_w:] = reg * torch.eye(dim_u, dtype=dtype, device=device)
    q = torch.zeros((dim,), dtype=dtype, device=device)
    q[:dim_w] = -mu.reshape(-1)

    # Equality: 1^T w_t = 1
    A = torch.zeros((H, dim), dtype=dtype, device=device)
    A[:, :dim_w] = torch.kron(torch.eye(H, dtype=dtype, device=device), torch.ones((1, N), dtype=dtype, device=device))
    b = torch.ones((H,), dtype=dtype, device=device)

    # Inequalities: w_t >= 0, u_t >= |w_t-w_{t-1}|, u_t >= 0, sum(u_t) <= tv_threshold
    nIneq = H * (4 * N + 1)
    G = torch.zeros((nIneq, dim), dtype=dtype, device=device)
    h = torch.zeros((nIneq,), dtype=dtype, device=device)
    I = torch.eye(N, dtype=dtype, device=device)

    row = 0
    for t in range(H):
        w = slice(t * N, (t + 1) * N)
        u = slice(dim_w + t * N, dim_w + (t + 1) * N)
        w_prev = slice((t - 1) * N, t * N)

        G[row:row + N, w] = -I  # w_t >= 0
        row += N

        G[row:row + N, w] = I
        G[row:row + N, u] = -I
        if t == 0:
            h[row:row + N] = z0
        else:
            G[row:row + N, w_prev] = -I
        row += N

        G[row:row + N, w] = -I
        G[row:row + N, u] = -I
        if t == 0:
            h[row:row + N] = -z0
        else:
            G[row:row + N, w_prev] = I
        row += N

        G[row:row + N, u] = -I  # u_t >= 0
        row += N

        G[row, u] = 1.0  # sum(u_t) <= lambda
        h[row] = float(tv_threshold)
        row += 1

    return Q, q, G, h, A, b, H, N, dim_w


class POptimizer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        ModelClass = _get_model_class(_getattr(configs, "model", "Linear"))
        self.model = ModelClass(configs)
        self.covupdater = CovUpdater(configs)
        self.algo, self.algo_type = _make_qp_layer(configs)
        self.eps = float(_getattr(configs, "eps", 1e-6))
        
        self.solve_type = "dense"
        
        # Cache for Cvxpylayer (created per problem size)
        self._cvxpy_cache = {}

    def _solve_qp(self, Q, q, G, h, A, b):
        dim = Q.shape[0]
        nIneq = G.shape[0]
        nEq = A.shape[0] if A is not None else 0
        
        # Ensure dense tensors for all solvers
        if Q.is_sparse:
            Q = Q.to_dense()
        if G.is_sparse:
            G = G.to_dense()
        if A is not None and A.is_sparse:
            A = A.to_dense()
        
        if self.algo_type == "dQP":
            out = self.algo(Q, q, G, h, A, b)
            x_star = out[0] if isinstance(out, (tuple, list)) else out
            if x_star.ndim > 1:
                x_star = x_star.squeeze(0)
            return x_star
        
        elif self.algo_type == "dXPP":
            out = self.algo(Q, q, G, h, A, b)
            x_star = out[0] if isinstance(out, (tuple, list)) else out
            if x_star.ndim > 1:
                x_star = x_star.squeeze(0)
            return x_star
        
        elif self.algo_type == "OptNet":
            # OptNet expects batched input: [batch, ...]
            Q_b = Q.unsqueeze(0)
            q_b = q.unsqueeze(0)
            G_b = G.unsqueeze(0)
            h_b = h.unsqueeze(0)
            A_b = A.unsqueeze(0) if A is not None else torch.zeros(1, 1, dim, dtype=Q.dtype, device=Q.device)
            b_b = b.unsqueeze(0) if b is not None else torch.zeros(1, 1, dtype=Q.dtype, device=Q.device)
            
            x_star, _ = self.algo(Q_b, q_b, G_b, h_b, A_b, b_b)
            return x_star.squeeze(0)
        
        elif self.algo_type == "Cvxpy":
            # Create or retrieve cached Cvxpylayer for this problem size
            cache_key = (dim, nIneq, nEq)
            if cache_key not in self._cvxpy_cache:
                Q_sqrt_ = cp.Parameter((dim, dim))
                q_ = cp.Parameter(dim)
                G_ = cp.Parameter((nIneq, dim))
                h_ = cp.Parameter(nIneq)
                x_ = cp.Variable(dim)
                obj = cp.Minimize(0.5 * cp.sum_squares(Q_sqrt_ @ x_) + q_.T @ x_)
                
                if nEq > 0:
                    A_ = cp.Parameter((nEq, dim))
                    b_ = cp.Parameter(nEq)
                    cons = [A_ @ x_ == b_, G_ @ x_ <= h_]
                    prob = cp.Problem(obj, cons)
                    layer = CvxpyLayer_fun(prob, parameters=[Q_sqrt_, q_, G_, h_, A_, b_], variables=[x_])
                else:
                    cons = [G_ @ x_ <= h_]
                    prob = cp.Problem(obj, cons)
                    layer = CvxpyLayer_fun(prob, parameters=[Q_sqrt_, q_, G_, h_], variables=[x_])
                self._cvxpy_cache[cache_key] = (layer, nEq > 0)
            
            layer, has_eq = self._cvxpy_cache[cache_key]
            
            # Add small regularization for Cholesky stability (Q may be semi-definite)
            Q_reg = Q + 1e-6 * torch.eye(dim, dtype=Q.dtype, device=Q.device)
            # Cvxpylayer needs Q_sqrt (upper triangular Cholesky factor)
            Q_sqrt = torch.linalg.cholesky(Q_reg, upper=True)
            
            if has_eq:
                x_star, = layer(Q_sqrt, q, G, h, A, b)
            else:
                x_star, = layer(Q_sqrt, q, G, h)
            return x_star
        
        elif self.algo_type == "SCQPTH":
            # SCQPTH: convert equality to inequality (Ax=b -> Ax<=b, -Ax<=-b)
            control = scqpth_control(eps_abs=self.eps, eps_rel=0)
            
            Q_b = Q.unsqueeze(0)
            q_b = q.unsqueeze(0).unsqueeze(-1)
            
            if nEq > 0:
                G_aug = torch.cat([G, A, -A], dim=0).unsqueeze(0)
                h_aug = torch.cat([h, b, -b], dim=0).unsqueeze(0).unsqueeze(-1)
            else:
                G_aug = G.unsqueeze(0)
                h_aug = h.unsqueeze(0).unsqueeze(-1)
            
            nIneq_aug = G_aug.shape[1]
            lb = -1.0e20 * torch.ones((1, nIneq_aug, 1), dtype=Q.dtype, device=Q.device)
            
            x_star, _ = SCQPTHNet(control)(Q=Q_b, p=q_b, A=G_aug, lb=lb, ub=h_aug)
            return x_star.squeeze(0).squeeze(-1)

    def forward(self, x0, x, opt=True, z0=None):
        pred = self.model(x)  # [B, H_out, N]
        _, cov = self.covupdater(x0)  # cov: [B,N,N]
        if not opt:
            return pred, cov

        B, H_out, N = pred.shape
        H = int(_getattr(self.configs, "horizon", H_out))
        H = H_out if H_out != H else H

        tv = _getattr(self.configs, "tv_lambda", None)
        tv = _getattr(self.configs, "turnover", tv)
        tv_threshold = float(_getattr(self.configs, "r", 0.0) if tv is None else tv)
        delta = float(_getattr(self.configs, "delta", 1.0))

        z0 = (torch.full((N,), 1.0 / N, dtype=pred.dtype, device=pred.device)
              if z0 is None else z0.reshape(-1).to(pred.device))

        w_plans = []
        for b in range(B):
            mu_b = pred[b, :H, :]
            try:
                cov_b, _ = update_cov(cov[b], cov[b].unsqueeze(0), H, x[b].unsqueeze(0), mu_b.unsqueeze(0))
            except Exception:
                cov_b = repeat_cov(cov[b], cov[b].unsqueeze(0), H)[0]

            Q, q, G, h, A, b_eq, H_eff, N_eff, dim_w = _build_multiperiod_qp(
                mu=mu_b,
                cov=cov_b,
                z0=z0.detach(),
                tv_threshold=tv_threshold,
                delta=delta,
            )

            x_star = self._solve_qp(Q, q, G, h, A, b_eq)  # [dim]
            w_flat = x_star[:dim_w]
            
            # Since CNNLSTM has been modified to output full [H, N], directly reshape here
            w_plan = w_flat.reshape(H_eff, N_eff)  # [H, N]
            w_plans.append(w_plan)

        w_plans = torch.stack(w_plans, dim=0)  # [B, H, N]
        return w_plans

    def evaluate(self, x0, x, opt=True, z0=None):
        pred = self.model(x)
        _, cov = self.covupdater(x0)
        if not opt:
            return pred, cov
        return self.forward(x0, x, opt=True, z0=z0)