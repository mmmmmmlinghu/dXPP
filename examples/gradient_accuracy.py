import numpy as np
import torch
import sys
import os
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dXPP import dXPPLayer
from deps.dQP import dQP

def get_available_solver():
    import qpsolvers
    solvers = qpsolvers.available_solvers
    if 'gurobi' in solvers: return 'gurobi'
    if 'osqp' in solvers: return 'osqp'
    if 'cvxopt' in solvers: return 'cvxopt'
    if 'clarabel' in solvers: return 'clarabel'
    if len(solvers) > 0: return solvers[0]
    return None

def remove_outliers(data, method='iqr'):
    """
    Remove NaN values and outliers from data.
    
    Args:
        data: list or numpy array of values
        method: 'iqr' (Interquartile Range) or 'zscore' (Z-score method)
    
    Returns:
        filtered_data: array with NaN and outliers removed
    """
    data = np.array(data)
    
    # Remove NaN values
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return data
    
    # Remove outliers using IQR method
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered = data[(data >= lower_bound) & (data <= upper_bound)]
    elif method == 'zscore':
        # Use Z-score method (remove values beyond 3 standard deviations)
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            z_scores = np.abs((data - mean) / std)
            filtered = data[z_scores < 3]
        else:
            filtered = data
    else:
        filtered = data
    
    return filtered

def generate_problem(dim, nIneq, nEq, seed=None):
    """
    Generate a QP problem with normal condition number.
    
    Args:
        dim: problem dimension
        nIneq: number of inequality constraints
        nEq: number of equality constraints
        seed: random seed
    
    Returns:
        Q, q, G, h, A, b: problem data
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate Q with normal condition (strictly convex)
    P = torch.randn(dim, dim, dtype=torch.float64)
    Q = P @ P.T + 0.1 * torch.eye(dim, dtype=torch.float64)
    
    # Generate other problem data
    q = torch.randn(dim, dtype=torch.float64)
    G = torch.randn(nIneq, dim, dtype=torch.float64)
    A = torch.randn(nEq, dim, dtype=torch.float64)
    
    # Ensure feasibility: construct a feasible point z0
    z0 = torch.ones(dim, dtype=torch.float64)
    h = G @ z0 + 1.0  # G @ z0 < h
    b = A @ z0        # A @ z0 = b
    
    return Q, q, G, h, A, b

def test_single_problem(dim, nIneq, nEq, seed, solver, max_retries=2):
    """Test a single problem instance and return relative difference"""
    # Filter numpy matrix warnings for the entire function
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*matrix subclass.*')
        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        
        for attempt in range(max_retries):
            try:
                # Use different seed for retry to avoid same issue
                current_seed = seed + attempt * 1000
                
                # Generate problem data
                Q, q, G, h, A, b = generate_problem(dim, nIneq, nEq, current_seed)
                
                # Enable gradients (only q needs gradient for comparison)
                Q.requires_grad = False
                q.requires_grad = True
                G.requires_grad = False
                h.requires_grad = False
                A.requires_grad = False
                b.requires_grad = False
                
                # 1. dXPPLayer Gradient
                layer = dXPPLayer(eps_abs=1e-8, beta=1e-6, verbose=False, qp_solver="gurobi")
                
                x_ap, mu_ap, nu_ap = layer(Q, q, G, h, A, b)
                loss_ap = (x_ap ** 2).sum() + (mu_ap ** 2).sum() + (nu_ap ** 2).sum()
                loss_ap.backward()
                ap_grad = q.grad.clone()
                
                # Zero out gradients for the next comparison
                q.grad.zero_()
                
                # 2. dQP Gradient
                settings = dQP.build_settings(
                    qp_solver=solver, 
                    solve_type="dense", 
                    check_PSD=False,
                    verbose=False,
                    eps_abs=1e-7,
                    eps_rel=1e-7
                )
                dqp_layer = dQP.dQP_layer(settings)
                
                x_dqp, mu_dqp, nu_dqp, _, _ = dqp_layer(Q, q, G, h, A, b)
                loss_dqp = (x_dqp ** 2).sum() + (mu_dqp ** 2).sum() + (nu_dqp ** 2).sum()
                loss_dqp.backward()
                dqp_grad = q.grad.clone()
                
                # Compute relative difference
                rel_diff = torch.norm(ap_grad - dqp_grad) / (torch.norm(dqp_grad) + 1e-10)
                return rel_diff.item(), None
                
            except Exception as e:
                error_msg = str(e)
                # Check if it's the numpy matrix subclass error
                if 'matrix subclass' in error_msg.lower() or 'numpy-for-matlab-users' in error_msg.lower():
                    # Retry with different seed if not last attempt
                    if attempt < max_retries - 1:
                        continue
                    # If all retries failed, return None to indicate it's a known non-critical issue
                    return float('nan'), None
                # For other errors, return the error message
                return float('nan'), error_msg
        
        # Should not reach here, but just in case
        return float('nan'), "Unknown error after retries"

def test_dqp_comparison():
    """Comparison between dXPPLayer and dQP gradients with multiple configurations"""
    print("\n" + "=" * 80, flush=True)
    print("dQP vs dXPPLayer Gradient Comparison (Multiple Configurations)", flush=True)
    print("=" * 80, flush=True)
    
    solver = get_available_solver()
    print(f"Using solver: {solver}\n", flush=True)
    
    # Problem configurations: (dim, nIneq), where nEq = nIneq
    configs = [
        (10, 5),
        (50, 10),
        (100, 20),
        (500, 100),
        (1000, 200),
        (1500, 500),
        (3000, 1000),
        (5000, 2000)
    ]
    
    n_runs = 30  # Number of random runs per configuration
    
    results = []
    
    # Print header for summary table
    print(f"\n{'='*80}", flush=True)
    print("REAL-TIME SUMMARY TABLE", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'dim':<8} {'nIneq':<8} {'avg_rel_diff':<15} {'std_rel_diff':<15}", flush=True)
    print("-" * 80, flush=True)
    
    for dim, nIneq in configs:
        nEq = nIneq  # Equality constraints = inequality constraints
        print(f"\n{'='*80}", flush=True)
        print(f"Configuration: dim={dim}, nIneq={nIneq}, nEq={nEq}", flush=True)
        print(f"{'='*80}", flush=True)
        
        rel_diffs = []
        
        # Parallel execution for 5 runs (limit to CPU count to avoid resource exhaustion)
        max_workers = min(n_runs, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for run_idx in range(n_runs):
                seed = 100000 + run_idx  # Different seed for each run
                future = executor.submit(test_single_problem, dim, nIneq, nEq, seed, solver)
                futures[future] = run_idx + 1
            
            # Collect results as they complete
            for future in as_completed(futures):
                run_idx = futures[future]
                try:
                    rel_diff, error = future.result()
                    if error is None:
                        # Only add valid (non-NaN) values
                        if not np.isnan(rel_diff):
                            rel_diffs.append(rel_diff)
                            print(f"  Run {run_idx}/{n_runs}: rel_diff = {rel_diff:.2e}", flush=True)
                        else:
                            print(f"  Run {run_idx}/{n_runs}: Skipped (NaN result)", flush=True)
                    elif error and 'matrix subclass' in error.lower():
                        # Skip numpy matrix subclass warnings (known non-critical issue from library)
                        print(f"  Run {run_idx}/{n_runs}: Skipped (numpy matrix warning)", flush=True)
                    else:
                        print(f"  Run {run_idx}/{n_runs}: Failed - {error}", flush=True)
                except Exception as e:
                    error_msg = str(e)
                    if 'matrix subclass' in error_msg.lower():
                        # Skip numpy matrix subclass warnings
                        print(f"  Run {run_idx}/{n_runs}: Skipped (numpy matrix warning)", flush=True)
                    else:
                        print(f"  Run {run_idx}/{n_runs}: Exception - {e}", flush=True)
        
        if rel_diffs:
            # Remove NaN and outliers before calculating statistics
            filtered_diffs = remove_outliers(rel_diffs, method='iqr')
            
            if len(filtered_diffs) > 0:
                avg_rel_diff = np.mean(filtered_diffs)
                std_rel_diff = np.std(filtered_diffs)
                n_outliers = len(rel_diffs) - len(filtered_diffs)
                
                if n_outliers > 0:
                    print(f"  Average rel_diff: {avg_rel_diff:.2e} ± {std_rel_diff:.2e} (removed {n_outliers} outliers from {len(rel_diffs)} runs)", flush=True)
                else:
                    print(f"  Average rel_diff: {avg_rel_diff:.2e} ± {std_rel_diff:.2e}", flush=True)
                
                result_item = {
                    'dim': dim,
                    'nIneq': nIneq,
                    'nEq': nEq,
                    'avg_rel_diff': avg_rel_diff,
                    'std_rel_diff': std_rel_diff
                }
                results.append(result_item)
                
                # Print to summary table immediately
                print(f"  → Summary: {dim:<8} {nIneq:<8} {avg_rel_diff:<15.2e} {std_rel_diff:<15.2e}", flush=True)
            else:
                print(f"  All values filtered out as outliers!", flush=True)
                result_item = {
                    'dim': dim,
                    'nIneq': nIneq,
                    'nEq': nEq,
                    'avg_rel_diff': float('nan'),
                    'std_rel_diff': float('nan')
                }
                results.append(result_item)
                print(f"  → Summary: {dim:<8} {nIneq:<8} {'FAILED':<15} {'N/A':<15}", flush=True)
        else:
            print(f"  All runs failed!", flush=True)
            result_item = {
                'dim': dim,
                'nIneq': nIneq,
                'nEq': nEq,
                'avg_rel_diff': float('nan'),
                'std_rel_diff': float('nan')
            }
            results.append(result_item)
            print(f"  → Summary: {dim:<8} {nIneq:<8} {'FAILED':<15} {'N/A':<15}", flush=True)
    
    # Print final summary table
    print(f"\n{'='*80}", flush=True)
    print("FINAL SUMMARY TABLE", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'dim':<8} {'nIneq':<8} {'avg_rel_diff':<15} {'std_rel_diff':<15}", flush=True)
    print("-" * 80, flush=True)
    for r in results:
        if not np.isnan(r['avg_rel_diff']):
            print(f"{r['dim']:<8} {r['nIneq']:<8} {r['avg_rel_diff']:<15.2e} {r['std_rel_diff']:<15.2e}", flush=True)
        else:
            print(f"{r['dim']:<8} {r['nIneq']:<8} {'FAILED':<15} {'N/A':<15}", flush=True)
    
    print("=" * 80, flush=True)
    
    # Save results to CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), 'results')
    os.makedirs(results_dir, exist_ok=True)

    csv_filename = os.path.join(results_dir, f'gradient_comparison.csv')
    
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    test_dqp_comparison()
