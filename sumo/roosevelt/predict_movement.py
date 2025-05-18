"""
Implement step 1: match route-movement flow using Sage traffic count
Solve for min ||AX - C||^2 with regularization
Solve by batch so it's more scalable
Visualize C_obs vs. C_est
"""
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import sparse

import cvxpy as cp
import numpy as np
from scipy import sparse

def solve_optimized(A_matrix, C_matrix, gamma=None, time_batch_size=100):
    """
    Smart solver that automatically chooses the best method:
    - No regularization: Uses default solver (OSQP/ECOS) per timestep
    - With regularization: Uses batched SCS with sparse matrices
    
    Parameters:
        gamma: None for no regularization, float for regularization strength
        time_batch_size: Number of timesteps to process together when regularized
    """
    
    if gamma is None:
        # Fast path - no regularization
        return _solve_batch(A_matrix, C_matrix)
    else:
        # Regularized path - use sparse batched solver
        return _solve_batched_regularized(
            A_matrix, C_matrix, gamma, time_batch_size
        )

def _solve_batch(A_matrix, C_matrix):
    '''
    Solve route-movement flow problem
    Step 1: 
    '''

    # === 5. Define and solve the optimization problem ===
    X = cp.Variable((len(routes), len(time_points)), nonneg=True)
    
    objective = cp.Minimize(cp.sum_squares(A_matrix @ X - C_matrix) )
    prob = cp.Problem(objective)
    prob.solve(verbose=True)

    print("\n=== Solver Status ===")
    print("Status:", prob.status)
    print("Optimal value:", prob.value)
    print("Solver:", prob.solver_stats.solver_name)
    print("Solve time:", prob.solver_stats.solve_time, "seconds")
    print("Number of iterations:", prob.solver_stats.num_iters)

    # Check matrix rank and conditioning
    print("\n=== Matrix Analysis ===")
    print("Rank of A:", np.linalg.matrix_rank(A_matrix))
    print("Shape of A:", A_matrix.shape)
    print("Condition number of A:", np.linalg.cond(A_matrix))

    if np.linalg.matrix_rank(A_matrix) < A_matrix.shape[1]:
        print("Warning: Problem is underdetermined - multiple solutions exist")
    else:
        print("Problem has a unique solution")

    # Post-processing
    X_val = np.array(X.value)
    X_val[np.abs(X_val) < 1e-5] = 0  # Set very small values to 0
    X_val = np.round(X_val)          # Round to nearest integer

    # === 6. Prepare results ===
    results_df = pd.DataFrame(
        X_val.T,  # Transpose to have time points as rows
        columns=routes,
        index=time_points
    ).reset_index().rename(columns={'index': 'Time'})

    # === 7. Save results ===
    results_df.to_csv("data/estimated_route_flows.csv", index=False)
    print("\nResults saved to 'data/estimated_route_flows.csv'")

    return X_val

def _create_diff_matrix(n_times):
    """Efficient sparse difference operator"""
    return sparse.diags([-1, 1], [0, 1], shape=(n_times-1, n_times))

def _solve_batched_regularized(A, C, gamma, batch_size):
    """Solve regularized problem in time batches while maintaining original matrix format"""
    n_routes = A.shape[1]
    n_times = C.shape[1]
    
    # Initialize full result matrix
    X_full = np.zeros((n_routes, n_times))
    
    # Precompute difference operator for regularization
    if batch_size > 1:
        D = _create_diff_matrix(batch_size)  # <-- ADD THIS LINE
        D = sparse.csc_matrix(D)  # Convert to efficient format

    # Convert to sparse if not already
    if not sparse.issparse(A):
        A = sparse.csc_matrix(A)
    if not sparse.issparse(C):
        C = sparse.csc_matrix(C)
    
    # Determine overlap between batches (for smooth regularization)
    overlap = 1  # One timepoint overlap between batches for continuity
    
    for i in range(0, n_times, batch_size - overlap):
        print(f"batch {round(i/batch_size)} / {int(n_times/batch_size)}")

        if i > 0:  # After first batch, start from overlap point
            i -= overlap
            
        # Get current batch bounds
        batch_start = i
        batch_end = min(i + batch_size, n_times)
        actual_batch_size = batch_end - batch_start
        
        # Skip if batch is too small (except for last batch)
        if actual_batch_size < 2 and i + batch_size < n_times:
            continue
            
        C_batch = C[:, i:batch_end]
        X_batch = cp.Variable((n_routes, actual_batch_size), nonneg=True)
        
        # Main residual term
        residual = cp.sum_squares(A @ X_batch - C_batch)
        
        # Efficient regularization term <-- MODIFIED SECTION
        if actual_batch_size > 1:
            if actual_batch_size == batch_size:
                # Use precomputed D if full batch
                temporal_reg = cp.sum_squares(X_batch @ D.T)
            else:
                # Handle partial batches with custom D
                D_partial = _create_diff_matrix(actual_batch_size)
                temporal_reg = cp.sum_squares(X_batch @ D_partial.T)
            objective = residual + gamma * temporal_reg
        else:
            objective = residual
        
        # Solve with SCS
        prob = cp.Problem(cp.Minimize(objective))
        prob.solve(
            solver=cp.SCS,
            verbose=False,
            max_iters=2000,
            eps=1e-3,
            use_indirect=True
        )
        print(prob.status)

        # Post-processing
        X_val = np.array(X_batch.value)
        X_val[np.abs(X_val) < 1e-5] = 0  # Set very small values to 0
        X_val = np.round(X_val)          # Round to nearest integer

        # Handle batch overlap
        if i == 0:
            # First batch - take all values
            X_full[:, batch_start:batch_end] = X_val
        else:
            # Subsequent batches - blend overlap region
            overlap_region = slice(batch_start, batch_start + overlap)
            new_region = slice(batch_start + overlap, batch_end)
            
            # Weighted average in overlap region (50/50)
            X_full[:, overlap_region] = 0.5 * (X_full[:, overlap_region] + X_val[:, :overlap])
            X_full[:, new_region] = X_val[:, overlap:]
    
    
    # === 6. Prepare results ===
    results_df = pd.DataFrame(
        X_full.T,  # Transpose to have time points as rows
        columns=routes,
        index=time_points
    ).reset_index().rename(columns={'index': 'Time'})

    # === 7. Save results ===
    results_df.to_csv("data/estimated_route_flows_reg.csv", index=False)
    print("\nResults saved to 'data/estimated_route_flows_reg.csv'")

    return X_full


def plot_comparison_heatmaps(C_matrix, AX_matrix, movements, time_points):
    """
    Plot side-by-side heatmaps comparing observed vs reconstructed matrices using Matplotlib
    
    Parameters:
        C_matrix: Observed movement counts (movements × time)
        AX_matrix: Reconstructed A@X values (movements × time) 
        movements: List of movement names for y-axis
        time_points: List of time points for x-axis
    """
    # Calculate difference
    diff = C_matrix - AX_matrix
    abs_diff = np.abs(diff)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Determine common scale for first two plots
    max_val = max(C_matrix.max(), AX_matrix.max())
    
    # Plot 1: Observed counts
    im1 = ax1.imshow(C_matrix, aspect='auto', cmap='YlOrRd', 
                    norm=colors.Normalize(vmin=0, vmax=max_val))
    ax1.set_title("Observed Counts (C_matrix)", pad=20)
    ax1.set_xlabel("Time Points")
    ax1.set_ylabel("Movement")
    fig.colorbar(im1, ax=ax1, label='Vehicle Count')
    
    # Plot 2: Reconstructed counts
    im2 = ax2.imshow(AX_matrix, aspect='auto', cmap='YlOrRd',
                    norm=colors.Normalize(vmin=0, vmax=max_val))
    ax2.set_title("Reconstructed Counts (A@X)", pad=20)
    ax2.set_xlabel("Time Points")
    fig.colorbar(im2, ax=ax2, label='Vehicle Count')
    
    # Plot 3: Difference (signed)
    im3 = ax3.imshow(diff, aspect='auto', cmap='coolwarm',
                    norm=colors.CenteredNorm())
    ax3.set_title("Difference (Observed - Reconstructed)", pad=20)
    ax3.set_xlabel("Time Points")
    fig.colorbar(im3, ax=ax3, label='Difference')

    # Configure ticks - handle potentially large time series
    num_time_points = C_matrix.shape[1]
    xtick_step = max(1, num_time_points // 20)  # Show ~20 ticks max
    xtick_positions = np.arange(0, num_time_points, xtick_step)
    
    # Convert time_points to datetime if they aren't already
    try:
        time_dates = pd.to_datetime(time_points)
        time_labels = [d.strftime('%m-%d %H:%M') for d in time_dates]
    except:
        # Fallback if conversion fails
        time_labels = [str(t) for t in time_points]
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(xtick_positions)
        
        # Set formatted time labels
        ax.set_xticklabels([time_labels[i] if i < len(time_labels) else '' 
                          for i in xtick_positions], rotation=45, ha='right')
        
        # Configure y-axis (movements)
        ytick_step = max(1, len(movements) // 10)  # Show ~10 movement labels
        ytick_positions = np.arange(0, len(movements), ytick_step)
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels([movements[i] for i in ytick_positions])
    
    plt.tight_layout()
    
    # Print statistics
    print("\n=== Reconstruction Quality ===")
    print(f"Mean Absolute Error: {abs_diff.mean():.2f}")
    print(f"Max Absolute Error: {abs_diff.max():.2f}")
    print(f"Percentage of points with <1 error: {(abs_diff < 1).mean()*100:.1f}%")
    
    # plt.show()
    plt.savefig("compare.png")

if __name__ == "__main__":

    # === 1. Read route-movement matrix (A matrix) ===
    A_df = pd.read_csv("data/UP_route_movement_matrix.csv")
    routes = A_df.columns[3:]  # Columns starting from 4th are routes
    movements = A_df[['Node', 'Direction']].astype(str).agg('_'.join, axis=1)

    # === 2. Read observation data (c_lt) ===
    c_df = pd.read_csv("data/Sage-0301/resampled_sage_data_1min.csv")
    c_df['Node_Dir'] = c_df[['Node', 'Direction']].astype(str).agg('_'.join, axis=1)

    # === 3. Get all time points and prepare the full C matrix ===
    time_points = sorted(c_df['Time'].unique())

    # Create a pivot table where rows are movements and columns are time points
    c_pivot = c_df.pivot_table(index='Node_Dir', columns='Time', values='Value', fill_value=0)

    # Align the c_pivot with the movements in A_df to ensure consistent ordering
    movement_index = A_df[['Node', 'Direction']].astype(str).agg('_'.join, axis=1)
    C_matrix = c_pivot.reindex(movement_index, fill_value=0).to_numpy()

    # === 4. Construct A matrix ===
    A_matrix = A_df[routes].to_numpy()



    # 1. Fast no-regularization version
    # X_no_reg = solve_optimized(A_matrix, C_matrix)

    # 2. Regularized version with batching
    X_reg = solve_optimized(A_matrix, C_matrix, gamma=0.1, time_batch_size=50)

    # Plot comparison
    plot_comparison_heatmaps(C_matrix, A_matrix @ X_reg, movements, time_points)
