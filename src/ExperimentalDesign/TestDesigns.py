import jax 
import jax.numpy as jnp
import jaxopt
import optax
import equinox as eqx
import diffrax
from utils.Solvers import solve, safe_solve
from utils.Tests import *

import time
from typing import Any, Callable, Optional, Tuple

from icecream import ic 

from matplotlib import pyplot as plt

import hydra
from omegaconf import OmegaConf
import wandb
import pickle

import numpy as np

import itertools



@eqx.filter_jit()
def optimize_simple(
    x: jnp.ndarray, 
    t: jnp.ndarray, 
    args_0: jnp.ndarray, 
    x0: Optional[jnp.ndarray] = None,
    max_iter: int = 1000, 
    tol: float = 1e-6, 
    optimizer_type: str = 'adam', 
    lr: float = 0.001, 
    vector_field: Optional[Callable] = None, 
    ic: bool = True,
    **optimizer_kwargs: Any
) -> Tuple[jnp.ndarray, float]:
    """
    Optimize a given objective function, optionally using .run() method.

    Parameters are the same as before...
    use_run_method: Flag to use the .run() method instead of manually running update_fun.

    return: Optimized values and final loss
    """

    def loss_function(x0, ti, args, vector_field, x):
        t, xi, results = safe_solve(ti, x0, args, vector_field)
        return jax.lax.cond(jnp.any(results == diffrax.RESULTS[0]), lambda x: jnp.mean((xi - x)**2), lambda x: jnp.inf, x)

    objective_fun = lambda args: loss_function(x[0], t, args, vector_field, x) if x0 is None else loss_function(x0, t, args, vector_field, x)
    
    if ic is True:
        
        def loss_function(args_x_0, ti, vector_field, x):
            
            args, x0 = args_x_0[0], args_x_0[1]
            xi = solve(ti, x0, args, vector_field)
            
            if jnp.any(jnp.isnan(xi)):
                return jnp.inf
            
            return jnp.mean((xi - x)**2)
        
        objective_fun = lambda args_x_0: loss_function(args_x_0, t, vector_field, x)

    # Initialize optimizer based on user choice
    if optimizer_type.lower() == 'lbfgs':
        optimizer = jaxopt.LBFGS(fun=objective_fun, maxiter=max_iter, tol=tol, **optimizer_kwargs)
    elif optimizer_type.lower() == 'adam':
        opt = optax.adam(lr)
        optimizer = jaxopt.OptaxSolver(fun=objective_fun, opt=opt, maxiter=max_iter, tol=tol, **optimizer_kwargs)
    elif optimizer_type.lower() == 'gd':
        optimizer = jaxopt.GradientDescent(fun=objective_fun, maxiter=max_iter, tol=tol, **optimizer_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    start_time = time.time()
    
    if ic is True:
        xi_final, state_final = optimizer.run(init_params=[args_0, x[0]]) if x0 is None else optimizer.run(init_params=[args_0, x0])
    else:
        xi_final, state_final = optimizer.run(init_params=args_0)
    
    end_time = time.time()
    
    print("Time elapsed: ", end_time - start_time)
    
    return xi_final



class SplitLikelihood():
    
    def __init__(self, key, data, times, std_dev, x0=None, K_folds=2):
        
        self.data = data[jnp.argsort(times)]
        self.times = jnp.sort(times)
        self.splits, self.splits_c = self.random_split(key, len(data), K_folds)
        # self.splits = self.splits[0].reshape(1, -1)
        # self.splits_c = self.splits_c[0].reshape(1, -1)
        
        # [plt.scatter(self.times[split], self.data[split]) for split in self.splits]
        # plt.show()
        self.MLEs = None
        self.total_MLE = None
        self.x0 = x0
        self.std_dev = std_dev
        
    
    def random_split(self, key, K_folds=1):
        total_points = self.data.shape[0]
        shuffled_indices = jax.random.permutation(key, jnp.arange(total_points))
        num_per_fold = total_points // K_folds

        fold_indices = [shuffled_indices[i * num_per_fold: (i + 1) * num_per_fold] for i in range(K_folds)]
        complement_indices = [jnp.setdiff1d(shuffled_indices, fold) for fold in fold_indices]

        return jnp.array(fold_indices), jnp.array(complement_indices)

        
    def get_MLE(self, field, w0, x0, split, max_iter=1000, tol=1e-6, optimizer_type='adam', lr=0.01, ic=False):
       
        t = self.times[split]
        data = self.data[split][jnp.argsort(t)]
        t=jnp.sort(t)

        return optimize_simple(
                data, 
                t, 
                x0=self.x0,
                args_0=w0, 
                max_iter=max_iter,
                tol=tol, 
                optimizer_type=optimizer_type, 
                lr=lr, 
                vector_field=field, 
                ic = ic)
    
    @staticmethod
    def exclude(array, idx):
        mask = jnp.ones_like(array, dtype=bool).at[idx].set(False)
        return array[mask]
    
    def fit(self, field, w0, max_iter=1000, tol=1e-6, optimizer_type='adam', lr=0.01):
        
        self.MLEs = jax.vmap(self.get_MLE, in_axes=[None, None, None, 0, None, None, None, None])(field, w0, self.x0, self.splits, max_iter, tol, optimizer_type, lr)
        
        self.MLE = self.get_MLE(field, w0, self.x0, jnp.arange(len(self.data)), max_iter, tol, optimizer_type, lr)        
        self.cross_L = jax.vmap(self.cross_likelihood_, in_axes=[None, 0, 0])(field, self.MLEs, self.splits_c)
    
    @staticmethod
    def likelihood(data, traj, split, std_dev):
        """
        Calculate the Gaussian likelihood of the data given a trajectory and noise variance.

        Parameters:
        - data: The observed data points.
        - traj: The expected trajectory (mean values).
        - split: Indices or boolean mask to select the data points to be included in the calculation.
        - variance: The variance of the Gaussian noise.

        Returns:
        - The likelihood of the selected data points.
        """
        
        selected_data = data[split]
        selected_traj = traj[split]

        # Calculate the Gaussian likelihood for each data point
        likelihoods = jax.scipy.stats.norm.pdf(selected_data, loc=selected_traj, scale=std_dev)

        # Multiply the likelihoods of all data points
        total_likelihood = jnp.prod(likelihoods)

        return total_likelihood
    
    def c_likelihood(self, field, t, x0, w, x):
        x_pred = solve(t, x0, w, field) 
        return self.likelihood(x, x_pred, self.splits_c, self.std_dev)

    def cross_likelihood_(self, field, MLE, c_split):
        x_pred = solve(self.times, self.x0, MLE, field) 
        return self.likelihood(self.data, x_pred, c_split, self.std_dev)
    
    @eqx.filter_jit
    def log_Tn(self, field, w):
        
        traj = solve(self.times, self.x0, w, field)
        likelihoods_top = jax.vmap(self.likelihood, in_axes=[None, None, 0, None])(self.data, traj, self.splits_c, self.std_dev)
        
        return jnp.log(jnp.mean(self.cross_L/likelihoods_top))
    
    def get_full_CI(self, field, alpha=0.05, delta_upper=[20, 20], delta_lower=[20, 20], num_points=20, plot_grid=False):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import jax.numpy as jnp

        avg_mle = jnp.array([60, 30.4])
        
        # Define the grid resolution for each parameter
        dx = (delta_upper[0] + delta_lower[0]) / num_points
        dy = (delta_upper[1] + delta_lower[1]) / num_points

        # Generate the grid
        x, y = jnp.mgrid[slice(avg_mle[0] - delta_lower[0], avg_mle[0] + delta_upper[0] + dx, dx),
                        slice(avg_mle[1] - delta_lower[1], avg_mle[1] + delta_upper[1] + dy, dy)]

        x_flat, y_flat = x.flatten(), y.flatten()

        # Compute SLR values
        slr_values = jax.vmap(lambda params: self.log_Tn(field, params))(jnp.column_stack((x_flat, y_flat)))
        z = slr_values.reshape(x.shape)

        # Calculate the threshold
        threshold = jnp.log(1 / alpha)

        # Identify parameters with SLR values under the threshold
        under_threshold = z < threshold
        params_under_threshold = jnp.column_stack((x[under_threshold], y[under_threshold]))

        if plot_grid:
            fig, ax = plt.subplots()
            # qm = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
            ax.contour(x, y, z, levels=[threshold], colors='red', linewidths=2)
            ax.set_title('SLR Heatmap Visualization')
            # fig.colorbar(qm, ax=ax)
            plt.xlabel('Parameter 1')
            plt.ylabel('Parameter 2')
            plt.show()

        return params_under_threshold
    
    def get_full_CI(self, field, alpha=0.05, delta_upper=[20, 20], delta_lower=[20, 20], num_points=20, plot_grid=False):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import jax.numpy as jnp

        avg_mle = jnp.array([60, 30.4])
        
        # Generate grids for each dimension
        grids = [jnp.linspace(avg_mle[i] - delta_lower[i], avg_mle[i] + delta_upper[i], num_points) for i in range(len(avg_mle))]

        # Generate a meshgrid (note: this may need adjustment for dimensions > 2)
        mesh = jnp.meshgrid(*grids, indexing='ij')

        # Flatten the meshgrid to create a list of parameter combinations
        param_combinations = jnp.stack([m.flatten() for m in mesh], axis=-1)

        # Compute SLR values
        slr_values = jax.vmap(lambda params: self.log_Tn(field, params))(param_combinations)
        z = slr_values.reshape([num_points] * len(avg_mle))

        # Calculate the threshold
        threshold = jnp.log(1 / alpha)

        # Identify parameters with SLR values under the threshold
        under_threshold = z < threshold
        params_under_threshold = param_combinations[under_threshold.flatten()]

        return params_under_threshold

    def get_SLR_mask(self, field, alpha=0.05, delta_upper=[20, 20], delta_lower=[20, 20], num_points=20):
        import matplotlib.pyplot as plt
        import jax.numpy as jnp

        avg_mle = jnp.array([60, 30.4])
        
        # Define the grid resolution for each parameter
        dx = (delta_upper[0] + delta_lower[0]) / num_points
        dy = (delta_upper[1] + delta_lower[1]) / num_points

        # Generate the grid
        x, y = jnp.mgrid[slice(avg_mle[0] - delta_lower[0], avg_mle[0] + delta_upper[0] + dx, dx),
                        slice(avg_mle[1] - delta_lower[1], avg_mle[1] + delta_upper[1] + dy, dy)]

        x_flat, y_flat = x.flatten(), y.flatten()

        # Compute SLR values for each grid point
        slr_values = jax.vmap(lambda params: self.log_Tn(field, params))(jnp.column_stack((x_flat, y_flat)))
        # z = slr_values.reshape(x.shape)

        # Calculate the threshold
        threshold = jnp.log(1 / alpha)

        # Create a binary 2D array for contour plotting
        z = jnp.where(slr_values.reshape(x.shape) < threshold, 1, 0)

        return z
    
    def compare_CI(self, other, field, delta_upper, delta_lower, alpha=0.05, num_points=20, mle=None, plot=False):
        import itertools
        import numpy as np
        # Ensure the MLE is set
        avg_mle = self.MLE if mle is None else mle
        # Number of parameters
        num_params = len(avg_mle)
        
        # Generate grids for each dimension
        grids = [jnp.linspace(avg_mle[i] - delta_lower[i], avg_mle[i] + delta_upper[i], num_points) for i in range(len(avg_mle))]

        mesh = jnp.meshgrid(*grids, indexing='ij')
        
        # Flatten the meshgrid to create a list of parameter combinations
        param_combinations = jnp.stack([m.flatten() for m in mesh], axis=-1)

        slr_values_self = jax.vmap(lambda params: self.log_Tn(field, params))(param_combinations)
        slr_values_other = jax.vmap(lambda params: other.log_Tn(field, params))(param_combinations)
        
        slr_values_self = slr_values_self.reshape([num_points] * num_params)
        slr_values_other = slr_values_other.reshape([num_points] * num_params)
        
        # Calculate the threshold
        threshold = jnp.log(1 / alpha)

        # Identify parameters with SLR values under the threshold
        under_threshold_self = slr_values_self < threshold
        under_threshold_other = slr_values_other < threshold
        
        params_self = jnp.column_stack([m[under_threshold_self] for m in mesh])
        params_other = jnp.column_stack([m[under_threshold_other] for m in mesh])

        if plot:

            # Number of plots required for the upper triangle (excluding the diagonal)
            num_plots = num_params * (num_params - 1) // 2

            # Determine the number of rows and columns for subplots
            num_rows = int(jnp.ceil(jnp.sqrt(num_plots)))
            num_cols = int(jnp.ceil(num_plots / num_rows))

            # Create a figure with subplots
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10), dpi=300)
            if not isinstance(axes, np.ndarray):
                axes = [axes]  # Wrap it in a list for consistency
            else:
                axes = axes.flatten()  # Flatten the axes array for easy indexing


            plot_index = 0  # Index to keep track of which subplot to use next

            for i, j in itertools.product(range(num_params), repeat=2):
                if i >= j:  # Skip redundant plots (lower triangle)
                    continue  # Only plot upper triangle (excluding diagonal)

                ax = axes[plot_index]
                plot_index += 1

                # Plot contour for the current object
                ax.contour(mesh[i], mesh[j], slr_values_self, levels=[threshold], colors='blue', linewidths=2)
                # Plot contour for the other object
                ax.contour(mesh[i], mesh[j], slr_values_other, levels=[threshold], colors='red', linewidths=2)

                # Set labels and titles
                ax.set_xlabel(f'Parameter {i+1}')
                ax.set_ylabel(f'Parameter {j+1}')

            # Remove unused subplots (if any)
            for k in range(plot_index, num_rows * num_cols):
                fig.delaxes(axes[k])

            # Add a legend and adjust layout
            fig.suptitle('95% Confidence Intervals', fontsize=14)
            fig.legend(['Experimental Design', 'Equidistant Design'], loc='upper right')
            plt.tight_layout()
            plt.show()

        return params_self, params_other

    def trajectory_CI(self, field, params_set, times = jnp.linspace(0, 10, 1000), plot=False):

        # Simulate trajectories for each set of parameters
        
        def l(params):
            return solve(times, self.x0, params, field)
        
        all_trajectories = jax.vmap(l)(params_set)
        
        mle_traj = solve(times, self.x0, self.MLE, field)

        # Convert list of trajectories to a 2D array (time x trajectories)
        all_trajectories = jnp.array(all_trajectories)

        # Determine min and max trajectories
        min_trajectory = jnp.min(all_trajectories, axis=0)
        max_trajectory = jnp.max(all_trajectories, axis=0)

        if plot:
            # Plotting
            fig, axs = plt.subplots(min_trajectory.shape[1], figsize=(12, 6), sharex=True)
            for i in range(min_trajectory.shape[1]):
                axs[i].fill_between(times, min_trajectory[:, i], max_trajectory[:, i], alpha=0.3)
                axs[i].plot(times, mle_traj[:, i], label='MLE Trajectory')
                axs[i].set_ylabel(f'Dimension {i+1}')
                axs[i].legend()
            plt.xlabel('Time')
            plt.show()
            
        return min_trajectory, max_trajectory, mle_traj
    
    def compare_trajectory_CI(self, other, field, CI_params_self, CI_params_other, times=jnp.linspace(0, 10, 1000), plot=False):
        """
        Compare and plot the confidence intervals of trajectories between two instances.

        Parameters:
        - other: Another instance for comparison.
        - field: Function representing the vector field.
        - CI_params_self: Parameters under the confidence interval for the current instance.
        - CI_params_other: Parameters under the confidence interval for the other instance.
        - times: Array of time points for the simulation.
        - plot: Boolean indicating whether to plot the trajectories.

        Returns:
        - Dictionary containing min, max, and MLE trajectories for both instances.
        """

        self_min, self_max, self_mle = self.trajectory_CI(field, CI_params_self, times)
        other_min, other_max, other_mle = other.trajectory_CI(field, CI_params_other, times)

        if plot:
            # Determine the number of dimensions
            num_dims = self_min.shape[1]
            # If there is only one dimension, wrap axs in a list
            if num_dims == 1:
                axs = [axs]
        
            # Plotting
            fig, axs = plt.subplots(num_dims, figsize=(12, 6 * num_dims), sharex=True)
            for i in range(num_dims):
                # Fill between for self confidence interval
                axs[i].fill_between(times, self_min[:, i], self_max[:, i], color='Blue', alpha=0.3)
                # Fill between for other confidence interval
                axs[i].fill_between(times, other_min[:, i], other_max[:, i], color='Red', alpha=0.3)
                # Plot MLE trajectories for self and other
                axs[i].plot(times, self_mle[:, i], color='Blue', label='Self MLE Trajectory')
                axs[i].plot(times, other_mle[:, i], color='Red', label='Other MLE Trajectory')
                axs[i].set_ylabel(f'Dimension {i+1}')
                if i == 0:
                    axs[i].legend()

            plt.xlabel('Time')
            plt.show()

        return {
            'self': [self_min, self_max, self_mle],
            'other': [other_min, other_max, other_mle]
        }
        
        

def compare_MLE(self, other, force_field, w0, rounds = 1):
    """
    Compare and plot the MLE trajectories between two instances.

    Parameters:
    - other: Another instance for comparison.
    - force_field: Function representing the vector field.
    - times: Array of time points for the simulation.
    - plot: Boolean indicating whether to plot the trajectories.

    Returns:
    - 2 Lists containing MLE estimates.
    """
    
    MLE_self = []
    MLE_other = []
    
    for i in range(len(self.data)):
        index= i+1
        self_data = self.data[:index]

        
        self_t = self.times[:index]
        self_data = self_data[jnp.argsort(self_t)]
        self_t = self_t[jnp.argsort(self_t)]
        
        if other != None:
                other_data = other.data[:index]
                other_t = other.times[:index]
                other_data = other_data[jnp.argsort(other_t)]
                other_t = other_t[jnp.argsort(other_t)]
        
        self_MLE = optimize_simple(
                self_data, 
                self_t, 
                x0=self.x0,
                args_0=w0, 
                max_iter=10000,
                tol=1e-6, 
                optimizer_type='adam', 
                lr=0.005, 
                vector_field=force_field, 
                ic = False)
        MLE_self.append(self_MLE)
        
        if other != None:
                other_MLE = optimize_simple(
                        other_data, 
                        other_t, 
                        x0=other.x0,
                        args_0=w0, 
                        max_iter=10000,
                        tol=1e-6, 
                        optimizer_type='adam', 
                        lr=0.005, 
                        vector_field=force_field, 
                        ic = False)
                MLE_other.append(other_MLE)
        
    return MLE_self, MLE_other

def process_rounds(rounds, key, t_range, n_points, true_args, true_x0, true_x0_, noise_std_dev, vector_field, ed, ed1, folds, w0):
    MLE_self_ = []
    MLE_other_ = []
    MLE_eq_1 = []
    MLE_eq_2 = []

    for i in range(1, rounds):
        key1, key2, key3, key4, key = jax.random.split(key, 5)
        skey1, skey2, skey3, skey4, skey = jax.random.split(key, 5)

        t_ed, data_ed, noisy_data_ed = generate_synthetic_data(t_range, n_points, true_args, true_x0, noise_std_dev, key1, vector_field, ed)
        t, data, noisy_data = generate_synthetic_data(t_range, n_points, true_args, true_x0_, noise_std_dev, key2, vector_field, ed1)
        t_eq, data_eq, noisy_data_eq = generate_synthetic_data(t_range, len(ed), true_args, true_x0, noise_std_dev, key3, vector_field, None)
        t_eq_, data_eq_, noisy_data_eq_ = generate_synthetic_data(t_range, len(ed), true_args, true_x0_, noise_std_dev, key3, vector_field, None)

        split_likelihood_ed = SplitLikelihood(skey1, noisy_data_ed, t_ed, x0=true_x0, std_dev=noise_std_dev, K_folds=folds)
        split_likelihood = SplitLikelihood(skey2, noisy_data, t, x0=true_x0_, std_dev=noise_std_dev, K_folds=folds)
        split_likelihood_eq_x0_ = SplitLikelihood(skey3, noisy_data_eq_, t_eq_, x0=true_x0_, std_dev=noise_std_dev, K_folds=folds)
        split_likelihood_eq_x0 = SplitLikelihood(skey4, noisy_data_eq, t_eq, x0=true_x0, std_dev=noise_std_dev, K_folds=folds)

        MLE_eq_x0, MLE_eq_x0_ = compare_MLE(split_likelihood_eq_x0, split_likelihood_eq_x0_, vector_field, w0)
        MLE_self, MLE_other = compare_MLE(split_likelihood_ed, split_likelihood, vector_field, w0)

        MLE_self_.append(jnp.array(MLE_self))
        MLE_other_.append(jnp.array(MLE_other))
        MLE_eq_1.append(MLE_eq_x0)
        MLE_eq_2.append(MLE_eq_x0_)

    return MLE_self_, MLE_other_, MLE_eq_1, MLE_eq_2

def save_to_file(MLEs, log_dir):    
    with open(log_dir, 'wb') as f:
        pickle.dump(MLEs, f)

def load_from_file(log_dir):
    with open(log_dir, 'rb') as f:
        MLEs = pickle.load(f)
    return MLEs

def cartesian_product(*arrays):
    """
    Input jax arrays and return their cartesian product.
    Args:
        array (jndarray): jndarray of shape (n, d)
    """
    
    return jnp.array(list(itertools.product(*arrays)))

# Test cartesian product:
def test_cartesian_product():
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5])
    c = jnp.array([6, 7])
    cp = cartesian_product(a, b, c)
    assert jnp.allclose(cp, jnp.array([[1, 4, 6], [1, 4, 7], [1, 5, 6], [1, 5, 7], [2, 4, 6], [2, 4, 7], [2, 5, 6], [2, 5, 7], [3, 4, 6], [3, 4, 7], [3, 5, 6], [3, 5, 7]]))

def perform_analysis(t_range, n_points, true_args, true_x0, w0, noise_std_dev, ed, no_ed = False, 
                     full_ed=False, noise_percentage=None, repeats=5, vector_field=None,
                     lr=0.001, max_steps=10000, **kwargs):
    
    key = jax.random.PRNGKey(0)

    @eqx.filter_jit()
    def round(key, t_range, n_points, true_args, true_x0, true_x0_, noise_std_dev, vector_field, ed, ed1, index):
        
        key1, key2, key3, key = jax.random.split(key, 4)
        
        t_ed, data_ed, noisy_data_ed = generate_synthetic_data(t_range, n_points, true_args, true_x0, noise_std_dev, key1, vector_field, ed[:index])
        noisy_data_ed = noisy_data_ed[jnp.argsort(t_ed)]
        t_ed = jnp.sort(t_ed)
        
        t, data, noisy_data = generate_synthetic_data(t_range, n_points, true_args, true_x0_, noise_std_dev, key2, vector_field, ed1[:index])
        noisy_data = noisy_data[jnp.argsort(t)]
        t = jnp.sort(t)
        
        t_eq, data_eq, noisy_data_eq = generate_synthetic_data(t_range, index, true_args, true_x0, noise_std_dev, key3, vector_field, None)
        t_eq_, data_eq_, noisy_data_eq_ = generate_synthetic_data(t_range, index, true_args, true_x0_, noise_std_dev, key3, vector_field, None)
        
        # datas = jnp.array([noisy_data_ed, noisy_data, noisy_data_eq, noisy_data_eq_])
        # times = jnp.array([t_ed, t, t_eq, t_eq_])
        # x0s = jnp.array([true_x0, true_x0_, true_x0, true_x0_])
        
        datas = jnp.array([noisy_data_ed, noisy_data_eq, noisy_data, noisy_data_eq_])
        times = jnp.array([t_ed, t_eq, t, t_eq_])
        x0s = jnp.array([true_x0, true_x0, true_x0_, true_x0_])

        MLES = jax.vmap(optimize_simple, in_axes=[0, 0, None, 0, None, None, None, None, None, None])(datas, times, 
                                            w0, x0s, 10000, 1e-6, 'adam', 0.005, vector_field, False)
        # Testing dummy MLEs
        
        # MLES = jax.random.normal(key, (4, 4))
        
        return MLES
    
    @eqx.filter_jit()
    def round2(key, t_range, n_points, true_args, true_x0, noise_std_dev, vector_field, ed, index):
        
        key1, key2, key3, key = jax.random.split(key, 4)
        
        t_ed, data_ed, noisy_data_ed = generate_synthetic_data(t_range, n_points, true_args, true_x0, noise_std_dev, key1, vector_field, ed[:index])
        noisy_data_ed = noisy_data_ed[jnp.argsort(t_ed)]
        t_ed = jnp.sort(t_ed)
        
        t_eq, data_eq, noisy_data_eq = generate_synthetic_data(t_range, index, true_args, true_x0, noise_std_dev, key3, vector_field, None)
        
        datas = jnp.array([noisy_data_ed, noisy_data_eq])
        times = jnp.array([t_ed, t_eq])
        x0s = jnp.array([true_x0, true_x0])

        MLES = jax.vmap(optimize_simple, in_axes=[0, 0, None, 0, None, None, None, None, None, None])(datas, times, 
                                            w0, x0s, 10000, 1e-6, 'adam', 0.001, vector_field, False)
        # Testing dummy MLEs
        
        # MLES = jax.random.normal(key, (4, 4))
        
        return MLES
    
    @eqx.filter_jit()
    def round3(key, t_range, n_points, true_args, true_x0s, noise_std_dev, vector_field, index, ed=None, no_ed=False, noise_percentage=None):
        
        key3, key = jax.random.split(key, 2)
        
        if no_ed:
            t, data, noisy_data = jax.vmap(generate_synthetic_data, 
                                                    in_axes=[None, None, None, 0, 
                                                            None, None, None, None, 
                                                            None])(t_range, index, true_args, 
                                                                    true_x0s, noise_std_dev,
                                                                    key3, vector_field, None,
                                                                    noise_percentage)
        else:
            t, data, noisy_data = jax.vmap(generate_synthetic_data, 
                                                    in_axes=[None, None, None, 0, 
                                                            None, None, None, None, 
                                                            None])(t_range, n_points, true_args, 
                                                                    true_x0s, noise_std_dev,
                                                                    key3, vector_field, ed[:index],
                                                                    noise_percentage)

        print("Data:", data)
        MLES = jax.vmap(optimize_simple, in_axes=[0, 0, None, 0, None, None, None, None, None, None])(noisy_data, t, 
                                            w0, true_x0s, max_steps, 1e-8, 'adam', lr, vector_field, False)
        
        # Test MLEs:
        # MLES = jax.random.normal(key, (len(true_x0s), len(true_args))) 
        
        return MLES
    if full_ed:
        
        keys = jax.random.split(key, repeats)
        key = jax.random.split(key, 1)[0]
        start = time.time()
        
        MLES = jax.vmap(round3, in_axes=[0, None, None, None, None, None, None, None, None, None, None])(
            keys, t_range, n_points, true_args, true_x0, noise_std_dev, vector_field, len(ed), ed, no_ed, noise_percentage)
        
        end = time.time()
        
        
        MLE_total = MLES.reshape(1, *MLES.shape) 
        plot = plot_results(MLE_total, true_args, labels=['Good IC ED',  'Good IC EQ ED', 'Bad IC ED', 'Bad IC EQ ED'], title=None)#r"$ ||\gamma - \hat{\gamma}||^2_2 $")
        
        wandb.log({"Time elapsed:": end - start,
                f"Experimental Design Performance:": wandb.Image(plot)})
        
        save_to_file(MLE_total, 'MLEs.pkl')
        
    else:
        for i in range(len(ed)):
            
            index = i + 1
            keys = jax.random.split(key, repeats)
            key = jax.random.split(key, 1)[0]
            
            start = time.time()
            # if true_x0_ is not None:
            #     MLES = jax.vmap(round, in_axes=[0, None, None, None, None, None, None, None, None, None, None])(
            #         keys, t_range, n_points, true_args, true_x0, true_x0_, noise_std_dev, vector_field, ed, ed1, index)

            MLES = jax.vmap(round3, in_axes=[0, None, None, None, None, None, None, None, None, None, None])(
                keys, t_range, n_points, true_args, true_x0, noise_std_dev, vector_field, index, ed, no_ed, noise_percentage)
            end = time.time()
            
            print(f"{index} MLES:", MLES)
            MLE_total = MLES.reshape(1, *MLES.shape) if i == 0 else jnp.concatenate([MLE_total, MLES.reshape(1, *MLES.shape)], axis = 0)
            plot = plot_results(MLE_total, true_args, labels=['Good IC ED',  'Good IC EQ ED', 'Bad IC ED', 'Bad IC EQ ED'], title=None)#r"$ ||\gamma - \hat{\gamma}||^2_2 $")
            
            wandb.log({"Time elapsed:": end - start,
                    f"Experimental Design Performance:": wandb.Image(plot)})
            
            save_to_file(MLE_total, 'MLEs.pkl')
        
    return MLE_total

def plot_results(MLE_reshaped, true_args, labels=None, title='MLE Results Comparison', point_format=['-', '-', '-', '-']):
    """
    Plots the mean and standard deviation of the difference between MLE_reshaped and true_args with customizable options.

    :param MLE_reshaped: The reshaped MLE results.
    :param true_args: The true argument values to compare against.
    :param labels: A list of labels for the plot lines. If None, default labels are used.
    :param title: Title of the plot.
    :param point_format: Format string for the plot points.
    """
    
    true_args_tile = jnp.tile(true_args, (*MLE_reshaped.shape[:-1], 1))
    self_dif = jnp.linalg.norm(MLE_reshaped - true_args_tile, axis=-1)
    self_dif_mean = jnp.mean(self_dif, axis=1)
    self_dif_std = jnp.std(self_dif, axis=1)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6)) #if MLE_reshaped.shape[2] > 2 else plt.subplots(figsize=(12, 6))

    # Plotting for the first subplot
    for i, (traj, var) in enumerate(zip(self_dif_mean.T, self_dif_std.T)):
        if i <= 1:  # Adjust the condition as needed
            label = labels[i] if labels and len(labels) > i else f'Var {i+1}'
            ax1.plot(traj, point_format[i], label=label)
            ax1.fill_between(range(len(traj)), traj - var, traj + var, alpha=0.3)

    ax1.set_xlabel('Number of deisgn points')
    ax1.set_ylabel(r"$ ||\gamma - \hat{\gamma}||^2_2 $")
    ax1.set_title(title)
    ax1.legend()
    # ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    
    # Plotting for the second subplot
    for i, (traj, var) in enumerate(zip(self_dif_mean.T, self_dif_std.T)):
        if i > 1:  # Adjust the condition as needed
            label = labels[i] if labels and len(labels) > i else f'Var {i+1}'
            ax2.plot(traj, point_format[i], label=label)
            ax2.fill_between(range(len(traj)), traj - var, traj + var, alpha=0.3)

    ax2.set_xlabel('Number of deisgn points')
    ax2.set_ylabel(r"$ ||\gamma - \hat{\gamma}||^2_2 $")
    ax2.set_title(title)
    ax2.legend()
    # ax2.set_yscale('log')  # Set y-axis to logarithmic scale
    
    return plt
    
    
def get_dif_mean(MLE, true_args):
    """
    Returns the ||\\gamma \\hat{\\gamma}||^2_2 of the results.

    :param MLE: The MLE results.
    :return: The mean of the MLE results.
    """
    return jnp.mean(jnp.linalg.norm(MLE-true_args, axis=-1), 0)

def get_dif_std(MLE, true_args):
    """
    Returns the standard deviation of ||\\gamma \\hat{\\gamma}||^2_2.

    :param MLE: The MLE results.
    :return: The standard deviation of the MLE results.
    """
    return jnp.std(jnp.mean(jnp.linalg.norm(MLE-true_args, axis=-1), axis= -2), axis=0)

def plot_results(MLE_reshaped, true_args, labels=None, title='MLE Results Comparison', point_format=['-', '-', '-', '-']):
    """
    Plots the mean and standard deviation of the difference between MLE_reshaped and true_args with customizable options.

    :param MLE_reshaped: The reshaped MLE results.
    :param true_args: The true argument values to compare against.
    :param labels: A list of labels for the plot lines. If None, default labels are used.
    :param title: Title of the plot.
    :param point_format: Format string for the plot points.
    """
    
    Means = jax.vmap(lambda x, y: get_dif_mean(x, y), in_axes = [0, None])(MLE_reshaped, true_args)
    mu = jnp.mean(Means, axis=1)
    var = jax.vmap(get_dif_std, in_axes = [0, None])(MLE_reshaped, true_args) if MLE_reshaped.shape[-2] > 1 else jnp.std(jnp.linalg.norm(MLE_reshaped - true_args, axis=-1), axis=1)[:, 0]

    # Create a figure with two subplots
    fig, ax1 = plt.subplots(1, figsize=(12, 6)) #if MLE_reshaped.shape[2] > 2 else plt.subplots(figsize=(12, 6))

    ax1.plot(mu, label = 'Mean')
    ax1.fill_between(range(len(mu)), mu - var, mu + var, alpha=0.3, label = 'Std')
    ax1.set_xlabel('Number of deisgn points')
    ax1.set_ylabel(r"$ ||\gamma - \hat{\gamma}||^2_2 $")
    ax1.set_title(title)
    ax1.legend()
    
    return plt

@hydra.main(config_path="conf/ExpTest", config_name="VDP")
def IC_Test(config):
    
    jax.config.update("jax_enable_x64", True)
    wandb.init(**config['logging'],
            config = OmegaConf.to_container(config, resolve=True))

    vector_fields = {'s1': s1_vfield, 
                     's2': s2_vfield,
                     's3': s3_vfield,
                     's4': s4_vfield,
                     's5': s5_vfield,
                     'osc': osc_vfield,
                     'osc2': osc_vfield2,
                     }
    
    upper_x0 = jnp.array(config['IC_Test']['upper_x0'])
    lower_x0 = jnp.array(config['IC_Test']['lower_x0'])
    n_per_dim = jnp.power(jnp.array(config['IC_Test']['x0_n_points']), 1/len(upper_x0)).astype(int)
    # x0s = cartesian_product(*[jnp.linspace(lower_x0[i], upper_x0[i], n_per_dim) for i in range(len(upper_x0))])
    x0s = jnp.linspace(lower_x0, upper_x0, config['IC_Test']['x0_n_points'])
    true_x0s = jnp.array(config['IC_Test']['true_x0']).reshape(1, -1) if config['IC_Test']['true_x0'] is not None else x0s
    
    ed = jax.random.randint(jax.random.PRNGKey(1), [len(jnp.array(config['IC_Test']['ed']))], minval=0, maxval=config['IC_Test']['n_points']) if config['IC_Test'].get('random_ed', False) else jnp.array(config['IC_Test']['ed'])
        
    MLE_total = perform_analysis(t_range = jnp.array(config['IC_Test']['t_range']),
                     n_points = config['IC_Test']['n_points'],
                     true_args = jnp.array(config['IC_Test']['true_args']),
                     true_x0= true_x0s,
                    #  true_x0_= jnp.array(config['IC_Test']['true_x0_']) if config['IC_Test']['true_x0_'] is not None else None,
                     w0 = jnp.array(config['IC_Test']['w0']),
                     noise_std_dev = config['IC_Test']['noise_std_dev'],
                     noise_percentage = config['IC_Test']['noise_percentage'],
                     full_ed=config['IC_Test'].get('full_ed', False),
                     ed = ed,
                     no_ed = config['IC_Test']['no_ed'],
                     repeats = config['IC_Test']['repeats'],
                     vector_field = vector_fields[config['IC_Test']['vector_field']],
                     lr = config['optimizer'].get('learning_rate', 0.001),
                     max_steps = config['optimizer'].get('max_steps', 10000)
                     )
    
    # Log the artifact to WandB
    artifact = wandb.Artifact(f"{config['IC_Test']['vector_field']}_STD{config['IC_Test']['noise_std_dev']}_NOED_{config['IC_Test']['no_ed']}", type="data")
    artifact.add_file('MLEs.pkl')
    wandb.log_artifact(artifact)

    wandb.finish()
    
if __name__ == "__main__":
    IC_Test()
    




