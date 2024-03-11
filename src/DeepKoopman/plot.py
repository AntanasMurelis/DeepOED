import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from ODE_Dataloader import solve 
from itertools import combinations
import collections
import numpy as np
from typing import List, Tuple

def plot_trajectories(time_span, trajectories, ax):
    num_vars = trajectories.shape[-1]
    if ax is None:
        fig, ax = plt.subplots(num_vars, 1, figsize=(8, 8*num_vars))  
    for var in range(num_vars):
        for i in range(len(trajectories)):
            ax[var].plot(time_span, trajectories[i, :, var], alpha=0.1)
        ax[var].set_xlabel(r"$t$ [Depth]")
        ax[var].set_ylabel(fr"$\mathbf{{x}}_{var}(t)$")

def plot_trajectories_3d(time_span, trajectories, ax=None):
    num_vars = trajectories.shape[-1]
    if ax is None:
        num_combinations = len(list(combinations(range(num_vars), 2)))
        fig = plt.figure(figsize=(8, 8 * num_combinations))
        axes = [fig.add_subplot(num_combinations, 1, i+1, projection='3d') for i in range(num_combinations)]
        for ax, (var1, var2) in zip(axes, combinations(range(num_vars), 2)):
            for i in range(len(trajectories)):
                ax.plot3D(trajectories[i, :, var1], trajectories[i, :, var2], time_span, alpha=0.1)
            ax.set_xlabel(fr"$\mathbf{{x}}_{var1}(t)$")
            ax.set_ylabel(fr"$\mathbf{{x}}_{var2}(t)$")
            ax.set_zlabel(r"$t$ [Depth]")
    else:
        for i in range(len(trajectories)):
            ax.plot3D(trajectories[i, :, 0], trajectories[i, :, 1], time_span, alpha=0.1)
        ax.set_xlabel(r"$\mathbf{x}_1(t)$")
        ax.set_ylabel(r"$\mathbf{x}_0(t)$")
        ax.set_zlabel(r"$t$ [Depth]")

    
def plot_state_space(trajectories, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

    for i in range(trajectories.shape[0]):
        ax.plot(trajectories[i, :, 0], trajectories[i, :, 1], alpha=0.1)
        
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    

def plot_state_space(trajectories, ax=None):
    num_vars = trajectories.shape[-1]
    if ax is None:
        num_combinations = len(list(combinations(range(num_vars), 2)))
        fig, axes = plt.subplots(num_combinations, 1, figsize=(8, 8 * num_combinations))
        if not isinstance(axes, collections.abc.Iterable):
            axes = [axes]
        for ax, (var1, var2) in zip(axes, combinations(range(num_vars), 2)):
            for i in range(trajectories.shape[0]):
                ax.plot(trajectories[i, :, var1], trajectories[i, :, var2], alpha=0.1)
            ax.set_xlabel(fr"$x_{var1}$")
            ax.set_ylabel(fr"$x_{var2}$")
    else:
        for i in range(trajectories.shape[0]):
            ax.plot(trajectories[i, :, 0], trajectories[i, :, 1], alpha=0.1)
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$x_1$")


def sample_initial_conditions(num_samples, ranges, key):
    keys = jax.random.split(key, len(ranges))
    initial_conditions = jnp.stack([jax.random.uniform(k, minval = r[0], maxval = r[1], shape = num_samples) for k, r in zip(keys, ranges)], axis=-1)
    return initial_conditions

def generate_trajectories(model, solve, time_span, initial_conditions, args, vector_field, plot_latent = False, from_latent=False, single = False):
    
    # Numerically integrate the ODEs to generate trajectories for each initial condition
    trajectories_solve = jax.vmap(solve, in_axes=(None, 0, None, None))(time_span, initial_conditions, args, vector_field)
    
    # Use the model to generate trajectories for each initial condition
    if single:
        get_sol = jax.vmap(model, in_axes=0)
        if plot_latent:
            trajectories_model = jax.vmap(get_sol, in_axes=0)(initial_conditions)[1]
        else:
            trajectories_model = jax.vmap(get_sol, in_axes=0)(initial_conditions)[0]
    else:
        if plot_latent:
            trajectories_model = jax.vmap(model, in_axes=0)(initial_conditions)[1]
        else:
            trajectories_model = jax.vmap(model, in_axes=0)(initial_conditions)[0]

    return trajectories_model, trajectories_solve


def plot_all_trajectories(time_span, trajectories_model, trajectories_solve, plot_2d = True, plot_3d = False, state_space = False):
    
    num_vars = trajectories_model.shape[-1]

    if plot_2d:
        plot_2d_trajectories(time_span, trajectories_model, trajectories_solve)
    if plot_3d and num_vars > 1:
        plot_3d_trajectories(time_span, trajectories_model, trajectories_solve) 
    if state_space: #and num_vars == 2:
        plot_state_space_diagrams(trajectories_model, trajectories_solve)

def plot_3d_trajectories(time_span, trajectories_model, trajectories_solve):
    
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Trajectories from NeuralODESolver')
    plot_trajectories_3d(time_span, trajectories_model, ax1)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Trajectories from numerical solution')
    plot_trajectories_3d(time_span, trajectories_solve, ax2)
    return fig

def plot_2d_trajectories(time_span, trajectories_model, trajectories_solve):
    num_vars = trajectories_model.shape[-1]
    fig, axs = plt.subplots(2, num_vars, figsize=(12,12))  # create a 2xN grid of subplots
    for var in range(num_vars):
        axs[0, var].set_title('Trajectories from NeuralODESolver -- D{}'.format(var))
        axs[1, var].set_title('Trajectories from numerical solution -- D{}'.format(var))
    plot_trajectories(time_span, trajectories_model, ax=axs[0])
    plot_trajectories(time_span, trajectories_solve, ax=axs[1])
    return fig


def plot_state_space_diagrams(trajectories_model, trajectories_solve):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    axs[0].set_title('State-Space Diagram from NeuralODESolver')
    plot_state_space(trajectories_model, ax=axs[0])

    axs[1].set_title('State-Space Diagram from numerical solution')
    plot_state_space(trajectories_solve, ax=axs[1])
    
    return fig


def sample_and_plot_trajectories(model, time_span, args, num_samples, ranges, key, plot_latent = False, from_latent = False, plot_2d = True, plot_3d = False, state_space=False):
    
    initial_conditions = sample_initial_conditions(num_samples, ranges, key)
    trajectories_model, trajectories_solve = generate_trajectories(model = model, 
                                                                   solve = solve, 
                                                                   time_span = time_span,
                                                                   initial_conditions = initial_conditions,
                                                                   args = args,
                                                                   vector_field = vector_field, 
                                                                   plot_latent = plot_latent,
                                                                   from_latent = from_latent)
    
    plot_all_trajectories(time_span, trajectories_model, trajectories_solve, plot_2d, plot_3d, state_space)


# def plot_trajectory_sets(set_1, set_2, time_points=None, parameters=None, title=None):
#     """
#     Plot two sets of trajectories overlapped.
    
#     Parameters:
#     - set_1: List of trajectories (each trajectory is a list of y-values) from set 1.
#     - set_2: List of trajectories (each trajectory is a list of y-values) from set 2.
#     - time_points: Optional list of time points to use as the X-axis. If not provided, 
#                    a range of integers starting from 0 will be used.
#     - parameters: Optional list of parameters that generated the trajectories. 
#                   Should have the same length as the number of trajectories.
#     """
    
#     plt.figure(figsize=(10, 6))
    
#     linestyles_set1 = '-'
#     linestyles_set2 = '--'
    
#     # Ensure both sets have the same number of trajectories
#     assert len(set_1) == len(set_2), "The two sets of trajectories must have the same number of trajectories."
    
#     # Default time points if none are provided
#     if time_points is None:
#         trajectory_length = len(set_1[0])
#         time_points = np.arange(trajectory_length)
    
#     num_trajectories = len(set_1)
    
#     # Check if parameters are provided and have the correct length
#     if parameters is not None:
#         assert len(parameters) == num_trajectories, "Parameters must have the same length as the number of trajectories."
    
#     # Generate random colors for each pair of trajectories
#     colors = [np.random.rand(3,) for _ in range(num_trajectories)]
    
#     # Plotting trajectories from set 1
#     for idx, trajectory in enumerate(set_1):
#         param_str = f'$p_1$: {parameters[idx, 0]}, $p_2$: {parameters[idx, 1]})' if parameters is not None else ''
#         plt.plot(time_points, trajectory, label=param_str, color=colors[idx], linestyle=linestyles_set1)
    
#     # Plotting trajectories from set 2
#     for idx, trajectory in enumerate(set_2):
#         plt.plot(time_points, trajectory, color=colors[idx], linestyle=linestyles_set2)
#     if title is not None:
#         plt.title(title)
#     else:
#         plt.title('Overlapped Trajectories from Two Sets')
    
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.show()
    
    
# def plot_trajectory_sets(set_1, set_2):
#     """
#     Plot two sets of trajectories overlapped.
    
#     Parameters:
#     - set_1: List of trajectories (each trajectory is a list of (x, y) tuples) from set 1.
#     - set_2: List of trajectories (each trajectory is a list of (x, y) tuples) from set 2.
#     """
    
#     plt.figure(figsize=(10, 6))
    
#     linestyles_set1 = '-'
#     linestyles_set2 = '--'
    
#     # Ensure both sets have the same number of trajectories
#     assert len(set_1) == len(set_2), "The two sets of trajectories must have the same number of trajectories."
    
#     num_trajectories = len(set_1)
    
#     # Generate random colors for each pair of trajectories
#     colors = [np.random.rand(3,) for _ in range(num_trajectories)]
    
#     # Plotting trajectories from set 1
#     for idx, trajectory in enumerate(set_1):
#         x, y = zip(*trajectory)
#         plt.plot(x, y, label=f'Set 1 - Trajectory {idx+1}', color=colors[idx], linestyle=linestyles_set1)
    
#     # Plotting trajectories from set 2
#     for idx, trajectory in enumerate(set_2):
#         x, y = zip(*trajectory)
#         plt.plot(x, y, color=colors[idx], linestyle=linestyles_set2)
    
#     plt.title('Overlapped Trajectories from Two Sets')
#     plt.xlabel('X')
#     plt.ylabel('Y')
    
#     return plt

def plot_trajectory_sets(set_1: List[List[Tuple[float, float]]], set_2: List[List[Tuple[float, float]]]) -> plt.Figure:
    """
    Plots two sets of trajectories overlapped on the same graph.
    
    Parameters:
    set_1 (List[List[Tuple[float, float]]]): List of trajectories where each trajectory is represented as a list of (x, y) tuples from set 1.
    set_2 (List[List[Tuple[float, float]]]): List of trajectories where each trajectory is represented as a list of (x, y) tuples from set 2.
    
    Returns:
    plt.Figure: A Matplotlib figure instance with the plot.
    """
    
    fig = plt.figure(figsize=(5, 5))
    
    linestyles_set1 = '-'
    linestyles_set2 = '--'
    
    # Ensure both sets have the same number of trajectories
    assert len(set_1) == len(set_2), "The two sets of trajectories must have the same number of trajectories."
    
    num_trajectories = len(set_1)
    
    # Generate random colors for each pair of trajectories
    colors = [np.random.rand(3,) for _ in range(num_trajectories)]
    
    # Plotting trajectories from set 1
    for idx, trajectory in enumerate(set_1):
        x, y = zip(*trajectory)
        plt.plot(x, y, label=f'Set 1 - Trajectory {idx+1}', color=colors[idx], linestyle=linestyles_set1)
    
    # Plotting trajectories from set 2
    for idx, trajectory in enumerate(set_2):
        x, y = zip(*trajectory)
        plt.plot(x, y, label=f'Set 2 - Trajectory {idx+1}', color=colors[idx], linestyle=linestyles_set2)
    
    plt.title('Overlapped Trajectories from Two Sets')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    return fig

def plot_error_trajectory(ts, predictions, sols):
    """
    Plot the error trajectory for multiple dimensions, with min-max error bounds
    for both squared and absolute errors.

    Parameters:
        ts: array
            Time points
        predictions: array
            Predicted values
        sols: array
            Ground-truth solutions
    Returns:
        fig: matplotlib.figure.Figure
            The generated figure object
    """
    
    num_dimensions = predictions.shape[-1]  # Number of dimensions
    num_plots = num_dimensions + 1  # Including one for total error
    
    fig = plt.figure(figsize=(20, 5))  # Create the main figure
    
    # Plot the total error
    ax1 = fig.add_subplot(1, num_plots, 1)
    
    total_abs_error = jnp.sum(jnp.abs(predictions - sols), axis=-1)
    total_squared_error = jnp.sum((predictions - sols)**2, axis=-1)
    
    # Calculate average, min, and max for total errors
    avg_total_abs_error = jnp.mean(total_abs_error, axis=0)
    min_total_abs_error = jnp.min(total_abs_error, axis=0)
    max_total_abs_error = jnp.max(total_abs_error, axis=0)
    
    avg_total_squared_error = jnp.mean(total_squared_error, axis=0)
    min_total_squared_error = jnp.min(total_squared_error, axis=0)
    max_total_squared_error = jnp.max(total_squared_error, axis=0)
    
    # Plot total errors with bounds
    ax1.plot(ts, avg_total_abs_error, lw=2, label='Average Abs Error')
    # ax1.fill_between(ts, min_total_abs_error, max_total_abs_error, alpha=0.2)
    
    # ax1.plot(ts, avg_total_squared_error, lw=2, label='Average Sq Error',)
    # ax1.fill_between(ts, min_total_squared_error, max_total_squared_error, alpha=0.2)
    
    ax1.set_xlabel('t')
    ax1.set_ylabel('Average Total Error')
    ax1.set_title('Average Total Error in Trajectories')
    ax1.legend()
    
    # Plot the individual errors
    for i in range(num_dimensions):
        ax = fig.add_subplot(1, num_plots, i + 2)
        
        abs_error = jnp.abs(predictions[:, :, i] - sols[:, :, i])
        squared_error = (predictions[:, :, i] - sols[:, :, i])**2
        
        # Calculate average, min, and max for individual dimension errors
        avg_abs_error = jnp.mean(abs_error, axis=0)
        min_abs_error = jnp.min(abs_error, axis=0)
        max_abs_error = jnp.max(abs_error, axis=0)
        
        avg_squared_error = jnp.mean(squared_error, axis=0)
        min_squared_error = jnp.min(squared_error, axis=0)
        max_squared_error = jnp.max(squared_error, axis=0)
        
        # Plot individual errors with bounds
        ax.plot(ts, avg_abs_error, lw=2, label=f'Average Abs Error')
        # ax.fill_between(ts, min_abs_error, max_abs_error, alpha=0.2)
        
        # ax.plot(ts, avg_squared_error, lw=2, label=f'Average Sq Error', )
        # ax.fill_between(ts, min_squared_error, max_squared_error, alpha=0.2)
        
        ax.set_xlabel('t')
        ax.set_ylabel(f'Average Error in Dim {i+1}')
        ax.legend()
    

    # Compute the time step
    delta_t = ts[1] - ts[0]
    
    # Compute the midpoint values of the average total absolute error
    midpoints = (avg_total_abs_error[:-1] + avg_total_abs_error[1:]) / 2
    
    # Compute the approximate integral using the midpoint rule
    approx_integral = np.sum(midpoints * delta_t)
    
    fig.suptitle('Average Error in Trajectories')
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust the layout
    
    return fig, approx_integral

def plot_parameter_error_trajectory(ts, predictions, sols):
    """
    Plot the error trajectory for multiple dimensions, with min-max error bounds
    for both squared and absolute errors.

    Parameters:
        ts: array
            Time points
        predictions: array
            Predicted values
        sols: array
            Ground-truth solutions
    Returns:
        fig: matplotlib.figure.Figure
            The generated figure object
    """
    
    num_dimensions = predictions.shape[-1]  # Number of dimensions
    num_plots = num_dimensions + 1  # Including one for total error
    
    fig = plt.figure(figsize=(20, 5))  # Create the main figure
    
    # Plot the total error
    ax1 = fig.add_subplot(1, num_plots, 1)
    
    diff_fun = lambda x, y: x - y
    
    difference = jax.vmap(diff_fun, in_axes=(0, 0))(predictions, sols)
    
    total_abs_error = jnp.sum(jnp.abs(difference), axis=-1)
    
    # Calculate average, min, and max for total errors
    avg_total_abs_error = jnp.mean(total_abs_error, axis=0)

    
    # Plot total errors with bounds
    ax1.plot(ts, avg_total_abs_error, lw=2, label='Average Abs Error')
    # ax1.fill_between(ts, min_total_abs_error, max_total_abs_error, alpha=0.2)
    
    # ax1.plot(ts, avg_total_squared_error, lw=2, label='Average Sq Error',)
    # ax1.fill_between(ts, min_total_squared_error, max_total_squared_error, alpha=0.2)
    
    ax1.set_xlabel('t')
    ax1.set_ylabel('Average Total Error')
    ax1.set_title('Average Total Error in Trajectories')
    ax1.legend()
    
    # Plot the individual errors
    # for i in range(num_dimensions):
    #     ax = fig.add_subplot(1, num_plots, i + 2)
        
    #     abs_error = jnp.abs(predictions[:, :, i] - sols[:, :, i])
    #     squared_error = (predictions[:, :, i] - sols[:, :, i])**2
        
    #     # Calculate average, min, and max for individual dimension errors
    #     avg_abs_error = jnp.mean(abs_error, axis=0)
        
    #     # Plot individual errors with bounds
    #     ax.plot(ts, avg_abs_error, lw=2, label=f'Average Abs Error')
        
    #     ax.set_xlabel('t')
    #     ax.set_ylabel(f'Average Error in Dim {i+1}')
    #     ax.legend()
    

    # Compute the time step
    delta_t = ts[1] - ts[0]
    
    # Compute the midpoint values of the average total absolute error
    midpoints = (avg_total_abs_error[:-1] + avg_total_abs_error[1:]) / 2
    
    # Compute the approximate integral using the midpoint rule
    approx_integral = np.sum(midpoints * delta_t)
    
    # fig.suptitle('Average Error in Trajectories')
    # plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust the layout
    
    return fig, approx_integral

def plot_time_trajectory_sets(set_1, set_2, time_points=None):
    """
    Plot two sets of trajectories overlapped.
    
    Parameters:
    - set_1: List of trajectories (each trajectory is a list of y-values) from set 1.
    - set_2: List of trajectories (each trajectory is a list of y-values) from set 2.
    - time_points: Optional list of time points to use as the X-axis. If not provided, 
                   a range of integers starting from 0 will be used.
    """
    
    fig = plt.figure(figsize=(10, 6))
    
    linestyles_set1 = '-'
    linestyles_set2 = '--'
    
    # Ensure both sets have the same number of trajectories
    assert len(set_1) == len(set_2), "The two sets of trajectories must have the same number of trajectories."
    
    # Default time points if none are provided
    if time_points is None:
        trajectory_length = len(set_1[0])
        time_points = np.arange(trajectory_length)
    
    num_trajectories = len(set_1)
    
    # Generate random colors for each pair of trajectories
    colors = [np.random.rand(3,) for _ in range(num_trajectories)]
    
    # Plotting trajectories from set 1
    for idx, trajectory in enumerate(set_1):
        plt.plot(time_points, trajectory, label=f'Set 1 - Trajectory {idx+1}', color=colors[idx], linestyle=linestyles_set1)
    
    # Plotting trajectories from set 2
    for idx, trajectory in enumerate(set_2):
        plt.plot(time_points, trajectory, label=f'Set 2 - Trajectory {idx+1}', color=colors[idx], linestyle=linestyles_set2)
    
    plt.title('Overlapped Trajectories from Two Sets')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    return fig
