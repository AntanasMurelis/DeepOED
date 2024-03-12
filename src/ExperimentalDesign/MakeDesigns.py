#----------------------------------------------------------------
# Latent Linear Experimental Designs
#----------------------------------------------------------------

import jax
import numpy as np
import jax.numpy as jnp
from jax import random as jrandom
import scipy
from icecream import ic

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns 

from utils.Solvers import solve, hess_solve
from utils.Tests import *
from DeepKoopman.Archs import load
from GP.GaussianProcess import GaussianProcess, ConstrainedGaussianProcess
from HermiteEmbedding import *
from ExperimentalDesign.Selection import GreedySelector

from scipy.optimize import minimize_scalar
from itertools import product

def cartesian_product(*arrays):
    return jnp.array(list(product(*arrays)))

#----------------------------------------------------------------
# Nullspace C
#----------------------------------------------------------------     
   
def get_Lt(dPhi, Phi, A_theta):
    
    I = jnp.identity(A_theta.shape[0])
    I_dPhi = jnp.kron(I, dPhi)
    A_Phi = jnp.kron(A_theta, Phi)
    
    return I_dPhi - A_Phi

def get_Lt_discrete(Phi_, Phi, A_theta):
    A_Phi = jax.vmap(lambda x: A_theta @ x)(Phi)
    return Phi_ - A_Phi

#----------------------------------------------------------------

# @jax.jit
def null_space(A, rcond=None):
    """
    Construct an orthonormal basis for the null space of A using SVD
    in JAX, in a JIT-compilable way.
    """
    # Compute the SVD of A
    u, s, vh = jax.scipy.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    ic(s)
    # Set the relative condition number
    if rcond is None:
        rcond = jnp.finfo(s.dtype).eps * max(M, N)

    # Determine the threshold below which singular values are considered zero
    tol = jnp.amax(s) * rcond

    # Create a mask for non-zero singular values
    rank = jnp.sum(s > tol, dtype=int)
    ic(rank)
    # Zero out rows of vh up to the calculated rank 
    
    mask = jnp.arange(vh.shape[0]) < rank
    
    Q = jnp.where(mask[:, None], 0, vh)
    return Q

def get_C(dPhi, Phi, A_theta):
    
    L = jax.vmap(get_Lt, in_axes = [0, 0, None])(dPhi, Phi, A_theta)
    L = L.reshape(-1, L.shape[-1])
    C_T = scipy.linalg.null_space(L.T)
    
    return C_T.T

def get_Q(dPhi, Phi, A_theta, rcond = 10e-5):
    
    L = jax.vmap(get_Lt, in_axes = [0, 0, None])(dPhi, Phi, A_theta)
    L = L.reshape(-1, L.shape[-1])
    Q = null_space(L, rcond=rcond)
    
    return Q

def get_Q_discrete(Phi_, Phi, A_theta, rcond = 10e-5):
    
    L = jax.vmap(get_Lt_discrete, in_axes = [0, 0, None])(Phi_, Phi, A_theta)
    L = L.reshape(-1, L.shape[-1])
    Q = null_space(L, rcond=rcond)
    return Q


#---Minimization of information criteria-------------------------

def minimize_scalarization(hermite, t_, constraint, design = 'A', lam = None, eta = None, sigma = None):
    
    """
    Minimize the scalarization function to find the optimal design.
    """
    
    Phi_t = jax.vmap(hermite.embed)(t_)
    dPhi_t = jax.vmap(jax.jacfwd(hermite.embed))(t_)  
    
    k = constraint.shape[0]
    
    def ext(Phi, k):
        return jnp.kron(jnp.identity(k), Phi).T    
    
    C = get_C(dPhi_t, Phi_t, constraint)  
      
    Phi_t = jax.vmap(ext, in_axes = [0, None])(Phi_t, k)
    
    selector = GreedySelector()
    
    eta, Info = selector.greedy_simple(Phi_t, C, 10, lam = 10, letter="A", repeats=False)

    return eta 


def minimize_scalarization_min_max(hermite, t_, constraint, design = 'A', lam = None, eta = None, rcond = 10e-5, sigma = 1):
    
    """
    Minimize the scalarization function to find the optimal design.
    """
    
    Phi_t = jax.vmap(hermite.embed)(t_)
    dPhi_t = jax.vmap(jax.jacfwd(hermite.embed))(t_)  
    
    k = constraint.shape[1]
    
    def ext(Phi, k):
        return jnp.kron(jnp.ones(k), Phi).T
    
    key = jax.random.PRNGKey(0)
    # constraint = jnp.array(constraint)
    # constraints = constraint + jax.random.normal(key, [5, constraint.shape[0], constraint.shape[1]]) * 100

    C = jax.vmap(get_Q, in_axes= [None, None, 0, None])(dPhi_t, Phi_t, constraint, rcond)
    
    Phi_t = jax.vmap(ext, in_axes = [0, None])(Phi_t, k)

    selector = GreedySelector()
    ic(Phi_t)
    
    eta, Info = selector.greedy_min_max(Phi_t, C, 15, lam = 10, letter="A", sigma = sigma, repeats=False)

    return eta 

def remove_empty_rows(arr):
    """
    Remove rows that are entirely zero across all batches.

    :param arr: 3D numpy array of shape [batch, rows, cols].
    :return: 3D numpy array with minimum rows needed.
    """
    # Step 1: Identify non-zero rows in each batch
    non_zero_rows = [np.any(batch, axis=1) for batch in arr]

    # Step 2: Find the union of non-zero rows across all batches
    rows_to_keep = np.any(non_zero_rows, axis=0)

    # Step 3: Slice the array to keep only the required rows
    return arr[:, rows_to_keep, :]


def minimize_scalarization_min_max_robust(hermite, t_, constraint, budget, lam = 0.1, eta = None, rcond = 10e-5, 
                                          sigma = 1, repeats = True, prior = None, bayes = False, mean_info = False,
                                          continuous = True, RA = None, RA_budget = None):
    
    """
    Minimize the scalarization function to find the optimal (greedy) robust design.
    """
    
    if continuous:
        Phi_t = jax.vmap(hermite.embed)(t_)
        dPhi_t = jax.vmap(jax.jacfwd(hermite.embed))(t_)  
    else:
        Phi_t = jax.vmap(hermite.embed)(t_[:-1])
        dPhi_t = jax.vmap(hermite.embed)(t_[1:])
    
    k = constraint.shape[-1]
    
    def ext(Phi, k):
        return jnp.kron(jnp.identity(k), Phi)
    
    if continuous:
        C = jax.vmap(get_Q, in_axes= [None, None, 0, None])(dPhi_t, Phi_t, constraint, rcond)
    else:
        C = jax.vmap(get_Q_discrete, in_axes= [None, None, 0, None])(dPhi_t, Phi_t, constraint, rcond)
        
    C = remove_empty_rows(C)
    
    Phi_t = jax.vmap(ext, in_axes = [0, None])(Phi_t, k)
    Phi_t = jax.vmap(lambda x: x.T @ x)(Phi_t)

    selector = GreedySelector()
    
    prior_ = sigma**2 * jnp.eye(Phi_t.shape[-1]) * lam if lam is not None else None
    prior__ = Phi_t[0] * prior + prior_ if prior is not None else prior_

    # eta, Info = selector.greedy_min_max_robust(Phi_t, C, budget, lam = lam, letter="A", repeats=False)
    eta, Info = selector.greedy_min_max_robust_RA(Phi_t, C, budget, lam = lam, letter="A", 
                                                      repeats=repeats, sigma=sigma, prior = prior__,
                                                      bayes=bayes, RA=RA, RA_budget=RA_budget)
    
    if mean_info:
        return eta, jnp.mean(jax.vmap(selector.A, in_axes=[None, 0])(Info, C))

    return eta 
#----------------------------------------------------------------


#----------------------------------------------------------------
# Continuous Designs
#----------------------------------------------------------------

def A_design_DCS(Phi_t, C, lam=None):
    
    n, o, d = Phi_t.shape
    k = C.shape[0]
    
    # Define the optimization variables
    eta = cp.Variable(n)
    u = cp.Variable(k)

    # Construct matrix V as a weighted sum of Phi_t.T @ Phi_t matrices
    
    V = sum(eta[i] * (Phi_t[i, :, :].T @ Phi_t[i, :, :]) for i in range(n))

    if lam is not None:
        V += lam * scipy.sparse.eye(d)

    objective = cp.Minimize(cp.sum(u))

    # Constraints
    constraints = [cp.sum(eta) == 1, eta >= 0]  # eta constraints

    # Semidefinite constraints for each i in range(k)
    I = np.eye(k) 
    for i in range(k):
        e_i = I[i, :].reshape(-1, 1)  # Unit vector for the ith element
        l = cp.reshape(u[i], (1, 1))
        G = cp.bmat([[V, C.T @ e_i], [e_i.T @ C, l]])  # Block matrix
        constraints.append(G >> 0)  # Semidefinite constraint

    # Problem and solver
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=True)  

    return eta.value

def minimize_scalarization_continuous(hermite, t_, constraint, design = 'A', lam = None, eta = None, rcond=10e-5, sigma = None):
    
    """
    Minimize the scalarization function to find the optimal design.
    """
    if constraint.ndim == 2:
        constraint = constraint.reshape(1, constraint.shape[0], constraint.shape[1])
    
    Phi_t = jax.vmap(hermite.embed)(t_)
    dPhi_t = jax.vmap(jax.jacfwd(hermite.embed))(t_)  
    
    k = constraint.shape[-1]
    
    def ext(Phi, k):
        return jnp.kron(jnp.identity(k), Phi) 
    
    C = jax.vmap(get_Q, in_axes = [None, None, 0])(dPhi_t, Phi_t, constraint)  
    C = remove_empty_rows(C)
    
    Phi_t = jax.vmap(ext, in_axes = [0, None])(Phi_t, k)
    
    if design == 'A':
        eta = list(map(lambda x: A_design_DCS(Phi_t, x, lam), C))
        return eta
    # if design == 'E':
    #     eta, t = E_design(Phi_t, C, lam)
    #     return eta, t
    return eta 

# @eqx.filter_jit
def Frank_Wolfe(objective, initial_distribution, tol=1e-5, iterations = 100, line_search = True):
    
    def simplex_projection(gradient):
        s = jnp.zeros_like(gradient)
        s = s.at[jnp.argmax(gradient)].set(1)
        return s    
    
    distribution = initial_distribution
    grad = jax.grad(objective)
    # prev_obj = 0
    for i in range(iterations):
        
        gradient = grad(distribution)
        s = simplex_projection(-gradient)
        
        if line_search:
            # Define the objective function along the line
            def line_objective(step_size):
                new_distribution = (1 - step_size) * distribution + step_size * s
                return objective(new_distribution)

            # Perform line search
            res = minimize_scalar(line_objective, bounds=(0, 1), method='bounded')
            step_size = res.x
        else:
            step_size = 2 / (i + 3)

        
        distribution = (1 - step_size) * distribution + step_size * s
        
    
    return distribution

def Exponentiated_Gradient_Descent(objective, initial_distribution, tol=1e-5, iterations=100, learning_rate=0.1):
    
    distribution = initial_distribution
    grad = jax.grad(objective)

    for i in range(iterations):
        gradient = grad(distribution)

        # Multiplicative update
        distribution *= jnp.exp(-learning_rate * gradient)

        # Normalization
        distribution /= jnp.sum(distribution)

        # Optional convergence check can be added here

    return distribution

def test_Frank_Wolfe():
        
        def objective(x):
            return x[0]**2 + x[1]**2
        
        initial_distribution = jnp.array([0.5, 0.5])
        
        distribution = Frank_Wolfe(objective, initial_distribution)
        
        ic(distribution)
        ic(objective(distribution))
        
        return None


def minimize_scalarization_continuous_FW(hermite, t_, constraint, sigma = None, design = 'A', lam = None, 
                                         eta = None, rcond=10e-5, iterations=100, prior = None, line_search=True, lr=0.1):
    
    """
    Minimize the scalarization function to find the optimal design.
    """
    if constraint.ndim == 2:
        constraint = constraint.reshape(1, constraint.shape[0], constraint.shape[1])
    
    Phi_t = jax.vmap(hermite.embed)(t_)
    dPhi_t = jax.vmap(jax.jacfwd(hermite.embed))(t_)  
    
    k = constraint.shape[-1]
    
    def ext(Phi, k):
        return jnp.kron(jnp.identity(k), Phi) 
    
    C = jax.vmap(get_Q, in_axes = [None, None, 0, None])(dPhi_t, Phi_t, constraint, rcond)  
    C = remove_empty_rows(C)
    
    Phi_t = jax.vmap(ext, in_axes = [0, None])(Phi_t, k)
    
    def Info(eta, Phi_i):
        return eta * Phi_i
    
    def A_obj(eta, Phi_t, C, lam, prior=None):
        Info_ = jnp.sum(jax.vmap(Info, in_axes = [0, 0])(eta, Phi_t), axis = 0)
        Info_ = Info_ + prior if prior is not None else Info_
        A = 1/sigma**2 * jnp.trace( C @ jax.numpy.linalg.pinv(Info_ + sigma**2 * lam * jnp.eye(len(Info_))) @ C.T )
        return A
    
    Phi_t_T_Phi_t = jax.vmap(lambda x: x.T @ x)(Phi_t)
    prior = Phi_t_T_Phi_t[0] * prior if prior is not None else None
    
    sum_C = lambda eta, C: jnp.sum(jax.vmap(A_obj, in_axes = [None, None, 0, None, None])(eta, Phi_t_T_Phi_t, C, lam, prior), axis = -1)
    
    objective = lambda eta: sum_C(eta, C)
    objective = jax.jit(objective)
    
    initial_distribution = jnp.ones(len(t_))/len(t_)
    
    if design == 'A':
        # eta = Frank_Wolfe(objective, initial_distribution, iterations=iterations, line_search=line_search, tol=1e-5)
        eta = Exponentiated_Gradient_Descent(objective, initial_distribution, iterations=iterations, learning_rate=lr)
        return eta

    return eta  
    

#------------------------------------------------------------------------------





        

#--Plotting--------------------------------------------------------------------
linear_field = lambda t, x, args: args @ x.T

import jax
import numpy as np
import matplotlib.pyplot as plt

def plot_eta(t, all_points, eta_t, sample_x0, vector_field, params, main_sample = False, eta_cont=None, ind_etas = None, design_sample = False, 
             FWHM = None, eta_t_robust = None, exclude_designs=False):
    """
    Plot the experimental design with eta_t marked on the x-axis and multidimensional sample trajectories.

    Parameters:
    - t: Time points (array) for plotting the complete trajectories.
    - all_points: All possible experimental design points (array).
    - eta_t: Suggested time points for sampling (array).
    - sample_x0: Sample initial conditions for trajectories (array).
    - vector_field: Function for the vector field.
    - params: Parameters for the vector field function.
    - solve: Function to solve the differential equation for trajectories.
    """

    # Convert JAX arrays to NumPy arrays for set operations
    all_points_np = np.array(all_points)
    eta_t_np = np.array(eta_t)
    t = jnp.sort(jnp.concatenate([eta_t_np, t, all_points])) #jnp.array(all_points_np) #jnp.sort(jnp.concatenate([eta_t_np, t]))
    
    # Compute sample trajectories
    if sample_x0.ndim != 1:
        map1 = jax.vmap(solve, in_axes=(None, 0, None, None))
        sample_trajectories = jax.vmap(map1, in_axes=(None, None, 0, None))(t, sample_x0, params, vector_field)
        sample_trajectories = sample_trajectories.reshape(-1, sample_trajectories.shape[-2],sample_trajectories.shape[-1])
    else:
        sample_trajectories = jax.vmap(solve, in_axes=(None, None, 0, None))(t, sample_x0, params, vector_field)

    # Determine the number of trajectories
    num_trajectories = sample_trajectories.shape[0]

    # Use a light color palette
    palette = sns.color_palette("pastel", num_trajectories)

    # Create the plot hq figure with higher DPI
    fig, ax = plt.subplots(dpi=150)

    #--Trajectories----------------------------------------------------
    for idx, trajectory in enumerate(sample_trajectories):
        for dim in range(trajectory.shape[1]):
            ax.plot(t, trajectory[:, dim], label=None, color=palette[idx])
            if design_sample and ind_etas is not None:
                # Identify the indices of eta_t in t
                # design_indices = np.isin(t, eta_t_np)

                # design_indices = np.isin(t, np.array(ind_etas[idx]))
                design_indices = np.isin(t, ind_etas[idx])
                # Plot the design sample points on the trajectories
                ax.scatter(t[design_indices], trajectory[design_indices, dim], marker='o', color=palette[idx], s=15)
                # ax.scatter(t[design_indices], trajectory[design_indices, dim], marker='o', color='lightgreen', s=15)

    #--ED-------------------------------------------------------------
    # Plot overall experimental design points
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Adjust the plot bottom space for individual designs
    if ind_etas is not None:
        relative_offset_factor = 0.015  # Adjust this factor as needed
        y_offset = y_range * relative_offset_factor
        # y_offset = 1.5  # Adjusted offset from the overall design
        extra_space = len(ind_etas) * y_offset if eta_t_robust is None else (len(ind_etas) + 1) * y_offset
        ax.set_ylim(y_min - extra_space, y_max)
        y_min, y_max = ax.get_ylim()  # Update y_min after adjusting the ylim
        y_min = y_min + y_offset 

    if eta_t_robust is not None:  
        ax.plot(all_points_np, [y_min] * len(all_points_np), 'o', markersize=4, color="grey", alpha=0.5)
        eta_t_robust_points = set(np.array(eta_t_robust)) & set(all_points_np)
        ax.plot(list(eta_t_robust_points), [y_min] * len(eta_t_robust_points), 'o', markersize=4, color="crimson", alpha=1.0, label='Robust ED')
        y_min =  y_min + y_offset
        
        
    # Plot main design points
    if not exclude_designs:
        ax.plot(all_points_np, [y_min] * len(all_points_np), 'o', markersize=4, color="grey", alpha=0.5)
        eta_t_points = set(eta_t_np) & set(all_points_np)
        ax.plot(list(eta_t_points), [y_min] * len(eta_t_points), 'o', markersize=4, color="forestgreen", alpha=1.0, label='Bayesian ED')
        

    if ind_etas is not None:
        ind_palette = sns.color_palette("pastel", len(ind_etas))
        for i, ind_eta in enumerate(ind_etas):
            ind_eta = np.array(ind_eta)
            ind_eta_points = set(ind_eta) & set(all_points_np)
            y_positions = [y_min + (i + 1) * y_offset] * len(all_points_np)
            ax.plot(all_points_np, y_positions, 'o', markersize=4, color="lightgrey", alpha=0.5)
            
            selected_y_positions = [y_min + (i + 1) * y_offset] * len(ind_eta_points)
            ax.plot(list(ind_eta_points), selected_y_positions, 'o', markersize=4, color=ind_palette[i])
            
            if design_sample or main_sample:
                # Iterate through each trajectory to plot the individual design points
                for dim in range(trajectory.shape[1]):
                    # Identify the indices of ind_eta in t
                    design_indices = np.isin(t, ind_eta) if not main_sample else np.isin(t, eta_t_np)
                    # Plot the individual design sample points on the trajectories
                    if main_sample:
                        ax.scatter(t[design_indices], sample_trajectories[i][design_indices, dim], marker='o', color='forestgreen', s=15)
                    else:
                        ax.scatter(t[design_indices], sample_trajectories[i][design_indices, dim], marker='o', color=ind_palette[i], s=15)


    #------------------------------------------------------------------

    # Add KDE plot if eta_cont is provided
    if eta_cont is not None:
        ax2 = ax.twinx()  # Create a second y-axis
        
        if FWHM is None:
            
            ax2.plot(all_points_np, eta_cont, color='crimson', label=f'Optimal ED')
            ax2.axis('off')
            # ax2.legend(loc='upper left')
        else:

        # Function to convert FWHM to sigma
            def fwhm2sigma(fwhm):
                return fwhm / jnp.sqrt(8 * jnp.log(2))

            # Assume that all_points_np and eta_cont are already defined numpy arrays
            # Convert them to JAX arrays
            # all_points_jax = jnp.array(all_points_np)
            all_points_jax = jnp.array(all_points_np)

            eta_cont_jax = jnp.array(eta_cont)

            # Set the FWHM (you might need to adjust this based on your data)
            sigma = fwhm2sigma(FWHM)

            # Define the Gaussian kernel function
            def gauss(x, mu=0.0, sigma=1.0):
                return jnp.exp(-((x - mu) / sigma) ** 2 / 2) / (sigma * jnp.sqrt(2 * jnp.pi))

            # Vectorize the gauss function over mu (data points)

            # Define the Gaussian smoothing function
            def smooth_point(x_position, x_vals, y_vals, sigma):
                kernel = gauss(x_vals, mu=x_position, sigma=sigma)
                kernel = kernel / jnp.sum(kernel)
                return jnp.sum(y_vals * kernel)

            # Vectorize the smoothing function
            smoothed_vals_vmap = jax.vmap(smooth_point, in_axes=(0, None, None, None))

            # Apply Gaussian smoothing using vmap
            smoothed_eta_cont = smoothed_vals_vmap(all_points_jax, all_points_jax, eta_cont_jax, sigma)
            # Plot the smoothed data
            ax2.plot(all_points_np, smoothed_eta_cont, label=f'Optimal ED')
            # Update the legend
            # ax2.legend(loc='upper left')
        
    # Setting labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Values')
    ax.set_title('Sample Trajectories with Experimental Design Points')
    # ax.legend()

    # Show plot
    plt.show()

    return None

def plot_design_phase(t, all_points, eta_t, sample_x0, vector_field, params, design_sample = True):

    # Convert JAX arrays to NumPy arrays for set operations
    all_points_np = np.array(all_points)
    eta_t_np = np.array(eta_t)
    t = jnp.array(all_points_np) #jnp.sort(jnp.concatenate([eta_t_np, t]))
    
    # Compute sample trajectories
    if sample_x0.ndim != 1:
        sample_trajectories = jax.vmap(solve, in_axes=(None, 0, 0, None))(t, sample_x0, params, vector_field)
    else:
        sample_trajectories = jax.vmap(solve, in_axes=(None, None, 0, None))(t, sample_x0, params, vector_field)

    # Determine the number of trajectories
    num_trajectories = sample_trajectories.shape[0]

    # Use a light color palette
    palette = sns.color_palette("pastel", num_trajectories)

    # Create the plot hq figure with higher DPI
    fig, ax = plt.subplots()#figsize=(6, 6), dpi=300)

    #--Trajectories----------------------------------------------------
    for idx, trajectory in enumerate(sample_trajectories):
        ax.plot(trajectory[:, 0], trajectory[:, 1], label=None, color=palette[idx])
        if design_sample:
            # Identify the indices of eta_t in t
            design_indices = np.isin(t, eta_t_np)
            # Plot the design sample points on the trajectories
            # ax.scatter(t[design_indices], trajectory[design_indices, dim], marker='x', color=palette[idx], s=15)
            ax.scatter( trajectory[design_indices, 0], trajectory[design_indices, 1], marker='o', color='lightgreen', s=15)

    plt.show()  

def plot_greedy_results(t, eta, ed_points, model, multi_args, parameters, eval_range=(0, 5), eta_cont = None, num_eval_points=1000):
    """
    Generalized function to plot results using Gaussian Processes and a provided model.

    Parameters:
    - t: Time points array.
    - eta: Indices for the selected time points.
    - ed_points: Experimental design points (pre-processed).
    - model: Model with methods for data processing (get_latent, get_K, decoder).
    - multi_args: Arguments for the model.
    - parameters: Parameters for the Gaussian Process model.
    - eval_range: Tuple indicating the range for evaluation points (start, end).
    - num_eval_points: Number of evaluation points.
    """

    # Use a light color palette
    num_trajectories = ed_points.shape[0]
    palette = sns.color_palette("pastel", num_trajectories)

    # Process data using the model
    traj_map_arg = jax.vmap(model.get_latent, in_axes=(0, None))
    latent_traj = jax.vmap(traj_map_arg, in_axes=[None, 0])(ed_points, multi_args)
    K_mats = jax.vmap(model.get_KoopmanK, in_axes=[None, 0])(None, multi_args)

    # Define evaluation points
    eval_points = jnp.linspace(*eval_range, num_eval_points)

    # Plotting
    plt.figure(figsize=(10, 6))
    for idx, (traj, constraint) in enumerate(zip(latent_traj, K_mats)):
        c_gp = ConstrainedGaussianProcess(parameters)
        c_gp.fit_(jnp.sort(t[eta]), traj, eval_points, constraint, noise_variance=1)
        mu, var = c_gp(eval_points, var=True)
        u_b, l_b = mu + var, mu - var

        map_back = jax.vmap(model.decoder)
        u_b_, l_b_, mu_ = map_back(u_b.T), map_back(l_b.T), map_back(mu.T)
        upper_bound_traj = u_b_[:, :ed_points.shape[-1]]
        lower_bound_traj = l_b_[:, :ed_points.shape[-1]]
        mu_traj = mu_[:, :ed_points.shape[-1]]

        # Plot mean trajectory and variance as shaded areas
        plt.plot(eval_points, mu_traj, color=palette[idx], linestyle='-')
        plt.fill_between(eval_points.flatten(), lower_bound_traj.flatten(), upper_bound_traj.flatten(), color=palette[idx], alpha=0.2)

    # Plotting experimental design points
    y_min, _ = plt.ylim()
    plt.plot(t, [y_min] * len(t), 'o', markersize=4, color="grey", alpha=0.5)
    
    # Overlay selected ED points in light red
    selected_ed_points = t[eta]
    plt.plot(selected_ed_points, [y_min] * len(selected_ed_points), 'o', markersize=4, color="deepskyblue", alpha=0.5, label='Robust ED Points')

    # Add KDE plot if eta_cont is provided
    if eta_cont is not None:
        ax2 = plt.gca().twinx()  # Create a second y-axis
        sns.kdeplot(eta_cont, ax=ax2, color="blue", label="Eta Cont Density", alpha=0.3)
        ax2.set_ylabel('Density')
        ax2.legend()

    # plt.legend()
    plt.show()
#------------------------------------------------------------------------------


def MM_Design():
    
    #---Step 1: Generate synthetic training data---------------------
    ti = jnp.linspace(0, 10, 10)
    t_ = jnp.linspace(0, 10, 100)
    args = jnp.array((60.0, 30.4)) # Assuming these are the true parameters for kcat and K_m
    multi_args = jnp.array([(60.0, 30.4), (30, 70), (45, 45), (100, 10)])
    # multi_args = jax.random.uniform(key = jax.random.PRNGKey(0), shape = (100, 2), minval = jnp.array([30, 30]), maxval=jnp.array([100, 100]))
    # multi_args = cartesian_product(jnp.linspace(30, 60, 10), jnp.linspace(50, 90, 10))
    true_x0 = jnp.array([110.0])  # True initial condition
    synthetic_data = solve(ti, true_x0, args, s3_vfield) # Generating synthetic data
    synthetic_data = synthetic_data + jax.random.normal(jax.random.PRNGKey(0), shape=synthetic_data.shape) * 1
    # Floor the data to zero
    synthetic_data = jnp.where(synthetic_data < 0, 0, synthetic_data)
    #----------------------------------------------------------------


    #---Step 2: Load the model---------------------------------------  
    # jax.config.update("jax_enable_x64", False)
    # model = load("/Users/antanas/GitRepo/NODE/Models/PLearnKoopman_MM_1D_Lip_10_16.eqx", type = 'PlearnKoopmanLip')

    # # model = load("/Users/antanas/GitRepo/NODE/Models/PLearnKoopman_MM_2D_Cons_K", type = 'PlearnKoopmanCL')
    # jax.config.update("jax_enable_x64", True)

    # embedded_data = jax.vmap(model.get_latent, in_axes=(0, None))(synthetic_data, args)
    # # embedded_data_1 = model.get_latent_series(ti, true_x0, args)
    # A_theta_ = model.encoderP(args)
    # # A_theta = model.get_K(args)
    # A_theta = model.get_naive(A_theta_)
    # A_thetas = jax.vmap(model.get_naive)(jax.vmap(model.encoderP)(multi_args))
    
    # # A_thetas = A_thetas[:4]
    # A_theta_ = A_thetas[-1].reshape(1, A_theta.shape[0], A_theta.shape[1])
    #----------------------------------------------------------------
    
    #---Step 2: Load the model---------------------------------------  
    jax.config.update("jax_enable_x64", False)
    model = load('/Users/antanas/GitRepo/NODE/Models/MM/MM_2D_0208.eqx', type = 'DynamicKoopman')
    # model = load("/Users/antanas/GitRepo/NODE/Models/PLearnKoopman_MM_2D_Cons_K", type = 'PlearnKoopmanCL')
    jax.config.update("jax_enable_x64", True)
    A_thetas = jax.vmap(lambda w: model.get_KoopmanK(None, w))(multi_args)
    #----------------------------------------------------------------


    # list(map(lambda x: ic(jnp.linalg.eigvals(x)), A_thetas))

    #---Step 3: Fit the GP-------------------------------------------
    H = HermiteLayer(scale=1, d=1, m=50, o=2)
    # H.constrained_optimization(ti, embedded_data, t_=t_, A_theta=A_theta)
    #----------------------------------------------------------------

    #---Step 4: Experimental Design-----------------------------------
    t_ = jnp.linspace(0, 10, 100)
    sigma = 5.0
    # lam = 0.01
    # sigma = 0.1
    lam = 0.0001
    prior= 100
    eta = minimize_scalarization_min_max_robust(H, t_, A_thetas, budget = 15, sigma = sigma,prior=prior, lam = lam, rcond = 10e-10, repeats = False, bayes=True)
    ic(eta)
    eta_minimax = minimize_scalarization_min_max_robust(H, t_, A_thetas, budget = 15, sigma = sigma, prior=prior, lam = lam, rcond = 10e-10, repeats = False, bayes=False)
   
    eta_ = None
    eta_ = list(map(lambda th: minimize_scalarization_min_max_robust(H, t_, th.reshape(1, th.shape[0], th.shape[1]), sigma = sigma, prior=prior,budget = 15, lam = lam, repeats = False), A_thetas))
    
    # print(list(map(lambda a: a.shape, A_thetas)))
    # eta_ = eta_.reshape(eta_.shape[0], eta_.shape[1])
    # etas = minimize_scalarization_continuous(H, t_, A_thetas, design = 'A', lam = 0.1)
    # eta_cont = sum(etas)/len(etas)
    eta_cont = None
    #----------------------------------------------------------------


    #---Step 5: Plot the results-------------------------------------
    plot_eta(t_, t_, t_[eta], true_x0, s3_vfield, multi_args, eta_cont = eta_cont, ind_etas = [t_[_] for _ in eta_] if eta_ is not None else None, design_sample=True, eta_t_robust=t_[eta_minimax] if eta_minimax is not None else None)
    multi_args_ = multi_args
    multi_args_embedded = jax.vmap(model.get_latent, in_axes=(None, 0))(true_x0 , multi_args_)
    K_mats = jax.vmap(model.get_K, in_axes=[None, 0])(None, multi_args_)
    plot_eta(t_, t_, t_[eta], multi_args_embedded, linear_field, K_mats) #, ind_etas = [t_[_] for _ in eta_])
    #-----------------------------------------------------------------


    #---Step 6: Test greedy-------------------------------------------
    ed_t = jnp.sort(t_[eta])
    traj_map_arg = jax.vmap(model.get_latent, in_axes=(0, None))
    ed_points = solve(ed_t, true_x0, args, s3_vfield)
    
    latent_traj = jax.vmap(traj_map_arg, in_axes=[None, 0])(ed_points, multi_args)

    K_mats = jax.vmap(model.get_K, in_axes=[None, 0])(None, multi_args)
    
    eval_points = jnp.linspace(0, 10, 1000)
    
    parameters = {'scale': 1, 
                'd': 1, 
                'm': 30, 
                'o': 2, 
    }
    
    plot_greedy_results(t_, eta, ed_points, model, multi_args, parameters, (0, 10), 100)
        
def S1_design():
    
    #---Step 1: Generate synthetic training data---------------------
    ti = jnp.linspace(0, 10, 10)
    t_ = jnp.linspace(0, 10, 100)
    args = jnp.array((-0.5, -4.1))# Assuming these are the true parameters for kcat and K_m
    # multi_args = cartesian_product(jnp.linspace(-1, -0.1, 2), jnp.linspace(-5, -1, 3))
    multi_args = cartesian_product(jnp.linspace(-5, -0.2, 15), jnp.linspace(-4, -0.2, 15))
    multi_args= jnp.array([-0.3, -2.0]).reshape(1, 2)
    
    # multi_args = args.reshape(1, -1)
    true_x0 = jnp.array([5.0, 0.0]) # True initial condition

    #----------------------------------------------------------------


    #---Step 2: Load the model---------------------------------------  
    jax.config.update("jax_enable_x64", False)
    model = load('/Users/antanas/GitRepo/NODE/Models/S1_0118/S1_3D.eqx', type='DynamicKoopman')   
    # model = load('/Users/antanas/GitRepo/NODE/Models/S1_0118/S1_5D_0207.eqx', type='DynamicKoopman')  

    jax.config.update("jax_enable_x64", True)


    A_thetas = jax.vmap(lambda w: model.get_KoopmanK(None, w))(multi_args)
    #----------------------------------------------------------------


    #---Step 3: Fit the GP-------------------------------------------
    H = HermiteLayer(scale=1, d=1, m=50, o=1)
    #----------------------------------------------------------------

    #---Step 4: Experimental Design-----------------------------------
    t_ = jnp.linspace(0, 10, 100)

    sigma = 2
    lam = 0.001
    prior = 100
    budget = 30
    rcond = 10e-10 
    key= jax.random.PRNGKey(0)
    
    # eta = minimize_nonlinear_likelihood(s1_vfield, t_, true_x0, multi_args, budget, sigma, lam = 0.0)
    # eta2 = minimize_scalarization_Likelihood(key, s1_vfield, t_, true_x0, args.reshape(1, -1), args, budget, sigma)

    # eta = minimize_scalarization_min_max_robust(H, t_, A_thetas, budget = budget, sigma = sigma, lam = lam, prior = prior, rcond = rcond, repeats = True, bayes=True)
    eta = jnp.array([25, 51, 74,  1,  2,  3,  5, 16, 99,  4, 15,  6, 18,  7, 23,  1, 16,
       3,  5, 13,  8, 35,  9, 41,  1, 20,  3, 94, 35, 18])
    ic(eta)

    eta_robust = None
    # eta_robust = minimize_scalarization_min_max_robust(H, t_, A_thetas, budget = budget, sigma = sigma, lam = lam, prior = prior, rcond = rcond, repeats = True, bayes=False)
    
    eta_ = None
    eta_ = [eta]
    # eta_ = list(map(lambda th: minimize_scalarization_min_max_robust(H, t_, th.reshape(1, th.shape[0], th.shape[1]), sigma = sigma, budget = budget, lam = lam, prior = prior, rcond = rcond, repeats = False, bayes=True), A_thetas))

    eta_cont = None
    # eta_cont = minimize_scalarization_continuous_FW(H, t_, A_thetas, design = 'A', lam = lam, rcond = 10e-10, sigma = sigma, iterations=10, prior = prior, line_search=True, lr = 0.01)
    #----------------------------------------------------------------


    #---Step 5: Plot the results-------------------------------------
    plot_eta(t_, t_, t_[eta], true_x0, s1_vfield, multi_args, eta_cont = eta_cont, ind_etas = [t_[_] for _ in eta_] if eta_ is not None else None, main_sample=True,
             design_sample = True, FWHM = 0.5, eta_t_robust=t_[eta_robust] if eta_robust is not None else None)
    # plot_design_phase(t_, t_, t_[eta], true_x0, s1_vfield, multi_args, design_sample = True)
    
    multi_args_ = multi_args
    multi_args_embedded = jax.vmap(model.get_latent, in_axes=(None, 0))(true_x0 , multi_args_)
    K_mats = jax.vmap(model.get_K, in_axes=[None, 0])(None, multi_args_)
    plot_eta(t_, t_, t_[eta], multi_args_embedded, linear_field, K_mats) #, ind_etas = [t_[_] for _ in eta_])
    #-----------------------------------------------------------------


    #---Step 6: Test greedy-------------------------------------------
    ed_t = jnp.sort(t_[eta])
    traj_map_arg = jax.vmap(model.get_latent, in_axes=(0, None))
    ed_points = solve(ed_t, true_x0, args, s3_vfield)
    
    latent_traj = jax.vmap(traj_map_arg, in_axes=[None, 0])(ed_points, multi_args)

    K_mats = jax.vmap(model.get_K, in_axes=[None, 0])(None, multi_args)
    
    eval_points = jnp.linspace(0, 10, 1000)
    
    parameters = {'scale': 1, 
                'd': 1, 
                'm': 50, 
                'o': 2, 
    }
    
    plot_greedy_results(t_, eta, ed_points, model, multi_args, parameters, (0, 10), 100)

def PK_design():
    
    #---Step 1: Generate synthetic training data---------------------
    ti = jnp.linspace(0, 10, 10)
    t_ = jnp.linspace(0, 10, 100)    
    # multi_args = cartesian_product(jnp.linspace(-1, -0.1, 2), jnp.linspace(-5, -1, 3))
    multi_args = cartesian_product(jnp.linspace(1, 3, 4), jnp.linspace(10.0, 30.0, 4), jnp.linspace(50.0, 70.0, 4))

    # multi_args = jnp.array([2.0, 5.0, 0.1]).reshape(1, -1)
    
    true_x0 = jnp.array([20.0, 0.0]) # True initial condition

    #----------------------------------------------------------------


    #---Step 2: Load the model---------------------------------------  
    jax.config.update("jax_enable_x64", False)
    model = load("/Users/antanas/GitRepo/NODE/Models/PK/PK_D3_0213.eqx", type = 'DynamicKoopman')
    jax.config.update("jax_enable_x64", True)


    A_thetas = jax.vmap(lambda w: model.get_KoopmanK(None, w))(multi_args)
    #----------------------------------------------------------------


    #---Step 3: Fit the GP-------------------------------------------
    H = HermiteLayer(scale=1, d=1, m=50, o=1)
    #----------------------------------------------------------------

    #---Step 4: Experimental Design-----------------------------------
    t_ = jnp.linspace(0, 10, 100)

    sigma = 5
    lam = 0.00001
    prior = 500
    budget = 50
    rcond = 10e-10 
    key= jax.random.PRNGKey(0)
    
    # eta = minimize_nonlinear_likelihood(s1_vfield, t_, true_x0, multi_args, budget, sigma, lam = 0.0)
    # eta2 = minimize_scalarization_Likelihood(key, s1_vfield, t_, true_x0, args.reshape(1, -1), args, budget, sigma)

    eta = minimize_scalarization_min_max_robust(H, t_, A_thetas, budget = budget, sigma = sigma, lam = lam, prior = prior, rcond = rcond, repeats = False, bayes=True)
    ic(eta)

    eta_robust = None
    # eta_robust = minimize_scalarization_min_max_robust(H, t_, A_thetas, budget = budget, sigma = sigma, lam = lam, prior = prior, rcond = rcond, repeats = True, bayes=False)
    
    eta_ = None
    # eta_ = list(map(lambda th: minimize_scalarization_min_max_robust(H, t_, th.reshape(1, th.shape[0], th.shape[1]), sigma = sigma, budget = budget, lam = lam, prior = prior, rcond = rcond, repeats = False, bayes=True), A_thetas))

    eta_cont = None
    # eta_cont = minimize_scalarization_continuous_FW(H, t_, A_thetas, design = 'A', lam = lam, rcond = 10e-10, sigma = sigma, iterations=10, prior = prior, line_search=True, lr = 0.01)
    #----------------------------------------------------------------


    #---Step 5: Plot the results-------------------------------------
    plot_eta(t_, t_, t_[eta], true_x0, s7_vfield, multi_args, eta_cont = eta_cont, ind_etas = [t_[_] for _ in eta_] if eta_ is not None else None,
             design_sample = True, FWHM = 0.5, eta_t_robust=t_[eta_robust] if eta_robust is not None else None)
    # plot_design_phase(t_, t_, t_[eta], true_x0, s1_vfield, multi_args, design_sample = True)
    
    multi_args_ = multi_args
    multi_args_embedded = jax.vmap(model.get_latent, in_axes=(None, 0))(true_x0 , multi_args_)
    K_mats = jax.vmap(model.get_K, in_axes=[None, 0])(None, multi_args_)
    plot_eta(t_, t_, t_[eta], multi_args_embedded, linear_field, K_mats) #, ind_etas = [t_[_] for _ in eta_])
    #-----------------------------------------------------------------


    #---Step 6: Test greedy-------------------------------------------
    ed_t = jnp.sort(t_[eta])
    traj_map_arg = jax.vmap(model.get_latent, in_axes=(0, None))
    ed_points = solve(ed_t, true_x0, args, s3_vfield)
    
    latent_traj = jax.vmap(traj_map_arg, in_axes=[None, 0])(ed_points, multi_args)

    K_mats = jax.vmap(model.get_K, in_axes=[None, 0])(None, multi_args)
    
    eval_points = jnp.linspace(0, 10, 1000)
    
    parameters = {'scale': 1, 
                'd': 1, 
                'm': 50, 
                'o': 2, 
    }
    
    plot_greedy_results(t_, eta, ed_points, model, multi_args, parameters, (0, 10), 100)

def VDP_design():
    
    #---Step 1: Generate synthetic training data---------------------
    ti = jnp.linspace(0, 10, 10)
    t_ = jnp.linspace(0, 10, 100)
    args = jnp.array([1.5]) # Assuming these are the true parameters for kcat and K_m
    # multi_args = jnp.array([(60.0, 30.4), (30, 70), (25, 45), (100, 10)])
    # multi_args = jax.random.uniform(key = jax.random.PRNGKey(0), shape = (100, 1), minval = jnp.array([0.2]), maxval=jnp.array([2.0]))
    multi_args = jnp.linspace(0.5, 3.0, 50).reshape(-1, 1)
    # multi_args = jnp.array([0.5, 1.0, 2.0, 3.0]).reshape(-1, 1)  
    # multi_args = jnp.array([1.5]).reshape(-1, 1) 
    # multi_args = args.reshape(1, -1)
    # print(multi_args)
    true_x0 = jnp.array([2.0, 2.0]) # True initial condition
    # true_x0 = cartesian_product(jnp.linspace(-4.0, 4.0, 10), jnp.linspace(-4.0, 4.0, 10))
    # true_x0 = jnp.array([1.5, 2.0]) # True initial condition
    # synthetic_data = solve(ti, true_x0, args, s5_vfield) # Generating synthetic data
    # synthetic_data = synthetic_data + jax.random.normal(jax.random.PRNGKey(0), shape=synthetic_data.shape) * 1
    # Floor the data to zero
    # synthetic_data = jnp.where(synthetic_data < 0, 0, synthetic_data)
    #----------------------------------------------------------------


    #---Step 2: Load the model---------------------------------------  
    jax.config.update("jax_enable_x64", False)
    model = load('/Users/antanas/GitRepo/NODE/Models/VDP/VDP_3D_0120.eqx', type='DynamicKoopman')
    # model = load('/Users/antanas/GitRepo/NODE/Models/VDP/VDP_5D_0218.eqx', type='DynamicKoopman')
    jax.config.update("jax_enable_x64", True)

    A_thetas = jax.vmap(lambda w: model.get_KoopmanK(None, w))(multi_args)
    #----------------------------------------------------------------

    #---Step 3: Fit the GP-------------------------------------------
    H = HermiteLayer(scale=1, d=1, m=50, o=1)
    #----------------------------------------------------------------

    #---Step 4: Experimental Design-----------------------------------
    t_ = jnp.linspace(0, 10, 100)
    sigma = 1.0
    lam = 0.001
    prior = 10.0
    eta = minimize_scalarization_min_max_robust(H, t_, A_thetas, budget = 15, sigma = sigma, lam = lam, 
                                                prior = prior, rcond = 10000e-15, repeats = True, bayes=False, design='D')
    # ic(eta)

    t_ = jnp.linspace(0, 10, 100)  # Time points
    x0 = jnp.array([2.0, 2.0])  # Initial condition
    Gamma = jnp.linspace(0.5, 4, 10).reshape(-1, 1)  # Parameter range
    true_gamma = jnp.array([1.5])  # True parameter value
    budget = 10  # Budget for the minimization
    sigma = 1.0  # Noise standard deviation
    num_samples = 100  # Number of samples for the Monte Carlo approximation
    key = jax.random.PRNGKey(0)
    
    # eta = minimize_scalarization_Likelihood(key, s5_vfield, t_, x0, Gamma, true_gamma, budget, sigma, num_samples=num_samples)
    eta_minimax = None
    # eta_minimax = minimize_scalarization_min_max_robust(H, t_, A_thetas, budget = 15, sigma = sigma, lam = lam, 
    #                                                     prior = prior, rcond = 10000e-15, repeats = True, bayes=False)
    eta_ = None    
    # eta_ = list(map(lambda th: minimize_scalarization_min_max_robust(H, t_, th.reshape(1, th.shape[0], th.shape[1]),
    #                                                                  sigma = sigma, rcond = 10000e-15, budget = 15, lam = lam, 
    #                                                                  repeats = True, prior=prior), A_thetas))
    
    eta_cont = None
    # eta_cont = minimize_scalarization_continuous_FW(H, t_, A_thetas, design = 'A', lam = lam, rcond = 10000e-15, sigma = sigma, 
    #                                                 iterations=2000, prior=None, line_search=True, lr = 1e-10)
    
    #----------------------------------------------------------------

    #---Step 5: Plot the results-------------------------------------
    plot_eta(t_, t_, t_[eta], true_x0, s5_vfield, multi_args, eta_cont = eta_cont, ind_etas = [t_[_] for _ in eta_] if eta_ is not None else None,
             design_sample = True, FWHM=0.01, eta_t_robust=t_[eta_minimax] if eta_minimax is not None else None)
    
    # plot_eta(t_, t_, t_[eta2], true_x0, s5_vfield, multi_args, eta_cont = eta_cont, ind_etas = [t_[_] for _ in eta_] if eta_ is not None else None, 
    #         design_sample = True, FWHM=0.5)
    
    plot_design_phase(t_, t_, t_[eta], true_x0, s5_vfield, multi_args, design_sample = True)
    
    multi_args_ = multi_args
    multi_args_embedded = jax.vmap(model.get_latent, in_axes=(None, 0))(true_x0 , multi_args_)
    K_mats = jax.vmap(model.get_KoopmanK, in_axes=[None, 0])(None, multi_args_)
    plot_eta(t_, t_, t_[eta], multi_args_embedded, linear_field, K_mats) #, ind_etas = [t_[_] for _ in eta_])
    #-----------------------------------------------------------------


    #---Step 6: Test greedy-------------------------------------------
    ed_t = jnp.sort(t_[eta])
    traj_map_arg = jax.vmap(model.get_latent, in_axes=(0, None))
    ed_points = solve(ed_t, true_x0, args, s5_vfield)
    
    latent_traj = jax.vmap(traj_map_arg, in_axes=[None, 0])(ed_points, multi_args)

    K_mats = jax.vmap(model.get_KoopmanK, in_axes=[None, 0])(None, multi_args)
    
    eval_points = jnp.linspace(0, 10, 1000)
    
    parameters = {'scale': 1, 
                'd': 1, 
                'm': 50, 
                'o': 2, 
    }
    
    plot_greedy_results(t_, eta, ed_points, model, multi_args, parameters, (0, 10), 100)
        
def linear_oscillator_design():
    
    #---Step 1: Generate synthetic training data---------------------
    ti = jnp.linspace(0, 10, 10)
    t_ = jnp.linspace(0, 10, 100)
    args = jnp.array([1.0]) # Assuming these are the true parameters for kcat and K_m
    # multi_args = jnp.array([(60.0, 30.4), (30, 70), (25, 45), (100, 10)])
    multi_args = jnp.linspace(0.2, 2.0, 1).reshape(-1, 1)
    true_x0 = jnp.array([2.0, 2.0]) # True initial condition
    
    jax.config.update("jax_enable_x64", True)

    A_thetas = jnp.array([[0, 1.0], [-1.0, 0]]).reshape(1, 2, 2)
    # A_thetas = jnp.array([[0, 0], [0, 0]]).reshape(1, 2, 2)

    multi_args = A_thetas
    #----------------------------------------------------------------

    #---Step 3: Fit the GP-------------------------------------------
    H = HermiteLayer(scale=1, d=1, m=50, o=1)
    #----------------------------------------------------------------

    #---Step 4: Experimental Design-----------------------------------
    t_ = jnp.linspace(0, 10, 100)
    sigma = 1.0
    lam = 0.1
    prior = 0.0
    eta = minimize_scalarization_min_max_robust(H, t_, A_thetas, budget = 10, sigma = sigma, lam = lam, prior = prior, rcond = 1e-10, repeats = True, bayes=False)
    ic(eta)

    
    t__ = jnp.linspace(0, 10, 100)  # Time points
    true_x0 = jnp.array([1.0, 2.0])  # Initial condition
    # Gamma = jnp.array([2.0, -1.0]).reshape(1, -1)

    # # true_gamma = jnp.array([-4.0, 1.0]) #jnp.array([1.5])  # True parameter value
    # budget = 30  # Budget for the minimization
    # sigma = 1.0  # Noise standard deviation
    # num_samples = 1  # Number of samples for the Monte Carlo approximation
    # key = jax.random.PRNGKey(0)
    
    # eta = minimize_nonlinear_likelihood(osc_vfield, t__, true_x0, Gamma, budget, sigma, lam = 0.01)
    # ic(eta)
    # eta2 = minimize_scalarization_Likelihood(key, osc_vfield, t_, true_x0, true_gamma.reshape(1, -1), true_gamma, budget, sigma)
    # eta2 = minimize_scalarization_Likelihood(key, osc_vfield, t__, true_x0, Gamma, true_gamma, budget, sigma, num_samples=num_samples)
    # ic(eta2)
    # eta_ = list(map(lambda th: minimize_scalarization_min_max_robust(H, t_, th.reshape(1, th.shape[0], th.shape[1]), sigma = sigma, budget = 20, lam = lam, repeats = False), A_thetas))
    
    # print(list(map(lambda a: a.shape, A_thetas)))
    eta_ = None
    # jnp.save('eta_cont', 
    
    
    eta_cont = None
    eta_cont = minimize_scalarization_continuous_FW(H, t_, A_thetas, design = 'A', lam = lam, rcond = 10e-14, sigma = sigma, iterations=10000, prior = prior, line_search=True, lr = 1e7)
    print(eta_cont)
    #----------------------------------------------------------------


    #---Step 5: Plot the results-------------------------------------
    plot_eta(t_, t_, t_[eta], true_x0, linear_field, multi_args, eta_cont = eta_cont, ind_etas = [t_[_] for _ in eta_] if eta_ is not None else None, 
             design_sample = True, FWHM=None)
    # plot_eta(t__, t__, t__[eta2], true_x0, osc_vfield, Gamma, eta_cont = eta_cont, ind_etas = [t__[_] for _ in eta_] if eta_ is not None else None, 
    #          design_sample = True, FWHM=0.001)
    # plot_design_phase(t_, t_, t_[eta], true_x0, linear_field, multi_args, design_sample = True)
    

    multi_args_ = multi_args
    # K_mats = jax.vmap(model.get_KoopmanK, in_axes=[None, 0])(None, multi_args_)
    # plot_eta(t_, t_, t_[eta], multi_args_embedded, linear_field, K_mats) #, ind_etas = [t_[_] for _ in eta_])
    #-----------------------------------------------------------------


    #---Step 6: Test greedy-------------------------------------------
    ed_t = jnp.sort(t_[eta])
    traj_map_arg = jax.vmap(model.get_latent, in_axes=(0, None))
    ed_points = solve(ed_t, true_x0, args, s5_vfield)
    
    latent_traj = jax.vmap(traj_map_arg, in_axes=[None, 0])(ed_points, multi_args)

    K_mats = jax.vmap(model.get_KoopmanK, in_axes=[None, 0])(None, multi_args)
    
    eval_points = jnp.linspace(0, 10, 1000)
    
    parameters = {'scale': 1, 
                'd': 1, 
                'm': 50, 
                'o': 2, 
    }
    
    # plot_greedy_results(t_, eta, ed_points, model, multi_args, parameters, (0, 10), 100)
        
def TMDD_design():
    # jax.config.update("jax_enable_x64", True)

    #---Step 1: Generate synthetic training data---------------------
    ti = jnp.linspace(0, 10, 10)
    t_ = jnp.linspace(0, 10, 1000)
    args =  jnp.array([9.549287 ,  0.01091326, 3.0324812 , 9.50335,    2.3925903 ])
    # multi_args = jnp.array([(60.0, 30.4), (30, 70), (25, 45), (100, 10)])
    multi_args = jax.random.uniform(key = jax.random.PRNGKey(0), shape = (100, 2), minval = jnp.array([-5.0, -5.0]), maxval=jnp.array([-0.1, -0.1]))
    multi_args = args.reshape(1,-1)
    true_x0 = jnp.array([0.0, 1500.0]) # True initial condition
    synthetic_data = solve(ti, true_x0, args, s4_vfield) # Generating synthetic data
    synthetic_data = synthetic_data + jax.random.normal(jax.random.PRNGKey(0), shape=synthetic_data.shape) * 1
    # Floor the data to zero
    synthetic_data = jnp.where(synthetic_data < 0, 0, synthetic_data)
    #----------------------------------------------------------------


    #---Step 2: Load the model---------------------------------------  
    jax.config.update("jax_enable_x64", False)
    model = load('/Users/antanas/GitRepo/NODE/Models/TMDD_0117/TMDD_2D.eqx', type='DynamicKoopman')    # model = load("/Users/antanas/GitRepo/NODE/Models/PLearnKoopman_MM_2D_Cons_K", type = 'PlearnKoopmanCL')
    jax.config.update("jax_enable_x64", True)

    # embedded_data = jax.vmap(model.get_latent, in_axes=(0, None))(synthetic_data[:, 1], args)
    # embedded_data_1 = model.get_latent_series(ti, true_x0, args)
    # A_theta_ = model.encoderP(args)
    # A_theta = model.get_K(args)
    # A_theta = model.get_naive(A_theta_)
    A_thetas = jax.vmap(lambda w: model.get_KoopmanK(None, w))(multi_args)
    ic(A_thetas.shape)
    # A_thetas = A_thetas[:4]
    # A_theta_ = A_thetas[0].reshape(1, A_theta.shape[0], A_theta.shape[1])
    #----------------------------------------------------------------

    # list(map(lambda x: ic(jnp.linalg.eigvals(x)), A_thetas))

    #---Step 3: Fit the GP-------------------------------------------
    H = HermiteLayer(scale=1, d=1, m=50, o=3)
    # H.constrained_optimization(ti, embedded_data, t_=t_, A_theta=A_theta)
    #----------------------------------------------------------------

    #---Step 4: Experimental Design-----------------------------------
    # t_ = jnp.linspace(0, 10, 100)
    sigma = 1
    lam = 0.01
    eta = minimize_scalarization_min_max_robust(H, t_, A_thetas, budget = 15, sigma = sigma, lam = lam, rcond = 10e-10, repeats = False, bayes=True)
    # eta_ = list(map(lambda th: minimize_scalarization_min_max_robust(H, t_, th.reshape(1, th.shape[0], th.shape[1]), sigma = sigma, budget = 20, lam = lam, repeats = False), A_thetas))
    
    # print(list(map(lambda a: a.shape, A_thetas)))
    eta_ = None
    ic(eta)
    # eta_ = eta_.reshape(eta_.shape[0], eta_.shape[1])
    # etas = minimize_scalarization_continuous(H, t_, A_thetas, design = 'A', lam = 0.1)
    # eta_cont = sum(etas)/len(etas)
    eta_cont = None
    #----------------------------------------------------------------
    t_ = jnp.linspace(0, 10, 1000)


    #---Step 5: Plot the results-------------------------------------
    plot_eta(t_, t_, t_[eta], true_x0, s4_vfield, multi_args, eta_cont, ind_etas = [t_[_] for _ in eta_] if eta_ is not None else None)
    # plot_eta(t_, t_, t_[eta], true_x0, s2_vfield, multi_args, eta_cont, ind_etas = [t_[_] for _ in eta_] if eta_ is not None else None)

    multi_args_ = multi_args
    multi_args_embedded = jax.vmap(model.get_latent, in_axes=(None, 0))(true_x0[1] , multi_args_)
    K_mats = jax.vmap(model.get_K, in_axes=[None, 0])(None, multi_args_)
    plot_eta(t_, t_, t_[eta], multi_args_embedded, linear_field, K_mats) #, ind_etas = [t_[_] for _ in eta_])
    #-----------------------------------------------------------------


    #---Step 6: Test greedy-------------------------------------------
    ed_t = jnp.sort(t_[eta])
    traj_map_arg = jax.vmap(model.get_latent, in_axes=(0, None))
    ed_points = solve(ed_t, true_x0, args, s3_vfield)
    
    latent_traj = jax.vmap(traj_map_arg, in_axes=[None, 0])(ed_points, multi_args)

    K_mats = jax.vmap(model.get_K, in_axes=[None, 0])(None, multi_args)
    
    eval_points = jnp.linspace(0, 10, 1000)
    
    parameters = {'scale': 1, 
                'd': 1, 
                'm': 30, 
                'o': 2, 
    }
    
    plot_greedy_results(t_, eta, ed_points, model, multi_args, parameters, (0, 10), 100)
        
def LV_Design():
    
    #---Step 1: Generate synthetic training data---------------------
    ti = jnp.linspace(0, 10, 10)
    t_ = jnp.linspace(0, 10, 100)
    args = jnp.array([1.3, 0.9, 1.6, 1.2]) 
    multi_args = jax.random.uniform(key = jax.random.PRNGKey(10), shape = (1, 4), minval = jnp.array([0.5, 0.5, 0.5, 0.5]),
                                    maxval = jnp.array([5, 5, 5, 5]))
    
    # multi_args = jnp.array([2.5, 1.2, 4.2, 2]).reshape(1, -1)
    
    # multi_args = jnp.array([(60.0, 30.4), (30, 70), (25, 45), (100, 10)])

    true_x0 = jnp.array([1.0, 1.0])  # True initial condition
    synthetic_data = solve(ti, true_x0, args, s2_vfield) # Generating synthetic data
    synthetic_data = synthetic_data + jax.random.normal(jax.random.PRNGKey(0), shape=synthetic_data.shape) * 1
    # Floor the data to zero
    synthetic_data = jnp.where(synthetic_data < 0, 0, synthetic_data)
    #----------------------------------------------------------------


    #---Step 2: Load the model---------------------------------------  
    jax.config.update("jax_enable_x64", False)
    # models = list(map(lambda x: load(x, type = 'PLearnKoop'), os.listdir("Users/antanas/GitRepo/NODE/Models/PLearnKoopman_LV_2D_Lip_10_16.eqx")))
    # model = load("/Users/antanas/GitRepo/NODE/Models/LV_10D_1e-5/PLearnKoopmanLV_10D.eqx", type = 'DynamicKoopman')
    model = load('/Users/antanas/GitRepo/NODE/Models/LV_10D_1e-4_800k/PLearnKoopman10D.eqx', type = 'DynamicKoopman')
    jax.config.update("jax_enable_x64", True)

    # embedded_data_1 = model.get_latent_series(ti, true_x0, args)
    # A_thetas = jax.vmap(lambda x: x.get_KoopmanK(None, None))(models)
    A_thetas = jax.vmap(lambda w: model.get_KoopmanK(None, w))(multi_args)
        #----------------------------------------------------------------

    #---Step 3: Fit the GP-------------------------------------------
    H = HermiteLayer(scale=1, d=1, m=50, o=10)
    #----------------------------------------------------------------

    #---Step 4: Experimental Design-----------------------------------
    t_ = jnp.linspace(0, 10, 100)
    sigma = 0.001
    lam = 0.1
    prior = 1
    eta = minimize_scalarization_min_max_robust(H, t_, A_thetas, budget = 20, sigma = sigma, prior= prior, lam = lam, rcond = 10e-13, bayes=True)
    
    eta_ = None
    print("Parameters:", multi_args)
    print("Experimental Design:", eta)
        
    eta_cont = None
    eta_cont = minimize_scalarization_continuous_FW(H, t_, A_thetas, design = 'A', lam = lam, rcond = 10e-13, sigma = sigma, iterations=20, prior=prior, line_search=True)
    #----------------------------------------------------------------

    # multi_args[-2].reshape(1, *multi_args[-1].shape)
    #---Step 5: Plot the results-------------------------------------
    # plt.plot(t_, eta)
    # plt.show()
    plot_eta(t_, t_, t_[eta], true_x0, s2_vfield, multi_args, eta_cont, ind_etas = [t_[_] for _ in eta_] if eta_ is not None else None, design_sample = False)
    multi_args_ = multi_args
    multi_args_embedded = jax.vmap(model.get_latent, in_axes=(None, 0))(true_x0 , multi_args_)
    # K_mats = jax.vmap(model.get_K, in_axes=[None, 0])(None, multi_args_)
    plot_eta(t_, t_, t_[eta], multi_args_embedded[-2].reshape(1, -1), linear_field, A_thetas[-2].reshape(1, *A_thetas[-2].shape))#, ind_etas = [t_[_] for _ in eta_])
    #-----------------------------------------------------------------


    #---Step 6: Test greedy-------------------------------------------
    # ed_t = jnp.sort(t_[eta])
    # traj_map_arg = jax.vmap(model.get_latent, in_axes=(0, None))
    # ed_points = solve(ed_t, true_x0, args, s3_vfield)
    

    # latent_traj = jax.vmap(traj_map_arg, in_axes=[None, 0])(ed_points, multi_args)

    # K_mats = jax.vmap(model.get_K, in_axes=[None, 0])(None, multi_args)
    
    # eval_points = jnp.linspace(0, 10, 1000)
    
    # parameters = {'scale': 1, 
    #             'd': 1, 
    #             'm': 50, 
    #             'o': 2, 
    # }
    
    # plot_greedy_results(t_, eta, ed_points, model, multi_args, parameters, (0, 10), 1000)
        
def Hill_Design():
    
    #---Step 1: Generate synthetic training data---------------------
    ti = jnp.linspace(0, 10, 10)
    t_ = jnp.linspace(0, 10, 100)
    args = jnp.array((60.0, 40.4, 0.8)) # Assuming these are the true parameters for kcat and K_m
    # multi_args = jnp.array([(60.0, 30.4), (30, 70), (25, 45), (100, 10)])
    # multi_args = jax.random.uniform(key = jax.random.PRNGKey(0), shape = (100, 2), minval = jnp.array([30, 30]), maxval=jnp.array([100, 100]))
    multi_args = args.reshape(1, -1)
    true_x0 = jnp.array([110.0])  # True initial condition
    #----------------------------------------------------------------


    #---Step 2: Load the model---------------------------------------  
    jax.config.update("jax_enable_x64", False)
    model = load("/Users/antanas/GitRepo/NODE/Models/Hill/Hill_2D.eqx", type = 'DynamicKoopman')
    # model = load("/Users/antanas/GitRepo/NODE/Models/PLearnKoopman_MM_2D_Cons_K", type = 'PlearnKoopmanCL')
    jax.config.update("jax_enable_x64", True)

    # embedded_data = jax.vmap(model.get_latent, in_axes=(0, None))(synthetic_data, args)
    # embedded_data_1 = model.get_latent_series(ti, true_x0, args)
    # A_theta_ = model.encoderP(args)
    # A_theta = model.get_K(args)
    # A_theta = model.get_naive(A_theta_)
    A_theta = jax.vmap(model.get_KoopmanK, in_axes=(None, 0))(None, multi_args)
    
    # A_thetas = A_thetas[:4]
    #----------------------------------------------------------------

    # list(map(lambda x: ic(jnp.linalg.eigvals(x)), A_thetas))

    #---Step 3: Fit the GP-------------------------------------------
    H = HermiteLayer(scale=1, d=1, m=50, o=2)
    # H.constrained_optimization(ti, embedded_data, t_=t_, A_theta=A_theta)
    #----------------------------------------------------------------

    #---Step 4: Experimental Design-----------------------------------
    t_ = jnp.linspace(0, 10, 100)
    sigma = 10
    lam = 0.001
    prior = 10
    eta = minimize_scalarization_min_max_robust(H, t_, A_theta, budget = 15, sigma = sigma, prior=prior, lam = lam, rcond = 10e-10, repeats = False, bayes=True)
    # eta_ = list(map(lambda th: minimize_scalarization_min_max_robust(H, t_, th.reshape(1, th.shape[0], th.shape[1]), sigma = sigma, budget = 20, lam = lam, repeats = False), A_thetas))
    
    # print(list(map(lambda a: a.shape, A_thetas)))
    eta_ = None
    ic(eta)
    # eta_ = eta_.reshape(eta_.shape[0], eta_.shape[1])
    # etas = minimize_scalarization_continuous(H, t_, A_thetas, design = 'A', lam = 0.1)
    # eta_cont = sum(etas)/len(etas)
    eta_cont = None
    #----------------------------------------------------------------


    #---Step 5: Plot the results-------------------------------------
    plot_eta(t_, t_, t_[eta], true_x0, s6_vfield, multi_args, eta_cont = eta_cont, ind_etas = [t_[_] for _ in eta_] if eta_ is not None else None)
    multi_args_ = multi_args
    multi_args_embedded = jax.vmap(model.get_latent, in_axes=(None, 0))(true_x0 , multi_args_)
    plot_eta(t_, t_, t_[eta], multi_args_embedded, linear_field, A_theta) #, ind_etas = [t_[_] for _ in eta_])
    #-----------------------------------------------------------------


    #---Step 6: Test greedy-------------------------------------------
    ed_t = jnp.sort(t_[eta])
    traj_map_arg = jax.vmap(model.get_latent, in_axes=(0, None))
    ed_points = solve(ed_t, true_x0, args, s3_vfield)
    
    latent_traj = jax.vmap(traj_map_arg, in_axes=[None, 0])(ed_points, multi_args)

    K_mats = jax.vmap(model.get_K, in_axes=[None, 0])(None, multi_args)
    
    eval_points = jnp.linspace(0, 10, 1000)
    
    parameters = {'scale': 1, 
                'd': 1, 
                'm': 30, 
                'o': 2, 
    }
    

if __name__ == "__main__":
    # PK_design()
    # MM_Design()
    # non_linear_LV_Design()
    # LV_Design()
    # S1_design()
    # VDP_design()
    linear_oscillator_design()
    # Hill_Design()

     
    

    
    
    
    