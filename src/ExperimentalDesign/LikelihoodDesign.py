#------------------------------------------------------------------------------------------------
# The non-linear Experimental Designs based on Fisher Information
#------------------------------------------------------------------------------------------------

import jax
import equinox as eqx
import jax.numpy as jnp
from icecream import ic

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns  # For a nicer color palette

from utils.Solvers import solve, hess_solve
from utils.Tests import *
from ExperimentalDesign.MakeDesigns import plot_eta


eqx.filter_jit()
def Likelihood_gamma(key, gamma, solver, t_old: jnp.ndarray, t_new, y_old, y_new, sigma):

    if t_old.size > 0:
        times = jnp.append(t_old, t_new)
        t_idx = jnp.argsort(times)
        y_total = jnp.concatenate([y_old, y_new])[t_idx]
        times = times[t_idx]
    else:
        times = t_new
        y_total = y_new

    y_pred = solver(times, gamma)
    return 1.0 / sigma * jnp.sum((y_pred - y_total)**2)

def minimize_scalarization_Likelihood(key, field, t_, x0, Gamma, true_gamma, budget = 10, sigma = None, num_samples = 1, lam = None):
    
    solver = lambda t, gamma: hess_solve(t, x0, gamma, field)
    y_old_list, t_old_list, eta = jnp.zeros_like(Gamma), [], []
    
    for i in range(budget):
        
        key, subkey = jax.random.split(key)
        
        fun = lambda g, t_n, y_n, y_old: Likelihood_gamma(subkey, g, solver, jnp.array(t_old_list), t_n.reshape(-1), y_old, y_n.reshape(1, -1), sigma)
        
        map = jax.vmap(fun, in_axes = [None, 0, 0, None])
        
        true_trajectories = jax.vmap(solve, in_axes = [None, None, 0, None])(t_, x0, Gamma, field)
        true_trajectories = true_trajectories + jax.random.normal(subkey, true_trajectories.shape) * sigma
        
        # Generate Hessians over noise realizations
        hessians = jax.hessian(map)
        hessians_t = jax.vmap(hessians, in_axes = [0, None, 0, 0])(Gamma, t_, true_trajectories, y_old_list)
        
        if hessians_t.ndim == 2:
            hessians_t = hessians_t.reshape(hessians_t.shape[0], hessians_t.shape[1], 1, 1)
        
        def scalarise_hessian(hessian):
            return jnp.trace(jnp.linalg.inv(hessian + 0.1 * jnp.eye(hessian.shape[0])))
            
        map2 = jax.jit(jax.vmap(scalarise_hessian))
        scalarised_hessians = jax.vmap(map2)(hessians_t)
        
        mean_v_gamma_new = jnp.mean(scalarised_hessians, axis = 0)
        print(mean_v_gamma_new)
        
        eta_new = jnp.argmin(mean_v_gamma_new)
        new_t = t_[eta_new]
        new_y = true_trajectories[:,eta_new]
        
        eta.append(eta_new)
        
        if i == 0:
            y_old_list = jnp.array(new_y)
            y_old_list = y_old_list.reshape(y_old_list.shape[0], 1, y_old_list.shape[1])
        else:
            new_y_reshaped = new_y.reshape(new_y.shape[0], 1, new_y.shape[1])
            y_old_list = jnp.concatenate([y_old_list, new_y_reshaped], axis=1)
            
        t_old_list.append(new_t)
        
    return jnp.array(eta)

def minimize_nonlinear_likelihood(field, t_, x0, Gamma, budget = 10, sigma = 1, lam = None):
        
        solver = lambda gamma, t: solve(t.reshape(-1), x0, gamma, field)
        get_jacobian = jax.jacobian(solver)
        
        # plt.plot(jnp.linspace(0, 1000, 1000), solver(Gamma[0], t_))
        
        def get_information_matrix(gamma, t):
            J = get_jacobian(gamma, t)[0]
            return 1/sigma**2 * J.T @ J
        
        map_time = jax.vmap(get_information_matrix, in_axes=[None, 0])
        map_time_gamma = jax.vmap(map_time, in_axes=[0, None])
        
        Info_matrices = map_time_gamma(Gamma, t_)
        
        eta = []
        
        scalarisation = lambda Info: jnp.trace(jnp.linalg.inv(Info + jnp.eye(Info.shape[0]) * lam))
        scalarisation = lambda Info: jnp.linalg.det(Info + jnp.eye(Info.shape[0]) * lam)
        
        def evaluate_scalarisation(Info, info_i):
            return scalarisation(Info + info_i)
        
        Info = jnp.zeros([Gamma.shape[0], Info_matrices.shape[-1], Info_matrices.shape[-2]])
        
        for i in range(budget):
            
            map_scalarisation_time = jax.vmap(evaluate_scalarisation, in_axes=[None, 0])
            map_scalarisation_gamma = jax.vmap(map_scalarisation_time, in_axes=[0, 0])
            
            scalarisations = map_scalarisation_gamma(Info, Info_matrices)
            # plt.plot(scalarisations[0])
            scalarisation_mean = jnp.mean(scalarisations, axis = 0)
            plt.plot(scalarisation_mean)
            # plt.plot(scalarisation_mean)
            # scalarisation_mean = jnp.where(jnp.isnan(scalarisation_mean), jnp.inf, scalarisation_mean)
            # scalarisation_mean = scalarisation_mean.at[0].set(jnp.inf)
            # ic(scalarisation_mean)  
            eta_new = jnp.argmax(scalarisation_mean)
            
            Info = Info + Info_matrices[:, eta_new]
            
            eta.append(eta_new)
            
        return jnp.array(eta)   

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

def minimize_scalarization_continuous_non_linear(field, x0, t_, Gamma, sigma = None, design = 'A', lam = None, 
                                         eta = None, iterations=100, prior = None, lr=0.1):
    
    """
    Minimize the scalarization function to find the optimal design.
    """
    
    
    solver = lambda gamma, t: solve(t.reshape(-1), x0, gamma, field)
    get_jacobian = jax.jacobian(solver)
    
    # plt.plot(jnp.linspace(0, 1000, 1000), solver(Gamma[0], t_))
    
    def get_information_matrix(gamma, t):
        J = get_jacobian(gamma, t)[0]
        return 1/sigma**2 * J.T @ J
    
    map_time = jax.vmap(get_information_matrix, in_axes=[None, 0])
    map_time_gamma = jax.vmap(map_time, in_axes=[0, None])
    
    Info_matrices = map_time_gamma(Gamma, t_)
    
    # scalarisation = lambda Info: jnp.trace(jnp.linalg.inv(Info + jnp.eye(Info.shape[0]) * lam))    
    
    def Info(eta, Phi_i):
        return eta * Phi_i
    
    def A_obj(eta, Phi_t, lam, prior=None):
        Info_ = jnp.sum(jax.vmap(Info, in_axes = [0, 0])(eta, Phi_t), axis = 0)
        Info_ = Info_ + prior if prior is not None else Info_
        A = 1/sigma**2 * jnp.trace( jax.numpy.linalg.pinv(Info_ + lam * jnp.eye(len(Info_))))
        return A
    
    sum_C = lambda eta: jnp.sum(jax.vmap(A_obj, in_axes = [None, 0, None, None])(eta, Info_matrices, lam, prior), axis = -1)
    
    objective = lambda eta: sum_C(eta)
    objective = jax.jit(objective)
    
    initial_distribution = jnp.ones(len(t_))/len(t_)
    
    if design == 'A':
        # eta = Frank_Wolfe(objective, initial_distribution, iterations=iterations, line_search=line_search, tol=1e-5)
        eta = Exponentiated_Gradient_Descent(objective, initial_distribution, iterations=iterations, learning_rate=lr)
        return eta

    return eta  
    

def test_minimize_scalarization_Likelihood():

    # Define test parameters
    jax.config.update("jax_enable_x64", True)
    
    t_ = jnp.linspace(0, 10, 1000)  # Time points    
    x0 = jnp.array([3.5, 0.6])  # Initial condition
    
    # Gamma = jnp.linspace(0, 2, 10)  # Parameter range
    true_gamma = jnp.array([1.5])  # True parameter value
    # Gamma = jnp.array([1.4]).reshape(1, -1) # Gamma parameter
    Gamma = jnp.linspace(5.0, 5.0, 1).reshape(-1, 1) # Gamma parameter
    budget = 5  # Budget for the minimization
    sigma = 1.0  # Noise standard deviation
    key = jax.random.PRNGKey(0)
    eta = minimize_nonlinear_likelihood(osc_vfield2, t_, x0, Gamma, budget, sigma, lam = 0.000001)
    ic(eta)
    
    eta_cont = minimize_scalarization_continuous_non_linear(osc_vfield2, x0, t_, Gamma, design = 'A', lam = 0.000001, sigma = sigma, iterations=20000, prior = None, lr = 1e1)
    ic(eta_cont)
    
    plot_eta(t_, t_, t_[eta], x0, osc_vfield2, Gamma, eta_cont=eta_cont, FWHM=None, exclude_designs=True)
    # eta = minimize_scalarization_Likelihood(key, osc_vfield2, t_, x0, true_gamma.reshape(1, -1), true_gamma, budget, sigma)
    # ic(eta)

        

if __name__ == "__main__":  
    
    test_minimize_scalarization_Likelihood()
        