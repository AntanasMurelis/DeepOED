import jax
from jax import numpy as jnp
from .Solvers import solve, safe_solve

#--------------------------------------------------------------------------------------------------------------
# Vector fields
#--------------------------------------------------------------------------------------------------------------
args_s1 = jnp.array((-0.05, -1.))
def s1_vfield(t, X, args):
    x, y = X
    mu, lam = args
    return jnp.stack([mu * x, lam * (y - x**2)], axis=-1)

args_s2 = jnp.array([0.66, 1.33, 1, 1])
def s2_vfield(t, X, args):
    x, y = X
    alpha, beta, delta, gamma = args
    return jnp.stack([alpha * x - beta * x * y, delta * x * y - gamma * y], axis=-1) 

args_s3 = jnp.array([1, 1])
def s3_vfield(t, X, args):
    kcat, K_m = args
    return -kcat * X / (K_m + X)



args_s4 = jnp.array([1, 1, 1, 1, 1])
def s4_vfield(t, X, args):
    A, L = X
    Vm, Km, k12, k21, kel = args
    ddt_L = -kel*L - Vm*L/(Km+L) - k12*L + k21*A
    ddt_A = k12*L - k21*A
    return jnp.stack([ddt_A, ddt_L], axis=-1)


args_s5 = jnp.array(1)
input_s5 = jnp.array([0, 0])
def s5_vfield(t, x, w):
    X, Y = x[0], x[1]
    dx = x[1] # dx/dt = y
    dy = w[0] * (1 - X**2) * Y - X  
    return jnp.stack([dx, dy], axis=-1)

def osc_vfield(t, x, w):
    dx = w[1] * x[1]
    dy = w[0] * x[0]
    return jnp.stack([dx, dy], axis=-1)

def osc_vfield2(t, x, w):
    dx = x[1]
    dy = w[0] * x[0]
    return jnp.stack([dx, dy], axis=-1)


args_s6 = jnp.array([1, 1, 1])
input_s6 = jnp.array([0])
def s6_vfield(t, X, args):
    kcat, K_m, gamma = args
    return -kcat * X / (K_m ** gamma + X)
#--------------------------------------------------------------------------------------------------------------


#--Synthetic data generation------------------------------------------------------------------------------------
def generate_synthetic_data(t_range, n_points, true_args, true_x0, noise_std_dev, key, vector_field, ed=None, noise_percentage = None):
    """
    Generate synthetic data with optional experimental design.

    Parameters:
    - t_range: Tuple indicating the range for time points (start, end).
    - n_points: Number of time points.
    - true_args: True arguments for the vector field.
    - true_x0: True initial condition.
    - noise_std_dev: Standard deviation of noise.
    - key: JAX random key for noise generation.
    - vector_field: Function for the vector field.
    - ed: List of indices for the experimental design (optional).
    """

    # Generate regular time points
    t = jnp.linspace(*t_range, n_points)
    # Generate synthetic data
    synthetic_data = solve(t, true_x0, true_args, vector_field)
    
    # Function to add proportional noise
    def add_proportional_noise(data):
        noise_scale = noise_percentage / 100 * data
        return data + noise_std_dev * jax.random.normal(key, data.shape) * noise_scale

    if ed is not None:
        # Select only the time points and data specified in the experimental design
        t_ed = t[ed]
        synthetic_data_ed = synthetic_data[ed]

        # Add independent noise for each time point in the experimental design
        noisy_synthetic_data_ed = synthetic_data_ed + noise_std_dev * jax.random.normal(key, synthetic_data_ed.shape) if noise_percentage is None else add_proportional_noise(synthetic_data_ed)
        # noisy_synthetic_data_ed = jnp.where(noisy_synthetic_data_ed < 0, 0, noisy_synthetic_data_ed)
        
        noisy_synthetic_data_ed = noisy_synthetic_data_ed[jnp.argsort(t_ed)]
        synthetic_data_ed = synthetic_data_ed[jnp.argsort(t_ed)]
        t_ed = t_ed[jnp.argsort(t_ed)]
        
        return t_ed, synthetic_data_ed, noisy_synthetic_data_ed
    else:
        # Add noise to all data points if no experimental design is provided
        noisy_synthetic_data = synthetic_data + noise_std_dev * jax.random.normal(key, synthetic_data.shape) if noise_percentage is None else add_proportional_noise(synthetic_data)
        # noisy_synthetic_data = jnp.where(noisy_synthetic_data < 0, 0, noisy_synthetic_data)

        return t, synthetic_data, noisy_synthetic_data
#--------------------------------------------------------------------------------------------------------------