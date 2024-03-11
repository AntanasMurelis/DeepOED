from jax import numpy as jnp
import jax
import diffrax
import equinox as eqx
from utils.Solvers import solve
import optax
import wandb

# A function that solves the system of ODEs
def solve(ts, X0, args, vector_field):
    solution = diffrax.diffeqsolve(diffrax.ODETerm(vector_field), 
                                #    diffrax.ImplicitEuler(),
                                #    diffrax.Kvaerno5(), # Implicit Solver
                                   diffrax.Tsit5(), 
                                #    diffrax.Dopri5(),
                                   t0=0.0, 
                                   t1=ts[-1]+0.0001, 
                                   dt0=0.00001, 
                                   y0=X0, 
                                   args=args, 
                                   saveat=diffrax.SaveAt(ts=ts),
                                #    stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-8),
                                   stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-8),
                                   max_steps=None
                                   )
    return solution.ys


@eqx.filter_jit  
def safe_solve(ts, X0, args, vector_field):
    solution = diffrax.diffeqsolve(diffrax.ODETerm(vector_field), 
                                   diffrax.Tsit5(), 
                                #    diffrax.Dopri5(),
                                #    diffrax.Kvaerno5(),
                                   t0=0.0, 
                                   t1=ts[-1]+0.0001, 
                                   dt0=0.00001, 
                                   y0=X0, 
                                   args=args, 
                                   max_steps=4000,
                                   saveat=diffrax.SaveAt(ts=ts),
                                #    stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-8),
                                   stepsize_controller=diffrax.PIDController(rtol = 1e-4, atol = 1e-6),
                                    throw=False
                                   )
    return solution.ts, solution.ys, solution.result

#------------------------------------------------------------



@eqx.filter_jit  
def _get_data(ts, X0, args, vector_field):
    
    solutions = jax.vmap(solve, in_axes=(None, 0, None, None))(ts, X0, args, vector_field)
    
    return ts, solutions

@eqx.filter_jit
def get_data(ts, args, dsize, dmin, dmax, vector_field, key):
    # 1. Get initial conditions
    key, X0_key = jax.random.split(key, 2)
    X0 = jax.random.uniform(X0_key, dsize, minval=dmin, maxval=dmax)
    # 2. Solve the system of ODEs using vmap
    ts, sols = _get_data(ts, X0, args, vector_field)
    return ts, sols


def dataloader(ts, args, key, vector_field, dmin, dmax, dsize=(1000, 2), batch_size=1):
    while True:
        key, d_key, p_key = jax.random.split(key, 3)
        ts, ys = get_data(ts, args, dsize, dmin, dmax, vector_field, d_key)
        indices = jnp.arange(ys.shape[0])
        indices = jax.random.shuffle(p_key, indices)
        start = 0
        end = batch_size
        while start < dsize[0]:
            perm_indices = indices[start:end]
            yield ts, ys[perm_indices], None
            start = end
            end = end + batch_size


# Random Parameter set-up
@eqx.filter_jit
def _get_data_rargs(ts, X0, args, vector_field):
    solutions = jax.vmap(solve, in_axes=(None, 0, 0, None))(ts, X0, args, vector_field)
    return ts, solutions

# Random Parameter set-up
@eqx.filter_jit
def _get_data_rargs_rts(ts, X0, args, vector_field):
    tsi, ys, result = jax.vmap(solve, in_axes=(0, 0, 0, None))(ts, X0, args, vector_field)
    return tsi, ys, result

# Random Parameter set-up
@eqx.filter_jit
def _get_data_rargs_rts_safe(ts, X0, args, vector_field):
    solutions = jax.vmap(safe_solve, in_axes=(0, 0, 0, None))(ts, X0, args, vector_field)
    return ts, solutions

@eqx.filter_jit
def get_data_rargs_rts(ts_l, ts_u, points, args, dsize, vector_field, dmin, dmax, min_p_noise, max_p_noise, key, **kwargs):
    # 1. Get initial conditions
    key, X0_key, ts_key = jax.random.split(key, 3)
    ts = jax.random.uniform(ts_key, (dsize[0], points-1), minval=ts_l, maxval=ts_u)
    zeros = jax.numpy.zeros((dsize[0], 1))
    ts = jax.numpy.concatenate((zeros, ts), axis=1)
    ts = jax.lax.sort(ts)
    r_args = jax.random.uniform(key, (dsize[0], len(args)), minval= 0, maxval=1) * (jnp.array(max_p_noise) - jnp.array(min_p_noise)) + (args - jnp.array(min_p_noise))
    X0 = jax.random.uniform(X0_key, dsize, minval=dmin, maxval=dmax)
    
    # 2. Solve the system of ODEs using vmap
    ts, sols = _get_data_rargs_rts(ts, X0, r_args, vector_field)

    return ts, sols, r_args

def dataloader_rp_rts(ts_l, ts_u, points, args, key, vector_field, dmin, dmax, min_p_noise, max_p_noise, 
                      time_increase=None, dsize=jnp.array((1024, 2)), batch_size=1):
    while True:
        if time_increase is not None:
            ts_u += time_increase # Increase the time horizon by the specified amount
        key, d_key, p_key = jax.random.split(key, 3)
        ts, ys, r_args = get_data_rargs_rts(ts_l, ts_u, points, args, dsize, vector_field, dmin, dmax, min_p_noise, max_p_noise, d_key)
        indices = jnp.arange(ys.shape[0])
        indices = jax.random.shuffle(p_key, indices)
        start = 0
        end = batch_size
        while start < dsize[0]:
            perm_indices = indices[start:end]
            yield ts[perm_indices], ys[perm_indices], r_args[perm_indices]
            start = end
            end = end + batch_size
            

@eqx.filter_jit
def get_rpoint_data(args, dsize, dmin, dmax, noise, key):
    key_r, key_d = jax.random.split(key, 2)
    r_args = jax.random.uniform(key_r, (dsize[0], len(args)), minval= noise[0], maxval=noise[1]) + args
    X0 = jax.random.uniform(key_d, dsize, minval=dmin, maxval=dmax)
    return None, X0, r_args
    
def dataloader_rp_js(key, args, vector_field, dmin, dmax, min_noise, max_noise, dsize=jnp.array((1000, 2)), batch_size=1):
    while True:        
        key, d_key, p_key = jax.random.split(key, 3)
        ts, ys, r_args = get_rpoint_data(args, dsize, dmin, dmax, min_noise, max_noise, d_key)
        indices = jnp.arange(ys.shape[0])
        indices = jax.random.shuffle(p_key, indices)
        start = 0
        end = batch_size
        while start < dsize[0]:
            perm_indices = indices[start:end]
            yield None, ys[perm_indices], r_args[perm_indices]
            start = end
            end = end + batch_size

@eqx.filter_jit
def get_data_rargs(ts,args, dsize, vector_field, dmin, dmax, min_p_noise, max_p_noise, key):
    # 1. Get initial conditions
    key, X0_key = jax.random.split(key, 2)
    r_args = jax.random.uniform(key, (dsize[0], len(args)), minval= 0, maxval=1) * (jnp.array(max_p_noise) - jnp.array(min_p_noise)) + (args - jnp.array(min_p_noise))
    X0 = jax.random.uniform(X0_key, dsize, minval=dmin, maxval=dmax)
    # 2. Solve the system of ODEs using vmap
    ts, sols = _get_data_rargs(ts, X0, r_args, vector_field)
    return ts, sols, r_args

def dataloader_rp(ts_l, ts_u, points, args, key, vector_field, dmin, dmax, min_p_noise, max_p_noise, dsize=jnp.array((1000, 2)), batch_size=1):
    ts = jnp.linspace(ts_l, ts_u, points)
    while True:
        key, d_key, p_key = jax.random.split(key, 3)
        ts_u, ys, r_args = get_data_rargs(ts, args, dsize, vector_field, dmin, dmax, min_p_noise, max_p_noise, d_key)
        ts_a = jnp.tile(ts_u, (dsize[0], 1))
        indices = jnp.arange(ys.shape[0])
        indices = jax.random.shuffle(p_key, indices)
        start = 0
        end = batch_size
        while start < dsize[0]:
            perm_indices = indices[start:end]
            yield ts_a[perm_indices], ys[perm_indices], r_args[perm_indices]
            start = end
            end = end + batch_size
       
       
#------------------------------------------------------------
# Safe set-up
#------------------------------------------------------------     

# Random Parameter set-up
@eqx.filter_jit
def _get_safe_data_rargs_rts(ts, X0, args, vector_field):
    tsi, ys, results = jax.vmap(safe_solve, in_axes=(0, 0, 0, None))(ts, X0, args, vector_field)
    return tsi, ys, results        
    
@eqx.filter_jit
def get_safe_data_rargs_rts(ts_l, ts_u, points, args, dsize, vector_field, dmin, dmax, min_p_noise, max_p_noise, key):
    # 1. Get initial conditions
    key, X0_key, ts_key = jax.random.split(key, 3)
    ts = jax.random.uniform(ts_key, (dsize[0], points-1), minval=ts_l, maxval=ts_u)
    zeros = jax.numpy.zeros((dsize[0], 1))
    ts = jax.numpy.concatenate((zeros, ts), axis=1)
    ts = jax.lax.sort(ts)
    r_args = jax.random.uniform(key, (dsize[0], len(args)), minval= 0, maxval=1) * (jnp.array(max_p_noise) - jnp.array(min_p_noise)) + jnp.array(min_p_noise)
    X0 = jax.random.uniform(X0_key, dsize, minval=0, maxval=1) * (jnp.array(dmax) - jnp.array(dmin)) + jnp.array(dmin)
    
    # 2. Solve the system of ODEs using vmap
    ts, sols, results = _get_safe_data_rargs_rts(ts, X0, r_args, vector_field)
    return ts, sols, r_args, results

@eqx.filter_jit
def get_safe_data_rargs_ts(ts_l, ts_u, points, args, dsize, vector_field, dmin, dmax, min_p_noise, max_p_noise, key):
    # 1. Get initial conditions
    key, X0_key = jax.random.split(key, 2)
    ts = jnp.linspace(start= ts_l, stop = ts_u, num=points, endpoint=False)
    ts = jnp.tile(ts, (dsize[0], 1))
    r_args = jax.random.uniform(key, (dsize[0], len(args)), minval= 0, maxval=1) * (jnp.array(max_p_noise) - jnp.array(min_p_noise)) + jnp.array(min_p_noise)
    X0 = jax.random.uniform(X0_key, dsize, minval=0, maxval=1) * (jnp.array(dmax) - jnp.array(dmin)) + jnp.array(dmin)
    
    # 2. Solve the system of ODEs using vmap
    ts, sols, results = _get_safe_data_rargs_rts(ts, X0, r_args, vector_field)
    return ts, sols, r_args, results


def filter_successful_solves(ts, ys, r_args, results):
    
    if jnp.all(results == 0):
        return ts, ys, r_args
        
    else:
        success_mask = results == 0
        
        # Find indices of failed simulations
        failure_indices = ~success_mask
        
        # Logging failed parameters and initial conditions
        failed_params = r_args[failure_indices]
        failed_initial_conditions = ys[failure_indices, 0, :]
    
        for params, init_cond in zip(failed_params, failed_initial_conditions):
            print(f"Failed parameters: {params}, Failed initial conditions: {init_cond}")
        
        # Return data corresponding to successful simulations
        return ts[success_mask], ys[success_mask], r_args[success_mask]


def safe_dataloader_rp_rts(ts_l, ts_u, points, args, key, vector_field, dmin, dmax, min_p_noise, max_p_noise, 
                      time_increase=None, dsize=jnp.array((1024, 2)), batch_size=1, limited_exposure = None, continuous = True, time_threshold=None, **kwargs):
    cumulative_time_increase = 0
    while True:
        # Increase the time horizon if specified, different logic for continuous and discrete cases
        if time_increase is not None:
            cumulative_time_increase += time_increase

            if continuous:
                ts_u += time_increase
            elif time_threshold is not None and cumulative_time_increase >= time_threshold:
                ts_u += time_threshold  # Update ts_u only when the threshold is reached
                points += 1  # Increase points
                cumulative_time_increase = cumulative_time_increase - time_threshold  # Reset cumulative time increase
                wandb.log({'points': points})
                wandb.log({'ts_u': ts_u})

        key, d_key, p_key = jax.random.split(key, 3)
        
        if continuous:
            ts, ys, r_args, results = get_safe_data_rargs_rts(ts_l, ts_u, points, args, dsize, vector_field, dmin, dmax, min_p_noise, max_p_noise, d_key)
        else:
            ts, ys, r_args, results = get_safe_data_rargs_ts(ts_l, ts_u, points, args, dsize, vector_field, dmin, dmax, min_p_noise, max_p_noise, d_key)
            
        ts, ys, r_args = filter_successful_solves(ts, ys, r_args, results)
        
        if limited_exposure is not None:
            ys = jnp.take(ys, jnp.array(limited_exposure), axis=-1)
        
        indices = jnp.arange(ys.shape[0])
        indices = jax.random.permutation(p_key, indices, independent = True)
        start = 0
        end = batch_size
        
        while start < len(indices):
            perm_indices = indices[start:end]
            yield ts[perm_indices], ys[perm_indices], r_args[perm_indices]
            start = end
            end = end + batch_size


#------------------------------------------------------------
# GLV set-up
#------------------------------------------------------------

eqx.filter_jit
def build_Allesina_Tang_normal(n, C, sigma, rho, d_min, d_max, key):
    key1, key2, key3 = jax.random.split(key, 3)

    # Sample coefficients in pairs
    Sigma = sigma**2 * jnp.array([[1, rho], [rho, 1]])
    mu = jnp.zeros(2)
    pairs = jax.random.multivariate_normal(key1, mean=mu, cov=Sigma, shape=(n * (n - 1) // 2,))

    # Build a completely filled matrix
    triu_indices = jnp.triu_indices(n, 1)
    # For the upper triangular part
    M_upper = jnp.zeros((n, n))
    M_upper = M_upper.at[triu_indices].set(pairs[:, 0])
    
    # For the lower triangular part
    M_lower = jnp.zeros((n, n))
    M_lower = M_lower.at[triu_indices[::-1]].set(pairs[:, 1])  # Reverse the order for lower triangular
    
    # Summing both parts to get the full matrix
    M = M_upper + M_lower
    
    # Determine which connections to retain
    random_vals = jax.random.uniform(key2, (n, n))
    Connections = jnp.triu((random_vals <= C).astype(jnp.int32), 1)  # Connections in upper triangle
    Connections = Connections + Connections.T  # Symmetric connections

    # Apply connections to M
    M = M * Connections
    # Randomly generate diagonal values
    d_values = jax.random.uniform(key3, (n,), minval=d_min, maxval=d_max)
    
    # Set the diagonal elements
    M = M.at[jnp.diag_indices(n)].set(-d_values)

    return M

@eqx.filter_jit  
def generate_glv_system(n, C, d_min, d_max, sigma, rho, r_min, r_max, key):

    subkey, r_key = jax.random.split(key, 2)  # Generate new subkeys for each matrix, for d, and for r
    
    # Generate a random r vector
    r = jax.random.uniform(r_key, (n,), minval=r_min, maxval=r_max)
    
    # Generate the interaction matrix A
    A = build_Allesina_Tang_normal(n, C, sigma, rho, d_min, d_max, subkey)

    # Stack r and A together into a single array
    r_and_A = jnp.concatenate([r.ravel(), A.ravel()], axis=0)
    
    return r_and_A

@eqx.filter_jit
def get_data_rargs_glv(ts_l, ts_u, points, n, C, diag_min, diag_max, sigma, rho, r_min,
                       r_max, d_min, d_max, dsize, key):
    
    # 1. Get initial conditions
    key, X0_key = jax.random.split(key, 2)
    r_args_keys = jax.random.split(key, dsize[0])
    
    ts = jnp.linspace(ts_l, ts_u, points)  # Assuming you want equally spaced time points
    
    r_args = jax.vmap(generate_glv_system, in_axes=(None, None, None, None, None, None, None, None, 0))(
        n, C, diag_min, diag_max, sigma, rho, r_min, r_max, r_args_keys)
    
    X0 = jax.random.uniform(X0_key, (dsize[0], n), minval=d_min, maxval=d_max)  # Assuming min and max values for initial conditions
    
    # 2. Solve the system of ODEs using vmap
    tsi, sols, _ = _get_data_rargs_glv(ts, X0, r_args)
    
    return tsi, sols, r_args, _

@jax.jit
def _get_data_rargs_glv(ts, X0, r_and_A):

    # Define the system of ODEs
    def glv(t, X, args):
        n = X.shape[-1]  # Determine n from the shape of X
        r = args[:n]  # Reshape the first n elements into the r vector
        A = args[n:].reshape((n, n))  # Reshape the remaining elements into the matrix A
        
        return X * (r + jnp.dot(A, X))
        
    tsi, solutions, _ = jax.vmap(safe_solve, in_axes=(None, 0, 0, None))(ts, X0, r_and_A, glv)
    
    return tsi, solutions, _

def dataloader_glv(ts_l, ts_u, points, n, C, diag_min, diag_max, sigma, rho, r_min, r_max, 
                   d_min, d_max, key, time_increase=None, dsize=jnp.array((1024, 2)), 
                   batch_size=1, **kwargs):
    while True:
        if time_increase is not None:
            ts_u += time_increase  # Increase the time horizon by the specified amount
        
        # Split the keys for data generation and shuffling
        key, d_key, p_key = jax.random.split(key, 3)
        
        # Generate GLV system data
        ts, ys, r_args, results = get_data_rargs_glv(ts_l, ts_u, points, n, C, diag_min, diag_max, 
                                             sigma, rho, r_min, r_max, d_min, d_max, dsize, d_key)
        
        # Create an array of indices and shuffle them
        indices = jnp.arange(ys.shape[0])
        indices = jax.random.shuffle(p_key, indices)
        
        # Yield data in batches
        start = 0
        end = batch_size
        while start < dsize[0]:
            perm_indices = indices[start:end]
            yield ts[perm_indices], ys[perm_indices], r_args[perm_indices]
            start = end
            end = end + batch_size
            
def safe_dataloader_glv(ts_l, ts_u, points, n, C, diag_min, diag_max, sigma, rho, r_min, r_max, 
                   d_min, d_max, key, time_increase=None, dsize=jnp.array((1024, 2)), 
                   batch_size=1, **kwargs):
    while True:
        if time_increase is not None:
            ts_u += time_increase  # Increase the time horizon by the specified amount
        
        # Split the keys for data generation and shuffling
        key, d_key, p_key = jax.random.split(key, 3)
        
        # Generate GLV system data
        ts, ys, r_args, results = get_data_rargs_glv(ts_l, ts_u, points, n, C, diag_min, diag_max, 
                                             sigma, rho, r_min, r_max, d_min, d_max, dsize, d_key)
                
        ts, ys, r_args = filter_successful_solves(ts, ys, r_args, results)
        
        # Create an array of indices and shuffle them
        indices = jnp.arange(ys.shape[0])
        indices = jax.random.shuffle(p_key, indices)
        
        # Yield data in batches
        start = 0
        end = batch_size
        while start < len(indices):
            perm_indices = indices[start:end]
            yield ts[perm_indices], ys[perm_indices], r_args[perm_indices]
            start = end
            end = end + batch_size