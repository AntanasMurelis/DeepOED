import jax
import jax.numpy as jnp
import jaxopt
import optax
import diffrax
from utils.Solvers import solve, safe_solve
import equinox as eqx
import optax
from DeepKoopman.Archs import load
from DeepKoopman.ODE_Dataloader import generate_glv_system
from matplotlib import pyplot as plt
from typing import Callable, Union, Any, Tuple, Optional

from utils.Tests import *
from utils.Solvers import solve
from HermiteEmbedding import ConstrainedHermiteLayer


import time

def optimize(
    xi: jnp.ndarray, 
    objective_fun: Callable,
    optimizer_type: str,
    lr: float,
    **optimizer_kwargs: Any
) -> Tuple[jnp.ndarray, float]:

    # Initialize optimizer based on user choice
    if optimizer_type.lower() == 'bfgs':
        optimizer = jaxopt.LBFGS(fun=objective_fun, **optimizer_kwargs)
    elif optimizer_type.lower() == 'adam':
        opt = optax.adam(lr)
        optimizer = jaxopt.OptaxSolver(fun=objective_fun, opt=opt, **optimizer_kwargs)
    elif optimizer_type.lower() == 'gd':
        optimizer = jaxopt.GradientDescent(fun=objective_fun, **optimizer_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    xi_final, state_final = optimizer.run(init_params=xi)
    return xi_final, objective_fun(xi_final)


@eqx.filter_jit()
def loss_solve_w_x0(args_x_0, ti, vector_field, x):
    args, x0 = args_x_0[0], args_x_0[1]
    xi = solve(ti, x0, args, vector_field)
    return jnp.mean((xi - x)**2)

@eqx.filter_jit()
def loss_solve_w(x0, w, ti, vector_field, x):
    _, xi, results = safe_solve(ti, x0, w, vector_field)
    # return jnp.mean((xi - x)**2) 
    return jax.lax.cond(jnp.any(results == diffrax.RESULTS.successful), 
                        lambda x: jnp.mean((xi - x)**2), lambda x: jnp.inf, x)

@eqx.filter_jit()
def loss_model_w_x0(x0, ti, model, x):
    d = x.shape[-1]
    x_, w = x0[:d], x0[d:]
    latent_array = model.get_latent_series(ti, x_, w)
    latent_x = jax.vmap(model.get_latent, in_axes=[0, None])(x, w)
    return jnp.mean(jnp.abs(latent_array - latent_x))

@eqx.filter_jit()
def loss_model_w(x0, w, ti, model, x):
    latent_array = model.get_latent_traj(ti, x0, w)
    latent_x = jax.vmap(model.get_latent, in_axes=[0, None])(x, w)
    return jnp.mean(jnp.abs(latent_array - latent_x))

@eqx.filter_jit()
def loss_Hermite_regression(constraint, ti, hermite_model, x):
    hermite_model.constrained_optimization(ti, x, constraint)
    x_pred = jax.vmap(hermite_model.mu)(ti)
    return jnp.mean((x_pred - x)**2)

@eqx.filter_jit()
def loss_nl_Hermite_regression(w, ti, hermite_model: ConstrainedHermiteLayer, model, x):
    
    embedded_data = jax.vmap(model.get_latent, in_axes=(0, None))(x, w)
    constraint = model.get_K(None, w)
    hermite_model.constrained_optimization(ti, embedded_data, constraint)
    x_pred = jax.vmap(hermite_model.mu)(ti)
    
    return jnp.mean((x_pred - embedded_data)**2)


def test_Hermite_opt():
    
    jax.config.update("jax_enable_x64", False)
    model = load("/Users/antanas/GitRepo/NODE/Models/PLearnKoopman_MM_1D_Lip_10_16.eqx", type = 'PlearnKoopmanLip')
    # model = load("/Users/antanas/GitRepo/NODE/Models/PLearnKoopman_MM_2D_Cons_K", type = 'PlearnKoopmanCL')
    jax.config.update("jax_enable_x64", True)


    ed = jnp.array([ 0, 22,  5,  1, 43, 17,  6,  4, 15, 64])#, 33,  7, 14,  2, 16,  8, 13,
                    #  3, 85, 51, 12, 34,  9, 23, 18, 32, 11, 24, 35, 50, 69, 98, 19, 31,
                    #  25, 52, 36, 21, 78, 30, 20, 37, 53, 10, 65, 49, 26, 38, 29, 39])
    
    # ed = None
    
    t_range = (0, 10)
    n_points = 100
    true_args = jnp.array((60.0, 30.4))
    true_x0 = jnp.array([110.0])
    noise_std_dev = 0.0
    key = jax.random.PRNGKey(0)

    t, synthetic_data, noisy_synthetic_data = generate_synthetic_data(
        t_range, n_points, true_args, true_x0, noise_std_dev, key, s3_vfield, ed=ed
    )
    
    parameters = {'scale': 1, 
            'd': 1, 
            'm': 50, 
            'o': 2, 
    }
    
    hermite_model = ConstrainedHermiteLayer(**parameters, t_ = jnp.linspace(0, 10, 100))
    objective_fun = lambda w: loss_nl_Hermite_regression(w, t, hermite_model, model, noisy_synthetic_data)
    
    w0 = jnp.array([50.0, 50.0])
    optimizer_kwargs = {'tol': 1e-8, 
                    'maxiter': 1000}
    
    w_pred, _ = optimize(w0, objective_fun, optimizer_type = 'adam',
                         lr = 0.1, **optimizer_kwargs)
    
    print("Prediciton: ", w_pred)
    
    return

def test_oscillator():
    
    jax.config.update("jax_enable_x64", True)
    ed = jnp.array([16, 36, 54, 73, 92, 99,  3, 26, 63, 47])
    ed1 = jnp.array([9, 24, 40, 56, 72, 99, 9, 24, 40, 56])
    # ed = jnp.array([ 25,  80, 129, 177, 226, 274, 322, 370, 417, 465, 512, 560, 608])
                    # 656, 704, 752, 800, 848, 896, 944, 992, 197, 246, 301, 355, 409,
                    # 460, 514, 567, 620, 673, 728, 782, 830, 882, 143, 690, 745, 633,
                    # 576, 523, 472, 417, 363, 312, 261, 194, 384, 439, 325, 495, 547,
                    # 602, 652, 700, 782, 827, 737, 583, 631, 529, 470, 681, 424, 372,
                    # 251, 300, 494, 558, 437, 613, 366, 493, 549, 662, 413, 601, 717,
                    # 777, 319, 456, 514, 564, 371, 258, 207, 102, 151,  44, 316, 267,
                    # 216, 168,  99, 416, 364, 471, 302, 655, 609], dtype = jnp.int32)
    # ed = jnp.linspace(0, 200, 10, dtype = jnp.int32)
    
    t_range = (0, 10)
    n_points = 100
    true_args = jnp.array([1.0])
    true_x0 = jnp.array([2.0, 4.0])
    noise_std_dev = 2.0
    key = jax.random.PRNGKey(1)
    
    t, synthetic_data, noisy_synthetic_data = generate_synthetic_data(
        t_range, n_points, true_args, true_x0, noise_std_dev, key, osc_vfield2, ed=ed
    )
    
    print("Time:", t)
    
    # plt.plot(t, noisy_synthetic_data[:, 0], label = 'x')
    # plt.plot(t, noisy_synthetic_data[:, 1], label = 'y')
    # plt.plot(t, synthetic_data[:, 0], label = 'x')
    # plt.plot(t, synthetic_data[:, 1], label = 'y')
    # plt.legend()
    # plt.show()
    
    
    objective_fun = lambda w: loss_solve_w(true_x0, w, t, osc_vfield2, noisy_synthetic_data)
    w0 = jnp.array([0.5])
    
    optimizer_kwargs = {'tol': 1e-10,    
                        'maxiter': 200000}
    
    w_pred, _ = optimize(w0, objective_fun, optimizer_type = 'adam',
                        lr = 0.001, **optimizer_kwargs)
    
    print("MSE: ", _)


    print("Prediciton: ", w_pred)
    print("True: ", true_args)
    print("Error: ", jnp.linalg.norm(w_pred - true_args))
    
def test_MM():
    
    jax.config.update("jax_enable_x64", False)
    # model = load("/Users/antanas/GitRepo/NODE/Models/PLearnKoopman_MM_1D_Lip_10_16.eqx", type='PlearnKoopmanLip')
    model = load('/Users/antanas/GitRepo/NODE/Models/MM/MM_2D_0208.eqx', type = 'DynamicKoopman')
    jax.config.update("jax_enable_x64", True)

    
    ed = jnp.array([ 0, 22,  5,  1, 43, 17,  6,  4, 15, 64, 33,  7, 14,  2, 16,  8, 13,
                3, 85, 51, 12, 34,  9, 23, 18, 32, 11, 24, 35, 50, 69, 98, 19, 31,
            25, 52, 36, 21, 78, 30, 20, 37, 53, 10, 65, 49, 26, 38, 29, 39])
    
    # ed = None
    t_range = (0, 10)
    n_points = 100
    true_args = jnp.array((70.0, 45.0))
    true_x0 = jnp.array([110.0])
    noise_std_dev = 5.0
    key = jax.random.PRNGKey(1)
    vector_field = s3_vfield

    t, synthetic_data, noisy_synthetic_data = generate_synthetic_data(
        t_range, n_points, true_args, true_x0, noise_std_dev, key, vector_field, ed=ed
    )
    
    objective_fun = lambda w: loss_model_w(true_x0, w, t, model, noisy_synthetic_data)
    
    w0 = jnp.array([100.0, 100.0])
    
    optimizer_kwargs = {'tol': 1e-10,    
                        'maxiter': 15000}
    
    w_pred, _ = optimize(w0, objective_fun, optimizer_type = 'adam',
                        lr = 0.1, **optimizer_kwargs)
    


    print(f"Optimized x0 for model: {w_pred}")
    # print(f"Loss for model: {l_model}")
    

    

def test_VDP():
    
    jax.config.update("jax_enable_x64", True)
    t_range = [0, 10]
    n_points = 100

    true_args = jnp.array([1.5]) # Assuming these are the true parameters for kcat and K_m

    true_x0 = jnp.array([0.5, 2.0]) # True initial condition
    noise_std_dev = 0.5
    key = jax.random.PRNGKey(0)
    vector_field = s5_vfield
    
    ed = jnp.array([25, 72, 99, 49, 87, 11, 37, 61, 99,  6, 85, 22, 35, 93, 73, 58, 99,
                47,  6, 16])

    
    t, synthetic_data, noisy_synthetic_data = generate_synthetic_data(
        t_range, n_points, true_args, true_x0, noise_std_dev, key, vector_field, ed=ed
    )
    
    objective_fun = lambda w: loss_solve_w(true_x0, w, t, vector_field, noisy_synthetic_data)
    
    w0 = jnp.array([0.2])
    
    optimizer_kwargs = {'tol': 1e-10,    
                        'maxiter': 150000}
    
    w_pred, _ = optimize(w0, objective_fun, optimizer_type = 'adam',
                        lr = 0.01, **optimizer_kwargs)
    

    print("Prediciton: ", w_pred)
    print("True: ", true_args)
    print("Error: ", jnp.linalg.norm(w_pred - true_args))
    
if __name__ == "__main__":
    test_oscillator()
    # test_VDP()
    # test_MM()



