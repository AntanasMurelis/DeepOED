from DeepKoopman.ODE_Dataloader import *
from DeepKoopman.Archs import *
import hydra
import time
import equinox as eqx
from omegaconf import OmegaConf
from omegaconf import DictConfig
import wandb
from DeepKoopman.plot import *
from typing import Callable, Tuple


#--------------------------------------------------------------------------------------------------------------
# Vector fields
#--------------------------------------------------------------------------------------------------------------
args_s1 = jnp.array((-0.05, -1.))
input_s1 = jnp.array([0.1, 0.1])
def s1_vfield(t, X, args):
    x, y = X
    mu, lam = args
    return jnp.stack([mu * x, lam * (y - x**2)], axis=-1)

args_s2 = jnp.array([0, 0, 0, 0])
input_s2 = jnp.array([0.1, 0.1])
def s2_vfield(t, X, args):
    x, y = X
    alpha, beta, delta, gamma = args
    return jnp.stack([alpha * x - beta * x * y, delta * x * y - gamma * y], axis=-1) 

args_s3 = jnp.array([1, 1])
input_s3 = jnp.array([0.1])
def s3_vfield(t, X, args):
    kcat, K_m = args
    return  - kcat * X / (K_m + X)

args_s4 = jnp.array([0, 0, 0, 0, 0])
input_s4 = jnp.array([0.1])
def s4_vfield(t, X, args):
    A, L = X
    Vm, Km, k12, k21, kel = args
    ddt_L = -kel*L - Vm*L/(Km+L) - k12*L + k21*A
    ddt_A = k12*L - k21*A
    return jnp.stack([ddt_A, ddt_L], axis=-1)

args_s5 = jnp.array(1).reshape(1, -1)
input_s5 = jnp.array([0, 0])
def s5_vfield(t, x, w):
    dx = x[1] # dx/dt = y
    dy = w[0] * (1 - x[0]**2) * x[1] - x[0]  
    return jnp.stack([dx, dy], axis=-1)

# Define the system of ODEs
def glv(t, X, args):
    n = X.shape[-1]  # Determine n from the shape of X
    r = args[:n]  # Reshape the first n elements into the r vector
    A = args[n:].reshape((n, n))  # Reshape the remaining elements into the matrix A
    return X * (r + jnp.dot(A, X))

args_s6 = jnp.array([1, 1, 1])
input_s6 = jnp.array([0])
def s6_vfield(t, X, args):
    kcat, K_m, gamma = args
    return -kcat * X ** gamma / (K_m**gamma + X**gamma)

args_s7 = jnp.array([0, 0, 0])
input_s7 = jnp.array([0.1, 0.0])
def s7_vfield(t, C, args):
    C1, C2 = C[0], C[1]
    k1, Vmax, Km = args[0], args[1], args[2]    
    dC1dt = -k1 * C1
    dC2dt = k1 * C1 - (Vmax * C2) / (Km + C2)
    return jnp.stack([dC1dt, dC2dt], axis = -1)

def vfield_params(name: str, key_data, kwargs) -> Tuple[jnp.ndarray, Callable, jnp.ndarray]:
    
    time_increase = kwargs['time_increase']['total_time_increase'] * ((kwargs['data']['dsize'][0] // kwargs['data']['batch_size']) / kwargs['training']['steps'])
    threshold = (kwargs['time_series']['ts_u'] - kwargs['time_series']['ts_l'])/(kwargs['time_series']['points']) if not kwargs['time_series']['continuous'] and time_increase != 0 else None
    
    if name == 's1':
        args = args_s1
        sample_input = input_s1
        vector_field = s1_vfield
        
        data = safe_dataloader_rp_rts(**kwargs['time_series'], **kwargs['data'], args=args, vector_field=vector_field, key=key_data, time_increase=time_increase, time_threshold=threshold)
        
    elif name == 's2':
        args = args_s2
        sample_input = input_s2
        vector_field = s2_vfield
        
        data = safe_dataloader_rp_rts(**kwargs['time_series'], **kwargs['data'], args=args, vector_field=vector_field, key=key_data, time_increase=time_increase, time_threshold=threshold)

    elif name == 's3':
        args = args_s3
        sample_input = input_s3
        vector_field = s3_vfield
        
        data = safe_dataloader_rp_rts(**kwargs['time_series'], **kwargs['data'], args=args, vector_field=vector_field, key=key_data, time_increase=time_increase, time_threshold=threshold)

    elif name == 's4':
        args = args_s4
        sample_input = input_s4
        vector_field = s4_vfield
        
        data = safe_dataloader_rp_rts(**kwargs['time_series'], **kwargs['data'], args=args, vector_field=vector_field, key=key_data, time_increase=time_increase, time_threshold=threshold)

    elif name == 's5':
        args = args_s5
        sample_input = input_s5
        vector_field = s5_vfield
        
        data = safe_dataloader_rp_rts(**kwargs['time_series'], **kwargs['data'], args=args, vector_field=vector_field, key=key_data, time_increase=time_increase, time_threshold=threshold)

    elif name == 's6':
        args = args_s6
        sample_input = input_s6
        vector_field = s6_vfield
        
        data = safe_dataloader_rp_rts(**kwargs['time_series'], **kwargs['data'], args=args, vector_field=vector_field, key=key_data, time_increase=time_increase, time_threshold=threshold)

    elif name == 's7':
        args = args_s7
        sample_input = input_s7
        vector_field = s7_vfield
        
        data = safe_dataloader_rp_rts(**kwargs['time_series'], **kwargs['data'], args=args, vector_field=vector_field, key=key_data, time_increase=time_increase, time_threshold=threshold)


    elif name == 'glv':
        vector_field = glv
        sample_input = jnp.zeros(kwargs['glv']['n'])
        args = jnp.zeros(kwargs['glv']['n']**2 + kwargs['glv']['n'])
        
        data = safe_dataloader_glv(**kwargs['time_series'], **kwargs['data'], **kwargs['glv'], key=key_data, time_increase=time_increase)
        
    else:
        raise ValueError('Vector field not found')
    return args, vector_field, sample_input, data
#--------------------------------------------------------------------------------------------------------------

def callback(model, args, vector_field, trajectories, step, cfg, points = 100):
    
    # Get validation data
    ts, sols, predictions, r_args, w_pred = get_validation_data(model, args = args, trajectories = trajectories,
                        vector_field = vector_field, key=jax.random.PRNGKey(5), dmin=cfg['data']['dmin'], dmax=cfg['data']['dmax'],
                        min_p_noise=cfg['data']['min_p_noise'], max_p_noise=cfg['data']['max_p_noise'], dsize = cfg['data']['dsize'][-1],
                        ts_l = cfg['time_series']['ts_l'], ts_u = cfg['time_series']['ts_u'] + cfg['time_increase']['total_time_increase'], 
                        points = points, continuous = cfg['time_series']['continuous'], limited_exposure = cfg['data'].get('limited_exposure', None))
    
    sample_indices = jax.random.randint(jax.random.PRNGKey(0), shape=(5,), minval=0, maxval=len(sols))
    
    print("Making images...")
    
    sample_sols, sample_predictions = sols[sample_indices], predictions[sample_indices]
    
    if cfg['vector_field']['name'] == 's2' or cfg['vector_field']['name'] == 's1':
        
        fig1 = plot_state_space_diagrams(sample_predictions, sample_sols)
        wandb.log({f"State Space: step {step}": wandb.Image(fig1)})
        
        fig2 = plot_2d_trajectories(ts, sample_predictions, sample_sols)
        wandb.log({f"2D Trajectories: step {step}": wandb.Image(fig2)})
        
        fig3 = plot_3d_trajectories(ts, sample_predictions, sample_sols)
        wandb.log({f"3D Trajectories: step {step}": wandb.Image(fig3)})
        
        fig4 = plot_trajectory_sets(sample_predictions, sample_sols)
        wandb.log({f"Overlapped Trajectories: step {step}": wandb.Image(fig4)})
        
        # fig5 = model.plot_loss_landscape(cfg['data']['dmin'], cfg['data']['dmax'], args, vector_field, n=1000)
        
        fig6, error_integral = plot_error_trajectory(ts, predictions, sols)
        wandb.log({f"Error Trajectories: step {step}": wandb.Image(fig6),
            f"Absolute Error Integral: t = [{cfg['time_series']['ts_l']}, {cfg['time_series']['ts_u'] + cfg['time_increase']['total_time_increase']}]": float(error_integral)})
                
    else:
        
        fig1 = plot_time_trajectory_sets(sample_sols, sample_predictions, ts)
        wandb.log({f"Time trajectory: step {step}": wandb.Image(fig1)})
        
        fig2, error_integral = plot_error_trajectory(ts, predictions, sols)
        wandb.log({f"Error Trajectories: step {step}": wandb.Image(fig2),
                   f"Absolute Error Integral: t = [{cfg['time_series']['ts_l']}, {cfg['time_series']['ts_u'] + cfg['time_increase']['total_time_increase']}]": float(error_integral)})
        
        fig3, error_integral_p = plot_parameter_error_trajectory(ts, w_pred, r_args)
        wandb.log({f"Parameter Error Trajectories: step {step}": wandb.Image(fig3),
            f"Parameter Absolute Error Integral: t = [{cfg['time_series']['ts_l']}, {cfg['time_series']['ts_u'] + cfg['time_increase']['total_time_increase']}]": float(error_integral_p)})
        
    return 
#--------------------------------------------------------------------------------------------------------------

def get_validation_data(model, args, trajectories, vector_field, key, dmin, dmax, min_p_noise, max_p_noise, dsize, ts_l, ts_u, points, continuous = True, limited_exposure = None):
    from DeepKoopman.ODE_Dataloader import _get_data_rargs
    
    ts = jnp.linspace(ts_l, ts_u, points) if continuous else jnp.linspace(ts_l, ts_u, points, endpoint=False)
    
    p_lower = args + jnp.array(min_p_noise)
    p_upper = args + jnp.array(max_p_noise)

    key, X0_key = jax.random.split(key, 2)
    r_args = jax.random.uniform(key, (trajectories, len(args)), minval= p_lower, maxval=p_upper)
    # r_args = jnp.linspace(p_lower, p_upper, trajectories)
    # r_args = generate_grid([1, 1],[100, 100], 3)
    
    X0 = jax.random.uniform(X0_key, shape=jnp.array([trajectories, dsize]), minval=jnp.array(dmin), maxval=jnp.array(dmax))
    
    ts, sols = _get_data_rargs(ts, X0, r_args, vector_field=vector_field)
    
    if limited_exposure is not None:
        # sols = jnp.take(sols, jnp.array(limited_exposure), axis=-1)
        X0 = jnp.take(X0, jnp.array(limited_exposure), axis=-1)

    predictions, wi, latent  = jax.vmap(model, in_axes=[None, 0, 0])(ts, X0, r_args)
        
    return ts, sols, predictions, r_args, wi

#--------------------------------------------------------------------------------------------------------------
def train_model(model, data, steps=10000, save_every=1000, 
                learning_rate=1e-2, key=jax.random.PRNGKey(0), 
                args=None, vector_field=None, mode='vanilla',
                model_name = "model", hyperparams = None, 
                callback=None, config = None, logging = True):
   
    # optim = opt.adam(learning_rate)
    optim = hydra.utils.instantiate(config['optimizer'])
    
    opt_state = optim.init(eqx.filter(model, filter_spec=eqx.is_array))

    if logging:
        
        #-----------------------------------------------------------------------------------------
        # WandB logging
        #-----------------------------------------------------------------------------------------
        wandb.login()
        # Initialize Weights and Biases
        wandb.init(
        **config['logging'], config = OmegaConf.to_container(config, resolve=True)
        )
    else:
        #-----------------------------------------------------------------------------------------
        # Testing WandB logging
        #-----------------------------------------------------------------------------------------
        wandb.init(mode='disabled')
    

    #-----------------------------------------------------------------------------------------
    # Higher tolerance Solvers:
    #-----------------------------------------------------------------------------------------
    # jax.config.update("jax_enable_x64", True)
    #-----------------------------------------------------------------------------------------
    
    print("Training model...")
    start_solve = time.time()
    #--Main Training Loop-----------------------------------------------------------------------
    for step, (ts, ys_i, wi) in zip(range(steps), data):
        
        start = time.time()
        solving_time = start - start_solve

        values, model, opt_state = model.make_step(ts, ys_i, wi, optim, opt_state)
        end = time.time()
        
        #--Log metrics with wandb-----------------------------------------------------------------
        if ((step % save_every) == 0 or step == steps - 1):
            
            metrics_to_log = {
                "Step": step,
                "Computation Time": end - start,
                "Solving Time": solving_time
            }
            
            metrics_to_log.update(values[1])
            wandb.log(metrics_to_log)

        start_solve = time.time()
    
    #--Save and Plots---------------------------------------------------------------------------
    save(model_name + '.eqx', hyperparams = hyperparams, model = model)
    
    if config['time_series'].get('continuous', True):
        callback(model, args, vector_field, 5000, step, config)
    else:
        callback(model, args, vector_field, 5000, step, config, points = int(config['time_series']['points'] + config['time_series']['points']/(config['time_series']['ts_u'] -  config['time_series']['ts_l']) * config['time_increase']['total_time_increase']))
    
    wandb.run.log_code("~/NODE", include_fn=lambda path: path.endswith(".py"))
    
    wandb.finish()
    
    return model
#---------------------------------------------------------------------------------------------


def make_arc_arrays(input_d, w_d, mid_d, latent_d, en, dn, pn, params_encoding=False, IC_K=False, only_IC_K = False, **kwargs):
    
    if params_encoding:
        en_i_d = input_d + w_d
        de_o_d = en_i_d
    else:
        en_i_d = input_d
        de_o_d = en_i_d + w_d
    
    if IC_K:
        pen_d = w_d + input_d
    else:
        pen_d = w_d
        
    if only_IC_K:
        pen_d = input_d
        
    d_array = [en_i_d] + [mid_d] * (en) + [latent_d]
    b_array = [latent_d] + [mid_d] * (dn) + [de_o_d]
    w_array = [pen_d] + [mid_d] * (pn) + [latent_d ** 2]
    
    return {'d_array': d_array, 
            'b_array': b_array, 
            'w_array': w_array}
    

#---------------------------------------------------------------------------------------------
@hydra.main(config_path="conf", config_name="MM")
def main(cfg: DictConfig):
    
    print(OmegaConf.to_yaml(cfg))
    
    cfg_native = cfg #OmegaConf.to_container(cfg, resolve=True)

    key = jax.random.PRNGKey(cfg_native['random']['seed'])
    key_data, key_model = jax.random.split(key, 2)

    params, vfield, sample_input, data = vfield_params(cfg_native['vector_field']['name'], key_data, cfg_native)
    
    model_hyperparameters = make_arc_arrays(input_d=sample_input.shape[-1], w_d=params.shape[-1], latent_d=cfg_native['Model']['Hyperparameters']['latent_dim'],
                                            params_encoding=cfg_native['Model']['Hyperparameters']['encoder']['include_w'],
                                            IC_K=cfg_native['Model']['Hyperparameters']['propagator']['include_IC'],
                                            only_IC_K=cfg_native['Model']['Hyperparameters']['propagator']['only_IC'],
                                            **cfg_native['architecture'])
        
    # Update the configuration with the generated architecture parameters
    if cfg.Model.Hyperparameters.encoder.params.layers is None:
        cfg.Model.Hyperparameters.encoder.params.layers = model_hyperparameters['d_array']
    if cfg.Model.Hyperparameters.decoder.params.layers is None:
        cfg.Model.Hyperparameters.decoder.params.layers = model_hyperparameters['b_array']
    if 'propagator' in cfg.Model.Hyperparameters and cfg.Model.Hyperparameters.propagator.encoder.params.layers is None:
        cfg.Model.Hyperparameters.propagator.encoder.params.layers = model_hyperparameters['w_array']

    model = hydra.utils.instantiate(OmegaConf.to_object(cfg_native['Model']['Architecture']), OmegaConf.to_object(cfg_native['Model']['Hyperparameters']), key=key_model)

    train_model(model=model, data=data, **cfg_native['training'], vector_field=vfield, 
                hyperparams=OmegaConf.to_object(cfg_native['Model']['Hyperparameters']), 
                key=key_model, args=params, callback=callback, config=cfg_native, logging=True)
#--------------------------------------------------------------------------------- 


#---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
