import jax

from jax import numpy as jnp
from jax import random as jrandom
import equinox as eqx
import equinox.nn as nn
from DeepKoopman.ODE_Dataloader import *
from DeepKoopman.Modules import *
import matplotlib.pyplot as plt
import json
from DeepKoopman.DynamicKoopman import PLearnKoop

#--------------------------------------------------------------------------------------------------------------
# Kernel Encoding:
#--------------------------------------------------------------------------------------------------------------

# Fourier feature mapping
def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2.*jnp.pi*x) @ B.T
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

def gen_B_gauss(key, d, B_out = None, scale = 1):
    if B_out is None:
        return None
    else:
        return jrandom.normal(key, (B_out, d)) * scale

#--------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------- 
# Linear Neural ODE solver: Using Koopman Operator
#-------------------------------------------------------------------------------------------------------------- 
class MLP(eqx.Module):
    d_array: list[int]
    layers: list[nn.Linear]
    n_layers:list[nn.LayerNorm]

    def __init__(self, d_array, key, layer_norm = False):
        keys = jax.random.split(key, len(d_array))
        self.d_array = d_array
        input_d_array = d_array[:-1]
        output_d_array = d_array[1:]
        self.layers = [nn.Linear(in_d, out_d, key = keys[i]) for i,
                       (in_d, out_d) in enumerate(zip(input_d_array, output_d_array))]
        
        if layer_norm: self.n_layers = [nn.LayerNorm(d) for d in output_d_array[:-1]]
        else: self.n_layers = None
        
    
    def __call__(self, x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = jax.nn.relu(x)
            if self.n_layers is not None: x = self.n_layers[i](x)
            
        x = self.layers[-1](x)
        
        return jnp.array(x)

#--------------------------------------------------------------------------------------------------------------
# LipMLP - Lip
#--------------------------------------------------------------------------------------------------------------
   
class LipMLP(eqx.Module):
    params_net: list[list[jnp.ndarray]]
    use_bias: bool
    n_layers: list[nn.LayerNorm]
    
    def __init__(self, d_array, key, use_bias=True, layer_norm=False):
    
        def init_W_b(size_out, size_in, key):
            wkey, bkey = jrandom.split(key, 2)
            lim = jnp.sqrt(2 / size_in)
            W = jrandom.uniform(wkey, (size_out, size_in), minval=-lim, maxval=lim)
            if use_bias:
                b = jrandom.uniform(bkey, (size_out,), minval=-lim, maxval=lim)
            else:
                b = None
            return jnp.array(W), b

        keys = jax.random.split(key, len(d_array))
        input_d_array = d_array[:-1]
        output_d_array = d_array[1:]
        self.params_net = []
        for i, (in_d, out_d) in enumerate(zip(input_d_array, output_d_array)):
            W, b = init_W_b(out_d, in_d, keys[i])
            c = jnp.max(jnp.sum(jnp.abs(W), axis=1))
            self.params_net.append([W, b, c])
            
        self.use_bias = use_bias
        if layer_norm: self.n_layers = [nn.LayerNorm(d) for d in output_d_array[:-1]]
        else: self.n_layers = None
        
    def weight_normalization(self, W, softplus_c):
        """
        Lipschitz weight normalization based on the L-infinity norm
        """
        absrowsum = jnp.sum(jnp.abs(W), axis=1)
        scale = jnp.minimum(1.0, softplus_c/absrowsum)
        return W * scale[:,None]

    def __call__(self, x):
        """
        Forward pass of a lipschitz MLP

        Inputs
        x: a query location in the space

        Outputs
        out: implicit function value at x
        """
        # forward pass
        for ii in range(len(self.params_net) - 1):
            W, b, c = self.params_net[ii]
            W = self.weight_normalization(W, jax.nn.softplus(c))
            x = jax.nn.relu(jnp.dot(W, x) + (b if self.use_bias is not None else 0))
            if self.n_layers is not None:
                x = self.n_layers[ii](x)  # Apply LayerNorm if enabled

        # final layer
        W, b, c = self.params_net[-1]
        W = self.weight_normalization(W, jax.nn.softplus(c))
        out = jnp.dot(W, x) + (b if self.use_bias is not None else 0)
        return out
    
    def get_lipschitz_loss(self):
        """
        This function computes the Lipschitz regularization
        """
        loss_lip = 1.0
        for ii in range(len(self.params_net)):
            W, b, c = self.params_net[ii]
            loss_lip = loss_lip * jax.nn.softplus(c)
        return loss_lip

    def normalize_params(self, params_net):
        """
        (Optional) After training, this function will clip network [W, b] based on learned lipschitz constants. Thus, one can use normal MLP forward pass during test time, which is a little bit faster.
        """
        params_final = []    
        for ii in range(len(params_net)):
            W, b, c = params_net[ii]
            W = self.weight_normalization(W, jax.nn.softplus(c))
            params_final.append([W, b])
        return params_final


    def forward_eval_single(self, params_final, x):
        """
        (Optional) this is a standard forward pass of a mlp. This is useful to speed up the performance during test time 
        """
        # forward pass
        for ii in range(len(params_final) - 1):
            W, b = params_final[ii]
            x = jax.nn.relu(jnp.dot(W, x) + b)
        W, b = params_final[-1]  # final layer
        out = jnp.dot(W, x) + b
        return out


class FourierEncoder(eqx.Module):
    B: jax.Array
     
    def __init__(self, d, key, scale = 1, B_out = None):        
        if B_out is None:
            self.B = None
        else:
            self.B = jrandom.normal(key, (B_out//2, d)) * scale

    def __call__(self, x):
        if self.B is None:
            return x
        else:
            x_proj = (2.*jnp.pi*x) @ self.B.T
            return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)



class SShootODESolver(eqx.Module):
    encoder: MLP
    decoder: MLP
    Koopman: eqx.nn.Linear
    inter_calls: int

    # Single Shooting ODE solver. Predicts trajectory in the latent space before mapping to 
    # original space.
    
    def __init__(self, d_array, inter_calls, key):
        
        self.inter_calls = inter_calls
        d_key, e_key, k_key = jax.random.split(key, 3)

        self.encoder = MLP(d_array, e_key)
        self.decoder = MLP(d_array[::-1], d_key)
        self.Koopman = eqx.nn.Linear(d_array[-1], d_array[-1], key = k_key, use_bias = False)
        
    def get_latent(self, x):
        return self.encoder(x)
    
    def from_latent(self, latent):
        
        def Koopman_layer(carry, hidden = None):
            return self.Koopman(carry), carry
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = self.inter_calls)
        
        predictions_m = jax.vmap(self.decoder)(latent_array)
        
        return predictions_m, latent_array
        
    def __call__(self, x):
        
        latent = self.encoder(x)
        
        def Koopman_layer(carry, hidden = None):
            return self.Koopman(carry), carry
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = self.inter_calls)
        
        predictions_m = jax.vmap(self.decoder)(latent_array)
        
        return predictions_m, latent_array

    #----------------------------------------------------------------------------------------------------------------
    # Loss and Gradients
    #----------------------------------------------------------------------------------------------------------------
    
    @eqx.filter_jit
    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return sum(jnp.vdot(x, x) for x in leaves)

    @eqx.filter_jit  
    def loss_fn(self, x):
        
        predictions, latent_arrays = self(x[0])
        latent_exp_arrays = jax.vmap(self.get_latent)(x[1:len(latent_arrays)])

        reconstruction_error = jnp.mean((predictions[0] - x[0])**2)
        prediction_error = jnp.mean((predictions[1:] - x[1:len(latent_arrays)])**2)
            
        latent_error = jnp.mean((latent_arrays[1:] - latent_exp_arrays)**2)
        
        l2 = self.l2_squared()
        
        return prediction_error + reconstruction_error + latent_error + 10**-8 * l2

    @eqx.filter_value_and_grad
    def batch_loss(self, x):
        assert len(x.shape) == 3
        return jnp.mean(jax.vmap(self.loss_fn)(x))
        
    @eqx.filter_jit
    def make_step(self, x, optim, opt_state):
        loss, grads = self.batch_loss(x)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
      
      
class MShootODESolver(eqx.Module):
    encoder: MLP
    decoder: MLP
    Koopman: eqx.nn.Linear
    inter_calls: int

    # The architecture should encode, pass through the linear layer, and decode
    
    def __init__(self, d_array, inter_calls, key):
        
        self.inter_calls = inter_calls
        d_key, e_key, k_key = jax.random.split(key, 3)

        self.encoder = MLP(d_array, e_key)
        self.decoder = MLP(d_array[::-1], d_key)
        self.Koopman = eqx.nn.Linear(d_array[-1], d_array[-1], key = k_key, use_bias = False)
        
    def get_latent(self, x):
        return self.encoder(x)
    
    def vector_field(self, t, X, args = None):
        return self.Koopman(X)
    
    def from_latent(self, latent):
            
        latent_ =  self.Koopman(latent)
        prediction = self.decoder(latent)
        
        return latent, latent_, prediction
    
    def __call__(self, x):
        
        latent = self.encoder(x)

        latent_array = self.Koopman(latent)
        reconstruction = self.decoder(latent)
        prediction = self.decoder(latent_array)
        
        return latent, latent_array, reconstruction, prediction

    #----------------------------------------------------------------------------------------------------------------
    # Loss and Gradients
    #----------------------------------------------------------------------------------------------------------------
    

    @eqx.filter_jit
    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return sum(jnp.vdot(x, x) for x in leaves)

    @eqx.filter_jit  
    def loss_fn(self, x):
        latent, latent_, reconstruction, predictions = jax.vmap(self)(x)
        reconstruction_error = jnp.mean((reconstruction - x)**2)
        prediction_error = jnp.mean((predictions[:-1] - x[1:])**2)
        latent_error = jnp.mean((latent[1:] - latent_[:-1])**2)
        return reconstruction_error + latent_error + prediction_error

    @eqx.filter_value_and_grad
    def batch_loss(self, x):
        assert len(x.shape) == 3
        return jnp.mean(jax.vmap(self.loss_fn, in_axes = (None, 0))(x))

    
    @eqx.filter_jit
    def make_step(self, x, optim, opt_state):
        loss, grads = self.batch_loss(model, x)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return loss, model, opt_state

  
    
#--------------------------------------------------------------------------------------------------------------
# Non-Linear Neural ODE solver: Using GRUnit instead of Koopman layer
#--------------------------------------------------------------------------------------------------------------
class NNLODESolver(eqx.Module):
    encoder: MLP
    decoder: MLP
    RNN: MLP
    inter_calls: int

    # The architecture should encode, pass through the linear layer, and decode
    
    def __init__(self, d_array, inter_calls, key):
        
        self.inter_calls = inter_calls
        
        d_key, e_key, k_key = jax.random.split(key, 3)

        self.encoder = MLP(d_array, e_key)
        self.decoder = MLP(d_array[::-1], d_key)

        self.RNN = eqx.nn.Linear(d_array[-1], d_array[-1], key = k_key, use_bias = True)
        
    def __call__(self, x):
        latent = self.encoder(x)
        latent = jax.nn.tanh(latent)
        def RNN(carry, hidden = None):
            return jax.nn.tanh(self.RNN(carry)), carry
        _, latent_array =  jax.lax.scan(RNN, latent, xs = None, length = self.inter_calls)

        predictions_m = jax.vmap(self.decoder)(latent_array)
        
        return predictions_m, latent_array
    
    

#--------------------------------------------------------------------------------------------------------------
# Jacobian Shooting Neural ODE solver: Using Jacobian in loss function
#--------------------------------------------------------------------------------------------------------------
class JShootODESolver(eqx.Module):
    encoder: MLP
    decoder: MLP
    Koopman: eqx.nn.Linear

    # The architecture should encode, pass through the linear layer, and decode
    # Using the Jacobian in loss function. Needs to implement variable output.
    
    
    def __init__(self, d_array, key):
        """JShootODESolver: Uses the jacobian in the loss function. 

        Args:
            d_array (array): Describes the symmetric structrue of the coder/decoder.
            key (KeyArray): Random key.
        """
        d_key, e_key, k_key = jax.random.split(key, 3)

        self.encoder = MLP(d_array, e_key)
        self.decoder = MLP(d_array[::-1], d_key)
        self.Koopman = eqx.nn.Linear(d_array[-1], d_array[-1], key = k_key, use_bias = False)
        
    def get_latent(self, x):
        return self.encoder(x)
    
    def vector_field(self, t, X, args = None):
        return self.Koopman(X)
    
    def from_latent(self, latent):
            
        latent_ =  self.Koopman(latent)
        prediction = self.decoder(latent)
        
        return latent, latent_, prediction
    
    def __call__(self, x, inter_calls = 1):
        
        latent = self.encoder(x)
        latent_array = self.Koopman(latent)
        predictions = self.decoder(latent)
        
        return latent, latent_array, predictions
    #----------------------------------------------------------------------------------------------------------------
    
    
    #----------------------------------------------------------------------------------------------------------------
    # Loss and Gradients
    #----------------------------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return sum(jnp.vdot(x, x) for x in leaves)

    @eqx.filter_jit
    def jkb_loss_fn(self, x, args, vector_field):

        x_dot = jax.vmap(vector_field, (None, 0, None))(None, x, args)

        def f(model, x, x_dot):
            return jax.jvp(model, (x,), (x_dot,))
        
        latent, dFdt  = jax.vmap(f, in_axes = (None, 0, 0))(self.get_latent, x, x_dot)
        latent, latent_, predictions = jax.vmap(self.from_latent)(latent)
        j_loss = jnp.mean((dFdt - latent_)**2)
        reconstruction_error = jnp.mean((predictions - x)**2)

        l2 = self.l2_squared(self)

        return reconstruction_error + j_loss + 10**-9 * l2

    @eqx.filter_value_and_grad
    def jkb_batch_loss(self, x, args, vector_field):
        assert len(x.shape) == 3
        return jnp.mean(jax.vmap(self.jkb_loss_fn, in_axes = (0, None, None))(x, args, vector_field))
    
    @eqx.filter_jit
    def make_step(self, x, args, vector_field, optim, opt_state):
        loss, grads = self.jkb_batch_loss(x, args, vector_field)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return loss, model, opt_state
    #----------------------------------------------------------------------------------------------------------------


class LatentODESolver(eqx.Module):
    encoder: MLP
    decoder: MLP
    Lie: eqx.nn.Linear
    inter_calls: int

    # The architecture should encode, pass through the linear layer, and decode
    
    def __init__(self, d_array, inter_calls, key):
        
        self.inter_calls = inter_calls
        d_key, e_key, k_key = jax.random.split(key, 3)

        self.encoder = MLP(d_array, e_key)
        self.decoder = MLP(d_array[::-1], d_key)
        self.Lie = eqx.nn.Linear(d_array[-1], d_array[-1], key = k_key, use_bias = False)
        
    def get_latent(self, x):
        return self.encoder(x)
    
    def vector_field(self, t, X, args = None):
        return self.Lie(X)
    
    def from_latent(self, latent):
            
        latent_ =  self.Lie(latent)
        prediction = self.decoder(latent)
        
        return latent, latent_, prediction
    
    def __call__(self, ts, x0):
        
        latent = self.encoder(x0)
        
        latent_ = diffrax.diffeqsolve(
            diffrax.ODETerm(self.vector_field),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=latent,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )

        prediction = jax.vmap(self.decoder)(latent_.ys)

        return latent_, prediction

    #----------------------------------------------------------------------------------------------------------------
    # Loss and Gradients
    #----------------------------------------------------------------------------------------------------------------
    @eqx.filter_value_and_grad
    def grad_loss(self, ti, yi):
        latent, y_pred = jax.vmap(self, in_axes=(None, 0))(ti, yi[:, 0])
        return jnp.mean((yi - y_pred) ** 2)

    @eqx.filter_jit
    def make_step(self, ti, yi, optim, opt_state):
        loss, grads = self.grad_loss(ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return loss, model, opt_state
    #----------------------------------------------------------------------------------------------------------------
        
class ExpL(eqx.Module):
    weight: jax.Array
    
    def __init__(self, d, key):
        self.weight = jax.random.normal(key, (d, d))
    
    @eqx.filter_jit
    def __call__(self, t, x):
        return jax.scipy.linalg.expm(self.weight * t) @ x
    
    def get_weight(self):
        return self.weight
    
# Biased Koopman:

class StableKoopman(eqx.Module):
    diag: jax.Array
    off_diag: jax.Array
    
    def __init__(self, d, key):
        self.diag = jax.random.normal(key, (d,))
        self.off_diag = jax.random.normal(key, (d-1,))
    
    def Koopman(self):
        return jnp.diag(-jnp.square(self.diag)) - jnp.diag(self.off_diag, k=1) + jnp.diag(self.off_diag, k=-1)

    @eqx.filter_jit
    def __call__(self, t, x):
        Koopman = self.Koopman()
        return jax.scipy.linalg.expm(Koopman * t) @ x
        
class LinearLatentODESolver(eqx.Module):
    encoder: MLP
    decoder: MLP
    ExpL: eqx.Module

    # The architecture should encode, pass through the linear layer, and decode
    
    def __init__(self, d_array, key):
        
        d_key, e_key, k_key = jax.random.split(key, 3)

        self.encoder = MLP(d_array, e_key)
        self.decoder = MLP(d_array[::-1], d_key)
        self.ExpL = StableKoopman(d_array[-1], key = k_key)
        
    def get_latent(self, x):
        return self.encoder(x)
    
    def vector_field(self, t, X, args = None):
        return self.ExpL.get_weight() @ X
    
    
    def __call__(self, ts, x0):
        
        latent = self.encoder(x0)
        
        latent_ = jax.vmap(self.ExpL, in_axes = [0, None])(ts, latent)
        prediction = jax.vmap(self.decoder)(latent_)

        return latent_, prediction


    @eqx.filter_jit
    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return sum(jnp.vdot(x, x) for x in leaves)

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        latent, y_pred = jax.vmap(model, in_axes=[None, 0])(ti, yi[:, 0])
        return jnp.mean((yi - y_pred) ** 2)

    @eqx.filter_jit
    def make_step(self, ti, yi, optim, opt_state):
        loss, grads = self.grad_loss(ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return loss, model, opt_state
    

class PLearnKoopmanCE(eqx.Module):
    encoderD: MLP
    encoderP: MLP
    decoder: MLP
    latent_dim: int
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, b_array, w_array,key, **kwargs):
        
#        assert b_array[-1] + w_array[0] == d_array[0]
        
        d_key, e_key, k_key = jax.random.split(key, 3)
        
        self.latent_dim = d_array[-1]
        
        self.encoderD = MLP(d_array, e_key, layer_norm = True)
        self.decoder = MLP(b_array, d_key, layer_norm = True)
        self.encoderP = MLP(w_array, key = k_key, layer_norm = True)
        
    def get_latent(self, x, wi):
        return self.encoderD(x)
    
    def get_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)
    
    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_diagonal(self, w):
        return jnp.diag(w)
    
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        latent = self.encoderD(x0)
        weights = self.encoderP(w)
        koop = self.get_naive(weights)
        
        '''
        # Discrete Koopman:
        def Koopman_layer(carry, hidden = None):
            return self.propagate_fixed_st(None, carry, koop), carry
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)
        '''

        # Continuous Koopman:
        latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, koop)
        predictions = jax.vmap(self.decoder)(latent_array)

        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]


    def l2_loss(self, w, alpha):
        return alpha * jnp.mean(w**2)

    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return jnp.mean(jnp.array([jnp.vdot(x, x) for x in leaves]))
    
    @eqx.filter_jit
    def loss_fn(self, ti, yi, wi, vector_field):
        
        # Reconstruction
        latent, y_pred, w_pred = self(ti, yi[0], wi)
        tile = jnp.tile(wi, (yi.shape[0],  1))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        y_w_pred = jnp.concatenate((y_pred, w_pred), axis = -1)
        # reconstruction = jnp.mean(jnp.mean((yi - y_pred)**2, axis = -1))
        reconstruction = jnp.mean(jnp.mean((yi_w - y_w_pred)**2, axis = -1))

        # reconstruction = jnp.mean(jnp.mean(jnp.abs(yi - y_pred), axis = -1))
        # reconstruction = jnp.mean(jnp.mean((jnp.linalg.norm(yi - y_pred, ord = 1, axis = -1)/jnp.linalg.norm(yi, ord=1, axis= -1)), axis = -1))
        
        # Latent Linearity Error:
        latent_traj = jax.vmap(self.get_latent, in_axes=[0, None])(yi, wi)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent - latent_traj), axis = -1))
        # latent_error = jnp.mean(jnp.mean((latent - latent_traj)**2, axis = -1))

        return reconstruction + latent_error, (reconstruction, latent_error) #+ PINN #+ L2_K #+ latent_error #+ 0.01 * inf_norm
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi, vector_field):
        # l2 = self.l2_squared(self.encoderD) + self.l2_squared(self.decoder)
        total_loss, (reconstruction, latent_error) = jax.vmap(self.loss_fn, in_axes = [0, 0, 0, None])(ti, yi, wi, vector_field)
        return jnp.mean(total_loss), (jnp.mean(reconstruction), jnp.mean(latent_error))

    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi, vector_field)
        updates, opt_state = optim.update(grads, opt_state, params = self)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state  
    
    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction

        # latent = jax.vmap(self.get_latent, in_axes=(0, None), )(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        latent_1 = jax.vmap(self.get_latent, in_axes=(0, None))
        latent = jax.vmap(latent_1, in_axes=(0, None))(yi, wi)

        # reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)
        reconstruction_1 = jax.vmap(self.decoder)
        reconstruction = jax.vmap(reconstruction_1)(latent)
        
        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        # Mean Version:
        # weight_reconstruction = jnp.mean((tile - reconstruction[:,:, yi.shape[-1]:]) ** 2, axis=-1)
        # variable_reconstruction = jnp.mean((yi - reconstruction[:, :, :yi.shape[-1]]) ** 2, axis=-1)
        # reconstruction_loss = jnp.mean((yi_w - reconstruction) ** 2, axis=-1)

        # Sum Version:
        weight_reconstruction = jnp.mean(jnp.abs(tile - reconstruction[:,:, yi.shape[-1]:]), axis=-1)
        variable_reconstruction = jnp.mean(jnp.abs(yi - reconstruction[:, :, :yi.shape[-1]]), axis=-1)
        reconstruction_loss = jnp.mean(jnp.abs(yi_w - reconstruction), axis=-1)

        # Compute Jacobian Loss
        # x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        x_dot_1 = jax.vmap(vector_field, in_axes=(None, 0, None))
        x_dot = jax.vmap(x_dot_1, in_axes=(None, 0, None))(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)

        # _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        jvp_1 = jax.vmap(jax.jvp, in_axes=(None, 0, 0))
        _, dFdt = jax.vmap(jvp_1, in_axes=(None, 0, 0))(f, (yi,), (x_dot,))

        latent_1_ = jax.vmap(lambda x: self.Koopman @ x)
        latent_ = jax.vmap(latent_1_)(latent)
        j_loss = jnp.mean(jnp.abs(dFdt - latent_), axis=-1)
        
        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, variable_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig   

    def get_lipshitz_L(self, lx, ux, lw, uw, key, n=100):
        
        # Jacobian wrt parameters calculation:
        key_x, key_w = jax.random.split(key, 2)
        
        # Sampels of X
        x_samples = jax.random.uniform(key_x, (n, len(lx))) * (ux - lx) + lx
        
        # Grid of w
        w_samples = jax.ramdom.uniform(key_w, (n, len(lw))) * (uw - lw) + lw
        
        def f(x):
            return lambda wi: self.get_latent(x, wi)
        
        norm_list = []
        for i in range(n):
            working_function = f(x_samples[i])
            jacobians = jax.vmap(jax.jacfwd(working_function))(w_samples)
            average_norm = jnp.mean(jnp.linalg.norm(jacobians, ord = jax.numpy.inf, axis = -1))
            norm_list.append(average_norm)
        
        return jnp.max(norm_list), jnp.mean(norm_list), jnp.var(norm_list)
            
            
class PLearnKoopmanLipCE(eqx.Module):
    encoderD: LipMLP
    encoderP: LipMLP
    decoder: MLP
    latent_dim: int
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, b_array, w_array, key, **kwargs):
        
#        assert b_array[-1] + w_array[0] == d_array[0]
        
        d_key, e_key, k_key = jax.random.split(key, 3)
        
        self.latent_dim = d_array[-1]
        
        self.encoderD = LipMLP(d_array, e_key, layer_norm = False)
        self.decoder = MLP(b_array, d_key, layer_norm = True)
        self.encoderP = LipMLP(w_array, key = k_key, layer_norm = True)
        
    def get_latent(self, x, wi):
        return self.encoderD(x)
    
    def get_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)
    
    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))

    def get_null_koop(self, w, nzero_rows= 1):
        num_zero_rows = min(nzero_rows, self.latent_dim)
        num_filled_rows = self.latent_dim - num_zero_rows
        num_elements = num_filled_rows * self.latent_dim
        koop = w[:num_elements].reshape((num_filled_rows, self.latent_dim))
        zeros_rows = jnp.zeros((num_zero_rows, self.latent_dim))
        koop = jnp.concatenate([koop, zeros_rows], axis=0)
        return koop
    
    def get_diagonal(self, w):
        return jnp.diag(w)
    
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        latent = self.encoderD(x0)
        weights = self.encoderP(w)
        koop = self.get_naive(weights)
        
        '''
        # Discrete Koopman:
        def Koopman_layer(carry, hidden = None):
            return self.propagate_fixed_st(None, carry, koop), carry
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)
        '''

        # Continuous Koopman:
        latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, koop)
        predictions = jax.vmap(self.decoder)(latent_array)

        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]


    def l2_loss(self, w, alpha):
        return alpha * jnp.mean(w**2)

    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return jnp.mean(jnp.array([jnp.vdot(x, x) for x in leaves]))
    
    @eqx.filter_jit
    def loss_fn(self, ti, yi, wi):
        
        # Reconstruction
        latent, y_pred, w_pred = self(ti, yi[0], wi)
        # tile = jnp.tile(wi, (yi.shape[0],  1))
        # yi_w = jnp.concatenate((yi, tile), axis = -1)
        # y_w_pred = jnp.concatenate((y_pred, w_pred), axis = -1)
        # reconstruction = jnp.mean(jnp.mean((yi - y_pred)**2, axis = -1))
        # reconstruction = jnp.mean(jnp.mean(jnp.abs(yi_w - y_w_pred), axis = -1))

        reconstruction = jnp.mean(jnp.mean(jnp.abs(yi - y_pred), axis = -1))
        # reconstruction = jnp.mean(jnp.mean((jnp.linalg.norm(yi - y_pred, ord = 1, axis = -1)/jnp.linalg.norm(yi, ord=1, axis= -1)), axis = -1))
        
        # Latent Linearity Error:
        latent_traj = jax.vmap(self.get_latent, in_axes=[0, None])(yi, wi)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent - latent_traj), axis = -1))
        # latent_error = jnp.mean(jnp.mean((latent - latent_traj)**2, axis = -1))

        return reconstruction + latent_error, (reconstruction, latent_error) #+ PINN #+ L2_K #+ latent_error #+ 0.01 * inf_norm
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi, vector_field):
        # l2 = self.l2_squared(self.encoderD) + self.l2_squared(self.decoder)
        total_loss, (reconstruction, latent_error) = jax.vmap(self.loss_fn, in_axes = [0, 0, 0])(ti, yi, wi)
        lipschitz_loss_encoderD = self.encoderD.get_lipschitz_loss()
        lipschitz_loss_encoderP = self.encoderP.get_lipschitz_loss()
        # lipschitz_loss_decoder = self.decoder.get_lipschitz_loss()

        return jnp.mean(total_loss) + 0.1 * lipschitz_loss_encoderD +  0.1 * lipschitz_loss_encoderP, {"Reconstruction": jnp.mean(reconstruction), 
                                                                                                        "Latent Error": jnp.mean(latent_error), 
                                                                                                        "Lipschitz Encoder Loss": lipschitz_loss_encoderD, 
                                                                                                        "Lipschitz K encoder Loss": lipschitz_loss_encoderP}#, lipschitz_loss_decoder)


    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi, vector_field)
        updates, opt_state = optim.update(grads, opt_state, params = self)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state  
    
    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction

        # latent = jax.vmap(self.get_latent, in_axes=(0, None), )(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        latent_1 = jax.vmap(self.get_latent, in_axes=(0, None))
        latent = jax.vmap(latent_1, in_axes=(0, None))(yi, wi)

        # reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)
        reconstruction_1 = jax.vmap(self.decoder)
        reconstruction = jax.vmap(reconstruction_1)(latent)
        
        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        # Mean Version:
        # weight_reconstruction = jnp.mean((tile - reconstruction[:,:, yi.shape[-1]:]) ** 2, axis=-1)
        # variable_reconstruction = jnp.mean((yi - reconstruction[:, :, :yi.shape[-1]]) ** 2, axis=-1)
        # reconstruction_loss = jnp.mean((yi_w - reconstruction) ** 2, axis=-1)

        # Sum Version:
        weight_reconstruction = jnp.mean(jnp.abs(tile - reconstruction[:,:, yi.shape[-1]:]), axis=-1)
        variable_reconstruction = jnp.mean(jnp.abs(yi - reconstruction[:, :, :yi.shape[-1]]), axis=-1)
        reconstruction_loss = jnp.mean(jnp.abs(yi_w - reconstruction), axis=-1)

        # Compute Jacobian Loss
        # x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        x_dot_1 = jax.vmap(vector_field, in_axes=(None, 0, None))
        x_dot = jax.vmap(x_dot_1, in_axes=(None, 0, None))(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)

        # _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        jvp_1 = jax.vmap(jax.jvp, in_axes=(None, 0, 0))
        _, dFdt = jax.vmap(jvp_1, in_axes=(None, 0, 0))(f, (yi,), (x_dot,))

        latent_1_ = jax.vmap(lambda x: self.Koopman @ x)
        latent_ = jax.vmap(latent_1_)(latent)
        j_loss = jnp.mean(jnp.abs(dFdt - latent_), axis=-1)
        
        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, variable_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig   

    def get_lipshitz_L(self, lx, ux, lw, uw, key, n=100):
        
        # Jacobian wrt parameters calculation:
        key_x, key_w = jax.random.split(key, 2)
        
        # Sampels of X
        x_samples = jax.random.uniform(key_x, (n, len(lx))) * (ux - lx) + lx
        
        # Grid of w
        w_samples = jax.ramdom.uniform(key_w, (n, len(lw))) * (uw - lw) + lw
        
        def f(x):
            return lambda wi: self.get_latent(x, wi)
        
        norm_list = []
        for i in range(n):
            working_function = f(x_samples[i])
            jacobians = jax.vmap(jax.jacfwd(working_function))(w_samples)
            average_norm = jnp.mean(jnp.linalg.norm(jacobians, ord = jax.numpy.inf, axis = -1))
            norm_list.append(average_norm)
        
        return jnp.max(norm_list), jnp.mean(norm_list), jnp.var(norm_list)
 
 
class PLearnKoopmanKDecoder(eqx.Module):
    encoderD: LipMLP
    encoderP: LipMLP
    decoder: MLP
    decoderP: LipMLP
    latent_dim: int
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, b_array, w_array, key, **kwargs):
                
        d_key, e_key, k_key = jax.random.split(key, 3)
        
        self.latent_dim = d_array[-1]
        
        self.encoderD = LipMLP(d_array, e_key, layer_norm = False)
        self.decoder = MLP(b_array, d_key, layer_norm = True)
        self.encoderP = LipMLP(w_array, key = k_key, layer_norm = True)
        self.decoderP = LipMLP(w_array[::-1], key = k_key, layer_norm = False)
        
    def get_latent(self, x, wi):
        return self.encoderD(x)
    
    def get_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)
    
    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_diagonal(self, w):
        return jnp.diag(w)
    
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        latent = self.encoderD(x0)
        weights = self.encoderP(w)
        koop = self.get_naive(weights)
        
        '''
        # Discrete Koopman:
        def Koopman_layer(carry, hidden = None):
            return self.propagate_fixed_st(None, carry, koop), carry
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)
        '''

        # Continuous Koopman:
        latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, koop)
        predictions = jax.vmap(self.decoder)(latent_array)
        
        params = self.decoderP(weights)

        return latent_array, predictions[:, :len(x0)], params


    def l2_loss(self, w, alpha):
        return alpha * jnp.mean(w**2)

    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return jnp.mean(jnp.array([jnp.vdot(x, x) for x in leaves]))
    
    @eqx.filter_jit
    def loss_fn(self, ti, yi, wi):
        
        # Reconstruction
        latent, y_pred, w_pred = self(ti, yi[0], wi)
        
        # tile = jnp.tile(wi, (yi.shape[0],  1))
        # yi_w = jnp.concatenate((yi, tile), axis = -1)
        # y_w_pred = jnp.concatenate((y_pred, w_pred), axis = -1)
        # reconstruction = jnp.mean(jnp.mean((yi - y_pred)**2, axis = -1))
        
        w_reconstruction = jnp.mean(jnp.mean(jnp.abs(wi - w_pred), axis = -1))
        reconstruction = jnp.mean(jnp.mean(jnp.abs(yi - y_pred), axis = -1))
                
        # Latent Linearity Error:
        latent_traj = jax.vmap(self.get_latent, in_axes=[0, None])(yi, wi)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent - latent_traj), axis = -1))

        return reconstruction + latent_error + w_reconstruction, (reconstruction, latent_error, w_reconstruction) #+ PINN #+ L2_K #+ latent_error #+ 0.01 * inf_norm
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi, vector_field):

        total_loss, (reconstruction, latent_error, w_reconstruction) = jax.vmap(self.loss_fn, in_axes = [0, 0, 0])(ti, yi, wi)
        lipschitz_loss_encoderD = self.encoderD.get_lipschitz_loss()
        lipschitz_loss_decoderP = self.decoderP.get_lipschitz_loss()
        lipschitz_loss_encoderP = self.encoderP.get_lipschitz_loss()


        return jnp.mean(total_loss) + 0.1 * lipschitz_loss_encoderD +  0.1 * lipschitz_loss_decoderP + 0.1 * lipschitz_loss_encoderP , {"Reconstruction": jnp.mean(reconstruction), 
                                                                                                                                        "Latent Error": jnp.mean(latent_error), 
                                                                                                                                        "Weight Reconstruction": jnp.mean(w_reconstruction),
                                                                                                                                        "Lipschitz Encoder Loss": lipschitz_loss_encoderD, 
                                                                                                                                        "Lipschitz K decoder Loss": lipschitz_loss_decoderP,
                                                                                                                                        "Lipschitz K encoder Loss": lipschitz_loss_encoderP}#, lipschitz_loss_decoder)


    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi, vector_field)
        updates, opt_state = optim.update(grads, opt_state, params = self)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state  
    
    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction

        # latent = jax.vmap(self.get_latent, in_axes=(0, None), )(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        latent_1 = jax.vmap(self.get_latent, in_axes=(0, None))
        latent = jax.vmap(latent_1, in_axes=(0, None))(yi, wi)

        # reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)
        reconstruction_1 = jax.vmap(self.decoder)
        reconstruction = jax.vmap(reconstruction_1)(latent)
        
        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        # Mean Version:
        # weight_reconstruction = jnp.mean((tile - reconstruction[:,:, yi.shape[-1]:]) ** 2, axis=-1)
        # variable_reconstruction = jnp.mean((yi - reconstruction[:, :, :yi.shape[-1]]) ** 2, axis=-1)
        # reconstruction_loss = jnp.mean((yi_w - reconstruction) ** 2, axis=-1)

        # Sum Version:
        weight_reconstruction = jnp.mean(jnp.abs(tile - reconstruction[:,:, yi.shape[-1]:]), axis=-1)
        variable_reconstruction = jnp.mean(jnp.abs(yi - reconstruction[:, :, :yi.shape[-1]]), axis=-1)
        reconstruction_loss = jnp.mean(jnp.abs(yi_w - reconstruction), axis=-1)

        # Compute Jacobian Loss
        # x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        x_dot_1 = jax.vmap(vector_field, in_axes=(None, 0, None))
        x_dot = jax.vmap(x_dot_1, in_axes=(None, 0, None))(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)

        # _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        jvp_1 = jax.vmap(jax.jvp, in_axes=(None, 0, 0))
        _, dFdt = jax.vmap(jvp_1, in_axes=(None, 0, 0))(f, (yi,), (x_dot,))

        latent_1_ = jax.vmap(lambda x: self.Koopman @ x)
        latent_ = jax.vmap(latent_1_)(latent)
        j_loss = jnp.mean(jnp.abs(dFdt - latent_), axis=-1)
        
        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, variable_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig   

    def get_lipshitz_L(self, lx, ux, lw, uw, key, n=100):
        
        # Jacobian wrt parameters calculation:
        key_x, key_w = jax.random.split(key, 2)
        
        # Sampels of X
        x_samples = jax.random.uniform(key_x, (n, len(lx))) * (ux - lx) + lx
        
        # Grid of w
        w_samples = jax.ramdom.uniform(key_w, (n, len(lw))) * (uw - lw) + lw
        
        def f(x):
            return lambda wi: self.get_latent(x, wi)
        
        norm_list = []
        for i in range(n):
            working_function = f(x_samples[i])
            jacobians = jax.vmap(jax.jacfwd(working_function))(w_samples)
            average_norm = jnp.mean(jnp.linalg.norm(jacobians, ord = jax.numpy.inf, axis = -1))
            norm_list.append(average_norm)
        
        return jnp.max(norm_list), jnp.mean(norm_list), jnp.var(norm_list)
                       
    
class PLearnKoopmanD(eqx.Module):
    encoderD: MLP
    encoderP: MLP
    decoder: MLP
    latent_dim: int
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, b_array, w_array,key, **kwargs):
        
#        assert b_array[-1] + w_array[0] == d_array[0]
        
        d_key, e_key, k_key = jax.random.split(key, 3)
        
        self.latent_dim = d_array[-1]
        
        self.encoderD = MLP(d_array, e_key, layer_norm = True)
        self.decoder = MLP(b_array, d_key, layer_norm=True)
        self.encoderP = MLP(w_array, key = k_key, layer_norm = True)
        
    def get_latent(self, x):
        return self.encoder(x)
    
    def get_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)
    
    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    def l2_loss(self, w, alpha):
        return alpha * jnp.mean(w**2)   
    
    def l2_sum_loss(self, w, alpha):
        return alpha * sum(w**2)   

    def get_latent(self, x, wi):
        x_w = jnp.concatenate((x, wi), axis = -1)
        return self.encoderD(x_w) 
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        x_w = jnp.concatenate((x0, w), axis = -1)
        
        nds = len(ts)
        
        latent = self.encoderD(x_w)
        weights = self.encoderP(w)
        koop = self.get_naive(weights)
        
        def Koopman_layer(carry, hidden = None):
            return self.propagate_fixed_st(None, carry, koop), carry
        
        _, latent_array = jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)
        
        predictions = jax.vmap(self.decoder)(latent_array)

        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]


    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return sum(jnp.vdot(x, x) for x in leaves)

    @eqx.filter_jit
    def loss_fn(self, ti, yi, wi):
        
        latent, y_pred, w_pred = self(ti, yi[0], wi)
        reconstruction = jnp.mean(jnp.mean(jnp.abs(yi - y_pred), axis = -1))
        latent_traj = jax.vmap(self.get_latent, in_axes=[0, None])(yi, wi)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent - latent_traj), axis = -1))

        weight_loss = self.l2_sum_loss(self.encoderP(wi), 1)

        return reconstruction + latent_error, (reconstruction, latent_error, weight_loss)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi):
        # l2 = self.l2_squared(self.encoderD) + self.l2_squared(self.decoder)
        total_loss, (reconstruction, latent_error, weight_loss) = jax.vmap(self.loss_fn, in_axes = [0, 0, 0])(ti, yi, wi)
        return jnp.mean(total_loss) + jnp.mean(weight_loss), (jnp.mean(reconstruction), jnp.mean(latent_error), jnp.mean(weight_loss))

    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state  
    
    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction

        # latent = jax.vmap(self.get_latent, in_axes=(0, None), )(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        latent_1 = jax.vmap(self.get_latent, in_axes=(0, None))
        latent = jax.vmap(latent_1, in_axes=(0, None))(yi, wi)

        # reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)
        reconstruction_1 = jax.vmap(self.decoder)
        reconstruction = jax.vmap(reconstruction_1)(latent)
        
        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        # Mean Version:
        # weight_reconstruction = jnp.mean((tile - reconstruction[:,:, yi.shape[-1]:]) ** 2, axis=-1)
        # variable_reconstruction = jnp.mean((yi - reconstruction[:, :, :yi.shape[-1]]) ** 2, axis=-1)
        # reconstruction_loss = jnp.mean((yi_w - reconstruction) ** 2, axis=-1)

        # Sum Version:
        weight_reconstruction = jnp.mean(jnp.abs(tile - reconstruction[:,:, yi.shape[-1]:]), axis=-1)
        variable_reconstruction = jnp.mean(jnp.abs(yi - reconstruction[:, :, :yi.shape[-1]]), axis=-1)
        reconstruction_loss = jnp.mean(jnp.abs(yi_w - reconstruction), axis=-1)

        # Compute Jacobian Loss
        # x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        x_dot_1 = jax.vmap(vector_field, in_axes=(None, 0, None))
        x_dot = jax.vmap(x_dot_1, in_axes=(None, 0, None))(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)

        # _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        jvp_1 = jax.vmap(jax.jvp, in_axes=(None, 0, 0))
        _, dFdt = jax.vmap(jvp_1, in_axes=(None, 0, 0))(f, (yi,), (x_dot,))

        latent_1_ = jax.vmap(lambda x: self.Koopman @ x)
        latent_ = jax.vmap(latent_1_)(latent)
        j_loss = jnp.mean(jnp.abs(dFdt - latent_), axis=-1)
        
        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, variable_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig   
    

class PLearnKoopman(eqx.Module):
    encoderD: MLP
    encoderP: MLP
    decoder: MLP
    latent_dim: int
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, b_array, w_array,key, **kwargs):
        
#        assert b_array[-1] + w_array[0] == d_array[0]
        
        d_key, e_key, k_key = jax.random.split(key, 3)
        
        self.latent_dim = d_array[-1]
        
        self.encoderD = MLP(d_array, e_key, layer_norm = True)
        self.decoder = MLP(b_array, d_key, layer_norm = True)
        self.encoderP = MLP(w_array, key = k_key, layer_norm = True)
        
    def get_latent(self, x, wi):
        x_w = jnp.concatenate((x, wi), axis = -1)
        return self.encoderD(x_w)
    
    def get_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)
    
    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_diagonal(self, w):
        return jnp.diag(w)
    
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        x_w = jnp.concatenate((x0, w), axis = -1)
        latent = self.encoderD(x_w)
        weights = self.encoderP(w)
        koop = self.get_naive(weights)
        
        '''
        # Discrete Koopman:
        def Koopman_layer(carry, hidden = None):
            return self.propagate_fixed_st(None, carry, koop), carry
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)
        '''

        # Continuous Koopman:
        latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, koop)
        predictions = jax.vmap(self.decoder)(latent_array)

        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]


    def l2_loss(self, w, alpha):
        return alpha * jnp.mean(w**2)

    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return jnp.mean(jnp.array([jnp.vdot(x, x) for x in leaves]))
    
    @eqx.filter_jit
    def loss_fn(self, ti, yi, wi, vector_field):
        
        # Reconstruction
        latent, y_pred, w_pred = self(ti, yi[0], wi)
        # tile = jnp.tile(wi, (yi.shape[0],  1))
        # yi_w = jnp.concatenate((yi, tile), axis = -1)
        # y_w_pred = jnp.concatenate((y_pred, w_pred), axis = -1)
        # reconstruction = jnp.mean(jnp.mean((yi - y_pred)**2, axis = -1))

        reconstruction = jnp.mean(jnp.mean(jnp.abs(yi - y_pred), axis = -1))
        # reconstruction = jnp.mean(jnp.mean((jnp.linalg.norm(yi - y_pred, ord = 1, axis = -1)/jnp.linalg.norm(yi, ord=1, axis= -1)), axis = -1))
        
        # Latent Linearity Error:
        latent_traj = jax.vmap(self.get_latent, in_axes=[0, None])(yi, wi)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent - latent_traj), axis = -1))
        # latent_error = jnp.mean(jnp.mean((latent - latent_traj)**2, axis = -1))

        return reconstruction + latent_error, (reconstruction, latent_error) #+ PINN #+ L2_K #+ latent_error #+ 0.01 * inf_norm
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi, vector_field):
        # l2 = self.l2_squared(self.encoderD) + self.l2_squared(self.decoder)
        total_loss, (reconstruction, latent_error) = jax.vmap(self.loss_fn, in_axes = [0, 0, 0, None])(ti, yi, wi, vector_field)
        return jnp.mean(total_loss), (jnp.mean(reconstruction), jnp.mean(latent_error))

    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi, vector_field)
        updates, opt_state = optim.update(grads, opt_state, params = self)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state  
    
    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction

        # latent = jax.vmap(self.get_latent, in_axes=(0, None), )(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        latent_1 = jax.vmap(self.get_latent, in_axes=(0, None))
        latent = jax.vmap(latent_1, in_axes=(0, None))(yi, wi)

        # reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)
        reconstruction_1 = jax.vmap(self.decoder)
        reconstruction = jax.vmap(reconstruction_1)(latent)
        
        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        # Mean Version:
        # weight_reconstruction = jnp.mean((tile - reconstruction[:,:, yi.shape[-1]:]) ** 2, axis=-1)
        # variable_reconstruction = jnp.mean((yi - reconstruction[:, :, :yi.shape[-1]]) ** 2, axis=-1)
        # reconstruction_loss = jnp.mean((yi_w - reconstruction) ** 2, axis=-1)

        # Sum Version:
        weight_reconstruction = jnp.mean(jnp.abs(tile - reconstruction[:,:, yi.shape[-1]:]), axis=-1)
        variable_reconstruction = jnp.mean(jnp.abs(yi - reconstruction[:, :, :yi.shape[-1]]), axis=-1)
        reconstruction_loss = jnp.mean(jnp.abs(yi_w - reconstruction), axis=-1)

        # Compute Jacobian Loss
        # x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        x_dot_1 = jax.vmap(vector_field, in_axes=(None, 0, None))
        x_dot = jax.vmap(x_dot_1, in_axes=(None, 0, None))(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)

        # _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        jvp_1 = jax.vmap(jax.jvp, in_axes=(None, 0, 0))
        _, dFdt = jax.vmap(jvp_1, in_axes=(None, 0, 0))(f, (yi,), (x_dot,))

        latent_1_ = jax.vmap(lambda x: self.Koopman @ x)
        latent_ = jax.vmap(latent_1_)(latent)
        j_loss = jnp.mean(jnp.abs(dFdt - latent_), axis=-1)
        
        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, variable_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig   

    def get_lipshitz_L(self, lx, ux, lw, uw, key, n=100):
        
        # Jacobian wrt parameters calculation:
        key_x, key_w = jax.random.split(key, 2)
        
        # Sampels of X
        x_samples = jax.random.uniform(key_x, (n, len(lx))) * (ux - lx) + lx
        
        # Grid of w
        w_samples = jax.random.uniform(key_w, (n, len(lw))) * (uw - lw) + lw
        
        def f(x):
            return lambda wi: self.get_latent(x, wi)
        
        norm_list = []
        for i in range(n):
            working_function = f(x_samples[i])
            jacobians = jax.vmap(jax.jacfwd(working_function))(w_samples)
            average_norm = jnp.mean(jnp.linalg.norm(jacobians, ord = jax.numpy.inf, axis = -1))
            norm_list.append(average_norm)
        
        return jnp.max(norm_list), jnp.mean(norm_list), jnp.var(norm_list)
            
class PLearnKoopmanLip(eqx.Module):
    encoderD: MLP
    encoderP: MLP
    decoder: MLP
    latent_dim: int
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, b_array, w_array, key, **kwargs):
        
#        assert b_array[-1] + w_array[0] == d_array[0]
        
        d_key, e_key, k_key = jax.random.split(key, 3)
        
        self.latent_dim = d_array[-1]
        
        self.encoderD = LipMLP(d_array, e_key, layer_norm = False)
        self.decoder = MLP(b_array, d_key, layer_norm = True)
        self.encoderP = LipMLP(w_array, key = k_key, layer_norm = True)
        
    def get_latent(self, x, wi):
        x_w = jnp.concatenate((x, wi), axis = -1)
        return self.encoderD(x_w)
    
    def get_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)
    
    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_diagonal(self, w):
        return jnp.diag(w)
   
   
    def get_null_koop(self, w, nzero_rows= 1):
        num_zero_rows = min(nzero_rows, self.latent_dim)
        num_filled_rows = self.latent_dim - num_zero_rows
        num_elements = num_filled_rows * self.latent_dim
        koop = w[:num_elements].reshape((num_filled_rows, self.latent_dim))
        zeros_rows = jnp.zeros((num_zero_rows, self.latent_dim))
        koop = jnp.concatenate([koop, zeros_rows], axis=0)
        return koop
     
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        x_w = jnp.concatenate((x0, w), axis = -1)
        latent = self.encoderD(x_w)
        weights = self.encoderP(w)
        
        koop = self.get_naive(weights)

        # koop = self.get_null_koop(weights)
        
        '''
        # Discrete Koopman:
        def Koopman_layer(carry, hidden = None):
            return self.propagate_fixed_st(None, carry, koop), carry
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)
        '''

        # Continuous Koopman:
        latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, koop)
        predictions = jax.vmap(self.decoder)(latent_array)

        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]


    def l2_loss(self, w, alpha):
        return alpha * jnp.mean(w**2)

    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return jnp.mean(jnp.array([jnp.vdot(x, x) for x in leaves]))
    
    @eqx.filter_jit
    def loss_fn(self, ti, yi, wi):
        
        # Reconstruction
        latent, y_pred, w_pred = self(ti, yi[0], wi)
        # tile = jnp.tile(wi, (yi.shape[0],  1))
        # yi_w = jnp.concatenate((yi, tile), axis = -1)
        # y_w_pred = jnp.concatenate((y_pred, w_pred), axis = -1)
        reconstruction = jnp.mean(jnp.mean((yi - y_pred)**2, axis = -1))
        # reconstruction = jnp.mean(jnp.mean(jnp.abs(yi_w - y_w_pred), axis = -1))

        # reconstruction = jnp.mean(jnp.mean(jnp.abs(yi - y_pred), axis = -1))
        # reconstruction = jnp.mean(jnp.linalg.norm(yi - y_pred, ord = 1, axis = -1)/( jnp.linalg.norm(y_pred, ord = 1, axis = -1) + jnp.linalg.norm(yi, ord=1, axis= -1)))
        
        # Latent Linearity Error:
        latent_traj = jax.vmap(self.get_latent, in_axes=[0, None])(yi, wi)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent - latent_traj), axis = -1))
        # latent_error = jnp.mean(jnp.linalg.norm(latent - latent_traj, ord = 1, axis = -1)/( jnp.linalg.norm(latent, ord = 1, axis = -1) + jnp.linalg.norm(latent_traj, ord=1, axis= -1)))

        # latent_error = jnp.mean(jnp.mean((latent - latent_traj)**2, axis = -1))

        return reconstruction + latent_error, (reconstruction, latent_error) #+ PINN #+ L2_K #+ latent_error #+ 0.01 * inf_norm
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi, vector_field):
        # l2 = self.l2_squared(self.encoderD) + self.l2_squared(self.decoder)
        total_loss, (reconstruction, latent_error) = jax.vmap(self.loss_fn, in_axes = [0, 0, 0])(ti, yi, wi)
        lipschitz_loss_encoderD = self.encoderD.get_lipschitz_loss()
        lipschitz_loss_encoderP = self.encoderP.get_lipschitz_loss()
        # lipschitz_loss_decoder = self.decoder.get_lipschitz_loss()

        return jnp.mean(total_loss) + 0.1 * lipschitz_loss_encoderD +  0.1 * lipschitz_loss_encoderP, {"Reconstruction": jnp.mean(reconstruction), 
                                                                                                        "Latent Error": jnp.mean(latent_error), 
                                                                                                        "Lipschitz Encoder Loss": lipschitz_loss_encoderD, 
                                                                                                        "Lipschitz K encoder Loss": lipschitz_loss_encoderP}#, lipschitz_loss_decoder)

    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi, vector_field)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state  
    
    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction

        # latent = jax.vmap(self.get_latent, in_axes=(0, None), )(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        latent_1 = jax.vmap(self.get_latent, in_axes=(0, None))
        latent = jax.vmap(latent_1, in_axes=(0, None))(yi, wi)

        # reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)
        reconstruction_1 = jax.vmap(self.decoder)
        reconstruction = jax.vmap(reconstruction_1)(latent)
        
        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        # Mean Version:
        # weight_reconstruction = jnp.mean((tile - reconstruction[:,:, yi.shape[-1]:]) ** 2, axis=-1)
        # variable_reconstruction = jnp.mean((yi - reconstruction[:, :, :yi.shape[-1]]) ** 2, axis=-1)
        # reconstruction_loss = jnp.mean((yi_w - reconstruction) ** 2, axis=-1)

        # Sum Version:
        weight_reconstruction = jnp.mean(jnp.abs(tile - reconstruction[:,:, yi.shape[-1]:]), axis=-1)
        variable_reconstruction = jnp.mean(jnp.abs(yi - reconstruction[:, :, :yi.shape[-1]]), axis=-1)
        reconstruction_loss = jnp.mean(jnp.abs(yi_w - reconstruction), axis=-1)

        # Compute Jacobian Loss
        # x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        x_dot_1 = jax.vmap(vector_field, in_axes=(None, 0, None))
        x_dot = jax.vmap(x_dot_1, in_axes=(None, 0, None))(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)

        # _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        jvp_1 = jax.vmap(jax.jvp, in_axes=(None, 0, 0))
        _, dFdt = jax.vmap(jvp_1, in_axes=(None, 0, 0))(f, (yi,), (x_dot,))

        latent_1_ = jax.vmap(lambda x: self.Koopman @ x)
        latent_ = jax.vmap(latent_1_)(latent)
        j_loss = jnp.mean(jnp.abs(dFdt - latent_), axis=-1)
        
        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, variable_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig   

    def get_lipshitz_L(self, lx, ux, lw, uw, key, n=100):
        
        # Jacobian wrt parameters calculation:
        key_x, key_w = jax.random.split(key, 2)
        
        # Sampels of X
        x_samples = jax.random.uniform(key_x, (n, len(lx))) * (ux - lx) + lx
        
        # Grid of w
        w_samples = jax.random.uniform(key_w, (n, len(lw))) * (uw - lw) + lw
        
        def f(x):
            return lambda wi: self.get_latent(x, wi)
        
        norm_list = []
        for i in range(n):
            working_function = f(x_samples[i])
            jacobians = jax.vmap(jax.jacfwd(working_function))(w_samples)
            average_norm = jnp.mean(jnp.linalg.norm(jacobians, ord = jax.numpy.inf, axis = -1))
            norm_list.append(average_norm)
        
        return jnp.max(norm_list), jnp.mean(norm_list), jnp.var(norm_list)

class PLearnKoopmanLipCK(eqx.Module):
    encoderD: LipMLP
    encoderP: MLP
    decoder: MLP
    latent_dim: int
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, b_array, w_array, key, **kwargs):
        
#        assert b_array[-1] + w_array[0] == d_array[0]
        
        d_key, e_key, k_key = jax.random.split(key, 3)
        
        self.latent_dim = d_array[-1]
        
        self.encoderD = LipMLP(d_array, e_key, layer_norm = False)
        self.decoder = MLP(b_array, d_key, layer_norm = True)
        self.encoderP = MLP(w_array, key = k_key, layer_norm = True)
        
    def get_latent(self, x, wi):
        x_w = jnp.concatenate((x, wi), axis = -1)
        return self.encoderD(x_w)
    
    def get_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)
    
    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_diagonal(self, w):
        return jnp.diag(w)
    
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        x_w = jnp.concatenate((x0, w), axis = -1)
        latent = self.encoderD(x_w)
        weights = self.encoderP(x_w)
        koop = self.get_naive(weights)
        
        '''
        # Discrete Koopman:
        def Koopman_layer(carry, hidden = None):
            return self.propagate_fixed_st(None, carry, koop), carry
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = len(ts))
        predictions = jax.vmap(self.decoder)(latent_array)
        '''
        
        # Continuous Koopman:
        latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, koop)
        predictions = jax.vmap(self.decoder)(latent_array)
        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]


    def l2_loss(self, w, alpha):
        return alpha * jnp.mean(w**2)

    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return jnp.mean(jnp.array([jnp.vdot(x, x) for x in leaves]))
    
    @eqx.filter_jit
    def loss_fn(self, ti, yi, wi):
        
        # Reconstruction
        latent, y_pred, w_pred = self(ti, yi[0], wi)
        tile = jnp.tile(wi, (yi.shape[0],  1))
        # yi_w = jnp.concatenate((yi, tile), axis = -1)
        # y_w_pred = jnp.concatenate((y_pred, w_pred), axis = -1)
        reconstruction = jnp.mean(jnp.mean(jnp.abs(yi - y_pred), axis = -1))
        weight_reconstruction = jnp.mean(jnp.abs(tile - w_pred), axis=-1)
        # reconstruction = jnp.mean(jnp.mean(jnp.abs(yi - y_pred), axis = -1))
        
        # Latent Linearity Error:
        latent_traj = jax.vmap(self.get_latent, in_axes=[0, None])(yi, wi)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent - latent_traj), axis = -1))
        # latent_error = jnp.mean(jnp.mean((latent - latent_traj)**2, axis = -1))

        return reconstruction + latent_error + weight_reconstruction, (reconstruction, latent_error, weight_reconstruction) #+ PINN #+ L2_K #+ latent_error #+ 0.01 * inf_norm
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi, vector_field):
        # l2 = self.l2_squared(self.encoderD) + self.l2_squared(self.decoder)
        total_loss, (reconstruction, latent_error, weight_reconstruction) = jax.vmap(self.loss_fn, in_axes = [0, 0, 0])(ti, yi, wi)
        lipschitz_loss_encoderD = self.encoderD.get_lipschitz_loss()
        # lipschitz_loss_encoderP = self.encoderP.get_lipschitz_loss()
        # lipschitz_loss_decoder = self.decoder.get_lipschitz_loss()

        return jnp.mean(total_loss) + 0.1 * lipschitz_loss_encoderD, {"Reconstruction": jnp.mean(reconstruction), 
                                                                      "Parameter Reconstruction": jnp.mean(weight_reconstruction),
                                                                        "Latent Error": jnp.mean(latent_error), 
                                                                        "Lipschitz Encoder Loss": lipschitz_loss_encoderD, 
                                                                        
                                                                        # "Lipschitz K encoder Loss": lipschitz_loss_encoderP,
                                                                        }#, lipschitz_loss_decoder)

    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi, vector_field)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state  
    
    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction

        # latent = jax.vmap(self.get_latent, in_axes=(0, None), )(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        latent_1 = jax.vmap(self.get_latent, in_axes=(0, None))
        latent = jax.vmap(latent_1, in_axes=(0, None))(yi, wi)

        # reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)
        reconstruction_1 = jax.vmap(self.decoder)
        reconstruction = jax.vmap(reconstruction_1)(latent)
        
        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        # Mean Version:
        # weight_reconstruction = jnp.mean((tile - reconstruction[:,:, yi.shape[-1]:]) ** 2, axis=-1)
        # variable_reconstruction = jnp.mean((yi - reconstruction[:, :, :yi.shape[-1]]) ** 2, axis=-1)
        # reconstruction_loss = jnp.mean((yi_w - reconstruction) ** 2, axis=-1)

        # Sum Version:
        weight_reconstruction = jnp.mean(jnp.abs(tile - reconstruction[:,:, yi.shape[-1]:]), axis=-1)
        variable_reconstruction = jnp.mean(jnp.abs(yi - reconstruction[:, :, :yi.shape[-1]]), axis=-1)
        reconstruction_loss = jnp.mean(jnp.abs(yi_w - reconstruction), axis=-1)

        # Compute Jacobian Loss
        # x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        x_dot_1 = jax.vmap(vector_field, in_axes=(None, 0, None))
        x_dot = jax.vmap(x_dot_1, in_axes=(None, 0, None))(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)

        # _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        jvp_1 = jax.vmap(jax.jvp, in_axes=(None, 0, 0))
        _, dFdt = jax.vmap(jvp_1, in_axes=(None, 0, 0))(f, (yi,), (x_dot,))

        latent_1_ = jax.vmap(lambda x: self.Koopman @ x)
        latent_ = jax.vmap(latent_1_)(latent)
        j_loss = jnp.mean(jnp.abs(dFdt - latent_), axis=-1)
        
        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, variable_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig   

    def get_lipshitz_L(self, lx, ux, lw, uw, key, n=100):
        
        # Jacobian wrt parameters calculation:
        key_x, key_w = jax.random.split(key, 2)
        
        # Sampels of X
        x_samples = jax.random.uniform(key_x, (n, len(lx))) * (ux - lx) + lx
        
        # Grid of w
        w_samples = jax.random.uniform(key_w, (n, len(lw))) * (uw - lw) + lw
        
        def f(x):
            return lambda wi: self.get_latent(x, wi)
        
        norm_list = []
        for i in range(n):
            working_function = f(x_samples[i])
            jacobians = jax.vmap(jax.jacfwd(working_function))(w_samples)
            average_norm = jnp.mean(jnp.linalg.norm(jacobians, ord = jax.numpy.inf, axis = -1))
            norm_list.append(average_norm)
        
        return jnp.max(norm_list), jnp.mean(norm_list), jnp.var(norm_list)

    
class PLearnKoopmanConvex(eqx.Module):
    encoderD: MLP
    encoderP: MLP
    decoder: MLP
    latent_dim: int
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, b_array, w_array, key, **kwargs):
        
#        assert b_array[-1] + w_array[0] == d_array[0]
        
        d_key, e_key, k_key = jax.random.split(key, 3)
        
        self.latent_dim = d_array[-1]
        
        self.encoderD = LipICNN(d_array, key = e_key)
        self.decoder = MLP(b_array, d_key, layer_norm = True)
        self.encoderP = LipICNN(w_array, key = k_key)
        
    @eqx.filter_jit
    def get_latent_series(self, ts, x0, w):        
        x_w = jnp.concatenate((x0, w), axis = -1)
        latent = self.encoderD(x_w)
        weights = self.encoderP(w)
        koop = self.get_naive(weights)
        latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, koop)
        return latent_array    
    
    @eqx.filter_jit
    def get_latent(self, x, wi):
        x_w = jnp.concatenate((x, wi), axis = -1)
        return self.encoderD(x_w)
    
    def get_L(self):
        return self.encoderD.get_lipschitz(), self.encoderP.get_lipschitz()
    
    def get_L_loss(self):
        return self.encoderD.get_lipschitz_loss(), self.encoderP.get_lipschitz_loss()
    def get_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)
    
    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_diagonal(self, w):
        return jnp.diag(w)
    
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        x_w = jnp.concatenate((x0, w), axis = -1)
        latent = self.encoderD(x_w)
        weights = self.encoderP(w)
        koop = self.get_naive(weights)
        
        '''
        # Discrete Koopman:
        def Koopman_layer(carry, hidden = None):
            return self.propagate_fixed_st(None, carry, koop), carry
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)
        '''

        # Continuous Koopman:
        latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, koop)
        predictions = jax.vmap(self.decoder)(latent_array)

        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]


    def l2_loss(self, w, alpha):
        return alpha * jnp.mean(w**2)

    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return jnp.mean(jnp.array([jnp.vdot(x, x) for x in leaves]))
    
    @eqx.filter_jit
    def loss_fn(self, ti, yi, wi):
        
        # Reconstruction
        latent, y_pred, w_pred = self(ti, yi[0], wi)
        # tile = jnp.tile(wi, (yi.shape[0],  1))
        # yi_w = jnp.concatenate((yi, tile), axis = -1)
        # y_w_pred = jnp.concatenate((y_pred, w_pred), axis = -1)
        # reconstruction = jnp.mean(jnp.mean((yi - y_pred)**2, axis = -1))

        reconstruction = jnp.mean(jnp.mean(jnp.abs(yi - y_pred), axis = -1))
        # reconstruction = jnp.mean(jnp.mean((jnp.linalg.norm(yi - y_pred, ord = 1, axis = -1)/jnp.linalg.norm(yi, ord=1, axis= -1)), axis = -1))
        
        # Latent Linearity Error:
        latent_traj = jax.vmap(self.get_latent, in_axes=[0, None])(yi, wi)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent - latent_traj), axis = -1))
        # latent_error = jnp.mean(jnp.mean((latent - latent_traj)**2, axis = -1))

        return reconstruction + latent_error, (reconstruction, latent_error) #+ PINN #+ L2_K #+ latent_error #+ 0.01 * inf_norm
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi, vector_field):
        # l2 = self.l2_squared(self.encoderD) + self.l2_squared(self.decoder)
        total_loss, (reconstruction, latent_error) = jax.vmap(self.loss_fn, in_axes = [0, 0, 0])(ti, yi, wi)
        lipschitz_loss_encoderD = self.encoderD.get_lipschitz_loss()
        lipschitz_loss_encoderP = self.encoderP.get_lipschitz_loss()
        # lipschitz_loss_decoder = self.decoder.get_lipschitz_loss()

        return jnp.mean(total_loss) + 0.1 * lipschitz_loss_encoderD + 0.1 * lipschitz_loss_encoderP, {"Reconstruction": jnp.mean(reconstruction), 
                                     "Latent Error": jnp.mean(latent_error),
                                     "Lipschitz Encoder Loss": lipschitz_loss_encoderD,
                                     "Lipschitz K encoder Loss": lipschitz_loss_encoderP,
                                     }

    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi, vector_field)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state  
    

class PLearnKoopmanFF(eqx.Module):
    f_encoder: FourierEncoder
    f_encoder_w: FourierEncoder
    encoderD: MLP
    encoderP: MLP
    decoder: MLP
    latent_dim: int
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, b_array, w_array,key, ff, scale, **kwargs):
            
        d_key, e_key, k_key, f_key, f_key2 = jax.random.split(key, 5)
        self.latent_dim = d_array[-1]
        if ff:          
            self.f_encoder = FourierEncoder(d = d_array[0], key = f_key, scale = scale, B_out= d_array[1])
            self.f_encoder_w = FourierEncoder(d = w_array[0], key = f_key2, scale = scale, B_out=w_array[1])

            self.encoderD = MLP(d_array[2:], e_key, layer_norm = True)
            self.decoder = MLP(b_array, d_key, layer_norm = True)
            self.encoderP = MLP(w_array[2:], key = k_key, layer_norm = True)
        else:
            self.f_encoder = FourierEncoder(d = d_array[0], key = f_key, scale = scale, B_out=None)
            self.f_encoder_w = FourierEncoder(d = w_array[0], key = f_key2, scale = scale, B_out=None)

            self.encoderD = MLP(d_array, e_key, layer_norm = True)
            self.decoder = MLP(b_array, d_key, layer_norm = True)
            self.encoderP = MLP(w_array, key = k_key, layer_norm = True)
        
    def get_latent(self, x, wi):
        x_w = jnp.concatenate((x, wi), axis = -1)
        platent = self.f_encoder(x_w)
        return self.encoderD(platent)
    
    def get_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)
    
    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_diagonal(self, w):
        return jnp.diag(w)
    
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        x_w = jnp.concatenate((x0, w), axis = -1)
        
        platent = self.f_encoder(x_w)
        latent = self.encoderD(platent)
        pweights = self.f_encoder_w(w)
        weights = self.encoderP(pweights)
        
        koop = self.get_naive(weights)
        
        '''
        # Discrete Koopman:
        def Koopman_layer(carry, hidden = None):
            return self.propagate_fixed_st(None, carry, koop), carry
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)
        '''

        # Continuous Koopman:
        latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, koop)
        predictions = jax.vmap(self.decoder)(latent_array)

        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]


    def l2_loss(self, w, alpha):
        return alpha * jnp.mean(w**2)

    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return jnp.mean(jnp.array([jnp.vdot(x, x) for x in leaves]))
    
    
    @eqx.filter_jit
    def loss_fn(self, ti, yi, wi, vector_field):
        
        # Reconstruction
        latent, y_pred, w_pred = self(ti, yi[0], wi)
        # tile = jnp.tile(wi, (yi.shape[0],  1))
        # yi_w = jnp.concatenate((yi, tile), axis = -1)
        # y_w_pred = jnp.concatenate((y_pred, w_pred), axis = -1)
        reconstruction = jnp.mean(jnp.mean((yi - y_pred)**2, axis = -1))
        
        # reconstruction = jnp.mean(jnp.mean((jnp.linalg.norm(yi - (y_pred), ord = 2, axis = -1)/jnp.linalg.norm(yi, ord=2, axis= -1)), axis = -1))
        
        # Latent Linearity Error:
        latent_traj = jax.vmap(self.get_latent, in_axes=[0, None])(yi, wi)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent - latent_traj), axis = -1))

        return reconstruction + latent_error, (reconstruction, latent_error) #+ PINN #+ L2_K #+ latent_error #+ 0.01 * inf_norm
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    def gen_filter_spec(self):
        filter_spec = jax.tree_util.tree_map(lambda _: True, self)
        filter_spec = eqx.tree_at(
        lambda tree: (tree.f_encoder.B, tree.f_encoder_w.B),
        filter_spec,
        replace=(False, False),
        )
        return filter_spec
        
        
    @staticmethod
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(diff_model, static_model, ti, yi, wi, vector_field):
        model = eqx.combine(diff_model, static_model)
        total_loss, (reconstruction, latent_error) = jax.vmap(model.loss_fn, in_axes = [0, 0, 0, None])(ti, yi, wi, vector_field)
        return jnp.mean(total_loss), (jnp.mean(reconstruction), jnp.mean(latent_error))

    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        diff_model, static_model = eqx.partition(self, self.gen_filter_spec())
        losses, grads = self.batch_loss(diff_model, static_model, ti, yi, wi, vector_field)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state  
    
    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction

        # latent = jax.vmap(self.get_latent, in_axes=(0, None), )(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        latent_1 = jax.vmap(self.get_latent, in_axes=(0, None))
        latent = jax.vmap(latent_1, in_axes=(0, None))(yi, wi)

        # reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)
        reconstruction_1 = jax.vmap(self.decoder)
        reconstruction = jax.vmap(reconstruction_1)(latent)
        
        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        # Sum Version:
        weight_reconstruction = jnp.mean(jnp.abs(tile - reconstruction[:,:, yi.shape[-1]:]), axis=-1)
        variable_reconstruction = jnp.mean(jnp.abs(yi - reconstruction[:, :, :yi.shape[-1]]), axis=-1)
        reconstruction_loss = jnp.mean(jnp.abs(yi_w - reconstruction), axis=-1)

        # Compute Jacobian Loss
        # x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        x_dot_1 = jax.vmap(vector_field, in_axes=(None, 0, None))
        x_dot = jax.vmap(x_dot_1, in_axes=(None, 0, None))(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)

        # _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        jvp_1 = jax.vmap(jax.jvp, in_axes=(None, 0, 0))
        _, dFdt = jax.vmap(jvp_1, in_axes=(None, 0, 0))(f, (yi,), (x_dot,))

        latent_1_ = jax.vmap(lambda x: self.Koopman @ x)
        latent_ = jax.vmap(latent_1_)(latent)
        j_loss = jnp.mean(jnp.abs(dFdt - latent_), axis=-1)
        
        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, variable_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig   


class PLearnKoopmanSin(eqx.Module):
    encoderD: MLP
    encoderP: MLP
    decoder: MLP
    latent_dim: int
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, b_array, w_array,key, **kwargs):
                
        d_key, e_key, k_key = jax.random.split(key, 3)
        self.latent_dim = d_array[-1]
        
        
        self.encoderD = MLP(d_array, e_key, layer_norm = True)
        self.decoder = MLP(b_array, d_key, layer_norm = True)
        self.encoderP = MLP(w_array, key = k_key, layer_norm = True)
        
    def get_latent(self, x, wi):
        x_w = jnp.concatenate((x, wi), axis = -1)
        return self.encoderD(x_w)
    
    def get_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)
    
    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_diagonal(self, w):
        return jnp.diag(w)
    
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        x_w = jnp.concatenate((x0, w), axis = -1)
        
        latent = self.encoderD(x_w)
        weights = self.encoderP(w)
        koop = self.get_naive(weights)
        
        '''
        # Discrete Koopman:
        def Koopman_layer(carry, hidden = None):
            return self.propagate_fixed_st(None, carry, koop), carry
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)
        '''

        # Continuous Koopman:
        latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, koop)
        predictions = jax.vmap(self.decoder)(latent_array)

        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]


    def l2_loss(self, w, alpha):
        return alpha * jnp.mean(w**2)

    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return jnp.mean(jnp.array([jnp.vdot(x, x) for x in leaves]))
    
    
    @eqx.filter_jit
    def loss_fn(self, ti, yi, wi, vector_field):
        
        # Reconstruction
        latent, y_pred, w_pred = self(ti, yi[0], wi)
        # tile = jnp.tile(wi, (yi.shape[0],  1))
        # yi_w = jnp.concatenate((yi, tile), axis = -1)
        # y_w_pred = jnp.concatenate((y_pred, w_pred), axis = -1)
        reconstruction = jnp.mean(jnp.mean((jnp.sin(yi) - y_pred)**2, axis = -1))
        
        # reconstruction = jnp.mean(jnp.mean((jnp.linalg.norm(yi - y_pred, ord = 2, axis = -1)/jnp.linalg.norm(yi_w, ord=2, axis= -1)), axis = -1))
        
        # Latent Linearity Error:
        latent_traj = jax.vmap(self.get_latent, in_axes=[0, None])(yi, wi)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent - latent_traj), axis = -1))
        # latent_error = jnp.mean(jnp.mean((latent - latent_traj)**2, axis = -1))

        return reconstruction + latent_error, (reconstruction, latent_error) #+ PINN #+ L2_K #+ latent_error #+ 0.01 * inf_norm
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi, vector_field):
        # l2 = self.l2_squared(self.encoderD) + self.l2_squared(self.decoder)
        total_loss, (reconstruction, latent_error) = jax.vmap(self.loss_fn, in_axes = [0, 0, 0, None])(ti, yi, wi, vector_field)
        return jnp.mean(total_loss), (jnp.mean(reconstruction), jnp.mean(latent_error))

    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi, vector_field)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state  
    
    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction

        # latent = jax.vmap(self.get_latent, in_axes=(0, None), )(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        latent_1 = jax.vmap(self.get_latent, in_axes=(0, None))
        latent = jax.vmap(latent_1, in_axes=(0, None))(yi, wi)

        # reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)
        reconstruction_1 = jax.vmap(self.decoder)
        reconstruction = jax.vmap(reconstruction_1)(latent)
        
        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        # Mean Version:
        # weight_reconstruction = jnp.mean((tile - reconstruction[:,:, yi.shape[-1]:]) ** 2, axis=-1)
        # variable_reconstruction = jnp.mean((yi - reconstruction[:, :, :yi.shape[-1]]) ** 2, axis=-1)
        # reconstruction_loss = jnp.mean((yi_w - reconstruction) ** 2, axis=-1)

        # Sum Version:
        weight_reconstruction = jnp.mean(jnp.abs(tile - reconstruction[:,:, yi.shape[-1]:]), axis=-1)
        variable_reconstruction = jnp.mean(jnp.abs(yi - reconstruction[:, :, :yi.shape[-1]]), axis=-1)
        reconstruction_loss = jnp.mean(jnp.abs(yi_w - reconstruction), axis=-1)

        # Compute Jacobian Loss
        # x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        x_dot_1 = jax.vmap(vector_field, in_axes=(None, 0, None))
        x_dot = jax.vmap(x_dot_1, in_axes=(None, 0, None))(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)

        # _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        jvp_1 = jax.vmap(jax.jvp, in_axes=(None, 0, 0))
        _, dFdt = jax.vmap(jvp_1, in_axes=(None, 0, 0))(f, (yi,), (x_dot,))

        latent_1_ = jax.vmap(lambda x: self.Koopman @ x)
        latent_ = jax.vmap(latent_1_)(latent)
        j_loss = jnp.mean(jnp.abs(dFdt - latent_), axis=-1)
        
        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, variable_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig   


class PLearnKoopmanCL(eqx.Module):
    encoderD: MLP
    decoder: MLP
    Koopman: jnp.ndarray
    latent_dim: int
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, b_array, w_array, key, **kwargs):
        
#        assert b_array[-1] + w_array[0] == d_array[0]
        
        d_key, e_key, k_key = jax.random.split(key, 3)
        
        self.latent_dim = d_array[-1]
        
        self.encoderD = MLP(d_array, e_key, layer_norm = True)
        self.decoder = MLP(d_array[::-1], d_key, layer_norm = True)
        
        lim = 1 / jnp.sqrt(d_array[-1])
        self.Koopman = jrandom.uniform(k_key, (d_array[-1], d_array[-1]), minval=-lim, maxval=lim)
        
    def get_latent(self, x, wi):
        x_w = jnp.concatenate((x, wi), axis = -1)
        return self.encoderD(x_w)
    
    def get_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)
    
    def get_K(self, w=None):
        return self.Koopman    
    
    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_diagonal(self, w):
        return jnp.diag(w)
    
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    def l2_squared(tree):
        leaves, _ = jax.tree_util.tree_flatten(tree)
        return sum(jnp.vdot(x, x) for x in leaves)
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        x_w = jnp.concatenate((x0, w), axis = -1)
        latent = self.encoderD(x_w)
        
        # Continuous Koopman:
        latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, self.Koopman)
        
        predictions = jax.vmap(self.decoder)(latent_array)
        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]
    
    
    def loss_fn(self, ti, yi, wi):
        
        # Reconstruction
        latent, y_pred, w_pred = self(ti, yi[0], wi)
        # tile = jnp.tile(wi, (yi.shape[0],  1))
        # yi_w = jnp.concatenate((yi, tile), axis = -1)
        # y_w_pred = jnp.concatenate((y_pred, w_pred), axis = -1)
        # reconstruction = jnp.mean(jnp.mean(jnp.abs(yi_w - y_w_pred), axis = -1))
        reconstruction = jnp.mean(jnp.mean(jnp.abs(y_pred - yi), axis = -1)) 
        
        # Latent Linearity Error:
        latent_traj = jax.vmap(self.get_latent, in_axes=[0, None])(yi, wi)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent - latent_traj), axis = -1))
        
        return reconstruction + latent_error, (reconstruction, latent_error) #+ PINN #+ L2_K #+ latent_error #+ 0.01 * inf_norm
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi):
        # l2 = self.l2_squared(self.encoderD) + self.l2_squared(self.decoder)
        total_loss, (reconstruction, latent_error) = jax.vmap(self.loss_fn, in_axes = [0, 0, 0])(ti, yi, wi)

        # lipschitz_loss_encoderD = self.encoderD.get_lipschitz_loss()
        # lipschitz_loss_encoderP = self.encoderP.get_lipschitz_loss()
        # lipschitz_loss_decoder = self.decoder.get_lipschitz_loss()

        return jnp.mean(total_loss), {"Reconstruction": jnp.mean(reconstruction), 
                                     "Latent Error": jnp.mean(latent_error),
                                    #  "Lipschitz Encoder Loss": lipschitz_loss_encoderD,
                                    #  "Lipschitz K encoder Loss": lipschitz_loss_encoderP,
                                     }
    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state  
    
    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction

        # latent = jax.vmap(self.get_latent, in_axes=(0, None), )(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        latent_1 = jax.vmap(self.get_latent, in_axes=(0, None))
        latent = jax.vmap(latent_1, in_axes=(0, None))(yi, wi)

        # reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)
        reconstruction_1 = jax.vmap(self.decoder)
        reconstruction = jax.vmap(reconstruction_1)(latent)
        
        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        # Mean Version:
        # weight_reconstruction = jnp.mean((tile - reconstruction[:,:, yi.shape[-1]:]) ** 2, axis=-1)
        # variable_reconstruction = jnp.mean((yi - reconstruction[:, :, :yi.shape[-1]]) ** 2, axis=-1)
        # reconstruction_loss = jnp.mean((yi_w - reconstruction) ** 2, axis=-1)

        # Sum Version:
        weight_reconstruction = jnp.mean(jnp.abs(tile - reconstruction[:,:, yi.shape[-1]:]), axis=-1)
        variable_reconstruction = jnp.mean(jnp.abs(yi - reconstruction[:, :, :yi.shape[-1]]), axis=-1)
        reconstruction_loss = jnp.mean(jnp.abs(yi_w - reconstruction), axis=-1)

        # Compute Jacobian Loss
        # x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        x_dot_1 = jax.vmap(vector_field, in_axes=(None, 0, None))
        x_dot = jax.vmap(x_dot_1, in_axes=(None, 0, None))(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)

        # _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        jvp_1 = jax.vmap(jax.jvp, in_axes=(None, 0, 0))
        _, dFdt = jax.vmap(jvp_1, in_axes=(None, 0, 0))(f, (yi,), (x_dot,))

        latent_1_ = jax.vmap(lambda x: self.Koopman @ x)
        latent_ = jax.vmap(latent_1_)(latent)
        j_loss = jnp.mean(jnp.abs(dFdt - latent_), axis=-1)
        
        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, variable_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig   
    

class PLearnKoopmanLipCL(eqx.Module):
    encoderD: MLP
    decoder: MLP
    Koopman: jnp.ndarray
    latent_dim: int
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, b_array, w_array, key, **kwargs):
        
#        assert b_array[-1] + w_array[0] == d_array[0]
        
        d_key, e_key, k_key = jax.random.split(key, 3)
        
        self.latent_dim = d_array[-1]
        
        self.encoderD = LipMLP(d_array, e_key, layer_norm = False)
        self.decoder = MLP(d_array[::-1], d_key, layer_norm = True)
        
        lim = 1 / jnp.sqrt(d_array[-1])
        self.Koopman = jrandom.uniform(k_key, (d_array[-1], d_array[-1]), minval=-lim, maxval=lim)
        
    def get_latent(self, x, wi):
        x_w = jnp.concatenate((x, wi), axis = -1)
        return self.encoderD(x_w)
    
    def get_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)
    
    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_diagonal(self, w):
        return jnp.diag(w)
    
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    def l2_squared(tree):
        leaves, _ = jax.tree_util.tree_flatten(tree)
        return sum(jnp.vdot(x, x) for x in leaves)
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        x_w = jnp.concatenate((x0, w), axis = -1)
        latent = self.encoderD(x_w)
        # # Continuous Koopman:
        latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, self.Koopman)
        predictions = jax.vmap(self.decoder)(latent_array)
        
        # Discrete Koopman:
        # def Koopman_layer(carry, hidden = None):
        #     return self.propagate_fixed_st(None, carry, self.Koopman), carry
        # _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = len(ts))
        # predictions = jax.vmap(self.decoder)(latent_array)
             
        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]
    
    
    def loss_fn(self, ti, yi, wi):
        
        # Reconstruction
        latent, y_pred, w_pred = self(ti, yi[0], wi)
        # tile = jnp.tile(wi, (yi.shape[0],  1))
        # yi_w = jnp.concatenate((yi, tile), axis = -1)
        # y_w_pred = jnp.concatenate((y_pred, w_pred), axis = -1)
        # reconstruction = jnp.mean(jnp.mean(jnp.abs(yi_w - y_w_pred), axis = -1))
        reconstruction = jnp.mean(jnp.mean(jnp.abs(y_pred - yi), axis = -1))         
        # Latent Linearity Error:
        latent_traj = jax.vmap(self.get_latent, in_axes=[0, None])(yi, wi)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent - latent_traj), axis = -1))
        
        return reconstruction + latent_error, (reconstruction, latent_error) #+ PINN #+ L2_K #+ latent_error #+ 0.01 * inf_norm
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi):
        # l2 = self.l2_squared(self.encoderD) + self.l2_squared(self.decoder)
        total_loss, (reconstruction, latent_error) = jax.vmap(self.loss_fn, in_axes = [0, 0, 0])(ti, yi, wi)

        lipschitz_loss_encoderD = self.encoderD.get_lipschitz_loss()
        # lipschitz_loss_encoderP = self.encoderP.get_lipschitz_loss()
        # lipschitz_loss_decoder = self.decoder.get_lipschitz_loss()

        return jnp.mean(total_loss) + 0.1 * lipschitz_loss_encoderD, {"Reconstruction": jnp.mean(reconstruction), 
                                     "Latent Error": jnp.mean(latent_error),
                                     "Lipschitz Encoder Loss": lipschitz_loss_encoderD,
                                    #  "Lipschitz K encoder Loss": lipschitz_loss_encoderP,
                                     }
    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state  
    
    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction

        # latent = jax.vmap(self.get_latent, in_axes=(0, None), )(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        latent_1 = jax.vmap(self.get_latent, in_axes=(0, None))
        latent = jax.vmap(latent_1, in_axes=(0, None))(yi, wi)

        # reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)
        reconstruction_1 = jax.vmap(self.decoder)
        reconstruction = jax.vmap(reconstruction_1)(latent)
        
        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        # Mean Version:
        # weight_reconstruction = jnp.mean((tile - reconstruction[:,:, yi.shape[-1]:]) ** 2, axis=-1)
        # variable_reconstruction = jnp.mean((yi - reconstruction[:, :, :yi.shape[-1]]) ** 2, axis=-1)
        # reconstruction_loss = jnp.mean((yi_w - reconstruction) ** 2, axis=-1)

        # Sum Version:
        weight_reconstruction = jnp.mean(jnp.abs(tile - reconstruction[:,:, yi.shape[-1]:]), axis=-1)
        variable_reconstruction = jnp.mean(jnp.abs(yi - reconstruction[:, :, :yi.shape[-1]]), axis=-1)
        reconstruction_loss = jnp.mean(jnp.abs(yi_w - reconstruction), axis=-1)

        # Compute Jacobian Loss
        # x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        x_dot_1 = jax.vmap(vector_field, in_axes=(None, 0, None))
        x_dot = jax.vmap(x_dot_1, in_axes=(None, 0, None))(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)

        # _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        jvp_1 = jax.vmap(jax.jvp, in_axes=(None, 0, 0))
        _, dFdt = jax.vmap(jvp_1, in_axes=(None, 0, 0))(f, (yi,), (x_dot,))

        latent_1_ = jax.vmap(lambda x: self.Koopman @ x)
        latent_ = jax.vmap(latent_1_)(latent)
        j_loss = jnp.mean(jnp.abs(dFdt - latent_), axis=-1)
        
        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, variable_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig   
    

    
class PLearnKoopmanTrajFree(eqx.Module):
    encoderD: MLP
    decoder: MLP
    Koopman: jnp.ndarray
    latent_dim: int
    
    # Architecture that learns the Koopman operator from random points and no trajectories. 
    # [x_0, w] -> encoder(x_0, w) -> latent -> Koopman @ latent -> Jakobian loss.
    #                                       -> decoder(latent) -> [x_0, w] reconstruction.
    
    def __init__(self, d_array,key, **kwargs):
        
        # assert b_array[-1] + w_array[0] == d_array[0]
        
        d_key, e_key, k_key = jax.random.split(key, 3)
        
        self.latent_dim = d_array[-1]
        
        self.encoderD = MLP(d_array, e_key, layer_norm = True)
        self.decoder = MLP(d_array[::-1], d_key, layer_norm = True)
        
        lim = 1 / jnp.sqrt(d_array[-1])
        self.Koopman = jrandom.uniform(k_key, (d_array[-1], d_array[-1]), minval=-lim, maxval=lim)
        
    def get_latent(self, x, wi):
        x_w = jnp.concatenate((x, wi), axis = -1)
        return self.encoderD(x_w)
    
    def reconstruct(self, latent):
        x_w = jnp.concatenate((x, wi), axis = -1)
        latent = self.encoderD(x_w)
        x_w_pred = self.decoder(latent)
        
        return x_w_pred
    
    def get_stable_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)
    
    def get_naive_Koopman(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_diagonal(self, w):
        return jnp.diag(w)
    
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        x_w = jnp.concatenate((x0, w), axis = -1)
        latent = self.encoderD(x_w)
        
        # Continuous Koopman:
        latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, self.Koopman)

        #----------------------------------------
        # Discrete Koopman:
        # dt = ts[1] - ts[0]
        # nds = len(ts)  
        
        # def Koopman_layer(carry, hidden = None):
        #     return carry + self.propagate_fixed_st(None, carry, dt * self.Koopman), carry
        # _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)
        #----------------------------------------
        
        predictions = jax.vmap(self.decoder)(latent_array)
        
        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]
    
    
    def l2_loss(self, w, alpha):
        return alpha * jnp.mean(w**2)
    
    @staticmethod
    def l2_squared(tree):
        leaves, _ = jax.tree_util.tree_flatten(tree)
        return jnp.mean(jnp.array([jnp.vdot(x, x) for x in leaves]))
    
    def loss_fn(self, ti, yi, wi, vector_field):
        
        # Reconstruction:
        yi_w = jnp.concatenate((yi, wi), axis = -1)
        latent = self.encoderD(yi_w)
        y_w_pred = self.decoder(latent)
        
        reconstruction = jnp.mean(jnp.mean((yi_w - y_w_pred)**2, axis = -1))
        
        # Jacobian Loss:
        x_dot = vector_field(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)
        
        _, dFdt = jax.jvp(f, (yi,), (x_dot,))
        latent_ = self.Koopman @ latent
        j_loss = jnp.mean(jnp.mean((dFdt - latent_)**2, axis = -1))
        
        # 2nd Jacobian Loss:
        _, dFidt = jax.jvp(self.decoder, (latent,), (latent_,))
        zeros = jnp.zeros_like(wi)
        x_dot_ = jnp.concatenate((x_dot, zeros), axis = -1)
        j2_loss = jnp.mean(jnp.mean((dFidt - x_dot_)**2, axis = -1))
        
        return reconstruction + j_loss + j2_loss, reconstruction, j_loss, j2_loss #+ PINN # + latent_error #+ 0.01 * inf_norm
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi, vector_field):
        # l2 = self.l2_squared(self.decoder) + self.l2_squared(self.encoderD)
        l2_norm = jnp.linalg.norm(self.Koopman, ord = 'fro')
        map1 = jax.vmap(self.loss_fn, in_axes = [None, 0, None, None])
        loss, reconstruction, j_loss, j2_loss = jax.vmap(map1, in_axes = [None, 0, 0, None])(ti, yi, wi, vector_field)
        return jnp.mean(loss) + l2_norm, (jnp.mean(reconstruction), jnp.mean(j_loss), jnp.mean(j2_loss), l2_norm)

    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi, vector_field)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state   
    
    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction

        # latent = jax.vmap(self.get_latent, in_axes=(0, None), )(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        latent_1 = jax.vmap(self.get_latent, in_axes=(0, None))
        latent = jax.vmap(latent_1, in_axes=(0, None))(yi, wi)

        # reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)
        reconstruction_1 = jax.vmap(self.decoder)
        reconstruction = jax.vmap(reconstruction_1)(latent)
        
        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        # Mean Version:
        # weight_reconstruction = jnp.mean((tile - reconstruction[:,:, yi.shape[-1]:]) ** 2, axis=-1)
        # variable_reconstruction = jnp.mean((yi - reconstruction[:, :, :yi.shape[-1]]) ** 2, axis=-1)
        # reconstruction_loss = jnp.mean((yi_w - reconstruction) ** 2, axis=-1)

        # Sum Version:
        weight_reconstruction = jnp.mean(jnp.abs(tile - reconstruction[:,:, yi.shape[-1]:]), axis=-1)
        variable_reconstruction = jnp.mean(jnp.abs(yi - reconstruction[:, :, :yi.shape[-1]]), axis=-1)
        reconstruction_loss = jnp.mean(jnp.abs(yi_w - reconstruction), axis=-1)

        # Compute Jacobian Loss
        # x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        x_dot_1 = jax.vmap(vector_field, in_axes=(None, 0, None))
        x_dot = jax.vmap(x_dot_1, in_axes=(None, 0, None))(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)

        # _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        jvp_1 = jax.vmap(jax.jvp, in_axes=(None, 0, 0))
        _, dFdt = jax.vmap(jvp_1, in_axes=(None, 0, 0))(f, (yi,), (x_dot,))

        latent_1_ = jax.vmap(lambda x: self.Koopman @ x)
        latent_ = jax.vmap(latent_1_)(latent)
        j_loss = jnp.mean(jnp.abs(dFdt - latent_), axis=-1)
        
        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, variable_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig   


class PLearnKoopmanMS(eqx.Module):
    encoderD: MLP
    encoderP: MLP
    decoder: MLP
    latent_dim: int
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, b_array, w_array,key, **kwargs):
        
#        assert b_array[-1] + w_array[0] == d_array[0]
        
        d_key, e_key, k_key = jax.random.split(key, 3)
        
        self.latent_dim = d_array[-1]
        
        self.encoderD = MLP(d_array, e_key)
        self.decoder = MLP(b_array, d_key)
        self.encoderP = MLP(w_array, key = k_key)
        
    def get_latent(self, x, wi):
        x_w = jnp.concatenate((x, wi), axis = -1)
        return self.encoderD(x_w)
    
    def get_stable_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)

    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_Koopman(self, p):
        w = self.encoderP(p)
        return self.get_naive(w)
        
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    @eqx.filter_jit
    def __call__(self,ts, x, w):
        
        x_w = jnp.concatenate((x, w), axis = -1)
        latent = self.encoderD(x_w)
        weights = self.encoderP(w)
        koop = self.get_naive(weights)
        
        latent_ = self.propagate(ts, latent, koop)
        reconstruction = self.decoder(latent)
        prediction = self.decoder(latent_)
        
        return latent, latent_, reconstruction, prediction

    @eqx.filter_jit
    def predict(self, x, w, nds):
        
        x_w = jnp.concatenate((x, w), axis = -1)
        
        latent = self.encoderD(x_w)
        weights = self.encoderP(w)
        koop = self.get_naive(weights)
        
        def Koopman_layer(carry, hidden = None):
            return self.propagate(None, carry, koop), carry
        
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)

        # latent_ = self.propagate_fixed_st(None, latent, koop)
        reconstruction = self.decoder(latent)
        predictions = jax.vmap(self.decoder)(latent_array)
            
        return reconstruction, predictions
    
    def l2_squared(self):
        leaves, _ = jax.tree_util.tree_flatten(self)
        return sum(jnp.vdot(x, x) for x in leaves)
    
    def loss_fn(self, ti, yi, wi, vector_field):
        
        # Reconstruction
        ts = ti[1:] - ti[:-1]
        latent, latent_, reconstruction, prediction = jax.vmap(self, in_axes=[0, 0, None])(ts, yi, wi)
        prediction_error = jnp.mean((yi[1:] - prediction[:-1]) ** 2)
        reconstruction = jnp.mean((yi - reconstruction) ** 2)
        
        # Latent Linearity Error:
        latent_error = jnp.mean((latent[1:] - latent_[:-1])**2)
        
        # Jacobina Loss:
        # x_dot = jax.vmap(vector_field, (None, 0, None))(None, yi, wi)

        # get_latent = lambda x: self.get_latent(x, wi)
        
        # def f(x, x_dot):
        #     return jax.jvp(get_latent, (x,), (x_dot,))
        # _, dFdt  = jax.vmap(f, in_axes = (0, 0))(yi, x_dot)
        
        # def Koop_latent(latent, koop):
        #     return koop @ latent
    
        # koop = self.get_Koopman(wi)
        
        # latent_koop = jax.vmap(Koop_latent, in_axes = [0, None])(latent, koop)
        # j_loss = jnp.mean((dFdt - latent_koop)**2)
    
        
        return reconstruction + prediction_error +  latent_error #+ j_loss
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad
    def batch_loss(self, ti, yi, wi, vector_field):
        l2 = self.l2_squared()
        return jnp.mean(jax.vmap(self.loss_fn, in_axes = [None, 0, 0, None])(ti, yi, wi, vector_field)) + 0.001 * l2


    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        loss, grads = self.batch_loss(ti, yi, wi, vector_field)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return loss, model, opt_state
    

class PLearnKoopmanMS_WI(eqx.Module):
    encoderD: MLP
    Koopman: jnp.ndarray
    decoder: MLP
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, key, **kwargs):
        
        d_key, e_key, k_key = jax.random.split(key, 3)
                
        self.encoderD = MLP(d_array, e_key, layer_norm = True)
        self.decoder = MLP(d_array[::-1], d_key, layer_norm = True)
        
        lim = 1 / jnp.sqrt(d_array[-1])
        self.Koopman = jrandom.uniform(k_key, (d_array[-1], d_array[-1]), minval=-lim, maxval=lim)
        
        
    def get_latent(self, x, wi):
        x_w = jnp.concatenate((x, wi), axis = -1)
        return self.encoderD(x_w)
    
    def get_stable_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)

    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_Koopman(self, p):
        w = self.encoderP(p)
        return self.get_naive(w)
        
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        x_w = jnp.concatenate((x0, w), axis = -1)
        latent = self.encoderD(x_w)
        
        # # Continuous Koopman:
        latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, self.Koopman)
        predictions = jax.vmap(self.decoder)(latent_array)
        # Discrete Koopman:
        
        # dt = ts[1] - ts[0]
        # nds = len(ts) - 1 
        
        # def Koopman_layer(carry, hidden = None):
        #     return self.propagate_fixed_st(None, carry, self.Koopman), carry
        
        # _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)
        
        predictions = jax.vmap(self.decoder)(latent_array)
        
        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]

    @eqx.filter_jit
    def predict(self, x, w, nds):
        
        x_w = jnp.concatenate((x, w), axis = -1)
        
        latent = self.encoderD(x_w)
        weights = self.encoderP(w)
        koop = self.get_naive(weights)
        
        def Koopman_layer(carry, hidden = None):
            return self.propagate(None, carry, koop), carry
        
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)

        # latent_ = self.propagate_fixed_st(None, latent, koop)
        reconstruction = self.decoder(latent)
        predictions = jax.vmap(self.decoder)(latent_array)
            
        return reconstruction, predictions
    
    def l2_squared(tree):
        leaves, _ = jax.tree_util.tree_flatten(tree)
        return sum(jnp.vdot(x, x) for x in leaves)
    
    def loss_fn(self, ti, yi, wi, forward, vector_field):
        
        # Reconstruction
        tile = jnp.tile(wi, (yi.shape[0],  1))
        x_w = jnp.concatenate((yi, tile), axis = -1)
        
        ts = ti[1:] - ti[:-1]
        latent = jax.vmap(self.get_latent, in_axes = [0, None])(yi, wi)
        
        latent_ = jax.vmap(self.propagate, in_axes = [0, 0, None])(ts, latent[:-1], self.Koopman)
        
        prediction = jax.vmap(self.decoder)(latent_)
        reconstruction = jax.vmap(self.decoder)(latent)
        
        # prediction_error = jnp.mean((x_w[1:] - prediction) ** 2)
        # reconstruction_error = jnp.mean((x_w - reconstruction) ** 2)
    
        prediction_traj_error = jnp.mean(jnp.abs(x_w[1:] - prediction), axis =-1)
        prediction_error = jnp.mean(prediction_traj_error)
        reconstruction_error = jnp.mean(jnp.mean(jnp.abs(x_w - reconstruction), axis = -1))

        # Latent Linearity Error:
        # latent_error = jnp.mean((latent[1:] - latent_)**2)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent[1:] - latent_), axis = -1))

        
        # Jacobina Loss:
        # x_dot = jax.vmap(vector_field, (None, 0, None))(None, yi, wi)

        # get_latent = lambda x: self.get_latent(x, wi)
        
        # def f(x, x_dot):
        #     return jax.jvp(get_latent, (x,), (x_dot,))
        # _, dFdt  = jax.vmap(f, in_axes = (0, 0))(yi, x_dot)
        
        # def Koop_latent(latent, koop):
        #     return koop @ latent
    
        # koop = self.get_Koopman(wi)
        
        # latent_koop = jax.vmap(Koop_latent, in_axes = [0, None])(latent, koop)
        # j_loss = jnp.mean((dFdt - latent_koop)**2)
        
        return reconstruction_error + latent_error + prediction_error, reconstruction_error, latent_error, prediction_error #+ j_loss
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi, vector_field):
        #l2 = self.l2_squared(self.encoderD) + self.l2_squared(self.decoder)
        total_loss, reconstruction_error, latent_error, prediction_error = jax.vmap(self.loss_fn, in_axes = [0, 0, 0, None])(ti, yi, wi, vector_field)
        return jnp.mean(total_loss), (jnp.mean(reconstruction_error), jnp.mean(latent_error), jnp.mean(prediction_error)) #+ 0.01 * l2

    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi, vector_field)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state

    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction

        # latent = jax.vmap(self.get_latent, in_axes=(0, None), )(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        latent_1 = jax.vmap(self.get_latent, in_axes=(0, None))
        latent = jax.vmap(latent_1, in_axes=(0, None))(yi, wi)

        # reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)
        reconstruction_1 = jax.vmap(self.decoder)
        reconstruction = jax.vmap(reconstruction_1)(latent)
        
        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        # Mean Version:
        # weight_reconstruction = jnp.mean((tile - reconstruction[:,:, yi.shape[-1]:]) ** 2, axis=-1)
        # variable_reconstruction = jnp.mean((yi - reconstruction[:, :, :yi.shape[-1]]) ** 2, axis=-1)
        # reconstruction_loss = jnp.mean((yi_w - reconstruction) ** 2, axis=-1)

        # Sum Version:
        weight_reconstruction = jnp.mean(jnp.abs(tile - reconstruction[:,:, yi.shape[-1]:]), axis=-1)
        variable_reconstruction = jnp.mean(jnp.abs(yi - reconstruction[:, :, :yi.shape[-1]]), axis=-1)
        reconstruction_loss = jnp.mean(jnp.abs(yi_w - reconstruction), axis=-1)

        # Compute Jacobian Loss
        # x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        x_dot_1 = jax.vmap(vector_field, in_axes=(None, 0, None))
        x_dot = jax.vmap(x_dot_1, in_axes=(None, 0, None))(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)

        # _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        jvp_1 = jax.vmap(jax.jvp, in_axes=(None, 0, 0))
        _, dFdt = jax.vmap(jvp_1, in_axes=(None, 0, 0))(f, (yi,), (x_dot,))

        latent_1_ = jax.vmap(lambda x: self.Koopman @ x)
        latent_ = jax.vmap(latent_1_)(latent)
        j_loss = jnp.mean(jnp.abs(dFdt - latent_), axis=-1)
        
        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, variable_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig
    
class PLearnKoopmanMS_WI(eqx.Module):
    encoderD: MLP
    Koopman: jnp.ndarray
    decoder: MLP
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, key, **kwargs):
        
        d_key, e_key, k_key = jax.random.split(key, 3)
                
        self.encoderD = MLP(d_array, e_key, layer_norm = True)
        self.decoder = MLP(d_array[::-1], d_key, layer_norm = True)
        
        lim = 1 / jnp.sqrt(d_array[-1])
        self.Koopman = jrandom.uniform(k_key, (d_array[-1], d_array[-1]), minval=-lim, maxval=lim)
        
        
    def get_latent(self, x, wi):
        x_w = jnp.concatenate((x, wi), axis = -1)
        return self.encoderD(x_w)
    
    def get_stable_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)

    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_Koopman(self, p):
        w = self.encoderP(p)
        return self.get_naive(w)
        
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        x_w = jnp.concatenate((x0, w), axis = -1)
        latent = self.encoderD(x_w)
        
        def Koopman_layer(carry, hidden = None):
            return self.propagate_fixed_st(None, carry, self.Koopman), carry
        
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = len(ts))

        # latent_ = self.propagate_fixed_st(None, latent, koop)
        predictions = jax.vmap(self.decoder)(latent_array)
            
        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]

    def predict_next(self, x_w, ts):
            
            latent = self.encoderD(x_w)
            latent_ = self.propagate(ts, latent, self.Koopman)
            prediction = self.decoder(latent_)
            
            return prediction
        
    @eqx.filter_jit
    def predict(self, x, w, nds):
        
        x_w = jnp.concatenate((x, w), axis = -1)
        
        latent = self.encoderD(x_w)
        
        def Koopman_layer(carry, hidden = None):
            return self.predict_next(None, carry, self.Koopman), carry
        
        _, predictions =  jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)

        # latent_ = self.propagate_fixed_st(None, latent, koop)
        reconstruction = self.decoder(latent)
        predictions = jnp.concatenate((reconstruction, predictions), axis = 0)
        
        return predictions[:, :len(x)], predictions[:, len(x):]
    
    @staticmethod
    def l2_squared(tree):
        leaves, _ = jax.tree_util.tree_flatten(tree)
        return jnp.mean(jnp.array([jnp.vdot(x, x) for x in leaves]))
    
    @staticmethod
    def split_array_with_kernel(arr: jnp.ndarray, kernel_size: int) -> jnp.ndarray:
        """
        Splits a JAX array into overlapping segments of size kernel_size.
        
        Parameters:
            arr (jnp.ndarray): The array to split.
            kernel_size (int): The size of the kernel.
        
        Returns:
            jnp.ndarray: A JAX array containing overlapping segments.
        """
        return jnp.stack([arr[i:i+kernel_size] for i in range(arr.shape[0]-kernel_size+1)], axis=0)

    def loss_fn(self, ti, yi, wi, vector_field, forward = 1):
        
        '''
        
        Parameters:
        ti: Time points (1D)
        yi: Observations (2D)
        wi: Additional arguments for the vector field (1D)
        vector_field: Function to compute the vector field 
        forward: Number of steps to predict forward (int)
        
        '''        
        # Reconstruction
        tile = jnp.tile(wi, (yi.shape[0],  1))
        x_w = jnp.concatenate((yi, tile), axis = -1)
        
        ts = ti[1:] - ti[:-1]
        
        # if forward > 1:
        # split_mapping = jax.vmap(s, in_axes = [0, None])
        if forward > 1:
            x_w_split = self.split_array_with_kernel(x_w, forward)
            x_split = self.split_array_with_kernel(yi, forward)
            
            latent_map = jax.vmap(self.get_latent, in_axes = [0, None])
            latent = jax.vmap(latent_map, in_axes = [0, None])(x_split, wi)
                    
            def Koopman_prop(carry, ts):
                return self.propagate_fixed_st(ts, carry, self.Koopman), carry
            
            # scan_over = jax.lax.scan(Koopman_prop, latent, xs = forward)
            def scan_Koop(latent, ts):
                _, latent_array = jax.lax.scan(Koopman_prop, latent, xs = None, length = ts)
                return latent_array
            
            
            scannable = latent[:, 0, :]
            latent_ = jax.vmap(scan_Koop, in_axes = [0, None])(scannable, forward)
            
            prediction_map = jax.vmap(self.decoder)
            prediction = jax.vmap(prediction_map, in_axes = [0])(latent_)
            
            prediction_traj_error = jnp.mean(jnp.abs(x_w_split - prediction), axis =-1)
            prediction_error = jnp.mean(jnp.mean(prediction_traj_error, axis=-1), axis = -1)
            
            latent_error = jnp.mean(jnp.mean(jnp.mean(jnp.abs(latent[:, 1:, :] - latent_[:, :-1, :]), axis = -1), axis = -1), axis = -1)
        
        else:
            
            latent = jax.vmap(self.get_latent, in_axes = [0, None])(yi, wi)
            
            
            latent_ = jax.vmap(self.propagate_fixed_st, in_axes = [None, 0, None])(None, latent[:-1], self.Koopman)
            
            prediction = jax.vmap(self.decoder)(latent_)
            reconstruction = jax.vmap(self.decoder)(latent)
            
            prediction_traj_error = jnp.mean(jnp.abs(x_w[1:] - prediction), axis =-1)
            prediction_error = jnp.mean(prediction_traj_error)
            
            reconstruction_error = jnp.mean(jnp.mean(jnp.abs(x_w - reconstruction), axis = -1))
            prediction_error = prediction_error + reconstruction_error
            
            latent_error = jnp.mean(jnp.mean(jnp.abs(latent[1:] - latent_), axis = -1))
        
        
        return latent_error + prediction_error, prediction_error, latent_error 
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi, vector_field):
        l2 = self.l2_squared(self.encoderD) + self.l2_squared(self.decoder)
        # spectral_norm = jnp.linalg.norm(self.Koopman, ord = 2)
        total_loss, prediction_error, latent_error = jax.vmap(self.loss_fn, in_axes = [None, 0, 0, None, None])(ti, yi, wi, vector_field, 1)
        return jnp.mean(total_loss) + 0.01 * l2, (jnp.mean(prediction_error), jnp.mean(latent_error), 0.01 * l2) #+ 0.01 * l2

    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi, vector_field)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state

    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction

        # latent = jax.vmap(self.get_latent, in_axes=(0, None), )(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        latent_1 = jax.vmap(self.get_latent, in_axes=(0, None))
        latent = jax.vmap(latent_1, in_axes=(0, None))(yi, wi)

        # reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)
        reconstruction_1 = jax.vmap(self.decoder)
        reconstruction = jax.vmap(reconstruction_1)(latent)
        
        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        # Mean Version:
        # weight_reconstruction = jnp.mean((tile - reconstruction[:,:, yi.shape[-1]:]) ** 2, axis=-1)
        # variable_reconstruction = jnp.mean((yi - reconstruction[:, :, :yi.shape[-1]]) ** 2, axis=-1)
        # reconstruction_loss = jnp.mean((yi_w - reconstruction) ** 2, axis=-1)

        # Sum Version:
        weight_reconstruction = jnp.mean(jnp.abs(tile - reconstruction[:,:, yi.shape[-1]:]), axis=-1)
        variable_reconstruction = jnp.mean(jnp.abs(yi - reconstruction[:, :, :yi.shape[-1]]), axis=-1)
        reconstruction_loss = jnp.mean(jnp.abs(yi_w - reconstruction), axis=-1)

        # Compute Jacobian Loss
        # x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        x_dot_1 = jax.vmap(vector_field, in_axes=(None, 0, None))
        x_dot = jax.vmap(x_dot_1, in_axes=(None, 0, None))(None, yi, wi)

        def f(x):
            return self.get_latent(x, wi)

        # _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        jvp_1 = jax.vmap(jax.jvp, in_axes=(None, 0, 0))
        _, dFdt = jax.vmap(jvp_1, in_axes=(None, 0, 0))(f, (yi,), (x_dot,))

        latent_1_ = jax.vmap(lambda x: self.Koopman @ x)
        latent_ = jax.vmap(latent_1_)(latent)
        j_loss = jnp.mean(jnp.abs(dFdt - latent_), axis=-1)
        
        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, variable_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig
    
class PLearnKoopmanMS_ResNet(eqx.Module):
    encoderD: MLP
    Koopman: jnp.ndarray
    decoder: MLP
    
    # Architecture that learns the Koopman operator from parameters.
    
    def __init__(self, d_array, key, **kwargs):
        
        d_key, e_key, k_key = jax.random.split(key, 3)
                
        self.encoderD = MLP(d_array, e_key, layer_norm = True)
        self.decoder = MLP(d_array[::-1], d_key, layer_norm = True)
        
        lim = 1 / jnp.sqrt(d_array[-1])
        self.Koopman = jrandom.uniform(k_key, (d_array[-1], d_array[-1]), minval=-lim, maxval=lim)
        
        
    def get_latent(self, x, wi):
        x_w = jnp.concatenate((x, wi), axis = -1)
        return self.encoderD(x_w)
    
    def get_stable_Koopman(self, w):
        K_D = -(w.shape[0]//-2)
        return jnp.diag(-jnp.square(w[:K_D])) - jnp.diag(w[K_D:], k=1) + jnp.diag(w[K_D:], k=-1)

    def get_naive(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
    def get_Koopman(self, p):
        w = self.encoderP(p)
        return self.get_naive(w)
        
    def propagate(self, t, x, w):
        return jax.scipy.linalg.expm(w * t) @ x
    
    def propagate_fixed_st(self, t, x, w):
        return w @ x
    
    @eqx.filter_jit
    def __call__(self, ts, x0, w):
        
        x_w = jnp.concatenate((x0, w), axis = -1)
        latent = self.encoderD(x_w)
        
        # # Continuous Koopman:
        # latent_array = jax.vmap(self.propagate, in_axes = [0, None, None])(ts, latent, self.Koopman)
        # predictions = jax.vmap(self.decoder)(latent_array)
        # Discrete Koopman:
        
        dt = ts[1] - ts[0]
        nds = len(ts) 
        
        def Koopman_layer(carry, hidden = None):
            return carry + self.propagate_fixed_st(None, carry, self.Koopman), carry
        
        _, latent_array =  jax.lax.scan(Koopman_layer, latent, xs = None, length = nds)
        
        predictions = jax.vmap(self.decoder)(latent_array)
        
        return latent_array, predictions[:, :len(x0)], predictions[:, len(x0):]

    @staticmethod
    def l2_squared(tree):
        leaves, _ = jax.tree_util.tree_flatten(tree)
        return sum(jnp.vdot(x, x) for x in leaves)
    
    def loss_fn(self, ti, yi, wi, vector_field):
        
        # Reconstruction
        tile = jnp.tile(wi, (yi.shape[0],  1))
        x_w = jnp.concatenate((yi, tile), axis = -1)
        
        ts = ti[1:] - ti[:-1]
        latent = jax.vmap(self.get_latent, in_axes = [0, None])(yi, wi)
        # latent_ = jax.vmap(self.propagate, in_axes = [0, 0, None])(ts, latent[:-1], self.Koopman)
        d_latent = jax.vmap(self.propagate_fixed_st, in_axes = [None, 0, None])(ts, latent[:-1], self.Koopman)

        latent_ = latent[:-1] + d_latent
        
        prediction = jax.vmap(self.decoder)(latent_)
        reconstruction = jax.vmap(self.decoder)(latent)
        
        # prediction_error = jnp.mean((x_w[1:] - prediction) ** 2)
        # reconstruction_error = jnp.mean((x_w - reconstruction) ** 2)
    
        prediction_traj_error = jnp.mean(jnp.abs(x_w[1:] - prediction), axis =-1)
        prediction_error = jnp.mean(prediction_traj_error)
        reconstruction_error = jnp.mean(jnp.mean(jnp.abs(x_w - reconstruction), axis = -1))

        # Latent Linearity Error:
        # latent_error = jnp.mean((latent[1:] - latent_)**2)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent[1:] - latent_), axis = -1))
        
        # Jacobina Loss:
        # x_dot = jax.vmap(vector_field, (None, 0, None))(None, yi, wi)

        # get_latent = lambda x: self.get_latent(x, wi)
        
        # def f(x, x_dot):
        #     return jax.jvp(get_latent, (x,), (x_dot,))
        # _, dFdt  = jax.vmap(f, in_axes = (0, 0))(yi, x_dot)
        
        # def Koop_latent(latent, koop):
        #     return koop @ latent
    
        # koop = self.get_Koopman(wi)
        
        # latent_koop = jax.vmap(Koop_latent, in_axes = [0, None])(latent, koop)
        # j_loss = jnp.mean((dFdt - latent_koop)**2)
        
        return reconstruction_error + latent_error + prediction_error, reconstruction_error, latent_error, prediction_error #+ prediction_error #+ j_loss
    
    def l2_Koopman(w):
        jnp.mean(w**2)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi, vector_field):
        l2 = self.l2_squared(self.encoderD) + self.l2_squared(self.decoder)
        total_loss, reconstruction_error, latent_error, prediction_error = jax.vmap(self.loss_fn, in_axes = [None, 0, 0, None])(ti, yi, wi, vector_field)
        return jnp.mean(total_loss) + 0.001 * l2, (jnp.mean(reconstruction_error), jnp.mean(latent_error), jnp.mean(prediction_error), 0.001 * l2) 


    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state, vector_field):
        losses, grads = self.batch_loss(ti, yi, wi, vector_field)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state

    def plot_loss_landscape(self, lb, ub, wi, vector_field, n=100):
        """
        Create a heatmap of the loss landscape.
        
        Parameters:
            ti: Time points
            lb: Lower bound for the plot
            ub: Upper bound for the plot
            wi: Additional arguments for the vector field
            vector_field: Function to compute the vector field
            ax: Matplotlib axis
            n: Number of points in the grid (optional)
        """
        
        # Generate a 2D grid for the x & y bounds
        y1, y2 = jnp.meshgrid(jnp.linspace(lb, ub, n), jnp.linspace(lb, ub, n))
        yi = jnp.stack([y1, y2], axis=-1)
        # Calculate latent and reconstruction
        latent = jax.vmap(self.get_latent, in_axes=(0, None))(yi.reshape(-1, 2), wi).reshape(n, n, -1)
        reconstruction = jax.vmap(self.decoder)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, 6)

        # Compute Reconstruction loss
        tile = jnp.tile(wi, ([n, n, 1]))
        yi_w = jnp.concatenate((yi, tile), axis = -1)
        
        weight_reconstruction = jnp.mean((tile - reconstruction[:,:, yi.shape[-1]:]) ** 2, axis=-1)
        weight_reconstruction = jnp.mean((yi - reconstruction[:, :, :yi.shape[-1]]) ** 2, axis=-1)
        reconstruction_loss = jnp.mean((yi_w - reconstruction) ** 2, axis=-1)

        # Compute Jacobian Loss
        x_dot = jax.vmap(vector_field, in_axes=(None, 0, None))(None, yi.reshape(-1, 2), wi).reshape(n, n, 2)
        
        def f(x):
            return self.get_latent(x, wi)

        _, dFdt = jax.vmap(jax.jvp, in_axes=(None, 0, 0))(f, (yi.reshape(-1, 2),), (x_dot.reshape(-1, 2),))
        dFdt = dFdt.reshape(n, n, -1)

        latent_ = jax.vmap(lambda x: self.Koopman @ x)(latent.reshape(-1, latent.shape[-1])).reshape(n, n, -1)
        j_loss = jnp.mean((dFdt - latent_) ** 2, axis=-1)

        # Create plots
        losses = [reconstruction_loss, weight_reconstruction, weight_reconstruction, j_loss]
        loss_types = ['All Reconstruction', 'Weights', 'Predictions', 'Jacobian']
        
        nrows = -(-len(losses) // 2)  # Equivalent to ceil(len(losses) / 2)
        ncols = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))
        axs = axs.ravel()  # Flatten the axis array for easier indexing

        for i, (loss, l_type) in enumerate(zip(losses, loss_types)):
            z = axs[i].pcolormesh(y1, y2, loss, cmap="RdBu")
            axs[i].set_title('Loss Landscape: ' + l_type)
            axs[i].axis([lb, ub, lb, ub])
            fig.colorbar(z, ax=axs[i])

        # Hide any remaining unused subplots
        for i in range(len(losses), nrows * ncols):
            axs[i].axis('off')
            
        return fig
    
    
#--------------------------------------------------------------------------------------------------------------
# Saving And Loading
#--------------------------------------------------------------------------------------------------------------

def save(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)
        
def load(filename, type = 'PLearnKoopman'):
    if type == 'DynamicKoopman':
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = PLearnKoop(hyperparams, key=jax.random.PRNGKey(0))
            return eqx.tree_deserialise_leaves(f, model)
    if type == 'PLearnKoopman':
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = PLearnKoopman(**hyperparams, key=jax.random.PRNGKey(0))
            return eqx.tree_deserialise_leaves(f, model)
    if type == 'PLearnKoopmanMS':
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = PLearnKoopmanMS(**hyperparams, key=jax.random.PRNGKey(0))
            return eqx.tree_deserialise_leaves(f, model) 
    if type == 'PlearnKoopmanMS_WI':
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = PLearnKoopmanMS_WI(**hyperparams, key=jax.random.PRNGKey(0))
            return eqx.tree_deserialise_leaves(f, model) 
    if type == 'PlearnKoopmanTrajFree':
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = PLearnKoopmanTrajFree(**hyperparams, key=jax.random.PRNGKey(0))
            return eqx.tree_deserialise_leaves(f, model) 
    if type == 'PlearnKoopmanCL':
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = PLearnKoopmanCL(**hyperparams, key=jax.random.PRNGKey(0))
            return eqx.tree_deserialise_leaves(f, model) 
    if type == 'PlearnKoopmanLipCL':
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = PLearnKoopmanLipCL(**hyperparams, key=jax.random.PRNGKey(0))
            return eqx.tree_deserialise_leaves(f, model) 
    if type == 'PlearnKoopmanCE':
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = PLearnKoopmanCE(**hyperparams, key=jax.random.PRNGKey(0))
            return eqx.tree_deserialise_leaves(f, model) 
    if type == 'PlearnKoopmanLipCE':
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = PLearnKoopmanLipCE(**hyperparams, key=jax.random.PRNGKey(0))
            return eqx.tree_deserialise_leaves(f, model) 
    if type == 'PlearnKoopmanLip':
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = PLearnKoopmanLip(**hyperparams, key=jax.random.PRNGKey(0))
            return eqx.tree_deserialise_leaves(f, model) 
    if type == 'PLearnKoopmanConvex':
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = PLearnKoopmanConvex(**hyperparams, key=jax.random.PRNGKey(0))
            return eqx.tree_deserialise_leaves(f, model) 
    if type == 'PLearnKoopmanConvexLip':
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = PLearnKoopmanConvexLip(**hyperparams, key=jax.random.PRNGKey(0))
            return eqx.tree_deserialise_leaves(f, model)
    if type == 'PLearnKoopmanKDecoder':
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = PLearnKoopmanKDecoder(**hyperparams, key=jax.random.PRNGKey(0))
            return eqx.tree_deserialise_leaves(f, model)  
    else:
        raise ValueError('Type not found')
    
    
#--------------------------------------------------------------------------------------------------------------

def main():
    key = jrandom.PRNGKey(0)
    d_array = [64, 128, 256]
    
    # Initialize LipMLP
    lipmlp = LipMLP(d_array, key, use_bias=True, layer_norm=False, activation=jax.nn.tanh)
    
    # Generate some random input
    x = jrandom.normal(key, (64,))
    
    # Forward pass
    output = lipmlp(x)
    
    # Get Lipschitz loss
    lip_loss = lipmlp.get_lipschitz_loss()
    
    print("Output:", output)
    print("Lipschitz Loss:", lip_loss)

if __name__ == '__main__':
    
    main()
