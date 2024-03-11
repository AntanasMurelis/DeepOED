import jax
from jax import numpy as jnp
import equinox as eqx
import equinox.nn as nn
from typing import Optional, Callable


def get_activation_function(name: str) -> Callable:
    
    activation_functions = {
        'relu': jax.nn.relu,
        'sigmoid': jax.nn.sigmoid,
        'tanh': jax.nn.tanh,
        'softplus': jax.nn.softplus,
        'elu': jax.nn.elu,
        'selu': jax.nn.selu,
        'gelu': jax.nn.gelu,
        'leaky_relu': jax.nn.leaky_relu,
        'log_sigmoid': jax.nn.log_sigmoid,
        'log_softmax': jax.nn.log_softmax,
        'None': lambda x: x,
        # Add more activations as needed
    }

    if name not in activation_functions:
        raise ValueError(f"Unknown activation function: {name}")

    return activation_functions[name]

class MLP(eqx.Module):
    d_array: list[int]
    layers: list[nn.Linear]
    n_layers:list[nn.LayerNorm]
    activation_fn: Callable

    def __init__(self, layers, key, activation='relu', layer_norm = False, **kwargs):
        
        keys = jax.random.split(key, len(layers))
        
        self.activation_fn = get_activation_function(activation)
        self.d_array = layers
        input_d_array = layers[:-1]
        output_d_array = layers[1:]
        self.layers = [nn.Linear(in_d, out_d, key = keys[i]) for i,
                       (in_d, out_d) in enumerate(zip(input_d_array, output_d_array))]
        
        if layer_norm: self.n_layers = [nn.LayerNorm(d) for d in output_d_array[:-1]]
        else: self.n_layers = None
        
    
    def __call__(self, x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = self.activation_fn(x)
            if self.n_layers is not None: x = self.n_layers[i](x)
            
        x = self.layers[-1](x)
        
        return jnp.array(x)
    
class LipMLP(eqx.Module):
    params_net: list[list[jnp.ndarray]]
    use_bias: bool = eqx.field(static=True)
    n_layers: list[nn.LayerNorm]
    activation_fn: Callable
    scale: float = eqx.field(static=True)
    
    def __init__(self, layers, key, scale = 1.0, use_bias=True, activation='relu', layer_norm=False, **kwargs):
    
        def init_W_b(size_out, size_in, key):
            wkey, bkey = jax.random.split(key, 2)
            lim = jnp.sqrt(2 / size_in)
            W = jax.random.uniform(wkey, (size_out, size_in), minval=-lim, maxval=lim)
            if use_bias:
                b = jax.random.uniform(bkey, (size_out,), minval=-lim, maxval=lim)
            else:
                b = None
            return jnp.array(W), b

        self.activation_fn = get_activation_function(activation)
        self.scale = scale
        
        keys = jax.random.split(key, len(layers))
        
        input_d_array = layers[:-1]
        output_d_array = layers[1:]
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
            x = self.activation_fn(jnp.dot(W, x) + (b if self.use_bias is not None else 0))
            
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
        return self.scale * loss_lip

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
    
def init_W_b(size_out, size_in, key, use_bias=True):
    wkey, bkey = jax.random.split(key, 2)
    lim = jnp.sqrt(2 / size_in)
    W = jax.random.uniform(wkey, (size_out, size_in), minval=-lim, maxval=lim)
    if use_bias:
        b = jax.random.uniform(bkey, (size_out,), minval=-lim, maxval=lim)
    else:
        b = jnp.zeros(size_out)
    return W, b

class LipNonNegativeLinear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    c: jnp.ndarray  # Lipschitz constant

    def __init__(self, in_d: int, out_d: int, key: jax.random.PRNGKey, use_bias: bool = True):
        wkey, bkey = jax.random.split(key, 2)
        
        self.weight = jax.random.uniform(wkey, (out_d, in_d)) * 1 / (in_d * out_d)
        self.c = jnp.max(jnp.sum(jnp.abs(self.weight), axis=1))  # Initialize Lipschitz constant

        if use_bias:
            self.bias = jax.random.normal(bkey, (out_d,)) * 0
        else:
            self.bias = jnp.zeros(out_d)

    def weight_normalization(self, W, softplus_c):
        """
        Lipschitz weight normalization based on the L-infinity norm
        """
        absrowsum = jnp.sum(jnp.abs(W), axis=1)
        scale = jnp.minimum(1.0, softplus_c / absrowsum)
        return W * scale[:, None]
    
    def get_lip(self):
        """
        This function computes the Lipschitz regularization
        """
        return jax.nn.softplus(self.c)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = jax.nn.softplus(self.weight) * 1e-3
        weight = self.weight_normalization(weight, jax.nn.softplus(self.c))  # Use the Lipschitz constant
        return jnp.dot(x, weight.T) + self.bias



class NonNegativeLinear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_d: int, out_d: int, key: jax.random.PRNGKey, use_bias: bool = True):
        wkey, bkey = jax.random.split(key, 2)
        lim = jnp.sqrt(2 / in_d)
        
        self.weight = jax.random.uniform(wkey, (out_d, in_d)) * 1 / (in_d * out_d)

        if use_bias:
            self.bias = jax.random.normal(bkey, (out_d,)) * 0
        else:
            self.bias = jnp.zeros(out_d)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = jax.nn.softplus(self.weight) * 1e-3
        return  jnp.dot(x, weight.T) + self.bias
    

class Linear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_d: int, out_d: int, key: jax.random.PRNGKey, use_bias: bool = True):
        wkey, bkey = jax.random.split(key, 2)
        lim = jnp.sqrt(2 / in_d)
        
        self.weight = jax.random.normal(wkey, (out_d, in_d)) * 1 / (in_d * out_d)
        if use_bias:
            self.bias = jax.random.normal(bkey, (out_d,)) * 0 # 1 / (in_d * out_d)  # Note: bias should have shape (out_d,)
        else:
            self.bias = jnp.zeros(out_d)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, self.weight.T) + self.bias

class LipLinear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    c: jnp.ndarray  # Lipschitz constant

    def __init__(self, in_d: int, out_d: int, key: jax.random.PRNGKey, use_bias: bool = True):
        wkey, bkey = jax.random.split(key, 2)
        lim = jnp.sqrt(2 / in_d)
        
        self.weight = jax.random.normal(wkey, (out_d, in_d)) * 1 / (in_d * out_d)
        self.c = jnp.max(jnp.sum(jnp.abs(self.weight), axis=1))  # Initialize Lipschitz constant

        if use_bias:
            self.bias = jax.random.normal(bkey, (out_d,)) * 0
        else:
            self.bias = jnp.zeros(out_d)

    def weight_normalization(self, W, softplus_c):
        """
        Lipschitz weight normalization based on the L-infinity norm
        """
        absrowsum = jnp.sum(jnp.abs(W), axis=1)
        scale = jnp.minimum(1.0, softplus_c / absrowsum)
        return W * scale[:, None]
    
    def get_lip(self):
        """
        This function computes the Lipschitz regularization
        """
        return jax.nn.softplus(self.c)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.weight
        weight = self.weight_normalization(weight, jax.nn.softplus(self.c))  # Use the Lipschitz constant
        return jnp.dot(x, weight.T) + self.bias


class ICNN(eqx.Module):
    W: list[NonNegativeLinear]
    A: list[Linear]

    def __init__(self, d_array: list[int], key: jax.random.PRNGKey):
        keys = jax.random.split(key, len(d_array) * 2)  # Twice the number of keys for A and A_ln
        self.W = [NonNegativeLinear(in_d, out_d, key=k) for in_d, out_d, k in zip(d_array[:-1], d_array[1:], keys[:len(d_array) - 1])]
        self.A = [Linear(d_array[0], out_d, key=k) for out_d, k in zip(d_array, keys[len(d_array)-1:len(d_array)*2-1])]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = jax.nn.relu(self.A[0](x))
        z = z * z
        for W, A in zip(self.W[:-1], self.A[1:-1]):
            z = jax.nn.relu(W(z) + A(x))
        y = self.W[-1](z) + self.A[-1](x)
        
        return y

class LipICNN(eqx.Module):
    W: list[NonNegativeLinear]  # Replace with LipNonNegativeLinear
    A: list[Linear]  # Replace with LipLinear

    def __init__(self, d_array: list[int], key: jax.random.PRNGKey):
        keys = jax.random.split(key, len(d_array) * 2)
        self.W = [LipNonNegativeLinear(in_d, out_d, key=k) for in_d, out_d, k in zip(d_array[:-1], d_array[1:], keys[:len(d_array) - 1])]
        self.A = [LipLinear(d_array[0], out_d, key=k) for out_d, k in zip(d_array, keys[len(d_array)-1:len(d_array)*2-1])]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = jax.nn.relu(self.A[0](x))
        z = z * z
        for W, A in zip(self.W[:-1], self.A[1:-1]):
            z = jax.nn.relu(W(z) + A(x))
        y = self.W[-1](z) + self.A[-1](x)
        
        return y

    def get_lipschitz_loss(self):
        """
        This function computes the overall Lipschitz regularization
        """
        loss_lip = 1.0
        for layer in self.W + self.A:
            loss_lip = loss_lip * layer.get_lip()
        return loss_lip


