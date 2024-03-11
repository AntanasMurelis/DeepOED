import jax
from jax import numpy as jnp
import equinox as eqx

from typing import Optional, Callable, Union
from .Modules import MLP, LipMLP

def module_factory(module_config, key):
    module_type = module_config['type']
    params = module_config.get('params', {})

    if module_type == "MLP":
        return MLP(key = key, **params)
    elif module_type == "LipMLP":
        return LipMLP(key = key, **params)
    # Add other conditions for different module types
    else:
        raise ValueError(f"Unknown module type: {module_type}")
    
def propagator_factory(propagator_config, key):
    
    propagator_type = propagator_config['type']

    if propagator_type == "Discrete":
        return DiscretePropagator(propagator_config, key)
    elif propagator_type == "Continuous":
        return ContinuousPropagator(propagator_config, key)
    else:
        raise ValueError(f"Unknown propagator type: {propagator_type}")

    

class BasePropagator(eqx.Module):
    
    encoderP: Optional[eqx.Module]
    Koop: Optional[jnp.ndarray]
    pre_encode: Optional[Callable]
    
    learn_koopman: bool = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(self, config, key):
        
        self.learn_koopman = config.get('learn_koopman', False)
        pre_encode1 = lambda x, w: jnp.concatenate((x, w), axis=-1) if config['include_IC'] else w
        self.pre_encode = lambda x, w: x if config['only_IC'] else pre_encode1(x, w)

        self.latent_dim = config['latent_dim']
        
        if self.learn_koopman:
            self.Koop = self.initialize_koopman_matrix(self.latent_dim, key)
            self.encoderP = None
        else:
            self.Koop = None
            d_key, _ = jax.random.split(key)
            self.encoderP = module_factory(config['encoder'], d_key)
            
    @staticmethod
    def initialize_koopman_matrix(latent_dim, key):
        dim = latent_dim
        lim = 1 / jnp.sqrt(dim) * 0.001
        k_key, _ = jax.random.split(key)
        return jax.random.uniform(k_key, (dim, dim), minval=-lim, maxval=lim)
    
    def LipLoss(self):
        if isinstance(self.encoderP, LipMLP):
            return self.encoderP.get_lipschitz_loss()
        return 0.0
    
    def get_KoopmanK(self, x, w):
        if self.learn_koopman:
            return self.Koop
        else:
            return self.encoderP(self.pre_encode(x, w)).reshape((self.latent_dim, self.latent_dim))
    
    def KoopmanK(self, w):
        return w.reshape((self.latent_dim, self.latent_dim))
    
class DiscretePropagator(BasePropagator):

    def __init__(self, config, key):
        super().__init__(config, key)

    def __call__(self, t, z0, x0, w):
        
        if self.Koop is not None:
            koopman_filters = self.compute_koopman_filters(self.Koop, len(t))
        else:
            koopman_matrix = self.encoderP(self.pre_encode(x0, w))
            koopman_filters = self.compute_koopman_filters(self.KoopmanK(koopman_matrix), len(t))
        # Apply each filter to the input x
        
        return jax.vmap(lambda k_filter: k_filter @ z0)(koopman_filters)
    
    def compute_koopman_filters(self, koopman_matrix, len_t):
        return jnp.array([jnp.linalg.matrix_power(koopman_matrix, t) for t in range(len_t)])
    
class ContinuousPropagator(BasePropagator):
    def __init__(self, config, key):
        super().__init__(config, key)

    def get_propagator(self, t, w):
        if self.Koop is not None:
            return jax.scipy.linalg.expm(self.Koop * t)
        else:
            return jax.scipy.linalg.expm(self.KoopmanK(w) * t)

    def __call__(self, t, z0, x0, w):
        if self.Koop is not None:
            koopman_matrix = self.Koop
        else:
            koopman_matrix = self.encoderP(self.pre_encode(x0, w))
        propagators = jax.vmap(self.get_propagator, in_axes=[0, None])(t, koopman_matrix)
        return jax.vmap(lambda propagator: propagator @ z0)(propagators)
    

class Encoder(eqx.Module):
    encoder: eqx.Module
    pre_encode: Callable
    
    def __init__(self, config, key):
        self.encoder = module_factory(config, key)
        self.pre_encode = lambda x, w: jnp.concatenate([x, w], axis=-1) if config['include_w'] else x
    
    def __call__(self, x, w):
        return self.encoder(self.pre_encode(x, w))
    
    def LipLoss(self):
        if isinstance(self.encoder, LipMLP):
            return self.encoder.get_lipschitz_loss()
        return 0.0  # No Lipschitz loss for non-LipMLP types
        
    
class Decoder(eqx.Module):
    decoder: eqx.Module
    
    def __init__(self, config, key):
        self.decoder = module_factory(config, key)
    
    def __call__(self, z):
        return jax.vmap(self.decoder)(z)
    
    def LipLoss(self):
        if isinstance(self.decoder, LipMLP):
            return self.decoder.get_lipschitz_loss()
        return 0.0  # No Lipschitz loss for non-LipMLP types


class KoopmanBase(eqx.Module):
    latent_dim: int = eqx.field(static=True) 

    def __init__(self, latent_dim, **kwargs):
        self.latent_dim = latent_dim

    def get_latent(self, x, w):
        raise NotImplementedError
    
    def loss_fn(self, ti, xi, wi):
        raise NotImplementedError
    
    def compute_w_loss(self, wi, w_pred):
        return jnp.mean(jnp.mean(jnp.abs(wi - w_pred), axis=-1))
    
    
class PLearnKoop(KoopmanBase):
    encoder: Encoder
    decoder: Decoder
    propagator: Optional[Union[DiscretePropagator, ContinuousPropagator]]
    
    latent_dim: int = eqx.field(static=True)
    learn_w: bool = eqx.field(static=True)

    def __init__(self, config, key):
        self.latent_dim = config['latent_dim'] 
        self.learn_w = config.get('learn_w', False)  # Set this based on the config

        # Initialize modules based on the configuration
        d_key, e_key, p_key = jax.random.split(key, 3)
        self.encoder = Encoder(config['encoder'], d_key)
        self.decoder = Decoder(config['decoder'], e_key)
        self.propagator = propagator_factory(config['propagator'], p_key)
        
    def __call__(self, t, x0, w):
        
        z0 = self.encoder(x0, w)
        latent_traj = self.propagator(t, z0, x0, w)
        x_pred = self.decoder(latent_traj)
        
        return x_pred[:, :len(x0)], x_pred[:, len(x0):], latent_traj

    def get_KoopmanK(self, x, w):
        return self.propagator.get_KoopmanK(x, w)

    @eqx.filter_jit
    def loss_fn(self, ti, xi, wi):
        # Reconstruction
        x_pred, w_pred, latent_traj_pred = self(ti, xi[0], wi)
        reconstruction = jnp.mean(jnp.mean(jnp.abs(xi - x_pred), axis=-1))

        # Latent Linearity Error:
        latent_traj = jax.vmap(self.encoder, in_axes=[0, None])(xi, wi)
        latent_error = jnp.mean(jnp.mean(jnp.abs(latent_traj_pred - latent_traj), axis=-1))

        total_loss = reconstruction + latent_error

        # Include w learning in the loss, if configured
        if self.learn_w:
            w_loss = self.compute_w_loss(wi, w_pred)
            total_loss += w_loss

        return total_loss, (reconstruction, latent_error,
                            w_loss if self.learn_w else 0.0)
    
    @eqx.filter_value_and_grad(has_aux=True)
    def batch_loss(self, ti, yi, wi):
        
        total_loss, (reconstruction, latent_error, w_loss) = jax.vmap(self.loss_fn, in_axes=[0, 0, 0])(ti, yi, wi)

        lipschitz_loss_encoderD = self.encoder.LipLoss()
        lipschitz_loss_decoder = self.decoder.LipLoss()
        lipschitz_loss_propagator = self.propagator.LipLoss() if hasattr(self.propagator, 'LipLoss') else 0.0
        
        loss = jnp.mean(total_loss) + lipschitz_loss_encoderD + lipschitz_loss_decoder + lipschitz_loss_propagator
        
        return loss, {"Reconstruction": jnp.mean(reconstruction), 
                    "Latent Error": jnp.mean(latent_error), 
                    "W Loss": jnp.mean(w_loss),
                    "Lipschitz Encoder Loss": lipschitz_loss_encoderD, 
                    "Lipschitz Decoder Loss": lipschitz_loss_decoder, 
                    "Lipschitz Propagator Loss": lipschitz_loss_propagator}

    @eqx.filter_jit
    def make_step(self, ti, yi, wi, optim, opt_state):
        losses, grads = self.batch_loss(ti, yi, wi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(self, updates)
        return losses, model, opt_state
    
    

#--Tests----------------------------------------------------------------------------------------------------------

mock_config = {
    'latent_dim': 128,
    'encoder': {
        'encoder': {
            'type': 'MLP',
            'params': {
                'd_array': jnp.array([128 + 128, 64, 128]),  # Assuming input dim = latent_dim + context dim
                'activation': 'relu',
                'layer_norm': False  # or True, if you want layer normalization
            },
            'include_w': True
        }
    },
    'decoder': {
        'decoder': {
            'type': 'MLP',
            'params': {
                'd_array': jnp.array([128, 64, 128]),  # Output dimension matches input dimension of the encoder
                'activation': 'sigmoid',
                'layer_norm': False  # or True, as per your requirements
            }
        }
    },
    'propagator': {
        'type': 'Discrete',
        'num_time_points': 10,
        'learn_koopman': True,
        'latent_dim': 128  # Matching the overall latent_dim of the model
    }
}

import unittest

class TestPLearnKoop(unittest.TestCase):

    def setUp(self):
        # Set up mock configuration
        self.mock_config = {
            'latent_dim': 128,
            'encoder': {
                'encoder': {
                    'type': 'MLP',
                    'params': {'layers': [64 + 128, 68, 128], 'activation': 'relu'},
                    'include_w': True
                }
            },
            'decoder': {
                'decoder': {
                    'type': 'MLP',
                    'params': {'layers': [128, 86, 64], 'activation': 'sigmoid'}
                }
            },
            'propagator': {
                'type': 'Discrete',
                'num_time_points': 10,
                'learn_koopman': True,
                'latent_dim': 128
            }
        }
        self.key = jax.random.PRNGKey(0)

    def test_module_factory(self):
        # Test module_factory function
        encoder = module_factory(self.mock_config['encoder']['encoder'], self.key)
        self.assertIsNotNone(encoder, "Encoder should not be None")

        decoder = module_factory(self.mock_config['decoder']['decoder'], self.key)
        self.assertIsNotNone(decoder, "Decoder should not be None")

    def test_propagator_factory(self):
        # Test propagator_factory function
        propagator = propagator_factory(self.mock_config['propagator'], self.key)
        self.assertIsNotNone(propagator, "Propagator should not be None")

    def test_PLearnKoop_initialization(self):
        # Test initialization of PLearnKoop
        model = PLearnKoop(self.mock_config, self.key)
        self.assertIsNotNone(model.encoder, "Model encoder should not be None")
        self.assertIsNotNone(model.decoder, "Model decoder should not be None")
        self.assertIsNotNone(model.propagator, "Model propagator should not be None")
        
    def test_PLearnKoop_forward_pass(self):
        # Test forward pass of PLearnKoop
        model = PLearnKoop(self.mock_config, self.key)
        t = jnp.array([0.0, 0.5, 1.0])  # Example time points
        x0 = jax.random.normal(self.key, (3, 64))  # Example initial state
        w = jax.random.normal(self.key, (3, 128))  # Example w values

        loss = model.batch_loss(t, x0, w)
        # self.assertEqual(x_pred.shape, (3, 64), "Shape of x_pred should match input x0")
        # self.assertEqual(latent_traj.shape, (3, 128), "Shape of latent_traj should match latent_dim")

class TestContinuousPropagator(unittest.TestCase):

    def setUp(self):
        """ Set up for the tests with a common configuration and key. """
        self.propagator_config = {
            "type": "Continuous",
            "latent_dim": 5,
            "learn_koopman": False,
            "encoderP": {
                "type": "MLP",
                "params": {"d_array": [5, 25], "activation": "relu"},
                "include_w": False
            }
        }
        self.key = jax.random.PRNGKey(0)
        self.x0 = jnp.array([1.0, 0.0, -1.0, 2.0, -2.0])
        self.times = jnp.linspace(0, 1, 5)

    def test_shape_and_finiteness(self):
        """ Test that the output has the correct shape and contains finite values. """
        propagator = ContinuousPropagator(self.propagator_config, self.key)
        propagated_states = propagator(self.times, self.x0, None)

        self.assertEqual(propagated_states.shape, (len(self.times), len(self.x0)),
                         "Output shape mismatch")
        self.assertTrue(jnp.all(jnp.isfinite(propagated_states)), 
                        "Non-finite values in output")


if __name__ == '__main__':
    unittest.main()
