import jax
import jax.numpy as jnp
from typing import Optional, Any, Tuple, Callable, Union, Sequence
from jax import ops
import numpy as np
import unittest
import numpy as np
import equinox as eqx
from icecream import ic
import cvxpy as cp

import jaxopt

def cartesian_product(arrays):
    """
    Generate a cartesian product of input arrays using JAX's meshgrid.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    """
    arrays = [jnp.array(x) for x in arrays]
    grids = jnp.meshgrid(*arrays, indexing='ij')
    cartesian_prod = jnp.stack(grids, axis=-1).reshape(-1, len(arrays))
    return cartesian_prod


class Embedding:#(eqx.Module):
    
    gamma: float
    n: float  # renamed from nu to maintain consistency
    m: int
    d: int
    nu: float
    diameter: float
    groups: Optional[Any] = eqx.field(static=True)
    kernel: str = eqx.field(static=True)
    approx: str = eqx.field(static=True)
    gradient_avail: int

    def __init__(self, gamma: float = 0.1, nu: float = 0.5, m: int = 100, 
                 d: int = 1, diameter: float = 1.0, groups: Optional[Any] = None,
                 kernel: str = "squared_exponential", approx: str = "rff"):
        """
        Initialize the Embedding class.

        Args:
            gamma: Bandwidth of the squared exponential kernel.
            nu: The parameter of the Matern family.
            m: Number of features.
            d: Dimension of the input space.
            diameter: Diameter of the domain.
            groups: Group identifiers.
            kernel: Type of kernel to use.
            approx: Approximation method to use.
        """
        self.gamma = gamma
        self.n = nu
        self.m = m
        self.d = d
        self.nu = nu
        self.diameter = diameter
        self.groups = groups
        self.kernel = kernel
        self.approx = approx
        self.gradient_avail = 0
        
        if self.m % 2 == 1:
            raise AssertionError("Number of random features has to be even.")

    def sample(self) -> None:
        """
        Placeholder for sampling method. Derived classes should implement this method.

        Args:
            None

        Raises:
            AttributeError: If called directly on base class.
        """
        raise AttributeError("Only derived classes can call this method.")

    def embed(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Placeholder for embed method. Derived classes should implement this method.

        Args:
            x: numpy array containing the points to be embedded in the format (n, d)

        Returns:
            y: numpy array containing the embedded points (n, m), where m is the embedding dimension.

        Raises:
            AttributeError: If called directly on base class.
        """
        raise AttributeError("Only derived classes can call this method.")

class QuadratureEmbedding(Embedding):
    scale: float
    W: jnp.ndarray  # Fourier feature weights
    weights: jnp.ndarray  # Fourier feature weights
    q: int  # Number of quadrature points

    def __init__(self, scale: float = 1.0, **kwargs: Any):
        """
        Initialize the QuadratureEmbedding class.

        Args:
            scale: Scaling factor for the quadrature.
            kwargs: Additional keyword arguments to be passed to the parent Embedding class.

        Returns:
            None
        """
        super().__init__(**kwargs)
        self.scale = float(scale)
        self.compute()

 
    @eqx.filter_jit()
    def embed(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the Fourier feature embedding of the input x.

        Args:
            x: Input array of shape (num_samples, input_dim).

        Returns:
            z: Fourier feature embedding of shape (num_samples, num_features).
        """
        d = self.d
        x = x.reshape(-1, d)
        q = jnp.dot(self.W[:, :d], x.T)
        cos_part = jnp.sqrt(self.weights.reshape(-1, 1)) * jnp.cos(q)
        sin_part = jnp.sqrt(self.weights.reshape(-1, 1)) * jnp.sin(q)
        z = jnp.vstack([cos_part, sin_part])
        z = jnp.squeeze(z)
        return z
    
    def reorder_complexity(self, omegas: jnp.ndarray, weights: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Reorder quadrature points and weights by frequency magnitude.

        Args:
            omegas: Frequencies.
            weights: Corresponding weights.

        Returns:
            new_omegas: Reordered frequencies.
            new_weights: Reordered weights.
        """
        abs_omegas = jnp.abs(omegas)
        order = jnp.argsort(abs_omegas)
        new_omegas = omegas[order]
        new_weights = weights[order]
        return new_omegas, new_weights

    def compute(self, complexity_reorder: bool = True) -> None:
         
        """
        Computes the tensor grid for Fourier features.
        """
        
        # Calculate the number of quadrature points and update self.m
        self.q = int(jnp.power(self.m // 2, 1. / self.d))
        # self.q = jnp.power(self.m // 2, 1. / self.d).astype(int)

        self.m = self.q ** self.d

        # Get the nodes and weights for quadrature
        omegas, weights = self.nodesAndWeights(self.q)

        # Optionally reorder for complexity
        if complexity_reorder:
            omegas, weights = self.reorder_complexity(omegas, weights)

        # Calculate the weights for Fourier features
        weights_ = cartesian_product([weights for _ in range(self.d)])
        self.weights = jnp.prod(weights_, axis=1)

        # Calculate the W matrix for Fourier features 
        self.W = cartesian_product([omegas for _ in range(self.d)])

        # Update the number of Fourier features
        self.m *= 2
        

    def transform(self) -> Any:
        """
        Compute the spectral density of a kernel using JAX's NumPy.
        """
        if self.kernel == "squared_exponential":
            p = lambda omega: jnp.exp(-jnp.sum(omega ** 2, axis=1).reshape(-1, 1) / 2 * (self.gamma ** 2)) * \
                            jnp.power((self.gamma / jnp.sqrt(2 * jnp.pi)), 1.) * jnp.power(jnp.pi / 2, 1.)
        elif self.kernel == "laplace":
            p = lambda omega: jnp.prod(1. / ((self.gamma ** 2) * (omega ** 2) + 1.), axis=1).reshape(-1, 1) * \
                            jnp.power(self.gamma / 2., 1.)
        elif self.kernel == "modified_matern":
            if self.nu == 2:
                p = lambda omega: jnp.prod(1. / ((self.gamma ** 2) * (omega ** 2) + 1.) ** self.nu, axis=1).reshape(-1, 1) * \
                                jnp.power(self.gamma * 1, 1.)
            # Add more conditions for other values of self.nu if needed
        return p

    def nodesAndWeights(self, q: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute nodes and weights of the quadrature scheme in 1D using JAX's NumPy.
        """        
        omegas, weights = np.polynomial.legendre.leggauss(2 * q)  

        omegas = omegas[q:]
        weights = 2 * weights[q:]

        omegas = ((omegas + 1.) / 2.) * jnp.pi
        sine_scale = (1. / (jnp.sin(omegas) ** 2))
        omegas = self.scale / jnp.tan(omegas)
        prob = self.transform()
        weights = self.scale * sine_scale * weights * prob(omegas.reshape(-1, 1)).flatten()
        return omegas, weights
    



class HermiteEmbedding(QuadratureEmbedding):
    def __init__(self, **kwargs: Any):
        """
        Initialize the HermiteEmbedding class.

        Args:
            kwargs: Additional keyword arguments to be passed to the parent QuadratureEmbedding class.

        Returns:
            None
        """
        super().__init__(**kwargs)
        if self.kernel != "squared_exponential":
            raise AssertionError("Hermite Embedding is allowed only with Squared Exponential Kernel")

    def nodesAndWeights(self, q: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute nodes and weights of the Hermite quadrature scheme in 1D.

        Args:
            q: Degree of quadrature.

        Returns:
            tuple(nodes, weights): Nodes and weights for Hermite quadrature.
        """
        nodes, weights = np.polynomial.hermite.hermgauss(2 * q)

        nodes = nodes[q:]
        weights = 2 * weights[q:]
        nodes = np.sqrt(2) * nodes / self.gamma
        weights = weights / np.sqrt(np.pi)

        return jnp.array(nodes), jnp.array(weights)

class HermiteLayer(HermiteEmbedding):
    
    B: jnp.ndarray
    o: int

    def __init__(self, scale: float, d: int, m: int, o: int, **kwargs) -> None:
        """
        scale: The scale parameter for the HermiteEmbedding.
        d: The dimensionality of the input space.
        m: The number of Hermite functions. (should be even)
        o: The number of output dimensions
        key: A JAX random key for initialization.
        kwargs: Additional keyword arguments to pass to HermiteEmbedding.

        return: None
        """
        super().__init__(gamma=scale, d=d, m=m, **kwargs)
        self.o = o
        self.B = None
        # self.B = self.init_B(m, o, key)


    def mu(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the HermiteLayer.

        Parameters:
        - t: Input tensor.

        Returns:
        - Output tensor after applying Hermite embedding and matrix multiplication.
        """
        return self.embed(t) @ self.B

    def func_embed(self, t):
        return jnp.kron(jnp.identity(self.o), self.embed(t))
    
    def mu(self, t: jnp.ndarray) -> jnp.ndarray:    
        return self.func_embed(t) @ self.B
    
    def get_B_vec(self):
        return self.B.flatten()  
    
    
    def constrained_optimization(self, ti, xi, t_=None, A_theta=None):
        
        # Compute the Psi matrices
        Psi_t = jax.vmap(self.embed)(ti) 
        if t_ is not None:
            Psi_t_ = jax.vmap(self.embed)(t_)
        
        # Create the Beta variable, assuming Beta is a matrix
        Beta = cp.Variable((Psi_t.shape[-1], self.o))
        diff = xi - Psi_t @ Beta
        
        if t_ is not None:
            jac = jax.vmap(jax.jacfwd(self.embed))(t_)
            
        objective = cp.Minimize(cp.norm(diff, 'fro')**2) 
        
        if A_theta is not None and t_ is not None:            
            constraints = [jac @ Beta == Psi_t_ @ (Beta @ A_theta.T)] 
        else:
            constraints = []
         
        # Define and solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)        
        self.B = Beta.value
        
        return Beta.value  # Return the optimized value of Beta

    def constrained_optimization(self, ti, xi, t_=None, A_theta=None):
        # Compute the Psi matrices
        Psi_t = jax.vmap(self.embed)(ti)

        if t_ is not None:
            Psi_t_ = jax.vmap(self.embed)(t_)
            jac = jax.vmap(jax.jacfwd(self.embed))(t_)

        # The dimension of Beta
        # beta_dim = Psi_t.shape[-1] * self.o

        # Create the objective matrix and vector (for quadratic programming)
        # The objective is ||xi - Psi_t @ Beta||^2_Fro
        Q = jax.numpy.kron(jax.numpy.eye(self.o), Psi_t.T @ Psi_t)
        c = -jax.numpy.ravel(xi.T @ Psi_t)

        if t_ is not None:
            jac = jax.vmap(jax.jacfwd(self.embed))(t_)

        if A_theta is not None and t_ is not None:
            # Constructing the linear constraint for QuadraticProgramming
            # The constraint is jac @ Beta == Psi_t_ @ (Beta @ A_theta^T)
            A_eq = jax.numpy.kron(jax.numpy.eye(self.o), jac) - jax.numpy.kron(A_theta, Psi_t_)
            b_eq = jax.numpy.zeros((A_eq.shape[0],))
            constraints = (A_eq, b_eq)
        else:
            constraints = None

        # Initialize the quadratic programming solver
        # qp_solver = jaxopt.EqualityConstrainedQP(tol=1e-10, maxiter=1000)
        qp_solver = jaxopt.OSQP()

        # Solve the problem
        sol = qp_solver.run(params_obj=(Q, c), params_eq=constraints if constraints else None).params.primal
        self.B = sol
        return self.B  # Return the optimized value of Beta

class ConstrainedHermiteLayer(HermiteEmbedding):
    
    Psi_t_: jnp.ndarray
    jac: jnp.ndarray
    
    def __init__(self, scale: float, d: int, m: int, o: int, t_: jnp.ndarray, **kwargs) -> None:
        
        super().__init__(gamma=scale, d=d, m=m, **kwargs)
        
        self.Psi_t_ = jax.vmap(self.embed)(t_)
        self.jac = jax.vmap(jax.jacfwd(self.embed))(t_)
        
        self.o = o
        self.B = None

    def mu(self, t: jnp.ndarray) -> jnp.ndarray:
        return self.func_embed(t) @ self.B
    
    def get_B_vec(self):
        return self.B.flatten()  
    
    def func_embed(self, t):
        return jnp.kron(jnp.identity(self.o), self.embed(t))
    
    def constrained_optimization(self, ti, xi, A_theta=None):

        Psi_t = jax.vmap(self.embed)(ti)

        # Create the objective matrix and vector (for quadratic programming)
        # The objective is ||xi - Psi_t @ Beta||^2_Fro
        Q = jax.numpy.kron(jax.numpy.eye(self.o), Psi_t.T @ Psi_t)
        c = -jax.numpy.ravel(xi.T @ Psi_t)

        if A_theta is not None:
            # Constructing the linear constraint for QuadraticProgramming
            # The constraint is jac @ Beta == Psi_t_ @ (Beta @ A_theta^T)
            A_eq = jax.numpy.kron(jax.numpy.eye(self.o), self.jac) - jax.numpy.kron(A_theta, self.Psi_t_)
            b_eq = jax.numpy.zeros((A_eq.shape[0],))
            constraints = (A_eq, b_eq)
        else:
            constraints = None

        # Initialize the quadratic programming solver
        # qp_solver = jaxopt.OSQP()
        qp_solver = jaxopt.EqualityConstrainedQP(tol=1e-10, maxiter=1000)

        # Solve the problem
        # qp_solver = jaxopt.CvxpyQP()
        sol = qp_solver.run(params_obj=(Q, c), params_eq=constraints if constraints else None).params.primal
        self.B = sol
        
        return self.B  # Return the optimized value of Beta    
        
    def A_theta_opt(self, ti, xi, A_theta_0):
        
        lambda A_theta: self.constrained_optimization(ti, xi, A_theta)
        
        def fun(A_theta):
            B = self.constrained_optimization(ti, xi, A_theta)
            xi_pred = jax.vmap(self.mu)(ti)
            return jnp.sum(jnp.abs(xi - xi_pred))
        
        

    # def constrained_optimization_(self, ti, xi, t_=None, A_theta=None):
        
        
    #     # Compute the Psi matrices
    #     I = jnp.identity(self.o)
        
    #     def fun(I, ti):
    #         return jnp.kron(I, self.embed(ti))
        
    #     Psi_t = jax.vmap(fun, in_axes=[None, 0])(I, ti) 
        
    #     if t_ is not None:
            
    #         def Psi_Theta(A_theta, t_):
    #             return jnp.kron(A_theta, self.embed(t_))
            
    #         Psi_t_ = jax.vmap(Psi_Theta, in_axes=[None, 0])(A_theta, t_)
        
    #     # Create the Beta variable, assuming Beta is a vector
    #     Beta = cp.Variable((Psi_t.shape[1] * self.o))
        
    #     def eval(Psi_t, Beta):
    #         return Psi_t @ Beta
        
    #     xi_pred = jax.vmap(eval, in_axes=[0, None])(Psi_t, Beta)
        
    #     diff = xi - xi_pred
        
    #     if t_ is not None:
    #         def get_jac(I, t_):
    #             return jnp.kron(I, jax.jacfwd(self.embed)(t_))
    #         jac = jax.vmap(get_jac, in_axes=[None, 0])(I,t_)
            
    #     objective = cp.Minimize(cp.norm(diff, 'fro')**2) 
        
    #     if A_theta is not None and t_ is not None:            
    #         constraints = [(jac - Psi_t_) @ Beta == 0] 
    #     else:
    #         constraints = []
         
    #     # Define and solve the problem
    #     prob = cp.Problem(objective, constraints)
    #     prob.solve(solver=cp.SCS)        
    #     self.B = Beta.value
        
    #     return Beta.value  # Return the optimized value of Beta

#--Tests----------------------------------------------------------------
class TestHermiteEmbedding(unittest.TestCase):
    
    def setUp(self):
        self.embedding = HermiteEmbedding(gamma=0.1, m=10, d=1, kernel="squared_exponential")

    def test_nodesAndWeights(self):
        nodes, weights = self.embedding.nodesAndWeights(10)
        self.assertEqual(nodes.shape, (10,))
        self.assertEqual(weights.shape, (10,))
        self.assertTrue(jnp.all(weights >= 0))

    def test_embed(self):
        x = jnp.array([[0.1], [0.2], [0.3]])
        embedded_x = self.embedding(x)
        self.assertEqual(embedded_x.shape, (3, self.embedding.m))

    def test_reorder_complexity(self):
        omegas = jnp.array([0.1, 0.5, 0.3])
        weights = jnp.array([0.2, 0.4, 0.1])
        new_omegas, new_weights = self.embedding.reorder_complexity(omegas, weights)
        self.assertTrue(jnp.all(jnp.diff(jnp.abs(new_omegas)) >= 0))

    def test_transform(self):
        p = self.embedding.transform()
        self.assertTrue(callable(p))

class TestHermiteEmbedding(unittest.TestCase):
    
    def test_nodesAndWeights(self):
        for m, d in [(100, 1), (200, 2), (50, 3)]:
            with self.subTest(m=m, d=d):
                embedding = HermiteEmbedding(gamma=0.1, m=m, d=d, kernel="squared_exponential")
                nodes, weights = embedding.nodesAndWeights(10)
                self.assertEqual(nodes.shape, (10,))
                self.assertEqual(weights.shape, (10,))
                self.assertTrue(jnp.all(weights >= 0))

    def test_embed(self):
        for m, d, x_shape in [(100, 1, (3, 1)), (200, 2, (3, 2)), (50, 3, (3, 3))]:
            with self.subTest(m=m, d=d, x_shape=x_shape):
                embedding = HermiteEmbedding(gamma=0.1, m=m, d=d, kernel="squared_exponential")
                x = np.random.rand(*x_shape)
                embedded_x = embedding(x)
                self.assertEqual(embedded_x.shape, (x_shape[0], embedding.m))

    def test_reorder_complexity(self):
        embedding = HermiteEmbedding(gamma=0.1, m=100, d=1, kernel="squared_exponential")
        omegas = jnp.array([0.1, 0.5, 0.3])
        weights = jnp.array([0.2, 0.4, 0.1])
        new_omegas, new_weights = embedding.reorder_complexity(omegas, weights)
        self.assertTrue(jnp.all(jnp.diff(jnp.abs(new_omegas)) >= 0))

    def test_transform(self):
        embedding = HermiteEmbedding(gamma=0.1, m=100, d=1, kernel="squared_exponential")
        p = embedding.transform()
        self.assertTrue(callable(p))

class TestCartesianJax(unittest.TestCase):
    
    def test_cartesian_jax(self):
        arrays = [[1, 2], [3, 4], [5, 6]]
        result = cartesian_product(arrays)
        
        # Expected result calculated manually
        expected = jnp.array([
            [1, 3, 5],
            [1, 3, 6],
            [1, 4, 5],
            [1, 4, 6],
            [2, 3, 5],
            [2, 3, 6],
            [2, 4, 5],
            [2, 4, 6]
        ])
        print(result)
        # Check if the result matches the expected output
        self.assertTrue(jnp.array_equal(result, expected))
#------------------------------------------------------------------------

if __name__ == '__main__':
    # unittest.main()
    
    from Archs import *
        #---Step 1: Generate synthetic training data---------------------
    ti = jnp.linspace(0, 10, 10)
    t_ = jnp.linspace(0, 10, 100)
    args = jnp.array([1.3, 0.9, 1.6, 1.2]) 
    multi_args = jax.random.uniform(key = jax.random.PRNGKey(10), shape = (1, 4), minval = jnp.array([0.5, 0.5, 0.5, 0.5]),
                                    maxval = jnp.array([5, 5, 5, 5]))
    
    # multi_args = jnp.array([2.5, 1.2, 4.2, 2]).reshape(1, -1)
    
    # multi_args = jnp.array([(60.0, 30.4), (30, 70), (25, 45), (100, 10)])

    true_x0 = jnp.array([5.0, 5.0])  # True initial condition

    # Get latent Trajectory:
    #----------------------------------------------------------------


    #---Step 2: Load the model---------------------------------------  
    # jax.config.update("jax_enable_x64", False)
    # # models = list(map(lambda x: load(x, type = 'PLearnKoop'), os.listdir("Users/antanas/GitRepo/NODE/Models/PLearnKoopman_LV_2D_Lip_10_16.eqx")))
    # model = load("/Users/antanas/GitRepo/NODE/Models/LV_10D_1e-5/PLearnKoopmanLV_10D.eqx", type = 'DynamicKoopman')
    # jax.config.update("jax_enable_x64", True)
    
    # Latent_Traj_fit = model.get_latent_traj(ti, true_x0, args)
    # Latent_Traj = model.get_latent_traj(t_, true_x0, args)

    # A_theta = model.get_KoopmanK(None, args)    
    
    linear_field = lambda t, x, args: args @ x.T
    
    args = jnp.array([[0, -1], [1, 0]]).reshape(2, 2)
    Latent_Traj_fit = solve(ti, true_x0, args, linear_field)
    Latent_Traj  = solve(t_, true_x0, args, linear_field)
    
    parameters = {'scale': 1.0, 
                'd': 1, 
                'm': 50, 
                'o': 2, 
    }
    # emb = HermiteLayer(**parameters) # Squared exponential with lenghtscale 0.5 with 100 basis functions 
    emb = ConstrainedHermiteLayer(**parameters, t_ = t_) # Squared exponential with lenghtscale 0.5 with 100 basis functions
    emb.constrained_optimization(ti, Latent_Traj_fit, A_theta = args)

    mu = jax.vmap(emb.mu)(t_)
    
    plt.plot(t_, mu, label = 'Hermite Embedding')
    plt.plot(t_, Latent_Traj, label = 'True Latent Trajectory')
    plt.legend()
    plt.show()


    # ic(jax.vmap(emb)(x)) # Embedding of x


