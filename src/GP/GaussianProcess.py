import jax.numpy as jnp
from jax import random as jrandom
from jax import vmap
import jax
from HermiteEmbedding import HermiteEmbedding, HermiteLayer, ConstrainedHermiteLayer
import scipy


import matplotlib.pyplot as plt
from icecream import ic

class GaussianProcess:
    
    HL: HermiteLayer
    
    def __init__(self, hermite_layer_params, noise_variance=1e-5):
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.K_inv = None  # Inverse of the kernel matrix
        self.HL = HermiteLayer(**hermite_layer_params)

    def get_B(self):
        return self.HL.B   
    
    def embed(self, t):
        return self.HL.embed(t)
    
    def __call__(self, X_new):
        
        # Transform the new data
        Phi_new = vmap(self.embed)(X_new)
        Phi_train = vmap(self.embed)(self.X_train)
        ic(self.K_inv.shape, Phi_new.shape, Phi_train.shape)
        # Compute the kernel between the new and training data
        K_s = Phi_train @ Phi_new.T
        
        
        # Compute the self-kernel of the new data
        K_ss = Phi_new @ Phi_new.T
        ic(X_new.shape)
        mu = jax.vmap(self.HL.mu)(X_new)
        ic(mu.shape)
        # Compute the predictive covariance
        cov = Phi_new @ self.K_inv @ Phi_new.T# + self.noise_variance * jnp.eye(len(X_new))
        ic
        cov *= self.noise_variance**2
        # cov = K_ss - jnp.dot(K_s.T, jnp.dot(self.K_inv, K_s))
        
        # Extract the variances (the diagonal elements of the covariance matrix)
        var = jnp.diag(cov).reshape(-1, 1)
        
        return mu, var  # Return the mean and variance for each point


    def fit(self, X_train, y_train, t_=None, A_theta=None, noise_variance=None):
        
        if noise_variance is not None:
            self.noise_variance = noise_variance
        
        self.X_train = X_train
        self.y_train = y_train
        
        # Optimize the HermiteLayer parameters
        self.HL.constrained_optimization(X_train, y_train, t_, A_theta)

        # Transform the training data
        Phi_train = jax.vmap(self.embed)(X_train)
        
        K = Phi_train.T @ Phi_train + self.noise_variance**2 * jnp.eye(len(Phi_train.T))
        # K = Phi_train @ Phi_train.T + self.noise_variance**2 * jnp.eye(len(Phi_train))
        self.K_inv = jnp.linalg.inv(K)
        
class ConstrainedGaussianProcess:
    
    HL: HermiteLayer
    
    def __init__(self, hermite_layer_params, noise_variance=1e-5):
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.K_inv = None  # Inverse of the kernel matrix
        self.HL = HermiteLayer(**hermite_layer_params)
    
    @property
    def get_B(self):
        return self.HL.B   
    
    def embed(self, t):
        return self.HL.embed(t)
    
    def b_kernel(self, t1, t2):
        return self.embed(t1).T @ self.embed(t2) * self.get_B.T @ self.get_B
    
    def build_K(self, t1, t2):
        rows = jax.vmap(self.b_kernel, in_axes=(None, 0))
        K = jax.vmap(rows, in_axes=(0, None))(t1, t2)
        return K
    
    def __call__(self, X_new, var=False):
        
        # Transform the new data
        Phi_new = vmap(self.embed)(X_new)
        # Phi_train = vmap(self.embed)(self.X_train)
        # ic(self.K_inv.shape, Phi_new.shape, Phi_train.shape)
        # Compute the kernel between the new and training data
        
        
        # K_s = self.get_B.T @ Phi_train.T @ Phi_new @ self.get_B
        # K_s = self.build_K(X_new, self.X_train)
        
        # Compute the self-kernel of the new data
        # K_ss = self.get_B.T @ Phi_new.T @ Phi_new @ self.get_B
        # K_ss = self.build_K(X_new, X_new)
        
        mu = jax.vmap(self.HL.mu)(X_new).T
        # Compute the predictive covariance
        # cov = Phi_new @ self.K_inv @ Phi_new.T# + self.noise_variance * jnp.eye(len(X_new))
        # ic
        # cov *= self.noise_variance**2
        # def get_covariance(B, Phi_new):
        #     ic(B.shape, Phi_new.shape)
        #     Phi = Phi_new.reshape(-1, 1)
            
        #     return  Phi.T @ Phi * B.T @ B
        
        # cov = jax.vmap(get_covariance, in_axes=[None, 0])(self.get_B, Phi_new)
        
        # cov = K_ss - jnp.dot(K_s.T, jnp.dot(self.K_inv, K_s))
        # ic(cov)
        # Extract the variances (the diagonal elements of the covariance matrix)
        
        # def get_var(cov):
            
        #     return jnp.diag(cov).reshape(-1, 1)
        
        # var = jax.vmap(get_var)(cov)
        # var = jnp.diag(cov).reshape(-1, 1)
        
        if var:
            P_c = self.C.T @ jnp.linalg.pinv(self.C.T, rcond=1e-10)
            
            def get_var(P_c, Phi):
                Phi_ = jnp.kron(jnp.identity(self.HL.o), Phi)
                K =  self.K_inv #jnp.identity(self.K_inv.shape[0]) 
                cov = Phi_ @ P_c @ K @ P_c.T @ Phi_.T
                return jnp.sqrt(jnp.diag(cov).reshape(-1))
            
            var = jax.vmap(get_var, in_axes=[None, 0], out_axes=-1)(P_c, Phi_new)
            # var = jnp.diag(cov).reshape(-1, 1)
            return mu, var
       
        # cov = K_ss - jnp.dot(K_s.T, jnp.dot(self.K_inv, K_s))
        # var = jnp.diag(cov).reshape(-1, 1)
        
        return mu, None # Return the mean and variance for each point


    def fit(self, X_train, y_train, t_=None, A_theta=None, noise_variance=None):
        
        if noise_variance is not None:
            self.noise_variance = noise_variance
        
        self.X_train = X_train
        self.y_train = y_train
        
        # Optimize the HermiteLayer parameters
        self.HL.constrained_optimization(X_train, y_train, t_, A_theta)

        # Transform the training data
        Phi_train = jax.vmap(self.embed)(X_train)
        
        ic(self.get_B.shape, Phi_train.shape)
        
        K = self.build_K(X_train, X_train) + self.noise_variance**2 * jnp.eye(len(X_train))
        
        # K = self.get_B.T @ Phi_train.T @ Phi_train @ self.get_B + self.noise_variance**2 * jnp.eye(self.get_B.shape[1])
        # K = Phi_train @ Phi_train.T + self.noise_variance**2 * jnp.eye(len(Phi_train))
        
        self.K_inv = jnp.linalg.pinv(K)
        
    def fit_(self, X_train, y_train, t_=None, A_theta=None, noise_variance=None):
        
        if noise_variance is not None:
            self.noise_variance = noise_variance
        
        self.X_train = X_train
        self.y_train = y_train
        
        # Optimize the HermiteLayer parameters
        self.HL.constrained_optimization(X_train, y_train, t_, A_theta)

        self.C = self.get_C(A_theta, t_)
        
        # Transform the training data
        Phi_train = jax.vmap(self.HL.func_embed)(X_train)
        
        Phi_train = Phi_train.reshape(-1, Phi_train.shape[-1])
        
        K = Phi_train.T @ Phi_train + self.noise_variance * jnp.eye(len(Phi_train.T))
        # K = K_ + self.noise_variance * jnp.eye(len(Phi_train.T))
        
        # self.K_inv = self.noise_variance * K_i @ K_ @ K_i 
        
        self.K_inv = self.noise_variance * jnp.linalg.inv(K)
    
    @staticmethod
    def get_Lt(dPhi, Phi, A_theta):
        
        I = jnp.identity(A_theta.shape[0])
        I_dPhi = jnp.kron(I, dPhi)
        A_Phi = jnp.kron(A_theta, Phi)
        
        return I_dPhi - A_Phi
    
    def get_C(self, A_theta, t_):
        
        Phi_train = jax.vmap(self.embed)(t_)
        dPhi_train = jax.vmap(jax.jacfwd(self.embed))(t_)
        
        L = jax.vmap(self.get_Lt, in_axes = [0, 0, None])(dPhi_train, Phi_train, A_theta)
        L = L.reshape(-1, L.shape[-1])
        C_T = scipy.linalg.null_space(L, rcond=1e-10)
        
        return C_T.T

class ConstrainedGaussianProcess2:
    
    HL: HermiteLayer
    
    def __init__(self, hermite_layer_params, noise_variance=1e-5):
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.K_inv = None  # Inverse of the kernel matrix
        self.HL = ConstrainedHermiteLayer(**hermite_layer_params)
    
    @property
    def get_B(self):
        return self.HL.B   
    
    def embed(self, t):
        return self.HL.embed(t)
    
    def b_kernel(self, t1, t2):
        return self.embed(t1).T @ self.embed(t2) * self.get_B.T @ self.get_B
    
    def build_K(self, t1, t2):
        rows = jax.vmap(self.b_kernel, in_axes=(None, 0))
        K = jax.vmap(rows, in_axes=(0, None))(t1, t2)
        return K
    
    def __call__(self, X_new, var=False):
        
        # Transform the new data
        Phi_new = vmap(self.embed)(X_new)
        mu = jax.vmap(self.HL.mu)(X_new).T
        
        if var:
            
            P_c = self.C.T @ jnp.linalg.pinv(self.C.T, rcond=1e-10)
            
            def get_var(P_c, Phi):
                Phi_ = jnp.kron(jnp.identity(self.HL.o), Phi)
                K =  self.K_inv 
                cov = Phi_ @ P_c @ K @ P_c.T @ Phi_.T
                return jnp.sqrt(jnp.diag(cov).reshape(-1))
            
            var = jax.vmap(get_var, in_axes=[None, 0], out_axes=-1)(P_c, Phi_new)
            return mu, var
       
        
        return mu, None # Return the mean and variance for each point


    def fit(self, X_train, y_train, t_=None, A_theta=None, noise_variance=None):
        
        if noise_variance is not None:
            self.noise_variance = noise_variance
        
        self.X_train = X_train
        self.y_train = y_train
        
        # Optimize the HermiteLayer parameters
        self.HL.constrained_optimization(X_train, y_train, t_, A_theta)

        # Transform the training data
        Phi_train = jax.vmap(self.embed)(X_train)
                
        K = self.build_K(X_train, X_train) + self.noise_variance**2 * jnp.eye(len(X_train))
        
        self.K_inv = jnp.linalg.pinv(K)
        
    def fit_(self, X_train, y_train, A_theta=None, noise_variance=None):
        
        if noise_variance is not None:
            self.noise_variance = noise_variance
        
        self.X_train = X_train
        self.y_train = y_train
        
        # Optimize the HermiteLayer parameters
        self.HL.constrained_optimization(X_train, y_train, A_theta)

        self.C = self.get_Q(A_theta)
        
        # Transform the training data
        Phi_train = jax.vmap(self.HL.func_embed)(X_train)
        
        Phi_train = Phi_train.reshape(-1, Phi_train.shape[-1])
        
        K = Phi_train.T @ Phi_train + self.noise_variance * jnp.eye(len(Phi_train.T))
        self.K_inv = self.noise_variance * jnp.linalg.inv(K)
    
    @staticmethod
    @jax.jit
    def null_space(A, rcond=None):
        """
        Construct an orthonormal basis for the null space of A using SVD
        in JAX, in a JIT-compilable way.
        """
        # Compute the SVD of A
        u, s, vh = jax.scipy.linalg.svd(A, full_matrices=True)
        M, N = u.shape[0], vh.shape[1]

        # Set the relative condition number
        if rcond is None:
            rcond = jnp.finfo(s.dtype).eps * max(M, N)

        # Determine the threshold below which singular values are considered zero
        tol = jnp.amax(s) * rcond

        # Create a mask for non-zero singular values
        rank = jnp.sum(s > tol, dtype=int)

        # Zero out rows of vh up to the calculated rank 
        
        mask = jnp.arange(vh.shape[0]) < rank
        
        Q = jnp.where(mask[:, None], 0, vh)

        # Reverse the order of the rows
        return Q
    
    @staticmethod
    def get_Lt(dPhi, Phi, A_theta):
        
        I = jnp.identity(A_theta.shape[0])
        I_dPhi = jnp.kron(I, dPhi)
        A_Phi = jnp.kron(A_theta, Phi)
        
        return I_dPhi - A_Phi
    
    
    def get_Q(self, A_theta):
        
        Phi = self.HL.Psi_t_
        dPhi = self.HL.jac
        L = jax.vmap(self.get_Lt, in_axes = [0, 0, None])(dPhi, Phi, A_theta)
        L = L.reshape(-1, L.shape[-1])
        Q = self.null_space(L, rcond=10e-10)
    
        return Q

    def get_C(self, A_theta):
        
        Phi_train = self.HL.Psi_t_
        dPhi_train = self.HL.jac
        
        L = jax.vmap(self.get_Lt, in_axes = [0, 0, None])(dPhi_train, Phi_train, A_theta)
        L = L.reshape(-1, L.shape[-1])
        C_T = scipy.linalg.null_space(L, rcond=1e-10)
        
        return C_T.T
                

if __name__ == "__main__": 
    
    
    from Archs import load
    import time
    from GaussianProcess import GaussianProcess
    from ODE_Dataloader import solve
    from main import s3_vfield

    # ti = jrandom.uniform(jrandom.PRNGKey(0), (10, 1), minval=0, maxval=5)
    ti = jnp.linspace(0, 5, 5)#, jnp.linspace(0, 0.1, 5)))
    # ti_ = jnp.sort(jnp.append(ti, ti_))
    # ti = jrandom.uniform(jrandom.PRNGKey(0), (10, 1), minval=0, maxval=5)
    # ti = jnp.sort(jnp.append(ti, jnp.linspace(4.9, 5, 1000)))
    t_ = jnp.linspace(0, 5, 1000)#, jnp.linspace(0, 5, 20))
    # t_ = jnp.append(t_, jnp.linspace(0, 0.01, 1000))
    args = jnp.array((10, 90) ) # Assuming these are the true parameters for kcat and K_m
    # args = jnp.array((30, 70))# (30, 70), (25, 45), (100, 10), (10, 100) ) # Assuming these are the
    true_x0 = jnp.array([110.0])  # True initial condition
    synthetic_data = solve(ti, true_x0, args, s3_vfield) # Generating synthetic data
    synthetic_data = synthetic_data# + jax.random.normal(jax.random.PRNGKey(0), shape=synthetic_data.shape) * 10
    
    jax.config.update("jax_enable_x64", False)
    model = load("/Users/antanas/GitRepo/NODE/Models/PLearnKoopman_MM_1D_Lip_10_16.eqx", type = 'PlearnKoopmanLip')
    jax.config.update("jax_enable_x64", True)

    embedded_data = jax.vmap(model.get_latent, in_axes=(0, None))(synthetic_data, args)
    embedded_data_1 = model.get_latent_series(ti, true_x0, args)

    A_theta_ = model.encoderP(args)
    A_theta = model.get_naive(A_theta_)
    parameters = {'scale': 1, 
                'd': 1, 
                'm': 30, 
                'o': 2, 
    }

    gp = ConstrainedGaussianProcess(parameters)
    gp.fit_(X_train=ti, y_train=embedded_data, t_=t_, A_theta=A_theta, noise_variance=0.0001)
    
    # # Generate time values
    time_vals = jnp.linspace(0, 5, 1000)

    # # Generate predictions from the HermiteLayer and the model
    preds, var = gp(time_vals, var=True)
    # # model_pred = jax.vmap(model.get_latent, in_axes=(0, None))(synthetic_data, args)

    # # Extract dimensions for plotting
    dim1_vals, dim2_vals = preds[0].reshape(-1, 1), preds[1].reshape(-1, 1)
    model_dim1_vals, model_dim2_vals = embedded_data[:, 0], embedded_data[:, 1]
    dim1_var, dim2_var = var[0].reshape(-1, 1), var[1].reshape(-1, 1)  # Assuming var has the same shape as preds

   
    plt.figure(figsize=(10, 6))

    # Plotting the dimensions with different line styles
    plt.plot(time_vals, dim1_vals, label='Latent Dimension 1', color='blue', linestyle='-', marker='o', markersize=0.01)
    plt.plot(time_vals, dim2_vals, label='Latent Dimension 2', color='red', linestyle='-', marker='o', markersize=0.01)

    # Plotting the variance as shaded areas
    plt.fill_between(time_vals.flatten(), (dim1_vals - dim1_var).flatten(), (dim1_vals + dim1_var).flatten(), color='blue', alpha=0.2)
    plt.fill_between(time_vals.flatten(), (dim2_vals - dim2_var).flatten(), (dim2_vals + dim2_var).flatten(), color='red', alpha=0.2)

    # Scatter plot for embedded data
    plt.scatter(ti, embedded_data[:, 0], label='Data Dimension 1', color='green', s=20, marker='o')
    plt.scatter(ti, embedded_data[:, 1], label='Data Dimension 2', color='orange', s=20, marker='o')

    # Adding grid, title, and labels
    plt.grid(True)
    plt.title("Linearised Michaelis-Menten System", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Dimension Value", fontsize=12)

    # Adjusting the legend
    plt.legend(loc='lower right', fontsize=10)

    plt.show()
    
    