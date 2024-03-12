#----------------------------------------------------------------
# Greedy selection of designs
#----------------------------------------------------------------

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.numpy.linalg import pinv, slogdet, eigvalsh, inv
import matplotlib.pyplot as plt
import equinox as eqx
import time
import scipy
from icecream import ic


class GreedySelector:
    
    def __init__(self):
        return
    

    @staticmethod
    @eqx.filter_jit
    def select_next_point_(X, C, Info, selection, criterion, repeats):
        n = X.shape[0]
        vals = jnp.inf * jnp.ones(n)

        def fun(Info, Xi):
            Info_new = Info + Xi @ Xi.T
            score = criterion(Info_new, C)
            return score
        
        vals = jax.vmap(fun, in_axes = [None, 0])(Info, X)
        
        vals = vals.at[selection].set(jnp.inf) if not repeats else vals
        next_point = jnp.argmin(vals)
        
        return next_point
   
    @staticmethod
    @jit
    def A(Info, C, prior=None):
        prior = prior if prior is not None else jnp.zeros_like(Info)
        return jnp.trace( C @ jax.numpy.linalg.pinv(Info + prior) @C.T )
    
    @staticmethod
    @jit
    def E(Info, C):
        return jnp.linalg.eigvalsh( C @ jax.numpy.linalg.inv(Info) @ C.T )[-1]
    
    @staticmethod
    @jit
    def D(Info, C):
        _, det = jnp.linalg.slogdet( C @ jax.numpy.linalg.inv(Info) @ C.T )
        return det
    
    
    def greedy_simple(self, X, C, budget, lam=0.1, letter="A", prior=None, repeats=False):
 
        m = C.shape[1]   

        selection = -1 * jnp.ones(budget, dtype = jnp.int32) 
        
        Info = prior + jnp.eye(m) * lam if prior is not None else jnp.eye(m) * lam
            
        if letter == "A":
            criterion = lambda Info, C: self.A(Info, C)
        elif letter == "E":
            criterion = lambda Info, C: self.E(Info, C)
        elif letter == "D":
            criterion = lambda Info, C: self.D(Info, C)
        else:
            raise NotImplementedError
        
        for _ in range(budget):
            
            k = self.select_next_point_(X=X, C=C, Info=Info, selection=selection, criterion=criterion, repeats=repeats)
            selection = selection.at[_].set(k)
            
            Info = Info + X[k] @ X[k].T
            
        return selection, Info
    
    
    
    @staticmethod
    # @eqx.filter_jit()
    def select_next_point_min_max(X, C, Info, selection, criterion, sigma, repeats, prior=None, bayes = False):
        
        n = X.shape[0]
        vals = jnp.inf * jnp.ones(n)

        def fun(Info, Xi, C, sigma, prior=None):
            Info_new = Info + Xi #Xi.T @ Xi Changed this
            score = 1/sigma**2 * criterion(Info_new, C, prior)
            return score
        
        map1 = jax.vmap(fun, in_axes = [None, 0, None, None, None])
        vals = jax.vmap(map1, in_axes = [None, None, 0, None, None])(Info, X, C, sigma, prior)
        
        if bayes:
            vals = jnp.sum(vals, axis=0)
        else:
            vals = jnp.max(vals, axis=0)
        
        vals = vals.at[selection[selection != -1]].set(jnp.inf) if not repeats else vals

        next_point = jnp.argmin(vals)

        return next_point
    
    @staticmethod
    def select_next_remove(XTX, C, Info, selection, criterion, sigma, prior = None, bayes = False):

        def fun(Info, Xi, C, sigma, prior=None):
            Info_new = Info - Xi #Xi.T @ Xi Changed this
            score = 1/sigma**2 * criterion(Info_new, C, prior)
            return score
                
        candidates = XTX[selection[selection != -1]]
        
        map1 = jax.vmap(fun, in_axes = [None, 0, None, None,None])
        vals = jax.vmap(map1, in_axes = [None, None, 0, None, None])(Info, candidates, C, sigma, prior)
        
        if bayes:
            vals = jnp.sum(vals, axis=0)
        else:
            vals = jnp.max(vals, axis=0)
            
        return jnp.argmin(vals)
        
    def greedy_multi_remove(self, XTX, C, Info, selection, budget, prior = None, sigma = 1, letter="A", bayes = False):
                            
        if letter == "A":
            criterion = lambda Info, C, prior: self.A(Info, C, prior)
        elif letter == "E":
            criterion = lambda Info, C: self.E(Info, C)
        elif letter == "D":
            criterion = lambda Info, C: self.D(Info, C)
        else:
            raise NotImplementedError
        
        for _ in range(budget):
            k = self.select_next_remove(XTX=XTX, C=C, Info=Info, selection=selection, criterion=criterion, sigma=sigma, prior=prior, bayes = bayes)
            selection = selection.at[k].set(-1)
            selection = jnp.concatenate((selection[selection != -1], selection[selection == -1]))
            Info = Info - XTX[k]
            
        return Info, selection
    
    def greedy_multi_add(self, XTX, C, Info, selection, budget, prior = None, sigma = 1, letter="A", bayes = False, repeats = False):
                                
        if letter == "A":
            criterion = lambda Info, C, prior: self.A(Info, C, prior)
        elif letter == "E":
            criterion = lambda Info, C: self.E(Info, C)
        elif letter == "D":
            criterion = lambda Info, C: self.D(Info, C)
        else:
            raise NotImplementedError
        
        for i, j in enumerate(jnp.where(selection == -1)[0]):
            
            if i >= budget:
                break
            
            k = self.select_next_point_min_max(X=XTX, C=C, Info=Info, selection=selection, 
                                               criterion=criterion, sigma=sigma, prior=prior,
                                               repeats=repeats, bayes=bayes)            
            selection = selection.at[j].set(k)
            Info = Info + XTX[k]
            
        return Info, selection             
                        
    
    def greedy_min_max(self, XTX, C, budget, lam=0.1, sigma = 1, letter="A", prior=None, repeats=False):
        
        m = XTX.shape[1]   
        
        selection = -1 * jnp.ones(budget, dtype = jnp.int32) 
        
        Info = prior + jnp.eye(m) * lam * sigma ** 2 if prior is not None else jnp.eye(m) * lam * sigma ** 2
            
        if letter == "A":
            criterion = lambda Info, C: self.A(Info, C)
        elif letter == "E":
            criterion = lambda Info, C: self.E(Info, C)
        elif letter == "D":
            criterion = lambda Info, C: self.D(Info, C)
        else:
            raise NotImplementedError
        

        for _ in range(budget):
            k = self.select_next_point_min_max(X=XTX, C=C, Info=Info, selection=selection, criterion=criterion, repeats=repeats)
            selection = selection.at[_].set(k)
            Info = Info + XTX[k] 

        return selection, Info
    
    @staticmethod
    @jit
    def A_robust(Reg, X, C):
        I = jnp.eye(X.shape[0]) * Reg
        Info_inv = jax.numpy.linalg.inv(I + X.T @ X)
        return jnp.trace((C @ Info_inv @ C.T))

    
    @staticmethod
    @eqx.filter_jit
    def select_next_point_min_max_robust(X, X_, C, Reg, selection, criterion, repeats):
        
        n = X.shape[0]
        vals = jnp.inf * jnp.ones(n)
        
        def fun(Reg, X_, Xi, C):
            
            if X_ is not None:
                X_i = jnp.concatenate((X_, Xi), axis=0)
            else:
                X_i = Xi
                
            score = criterion(Reg, X_i, C)
            return score
        
        map1 = jax.vmap(fun, in_axes = [None, None, 0, None])
        vals = jax.vmap(map1, in_axes = [None, None, None, 0])(Reg, X_, X, C)
        vals = jnp.max(vals, axis=0)
        vals = vals.at[selection].set(jnp.inf) if not repeats else vals
        next_point = jnp.argmin(vals)
        
        return next_point
    
    def greedy_min_max_robust(self, X, C, budget, lam=0.1, letter="A", prior=None, repeats=False):
        
        m = X.shape[1]   

        selection = -1 * jnp.ones(budget, dtype = jnp.int32) 
        
        Reg = lam
        
        if letter == "A":
            criterion = lambda Reg, X, C: self.A_robust(Reg, X, C)
        else:
            raise NotImplementedError

        X_ = None
        
        for _ in range(budget):
            k = self.select_next_point_min_max_robust(X=X, X_=X_, C=C, Reg=Reg, selection=selection, criterion=criterion, repeats=repeats)
            selection = selection.at[_].set(k)
            if X_ is not None:
                X_ = jnp.concatenate((X_, X[k]), axis=0)
            else:
                X_ = X[k]
        return selection, X_
   
    def greedy_min_max_robust_simple(self, XTX, C, budget, lam=0.1, sigma = 1, letter="A", 
                                     prior=None, repeats=False, bayes = False):
        
        m = XTX.shape[-1]   

        selection = -1 * jnp.ones(budget, dtype = jnp.int32) 
        
        Info = jnp.zeros_like(XTX[0])

        if letter == "A":
            criterion = lambda Info, C, prior: self.A(Info, C, prior = prior)
        elif letter == "E":
            criterion = lambda Info, C: self.E(Info, C)
        elif letter == "D":
            criterion = lambda Info, C: self.D(Info, C)
        else:
            raise NotImplementedError
        
        for _ in range(budget):
            
            k = self.select_next_point_min_max(X=XTX, C=C, Info=Info, sigma = sigma,
                                               selection=selection, criterion=criterion, 
                                               prior=prior, repeats=repeats, bayes = bayes)
                        
            selection = selection.at[_].set(k)
            Info = Info + XTX[k] #X[k].T @ X[k]
            
        return selection, Info
    
    def greedy_min_max_robust_RA(self, XTX, C, budget, lam=0.1, sigma = 1, letter="A", 
                                     prior=None, repeats=False, bayes = False, 
                                     RA = None, RA_budget = None):
        
        m = XTX.shape[-1]   

        selection = -1 * jnp.ones(budget, dtype = jnp.int32) 
        
        Info = jnp.zeros_like(XTX[0])

        if letter == "A":
            criterion = lambda Info, C, prior: self.A(Info, C, prior)
        elif letter == "E":
            criterion = lambda Info, C: self.E(Info, C)
        elif letter == "D":
            criterion = lambda Info, C: self.D(Info, C)
        else:
            raise NotImplementedError
        

            
        for _ in range(budget):
            
            k = self.select_next_point_min_max(X=XTX, C=C, Info=Info, sigma = sigma,
                                               selection=selection, criterion=criterion, 
                                               prior=prior, repeats=repeats, bayes = bayes)
            
            selection = selection.at[_].set(k)
            Info = Info + XTX[k] #X[k].T @ X[k]
        
        if RA is not None and RA_budget is not None:
            
            if RA_budget >= budget:
                raise ValueError("Remove-Add budget must be less than the actual budget")
            else:
                for _ in range(RA):
                    
                    Info, selection = self.greedy_multi_remove(XTX, C, Info, selection, RA_budget, 
                                                               prior=prior, sigma=sigma, letter=letter)
                    Info, selection = self.greedy_multi_add(XTX, C, Info, selection, RA_budget, 
                                                            prior=prior, sigma=sigma, letter=letter)
                    
        return selection, Info
        
    
if __name__ == "__main__":
    # Create mock data for X and C
    n, m, o, k = 20, 5, 3, 4  # example dimensions
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (n, m) )# Random data for X
    C = jax.random.normal(key, (k, k, m) )   # Random data for C

    zeros = jnp.zeros((k, k, m))
    C = jnp.concatenate((zeros, C), axis=1)
    
    # Initialize class and call method
    gs = GreedySelector()
    budget = 10  # example budget
    
    time_start = time.time()
    selection, Info = gs.greedy_min_max_robust(X, C, budget, letter="A")
    time_end = time.time()
    # Print results
    print("Selection:", selection)
    print("Info Matrix:", Info)
    print("Time taken:", time_end - time_start)