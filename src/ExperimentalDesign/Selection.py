import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.numpy.linalg import pinv, slogdet, eigvalsh, inv
import matplotlib.pyplot as plt
import equinox as eqx
import time
import scipy
from icecream import ic

class GreedyBasis:

    def __init__(self):
        pass

    @staticmethod
    @jit
    def P_X(X):
        return pinv(X) @ X

    def W_lambda(self, X, C, lam):
        return C @ pinv(X.T @ X + jnp.eye(X.shape[1]) * lam) @ C.T

    @staticmethod
    @jit
    def bias_variance_matrix(X, C, sigma, tau, lam, prior):
        if lam is None:
            pinv = pinv(X.T @ X)
            P = pinv @ (X.T @ X)
            var = (sigma ** 2) * (C @ pinv @ C.T)
            bias = (tau ** 2) * C @ (jnp.eye(X.shape[1]) - P) @ C.T
            return jnp.trace(var), jnp.trace(bias), jnp.trace(var + bias)
        else:
            I = jnp.eye(X.shape[0])
            if prior is not None:
                Kinv = inv(sigma ** 2 * (I + prior) + lam * X @ X.T)
            else:
                Kinv = inv(sigma ** 2 * I + lam * X @ X.T)
            full = -lam ** 2 * C @ X.T @ Kinv @ X @ C.T
            return 0., 0., jnp.trace(full)

    @staticmethod
    @jit
    def select_next_point(X, selection, bias_variance_func, mode, repeats):
        n = X.shape[0]
        vals = []
        indvals = []

        for j in range(n):
            if j not in selection or repeats:
                x = X[j, :].reshape(1, -1)
                Xnew = jnp.concatenate((X[selection, :], x))
                score_var, score_bias, score_combined = bias_variance_func(Xnew)
                if mode == 'bias_var':
                    score = score_combined
                elif mode == 'bias':
                    score = score_bias
                elif mode == "var":
                    score = score_var
                vals.append(score)
                indvals.append(j)

        vals = jnp.array(vals)
        indvals = jnp.array(indvals)
        next_point = jnp.argmin(vals)
        return indvals[next_point]

    @staticmethod
    @jit
    def greedy_basis(self, C, X, sigma, tau, budget, eps=None, lam=1e-1, repeats=False,
                     scalarization='A', mode='bias_var', visualize=False, verbose=False, rank1=False, prior=None):

        d = X.shape[1]
        n = X.shape[0]
        I = jnp.eye(d)
        k = C.shape[0]

        variance = lambda X: (sigma ** 2) * self.W_lambda(X, C, lam)
        bias = lambda X: (tau ** 2) * C @ (self.P_X(X) - I) @ C.T @ self.W_lambda(X, C, lam) @ C @ (self.P_X(X) - I) @ C.T

        bias_variance_func = jit(lambda X: self.bias_variance_matrix(X, C, sigma, tau, lam, prior))

        selection = []
        if eps is None:
            for _ in range(budget):
                next_point = self.select_next_point(X, selection, bias_variance_func, mode, repeats)
                if verbose:
                    print("Adding:", next_point)
                selection.append(next_point)
            value = self.bias_variance_matrix(X[selection], C, sigma, tau, lam, prior)
            return selection, value
        else:
            bias_score = 1e10
            while (bias_score > eps or len(selection) < k) and (len(selection) < budget):
                next_point = self.select_next_point(X, selection, bias_variance_func, mode, repeats)
                selection.append(next_point)
                bias_score = bias_variance_func(X[selection])[1]
                if verbose:
                    print(len(selection), selection, bias_score, eps)
            return selection, bias(X[selection])


    @staticmethod
    @jit
    def greedy_fast(C, X, sigma, tau, budget, eps=None, lam=1e-1, repeats=False, scalarization='A', mode='bias_var', verbose=False):
        assert scalarization == 'A'
        assert mode == 'bias_var'

        score = jnp.linalg.norm(X @ C.T, axis=1)
        start = jnp.argmax(score)
        Xbasis = X[start, :].reshape(1, -1)
        I = jnp.eye(X.shape[1])
        selection = [start]

        for i in range(budget):
            invV = jnp.linalg.inv(Xbasis.T @ Xbasis + I * lam * sigma ** 2)
            V_curr = C @ invV @ C.T
            vals = X @ invV @ C.T  # (n, d)
            bottom_score = 1 + jnp.einsum('ij,jk,ki->i', X, invV, X.T)
            top_score = jnp.einsum('ij,ij->i', vals, vals)
            scores = sigma ** 2 * (jnp.trace(V_curr) + top_score / bottom_score)
            k = jnp.argmax(scores)
            selection.append(k)
            Xbasis = X[selection, :]
            if verbose:
                print("Adding", k)
        return selection, 0
    
    @staticmethod
    @jit
    def select_next_point(X, selection, bias_variance_func, mode, repeats):
        n = X.shape[0]
        vals = []
        indvals = []

        for j in range(n):
            if j not in selection or repeats:
                x = X[j, :].reshape(1, -1)
                Xnew = jnp.concatenate((X[selection, :], x))
                score_var, score_bias, score_combined = bias_variance_func(Xnew)
                if mode == 'bias_var':
                    score = score_combined
                elif mode == 'bias':
                    score = score_bias
                elif mode == "var":
                    score = score_var
                vals.append(score)
                indvals.append(j)

        vals = jnp.array(vals)
        indvals = jnp.array(indvals)
        next_point = jnp.argmin(vals)
        return indvals[next_point]
    
    @staticmethod
    @eqx.filter_jit
    def select_next_point_(X, C, Info, selection, criterion, repeats):
        
        n = X.shape[0]
        vals = []
        indvals = []


        def fun_body(i, lst):
            vals, indvals = lst
            if i not in selection or repeats:
                x = X[i]
                Info_new = Info + x @ x.T
                score = criterion(Info_new, C)
                vals.append(score)
                indvals.append(i)
            return [vals, indvals]

        vals, indvals = jax.lax.fori_loop(0, n, fun_body, [vals, indvals])


        # for j in range(n):
        #     if j not in selection or repeats:
        #         x = X[j]
        #         Info_new = Info + x @ x.T
        #         score = criterion(Info_new, C)
        #         vals.append(score)
        #         indvals.append(j)
                
                
        vals = jnp.array(vals)
        indvals = jnp.array(indvals)
        next_point = jnp.argmax(vals)
        return indvals[next_point]
   
    @staticmethod
    @eqx.filter_jit
    def A(Info, C):
        return jnp.trace( C @ ((Info) ** -1) @C.T )
    
    def greedy_simple(self, X, C, budget, lam=0.1, letter="A", prior=None, repeats=False):
 
        m = C.shape[1]       
        selection = []
        
        Info = prior if prior + jnp.eye(m) * lam is not None else jnp.eye(m) * lam
            
        if letter == "A":
            criterion = lambda Info, C: self.A(Info, C)
        else:
            raise NotImplementedError
        
        for _ in range(budget):
            
            k = self.select_next_point_(X, C, Info, selection, criterion=criterion, repeats=repeats)
            selection.append(k)
            Info = Info + X[k] @ X[k].T
            
        return selection, Info


class GreedySelector:
    
    def __init__(self):
        return
    
    #----------------------------------------------------------------
    # For loop based - good when budget is large
    #----------------------------------------------------------------
    @staticmethod
    @eqx.filter_jit
    def select_next_point_(X, C, Info, selection, criterion, repeats):
        n = X.shape[0]
        vals = jnp.inf * jnp.ones(n)

        selected_mask = jnp.isin(jnp.arange(n), selection)

        def true_fun(i, vals):
            x = X[i]
            Info_new = Info + x @ x.T
            score = criterion(Info_new, C)
            vals = vals.at[i].set(score)
            return vals

        def false_fun(i, vals):
            return vals

        def fun_body(i, vals):
            condition = jnp.logical_or(~selected_mask[i], repeats)
            return jax.lax.cond(condition, true_fun, false_fun, i, vals)

        vals = jax.lax.fori_loop(0, n, fun_body, vals)
        
        next_point = jnp.argmin(vals)
        
        return next_point


    #----------------------------------------------------------------
    # vmap based - wins everytime...
    #----------------------------------------------------------------
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
        # Info_inv = jax.numpy.linalg.inv(Info)
        # tr = jnp.trace( C @ Info_inv @C.T )
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
    
    
    
    #----------------------------------------------------------------
    # vmap based - wins everytime...
    #----------------------------------------------------------------
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
        # vals = vals - jnp.mean(vals)
        # ic(vals)
        
        if bayes:
            vals = jnp.sum(vals, axis=0)
        else:
            vals = jnp.max(vals, axis=0)
        
        # def true_fun(vals):
        #     return vals.at[selection[selection != -1]].set(jnp.inf)

        # def false_fun(vals):
        #     return vals

        # vals = jax.lax.cond(not repeats, lambda _: true_fun(vals), lambda _: false_fun(vals), operand=None)
        vals = vals.at[selection[selection != -1]].set(jnp.inf) if not repeats else vals

        next_point = jnp.argmin(vals)
        
        # plt.plot(vals, label = "At: " + str(next_point))
        # plt.legend()
        # plt.show()
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
            # ic("Before", selection)
            selection = selection.at[k].set(-1)
            # put all -1s at the end
            selection = jnp.concatenate((selection[selection != -1], selection[selection == -1]))
            # ic("After", selection)
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
            # ic("Before", selection)
            
            selection = selection.at[j].set(k)
            # ic("After", selection)
            Info = Info + XTX[k]
            
        return Info, selection             
                        
    
    def greedy_min_max(self, XTX, C, budget, lam=0.1, sigma = 1, letter="A", prior=None, repeats=False):
        
        m = XTX.shape[1]   
        #changed this
        # Info_X = jax.vmap(lambda x: x.T @ x)(X)
        
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
                                                #Changed This
            k = self.select_next_point_min_max(X=XTX, C=C, Info=Info, selection=selection, criterion=criterion, repeats=repeats)
            selection = selection.at[_].set(k)
            Info = Info + XTX[k] #X[k].T @ X[k]

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
        
        # Reg = prior + jnp.eye(m) * lam if prior is not None else jnp.eye(m) * lam
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
        
        # Info = prior + jnp.eye(m) * lam * sigma ** 2 if prior is not None else jnp.eye(m) * lam * sigma ** 2
        Info = jnp.zeros_like(XTX[0])
        
                #changed this
        # Info_X = jax.vmap(lambda x: x.T @ x)(X)
        # Info_X = XTX
        if letter == "A":
            criterion = lambda Info, C, prior: self.A(Info, C, prior = prior)
        elif letter == "E":
            criterion = lambda Info, C: self.E(Info, C)
        elif letter == "D":
            criterion = lambda Info, C: self.D(Info, C)
        else:
            raise NotImplementedError
        
        # def plot_info(t_, Info_matrices, k=None):
        #     # for matrix in Info_matrices:
        #     plt.plot(t_, jax.vmap(lambda matrix: 1/sigma**2 * jnp.trace( C[0] @ jax.numpy.linalg.inv(matrix + sigma**2 * lam * jnp.eye(matrix.shape[0]))@ C[0].T))(Info_matrices), label = f"At: {k}")
        #     print
        #     plt.legend()
        #     plt.show()
            
        for _ in range(budget):
            
            k = self.select_next_point_min_max(X=XTX, C=C, Info=Info, sigma = sigma,
                                               selection=selection, criterion=criterion, 
                                               prior=prior, repeats=repeats, bayes = bayes)
            
            # plot_info(range(len(Info_X)), Info + Info_X, k)            
            
            selection = selection.at[_].set(k)
            Info = Info + XTX[k] #X[k].T @ X[k]
            
        return selection, Info
    
    def greedy_min_max_robust_RA(self, XTX, C, budget, lam=0.1, sigma = 1, letter="A", 
                                     prior=None, repeats=False, bayes = False, 
                                     RA = None, RA_budget = None):
        
        m = XTX.shape[-1]   

        selection = -1 * jnp.ones(budget, dtype = jnp.int32) 
        
        # Info = prior + jnp.eye(m) * lam * sigma ** 2 if prior is not None else jnp.eye(m) * lam * sigma ** 2
        Info = jnp.zeros_like(XTX[0])
                #changed this
        # Info_X = jax.vmap(lambda x: x.T @ x)(X)
        # Info_X = XTX
        if letter == "A":
            criterion = lambda Info, C, prior: self.A(Info, C, prior)
        elif letter == "E":
            criterion = lambda Info, C: self.E(Info, C)
        elif letter == "D":
            criterion = lambda Info, C: self.D(Info, C)
        else:
            raise NotImplementedError
        
        # def plot_info(t_, Info_matrices, k=None):
        #     # for matrix in Info_matrices:
        #     plt.plot(t_, jax.vmap(lambda matrix: 1/sigma**2 * jnp.trace( C[0] @ jax.numpy.linalg.inv(matrix + sigma**2 * lam * jnp.eye(matrix.shape[0]))@ C[0].T))(Info_matrices), label = f"At: {k}")
        #     print
        #     plt.legend()
        #     plt.show()
            
        for _ in range(budget):
            
            k = self.select_next_point_min_max(X=XTX, C=C, Info=Info, sigma = sigma,
                                               selection=selection, criterion=criterion, 
                                               prior=prior, repeats=repeats, bayes = bayes)
            
            # plot_info(range(len(Info_X)), Info + Info_X, k)            
            
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
    
    
    # def LogLikelihoodSelection(self,)
    
    
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