from jax import numpy as jnp
import diffrax
import equinox as eqx

@eqx.filter_jit  
def solve(ts, X0, args, vector_field):
    solution = diffrax.diffeqsolve(diffrax.ODETerm(vector_field), 
                                   diffrax.Tsit5(), 
                                #    diffrax.Dopri5(),
                                #    diffrax.Kvaerno5(),
                                #    diffrax.RadauIIA5()
                                   t0=0, 
                                   t1=ts[-1], 
                                   dt0=0.0001, 
                                   y0=X0, 
                                   args=args, 
                                   saveat=diffrax.SaveAt(ts=ts),
                                   stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-9),
                                #    stepsize_controller=diffrax.PIDController(rtol = 1e-4, atol = 1e-6),
                                   max_steps=3000,
                                   throw=False
                                   )
    return solution.ys

@eqx.filter_jit  
def hess_solve(ts, X0, args, vector_field):
    solution = diffrax.diffeqsolve(diffrax.ODETerm(vector_field), 
                                   diffrax.Dopri5(scan_kind="bounded"), 
                                #    diffrax.Dopri5(),
                                #    diffrax.Kvaerno5(),
                                   t0=0, 
                                   t1=ts[-1]+0.0001, 
                                   dt0=0.0001, 
                                   y0=X0, 
                                   args=args, 
                                   saveat=diffrax.SaveAt(ts=ts),
                                #    stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-8),
                                   stepsize_controller=diffrax.PIDController(rtol = 1e-4, atol = 1e-6),
                                   adjoint=diffrax.DirectAdjoint(),
                                #    max_steps=1000,
                                #    throw=False
                                   )
    return solution.ys


@eqx.filter_jit 
def termination_solver(ts, X0, args, vector_field, term_norm=1e-5):
    
    def termination_condition(state, **kwargs):
        # Here, `state.y` is the solution at the current timestep `state.tprev`.
        # You would define a condition that, when true, should terminate the solve.
        # For example, we check if the norm of the derivative is close to zero.
        derivative_norm = jnp.linalg.norm(vector_field(state.tprev, state.y, args))
        return derivative_norm < term_norm  # Replace with your chosen threshold
    
    event = diffrax.DiscreteTerminatingEvent(cond_fn=termination_condition)
    solution = diffrax.diffeqsolve(diffrax.ODETerm(vector_field), 
                                   diffrax.Tsit5(), 
                                #    diffrax.Dopri5(),
                                #    diffrax.Kvaerno5(),
                                #    diffrax.RadauIIA5()
                                   discrete_terminating_event=event,
                                   t0=0, 
                                   t1=ts[-1], 
                                   dt0=0.0001, 
                                   y0=X0, 
                                   args=args, 
                                   saveat=diffrax.SaveAt(ts=ts),
                                   stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-9),
                                #    stepsize_controller=diffrax.PIDController(rtol = 1e-4, atol = 1e-6),
                                   max_steps=3000,
                                   throw=False
                                   )
    return solution.ys, solution.ts


@eqx.filter_jit  
def safe_solve(ts, X0, args, vector_field):
    solution = diffrax.diffeqsolve(diffrax.ODETerm(vector_field), 
                                   diffrax.Tsit5(), 
                                #    diffrax.Dopri5(),
                                #    diffrax.Kvaerno5(),
                                   t0=0, 
                                   t1=ts[-1], 
                                   dt0=0.0001, 
                                   y0=X0, 
                                   args=args, 
                                   max_steps=1000,
                                   saveat=diffrax.SaveAt(ts=ts),
                                   stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-8),
                                #    stepsize_controller=diffrax.PIDController(rtol = 1e-4, atol = 1e-6),
                                    throw=False
                                   )
    return solution.ts, solution.ys, solution.result