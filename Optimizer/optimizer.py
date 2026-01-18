import numpy as np
import cvxpy as cp

### TO DO: ADJUST FOR LOG RETURNS ### 
def optimize_weights(mu, cov_mat, u=0.10):

    # num assets - fill with data once we have it (use len)
    N = len(mu)

    # Decision variable - weights of the stocks
    x = cp.Variable(N)

    # Objective function - minimize variance
    objective = cp.Minimize(cp.quad_form(x, cov_mat))

    # Constraints 
    # - Should sum to 1
    # - targert return (tbd)
    # - long only (tbd)
    constraints = [
        cp.sum(x) == 1,
        #mu @ x >= u,
        x >= 0   # long-only (remove if shorting allowed)
    ]

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # debugging
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed: {problem.status}")
    
    # extract values
    opt_weights = x.value

    # return weights
    return opt_weights

