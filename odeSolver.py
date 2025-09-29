from scipy.integrate import solve_ivp
from dynamics import quadrotor_dynamics

def run_simulation(trajectory, initial_state, params, controllers, gusts_on, dt):
    total_time = trajectory['time'][-1]
    sol = solve_ivp(quadrotor_dynamics, [0, total_time], initial_state, 
                    args=(params, trajectory, controllers, gusts_on, dt), 
                    dense_output=True, t_eval=trajectory['time'], method='RK45')
    return sol