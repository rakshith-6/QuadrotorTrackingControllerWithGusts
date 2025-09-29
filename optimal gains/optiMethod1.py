import numpy as np
from scipy.optimize import minimize

from controllers.pid import create_controllers
from odeSolver import run_simulation

def cost_function(gains_flat, trajectory, params, initial_state, dt):

    gains = {
        'x': {'Kp': gains_flat[0], 'Ki': gains_flat[1], 'Kd': gains_flat[2]},
        'y': {'Kp': gains_flat[3], 'Ki': gains_flat[4], 'Kd': gains_flat[5]},
        'z': {'Kp': gains_flat[6], 'Ki': gains_flat[7], 'Kd': gains_flat[8]},

        'phi': {'Kp': 0.2, 'Ki': 0.0, 'Kd': 0.05},
        'theta': {'Kp': 0.2, 'Ki': 0.0, 'Kd': 0.05},
        'psi': {'Kp': 0.2, 'Ki': 0.0, 'Kd': 0.01}
    }
    controllers = create_controllers(gains)

    sol = run_simulation(trajectory, initial_state, params, controllers, gusts_on=False, dt=dt)

    # Cost calculation - The mean position error norm
    pos_des = trajectory['interp_pos'](sol.t)
    pos_actual = sol.y[:3, :].T
    error_norm = np.linalg.norm(pos_des - pos_actual, axis=1)
    cost = np.mean(error_norm)
    
    print(f"Current Cost: {cost:.4f}")
    
    return cost

def optimize_gains(trajectory, params, initial_state, initial_gains_dict, dt):

    initial_gains_flat = [
        initial_gains_dict['x']['Kp'], initial_gains_dict['x']['Ki'], initial_gains_dict['x']['Kd'],
        initial_gains_dict['y']['Kp'], initial_gains_dict['y']['Ki'], initial_gains_dict['y']['Kd'],
        initial_gains_dict['z']['Kp'], initial_gains_dict['z']['Ki'], initial_gains_dict['z']['Kd']
    ]

    print("--- Starting PID Gain Optimization ---")
    bounds = [(0, 20)] * len(initial_gains_flat)

    # Optimizer
    result = minimize(
        cost_function,
        initial_gains_flat,
        args=(trajectory, params, initial_state, dt),
        method='Nelder-Mead',
        options={'maxiter': 75, 'disp': True}
    )

    print("--- Optimization Finished ---")
    optimal_gains_flat = result.x
    
    optimal_gains_dict = {
        'x': {'Kp': optimal_gains_flat[0], 'Ki': optimal_gains_flat[1], 'Kd': optimal_gains_flat[2]},
        'y': {'Kp': optimal_gains_flat[3], 'Ki': optimal_gains_flat[4], 'Kd': optimal_gains_flat[5]},
        'z': {'Kp': optimal_gains_flat[6], 'Ki': optimal_gains_flat[7], 'Kd': optimal_gains_flat[8]},
        'phi': {'Kp': 0.2, 'Ki': 0.0, 'Kd': 0.05},
        'theta': {'Kp': 0.2, 'Ki': 0.0, 'Kd': 0.05},
        'psi': {'Kp': 0.2, 'Ki': 0.0, 'Kd': 0.01}
    }
    
    return optimal_gains_dict