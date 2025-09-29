import numpy as np

from traj import generate_circle_trajectory, generate_rectangle_trajectory
from odeSolver import run_simulation
from assets.plot import post_process_data, plot_detailed_results, plot_comparison_results 
from optimalGains.optiMethod1 import optimize_gains
from controllers.pid import create_controllers

if __name__ == '__main__':
    params = {'m': 0.034, 'g': 9.81, 'I': np.diag([2.3951e-5, 2.3951e-5, 3.2347e-5])}
    dt = 0.01

    gains_circle_no_gust = {
        'x': {'Kp': 2.3, 'Ki': 0.01, 'Kd': 1.5},
        'y': {'Kp': 1.8, 'Ki': 0.006, 'Kd': 2.5},
        'z': {'Kp': 6.7, 'Ki': 0.02, 'Kd': 2.5},
        'phi': {'Kp': 0.2, 'Ki': 0.0, 'Kd': 0.05},
        'theta': {'Kp': 0.2, 'Ki': 0.0, 'Kd': 0.05},
        'psi': {'Kp': 0.2, 'Ki': 0.0, 'Kd': 0.01}
    }
    gains_circle_with_gust = {
        'x': {'Kp': 2.3, 'Ki': 1.4, 'Kd': 2.3},
        'y': {'Kp': 2.3, 'Ki': 1.4, 'Kd': 2.3},
        'z': {'Kp': 6.0, 'Ki': 0.25, 'Kd': 3.2},
        'phi': {'Kp': 0.35, 'Ki': 0.0, 'Kd': 0.1},
        'theta': {'Kp': 0.35, 'Ki': 0.0, 'Kd': 0.1},
        'psi': {'Kp': 0.4, 'Ki': 0.0, 'Kd': 0.15}
    }
    gains_rect_no_gust = {
        'x': {'Kp': 2.3, 'Ki': 0.01, 'Kd': 1.5},
        'y': {'Kp': 1.5, 'Ki': 0.007, 'Kd': 2.5},
        'z': {'Kp': 6.5, 'Ki': 0.02, 'Kd': 2.7},
        'phi': {'Kp': 0.25, 'Ki': 0.0, 'Kd': 0.08},
        'theta': {'Kp': 0.25, 'Ki': 0.0, 'Kd': 0.08},
        'psi': {'Kp': 0.2, 'Ki': 0.0, 'Kd': 0.01}
    }
    gains_rect_with_gust = {
        'x': {'Kp': 3.0, 'Ki': 0.6, 'Kd': 2.0, 'integral_limit': 0.5},
        'y': {'Kp': 3.0, 'Ki': 0.6, 'Kd': 2.0, 'integral_limit': 0.5},
        'z': {'Kp': 5.0, 'Ki': 0.5, 'Kd': 3.0},
        'phi': {'Kp': 0.4, 'Ki': 0.0, 'Kd': 0.12},
        'theta': {'Kp': 0.4, 'Ki': 0.0, 'Kd': 0.12},
        'psi': {'Kp': 0.45, 'Ki': 0.0, 'Kd': 0.18}
    }
    
    # CIRCLE SIMULATIOM

    print("Generating Circle Trajectory...")
    circle_traj = generate_circle_trajectory(dt=dt)
    initial_state_circle = np.array([1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    print("Running Circle Simulation WITHOUT Gusts...")
    controllers_c_ng = create_controllers(gains_circle_no_gust)
    sol_c_ng = run_simulation(circle_traj, initial_state_circle, params, controllers_c_ng, gusts_on=False, dt=dt)
    processed_c_ng = post_process_data(sol_c_ng, circle_traj, params, controllers_c_ng, dt=dt)
    plot_detailed_results(sol_c_ng, circle_traj, processed_c_ng, "Circle Tracking without Gusts)")
    
    print("Running Circle Simulation WITH Gusts...")
    controllers_c_wg = create_controllers(gains_circle_with_gust)
    sol_c_wg = run_simulation(circle_traj, initial_state_circle, params, controllers_c_wg, gusts_on=True, dt=dt)
    processed_c_wg = post_process_data(sol_c_wg, circle_traj, params, controllers_c_wg, dt=dt)
    plot_detailed_results(sol_c_wg, circle_traj, processed_c_wg, "Circle Tracking with Gusts")

    # Optimal gains and manual tuned gains comparision

    # Find Optimal gains
    optimal_gains = optimize_gains(circle_traj, params, initial_state_circle, gains_circle_no_gust, dt)
    print("\nOptimal Gains Found:")
    for axis, gains in optimal_gains.items():
        if axis in ['x', 'y', 'z']:
             print(f"  {axis}: Kp={gains['Kp']:.3f}, Ki={gains['Ki']:.3f}, Kd={gains['Kd']:.3f}")

    # Run with Optimally tuned gains
    print("\nRunning simulation with OPTIMALLY tuned gains...")
    controllers_optimal = create_controllers(optimal_gains)
    sol_optimal = run_simulation(circle_traj, initial_state_circle, params, controllers_optimal, gusts_on=False, dt=dt)
    print("Simulation with optimal gains complete.")

    # Plot the comparison
    plot_comparison_results(sol_c_ng, sol_optimal, circle_traj, "Manual vs. Optimal Gains Comparison (Circle, No Gusts)")

    # RECTANGLE SIMULATION

    print("\nGenerating Rectangle Trajectory...")
    rect_traj = generate_rectangle_trajectory(dt=dt)
    initial_state_rect = np.array([2.0, 2.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    print("Running Rectangle Simulation WITHOUT Gusts...")
    controllers_r_ng = create_controllers(gains_rect_no_gust)
    sol_r_ng = run_simulation(rect_traj, initial_state_rect, params, controllers_r_ng, gusts_on=False, dt=dt)
    processed_r_ng = post_process_data(sol_r_ng, rect_traj, params, controllers_r_ng, dt=dt)
    plot_detailed_results(sol_r_ng, rect_traj, processed_r_ng, "Rectangle Tracking without Gusts)")
    
    print("Running Rectangle Simulation WITH Gusts...")
    controllers_r_wg = create_controllers(gains_rect_with_gust)
    sol_r_wg = run_simulation(rect_traj, initial_state_rect, params, controllers_r_wg, gusts_on=True, dt=dt)
    processed_r_wg = post_process_data(sol_r_wg, rect_traj, params, controllers_r_wg, dt=dt)
    plot_detailed_results(sol_r_wg, rect_traj, processed_r_wg, "Rectangle Tracking with Gusts") 

    # Optimal gains and manual tuned gains comparision

    # Find Optimal gains
    optimal_r_gains = optimize_gains(rect_traj, params, initial_state_rect, gains_rect_no_gust, dt)
    print("\nOptimal Gains Found:")
    for axis, gains in optimal_r_gains.items():
        if axis in ['x', 'y', 'z']:
             print(f"  {axis}: Kp={gains['Kp']:.3f}, Ki={gains['Ki']:.3f}, Kd={gains['Kd']:.3f}")

    # Run with Optimally tuned gains
    print("\nRunning simulation with OPTIMALLY tuned gains...")
    controllers_r_optimal = create_controllers(optimal_r_gains)
    sol_r_optimal = run_simulation(rect_traj, initial_state_rect, params, controllers_r_optimal, gusts_on=False, dt=dt)
    print("Simulation with optimal gains complete.")

    # Plot the comparison
    plot_comparison_results(sol_r_ng, sol_r_optimal, rect_traj, "Manual vs. Optimal Gains Comparison (Rectangle, No Gusts)")