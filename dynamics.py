import numpy as np

def quadrotor_dynamics(t, state, params, trajectory, controllers, gusts_on, dt):
    m, g, I = params['m'], params['g'], params['I']
    pos, vel, angles, rates = state[0:3], state[3:6], state[6:9], state[9:12]
    phi, theta, psi = angles
    p, q, r = rates

    # Desired states of the trajectory
    pos_des = trajectory['interp_pos'](t)
    vel_des = trajectory['interp_vel'](t)
    accel_des = trajectory['interp_accel'](t) # The feedforward acceleration
    psi_des = 0.0

    # OUTER LOOP(Position control)

    # Errors in position and velocity
    pos_error = pos_des - pos
    vel_error = vel_des - vel

    # PID + Feedforward Control Law -> Calculate feedback acceleration from PID controllers
    feedback_ax = controllers['x'].update(pos_error[0], vel_error[0], dt)
    feedback_ay = controllers['y'].update(pos_error[1], vel_error[1], dt)
    feedback_az = controllers['z'].update(pos_error[2], vel_error[2], dt)

    # Combination of the feedforward and feedback
    ax_des = accel_des[0] + feedback_ax
    ay_des = accel_des[1] + feedback_ay
    az_des = accel_des[2] + feedback_az

    # Calculation of total thrust command from desired vertical acceleration
    thrust_cmd = m * (g + az_des)
    T = max(0.1, thrust_cmd)

    # phi desired calculation 
    c_psi, s_psi = np.cos(psi_des), np.sin(psi_des)
    sin_phi_des_val = (1.0 / T) * (m * ax_des * s_psi - m * ay_des * c_psi)
    phi_des = np.arcsin(np.clip(sin_phi_des_val, -1.0, 1.0))

    # theta desired calculation
    cos_phi_des = np.cos(phi_des)
    if abs(cos_phi_des) < 1e-6: 
        cos_phi_des = 1e-6
    sin_theta_des_val = (1.0 / (T * cos_phi_des)) * (m * ax_des * c_psi + m * ay_des * s_psi)
    theta_des = np.arcsin(np.clip(sin_theta_des_val, -1.0, 1.0))

    # INNER LOOP(Attitude control)
    angle_error = np.array([phi_des - phi, theta_des - theta, psi_des - psi])
    angle_error = np.clip(angle_error, -np.pi/2, np.pi/2)
    
    tau = np.array([
        controllers['phi'].update(angle_error[0], -p, dt), 
        controllers['theta'].update(angle_error[1], -q, dt),
        controllers['psi'].update(angle_error[2], -r, dt)
    ])

    # Disturbances
    Fg = np.zeros(3)
    if gusts_on:
        Fg = np.array([0.05, 0.05, 0.0])

    # R - B to I
    R = np.array([
        [np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi), np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi)],
        [np.sin(psi)*np.cos(theta), np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi)],
        [-np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)]
    ])

    # Dynamic equation
    accel = (R @ np.array([0, 0, T]) + np.array([0, 0, -m * g]) - Fg) / m
    
    omega = np.array([p, q, r])
    ang_accel = np.linalg.inv(I) @ (tau - np.cross(omega, I @ omega))
    
    kinematic_matrix = np.array([
        [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
    ])
    angle_rates = kinematic_matrix @ omega
    
    return np.concatenate([vel, accel, angle_rates, ang_accel])