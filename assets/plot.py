def post_process_data(sol, trajectory, params, controllers, dt):
    m, g = params['m'], params['g']
    phi_des_hist, theta_des_hist = [], []

    for i, t in enumerate(sol.t):
        state = sol.y[:, i]
        pos, vel = state[0:3], state[3:6]
        
        pos_des = trajectory['interp_pos'](t)
        vel_des = trajectory['interp_vel'](t)
        accel_des = trajectory['interp_accel'](t)
        
        pos_error, vel_error = pos_des - pos, vel_des - vel
        
        feedback_ax = controllers['x'].update(pos_error[0], vel_error[0], dt)
        feedback_ay = controllers['y'].update(pos_error[1], vel_error[1], dt)
        feedback_az = controllers['z'].update(pos_error[2], vel_error[2], dt)
        
        ax_des = accel_des[0] + feedback_ax
        ay_des = accel_des[1] + feedback_ay
        az_des = accel_des[2] + feedback_az
        
        thrust_cmd = m * (g + az_des)
        T = max(0.1, thrust_cmd)
        
        sin_phi_des_val = (1.0/T) * (m * ax_des * np.sin(0) - m * ay_des * np.cos(0))
        phi_des = np.arcsin(np.clip(sin_phi_des_val, -1.0, 1.0))
        
        cos_phi_des = np.cos(phi_des)
        if abs(cos_phi_des) < 1e-6: cos_phi_des = 1e-6
        sin_theta_des_val = (1.0/(T*cos_phi_des)) * (m*ax_des*np.cos(0) + m*ay_des*np.sin(0))
        theta_des = np.arcsin(np.clip(sin_theta_des_val, -1.0, 1.0))
        
        phi_des_hist.append(phi_des)
        theta_des_hist.append(theta_des)

    pos_des_hist = trajectory['interp_pos'](sol.t)
    pos_err = pos_des_hist - sol.y[0:3, :].T
    
    att_actual = sol.y[6:9, :]
    att_des = np.array([phi_des_hist, theta_des_hist, np.zeros_like(sol.t)])
    att_err = att_des - att_actual
    
    return {'pos_err': pos_err, 'att_err': att_err}

def plot_detailed_results(sol, trajectory, processed_data, title):
    t = sol.t
    pos_des = trajectory['pos']
    pos_actual = sol.y[:3, :].T
    pos_err = processed_data['pos_err']
    att_err = processed_data['att_err']
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # 3D plot of trajectory tracking
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(pos_des[:,0], pos_des[:,1], pos_des[:,2], 'k--', label='Desired')
    ax1.plot(pos_actual[:,0], pos_actual[:,1], pos_actual[:,2], 'b', label='Actual')
    ax1.set_title('3D Plot of Trajectory Tracking'); ax1.set_xlabel('X (m)'); 
    ax1.set_ylabel('Y (m)'); 
    ax1.set_zlabel('Z (m)'); 
    ax1.legend()

    # Position Errors
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t, pos_err[:, 0], label='Error X'); 
    ax2.plot(t, pos_err[:, 1], label='Error Y'); 
    ax2.plot(t, pos_err[:, 2], label='Error Z')
    ax2.set_title('Position Errors'); 
    ax2.set_xlabel('Time (s)'); 
    ax2.set_ylabel('Error (m)'); 
    ax2.grid(True); ax2.legend()
    
    # Attitude Errors
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t, np.rad2deg(att_err[0, :]), label='Error Roll (φ)'); 
    ax3.plot(t, np.rad2deg(att_err[1, :]), label='Error Pitch (θ)'); 
    ax3.plot(t, np.rad2deg(att_err[2, :]), label='Error Yaw (ψ)')
    ax3.set_title('Attitude Errors'); 
    ax3.set_xlabel('Time (s)'); 
    ax3.set_ylabel('Error (degrees)'); 
    ax3.grid(True); ax3.legend()

    # 4. Position Error Norm
    ax4 = fig.add_subplot(2, 2, 4)
    error_norm = np.linalg.norm(pos_err, axis=1)
    ax4.plot(t, error_norm); ax4.set_title('Position Error Norm'); ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Error (m)'); ax4.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

def plot_comparison_results(sol_manual, sol_optimal, trajectory, title):
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(title, fontsize=16)

    # 3D Trajectory Comparison
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    pos_des = trajectory['pos']
    ax1.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2], 'k--', label='Desired')
    ax1.plot(sol_manual.y[0, :], sol_manual.y[1, :], sol_manual.y[2, :], 'b', label='Manual Gains')
    ax1.plot(sol_optimal.y[0, :], sol_optimal.y[1, :], sol_optimal.y[2, :], 'r', label='Optimal Gains')
    ax1.set_title('3D Trajectory Tracking Comparison')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    ax1.legend()

    # Position Error Norm Comparison
    ax2 = fig.add_subplot(1, 2, 2)
    pos_des_interp = trajectory['interp_pos'](sol_manual.t)
    
    error_norm_manual = np.linalg.norm(pos_des_interp - sol_manual.y[:3, :].T, axis=1)
    error_norm_optimal = np.linalg.norm(pos_des_interp - sol_optimal.y[:3, :].T, axis=1)
    
    ax2.plot(sol_manual.t, error_norm_manual, 'b', label=f'Manual Gains (Mean Err: {np.mean(error_norm_manual):.3f} m)')
    ax2.plot(sol_optimal.t, error_norm_optimal, 'r', label=f'Optimal Gains (Mean Err: {np.mean(error_norm_optimal):.3f} m)')
    ax2.set_title('Position Error Norm Comparison')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Error (m)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()