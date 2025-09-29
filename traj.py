import numpy as np
import scipy.interpolate as interp1d

# CIRCLE TRAJECTORY

def generate_circle_trajectory(total_time = 25.0, takeoff_time = 4.0, landing_time = 4.0, dt = 0.01):
    circle_time = total_time - takeoff_time - landing_time
    if circle_time <= 0: 
        raise ValueError("Total time is too short.")
    
    times = np.arange(0, total_time, dt)
    pos_ref = np.zeros((len(times), 3))
    vel_ref = np.zeros((len(times), 3))
    accel_ref = np.zeros((len(times), 3))
    omega = 2 * np.pi / circle_time

    for i, t in enumerate(times):
        if t < takeoff_time:
            z, vz = t / takeoff_time, 1 / takeoff_time
            pos_ref[i], vel_ref[i] = [1.0, 0.0, z], [0.0, 0.0, vz]     
        elif t < takeoff_time + circle_time:
            t_circle = t - takeoff_time
            x = np.cos(omega * t_circle)
            y = np.sin(omega * t_circle)
            vx = -omega * np.sin(omega * t_circle)
            vy = omega * np.cos(omega * t_circle)
            ax = -omega**2 * np.cos(omega * t_circle)
            ay = -omega**2 * np.sin(omega * t_circle)
            pos_ref[i] = [x, y, 1.0]
            vel_ref[i] = [vx, vy, 0.0]
            accel_ref[i] = [ax, ay, 0.0]
        else:
            t_land = t - (takeoff_time + circle_time)
            z = 1.0 - (t_land / landing_time)
            vz = -1.0 / landing_time if z > 0 else 0.0
            pos_ref[i] = [1.0, 0.0, max(0, z)]
            vel_ref[i] = [0.0, 0.0, vz]

    interp_pos = interp1d(times, pos_ref, axis=0, bounds_error=False, fill_value=(pos_ref[0], pos_ref[-1]))
    interp_vel = interp1d(times, vel_ref, axis=0, bounds_error=False, fill_value=(vel_ref[0], vel_ref[-1]))
    interp_accel = interp1d(times, accel_ref, axis=0, bounds_error=False, fill_value=(accel_ref[0], accel_ref[-1]))
    
    return {
        'time': times, 'pos': pos_ref, 'vel': vel_ref, 'accel': accel_ref,
        'interp_pos': interp_pos, 'interp_vel': interp_vel, 'interp_accel': interp_accel
    }

# RECTANGLE TRAJECTORY

def generate_rectangle_trajectory(time_per_side = 5.0, takeoff_time = 4.0, landing_time = 4.0, hover_time = 1.0, dt = 0.01):
    vertices = [np.array([2,2,1]), 
                np.array([2,1,1]), 
                np.array([1,1,1]), 
                np.array([1,2,1]), 
                np.array([2,2,1])
                ]
    start_land_pos = np.array([2.0, 2.0, 0.0])
    total_time = takeoff_time + len(vertices) * hover_time + (len(vertices) - 1) * time_per_side + landing_time
    
    times = np.arange(0, total_time, dt)
    pos_ref = np.zeros((len(times), 3))
    vel_ref = np.zeros((len(times), 3))
    accel_ref = np.zeros((len(times), 3))

    current_time = 0.0
    takeoff_end_time = current_time + takeoff_time
    idx = (times >= current_time) & (times < takeoff_end_time); t_phase = times[idx] - current_time
    pos_ref[idx] = start_land_pos + (vertices[0] - start_land_pos) * (t_phase / takeoff_time)[:, np.newaxis]
    vel_ref[idx] = (vertices[0] - start_land_pos) / takeoff_time
    
    current_time = takeoff_end_time
    for i in range(len(vertices) - 1):
        hover_end_time = current_time + hover_time
        idx = (times >= current_time) & (times < hover_end_time); pos_ref[idx] = vertices[i]
        current_time = hover_end_time
        move_end_time = current_time + time_per_side
        idx = (times >= current_time) & (times < move_end_time); t_phase = times[idx] - current_time
        direction = vertices[i+1] - vertices[i]
        pos_ref[idx] = vertices[i] + direction * (t_phase / time_per_side)[:, np.newaxis]
        vel_ref[idx] = direction / time_per_side
        current_time = move_end_time
    hover_end_time = current_time + hover_time
    idx = (times >= current_time) & (times < hover_end_time); pos_ref[idx] = vertices[-1]
    
    current_time = hover_end_time
    landing_end_time = current_time + landing_time
    idx = (times >= current_time) & (times < landing_end_time); t_phase = times[idx] - current_time
    pos_ref[idx] = vertices[-1] + (start_land_pos - vertices[-1]) * (t_phase / landing_time)[:, np.newaxis]
    vel_ref[idx] = (start_land_pos - vertices[-1]) / landing_time
    idx = times >= landing_end_time; pos_ref[idx] = start_land_pos
    
    interp_pos = interp1d(times, pos_ref, axis=0, bounds_error=False, fill_value=(pos_ref[0], pos_ref[-1]))
    interp_vel = interp1d(times, vel_ref, axis=0, bounds_error=False, fill_value=(vel_ref[0], vel_ref[-1]))
    interp_accel = interp1d(times, accel_ref, axis=0, bounds_error=False, fill_value=(accel_ref[0], accel_ref[-1]))
    
    return {
        'time': times, 'pos': pos_ref, 'vel': vel_ref, 'accel': accel_ref,
        'interp_pos': interp_pos, 'interp_vel': interp_vel, 'interp_accel': interp_accel
    }

