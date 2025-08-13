import os
import math
import numpy as np
from typing import Tuple

def load_traj(
        path: str,
        file_name: str = "trajectory.npy",
        sub_folder: str = "trajectory/"
        ) -> np.array:
    """Loads ego-vehicle trajectory from numpy file.

    Args:
        path: folder/to/parent/folder
        sub_folder: sub/folder
        file_name: filename.npy

    Returns:
        traj: (T, 3)
    """

    return np.load(os.path.join(path,sub_folder, file_name))

def slice_traj(
        traj: np.array,
        time_step: int
) -> np.array:
    """Slices a ego-vehicle trajectory.

    Args:
        traj: (T, 3)
        time_step: cutting point

    Returns:
        traj: (T - time_step, 3)
    """
    
    return traj[time_step:,...]

def down_sample_traj(
        traj: np.array,
        rec_freq: int = 10, # Hz
        out_freq: int = 4, # Hz
        max_time: float = 9.0 # seconds
) -> Tuple[np.array, np.array, np.array]:
    """Slices a ego-vehicle trajectory.

    Args:
        traj: (T_rHz, 3)
        rec_freq: recording frequency
        out_freq: output frequency
        max_time: maximum scenario length

    Returns:
        traj: (T_oHz, 3)
    """

    assert rec_freq > out_freq, f"Output frequency ({out_freq} Hz) must be higher than the recording frequency ({rec_freq} Hz)!"
    
    total_time = len(traj) / rec_freq
    
    if total_time < max_time:

        pad_len = int((max_time-total_time)*rec_freq)
        valid = np.concatenate([
            np.ones(len(traj), dtype=np.int8),
            np.zeros(pad_len, dtype=np.int8)
        ])
        traj = np.pad(traj, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
    

    t_target = np.arange(0, max_time, 1 / out_freq)
    idx = np.round(t_target * rec_freq).astype(int)
    idx = np.clip(idx, 0, len(traj) - 1)

    return traj[idx], valid[idx], idx

def to_agent_centric_traj(
        traj: np.array,
        valid: np.array,
        current_t: int = 4, # seconds
        freq: int = 4
) -> np.array:
    
    """Transforms a scene centric traj into a agent centric one.

    Args:
        sc_traj: (T, 3)
        valid: valid mask (T,)
        current_t: reference center time
        freq: sample frequency

    Returns:
        ac_traj: (T, 3)
    """
    
    current_idx = (current_t * freq) -1
    assert valid[current_idx] != 0, f"Main timestep {current_idx} must be valid!"
    center = traj[current_idx]

    traj -= center # translate

    yaw = math.atan(traj[current_idx+1, 1] / traj[current_idx+1, 0])

    R = np.array([
        [ np.cos(-yaw), -np.sin(-yaw), 0],
        [ np.sin(-yaw),  np.cos(-yaw), 0],
        [ 0,               0,              1]
    ])

    traj = traj @ R.T # rotate

    return traj * valid[:, None]

def derivative(
        x : np.array,
        valid: np.array,
        freq: int = 4
) -> np.array:
    """Calculates the derivative of a tensor.

    Args:
        x: (T, 3)
        valid: valid mask (T,)
        freq: sample frequency

    Returns:
        dx: (T, 3)
    """
    
    dt = 1.0 / float(freq)
    ts = len(x)

    valid = valid.astype(bool)
    dx = np.zeros_like(x)

    for i in range(ts):
        if i>0 and i<ts-1 and valid[i-1] and valid[i+1]:
            dx[i] = (x[i+1] - x[i-1]) / (2*dt)
        elif i<ts-1 and valid[i+1]:
            dx[i] = (x[i+1] - x[i]) / dt
        elif i>0 and valid[i-1]:
            dx[i] = (x[i] - x[i-1]) / dt
    return dx

def compute_current_twist(
        vel: np.array, # agent centric
        acc: np.array, # agent centric
        valid: np.array,
        current_t: int = 4, # seconds
        freq: int = 4,
        eps_speed: float = 1e-3
) -> np.array:
    """Computes twist at current step.

    Args:
        vel: velocity (T, 3)
        acc: acceleration (T, 3)
        valid: valid mask (T,)
        current_t: reference center time
        freq: sample frequency
        eps_speed: threshold for angular vel calcualtion

    Returns:
        twist: (6,)
    """

    current_idx = (current_t * freq) -1
    assert valid[current_idx] != 0, f"Main timestep {current_idx} must be valid!"

    vx, vy, vz = vel[current_idx]
    ax, ay, az = acc[current_idx]

    v_xy = np.linalg.norm(vel[current_idx, :2], axis=-1)

    if v_xy > eps_speed:
        wz = (vx * ay - vy * ax) / (v_xy ** 2)
    else:
        wz = 0.0

    twist =  np.array(
        [
            vx,
            vy, 
            vz, 
            0.0, 
            0.0, 
            wz
        ], 
        dtype=np.float32
    )

    return twist

def load_pointcloud(
        path: str,
        file_name: str,
        sub_folder: str = "point_clouds/default/"
):
    """Loads point cloud from .pcd file.

    Args:
        path: folder/to/parent/folder
        sub_folder: sub/folder
        file_name: filename.pcd

    Returns:
        point_cloud: (P, 3)
    """
    file_path = os.path.join(path, sub_folder, file_name)

    with open(file_path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line.startswith('DATA'):
                data_format = line.split()[1].lower()
                break

        # get width, height
        width = int(next(h for h in header if h.startswith('WIDTH')).split()[1])
        height = int(next(h for h in header if h.startswith('HEIGHT')).split()[1])
        num_points = width * height

        # get fields, size, type
        fields = next(h for h in header if h.startswith('FIELDS')).split()[1:]
        sizes = list(map(int, next(h for h in header if h.startswith('SIZE')).split()[1:]))
        types = next(h for h in header if h.startswith('TYPE')).split()[1:]
        counts = list(map(int, next(h for h in header if h.startswith('COUNT')).split()[1:]))

        
        np_types = []
        for size, typ in zip(sizes, types):
            if typ == 'F':
                np_types.append(f'<f{size}')
            elif typ == 'U':
                np_types.append(f'<u{size}')
            elif typ == 'I':
                np_types.append(f'<i{size}')
            else:
                raise ValueError(f"Unsupported PCD field type: {typ}")

        dtype = np.dtype({'names': fields, 'formats': np_types})

        if data_format == 'ascii':
            points = np.loadtxt(f, dtype=dtype)
        elif data_format.startswith('binary'):
            points = np.frombuffer(f.read(), dtype=dtype, count=num_points)
        else:
            raise ValueError(f"Unsupported PCD DATA format: {data_format}")

    xyz = np.vstack([points['x'], points['y'], points['z']]).T.astype(np.float32)
    return xyz