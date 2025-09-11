import os
import yaml
import math
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
from typing import Tuple, List, Dict

class MRTE2E():
    def __init__(
            self,
            root: str,
            rec_freq: int,
            out_freq: int,
            hist_time: float,
            fut_time: float,
            expected_img_shape: Tuple[int, int, int],
            expected_pc_shape: Tuple[int, int]
    ):
        self.root = root

        self.rec_freq = rec_freq
        self.out_freq = out_freq
        self.hist_time = hist_time
        self.fut_time = fut_time

        self.expected_img_shape = expected_img_shape
        self.expected_pc_shape = expected_pc_shape

        self.current_t = int(self.hist_time * self.out_freq)

        self.total_time = hist_time + fut_time

        self.base_sub_folders = [
            "camera_front",
            "camera_front_right",
            "camera_front_left",
            "camera_back",
            "camera_back_right",
            "camera_back_left",
            "point_clouds",
            "trajectory",
        ]

        self.inter_sub_folders = [
            "annotations"
        ]

        self.rec_folder_base = "mrt_e2e_rec_"

    def check_recordings(
            self,
            root: str,
    ) -> Tuple[bool, List]:
        """
        Checks if all recordings have expected subfolders
        
        Args:
            root: path/to/mrt/dataset
        
        Returns:
            OK: True if data is correct
            recordings: sorted folder names
        """

        try:
            entries = os.listdir(root)
        except OSError as e:
            print(f"ERROR: cannot list '{root}': {e}")
            return False, []
        
        recordings = []
        for rec in entries:
            path = os.path.join(root, rec)

            if os.path.isdir(path) and rec.startswith(self.rec_folder_base):
                suffix = rec[len(self.rec_folder_base):]
                if suffix.isdigit():
                    recordings.append((int(suffix), rec))

        if not recordings:
            print(f"ERROR: No recordings have been found at: {root}")
            return False, []

        recordings.sort(key=lambda t:t[0])

        # check recordings contain subfolders
        for idx, rec in recordings:

            rec_path = os.path.join(root, rec)
            missing = [sub for sub in self.base_sub_folders
                       if not os.path.isdir(os.path.join(rec_path, sub))]

            if missing:
                for miss in missing:
                    print(f"ERROR: missing {miss} in {rec_path}")
                return False, []

        return True, recordings
    
    def load_scenario_annotations(
            self,
            root: str,
            name: str
        ) -> List:
        """
        For each recording, loads scenario starting points and annotated intent.
        """
        path = os.path.join(root,name,"scenarios.json")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

            if name != data["name"]:
                raise ValueError(f"Recording folder name ({name}) does not match ({data['name']})")

            if self.out_freq != int(data["frequency"]):
                raise ValueError(f"File frequency {data['frequency']} Hz does not match selected frequency {self.out_freq} Hz")
        
            if self.hist_time != int(data["history_duration"]):
                raise ValueError(f"File history duration {data['history_duration']} s does not match selected duration {self.hist_time} s")
            
            if self.fut_time != int(data["future_duration"]):
                raise ValueError(f"File future duration {data['future_duration']} s does not match selected duration {self.fut_time} s")

            return data['scenarios']
    
    def load(self) -> List:

        """
        Loads dataset recordings file names into a list

        Returns:
            recordings: List(str)
        """

        # Check dataset folder structure
        success, recordings = self.check_recordings(self.root)

        if not success:
            raise ValueError(f"No valid data was found at {self.root}.")

        print(f"\nFound {len(recordings)} recording{'s'*(len(recordings)>1)} from MRT-E2E")

        # Load sensor configuration
        self.sensor_config = self.load_config(self.root)

        # Prepare Front camera and LiDAR calibration
        self.raw = self.sensor_config["raw"]
        self.sensor_lidar_box_top_center = self.sensor_config["sensor_lidar_box_top_center"]
        self.sensor_camera_box_ring_front = self.sensor_config["sensor_camera_box_ring_front"]
        self.sensor_camera_box_ring_front["inverse_extrinsics"] = np.linalg.inv(np.array(self.sensor_camera_box_ring_front["extrinsics"]))
        self.sensor_lidar_box_top_center["inverse_extrinsics"] = np.linalg.inv(np.array(self.sensor_lidar_box_top_center["extrinsics"]))

        return recordings
    
    def load_config(
            self, 
            path: str, 
            filename: str = "sensor_cfg.yaml"
        ) -> dict:
        """Load YAML into dicts (tuples preserved)."""

        yaml.SafeDumper.add_representer(tuple, self.tuple_representer)
        yaml.SafeLoader.add_constructor(u'!tuple', self.tuple_constructor)

        with open(os.path.join(path,filename), "r") as f:
            return yaml.safe_load(f)
        
    def tuple_representer(self, dumper, data):
        """
        YAML utils for tuples
        """
        return dumper.represent_sequence(u'!tuple', data)

    def tuple_constructor(self, loader, node):
        """
        YAML utils for tuples
        """
        return tuple(loader.construct_sequence(node))

    def get_idx(self) -> np.array:

        """
        Gets RELATIVE indices for timestep selection based on scenario duration and frequency
        """

        t_target = np.arange(0, int(self.total_time), 1 / self.out_freq)
        
        return np.round(t_target * self.rec_freq).astype(int)
    
    def process_traj(
            self,
            path: str,
            mode: str,
            e2ed_data: Dict,
            time_step: int,
            idx: np.array,
            like_waymo: bool = True
        )-> None:
        """
        Loads trajectory data into the e2ed_data dictionary
        """

        # get traj from file
        traj = self.load_traj(path, sub_folder= self.base_sub_folders[-1])

        # select scenario starting time step
        traj = self.slice_traj(traj, time_step)

        # get traj and valid mask at out freq
        traj, valid = self.down_sample_traj(
            traj, 
            idx, 
            self.rec_freq, 
            self.out_freq, 
            self.total_time
            )

        # convert to agent centric
        pos = self.to_agent_centric_traj(
            traj,
            valid,
            int(self.hist_time),
            self.out_freq
            )
        
        # calculates velocities
        vel = self.derivative(
            pos,
            valid,
            self.out_freq
        )

        # calculates accelerations
        acc = self.derivative(
            vel,
            valid,
            self.out_freq
        )

        
        if like_waymo:
            if mode != "test":
                e2ed_data["agent/pos"] = pos[:,:2]
                e2ed_data["agent/valid"] = valid

                e2ed_data["gt/pos"] = pos[self.current_t:,:2]

            e2ed_data["history/agent/pos"] = pos[:self.current_t,:2]
            e2ed_data["history/agent/vel"] = vel[:self.current_t,:2]
            e2ed_data["histoy/agent/acc"] = acc[: self.current_t,:2]
            e2ed_data["history/agent/valid"] = valid[:self.current_t]
        
        else:
            if mode != "test":
                e2ed_data["agent/pos"] = pos[:,:]
                e2ed_data["agent/vel"] = vel[:,:]
                e2ed_data["agent/acc"] = acc[:,:]
                e2ed_data["agent/valid"] = valid

                e2ed_data["gt/pos"] = pos[self.current_t:,:]

            e2ed_data["history/agent/pos"] = pos[:self.current_t,:]
            e2ed_data["history/agent/vel"] = vel[:self.current_t,:]
            e2ed_data["histoy/agent/acc"] = acc[: self.current_t,:]
            e2ed_data["history/agent/valid"] = valid[:self.current_t]

    def process_sensors(
            self,
            path: str,
            mode: str,
            e2ed_data: Dict,
            time_step: int,
            idx: np.array,
            like_waymo: bool = True,
            img_ext: str = ".jpg",
            point_cloud_ext: str = ".pcd"
        )-> None:
        """
        Loads sensor data into the e2ed_data dictionary
        """

        # CAMERAS
        for cam in self.base_sub_folders[:-2]:
            video_buffer = []
            video_valid_buffer = []
            for i in idx:

                file_name = f"{i+time_step:010d}{img_ext}"

                try:
                    img = self.load_img(
                        path,
                        file_name,
                        cam
                    )
                    valid_img = np.ones(1)
                except:
                    img = np.zeros(self.expected_img_shape)
                    valid_img = np.zeros(1)

                if i == idx[self.current_t-1]:
                    e2ed_data[f"agent/{cam.removeprefix('camera_')}/img"] = img
                    e2ed_data[f"agent/{cam.removeprefix('camera_')}/valid_img"] = valid_img

                elif not like_waymo and i <= idx[self.current_t-1]:
                    video_buffer.append(img)
                    video_valid_buffer.append(valid_img)

            if not like_waymo:
                e2ed_data[f"agent/{cam.removeprefix('camera_')}/video"] = np.stack(video_buffer, axis=0)
                e2ed_data[f"agent/{cam.removeprefix('camera_')}/valid_video"] = np.stack(video_buffer, axis=0)


        # POINTCLOUD
        if not like_waymo:
            pc_buffer = []
            for i in idx:
                file_name = f"{i+time_step:010d}{point_cloud_ext}"

                try:
                    pc = self.load_pointcloud(
                        path,
                        file_name
                    )

                except:
                    pc = np.zeros(self.expected_pc_shape)

                pc_buffer.append(pc)

            e2ed_data[f"agent/point_clouds"] = np.stack(pc_buffer, axis=0)

        # DEPTH IMAGES
        if not like_waymo:

            pc = e2ed_data[f"agent/point_clouds"][self.current_t-1]

            lidar_extrinsics = np.array(self.sensor_lidar_box_top_center["extrinsics"])
            camera_extrinsics = np.array(self.sensor_camera_box_ring_front["extrinsics"])

            projected_points, valid_points = self.project_lidar_to_camera_raw(
                points= pc,
                lidar_extrinsics= lidar_extrinsics,
                camera_extrinsics= camera_extrinsics,
                camera_intrinsics= self.raw["intrinsics"],
                image_size = self.raw["image_size"]
            )

            e2ed_data["agent/front/depth_img"] = projected_points
            e2ed_data["agent/front/depth_valid"] = valid_points

    def process_ann(
            self,
            path: str,
            idx: np.array,
            e2ed_data: Dict,
            file_name: str = "default.json",
            sub_folder: str = "annotations/",
            max_agents: int = 64,
            ) -> np.array:
        """Loads traffic agent annotations.

        Args:
            path: folder/to/parent/folder
            idx: np.array
            sub_folder: sub/folder
            file_name: filename.json

        Returns:
            pos:   (N, T, 3)
            yaw:   (N, T,)
            size:  (N, T, 3)
            valid: (N, T,)
            type:  (N,)
        """

        with open(os.path.join(path, sub_folder, file_name), "r", encoding="utf-8") as file:
            data = json.load(file)


            ann_pos = {}
            ann_yaw = {}
            ann_size = {}
            ann_valid = {}
            ann_type = {}

            for cnt, i in enumerate(idx):
                not_updated = {k: True for k in ann_pos}
                for frame in data['items']:
                    if i == int(frame['id']):
                        
                        for ann in frame['annotations']:

                            track_id = ann['attributes']['track_id']

                            if track_id not in ann_pos and len(ann_pos) < max_agents:
                                ann_pos[track_id] = [[0.0, 0.0, 0.0] for _ in range(cnt)] + [ann['position']]
                                ann_yaw[track_id] = [0.0 for _ in range(cnt)] + [ann['rotation'][2]]
                                ann_size[track_id] = [[0.0, 0.0, 0.0] for _ in range(cnt)] + [ann['scale']]
                                ann_type[track_id] = ann['label_id']
                                ann_valid[track_id] = [0.0 for _ in range(cnt)] + [1]

                            else:
                                ann_pos[track_id].append(ann['position'])
                                ann_yaw[track_id].append(ann['rotation'][2])
                                ann_size[track_id].append(ann['scale'])
                                ann_valid[track_id].append(1)

                                not_updated[track_id] = False
                        break

                for pad_idx in not_updated:

                    if not_updated[pad_idx]:

                        ann_pos[pad_idx].append([0.0, 0.0, 0.0])
                        ann_yaw[pad_idx].append(0.0)
                        ann_size[pad_idx].append([0.0, 0.0, 0.0])
                        ann_valid[pad_idx].append(0)

                        not_updated[pad_idx] = False
        
            ann_pos = np.array(list(ann_pos.values()))
            ann_yaw = np.array(list(ann_yaw.values()))
            ann_size = np.array(list(ann_size.values()))
            ann_valid = np.array(list(ann_valid.values()))
            ann_type = np.array(list(ann_type.values()))
            
                                
            pad_size = int(max_agents - ann_pos.shape[0])

            if pad_size > 0:

                ann_pos = np.concatenate(
                    [
                        ann_pos,
                        np.zeros((pad_size, idx.shape[0], 3), dtype=float)
                    ],
                    axis=0
                )

                ann_yaw = np.concatenate(
                    [
                        ann_yaw,
                        np.zeros((pad_size, idx.shape[0]), dtype=float)
                    ],
                    axis=0
                )

                ann_size = np.concatenate(
                    [
                        ann_size,
                        np.zeros((pad_size, idx.shape[0], 3), dtype=float)
                    ],
                    axis=0
                )

                ann_valid = np.concatenate(
                    [
                        ann_valid,
                        np.zeros((pad_size, idx.shape[0]), dtype=float)
                    ],
                    axis=0
                )

                ann_type = np.concatenate(
                    [
                        ann_type,
                        np.zeros((pad_size), dtype=float)
                    ],
                    axis=0
                )

            e2ed_data["other/pos"] = ann_pos
            e2ed_data["other/yaw"] = ann_yaw
            e2ed_data["other/size"] = ann_size
            e2ed_data["other/valid"] = ann_valid
            e2ed_data["other/type"] = ann_type

    def project_lidar_to_camera_raw(
            self, 
            points: np.array, 
            lidar_extrinsics: np.array, 
            camera_extrinsics: np.array, 
            camera_intrinsics: np.array, 
            image_size: Tuple
            )-> Tuple[np.array, np.array]:
        """
        Project 3D LiDAR points to 2D camera image coordinates
        
        Args:
            points: Nx3 array of 3D points in LiDAR coordinate system
            lidar_extrinsics: 4x4 transformation matrix from LiDAR to world
            camera_extrinsics: 4x4 transformation matrix from camera to world
            camera_intrinsics: List of 15 parameters for raw camera model:
                            [cx, cy, f, k1, k2, k3, p1, p2, k4, k5, k6, s1, s2, s3, s4]
            image_size: (width, height) of the image
        
        Returns:
            projected_points: Nx2 array of 2D image coordinates
            valid_mask: boolean mask for points that are in front of camera and within image bounds
        """
        # Convert to homogeneous coordinates
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Transform from LiDAR to world coordinates
        points_world = (lidar_extrinsics @ points_homogeneous.T).T
        
        # Transform from world to camera coordinates
        camera_inverse = np.linalg.inv(camera_extrinsics)
        points_camera = (camera_inverse @ points_world.T).T
        
        # Extract 3D coordinates
        X, Y, Z = points_camera[:, 0], points_camera[:, 1], points_camera[:, 2]
        
        # Only keep points in front of the camera (positive Z)
        front_mask = Z > 0
        
        # Project to image coordinates using raw camera model (CAMERA_MODEL_NON_SVP)
        # Extract parameters from intrinsics
        cx = camera_intrinsics[0]  # Principal point x
        cy = camera_intrinsics[1]  # Principal point y
        f = camera_intrinsics[2]   # Focal length
        
        # Distortion coefficients
        k1, k2, k3 = camera_intrinsics[3], camera_intrinsics[4], camera_intrinsics[5]
        p1, p2 = camera_intrinsics[6], camera_intrinsics[7]
        k4, k5, k6 = camera_intrinsics[8], camera_intrinsics[9], camera_intrinsics[10]
        s1, s2, s3, s4 = camera_intrinsics[11], camera_intrinsics[12], camera_intrinsics[13], camera_intrinsics[14]
        
        # Normalize coordinates
        x_norm = X / Z
        y_norm = Y / Z
        
        # Calculate r^2
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r2 * r4
        
        # Radial distortion
        radial_distortion = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + k4*r2 + k5*r4 + k6*r6)
        
        # Tangential distortion
        tangential_x = 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
        tangential_y = p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm
        
        # Apply distortions
        x_distorted = x_norm * radial_distortion + tangential_x
        y_distorted = y_norm * radial_distortion + tangential_y
        
        # Add thin prism distortion (s1, s2, s3, s4 parameters)
        x_distorted += s1*r2 + s2*r4
        y_distorted += s3*r2 + s4*r4
        
        # Project to image coordinates
        u = f * x_distorted + cx
        v = f * y_distorted + cy
        
        # Stack u, v coordinates
        projected_points = np.column_stack([u, v])
        
        # Check which points are within image boundaries
        width, height = image_size
        bounds_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        
        # Combine masks
        valid_mask = front_mask & bounds_mask

        projected_points = projected_points[:, :, None] * Z[:, None, None]
        
        return projected_points, valid_mask
    
    def load_traj(
            self,
            path: str,
            file_name: str = "trajectory.npy",
            sub_folder: str = "trajectory/"
            ) -> np.array:
        """
        Loads ego-vehicle trajectory from numpy file.

        Args:
            path: folder/to/parent/folder
            sub_folder: sub/folder
            file_name: filename.npy

        Returns:
            traj: (T, 3)
        """

        return np.load(os.path.join(path,sub_folder, file_name))

    def slice_traj(
            self,
            traj: np.array,
            time_step: int
    ) -> np.array:
        """
        Slices a ego-vehicle trajectory.

        Args:
            traj: (T, 3)
            time_step: cutting point

        Returns:
            traj: (T - time_step, 3)
        """
        
        return traj[time_step:,...]

    def down_sample_traj(
            self,
            traj: np.array,
            idx: np.array,
            rec_freq: int = 10, # Hz
            out_freq: int = 4, # Hz
            max_time: float = 9.0 # seconds
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Slices a ego-vehicle trajectory.

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
        
        idx = np.clip(idx, 0, len(traj) - 1)

        return traj[idx], valid[idx]

    def to_agent_centric_traj(
            self,
            traj: np.array,
            valid: np.array,
            current_t: int = 4, # seconds
            freq: int = 4
    ) -> np.array:
        
        """
        Transforms a scene centric traj into a agent centric one.

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
            self,
            x : np.array,
            valid: np.array,
            freq: int = 4
    ) -> np.array:
        """
        Calculates the derivative of a tensor.

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
            self,
            vel: np.array, # agent centric
            acc: np.array, # agent centric
            valid: np.array,
            current_t: int = 4, # seconds
            freq: int = 4,
            eps_speed: float = 1e-3
    ) -> np.array:
        """
        Computes twist at current step.

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
            self,
            path: str,
            file_name: str,
            sub_folder: str = "point_clouds/default/"
    )-> np.array:
        """
        Loads point cloud from .pcd file.

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

    def load_img(
            self,
            path: str,
            file_name: str,
            sub_folder: str
    )-> np.array:
        """
        Loads image as np.array

        Args:
            path: folder/to/parent/folder
            sub_folder: sub/folder
            file_name: img.jpg

        Returns:
            img: (H, W, 3)
        """
        file_path = os.path.join(path, sub_folder, file_name)

        img = np.array(Image.open(file_path))

        return img
