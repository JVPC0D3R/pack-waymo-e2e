import os
import numpy as np
import tensorflow as tf
from typing import Dict

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2

def load_waymo(data_path: str):
    """Projects from vehicle coordinate system to image with global shutter.

    Args:
        data_path: Waymo End2End dataset path

    Returns:
        Tuple of lists of train, validation, and test files
    """

    all_files = os.listdir(data_path)

    train = sorted(
    os.path.join(data_path, f)
    for f in all_files
    if f.startswith('training_') and '.tfrecord' in f
    )

    val = sorted(
        os.path.join(data_path, f)
        for f in all_files
        if f.startswith('val_') and '.tfrecord' in f
    )

    test = sorted(
        os.path.join(data_path, f)
        for f in all_files
        if f.startswith('test_') and '.tfrecord' in f
    )

    return train, val, test

def get_ego_img(data: wod_e2ed_pb2.E2EDFrame, e2ed_data: Dict) -> None:
    """Gets ego-vehicle view at current frame.

    Args:
        data: Waymo E2EDFrame scenario
        e2ed_data: h5 dictionary

    """

    order = [2,1,3,4,5,6,7,8]
    keys = {
        2: "front_left",
        1: "front",
        3: "front_right",
        4: "side_left",
        5: "side_right",
        6: "rear_left",
        7: "rear",
        8: "rear_right"
    }

    for camera_name in order:
        for index, image_content in enumerate(data.frame.images):
            if image_content.name == camera_name:
                calibration = data.frame.context.camera_calibrations[index]
                image = tf.io.decode_image(image_content.image).numpy()

                e2ed_data[f"agent/{keys[camera_name]}/img"] = image
                e2ed_data[f"agent/{keys[camera_name]}/calib/intr"] = np.array(calibration.intrinsic)
                e2ed_data[f"agent/{keys[camera_name]}/calib/extr"] = np.array(calibration.extrinsic.transform)
                break

def get_ego_states(
        data: wod_e2ed_pb2.E2EDFrame, 
        e2ed_data: Dict, 
        mode: str)-> None:
    """Gets ego-vehicle past and future states.

    Args:
        data: Waymo E2EDFrame scenario
        e2ed_data: h5 dictionary

    """

    # GO_STRAIGHT = 1 | GO_LEFT = 2 | GO_RIGHT = 3
    e2ed_data["agent/intent"] = np.array([data.intent,])
    
    # Current twist
    e2ed_data["agent/vel"] = np.stack(
        [
            data.frame.images[0].velocity.v_x,
            data.frame.images[0].velocity.v_y,
            data.frame.images[0].velocity.v_z,
            data.frame.images[0].velocity.w_x,
            data.frame.images[0].velocity.w_y,
            data.frame.images[0].velocity.w_z,
        ],
        axis=0,
    ) # [6] 

    # Current pose (identity matrix)
    e2ed_data["hisotry/agent/pose"] = np.array(data.frame.images[0].pose.transform)

    # Total traj (x, y)
    if mode == "train" or mode == "val":
        e2ed_data["agent/pos"] = np.stack(
            [np.concatenate(
            [data.past_states.pos_x, data.future_states.pos_x], axis=0
            ),
            np.concatenate(
            [data.past_states.pos_y, data.future_states.pos_y], axis=0
            )]
            , axis = 1
        ) # [36, 2]

    # History positions (x, y)
    e2ed_data["history/agent/pos"] = np.stack(
        [
        data.past_states.pos_x,
        data.past_states.pos_y
        ],
        axis=1
    ) # [16, 2] 4 seconds 4 Hz

    e2ed_data["history/agent/vel"] = np.stack(
        [
        data.past_states.vel_x, 
        data.past_states.vel_y
        ], 
        axis=1
    ) # no z available [16, 2]

    e2ed_data["history/agent/acc"] = np.stack(
        [data.past_states.accel_x, data.past_states.accel_y], axis=1
    ) # no z available [16, 2]

    # Rater Feedback
    if mode == "val":
        e2ed_data["gt/preference_scores"] = np.stack([
            data.preference_trajectories[0].preference_score,
            data.preference_trajectories[1].preference_score,
            data.preference_trajectories[2].preference_score
        ])
    #TODO add rater trajs

    # Future positions (x, y, z)
    if mode == "train" or mode == "val":
        e2ed_data["gt/pos"] = np.stack(
            [
            data.future_states.pos_x,
            data.future_states.pos_y, 
            data.future_states.pos_z
            ], 
            axis=1
        ) # [20, 3] 5 seconds 4 Hz