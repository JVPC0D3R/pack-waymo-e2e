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
        2: "img_front_left",
        1: "img_front",
        3: "img_front_right",
        4: "img_side_left",
        5: "img_side_right",
        6: "img_rear_left",
        7: "img_rear",
        8: "img_rear_right"
    }

    for camera_name in order:
        for index, image_content in enumerate(data.frame.images):
            if image_content.name == camera_name:
                calibration = data.frame.context.camera_calibrations[index]
                image = tf.io.decode_image(image_content.image).numpy()

                e2ed_data[f"agent/{keys[camera_name]}"] = image
                break

def get_ego_states(data: wod_e2ed_pb2.E2EDFrame, e2ed_data: Dict) -> None:
    """Gets ego-vehicle past and future states.

    Args:
        data: Waymo E2EDFrame scenario
        e2ed_data: h5 dictionary

    """

    e2ed_data["agent/pos"] = np.stack(
        [np.concatenate(
        [data.past_states.pos_x, data.future_states.pos_x], axis=0
        ),
        np.concatenate(
        [data.past_states.pos_y, data.future_states.pos_y], axis=0
        )]
        , axis = 1
    ) # no z available in the past [36, 2]

    e2ed_data["history/agent/pos"] = np.stack(
        [data.past_states.pos_x, data.past_states.pos_y], axis=1
    ) # no z available [16, 2]

    e2ed_data["gt/pos"] = np.stack(
        [data.future_states.pos_x, data.future_states.pos_y, data.future_states.pos_z], axis=1
    ) # [20, 3]

    e2ed_data["agent/vel"] = np.stack(
        [np.concatenate(
        [data.past_states.vel_x, data.future_states.vel_x], axis=0
        ),
        np.concatenate(
        [data.past_states.vel_y, data.future_states.vel_y], axis=0
        )]
        , axis = 1
    ) # no z available in the past [36, 2]

    e2ed_data["history/agent/pos"] = np.stack(
        [data.past_states.vel_x, data.past_states.vel_y], axis=1
    ) # no z available [16, 2]