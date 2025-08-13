import os
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2

from h5_utils import write_element_to_hptr_h5_file
from waymo_utils import load_waymo, get_ego_img, get_ego_states
from view import plot_scenario

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "val", "test"]
    )

    parser.add_argument(
        "--input",
        type=str,
        default="/dir/to/waymo_e2e/",
        help="The root directory of the Waymo.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/dir/to/output/path/",
        help="Within this directory the files training.h5, validation.h5, and testing.h5 are generated.",
    )

    parser.add_argument(
    "--show",
    action="store_true",
    help="Show plots when enabled."
    )

    args = parser.parse_args()
    mode = args.mode
    in_path = args.input
    out_path = args.output
    show = args.show

    raw_train, raw_val, raw_test = load_waymo(in_path)

    if mode == "train":
        out_file = out_path + "training.h5"
        filenames = tf.io.matching_files(raw_train)

    elif mode == "val":
        out_file = out_path + "validation.h5"
        filenames = tf.io.matching_files(raw_val)

    elif mode == "test":
        out_file = out_path + "testing.h5"
        filenames = tf.io.matching_files(raw_test)

    dataset = tf.data.TFRecordDataset(filenames, compression_type='')
    dataset_iter = dataset.as_numpy_iterator()

    num_samples = 0

    for bytes, filename in tqdm(zip(dataset_iter, filenames), desc = f"Processing {mode} split", total = len(filenames)):
        data = wod_e2ed_pb2.E2EDFrame()
        data.ParseFromString(bytes)

        metadata = {}
        metadata["scenario_id"] = os.path.basename(filename.numpy().decode('utf-8'))

        e2ed_data = {}

        get_ego_img(data, e2ed_data)
        get_ego_states(data, e2ed_data, mode)

        if show:
            plot_scenario(e2ed_data)

        write_element_to_hptr_h5_file(
            out_file, str(num_samples), e2ed_data, metadata
        )
        
        num_samples += 1