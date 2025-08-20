import os
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse
from typing import Tuple

from view.view import plot_scenario

from utils.h5_utils import write_element_to_hptr_h5_file
from utils.mrt_e2e_utils import MRTE2E 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "val", "test"]
    )

    parser.add_argument(
        "--input",
        type=str,
        default="/dir/to/mrt_e2e/",
        help="The root directory of mrt-e2e.",
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

    parser.add_argument(
    "--img_shape",
    type=int,
    nargs=3,
    metavar=("H","W","C"),
    help="Camera image shape"
    )

    parser.add_argument(
    "--pc_shape",
    type=int,
    nargs=2,
    metavar=("P","XYZ"),
    help="Point cloud shape"
    )


    args = parser.parse_args()
    mode = args.mode
    in_path = args.input
    out_path = args.output
    show = args.show

    img_shape = args.img_shape
    pc_shape = args.pc_shape

    rec_freq = 10
    out_freq = 4
    hist_time = 4.0
    fut_time = 5.0

    mrt = MRTE2E(
        root = in_path,
        rec_freq = rec_freq,
        out_freq = out_freq,
        hist_time = hist_time,
        fut_time = fut_time,
        expected_img_shape = img_shape,
        expected_pc_shape = pc_shape
    )

    recordings = mrt.load()

    pbar = tqdm(recordings, desc="Processing data...")
    for rec in pbar:
        pbar.set_postfix(file=rec[1])
        for ann in mrt.load_scenario_annotations(in_path, rec[1]):

            e2ed_data = {}

            start = int(ann["start"])
            e2ed_data["agent/intent"] = np.array([int(ann["intent"])])

            idx = mrt.get_idx()

            mrt.process_sensors(
                os.path.join(in_path, rec[1]),
                mode,
                e2ed_data,
                start,
                idx,
                like_waymo = True
                )

            mrt.process_traj(
                os.path.join(in_path, rec[1]),
                mode,
                e2ed_data,
                start,
                idx,
                like_waymo = True
                )
    
            if show:
                plot_scenario(e2ed_data)

            #TODO save h5 files

