import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
from auvlib.bathy_maps import mesh_map, base_draper
from auvlib.data_tools import csv_data
from argparse import ArgumentParser
from tqdm import tqdm


def get_draper(mesh_path, svp_path, resolution):
    """
    Returns a BaseDraper object from a mesh file and a sound velocity profile file.
    """
    mesh_data = np.loadtxt(mesh_path, delimiter=" ")
    # Negate depth since AuvLib used ENU convention
    mesh_data[:, 2] = -mesh_data[:, 2]

    V, F, bounds = mesh_map.mesh_from_cloud(mesh_data, resolution)
    svp = csv_data.csv_asvp_sound_speed.parse_file(svp_path)

    draper = base_draper.BaseDraper(V, F, bounds, svp)
    return draper


def generate_groundtruth_for_file(data_path, draper, num_beams=400, mbes_angle=120):
    """
    Generate groundtruth data using a BaseDraper object and the AUV trajectory in
    the given data file.
    """
    T_mbes_RX = np.array([3.52, -0.003, -0.33])
    R_mbes_yaw = R.from_euler("zyx", degrees=True, angles=[0.1, 0, 0])
    mbes_angle = np.deg2rad(mbes_angle)

    data = np.load(data_path, allow_pickle=True)
    raw_pings = np.stack([data["X"], data["Y"], data["Z_relative"]], axis=-1)
    draping_results = np.zeros_like(raw_pings)
    # Transform AUV position from NED to ENU
    auv_pos = np.stack(
        [data["easting"][:, 0], data["northing"][:, 0], -data["depth"][:, 0]], axis=-1
    )
    # Transform AUV yaw from NED to ENU
    auv_yaw = np.zeros_like(auv_pos)
    auv_yaw[:, 0] = 90 - data["heading"][:, 0]

    for idx, pos in tqdm(enumerate(auv_pos), total=auv_pos.shape[0]):
        R_auv = R.from_euler("zyx", degrees=True, angles=auv_yaw[idx])
        pos = pos + R_auv.apply(T_mbes_RX)
        rotation = R_auv * R_mbes_yaw

        draped_ping, draping_idx = draper.project_mbes_with_hits_idx(
            pos, rotation.as_matrix(), num_beams, mbes_angle
        )
        valid_idx = np.where(draping_idx.flatten())

        # compute Z relative to the auv position
        draped_ping[:, -1] = pos[-1] - draped_ping[:, -1]
        draping_results[idx, valid_idx] = draped_ping
    return draping_results

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--mesh_path', type=str, required=True)
    parser.add_argument('--svp_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--resolution', type=float, required=True) #meters
    parser.add_argument('--num_beams', type=int, default=400)
    parser.add_argument('--mbes_angle', type=int, default=120)
    return parser.parse_args()

def main():
    args = parse_args()
    parent_path = os.path.dirname(args.data_path)
    data_filename = os.path.basename(args.data_path).split(".")[0]
    output_path = os.path.join(parent_path, f"{data_filename}_draping_gt_{args.resolution}m.npy")

    draper = get_draper(args.mesh_path, args.svp_path, args.resolution)
    draping_results = generate_groundtruth_for_file(args.data_path, draper, args.num_beams, args.mbes_angle)
    np.save(output_path, draping_results)

if __name__ == '__main__':
    main()