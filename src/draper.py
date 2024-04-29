import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
from auvlib.bathy_maps import mesh_map, base_draper
from auvlib.data_tools import csv_data
from argparse import ArgumentParser
from tqdm import tqdm
import pyproj


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


def generate_groundtruth_for_file(data_path, draper, crs_code=32632, interp=False):
    """
    Generate groundtruth data using a BaseDraper object and the AUV trajectory in
    the given data file.
    """
    # set up the proj for meridian convergence
    crs = pyproj.CRS.from_epsg(int(crs_code))
    proj = pyproj.Proj(crs)

    #T_mbes_RX = np.array([3.52, -0.003, -0.33])
    #T_mbes_RX = np.array([3.809, -0.004, -0.361])
    T_mbes_RX = np.array([3.809, -0.004, -0.361])
    #R_mbes_full = R.from_euler("zyx", degrees=True, angles=[0.1, 0.4, 0.149])
    R_mbes_full = R.from_euler("zyx", degrees=True, angles=[0.1, 0, 0])

    data = np.load(data_path, allow_pickle=True)
    raw_pings = np.stack([data["X"], data["Y"], data["Z_relative"]], axis=-1)
    draping_results = np.zeros_like(raw_pings)
    if interp:
        draping_results = np.zeros((raw_pings.shape[0], 800, 3))
    # Transform AUV angles from NED to ENU (flip the angles)
    raw_angles = np.deg2rad(data['angle'][:, ::-1])

    # Transform AUV position from NED to ENU
    auv_pos = np.stack(
        [data["easting"][:, 0], data["northing"][:, 0], -data["depth"][:, 0]], axis=-1
    )

    # Transform AUV yaw from NED to ENU
    auv_yaw = np.zeros_like(auv_pos)
    auv_yaw[:, 0] = 90 - data["heading"][:, 0]
    #auv_yaw[:, 1] = -data["pitch"][:, 0]
    #auv_yaw[:, 2] = data["roll"][:, 0]

    # Parse latitude and longitude, needed for meridian convergence
    auv_latlon = np.stack([data["lat"][:, 0], data["long"][:, 0]], axis=-1)

    for idx, pos in tqdm(enumerate(auv_pos), total=auv_pos.shape[0]):
        # apply meridian convergence to the yaw angle
        auv_yaw_convergence = auv_yaw[idx].copy()
        lat, lon = auv_latlon[idx]
        conv = proj.get_factors(lon, lat).meridian_convergence
        auv_yaw_convergence[0] += conv

        R_auv = R.from_euler("zyx", degrees=True, angles=auv_yaw_convergence)
        pos = pos + R_auv.apply(T_mbes_RX)
        rotation = R_auv * R_mbes_full

        raw_angles_idx = raw_angles[idx]

        if interp:
            x_axis = np.arange(400)
            x_double = np.arange(0, 400, .5)
            raw_angles_idx = np.interp(x_double, x_axis, raw_angles_idx)

        draped_ping, draping_idx = draper.project_mbes_with_hits_idx_given_angles(
            pos, rotation.as_matrix(), raw_angles_idx,
        )
        valid_idx = np.where(draping_idx.flatten())

        # compute Z relative to the auv position
        draped_ping[:, -1] = pos[-1] - draped_ping[:, -1]

        # apply the transformation to the draping results (constant offsets)
        draped_ping[:, -1] -= T_mbes_RX[-1]

        draping_results[idx, valid_idx] = draped_ping
    print(f'% points without draping hits: {compute_percentage_points_without_draping_hits(draping_results):.2f}')
    return draping_results

def compute_percentage_points_without_draping_hits(draping_results):
    """
    Compute the percentage of points without draping hits in the draping results.
    """
    num_points = np.prod(draping_results.shape[:2])
    num_points_without_draping_hits = np.sum(np.all(draping_results == 0, axis=-1))
    return num_points_without_draping_hits / num_points * 100

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--mesh_path', type=str, required=True)
    parser.add_argument('--svp_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--resolution', type=float, required=True, help='mesh resolution in meters')
    parser.add_argument('--crs_code', type=int, default=32632, help='CRS code for meridian convergence')
    parser.add_argument('--interp', action='store_true', help='Interpolate the angles double the angular resolution')
    parser.add_argument('--suffix', type=str, default='')
    return parser.parse_args()

def main():
    args = parse_args()
    parent_path = os.path.dirname(args.data_path)
    data_filename = os.path.basename(args.data_path).split(".")[0]
    output_path = os.path.join(parent_path, f"{data_filename}_draping_gt_{args.resolution}m_{args.suffix}.npy")

    draper = get_draper(args.mesh_path, args.svp_path, args.resolution)
    draping_results = generate_groundtruth_for_file(args.data_path, draper,
                                                    crs_code=args.crs_code,
                                                    interp=args.interp)
    np.save(output_path, draping_results)

if __name__ == '__main__':
    main()