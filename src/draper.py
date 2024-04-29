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

    # T_mbes_RX = np.array([3.52, -0.003, -0.33])
    # T_mbes_RX = np.array([3.809, -0.004, -0.361])
    T_mbes_RX = np.array([3.809, -0.004, -0.361])
    # R_mbes_full = R.from_euler("zyx", degrees=True, angles=[0.1, 0.4, 0.149])
    R_mbes_full = R.from_euler("zyx", degrees=True, angles=[0.1, 0, 0])

    data = np.load(data_path, allow_pickle=True)
    raw_pings = np.stack([data["X"], data["Y"], data["Z_relative"]], axis=-1)
    draping_results = np.zeros_like(raw_pings)
    if interp:
        draping_results = np.zeros((raw_pings.shape[0], 800, 3))
    # Transform AUV angles from NED to ENU (flip the angles)
    raw_angles = np.deg2rad(data["angle"][:, ::-1])

    # Transform AUV position from NED to ENU
    auv_pos = np.stack(
        [data["easting"][:, 0], data["northing"][:, 0], -data["depth"][:, 0]], axis=-1
    )

    # Transform AUV yaw from NED to ENU
    auv_yaw = np.zeros_like(auv_pos)
    auv_yaw[:, 0] = 90 - data["heading"][:, 0]
    # auv_yaw[:, 1] = -data["pitch"][:, 0]
    # auv_yaw[:, 2] = data["roll"][:, 0]

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
            x_double = np.arange(0, 400, 0.5)
            raw_angles_idx = np.interp(x_double, x_axis, raw_angles_idx)

        draped_ping, draping_idx = draper.project_mbes_with_hits_idx_given_angles(
            pos,
            rotation.as_matrix(),
            raw_angles_idx,
        )
        valid_idx = np.where(draping_idx.flatten())

        # compute Z relative to the auv position
        draped_ping[:, -1] = pos[-1] - draped_ping[:, -1]

        # apply the transformation to the draping results (constant offsets)
        draped_ping[:, -1] -= T_mbes_RX[-1]

        draping_results[idx, valid_idx] = draped_ping
    print(
        f"% points without draping hits: {compute_percentage_points_without_draping_hits(draping_results):.2f}"
    )
    return draping_results


def compute_percentage_points_without_draping_hits(draping_results):
    """
    Compute the percentage of points without draping hits in the draping results.
    """
    num_points = np.prod(draping_results.shape[:2])
    num_points_without_draping_hits = np.sum(np.all(draping_results == 0, axis=-1))
    return num_points_without_draping_hits / num_points * 100


def drape_one_file(
    data_path, mesh_path, svp_path, out_path, resolution, crs_code, interp=False
):
    draper = get_draper(mesh_path, svp_path, resolution)
    draping_results = generate_groundtruth_for_file(data_path, draper, crs_code, interp)
    np.save(out_path, draping_results)


def construct_patches(draping_res_folder, pings_per_patch, beams_per_patch):
    """
    Construct patches from the draping results in the given folder.
    """
    all_data = []
    draping_files = sorted(
        [x for x in os.listdir(draping_res_folder) if x.endswith(".npy")],
        key=lambda x: int(x.split("-")[1]),
    )
    for draping_file in tqdm(draping_files):
        draping_path = os.path.join(draping_res_folder, draping_file)
        draping_results = np.load(draping_path, allow_pickle=True)
        all_data.append(draping_results)
    all_data = np.concatenate(all_data, axis=0)

    patch_folder = os.path.join(
        draping_res_folder, f"patches_{pings_per_patch}pings_{beams_per_patch}beams"
    )
    os.makedirs(patch_folder, exist_ok=True)
    num_patches = 0
    num_pings, num_beams, _ = all_data.shape
    for i in tqdm(range(0, num_pings, pings_per_patch)):
        for j in range(0, num_beams, beams_per_patch):
            patch = all_data[i : i + pings_per_patch, j : j + beams_per_patch]
            valid_mask = ~np.ma.masked_less_equal(patch[:, :, 2], 0).mask
            np.savez(
                os.path.join(patch_folder, f"patch_{num_patches}.npz"),
                data=patch,
                valid_mask=valid_mask,
            )
            num_patches += 1
    print(f"Created {num_patches} patches.")
    return num_patches


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_folder",
        type=str,
        required=False,
        help="Folder containing the mesh and data files",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        required=False,
        help="Path to the mesh file (if data_folder is not provided)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        help="Path to the data file (if data_folder is not provided)",
    )
    parser.add_argument("--svp_path", type=str, required=True)
    parser.add_argument(
        "--resolution", type=float, required=True, help="mesh resolution in meters"
    )
    parser.add_argument(
        "--crs_code", type=int, default=32632, help="CRS code for meridian convergence"
    )
    parser.add_argument(
        "--interp",
        action="store_true",
        help="Interpolate the angles double the angular resolution",
    )
    parser.add_argument("--suffix", type=str, default="")

    # args for creating patches
    parser.add_argument("--create_patches", action="store_true")
    parser.add_argument("--pings_per_patch", type=int, default=32)
    parser.add_argument("--beams_per_patch", type=int, default=400)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.data_folder:
        mesh_folder = os.path.join(args.data_folder, "mesh")
        data_folder = os.path.join(args.data_folder, "merged")
        draping_res_folder = os.path.join(
            args.data_folder, f"draping_{args.resolution}m_{args.suffix}"
        )
        os.makedirs(draping_res_folder, exist_ok=True)

        for data_file in sorted(
            os.listdir(mesh_folder), key=lambda x: int(x.split("-")[1])
        ):
            print(f"\nProcessing {data_file}...")
            data_path = os.path.join(data_folder, f"{data_file}.npz")
            mesh_path = os.path.join(mesh_folder, data_file)
            out_path = os.path.join(draping_res_folder, data_file)
            drape_one_file(
                data_path,
                mesh_path,
                args.svp_path,
                out_path,
                args.resolution,
                args.crs_code,
                args.interp,
            )

        if args.create_patches:
            construct_patches(
                draping_res_folder, args.pings_per_patch, args.beams_per_patch
            )
    else:
        parent_path = os.path.dirname(args.data_path)
        data_filename = os.path.basename(args.data_path).split(".")[0]
        output_path = os.path.join(
            parent_path,
            f"{data_filename}_draping_gt_{args.resolution}m_{args.suffix}.npy",
        )
        drape_one_file(
            args.data_path,
            args.mesh_path,
            args.svp_path,
            output_path,
            args.resolution,
            args.crs_code,
            args.interp,
        )


if __name__ == "__main__":
    main()
