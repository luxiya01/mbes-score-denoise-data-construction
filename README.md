# MBES Score-Denoising Ground Truth Data Generation
This repository contains code to generate ground truth for [Score-Based Multibeam Point Cloud Denoising (AUV Symposium 2024)](https://github.com/luxiya01/mbes-score-denoise) using exports from [NaviEdit](https://www.eiva.com/products/navisuite/navisuite-processing-software/naviedit-pro).

## Data Generation
### Patch Generation from NaviEdit Exported Data
To generate data patches of size (`<pings_per_patch>` x `<beams_per_patch>`), run the following:
```bash
python src/data_generation.py \
       --data_folder <path_to_data_folder> \
       --pings_per_patch <number_of_pings_per_multibeam_patch> \
       --beams_per_patch <number_of_beams_per_multibeam_patch>
```
- `--pings_per_patch`: Number of consecutive multibeam pings per patch (default: `32`)
- `--beams_per_patch`: Number of beams per multibeam ping (default: `400`)
- `--data_folder`: Path to the folder containing NaviEdit exported raw multibeam sonar data

The `data_folder` stores ASCII format data and should follow this directory structure:
```bash
<data_folder>/
├── /pings-xyz
├── /angle-quality
├── /intensity
├── /cleaned
└── /vehicle-pos
```
- `pings-xyz`: Contains raw multibeam XYZ measurements
- `angle-quality`: Contains raw angles and qualities of each multibeam measurements
- `intensity`: Contains raw intensity measurements
- `cleaned`: Contains manually cleaned multibeam data with boolean rejection masks
- `vehicle-pos`: Contains vehicle position data

### Ground Truth Point Cloud Patch Construction
To generate ground truth multibeam point cloud patches, we use the draping functionality from [AuvLib](https://github.com/nilsbore/auvlib):
```
python src/draper.py
       --data_folder <folder name> \
       --data_path <path to data folder containing processed raw multibeam patches> \
       --mesh_path <path to mesh in .xyz format> \
       --resolution <resolution in meters> \
       --suffix <suffix for the output folder> \
       --svp_path <path to svp file> \
       --crs_code <CRS code for the UTM zone> \
       --interp \
       --create_patches \
       --pings_per_patch <number_of_pings_per_multibeam_patch> \
       --beams_per_patch <number_of_beams_per_multibeam_patch>
```
- `--data_folder`: Path to the parent folder containing the ground truth mesh and the patches generated using `src/data_generation.py`.
- `--data_path`: Path to the patches data generated using `src/data_generation.py`. Needed if `data_folder` is not provided.
- `--mesh_path`: Path to mesh in .xyz format. Needed if `data_folder` is not provided.
- `--resolution`: The provided mesh resolution in meters. Used by the AuvLib draper.
- `--suffix`: Suffix for the output folder. The draping results will be stored in the folder called `draping_{args.resolution}m_{args.suffix}`.
- `--svp_path`: Path to the Sound Velocity Profile (SVP) file. Used by the AuvLib draper.
- `--crs_code`: Coordinate Reference System (CRS) code of the provided data. Used for meridian convergence computation in UTM zones.
- `--interp`: If set, doubles the angular resolution of the simulated multibeam in draping through linear interpolation. This effectively doubles the number of beams per multibeam ping.
- `--create_patches`: If set, point cloud patches of size `<pings_per_patch>` x `<beams_per_patch>` will be generated using the ground truth draping results.
- `--pings_per_patch`: Number of consecutive multibeam pings per patch (default: `32`)
- `--beams_per_patch`: Number of beams per multibeam ping (default: `400`)
