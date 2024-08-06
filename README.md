# MBES Score-Denoising Ground Truth Data Generation
This repository contains code to generate ground truth for [Score-Based Multibeam Point Cloud Denoising (AUV Symposium 2024)](https://github.com/luxiya01/mbes-score-denoise) using exports from NaviEdit.


## Installation
#TODO

## Data Generation
### Patch Generation from Raw Data
To generate data patches of size (<pings_per_patch> x <beams_per_patch>), run the following:
```
python data_generation.py \
       --data_folder <folder name> \
       --pings_per_patch <pings per patch (default = 32)> \
       --beams_per_patch <beams per patch (default = 400)>
```
The provided data_folder should have the following structure:
#TODO

### Ground Truth Point Cloud Patch Construction
To generate ground truth MBES point cloud patches, run the following:
```
python draper.py
       --data_folder <folder name> \
       --resolution <resolution in meters> \
       --svp_path <path to svp file> \
       --mesh_path <path to mesh in .xyz format> \
       --crs_code <CRS code for the UTM zone> \
       --create_patches
```
The provided data_folder should have the following structure:
#TODO
