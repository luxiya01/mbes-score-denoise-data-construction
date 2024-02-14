from collections import defaultdict
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def interpolate_array(arr, rejected, method='nearest', resolution=1):
    """
    Filter an array based on a boolean filter and interpolate missing values.
    Method: 'linear', 'nearest', 'cubic'
    """
    # set up the grid
    x = np.arange(0, arr.shape[0], step=resolution)
    y = np.arange(0, arr.shape[1], step=resolution)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    # filter array and get indices where it is not nan
    valid_idx = np.where(~rejected)

    # interpolate the data
    arr_interp = griddata(
        valid_idx,
        arr[valid_idx],
        (xv, yv),
        method=method,
    )
    return arr_interp

class DataProcessor:
    """
    Class to process the raw data .xyz and .xyzi files exported from NaviEdit into numpy arrays
    """
    def __init__(self, folder, logger=None) -> None:
        self.logger = self.setup_logger(logger)
        self.folder = folder
        self._setup_subfolders()
        self.files_dict = self._get_files_dict()

    def __len__(self):
        return len(self.files_dict)

    def setup_logger(self, logger):
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def _setup_subfolders(self):
        self.pings_folder = os.path.join(self.folder, 'pings-xyz')
        self.angle_folder = os.path.join(self.folder, 'angle-quality')
        self.intensity_folder = os.path.join(self.folder, 'intensity')
        self.cleaned_folder = os.path.join(self.folder, 'cleaned')
        self.motion_folder = os.path.join(self.folder, 'motion')
        self.out_folder = os.path.join(self.folder, 'merged')
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

    def _get_files_dict(self):
        files_dict = {}
        for filename in sorted(os.listdir(self.pings_folder),
                               key=lambda x: int(x.split('-')[1])):
            idx = int(filename.split('-')[1])
            files_dict[idx] = {
                'pings': os.path.join(self.pings_folder, filename),
                'angle_quality': os.path.join(self.angle_folder, filename),
                'intensity': os.path.join(self.intensity_folder, f'{filename}i'),
                'cleaned': os.path.join(self.cleaned_folder, filename),
                'motion': os.path.join(self.motion_folder, f'{filename.split(".")[0]}.csv'),
                'out': os.path.join(self.out_folder, filename) + '.npz',
            }
        return files_dict

    def _parse_angle_quality(self, idx):
        filename = self.files_dict[idx]['angle_quality']
        scan_no_list = []
        beam_id_list = []
        x_list = []
        y_list = []
        z_list = []
        angle_list = []
        quality_list = []

        with open(filename) as f:
            current_scan_no = None
            current_beam_id = None
            for line in f:
                scan_match = re.match(r'Scan no.: (\d+)\s+(\d+) beams', line)
                if scan_match:
                    current_scan_no = int(scan_match.group(1))
                    num_beams = int(scan_match.group(2))
                    current_beam_id = 0

                data_match = re.match(r'\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)', line)
                if data_match:
                    scan_no_list.append(current_scan_no)
                    beam_id_list.append(current_beam_id)
                    x_list.append(float(data_match.group(1)))
                    y_list.append(float(data_match.group(2)))
                    z_list.append(float(data_match.group(3)))
                    angle_list.append(float(data_match.group(4)))
                    quality_list.append(int(data_match.group(5)))
                    current_beam_id += 1
        angle_quality_df = pd.DataFrame({
            'scan_no': scan_no_list,
            'beam_id': beam_id_list,
            'X': x_list,
            'Y': y_list,
            'Z': z_list,
            'angle': angle_list,
            'quality': quality_list,
        })
        return angle_quality_df

    def _parse_intensity(self, idx):
        filename = self.files_dict[idx]['intensity']
        intensity_list = []
        with open(filename) as f:
            for line in f:
                intensity_list.append(int(line.split(' ')[-1]))
        return intensity_list
        
    def _parse_cleaned(self, idx):
        filename = self.files_dict[idx]['cleaned']

        cleaned_df = pd.read_csv(filename,
                         header=None, sep=' ',
                         names=['X', 'Y', 'Z', 'rejection_flag'])
        cleaned_df['rejected'] = cleaned_df['rejection_flag'] == 'R'
        cleaned_df['rejected'].fillna(False, inplace=True)
        self.logger.info(f'Cleaned data shape: {cleaned_df.shape}')
        return cleaned_df[['X', 'Y', 'Z', 'rejected']]

    def _parse_motion(self, idx):
        filename = self.files_dict[idx]['motion']
        motion_df = pd.read_csv(filename, dtype={'Date': str, 'Time': str})
        motion_df['datetime'] = pd.to_datetime(motion_df['Date'] + motion_df['Time'],
                                               format='%Y-%m-%d%H:%M:%S.%f')
        self.logger.info(f'Motion data shape: {motion_df.shape}')
        return motion_df

    
    def _parse_raw_pings(self, idx):
        filename = self.files_dict[idx]['pings']
        df = pd.read_csv(filename, sep='\t',
                         dtype={'yyyy': str,
                                'mmddhhmm': str,
                                'ss.ss': str})
        # set scan_no to start from 0 and cumulate per ping
        df['scan_no'] = (df['Ping No'] != df['Ping No'].shift(1)).cumsum() - 1
        df['beam_id'] = df['Beam No']

        # convert time info to datetime column
        df['datetime'] = pd.to_datetime(df['yyyy'].astype(str) + df['mmddhhmm'].astype(str) 
                                            + df['ss.ss'].astype(str), format='%Y%m%d%H%M%S.%f')
        self.logger.info(f'Raw pings shape: {df.shape}')
        return df
        
    def get_angle_quality_and_intensity_df(self, idx):
        angle_quality_df = self._parse_angle_quality(idx)
        intensity_list = self._parse_intensity(idx)
        angle_quality_df['intensity'] = intensity_list
        self.logger.info(f'Angle quality and intensity shape: {angle_quality_df.shape}')
        return angle_quality_df

    def get_processed_data_df(self, idx):
        """
        Get the processed data dataframe for a given index.

        Parameters:
        - idx (int): The index of the data.

        Returns:
        - processed_df (pandas.DataFrame): The processed data dataframe.
        """
        # Load and parse relevant raw data
        pings_df = self._parse_raw_pings(idx)
        cleaned_df = self._parse_cleaned(idx)
        angle_quality_df = self.get_angle_quality_and_intensity_df(idx)
        motion_df = self._parse_motion(idx)[['datetime', 'Depth']]

        # Merge pings_df and motion_df based on datetime
        pings_df = pd.merge_asof(pings_df.sort_values('datetime'),
                                 motion_df.sort_values('datetime'),
                                 on='datetime',
                                 direction='nearest')
        pings_df['Z_relative'] = pings_df['Z'] - pings_df['Depth']

        # Merge the three dataframes based on XYZ
        processed_df = pd.merge(pings_df, angle_quality_df, on=['scan_no', 'beam_id', 'X', 'Y', 'Z'], how='outer')
        processed_df = pd.merge(processed_df, cleaned_df, on=['X', 'Y', 'Z'], how='outer')


        # Drop any duplicated rows
        processed_df.drop_duplicates(subset=['Ping No', 'Beam No', 'X', 'Y', 'Z'], inplace=True)
        # Drop any rows with NaN values in Ping No and Beam No
        processed_df.dropna(subset=['Ping No', 'Beam No'], inplace=True)

        self.logger.info(f'Processed data shape: {processed_df.shape}')
        return processed_df
    
    def process_data_to_numpy(self, idx):
        """
        Process the data for a given index and save it to a numpy file.
        Returns True if the data was processed and saved, False if it already exists.
        """
        out_filepath = self.files_dict[idx]['out']
        # check whether the file already exists
        if os.path.exists(out_filepath):
            self.logger.info(f'Processed data already exists at {out_filepath}')
            return False
        processed_df = self.get_processed_data_df(idx)
        pivot = processed_df.pivot(index='Ping No',
                                   columns='Beam No',
                                   values=[
                                       'X', 'Y', 'Z', 'Z_relative', 'Depth',
                                       'intensity', 'angle', 'quality', 'rejected',
                                       'Roll', 'Pitch', 'Heading', 'Heave',
                                       'datetime'
                                  ])
        data = {
            'X': pivot['X'].to_numpy().astype(np.float32),
            'Y': pivot['Y'].to_numpy().astype(np.float32),
            'Z': pivot['Z'].to_numpy().astype(np.float32),
            'Z_relative': pivot['Z_relative'].to_numpy().astype(np.float32),
            'depth': pivot['Depth'].to_numpy().astype(np.float32),
            'intensity': pivot['intensity'].to_numpy().astype(np.int),
            'angle': pivot['angle'].to_numpy().astype(np.float32),
            'quality': pivot['quality'].to_numpy().astype(np.int),
            'rejected': pivot['rejected'].to_numpy().astype(np.bool),
            'roll': pivot['Roll'].to_numpy().astype(np.float32),
            'pitch': pivot['Pitch'].to_numpy().astype(np.float32),
            'heading': pivot['Heading'].to_numpy().astype(np.float32),
            'heave': pivot['Heave'].to_numpy().astype(np.float32),
            'datetime': pivot['datetime'].to_numpy().astype(np.datetime64),
        }
        for method in ['linear', 'nearest', 'cubic']:
            data[f'X_interp_{method}'] = interpolate_array(data['X'], data['rejected'], method=method)
            data[f'Y_interp_{method}'] = interpolate_array(data['Y'], data['rejected'], method=method)
            data[f'Z_interp_{method}'] = interpolate_array(data['Z'], data['rejected'], method=method)
            data[f'Z_relative_interp_{method}'] = interpolate_array(data['Z_relative'], data['rejected'], method=method)
        np.savez(out_filepath, **data)
        self.logger.info(f'Processed data saved to {out_filepath}')
        return True

    def plot_numpy_data(self, idx, savefig=True, suffix='', num_rows=2, filter_rejected=False,
                        keys=['X', 'Y', 'Z', 'Z_relative', 'intensity', 'depth',
                              'quality', 'rejected', 'angle', 'roll', 'pitch', 'heading']):
        filename = self.files_dict[idx]['out']
        data = np.load(filename)
        num_columns = (len(keys)+1)//num_rows if num_rows > 1 else len(keys)
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns*3, num_columns*3))
        for i, key in enumerate(keys):
            if num_rows == 1:
                a = axes[i]
            else:
                a = axes[i//num_columns, i%num_columns]
            im = data[key]
            if filter_rejected:
                im = im.astype(float)
                im[data['rejected']] = np.nan
            fig.colorbar(a.imshow(im), ax=a)
            a.set_title(key)
        if savefig:
            fig.tight_layout()
            filename = os.path.basename(filename).split('.')[0]
            figname = f'{filename}{suffix}-filtered.png' if filter_rejected else f'{filename}{suffix}.png'
            fig.savefig(figname, dpi=150, bbox_inches='tight')
            plt.close(fig)

    def process_folder(self, plot=True):
        for idx in tqdm(self.files_dict.keys()):
            try:
                self.logger.info(f'Processing idx={idx}...')
                success = self.process_data_to_numpy(idx)
                if success and plot:
                    self.plot_numpy_data(idx)
                    self.plot_numpy_data(idx, filter_rejected=True)
                    for method in ['linear', 'nearest', 'cubic']:
                        suffix = f'_interp_{method}'
                        self.plot_numpy_data(idx, suffix=suffix,
                                             keys=[f'X{suffix}',
                                                   f'Y{suffix}',
                                                   f'Z{suffix}',
                                                   f'Z_relative{suffix}'],
                                             num_rows=1, filter_rejected=False)
            except Exception as e:
                self.logger.error(f'Error processing file idx={idx}: {e}')

    def merge_all_npz_data(self):
        """
        Merge all the processed data into a single .npz file,
        requires that all the .xyz data to be processed, e.g.
        using the process_folder() method.
        """
        all_data_path = os.path.join(self.out_folder, 'all_data.npz')
        if os.path.exists(all_data_path):
            self.logger.info(f'Using existing all_data.npz at {all_data_path}')
            return np.load(all_data_path, allow_pickle=True)

        # If not all processed data exists, raise an error
        if not os.path.exists(self.out_folder):
            self.logger.error('No processed data found. Run process_folder() first.')
        for k, v in self.files_dict.items():
            if not os.path.exists(v['out']):
                self.logger.error(f'Processed data not found for file idx {k}. Run process_folder() first.')

        # Load and concatenate all numpy data
        self.logger.info('Merging all data...')
        all_data_dict = defaultdict(list)
        for idx, filepaths in tqdm(self.files_dict.items()):
            data = np.load(filepaths['out'], allow_pickle=True)
            for file in data.files:
                all_data_dict[file].append(data[file])
        for k, v in all_data_dict.items():
            all_data_dict[k] = np.concatenate(v, axis=0)
        self.logger.info(f'All data shape: {all_data_dict["X"].shape}')
        np.savez(all_data_path, **all_data_dict)
        return all_data_dict

    def create_patches(self, pings_per_patch, beams_per_patch):
        all_data = self.merge_all_npz_data()
        folder_name = f'patches_{pings_per_patch}pings_{beams_per_patch}_beams'
        patches_folder = os.path.join(self.out_folder, folder_name)
        if not os.path.exists(patches_folder):
            os.makedirs(patches_folder)
        else:
            self.logger.warning(f'Patches folder {patches_folder} already exists. Overwriting...')

        self.logger.info(f'Creating patches with size ({pings_per_patch}, {beams_per_patch})...')
        num_pings, num_beams = all_data['X'].shape
        num_patches = 0
        for i in tqdm(range(0, num_pings, pings_per_patch)):
            for j in tqdm(range(0, num_beams, beams_per_patch)):
                patch = {k: v[i:i+pings_per_patch, j:j+beams_per_patch]
                         for k, v in all_data.items()}
                patch['idx'] = num_patches
                patch['start_ping'] = i
                patch['start_beam'] = j
                patch['end_ping'] = i+pings_per_patch
                patch['end_beam'] = j+beams_per_patch
                num_patches += 1
                np.savez(os.path.join(patches_folder, f'patch_{num_patches}.npz'), **patch)
                del patch
        self.logger.info(f'Created {num_patches} patches\n'
                         f'saved to {patches_folder}')