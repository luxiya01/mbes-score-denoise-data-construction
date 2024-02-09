import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def interpolate_array(arr, rejected, method='linear', resolution=1):
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
    def __init__(self, folder) -> None:
        self.folder = folder
        self._setup_subfolders()
        self.files_dict = self._get_files_dict()

    def __len__(self):
        return len(self.files_dict)

    def _setup_subfolders(self):
        self.pings_folder = os.path.join(self.folder, 'pings-xyz')
        self.angle_folder = os.path.join(self.folder, 'angle-quality')
        self.intensity_folder = os.path.join(self.folder, 'intensity')
        self.cleaned_folder = os.path.join(self.folder, 'cleaned')
        self.out_folder = os.path.join(self.folder, 'merged')
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

    def _get_files_dict(self):
        files_dict = {}
        for filename in os.listdir(self.pings_folder):
            idx = int(filename.split('-')[1])
            files_dict[idx] = {
                'pings': os.path.join(self.pings_folder, filename),
                'angle_quality': os.path.join(self.angle_folder, filename),
                'intensity': os.path.join(self.intensity_folder, f'{filename}i'),
                'cleaned': os.path.join(self.cleaned_folder, filename),
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
        logging.info(f'Cleaned data shape: {cleaned_df.shape}')
        return cleaned_df[['X', 'Y', 'Z', 'rejected']]
    
    def _parse_raw_pings(self, idx):
        filename = self.files_dict[idx]['pings']
        df = pd.read_csv(filename, sep='\t')
        # set scan_no to start from 0 and cumulate per ping
        df['scan_no'] = (df['Ping No'] != df['Ping No'].shift(1)).cumsum() - 1
        df['beam_id'] = df['Beam No']

        # convert time info to datetime column
        df['datetime'] = pd.to_datetime(df['yyyy'].astype(str) + df['mmddhhmm'].astype(str) 
                                            + df['ss.ss'].astype(str), format='%Y%m%d%H%M%S.%f')
        logging.info(f'Raw pings shape: {df.shape}')
        return df
        
    def get_angle_quality_and_intensity_df(self, idx):
        angle_quality_df = self._parse_angle_quality(idx)
        intensity_list = self._parse_intensity(idx)
        angle_quality_df['intensity'] = intensity_list
        logging.info(f'Angle quality and intensity shape: {angle_quality_df.shape}')
        return angle_quality_df

    def get_processed_data_df(self, idx):
        """
        Get the processed data dataframe for a given index.

        Parameters:
        - idx (int): The index of the data.

        Returns:
        - processed_df (pandas.DataFrame): The processed data dataframe.
        """
        pings_df = self._parse_raw_pings(idx)
        cleaned_df = self._parse_cleaned(idx)
        angle_quality_df = self.get_angle_quality_and_intensity_df(idx)

        # Merge the three dataframes based on XYZ
        processed_df = pd.merge(pings_df, angle_quality_df, on=['scan_no', 'beam_id', 'X', 'Y', 'Z'], how='outer')
        processed_df = pd.merge(processed_df, cleaned_df, on=['X', 'Y', 'Z'], how='outer')

        # Drop any duplicated rows
        processed_df.drop_duplicates(subset=['Ping No', 'Beam No', 'X', 'Y', 'Z'], inplace=True)
        # Drop any rows with NaN values in Ping No and Beam No
        processed_df.dropna(subset=['Ping No', 'Beam No'], inplace=True)

        logging.info(f'Processed data shape: {processed_df.shape}')
        return processed_df
    
    def process_data_to_numpy(self, idx):
        out_filepath = self.files_dict[idx]['out']
        # check whether the file already exists
        if os.path.exists(out_filepath):
            logging.info(f'Processed data already exists at {out_filepath}')
            return
        processed_df = self.get_processed_data_df(idx)
        pivot = processed_df.pivot(index='Ping No',
                                   columns='Beam No',
                                   values=[
                                       'X', 'Y', 'Z', 'intensity',
                                       'angle', 'quality', 'rejected',
                                       'Roll', 'Pitch', 'Heading', 'Heave',
                                       'datetime'
                                  ])
        data = {
            'X': pivot['X'].to_numpy().astype(np.float32),
            'Y': pivot['Y'].to_numpy().astype(np.float32),
            'Z': pivot['Z'].to_numpy().astype(np.float32),
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
        data['X_interp'] = interpolate_array(data['X'], data['rejected'])
        data['Y_interp'] = interpolate_array(data['Y'], data['rejected'])
        data['Z_interp'] = interpolate_array(data['Z'], data['rejected'])
        np.savez(out_filepath, **data)
        logging.info(f'Processed data saved to {out_filepath}')
        return processed_df, pivot

    def plot_numpy_data(self, idx, savefig=True, suffix='', num_rows=2, filter_rejected=False,
                        keys=['X', 'Y', 'Z', 'intensity', 'angle', 'quality', 'rejected', 'roll', 'pitch', 'heading']):
        filename = self.files_dict[idx]['out']
        data = np.load(filename)
        num_columns = (len(keys)+1)//num_rows
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns*3, num_columns*3))
        for i, key in enumerate(keys):
            a = axes[i//num_columns, i%num_columns]
            im = data[key]
            if filter_rejected:
                im[data['rejected']] = np.nan
            fig.colorbar(a.imshow(im), ax=a)
            a.set_title(key)
        if savefig:
            fig.tight_layout()
            figname = f'{filename}{suffix}-filtered.png' if filter_rejected else f'{filename}{suffix}.png'
            fig.savefig(os.path.join(self.out_folder, figname),
                                    dpi=150, bbox_inches='tight')
            plt.close(fig)

    def process_folder(self, plot=True):
        for idx in tqdm(self.files_dict.keys()):
            try:
                logging.info(f'Processing idx={idx}...')
                self.process_data_to_numpy(idx)
                if plot:
                    self.plot_numpy_data(idx)
                    self.plot_numpy_data(idx, filter_rejected=True)
                    self.plot_numpy_data(idx, suffix='-interp', keys=['X_interp', 'Y_interp', 'Z_interp'],
                                         num_rows=1, filter_rejected=False)
            except Exception as e:
                logging.error(f'Error processing file idx={idx}: {e}')