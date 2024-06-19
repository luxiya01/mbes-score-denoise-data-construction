from data_processing import DataProcessor
import numpy as np
import os
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--pings_per_patch', type=int, default=32)
    parser.add_argument('--beams_per_patch', type=int, default=400)
    parser.add_argument('--plot', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    processor = DataProcessor(folder=args.data_folder)
    processor.process_folder(plot=args.plot)

    all_data_path = os.path.join(processor.out_folder, 'all_data.npz')
    processor.split_data_into_train_and_test(all_data_path, test_pings=[84487, 148283]) # start and end idx of test pings

    filenames = ['train_data_part1.npz', 'train_data_part2.npz', 'test_data.npz']
    for f in filenames:
        filepath = os.path.join(processor.out_folder, f)
        processor.create_patches(filepath, args.pings_per_patch, args.beams_per_patch)

if __name__ == '__main__':
    main()