from data_processing import DataProcessor
import numpy as np
import os
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--pings_per_patch', type=int, default=64)
    parser.add_argument('--beams_per_patch', type=int, default=400)
    parser.add_argument('--plot', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    processor = DataProcessor(folder=args.data_folder)
    processor.process_folder(plot=args.plot)
    processor.create_patches(args.pings_per_patch, args.beams_per_patch)

if __name__ == '__main__':
    main()