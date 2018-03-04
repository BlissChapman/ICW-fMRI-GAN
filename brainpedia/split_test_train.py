import argparse
import os
import random
import shutil
import sys

parser = argparse.ArgumentParser(description="Utility script that given a folder of Brainpedia data can generate two new folders representing a random train/test split.")
parser.add_argument('data_dir', help='the directory containing Brainpedia data')
parser.add_argument('train_output_dir', help='the directory to output training data')
parser.add_argument('fraction_test', type=float, help='the fraction of data to withold as test data')
parser.add_argument('test_output_dir', help='the directory to output test data')
args = parser.parse_args()


# Check for existence of data directory
if not os.path.isdir(args.data_dir):
    sys.exit("Could not find data directory at: '{0}'".format(args.data_dir))

# Create output directories as needed:
if not os.path.isdir(args.train_output_dir):
    os.makedirs(args.train_output_dir)

if not os.path.isdir(args.test_output_dir):
    os.makedirs(args.test_output_dir)

# Compute random split of train/test data filenames:
data_dir_filenames = os.listdir(args.data_dir)
all_data_filenames = [fname for fname in data_dir_filenames if fname[-6:] == 'nii.gz']
random.shuffle(all_data_filenames)

test_split_idx = int(args.fraction_test * len(all_data_filenames))
test_data_filenames = all_data_filenames[:test_split_idx]
train_data_filenames = all_data_filenames[test_split_idx:]

# Copy data in filenames list + associated metadata from data dir to output dir


def copy_data(data_filenames, output_dir):
    for data_file_name in data_filenames:
        metadata_file_name = data_file_name.split('.')[0] + '_metadata.json'

        shutil.copyfile(args.data_dir + data_file_name, output_dir + data_file_name)
        shutil.copyfile(args.data_dir + metadata_file_name, output_dir + metadata_file_name)


copy_data(train_data_filenames, args.train_output_dir)
copy_data(test_data_filenames, args.test_output_dir)
