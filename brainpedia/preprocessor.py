import json
import numpy as np
import os
import pickle

from brainpedia.fmri_processing import resample_brain_img, normalize_brain_img_data
from nilearn.image import load_img, resample_img
from nilearn.input_data import NiftiMasker
import nilearn.masking as masking


class Preprocessor:
    """
    """

    def __init__(self,
                 data_dirs,
                 output_dir,
                 scale,
                 brain_data_filename,
                 brain_data_mask_filename,
                 brain_data_tags_filename,
                 brain_data_tags_encoding_filename,
                 brain_data_tags_decoding_filename):
        self.data_dirs = data_dirs
        self.output_dir = output_dir
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        self.scale = scale
        self.brain_data_path = self.output_dir + brain_data_filename
        self.brain_data_mask_path = self.output_dir + brain_data_mask_filename
        self.brain_data_tags_path = self.output_dir + brain_data_tags_filename
        self.brain_data_tags_encoding_path = self.output_dir + brain_data_tags_encoding_filename
        self.brain_data_tags_decoding_path = self.output_dir + brain_data_tags_decoding_filename

    def brain_data(self):
        if not self.data_is_preprocessed():
            self._run()
        return pickle.load(open(self.brain_data_path, 'rb'))

    def brain_data_mask(self):
        if not self.data_is_preprocessed():
            self._run()
        return pickle.load(open(self.brain_data_mask_path, 'rb'))

    def brain_data_tags(self):
        if not self.data_is_preprocessed():
            self._run()
        return pickle.load(open(self.brain_data_tags_path, 'rb'))

    def brain_data_tags_encoding(self):
        if not self.data_is_preprocessed():
            self._run()
        return pickle.load(open(self.brain_data_tags_encoding_path, 'rb'))

    def brain_data_tags_decoding(self):
        if not self.data_is_preprocessed():
            self._run()
        return pickle.load(open(self.brain_data_tags_decoding_path, 'rb'))

    def data_is_preprocessed(self):
        return os.path.isfile(self.brain_data_path) \
            and os.path.isfile(self.brain_data_mask_path) \
            and os.path.isfile(self.brain_data_tags_path) \
            and os.path.isfile(self.brain_data_tags_encoding_path) \
            and os.path.isfile(self.brain_data_tags_decoding_path)

    def all_data_paths(self):
        data_paths = []
        for data_dir in self.data_dirs:
            for filename in os.listdir(data_dir):
                data_paths.append(data_dir + filename)
        return data_paths

    def _run(self):
        # NOTE TO FUTURE READERS:
        # This code is extremely memory inefficient.  I am currently working
        # with a small dataset which allows for this inefficiency.
        # Refactoring will be necessary at larger scales.

        print("========== PREPROCESSING ==========")

        # Set up structures to hold brain data, brain imgs, associated tags, and 1-hot mapping
        brain_imgs = []
        brain_data = []
        brain_data_tags = []

        tag_encoding_count = 0
        brain_data_tag_encoding_map = {}
        brain_data_tag_decoding_map = {}

        # Retrieve names of all files in data_dirs
        collection_filenames = self.all_data_paths()

        # Loop over data files:
        total_num_files = len(collection_filenames)
        num_processed = 0

        for filename in collection_filenames:
            num_processed += 1
            print("PERCENT COMPLETE: {0:.2f}%\r".format(100.0 * float(num_processed) / float(total_num_files)), end='')

            # Ignore files that are not images.
            if filename[-2:] != 'gz':
                continue

            # Load brain image.
            brain_img = load_img(filename)
            brain_imgs.append(brain_img)

            # Load brain image metadata.
            metadata_tags = self.labels_for_brain_image(filename)
            brain_data_tags.append(metadata_tags)

            # Downsample brain image.
            downsampled_brain_img = resample_brain_img(brain_img, scale=self.scale)
            downsampled_brain_img_data = downsampled_brain_img.get_data()

            # Normalize brain image data.
            normalized_downsampled_brain_img_data = normalize_brain_img_data(downsampled_brain_img_data)
            brain_data.append(normalized_downsampled_brain_img_data)

            # Build one hot encoding map.
            for tag in metadata_tags:
                if tag not in brain_data_tag_encoding_map:
                    brain_data_tag_encoding_map[tag] = tag_encoding_count
                    brain_data_tag_decoding_map[tag_encoding_count] = tag
                    tag_encoding_count += 1

        # Compute dataset mask.
        print("Computing dataset mask...\r", end='')
        brain_data_mask = masking.compute_background_mask(brain_imgs)

        # 1-hot encode all brain data tags.
        print("1-hot encoding brain data tags...\r", end='')
        num_unique_labels = len(brain_data_tag_encoding_map.items())

        for i in range(len(brain_data_tags)):
            tags = brain_data_tags[i]
            tags_encoding = np.zeros(num_unique_labels)
            for tag in tags:
                tag_one_hot_encoding_idx = brain_data_tag_encoding_map[tag]
                tags_encoding[tag_one_hot_encoding_idx] = 1
            brain_data_tags[i] = tags_encoding

        # Write preprocessed brain data out as binary files.
        print("Writing preprocessed brain data out to files...\r", end='')
        brain_data_f = open(self.brain_data_path, 'wb')
        brain_data_mask_f = open(self.brain_data_mask_path, 'wb')
        brain_data_tags_f = open(self.brain_data_tags_path, 'wb')
        brain_data_tags_encoding_f = open(self.brain_data_tags_encoding_path, 'wb')
        brain_data_tags_decoding_f = open(self.brain_data_tags_decoding_path, 'wb')
        print("                                                 \r", end='')

        pickle.dump(brain_data, brain_data_f)
        pickle.dump(brain_data_mask, brain_data_mask_f)
        pickle.dump(brain_data_tags, brain_data_tags_f)
        pickle.dump(brain_data_tag_encoding_map, brain_data_tags_encoding_f)
        pickle.dump(brain_data_tag_decoding_map, brain_data_tags_decoding_f)

    def labels_for_brain_image(self, brain_image_path):
        metadata_file_path = brain_image_path.split('.')[0] + '_metadata.json'
        metadata_json = json.load(open(metadata_file_path, 'r'))
        tags = metadata_json['tags'][:-1]
        return tags.split(',')
