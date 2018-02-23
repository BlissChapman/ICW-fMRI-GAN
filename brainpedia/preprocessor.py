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
                 data_dir,
                 scale,
                 brain_data_filename,
                 brain_data_mask_filename,
                 brain_data_tags_filename,
                 brain_data_tags_encoding_filename,
                 brain_data_tags_decoding_filename,
                 augmented_data_dir=None):
        self.output_dir = data_dir
        if augmented_data_dir:
            self.output_dir += 'augmented_preprocessed/'
        else:
            self.output_dir += 'preprocessed/'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        self.data_dir = data_dir
        self.scale = scale
        self.brain_data_path = self.output_dir + brain_data_filename
        self.brain_data_mask_path = self.output_dir + brain_data_mask_filename
        self.brain_data_tags_path = self.output_dir + brain_data_tags_filename
        self.brain_data_tags_encoding_path = self.output_dir + brain_data_tags_encoding_filename
        self.brain_data_tags_decoding_path = self.output_dir + brain_data_tags_decoding_filename
        self.augmented_data_dir = augmented_data_dir

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

        # Retrieve names of all files in data_dir and augmented_data_dir
        base_dir = self.data_dir + 'neurovault/collection_1952/'
        collection_filenames = os.listdir(base_dir)
        collection_filenames = [self.data_dir + 'neurovault/collection_1952/' + filename for filename in collection_filenames]

        if self.augmented_data_dir is not None:
            for filename in os.listdir(self.augmented_data_dir):
                collection_filenames.append(self.augmented_data_dir + filename)

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
            metadata_tag = self.label_for_brain_image(filename)
            brain_data_tags.append(metadata_tag)

            # Downsample brain image.
            downsampled_brain_img = resample_brain_img(brain_img, scale=self.scale)
            downsampled_brain_img_data = downsampled_brain_img.get_data()

            # Normalize brain image data.
            normalized_downsampled_brain_img_data = normalize_brain_img_data(downsampled_brain_img_data)
            brain_data.append(normalized_downsampled_brain_img_data)

            # Build one hot encoding map.
            if metadata_tag not in brain_data_tag_encoding_map:
                brain_data_tag_encoding_map[metadata_tag] = tag_encoding_count
                brain_data_tag_decoding_map[tag_encoding_count] = metadata_tag
                tag_encoding_count += 1

        # Compute dataset mask.
        print("Computing dataset mask...\r", end='')
        brain_data_mask = masking.compute_background_mask(brain_imgs)

        # 1-hot encode all brain data tags.
        print("1-hot encoding brain data tags...\r", end='')
        num_unique_labels = len(brain_data_tag_encoding_map.items())

        for i in range(len(brain_data_tags)):
            tag = brain_data_tags[i]

            tags_encoding = np.zeros(num_unique_labels)
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

        pickle.dump(brain_data, brain_data_f)
        pickle.dump(brain_data_mask, brain_data_mask_f)
        pickle.dump(brain_data_tags, brain_data_tags_f)
        pickle.dump(brain_data_tag_encoding_map, brain_data_tags_encoding_f)
        pickle.dump(brain_data_tag_decoding_map, brain_data_tags_decoding_f)

    def label_for_brain_image(self, brain_image_path):
        metadata_file_path = brain_image_path.split('.')[0] + '_metadata.json'
        metadata_json = json.load(open(metadata_file_path, 'r'))
        return metadata_json['tags'][:-1]
