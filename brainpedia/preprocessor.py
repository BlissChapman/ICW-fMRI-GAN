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
                 multi_tag_label_encoding,
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
        self.multi_tag_label_encoding = multi_tag_label_encoding
        self.brain_data_path = self.output_dir + brain_data_filename
        self.brain_data_mask_path = self.output_dir + brain_data_mask_filename
        self.brain_data_tags_path = self.output_dir + brain_data_tags_filename
        self.brain_data_tags_encoding_path = self.output_dir + brain_data_tags_encoding_filename
        self.brain_data_tags_decoding_path = self.output_dir + brain_data_tags_decoding_filename

        self.brain_data_tag_decoding_map = None
        self.brain_data_tag_encoding_map = None

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
        if self.brain_data_tag_encoding_map is None:
            if not self.data_is_preprocessed():
                self._run()
            else:
                self.brain_data_tag_encoding_map = pickle.load(open(self.brain_data_tags_encoding_path, 'rb'))
        return self.brain_data_tag_encoding_map

    def brain_data_tags_decoding(self):
        if self.brain_data_tag_decoding_map is None:
            if not self.data_is_preprocessed():
                self._run()
            else:
                self.brain_data_tag_decoding_map = pickle.load(open(self.brain_data_tags_decoding_path, 'rb'))
        return self.brain_data_tag_decoding_map

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
        self.brain_data_tag_encoding_map = {}
        self.brain_data_tag_decoding_map = {}

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
            if self.multi_tag_label_encoding:
                for tag in metadata_tags:
                    if tag not in self.brain_data_tag_encoding_map:
                        self.brain_data_tag_encoding_map[tag] = tag_encoding_count
                        self.brain_data_tag_decoding_map[tag_encoding_count] = tag
                        tag_encoding_count += 1
            else:
                # Merge all metadata tags into a single string:
                metadata_tags.sort()
                metadata_tags = ",".join(metadata_tags)

                if metadata_tags not in self.brain_data_tag_encoding_map:
                    self.brain_data_tag_encoding_map[metadata_tags] = tag_encoding_count
                    self.brain_data_tag_decoding_map[tag_encoding_count] = metadata_tags
                    tag_encoding_count += 1

        # Compute dataset mask.
        print("Computing dataset mask...\r", end='')
        brain_data_mask = masking.compute_background_mask(brain_imgs)

        # 1-hot encode all brain data tags.
        print("1-hot encoding brain data tags...\r", end='')
        for i in range(len(brain_data_tags)):
            brain_data_tags[i] = self.encode_label(brain_data_tags[i])

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
        pickle.dump(self.brain_data_tag_encoding_map, brain_data_tags_encoding_f)
        pickle.dump(self.brain_data_tag_decoding_map, brain_data_tags_decoding_f)

    def labels_for_brain_image(self, brain_image_path):
        metadata_file_path = brain_image_path.split('.')[0] + '_metadata.json'
        metadata_json = json.load(open(metadata_file_path, 'r'))
        tags = metadata_json['tags'][:-1]
        return tags.split(',')

    def encode_label(self, tags):
        encoding_map = self.brain_data_tags_encoding()
        tags_encoding = np.zeros(len(encoding_map.items()))

        if self.multi_tag_label_encoding:
            for tag in tags:
                tag_one_hot_encoding_idx = encoding_map[tag]
                tags_encoding[tag_one_hot_encoding_idx] = 1
            return tags_encoding
        else:
            sorted_tags = sorted(tags)
            sorted_tags = ",".join(sorted_tags)
            tag_class = encoding_map[sorted_tags]
            tags_encoding[tag_class] = 1.0
            return tags_encoding

    def decode_label(self, encoded_label):
        decoding_map = self.brain_data_tags_decoding()

        if self.multi_tag_label_encoding:
            tags = []
            for i in range(len(encoded_label)):
                if encoded_label[i] == 1.0:
                    tag = decoding_map[i]
                    tags.append(tag)
            return tags
        else:
            encoded_label_class = np.argmax(encoded_label)
            decoded_label = decoding_map[encoded_label_class]
            return decoded_label.split(',')
