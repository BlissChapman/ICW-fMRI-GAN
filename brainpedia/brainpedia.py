import numpy as np
import torch

from brainpedia.preprocessor import Preprocessor
from nilearn.datasets import fetch_neurovault_ids


class Brainpedia:
    """
    """

    def __init__(self, data_dirs, cache_dir, scale, multi_tag_label_encoding):
        self.data_dirs = data_dirs
        self.cache_dir = cache_dir
        self.multi_tag_label_encoding = multi_tag_label_encoding

        multi_tag_str = 'multi_tag' if multi_tag_label_encoding else 'combined_tag'
        brain_data_filename = "brain_data_{0}_{1}.pkl".format(multi_tag_str, scale)
        brain_data_mask_filename = "brain_data_mask_{0}_{1}.pkl".format(multi_tag_str, scale)
        brain_data_tags_filename = "brain_data_tags_{0}_{1}.pkl".format(multi_tag_str, scale)
        brain_data_tags_encoding_filename = "brain_data_tags_encoding_{0}_{1}.pkl".format(multi_tag_str, scale)
        brain_data_tags_decoding_filename = "brain_data_tags_decoding_{0}_{1}.pkl".format(multi_tag_str, scale)

        self.preprocessor = Preprocessor(data_dirs=self.data_dirs,
                                         output_dir=self.cache_dir,
                                         scale=scale,
                                         multi_tag_label_encoding=multi_tag_label_encoding,
                                         brain_data_filename=brain_data_filename,
                                         brain_data_mask_filename=brain_data_mask_filename,
                                         brain_data_tags_filename=brain_data_tags_filename,
                                         brain_data_tags_encoding_filename=brain_data_tags_encoding_filename,
                                         brain_data_tags_decoding_filename=brain_data_tags_decoding_filename)

    def all_data(self):
        # Load data from preprocessed binary files.
        brain_data = np.array(self.preprocessor.brain_data())
        brain_data_tags = np.array(self.preprocessor.brain_data_tags())
        return brain_data, brain_data_tags

    def train_test_split(self):
        # Load all data from preprocessed binary files.
        brain_data = self.preprocessor.brain_data()
        brain_data_tags = self.preprocessor.brain_data_tags()
        epoch_length = len(brain_data_tags)

        # Shuffle data
        rng_state = np.random.get_state()
        np.random.shuffle(brain_data)
        np.random.set_state(rng_state)
        np.random.shuffle(brain_data_tags)

        # Split into training and test sets
        end_train_data_idx = int(epoch_length * (3 / 4))
        train_brain_data = np.array(brain_data[:end_train_data_idx])
        train_brain_data_tags = np.array(brain_data_tags[:end_train_data_idx])
        test_brain_data = np.array(brain_data[end_train_data_idx:])
        test_brain_data_tags = np.array(brain_data_tags[end_train_data_idx:])

        return train_brain_data, train_brain_data_tags, test_brain_data, test_brain_data_tags

    def batch_generator(brain_data, brain_data_tags, batch_size, cuda):
        epoch_length = len(brain_data_tags)

        while True:
            # Shuffle data between epochs:
            rng_state = np.random.get_state()
            np.random.shuffle(brain_data)
            np.random.set_state(rng_state)
            np.random.shuffle(brain_data_tags)

            for i in range(0, epoch_length, batch_size):
                # Retrieve data and tags.
                batch_end_idx = i + batch_size
                brain_data_batch = np.array(brain_data[i:batch_end_idx])
                brain_data_tags_batch = np.array(brain_data_tags[i:batch_end_idx])

                # Create torch tensors
                brain_data_batch = torch.Tensor(brain_data_batch)
                brain_data_tags_batch = torch.Tensor(brain_data_tags_batch)

                if cuda:
                    brain_data_batch = brain_data_batch.cuda()
                    brain_data_tags_batch = brain_data_tags_batch.cuda()

                yield (brain_data_batch, brain_data_tags_batch)

    def all_brain_image_paths(self):
        all_data_paths = self.preprocessor.all_data_paths()
        brain_img_data_paths = []
        for p in all_data_paths:
            if p[-6:] == 'nii.gz':
                brain_img_data_paths.append(p)
        return brain_img_data_paths

    def sample_shapes(self):
        brain_data = self.preprocessor.brain_data()
        brain_data_tags = self.preprocessor.brain_data_tags()
        return (brain_data[0].shape, brain_data_tags[0].shape)
