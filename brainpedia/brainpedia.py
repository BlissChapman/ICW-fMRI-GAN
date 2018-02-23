import numpy as np
import os
import torch

from brainpedia.preprocessor import Preprocessor
from nilearn.datasets import fetch_neurovault_ids


class Brainpedia:
    """
    """

    def __init__(self, data_dir, scale, augmented_data_dir=None):
        self.data_dir = data_dir
        self.augmented_data_dir = augmented_data_dir
        self.preprocessor = Preprocessor(data_dir=self.data_dir,
                                         scale=scale,
                                         brain_data_filename="brain_data_{0}.pkl".format(scale),
                                         brain_data_mask_filename="brain_data_mask_{0}.pkl".format(scale),
                                         brain_data_tags_filename="brain_data_tags_{0}.pkl".format(scale),
                                         brain_data_tags_encoding_filename="brain_data_tags_encoding_{0}.pkl".format(scale),
                                         brain_data_tags_decoding_filename="brain_data_tags_decoding_{0}.pkl".format(scale),
                                         augmented_data_dir=self.augmented_data_dir)

        # Load raw collection from neurovault.org if necessary:
        if not os.path.isdir(self.data_dir):
            print("No data directory detected.  Attempting to load collection 1952 from Neurovault.org.")
            _ = fetch_neurovault_ids(collection_ids=[1952], data_dir=self.data_dir, verbose=2)

    def batch_generator(self, batch_size, cuda):
        # Load data from preprocessed binary files.
        brain_data = self.preprocessor.brain_data()
        brain_data_tags = self.preprocessor.brain_data_tags()
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
                brain_data_tags_batch = brain_data_tags[i:batch_end_idx]

                # Create torch tensors
                brain_data_batch = torch.Tensor(brain_data_batch)
                brain_data_tags_batch = torch.Tensor(brain_data_tags_batch)

                if cuda:
                    brain_data_batch = brain_data_batch.cuda()
                    brain_data_tags_batch = brain_data_tags_batch.cuda()

                yield (brain_data_batch, brain_data_tags_batch)

    def all_brain_image_paths(self):
        collection_path = self.data_dir + 'neurovault/collection_1952/'
        all_data_paths = os.listdir(collection_path)
        if self.augmented_data_dir is not None:
            all_data_paths.append(os.listdir(self.augmented_data_dir))

        brain_img_data_paths = []
        for p in all_data_paths:
            if p[-6:] == 'nii.gz':
                brain_img_data_paths.append(collection_path + p)
        return brain_img_data_paths

    def sample_shapes(self):
        brain_data = self.preprocessor.brain_data()
        brain_data_tags = self.preprocessor.brain_data_tags()
        return (brain_data[0].shape, brain_data_tags[0].shape)

    def decode_label(self, encoded_label):
        brain_data_tag_decoding_map = self.preprocessor.brain_data_tags_decoding()
        return brain_data_tag_decoding_map[np.argmax(encoded_label)]
