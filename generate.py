import argparse
import datetime
import models.ICW_FMRI_GAN
import json
import nibabel
import numpy as np
import os
import shutil
import torch
import utils.utils

from brainpedia.brainpedia import Brainpedia
from brainpedia.fmri_processing import invert_preprocessor_scaling
from torch.autograd import Variable

parser = argparse.ArgumentParser(description="Generate specified number of samples from trained generator and writes to specified output directory.")
parser.add_argument('generator_state_dict_path', help='path to a file containing the generative model state dict')
parser.add_argument('train_data_dir', help='the directory containing real fMRI data to train on')
parser.add_argument('train_data_dir_cache', help='the directory to use as a cache for the train_data_dir preprocessing')
parser.add_argument('num_samples', type=int, help='the number of samples to generate')
parser.add_argument('output_dir', help='the directory to save generated samples')
args = parser.parse_args()

# ========== HOUSEKEEPING ==========
CUDA = torch.cuda.is_available()
if CUDA:
    print("Using GPU optimizations!")

np.random.seed(1)
torch.manual_seed(1)
if CUDA:
    torch.cuda.manual_seed(1)

shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)

# ========== Hyperparameters ==========
DOWNSAMPLE_SCALE = 0.25
MODEL_DIMENSIONALITY = 64
CONDITONING_DIMENSIONALITY = 5
NOISE_SAMPLE_LENGTH = 128

description_f = open(args.output_dir + 'collection_metadata.txt', 'w')
description_f.write('DATE: {0}\n\n'.format(datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')))
description_f.write('DOWNSAMPLE_SCALE: {0}\n'.format(DOWNSAMPLE_SCALE))
description_f.write('MODEL_DIMENSIONALITY: {0}\n'.format(MODEL_DIMENSIONALITY))
description_f.write('CONDITONING_DIMENSIONALITY: {0}\n'.format(CONDITONING_DIMENSIONALITY))
description_f.write('NOISE_SAMPLE_LENGTH: {0}\n'.format(NOISE_SAMPLE_LENGTH))
description_f.close()

# ========== Data ==========
brainpedia = Brainpedia(data_dirs=[args.train_data_dir],
                        cache_dir=args.train_data_dir_cache,
                        scale=DOWNSAMPLE_SCALE)
all_brain_data, all_brain_data_tags = brainpedia.all_data()
brainpedia_generator = Brainpedia.batch_generator(all_brain_data, all_brain_data_tags, 1, CUDA)
brain_data_shape, brain_data_tag_shape = brainpedia.sample_shapes()

# ========== Models ==========
generator = models.ICW_FMRI_GAN.Generator(input_size=NOISE_SAMPLE_LENGTH,
                                          output_shape=brain_data_shape,
                                          dimensionality=MODEL_DIMENSIONALITY,
                                          num_classes=brain_data_tag_shape[0],
                                          conditioning_dimensionality=CONDITONING_DIMENSIONALITY,
                                          cudaEnabled=CUDA)
generator.load_state_dict(torch.load(args.generator_state_dict_path))

# ========== Sample Generation ==========
for step in range(args.num_samples):
    # Draw batch of real data and labels from brainpedia
    real_brain_img_data_batch, labels_batch = next(brainpedia_generator)
    real_brain_img_data_batch = Variable(real_brain_img_data_batch)
    labels_batch = Variable(labels_batch)

    # Generate synthetic brain image data with the same label as the real data
    noise_sample = Variable(utils.utils.noise(size=(labels_batch.shape[0], NOISE_SAMPLE_LENGTH), cuda=CUDA))
    sythetic_brain_img_data = generator(noise_sample, labels_batch)

    # Upsample synthetic brain image data
    synthetic_sample_data = sythetic_brain_img_data[0].cpu().data.numpy().squeeze()
    upsampled_synthetic_brain_img = invert_preprocessor_scaling(synthetic_sample_data, brainpedia.preprocessor)

    # Save upsampled synthetic brain image data
    synthetic_sample_output_path = "{0}image_{1}.nii.gz".format(args.output_dir, step)
    nibabel.save(upsampled_synthetic_brain_img, synthetic_sample_output_path)

    # Save synthetic brain image metadata
    with open("{0}image_{1}_metadata.json".format(args.output_dir, step), 'w') as metadata_f:
        tags = ""
        for sample_label in brainpedia.decode_label(labels_batch.data[0]):
            tags += sample_label + ','

        json.dump({'tags': tags}, metadata_f)

    # Logging
    print("PERCENT GENERATED: {0:.2f}%\r".format(100.0 * float(step) / float(args.num_samples)), end='')
