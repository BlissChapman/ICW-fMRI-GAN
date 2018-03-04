import matplotlib
matplotlib.use('Agg')

import argparse
import datetime
import models.ICW_FMRI_GAN
import nibabel
import numpy as np
import os
import shutil
import timeit
import torch
import utils.utils

from brainpedia.brainpedia import Brainpedia
from brainpedia.fmri_processing import invert_preprocessor_scaling
from evaluation.neurosynth import avg_correlation_of_image_to_images_in_brainpedia_with_same_tags
from utils.plot import Plot
from torch.autograd import Variable

parser = argparse.ArgumentParser(description="Train ICW_FMRI_GAN.")
parser.add_argument('train_data_dir', help='the directory containing real fMRI data to train on')
parser.add_argument('train_data_dir_cache', help='the directory to use as a cache for the train_data_dir preprocessing')
parser.add_argument('output_dir', help='the directory to save training results')
args = parser.parse_args()

# ========== OUTPUT DIRECTORIES ==========
DATA_OUTPUT_DIR = args.output_dir + 'data/'
VIS_OUTPUT_DIR = args.output_dir + 'visualizations/'
MODEL_OUTPUT_DIR = args.output_dir + 'models/'

shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)
os.makedirs(DATA_OUTPUT_DIR)
os.makedirs(VIS_OUTPUT_DIR)
os.makedirs(MODEL_OUTPUT_DIR)

# ========== Hyperparameters ==========
DOWNSAMPLE_SCALE = 0.25
TRAINING_STEPS = 200000
BATCH_SIZE = 50
MODEL_DIMENSIONALITY = 64
CONDITONING_DIMENSIONALITY = 5
CRITIC_UPDATES_PER_GENERATOR_UPDATE = 5
LAMBDA = 10
VISUALIZATION_INTERVAL = 1000
NOISE_SAMPLE_LENGTH = 128

description_f = open(args.output_dir + 'description.txt', 'w')
description_f.write('DATE: {0}\n\n'.format(datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')))
description_f.write('DOWNSAMPLE_SCALE: {0}\n'.format(DOWNSAMPLE_SCALE))
description_f.write('TRAINING_STEPS: {0}\n'.format(TRAINING_STEPS))
description_f.write('BATCH_SIZE: {0}\n'.format(BATCH_SIZE))
description_f.write('MODEL_DIMENSIONALITY: {0}\n'.format(MODEL_DIMENSIONALITY))
description_f.write('CONDITONING_DIMENSIONALITY: {0}\n'.format(CONDITONING_DIMENSIONALITY))
description_f.write('CRITIC_UPDATES_PER_GENERATOR_UPDATE: {0}\n'.format(CRITIC_UPDATES_PER_GENERATOR_UPDATE))
description_f.write('LAMBDA: {0}\n'.format(LAMBDA))
description_f.write('VISUALIZATION_INTERVAL: {0}\n'.format(VISUALIZATION_INTERVAL))
description_f.write('NOISE_SAMPLE_LENGTH: {0}\n'.format(NOISE_SAMPLE_LENGTH))
description_f.close()

# ========== HOUSEKEEPING ==========
CUDA = torch.cuda.is_available()
if CUDA:
    print("Using GPU optimizations!")

np.random.seed(1)
torch.manual_seed(1)
if CUDA:
    torch.cuda.manual_seed(1)

# ========== Data ==========
brainpedia = Brainpedia(data_dirs=[args.train_data_dir],
                        cache_dir=args.train_data_dir_cache,
                        scale=DOWNSAMPLE_SCALE)
all_brain_data, all_brain_data_tags = brainpedia.all_data()
brainpedia_generator = brainpedia.batch_generator(all_brain_data, all_brain_data_tags, BATCH_SIZE, CUDA)
brain_data_shape, brain_data_tag_shape = brainpedia.sample_shapes()

# ========== Models ==========
generator = models.ICW_FMRI_GAN.Generator(input_size=NOISE_SAMPLE_LENGTH,
                                          output_shape=brain_data_shape,
                                          dimensionality=MODEL_DIMENSIONALITY,
                                          num_classes=brain_data_tag_shape[0],
                                          conditioning_dimensionality=CONDITONING_DIMENSIONALITY,
                                          cudaEnabled=CUDA)
critic = models.ICW_FMRI_GAN.Critic(dimensionality=MODEL_DIMENSIONALITY,
                                    num_classes=brain_data_tag_shape[0],
                                    conditioning_dimensionality=CONDITONING_DIMENSIONALITY,
                                    cudaEnabled=CUDA)

# ========= Training =========
critic_losses_per_vis_interval = []
generator_losses_per_vis_interval = []

running_critic_loss = 0.0
running_generator_loss = 0.0
running_batch_start_time = timeit.default_timer()

for training_step in range(1, TRAINING_STEPS + 1):
    print("BATCH: [{0}/{1}]\r".format(training_step % VISUALIZATION_INTERVAL, VISUALIZATION_INTERVAL), end='')

    # Train critic
    for critic_step in range(CRITIC_UPDATES_PER_GENERATOR_UPDATE):
        real_brain_img_data_batch, labels_batch = next(brainpedia_generator)
        real_brain_img_data_batch = Variable(real_brain_img_data_batch)
        labels_batch = Variable(labels_batch)

        noise_sample_c = Variable(utils.utils.noise(size=(labels_batch.shape[0], NOISE_SAMPLE_LENGTH), cuda=CUDA))
        synthetic_brain_img_data_batch = generator(noise_sample_c, labels_batch)
        critic_loss = critic.train(real_brain_img_data_batch, synthetic_brain_img_data_batch, labels_batch, LAMBDA)
        running_critic_loss += critic_loss.data[0]

    # Train generator
    noise_sample_g = Variable(utils.utils.noise(size=(labels_batch.shape[0], NOISE_SAMPLE_LENGTH), cuda=CUDA))
    synthetic_brain_img_data_batch = generator(noise_sample_g, labels_batch)
    critic_output = critic(synthetic_brain_img_data_batch, labels_batch)
    generator_loss = generator.train(critic_output)
    running_generator_loss += generator_loss.data[0]

    # Visualization
    if training_step % VISUALIZATION_INTERVAL == 0:
        # Timing
        running_batch_elapsed_time = timeit.default_timer() - running_batch_start_time
        running_batch_start_time = timeit.default_timer()

        num_training_batches_remaining = (TRAINING_STEPS - training_step) / BATCH_SIZE
        estimated_minutes_remaining = (num_training_batches_remaining * running_batch_elapsed_time) / 60.0

        print("===== TRAINING STEP {} | ~{:.0f} MINUTES REMAINING =====".format(training_step, estimated_minutes_remaining))
        print("CRITIC LOSS:     {0}".format(running_critic_loss))
        print("GENERATOR LOSS:  {0}\n".format(running_generator_loss))

        # Loss histories
        critic_losses_per_vis_interval.append(running_critic_loss)
        generator_losses_per_vis_interval.append(running_generator_loss)
        running_critic_loss = 0.0
        running_generator_loss = 0.0

        Plot.plot_histories([critic_losses_per_vis_interval],
                            ["Critic"],
                            "{0}critic_loss_history.png".format(MODEL_OUTPUT_DIR))
        Plot.plot_histories([generator_losses_per_vis_interval],
                            ["Generator"],
                            "{0}generator_loss_history.png".format(MODEL_OUTPUT_DIR))

        # Save model at checkpoint
        torch.save(generator.state_dict(), "{0}generator".format(MODEL_OUTPUT_DIR))
        torch.save(critic.state_dict(), "{0}critic".format(MODEL_OUTPUT_DIR))

        # Upsample and save samples
        sample_tags = brainpedia.decode_label(labels_batch.data[0])
        real_sample_data = real_brain_img_data_batch[0].cpu().data.numpy().squeeze()
        synthetic_sample_data = synthetic_brain_img_data_batch[0].cpu().data.numpy().squeeze()
        upsampled_real_brain_img = invert_preprocessor_scaling(real_sample_data, brainpedia.preprocessor)
        upsampled_synthetic_brain_img = invert_preprocessor_scaling(synthetic_sample_data, brainpedia.preprocessor)

        real_sample_output_path = "{0}sample_{1}_real.nii.gz".format(DATA_OUTPUT_DIR, training_step)
        synthetic_sample_output_path = "{0}sample_{1}_synthetic.nii.gz".format(DATA_OUTPUT_DIR, training_step)

        nibabel.save(upsampled_real_brain_img, real_sample_output_path)
        nibabel.save(upsampled_synthetic_brain_img, synthetic_sample_output_path)

        # Compute correlation scores
        real_sample_correlation = avg_correlation_of_image_to_images_in_brainpedia_with_same_tags(image_path=real_sample_output_path,
                                                                                                  brainpedia=brainpedia,
                                                                                                  tags=sample_tags)
        synthetic_sample_correlation = avg_correlation_of_image_to_images_in_brainpedia_with_same_tags(image_path=synthetic_sample_output_path,
                                                                                                       brainpedia=brainpedia,
                                                                                                       tags=sample_tags)

        # Visualize samples
        title = "{0}".format(sample_tags)
        Plot.plot_sample_brain_data(real_sample_brain_img=upsampled_real_brain_img,
                                    synthetic_sample_brain_img=upsampled_synthetic_brain_img,
                                    real_sample_correlation=real_sample_correlation,
                                    synthetic_sample_correlation=synthetic_sample_correlation,
                                    output_file="{0}sample_{1}".format(VIS_OUTPUT_DIR, training_step),
                                    title=title)
