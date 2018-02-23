import argparse
import datetime
import numpy as np
import os
import shutil
import torch

from brainpedia.brainpedia import Brainpedia
from models.classifier import Classifier
from torch.autograd import Variable


parser = argparse.ArgumentParser(description="Evaluate trained generator on real and synthetic data.")
parser.add_argument('generator_state_dict_path', help='path to a file containing the generative model state dict')
parser.add_argument('data_dir', help='the directory containing real fMRI data')
parser.add_argument('augmented_data_dir', help='the directory containing synthetic fMRI data')
parser.add_argument('output_dir', help='the directory to save evaluation results')
args = parser.parse_args()

# ========== HOUSEKEEPING ==========
CUDA = torch.cuda.is_available()
if CUDA:
    print("Using GPU optimizations!")

np.random.seed(1)
torch.manual_seed(1)
if CUDA:
    torch.cuda.manual_seed(1)

# ========== HYPERPARAMETERS ==========
shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)

DOWNSAMPLE_SCALE = 0.25
TRAINING_STEPS = 1000
MODEL_DIMENSIONALITY = 64
BATCH_SIZE = 50
VISUALIZATION_INTERVAL = 10
NOISE_SAMPLE_LENGTH = 128

description_f = open(args.output_dir + '/collection_metadata.txt', 'w')
description_f.write('DATE: {0}\n\n'.format(datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')))
description_f.write('DOWNSAMPLE_SCALE: {0}\n'.format(DOWNSAMPLE_SCALE))
description_f.write('TRAINING_STEPS: {0}\n'.format(TRAINING_STEPS))
description_f.write('BATCH_SIZE: {0}\n'.format(BATCH_SIZE))
description_f.write('MODEL_DIMENSIONALITY: {0}\n'.format(MODEL_DIMENSIONALITY))
description_f.write('VISUALIZATION_INTERVAL: {0}\n'.format(VISUALIZATION_INTERVAL))
description_f.write('NOISE_SAMPLE_LENGTH: {0}\n'.format(NOISE_SAMPLE_LENGTH))
description_f.close()

# ========== DATA ==========
brainpedia = Brainpedia(data_dir=args.data_dir, scale=DOWNSAMPLE_SCALE)
brainpedia_generator = brainpedia.batch_generator(BATCH_SIZE, CUDA)
brain_data_shape, brain_data_tag_shape = brainpedia.sample_shapes()

augmented_brainpedia = Brainpedia(data_dir=args.data_dir, scale=DOWNSAMPLE_SCALE, augmented_data_dir=args.augmented_data_dir)
augmented_brainpedia_generator = augmented_brainpedia.batch_generator(BATCH_SIZE, CUDA)
augmented_brain_data_shape, augmented_brain_data_tag_shape = augmented_brainpedia.sample_shapes()

# ========== MODELS ==========
nn_classifier = Classifier(dimensionality=MODEL_DIMENSIONALITY,
                           num_classes=brain_data_tag_shape[0],
                           cudaEnabled=CUDA)
augmented_nn_classifier = Classifier(dimensionality=MODEL_DIMENSIONALITY,
                                     num_classes=augmented_brain_data_tag_shape[0],
                                     cudaEnabled=CUDA)

# ========== TRAINING ===========
nn_classifier_loss_per_vis_interval = []
augmented_nn_classifier_loss_per_vis_interval = []

running_nn_classifier_loss = 0.0
running_augmented_nn_classifier_loss = 0.0

for training_step in range(1, TRAINING_STEPS + 1):
    print("BATCH: [{0}/{1}]\r".format(training_step % VISUALIZATION_INTERVAL, VISUALIZATION_INTERVAL), end='')

    # Retrieve [REAL] brain image data batch:
    brain_img_data_batch, labels_batch = next(brainpedia_generator)
    brain_img_data_batch = Variable(brain_img_data_batch)
    labels_batch = Variable(labels_batch)

    # Retrieve [REAL + SYNTHETIC] brain image data batch:
    augmented_brain_img_data_batch, augmented_labels_batch = next(augmented_brainpedia_generator)
    augmented_brain_img_data_batch = Variable(augmented_brain_img_data_batch)
    augmented_labels_batch = Variable(augmented_labels_batch)

    # Train classifiers:
    nn_classifier_loss = nn_classifier.train(brain_img_data_batch, labels_batch)
    nn_classifier_augmented_loss = augmented_nn_classifier.train(augmented_brain_img_data_batch, augmented_labels_batch)

    running_nn_classifier_loss += nn_classifier_loss.data[0]
    running_augmented_nn_classifier_loss += nn_classifier_augmented_loss.data[0]

    # Visualization:
    if training_step % VISUALIZATION_INTERVAL == 0:
        print("===== TRAINING STEP {0} / {1} =====".format(training_step, TRAINING_STEPS))
        print("NN CLASSIFIER LOSS:            {0}".format(running_nn_classifier_loss))
        print("NN AUGMENTED CLASSIFIER LOSS:  {0}\n".format(running_augmented_nn_classifier_loss))

        # Loss histories
        nn_classifier_loss_per_vis_interval.append(running_nn_classifier_loss)
        augmented_nn_classifier_loss_per_vis_interval.append(running_augmented_nn_classifier_loss)
        running_nn_classifier_loss = 0.0
        running_augmented_nn_classifier_loss = 0.0

# print(len(augmented_brainpedia.preprocessor.brain_data_tags()))
