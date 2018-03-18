import sys
sys.path.append(".")
import argparse
import datetime
import numpy as np
import os
import shutil
import torch

from brainpedia.brainpedia import Brainpedia
from models.classifier import Classifier
from scipy.stats import entropy
from torch.autograd import Variable


parser = argparse.ArgumentParser(description="Train classifiers on real and synthetic data.")
parser.add_argument('synthetic_data_dir', help='the directory containing synthetic fMRI data')
parser.add_argument('synthetic_data_dir_cache', help='the directory to use as a cache for the preprocessed synthetic fMRI data')
parser.add_argument('classifier_state_dict_path', help='path to a file containing the classifier model state dict')
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

shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)

# ========== HYPERPARAMETERS ==========
DOWNSAMPLE_SCALE = 0.25
MULTI_TAG_LABEL_ENCODING = False
CLASSIFIER_DIMENSIONALITY = 64
BATCH_SIZE = 16

results_f = open(args.output_dir + 'results.txt', 'w')
results_f.write('DATE: {0}\n\n'.format(datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')))
results_f.write('DOWNSAMPLE_SCALE: {0}\n'.format(DOWNSAMPLE_SCALE))
results_f.write('MULTI_TAG_LABEL_ENCODING: {0}\n'.format(MULTI_TAG_LABEL_ENCODING))
results_f.write('CLASSIFIER_DIMENSIONALITY: {0}\n'.format(CLASSIFIER_DIMENSIONALITY))
results_f.write('BATCH_SIZE: {0}\n'.format(BATCH_SIZE))
results_f.write('=====================================================\n\n\n')

# ========== INCEPTION SCORE ==========


def inception_score(path_to_generated_imgs_dir,
                    path_to_generated_imgs_dir_cache,
                    downsample_scale,
                    path_to_classifier,
                    classifier_dimensionality,
                    cuda_enabled,
                    batch_size,
                    splits):
    # Set up data
    generated_brainpedia = Brainpedia(data_dirs=[path_to_generated_imgs_dir],
                                      cache_dir=path_to_generated_imgs_dir_cache,
                                      scale=downsample_scale,
                                      multi_tag_label_encoding=MULTI_TAG_LABEL_ENCODING)
    generated_brain_data_shape, generated_brain_data_tag_shape = generated_brainpedia.sample_shapes()
    all_generated_brain_data, all_generated_brain_data_tags = generated_brainpedia.all_data()
    all_generated_brain_data = Variable(torch.Tensor(all_generated_brain_data))

    if cuda_enabled:
        all_generated_brain_data = all_generated_brain_data.cuda()

    # Load classifier model
    classifier = Classifier(dimensionality=classifier_dimensionality,
                            num_classes=generated_brain_data_tag_shape[0],
                            cudaEnabled=cuda_enabled)
    classifier.load_state_dict(torch.load(path_to_classifier))

    # Compute predictions
    predictions = classifier.forward(all_generated_brain_data).data.cpu().numpy()

    # Now compute the mean kl-div
    N = len(all_generated_brain_data)
    split_scores = []

    for k in range(splits):
        part = predictions[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


inception_score = inception_score(path_to_generated_imgs_dir=args.synthetic_data_dir,
                                  path_to_generated_imgs_dir_cache=args.synthetic_data_dir_cache,
                                  downsample_scale=DOWNSAMPLE_SCALE,
                                  path_to_classifier=args.classifier_state_dict_path,
                                  classifier_dimensionality=CLASSIFIER_DIMENSIONALITY,
                                  cuda_enabled=CUDA,
                                  batch_size=BATCH_SIZE,
                                  splits=10)

inception_score_str = "INCEPTION SCORE: {0}".format(inception_score)
print(inception_score_str)
results_f.write(inception_score_str)
