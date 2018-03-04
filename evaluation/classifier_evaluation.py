import matplotlib
matplotlib.use('Agg')
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
from sklearn.svm import LinearSVC
from torch.autograd import Variable
from utils.plot import Plot


parser = argparse.ArgumentParser(description="Train classifiers on real and synthetic data.")
parser.add_argument('train_data_dir', help='the directory containing real fMRI data to train on')
parser.add_argument('train_data_dir_cache', help='the directory to use as a cache for the train_data_dir preprocessing')
parser.add_argument('synthetic_data_dir', help='the directory containing synthetic fMRI data to train on')
parser.add_argument('synthetic_data_dir_cache', help='the directory to use as a cache for the synthetic_data_dir preprocessing')
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
TRAINING_STEPS = 200000
MODEL_DIMENSIONALITY = 64
BATCH_SIZE = 16
VISUALIZATION_INTERVAL = 1000

results_f = open(args.output_dir + 'results.txt', 'w')
results_f.write('DATE: {0}\n\n'.format(datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')))
results_f.write('DOWNSAMPLE_SCALE: {0}\n'.format(DOWNSAMPLE_SCALE))
results_f.write('TRAINING_STEPS: {0}\n'.format(TRAINING_STEPS))
results_f.write('BATCH_SIZE: {0}\n'.format(BATCH_SIZE))
results_f.write('MODEL_DIMENSIONALITY: {0}\n'.format(MODEL_DIMENSIONALITY))
results_f.write('VISUALIZATION_INTERVAL: {0}\n\n\n'.format(VISUALIZATION_INTERVAL))

# ========== DATA ==========
# Real data:
brainpedia = Brainpedia(data_dirs=[args.train_data_dir],
                        cache_dir=args.train_data_dir_cache,
                        scale=DOWNSAMPLE_SCALE)
train_brain_data, train_brain_data_tags, test_brain_data, test_brain_data_tags = brainpedia.train_test_split()

# Build real data generator:
train_generator = brainpedia.batch_generator(train_brain_data, train_brain_data_tags, BATCH_SIZE, CUDA)
brain_data_shape, brain_data_tag_shape = brainpedia.sample_shapes()

# Synthetic data:
synthetic_brainpedia = Brainpedia(data_dirs=[args.synthetic_data_dir],
                                  cache_dir=args.synthetic_data_dir_cache,
                                  scale=DOWNSAMPLE_SCALE)
synthetic_all_brain_data, synthetic_all_brain_data_tags = synthetic_brainpedia.all_data()

# Build synthetic data generator:
synthetic_train_generator = synthetic_brainpedia.batch_generator(synthetic_all_brain_data, synthetic_all_brain_data_tags, BATCH_SIZE, CUDA)
synthetic_brain_data_shape, synthetic_brain_data_tag_shape = synthetic_brainpedia.sample_shapes()


# ========== UTILS ==========
def hash_encoded_label(encoded_label):
    # create tuple representing indices that are 1
    indices_list = []
    for i in range(len(encoded_label)):
        if encoded_label[i] == 1:
            indices_list.append(i)
    return tuple(indices_list)


class_ctr = -1
class_encoding_map = {}


def class_from_encoding(encoded_label):
    global class_ctr, class_encoding_map

    label_key = hash_encoded_label(encoded_label)
    if label_key in class_encoding_map:
        return class_encoding_map[label_key]
    else:
        class_ctr += 1
        class_encoding_map[label_key] = class_ctr
        return class_ctr


def n_hot_encode(l, n):
    ret = np.zeros(len(l))
    max_n_indices = l.argsort()[-n:]
    for max_idx in max_n_indices:
        ret[max_idx] = 1.0
    return ret


# ========== SVMs ==========
results_f.write('===================== [SVM] ====================\n')
svm_classifier = LinearSVC(multi_class='ovr', random_state=0)
synthetic_svm_classifier = LinearSVC(multi_class='ovr', random_state=0)

# Flatten data into one dimension:
flattened_train_brain_data = train_brain_data.reshape(train_brain_data.shape[0], -1)
flattened_test_brain_data = test_brain_data.reshape(test_brain_data.shape[0], -1)
flattened_synthetic_all_brain_data = synthetic_all_brain_data.reshape(synthetic_all_brain_data.shape[0], -1)

# Convert tags into class values using custom encoding implementation:
class_encoded_train_brain_data_tags = np.array([class_from_encoding(brain_data_tag) for brain_data_tag in train_brain_data_tags])
class_encoded_test_brain_data_tags = np.array([class_from_encoding(brain_data_tag) for brain_data_tag in test_brain_data_tags])

# Since synthetic data was encoded differently than real data, it must be
# 1) decoded into the raw tags
# 2) encoded using the same method as the real data
# 3) converted into a class number with the same method as the real data
decoded_synthetic_brain_data_tags = [synthetic_brainpedia.decode_label(brain_data_tag) for brain_data_tag in synthetic_all_brain_data_tags]
reencoded_synthetic_brain_data_tags = [brainpedia.encode_label(decoded_brain_data_tag) for decoded_brain_data_tag in decoded_synthetic_brain_data_tags]
class_encoded_synthetic_all_brain_data_tags = np.array([class_from_encoding(brain_data_tag) for brain_data_tag in reencoded_synthetic_brain_data_tags])

# Train:
print("Training SVMs...")
svm_classifier.fit(flattened_train_brain_data, class_encoded_train_brain_data_tags)
synthetic_svm_classifier.fit(flattened_synthetic_all_brain_data, class_encoded_synthetic_all_brain_data_tags)

# Compute accuracy:
print("Evaluating SVMs...")
svm_classifier_score = svm_classifier.score(flattened_test_brain_data, class_encoded_test_brain_data_tags)
synthetic_svm_classifier_score = synthetic_svm_classifier.score(flattened_test_brain_data, class_encoded_test_brain_data_tags)

# Save SVM results:
print("SVM CLASSIFIER TEST ACCURACY: {0:.2f}%".format(100 * svm_classifier_score))
print("SYNTHETIC SVM TEST ACCURACY: {0:.2f}%\n".format(100 * synthetic_svm_classifier_score))
results_f.write("SVM CLASSIFIER TEST ACCURACY: {0:.2f}%\n".format(100 * svm_classifier_score))
results_f.write("SYNTHETIC SVM TEST ACCURACY: {0:.2f}%\n\n".format(100 * synthetic_svm_classifier_score))


# ========== NEURAL NETWORKS ==========
results_f.write('===================== [Neural Networks] ====================\n')
nn_classifier = Classifier(dimensionality=MODEL_DIMENSIONALITY,
                           num_classes=brain_data_tag_shape[0],
                           cudaEnabled=CUDA)
synthetic_nn_classifier = Classifier(dimensionality=MODEL_DIMENSIONALITY,
                                     num_classes=synthetic_brain_data_tag_shape[0],
                                     cudaEnabled=CUDA)


def compute_nn_accuracy(nn_classifier, synthetic_nn_classifier, brain_data, brain_data_tags):
    brain_data = Variable(torch.Tensor(brain_data))
    if CUDA:
        brain_data = brain_data.cuda()

    # Generate predictions on test set:
    nn_classifier_predictions = nn_classifier.forward(brain_data).data.numpy()
    synthetic_nn_classifier_predictions = synthetic_nn_classifier.forward(brain_data).data.numpy()
    random_guesses = np.array(brain_data_tags).copy()
    np.random.shuffle(random_guesses)

    # Count number of correct predictions:
    total_tests = len(brain_data_tags)
    num_nn_classifier_correct = 0
    num_synthetic_nn_classifier_correct = 0
    num_rand_guesses_correct = 0
    num_same_guesses = 0

    for i in range(total_tests):
        true_tags = brainpedia.decode_label(brain_data_tags[i])
        num_tags = len(true_tags)

        # n-hot encode predictions
        nn_predicted_tags = n_hot_encode(nn_classifier_predictions[i], num_tags)
        synthetic_nn_predicted_tags = n_hot_encode(synthetic_nn_classifier_predictions[i], num_tags)
        random_predicted_tags = n_hot_encode(random_guesses[i], num_tags)

        # decode predictions
        nn_predicted_tags = brainpedia.decode_label(nn_predicted_tags)
        synthetic_nn_predicted_tags = synthetic_brainpedia.decode_label(synthetic_nn_predicted_tags)
        random_predicted_tags = brainpedia.decode_label(random_predicted_tags)

        # prepare for comparison
        true_tags = set(true_tags)
        nn_predicted_tags = set(nn_predicted_tags)
        synthetic_nn_predicted_tags = set(synthetic_nn_predicted_tags)
        random_predicted_tags = set(random_predicted_tags)

        # count correct predictions
        num_nn_classifier_correct += (nn_predicted_tags == true_tags)
        num_synthetic_nn_classifier_correct += (synthetic_nn_predicted_tags == true_tags)
        num_rand_guesses_correct += (random_predicted_tags == true_tags)
        num_same_guesses += (nn_predicted_tags == synthetic_nn_predicted_tags)

    # Compute accuracy:
    nn_accuracy = float(num_nn_classifier_correct) / float(total_tests)
    nn_synthetic_accuracy = float(num_synthetic_nn_classifier_correct) / float(total_tests)
    random_accuracy = float(num_rand_guesses_correct) / float(total_tests)
    fraction_same_guesses = float(num_same_guesses) / float(total_tests)

    return nn_accuracy, nn_synthetic_accuracy, random_accuracy, fraction_same_guesses


# ========== TRAINING ===========
nn_classifier_loss_per_vis_interval = []
synthetic_nn_classifier_loss_per_vis_interval = []

nn_classifier_test_acc_per_vis_interval = []
synthetic_nn_classifier_test_acc_per_vis_interval = []
random_classifier_test_acc_per_vis_interval = []

running_nn_classifier_loss = 0.0
running_synthetic_nn_classifier_loss = 0.0

for training_step in range(1, TRAINING_STEPS + 1):
    print("BATCH: [{0}/{1}]\r".format(training_step % VISUALIZATION_INTERVAL, VISUALIZATION_INTERVAL), end='')

    # Retrieve [REAL] brain image data batch:
    brain_img_data_batch, labels_batch = next(train_generator)
    brain_img_data_batch = Variable(brain_img_data_batch)
    labels_batch = Variable(labels_batch)

    # Retrieve [REAL + SYNTHETIC] brain image data batch:
    synthetic_brain_img_data_batch, synthetic_labels_batch = next(synthetic_train_generator)
    synthetic_brain_img_data_batch = Variable(synthetic_brain_img_data_batch)
    synthetic_labels_batch = Variable(synthetic_labels_batch)

    # Train classifiers:
    nn_classifier_loss = nn_classifier.train(brain_img_data_batch, labels_batch)
    nn_classifier_synthetic_loss = synthetic_nn_classifier.train(synthetic_brain_img_data_batch, synthetic_labels_batch)

    running_nn_classifier_loss += nn_classifier_loss.data[0]
    running_synthetic_nn_classifier_loss += nn_classifier_synthetic_loss.data[0]

    # Visualization:
    if training_step % VISUALIZATION_INTERVAL == 0:
        # Compute accuracy stats on test set:
        nn_test_accuracy, nn_test_synthetic_accuracy, random_test_accuracy, fraction_test_same_guesses = compute_nn_accuracy(nn_classifier, synthetic_nn_classifier, test_brain_data, test_brain_data_tags)
        nn_classifier_test_acc_per_vis_interval.append(nn_test_accuracy)
        synthetic_nn_classifier_test_acc_per_vis_interval.append(nn_test_synthetic_accuracy)
        random_classifier_test_acc_per_vis_interval.append(random_test_accuracy)

        # Logging:
        print("===== TRAINING STEP {0} / {1} =====".format(training_step, TRAINING_STEPS))
        print("NN CLASSIFIER LOSS:                        {0}".format(running_nn_classifier_loss))
        print("NN SYNTHETIC CLASSIFIER LOSS:              {0}\n".format(running_synthetic_nn_classifier_loss))

        print("NN CLASSIFIER TEST ACCURACY:               {0:.2f}%".format(100.0 * nn_test_accuracy))
        print("NN SYNTHETIC CLASSIFIER TEST ACCURACY:     {0:.2f}%".format(100.0 * nn_test_synthetic_accuracy))
        print("RANDOM CLASSIFIER TEST ACCURACY:           {0:.2f}%".format(100.0 * random_test_accuracy))
        print("PERCENT TEST SAME GUESSES:                 {0:.2f}%\n".format(100.0 * fraction_test_same_guesses))

        # Loss histories
        nn_classifier_loss_per_vis_interval.append(running_nn_classifier_loss)
        synthetic_nn_classifier_loss_per_vis_interval.append(running_synthetic_nn_classifier_loss)
        running_nn_classifier_loss = 0.0
        running_synthetic_nn_classifier_loss = 0.0

        Plot.plot_histories([nn_classifier_loss_per_vis_interval, synthetic_nn_classifier_loss_per_vis_interval],
                            ['[REAL] Loss', '[REAL+SYNTHETIC] Loss'],
                            "{0}loss_history".format(args.output_dir))
        Plot.plot_histories([nn_classifier_test_acc_per_vis_interval, synthetic_nn_classifier_test_acc_per_vis_interval, random_classifier_test_acc_per_vis_interval],
                            ['[REAL] Test Accuracy', '[REAL+SYNTHETIC] Test Accuracy', '[RANDOM] Test Accuracy'],
                            "{0}accuracy_history".format(args.output_dir))

        # Save model at checkpoint
        torch.save(nn_classifier.state_dict(), "{0}nn_classifier".format(args.output_dir))
        torch.save(synthetic_nn_classifier.state_dict(), "{0}synthetic_nn_classifier".format(args.output_dir))


# Svae final NN classifier results to results_f:
results_f.write("NN CLASSIFIER TEST ACCURACY:               {0:.2f}%\n".format(100.0 * nn_test_accuracy))
results_f.write("NN SYNTHETIC CLASSIFIER TEST ACCURACY:     {0:.2f}%\n".format(100.0 * nn_test_synthetic_accuracy))
results_f.write("RANDOM CLASSIFIER TEST ACCURACY:           {0:.2f}%\n".format(100.0 * random_test_accuracy))
results_f.write("PERCENT TEST SAME GUESSES:                 {0:.2f}%\n".format(100.0 * fraction_test_same_guesses))
results_f.close()
