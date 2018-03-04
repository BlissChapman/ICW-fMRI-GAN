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
TRAINING_STEPS = 50000
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

# Synthetic data:
synthetic_brainpedia = Brainpedia(data_dirs=[args.synthetic_data_dir],
                                  cache_dir=args.synthetic_data_dir_cache,
                                  scale=DOWNSAMPLE_SCALE)
synthetic_brain_data, synthetic_brain_data_tags = synthetic_brainpedia.all_data()

# Since synthetic data was encoded differently than real data, it must be
# 1) decoded into the raw tags
# 2) re-encoded using the same method as the real data
decoded_synthetic_brain_data_tags = [synthetic_brainpedia.decode_label(brain_data_tag) for brain_data_tag in synthetic_brain_data_tags]
synthetic_brain_data_tags = [brainpedia.encode_label(decoded_brain_data_tag) for decoded_brain_data_tag in decoded_synthetic_brain_data_tags]

# Build real data generator:
train_generator = Brainpedia.batch_generator(train_brain_data, train_brain_data_tags, BATCH_SIZE, CUDA)

# Build synthetic data generator:
synthetic_train_generator = Brainpedia.batch_generator(synthetic_brain_data, synthetic_brain_data_tags, BATCH_SIZE, CUDA)

# Build mixed data generator:
mixed_50_brain_data = np.concatenate((train_brain_data, synthetic_brain_data))
mixed_50_brain_data_tags = np.concatenate((train_brain_data_tags, synthetic_brain_data_tags))
mixed_50_train_generator = Brainpedia.batch_generator(mixed_50_brain_data, mixed_50_brain_data_tags, BATCH_SIZE, CUDA)


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
mixed_50_svm_classifier = LinearSVC(multi_class='ovr', random_state=0)

# Flatten data into one dimension:
flattened_train_brain_data = train_brain_data.reshape(train_brain_data.shape[0], -1)
flattened_test_brain_data = test_brain_data.reshape(test_brain_data.shape[0], -1)
flattened_synthetic_brain_data = synthetic_brain_data.reshape(synthetic_brain_data.shape[0], -1)
flattened_mixed_50_brain_data = mixed_50_brain_data.reshape(mixed_50_brain_data.shape[0], -1)

# Convert tags into class values using custom encoding implementation:
class_encoded_train_brain_data_tags = np.array([class_from_encoding(brain_data_tag) for brain_data_tag in train_brain_data_tags])
class_encoded_test_brain_data_tags = np.array([class_from_encoding(brain_data_tag) for brain_data_tag in test_brain_data_tags])
class_encoded_synthetic_brain_data_tags = np.array([class_from_encoding(brain_data_tag) for brain_data_tag in synthetic_brain_data_tags])
class_encoded_mixed_50_brain_data_tags = np.array([class_from_encoding(brain_data_tag) for brain_data_tag in mixed_50_brain_data_tags])

# Train:
print("Training SVMs...")
svm_classifier.fit(flattened_train_brain_data, class_encoded_train_brain_data_tags)
synthetic_svm_classifier.fit(flattened_synthetic_brain_data, class_encoded_synthetic_brain_data_tags)
mixed_50_svm_classifier.fit(flattened_mixed_50_brain_data, class_encoded_mixed_50_brain_data_tags)

# Compute accuracy:
print("Evaluating SVMs...")
svm_classifier_score = svm_classifier.score(flattened_test_brain_data, class_encoded_test_brain_data_tags)
synthetic_svm_classifier_score = synthetic_svm_classifier.score(flattened_test_brain_data, class_encoded_test_brain_data_tags)
mixed_50_svm_classifier_score = mixed_50_svm_classifier.score(flattened_test_brain_data, class_encoded_test_brain_data_tags)

# Save SVM results:
print("SVM CLASSIFIER TEST ACCURACY: {0:.2f}%".format(100 * svm_classifier_score))
print("SYNTHETIC SVM TEST ACCURACY: {0:.2f}%".format(100 * synthetic_svm_classifier_score))
print("MIXED 50 SVM TEST ACCURACY: {0:.2f}%\n".format(100 * mixed_50_svm_classifier_score))
results_f.write("SVM CLASSIFIER TEST ACCURACY: {0:.2f}%\n".format(100 * svm_classifier_score))
results_f.write("SYNTHETIC SVM TEST ACCURACY: {0:.2f}%\n".format(100 * synthetic_svm_classifier_score))
results_f.write("MIXED 50 SVM TEST ACCURACY: {0:.2f}%\n\n".format(100 * mixed_50_svm_classifier_score))


# ========== NEURAL NETWORKS ==========
results_f.write('===================== [Neural Networks] ====================\n')
brain_data_shape, brain_data_tag_shape = brainpedia.sample_shapes()

nn_classifier = Classifier(dimensionality=MODEL_DIMENSIONALITY,
                           num_classes=brain_data_tag_shape[0],
                           cudaEnabled=CUDA)
synthetic_nn_classifier = Classifier(dimensionality=MODEL_DIMENSIONALITY,
                                     num_classes=brain_data_tag_shape[0],
                                     cudaEnabled=CUDA)
mixed_50_nn_classifier = Classifier(dimensionality=MODEL_DIMENSIONALITY,
                                    num_classes=brain_data_tag_shape[0],
                                    cudaEnabled=CUDA)


def compute_accuracies(classifiers, brain_data, brain_data_tags):
    brain_data = Variable(torch.Tensor(brain_data))
    if CUDA:
        brain_data = brain_data.cuda()

    # Generate predictions on test set:
    classifier_predictions = []
    for classifier in classifiers:
        prediction = classifier.forward(brain_data).data.cpu().numpy()
        classifier_predictions.append(prediction)

    # Count number of correct predictions:
    num_classifiers = len(classifiers)
    classifiers_num_correct = np.zeros(num_classifiers)
    total_tests = len(brain_data_tags)

    for i in range(total_tests):
        true_tags = brainpedia.decode_label(brain_data_tags[i])
        num_tags = len(true_tags)

        # n-hot encode predictions
        classifier_predicted_tags = [n_hot_encode(predictions[i], num_tags) for predictions in classifier_predictions]

        # decode predictions
        classifier_predicted_tags = [brainpedia.decode_label(predicted_tags) for predicted_tags in classifier_predicted_tags]

        # prepare for comparison
        true_tags = set(true_tags)
        classifier_predicted_tags = [set(predicted_tags) for predicted_tags in classifier_predicted_tags]

        # count correct predictions
        for i in range(num_classifiers):
            classifiers_num_correct[i] += (classifier_predicted_tags[i] == true_tags)

    # Compute accuracy:
    classifiers_accuracies = [float(num_correct) / float(total_tests) for num_correct in classifiers_num_correct]

    return classifiers_accuracies


# ========== TRAINING ===========
nn_classifier_loss_per_vis_interval = []
synthetic_nn_classifier_loss_per_vis_interval = []
mixed_50_nn_classifier_loss_per_vis_interval = []

nn_classifier_test_acc_per_vis_interval = []
synthetic_nn_classifier_test_acc_per_vis_interval = []
mixed_50_nn_classifier_test_acc_per_vis_interval = []

running_nn_classifier_loss = 0.0
running_synthetic_nn_classifier_loss = 0.0
running_mixed_50_nn_classifier_loss = 0.0

for training_step in range(1, TRAINING_STEPS + 1):
    print("BATCH: [{0}/{1}]\r".format(training_step % VISUALIZATION_INTERVAL, VISUALIZATION_INTERVAL), end='')

    # Retrieve [REAL] brain image data batch:
    brain_img_data_batch, labels_batch = next(train_generator)
    brain_img_data_batch = Variable(brain_img_data_batch)
    labels_batch = Variable(labels_batch)

    # Retrieve [SYNTHETIC] brain image data batch:
    synthetic_brain_img_data_batch, synthetic_labels_batch = next(synthetic_train_generator)
    synthetic_brain_img_data_batch = Variable(synthetic_brain_img_data_batch)
    synthetic_labels_batch = Variable(synthetic_labels_batch)

    # Retrieve [REAL + SYNTHETIC] brain image data batch:
    mixed_50_brain_img_data_batch, mixed_50_labels_batch = next(mixed_50_train_generator)
    mixed_50_brain_img_data_batch = Variable(mixed_50_brain_img_data_batch)
    mixed_50_labels_batch = Variable(mixed_50_labels_batch)

    # Train classifiers:
    nn_classifier_loss = nn_classifier.train(brain_img_data_batch, labels_batch)
    nn_classifier_synthetic_loss = synthetic_nn_classifier.train(synthetic_brain_img_data_batch, synthetic_labels_batch)
    nn_classifier_mixed_50_loss = mixed_50_nn_classifier.train(mixed_50_brain_img_data_batch, mixed_50_labels_batch)

    running_nn_classifier_loss += nn_classifier_loss.data[0]
    running_synthetic_nn_classifier_loss += nn_classifier_synthetic_loss.data[0]
    running_mixed_50_nn_classifier_loss += nn_classifier_mixed_50_loss.data[0]

    # Visualization:
    if training_step % VISUALIZATION_INTERVAL == 0:
        # Compute accuracy stats on test set:
        accuracies = compute_accuracies([nn_classifier, synthetic_nn_classifier, mixed_50_nn_classifier],
                                        test_brain_data,
                                        test_brain_data_tags)
        nn_classifier_test_acc_per_vis_interval.append(accuracies[0])
        synthetic_nn_classifier_test_acc_per_vis_interval.append(accuracies[1])
        mixed_50_nn_classifier_test_acc_per_vis_interval.append(accuracies[2])

        # Logging:
        print("===== TRAINING STEP {0} / {1} =====".format(training_step, TRAINING_STEPS))
        print("NN CLASSIFIER LOSS:                        {0}".format(running_nn_classifier_loss))
        print("NN SYNTHETIC CLASSIFIER LOSS:              {0}".format(running_synthetic_nn_classifier_loss))
        print("NN MIXED 50 CLASSIFIER LOSS:              {0}\n".format(running_mixed_50_nn_classifier_loss))

        print("NN CLASSIFIER TEST ACCURACY:               {0:.2f}%".format(100.0 * accuracies[0]))
        print("NN SYNTHETIC CLASSIFIER TEST ACCURACY:     {0:.2f}%".format(100.0 * accuracies[1]))
        print("NN MIXED 50 CLASSIFIER TEST ACCURACY:      {0:.2f}\n\n%".format(100.0 * accuracies[2]))

        # Loss histories
        nn_classifier_loss_per_vis_interval.append(running_nn_classifier_loss)
        synthetic_nn_classifier_loss_per_vis_interval.append(running_synthetic_nn_classifier_loss)
        mixed_50_nn_classifier_loss_per_vis_interval.append(running_mixed_50_nn_classifier_loss)
        running_nn_classifier_loss = 0.0
        running_synthetic_nn_classifier_loss = 0.0
        running_mixed_50_nn_classifier_loss = 0.0

        Plot.plot_histories([nn_classifier_loss_per_vis_interval, synthetic_nn_classifier_loss_per_vis_interval, mixed_50_nn_classifier_loss_per_vis_interval],
                            ['[REAL] Loss', '[SYNTHETIC] Loss', '[REAL + SYNTHETIC] Loss'],
                            "{0}loss_histories".format(args.output_dir))
        Plot.plot_histories(accuracies,
                            ['[REAL] Test Accuracy', '[SYNTHETIC] Test Accuracy', '[REAL + SYNTHETIC] Test Accuracy'],
                            "{0}accuracy_histories".format(args.output_dir))

        # Save model at checkpoint
        torch.save(nn_classifier.state_dict(), "{0}nn_classifier".format(args.output_dir))
        torch.save(synthetic_nn_classifier.state_dict(), "{0}synthetic_nn_classifier".format(args.output_dir))
        torch.save(mixed_50_nn_classifier.state_dict(), "{0}mixed_50_nn_classifier".format(args.output_dir))


# Svae final NN classifier results to results_f:
results_f.write("NN CLASSIFIER TEST ACCURACY:               {0:.2f}%\n".format(100.0 * nn_test_accuracy))
results_f.write("NN SYNTHETIC CLASSIFIER TEST ACCURACY:     {0:.2f}%\n".format(100.0 * nn_test_synthetic_accuracy))
results_f.write("NN MIXED 50 CLASSIFIER TEST ACCURACY:      {0:.2f}%\n".format(100.0 * nn_test_mixed_50_accuracy))
results_f.write("RANDOM CLASSIFIER TEST ACCURACY:           {0:.2f}%\n".format(100.0 * random_test_accuracy))
results_f.write("PERCENT TEST SAME GUESSES:                 {0:.2f}%\n".format(100.0 * fraction_test_same_guesses))
results_f.close()
