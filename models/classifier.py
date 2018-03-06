import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad


class Classifier(nn.Module):
    """
    """

    def __init__(self, dimensionality, num_classes, cudaEnabled):
        super(Classifier, self).__init__()

        self.conv_1 = nn.Conv3d(1, dimensionality, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(2, 2, 2))
        self.dropout_1 = nn.Dropout(0.3)
        self.conv_2 = nn.Conv3d(dimensionality, 2 * dimensionality, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.dropout_2 = nn.Dropout(0.3)
        self.conv_3 = nn.Conv3d(2 * dimensionality, 4 * dimensionality, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(2, 1, 2))
        self.dropout_3 = nn.Dropout(0.3)
        self.fc_1 = nn.Linear((2 * 2 * 2) * 4 * dimensionality, 128)
        self.dropout_4 = nn.Dropout(0.3)
        self.fc_2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.dimensionality = dimensionality
        self.num_classes = num_classes
        self.cudaEnabled = cudaEnabled
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=(0.5, 0.9))

        if cudaEnabled:
            self.cuda()

    def forward(self, images, train=False):
        out = images.view(-1, 1, 13, 15, 11)

        # Conv 1
        out = self.conv_1(out)
        out = F.leaky_relu(out, inplace=True)
        out = self.dropout_1(out)

        # Conv 2
        out = self.conv_2(out)
        out = F.leaky_relu(out, inplace=True)
        out = self.dropout_2(out)

        # Conv 3
        out = self.conv_3(out)
        out = F.leaky_relu(out, inplace=True)
        out = self.dropout_3(out)

        # Linear and reshape
        out = out.view(out.shape[0], (2 * 2 * 2) * 4 * self.dimensionality)
        out = self.fc_1(out)
        out = self.dropout_4(out)

        out = self.fc_2(out)

        # Softmax
        if not train:
            out = F.softmax(out, dim=1)

        return out

    def train(self, real_images, labels):
        # Housekeeping
        self.zero_grad()

        # Classifier output:
        classifier_output = self.forward(real_images, train=True)

        # Compute classifier loss:
        classifier_loss = F.binary_cross_entropy_with_logits(classifier_output, labels)

        # Compute gradients:
        classifier_loss.backward()

        # Optimize critic's parameters
        self.optimizer.step()

        return classifier_loss
