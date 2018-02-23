import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad


class Generator(nn.Module):
    """
    """

    def __init__(self, input_size, output_shape, dimensionality, num_classes, conditioning_dimensionality, cudaEnabled):
        super(Generator, self).__init__()

        # Linear and reshape
        self.fc_1 = nn.Linear(input_size + num_classes, (2 * 2 * 2) * 4 * dimensionality)
        self.fc_labels_1 = nn.Linear(num_classes, (2 * 2 * 2) * conditioning_dimensionality)

        # Deconv 1
        self.deconv_1 = nn.ConvTranspose3d(4 * dimensionality + conditioning_dimensionality,
                                           2 * dimensionality,
                                           kernel_size=(4, 4, 4),
                                           stride=(2, 2, 2),
                                           padding=(1, 1, 1))
        self.deconv_1_bn = nn.BatchNorm3d(2 * dimensionality)

        # Deconv 2
        self.fc_labels_2 = nn.Linear(num_classes, (4 * 4 * 4) * conditioning_dimensionality)
        self.deconv_2 = nn.ConvTranspose3d(2 * dimensionality + conditioning_dimensionality,
                                           dimensionality,
                                           kernel_size=(4, 4, 4),
                                           stride=(2, 2, 2),
                                           padding=(1, 1, 1))
        self.deconv_2_bn = nn.BatchNorm3d(dimensionality)

        # Deconv 3
        self.fc_labels_3 = nn.Linear(num_classes, (8 * 8 * 8) * conditioning_dimensionality)
        self.deconv_3 = nn.ConvTranspose3d(dimensionality + conditioning_dimensionality,
                                           1,
                                           kernel_size=(4, 4, 4),
                                           stride=(2, 2, 2),
                                           padding=(1, 1, 1))

        # Linear and reshape
        output_shape_dimensionality = 1
        for s in list(output_shape):
            output_shape_dimensionality *= s
        self.fc_2 = nn.Linear(16 * 16 * 16, output_shape_dimensionality)

        self.input_size = input_size
        self.output_shape = output_shape
        self.dimensionality = dimensionality
        self.num_classes = num_classes
        self.conditioning_dimensionality = conditioning_dimensionality
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.cudaEnabled = cudaEnabled

        if cudaEnabled:
            self.cuda()

    def forward(self, noise, labels):
        # NOTE: deconv output_size = stride*(input_size - 1) + kernel_size - 2*padding

        # Concatenate noise and labels for input layer:
        out = torch.cat((noise, labels), 1)
        out = out.view(-1, 1, self.input_size + self.num_classes)

        # Linear and reshape brain data:
        out = self.fc_1(out)
        out = F.leaky_relu(out, inplace=True)
        out = out.view(-1, 4 * self.dimensionality, 2, 2, 2)

        # Linear and reshape labels:
        labels_out = self.fc_labels_1(labels)
        labels_out = F.tanh(labels_out)
        labels_out = labels_out.view(-1, self.conditioning_dimensionality, out.shape[2], out.shape[3], out.shape[4])
        out = torch.cat((out, labels_out), 1)

        # Deconv 1
        out = self.deconv_1(out)
        out = self.deconv_1_bn(out)
        out = F.leaky_relu(out, inplace=True)

        # Linear and reshape label volume:
        labels_out = self.fc_labels_2(labels)
        labels_out = F.tanh(labels_out)
        labels_out = labels_out.view(-1, self.conditioning_dimensionality, out.shape[2], out.shape[3], out.shape[4])
        out = torch.cat((out, labels_out), 1)

        # Deconv 2
        out = self.deconv_2(out)
        out = self.deconv_2_bn(out)
        out = F.leaky_relu(out, inplace=True)

        # Linear and reshape label volume:
        labels_out = self.fc_labels_3(labels)
        labels_out = F.tanh(labels_out)
        labels_out = labels_out.view(-1, self.conditioning_dimensionality, out.shape[2], out.shape[3], out.shape[4])
        out = torch.cat((out, labels_out), 1)

        # Deconv 3
        out = self.deconv_3(out)
        out = F.tanh(out)

        # Linear and reshape
        out = out.view(-1, 16 * 16 * 16)
        out = self.fc_2(out)

        batch_output_shape = (-1,) + self.output_shape
        out = out.view(batch_output_shape)
        return out

    def train(self, critic_outputs):
        self.zero_grad()
        g_loss = -torch.mean(critic_outputs)
        g_loss.backward()
        self.optimizer.step()
        return g_loss


class Critic(nn.Module):
    """
    """

    def __init__(self, dimensionality, num_classes, conditioning_dimensionality, cudaEnabled):
        super(Critic, self).__init__()

        # Conv 1
        self.fc_labels_1 = nn.Linear(num_classes, (13 * 15 * 11) * conditioning_dimensionality)
        self.conv_1 = nn.Conv3d(1 + conditioning_dimensionality, dimensionality, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(2, 2, 2))

        # Conv 2
        self.fc_labels_2 = nn.Linear(num_classes, (7 * 8 * 6) * conditioning_dimensionality)
        self.conv_2 = nn.Conv3d(dimensionality + conditioning_dimensionality, 2 * dimensionality, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv_2_bn = nn.BatchNorm3d(2 * dimensionality)

        # Conv 3
        self.fc_labels_3 = nn.Linear(num_classes, (3 * 4 * 3) * conditioning_dimensionality)
        self.conv_3 = nn.Conv3d(2 * dimensionality + conditioning_dimensionality, 4 * dimensionality, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(2, 1, 2))
        self.conv_3_bn = nn.BatchNorm3d(4 * dimensionality)

        # Linear and reshape
        self.fc_1 = nn.Linear((2 * 2 * 2) * 4 * dimensionality, 128)

        # Linear
        self.fc_2 = nn.Linear(128 + num_classes, 1)

        self.dimensionality = dimensionality
        self.num_classes = num_classes
        self.conditioning_dimensionality = conditioning_dimensionality
        self.cudaEnabled = cudaEnabled
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=(0.5, 0.9))

        if cudaEnabled:
            self.cuda()

    def forward(self, images, labels):
        # NOTE conv output_size = (input_width − kernel_size + 2*padding)/stride + 1
        out = images.view(-1, 1, 13, 15, 11)

        # Linear and reshape label volume:
        labels_out = self.fc_labels_1(labels)
        labels_out = F.tanh(labels_out)
        labels_out = labels_out.view(-1, self.conditioning_dimensionality, out.shape[2], out.shape[3], out.shape[4])
        out = torch.cat((out, labels_out), 1)

        # Conv 1
        out = self.conv_1(out)
        out = F.leaky_relu(out, inplace=True)

        # Linear and reshape label volume:
        labels_out = self.fc_labels_2(labels)
        labels_out = F.tanh(labels_out)
        labels_out = labels_out.view(-1, self.conditioning_dimensionality, out.shape[2], out.shape[3], out.shape[4])
        out = torch.cat((out, labels_out), 1)

        # Conv 2
        out = self.conv_2(out)
        out = self.conv_2_bn(out)
        out = F.leaky_relu(out, inplace=True)

        # Linear and reshape label volume:
        labels_out = self.fc_labels_3(labels)
        labels_out = F.tanh(labels_out)
        labels_out = labels_out.view(-1, self.conditioning_dimensionality, out.shape[2], out.shape[3], out.shape[4])
        out = torch.cat((out, labels_out), 1)

        # Conv 3
        out = self.conv_3(out)
        out = self.conv_3_bn(out)
        out = F.leaky_relu(out, inplace=True)

        # Linear and reshape
        out = out.view(out.shape[0], (2 * 2 * 2) * 4 * self.dimensionality)
        out = self.fc_1(out)

        # Linear
        out = torch.cat((out, labels), 1)
        out = self.fc_2(out)

        return out

    def train(self, real_images, fake_images, labels, LAMBDA):
        # Housekeeping
        self.zero_grad()

        # Compute gradient penalty:
        random_samples = torch.rand(real_images.size())
        interpolated_random_samples = random_samples * real_images.data.cpu() + ((1 - random_samples) * fake_images.data.cpu())
        interpolated_random_samples = Variable(interpolated_random_samples, requires_grad=True)
        if self.cudaEnabled:
            interpolated_random_samples = interpolated_random_samples.cuda()

        critic_random_sample_output = self(interpolated_random_samples, labels)
        grad_outputs = torch.ones(critic_random_sample_output.size())
        if self.cudaEnabled:
            grad_outputs = grad_outputs.cuda()

        gradients = grad(outputs=critic_random_sample_output,
                         inputs=interpolated_random_samples,
                         grad_outputs=grad_outputs,
                         create_graph=True,
                         retain_graph=True,
                         only_inputs=True)[0]
        if self.cudaEnabled:
            gradients = gradients.cuda()

        gradient_penalty = LAMBDA * ((gradients.norm(2) - 1) ** 2).mean()
        if self.cudaEnabled:
            gradient_penalty = gradient_penalty.cuda()

        # Critic output:
        critic_real_output = self(real_images, labels)
        critic_fake_output = self(fake_images, labels)

        # Compute loss
        critic_loss = -(torch.mean(critic_real_output) - torch.mean(critic_fake_output)) + gradient_penalty
        critic_loss.backward()

        # Optimize critic's parameters
        self.optimizer.step()

        return critic_loss
