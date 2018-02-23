import matplotlib.pyplot as plt

from nilearn import plotting


class Plot:

    def plot_sample_brain_data(real_sample_brain_img,
                               synthetic_sample_brain_img,
                               real_sample_correlation,
                               synthetic_sample_correlation,
                               output_file,
                               title=None):

        figure = plt.figure(figsize=(10, 10))
        figure.text(0.5, 0.5, "[REAL] Neurosynth decoder correlation with '{0}'\n  examples in Brainpedia: {1:.4f}".format(title, real_sample_correlation), ha='center')
        figure.text(0.5, 0.05, "[SYNTHETIC] Neurosynth decoder correlation with '{0}'\n  examples in Brainpedia: {1:.4f}".format(title, synthetic_sample_correlation), ha='center')

        real_brain_img_axes = plt.subplot(2, 1, 1)
        synthetic_brain_img_axes = plt.subplot(2, 1, 2)

        plotting.plot_glass_brain(real_sample_brain_img, threshold='auto', title="[REAL] " + title, axes=real_brain_img_axes)
        plotting.plot_glass_brain(synthetic_sample_brain_img, threshold='auto', title="[SYNTHETIC] " + title, axes=synthetic_brain_img_axes)

        figure.savefig(output_file)
        plt.close()

    def plot_loss_histories(critic_losses, generator_losses, output_path):
        plt.figure(figsize=(30, 20))
        plt.plot(critic_losses)
        plt.plot(generator_losses)
        plt.legend(['Critic Loss', 'Generator Loss'])
        plt.savefig(output_path)
        plt.close()
