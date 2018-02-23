from neurosynth.analysis import decode

NEUROSYNTH_DECODER = None


def avg_correlation_of_image_to_images_in_brainpedia_with_same_label(image_path, brainpedia, label):
    """
    Computes correlation of the image stored at image_path to all other images
    in brainpedia with the given label.
    """

    global NEUROSYNTH_DECODER

    if NEUROSYNTH_DECODER is None:
        print("Building neurosynth decoder...\r", end='')
        NEUROSYNTH_DECODER = decode.Decoder(mask=brainpedia.preprocessor.brain_data_mask(),
                                            features=brainpedia.all_brain_image_paths())

    # Compute correlation to all images in neurosynth:
    correlations = NEUROSYNTH_DECODER.decode([image_path])

    # Average correlations to images in brainpedia with the same label:
    sum_correlations_same_label = 0
    num_correlations_same_label = 0
    for filename, correlation in correlations.itertuples():
        filename_label = brainpedia.preprocessor.label_for_brain_image(filename)
        if filename_label == label:
            sum_correlations_same_label += correlation
            num_correlations_same_label += 1

    return sum_correlations_same_label / num_correlations_same_label
