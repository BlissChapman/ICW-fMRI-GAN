from neurosynth.analysis import decode

NEUROSYNTH_DECODER = None


def avg_correlation_of_image_to_images_in_brainpedia_with_same_tags(image_path, brainpedia, tags):
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

    # Average correlations to images in brainpedia with the same labels:
    sum_correlations_same_label = 0
    num_correlations_same_label = 0
    for filename, correlation in correlations.itertuples():
        filename_labels = brainpedia.preprocessor.labels_for_brain_image(filename)

        all_tags_match = True
        for filename_label in filename_labels:
            if filename_label not in tags:
                all_tags_match = False
                break

        if all_tags_match:
            sum_correlations_same_label += correlation
            num_correlations_same_label += 1

    return sum_correlations_same_label / num_correlations_same_label
