import nibabel
import nilearn.masking as masking
import numpy as np

from nilearn.input_data import NiftiMasker
from nilearn.image import load_img, resample_img


def invert_preprocessor_scaling(brain_data, preprocessor):
    brain_data_mask = preprocessor.brain_data_mask()

    # Rescale
    preprocessed_shape = tuple([int(float(d) * preprocessor.scale) for d in brain_data_mask.shape])
    rescale_multiplier = float(brain_data_mask.shape[0]) / float(preprocessed_shape[0])

    rescaled_affine = brain_data_mask.get_affine().copy()
    rescaled_affine[:3, :3] *= rescale_multiplier

    # Resample
    brain_img = nibabel.Nifti1Image(brain_data, rescaled_affine)
    rescaled_brain_img = resample_img(brain_img,
                                      target_affine=brain_data_mask.get_affine(),
                                      target_shape=brain_data_mask.shape,
                                      interpolation='continuous')

    # Zero out non-grey matter voxels in rescaled data
    zero_masker = NiftiMasker(mask_img=brain_data_mask, standardize=False)
    zero_masker.fit()
    rescaled_masked_brain_img = zero_masker.inverse_transform(zero_masker.transform(rescaled_brain_img))

    return rescaled_masked_brain_img


def resample_brain_img(brain_img, scale):
    brain_img = nibabel.funcs.squeeze_image(brain_img)

    # Rescale
    rescaled_shape = tuple([int(float(d) * scale) for d in brain_img.shape])
    rescale_multiplier = float(brain_img.shape[0]) / float(rescaled_shape[0])

    rescaled_affine = brain_img.get_affine().copy()
    rescaled_affine[:3, :3] *= rescale_multiplier

    # Resample
    resampled_brain_img = resample_img(brain_img,
                                       target_affine=rescaled_affine,
                                       target_shape=rescaled_shape,
                                       interpolation='continuous')
    return resampled_brain_img


def normalize_brain_img_data(brain_img_data):
    _max = np.max(brain_img_data)
    _min = np.min(brain_img_data)
    return np.array([((brain_img_data - _min) / (_max - _min))])
