import os
import random
import numpy as np
import SimpleITK as sitk
# import torch


def image_equal(image1: sitk.Image, image2: sitk.Image, type_check=True,
                tolerance=1e-6):
    if type_check is True:
        # Check for equality of voxel types
        if image1.GetPixelIDValue() != image2.GetPixelIDValue():
            return False
    if image1.GetDimension() != image2.GetDimension():
        return False
    if image1.GetSize() != image2.GetSize():
        return False
    # Check the equality of image spacings
    if np.linalg.norm(np.array(image1.GetSpacing()) -
                      np.array(image2.GetSpacing())) > tolerance:
        return False
    # Check the equality of image origins
    if np.linalg.norm(np.array(image1.GetOrigin()) -
                      np.array(image2.GetOrigin())) > tolerance:
        return False
    # Check the array equality
    arry1 = sitk.GetArrayFromImage(image1)
    arry2 = sitk.GetArrayFromImage(image2)
    if np.linalg.norm(arry1 - arry2) > tolerance:
        return False
    return True


def read_image(path):
    # Reading DICOM from a directory
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(path)
        if len(series_IDs) > 1:
            msg = ('Only One image is allowed in a directory. There are '
                   f'{len(series_IDs)} Series IDs (images) in {path}.')
            raise ValueError(msg)
        if len(series_IDs) == 0:
            msg = f'There are not dicom files in {path}.'
            raise ValueError(msg)
        series_id = series_IDs[0]
        dicom_names = reader.GetGDCMSeriesFileNames(path, series_id)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
    elif os.path.isfile(path):
        image = sitk.ReadImage(path)
    return image


def get_stats(image):
    """Computes minimum, maximum, sum, mean, variance, and standard deviation of
        an image.
    Args:
        image:
    Returns:
        returns statistical values of type dictionary include keys 'min',
            'max', 'mean', 'std', and 'var'.
    """
    stat_fileter = sitk.StatisticsImageFilter()
    stat_fileter.Execute(image)
    image_info = {}
    image_info.update({'min': stat_fileter.GetMinimum()})
    image_info.update({'max': stat_fileter.GetMaximum()})
    image_info.update({'mean': stat_fileter.GetMean()})
    image_info.update({'std': stat_fileter.GetSigma()})
    image_info.update({'var': stat_fileter.GetVariance()})
    return image_info

def check_dimensions(image, mask):
    """ Check if the size of the image and mask are equal."""
    if (mask is not None) and (image is not None):
        if image.GetSize() != mask.GetSize():
            msg = 'image and mask size should be equal, but ({}) != ({}).'
            raise ValueError(msg.format(', '.format([str(x) for x in
                                                     image.GetSize()]),
                                        ', '.format([str(x) for x in
                                                     mask.GetSize()])))


class Label(object):
    """
    """
    def __init__(self, fully_connected=False, input_foreground_value=1,
                 output_background_value=0, dtype=sitk.sitkUInt8):
        self.input_foreground_value = input_foreground_value
        self.output_background_value = output_background_value
        self.fully_connected = fully_connected
        self.dtype = dtype

    def __call__(self, binary_image):
        mask = sitk.BinaryImageToLabelMap(binary_image,
            fullyConnected=self.fully_connected,
            inputForegroundValue=self.input_foreground_value,
            outputBackgroundValue=self.output_background_value)
        mask = sitk.Cast(mask, pixelID=self.dtype)
        return mask

    def __repr__(self):
        msg = ('{} (fully_connected={}, input_foreground_value={}, '
               'output_background_value={}, dtype={}')
        return msg.format(self.__class__.__name__,
                          self.fully_connected,
                          self.input_foreground_value,
                          self.output_background_value,
                          self.dtype)


