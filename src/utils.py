import os
import random
import itertools
import numpy as np
import SimpleITK as sitk

def refrence_free_3D_resample(image, transformation, interpolator,
                              default_value, image_type=None, spacing=None,
                              direction=None):
    """Do resampling without reference.

    Args:
        image: A SimpleITK image.
        transformation: A transformation to be applied to the image.
        interpolator: he interpolator used for image interpolation after
            applying transformation.
        default_value: The default values used for voxel values.
        image_type: The data type used for casting the resampled image.
        size: The size of the image after resampling.
        spacing: The spacing of the image after resampling.
        direction: The direction of the image after resampling.

    Returns:

    """
    extreme_indecies = list(itertools.product(*zip([0, 0, 0], image.GetSize())))
    extreme_points = [image.TransformIndexToPhysicalPoint(idx)
                      for idx in extreme_indecies]
    inv_transform = transformation.GetInverse()
    # Calculate the coordinates of the inversed extream points
    extreme_points_transformed = [inv_transform.TransformPoint(point)
                                  for point in extreme_points]
    min_values = np.array((extreme_points_transformed)).min(axis=0)
    max_values = np.array((extreme_points_transformed)).max(axis=0)

    if spacing is None:
        spacing = image.GetSpacing()
    if direction is None:
        direction = image.GetDirection()
    # Minimal x,y, and coordinates are the new origin.
    origin = min_values.tolist()
    # Compute grid size based on the physical size and spacing.
    size = [int((max_val - min_val) / s)
            for min_val, max_val, s in zip(min_values, max_values, spacing)]
    return sitk.Resample(image, size, transformation, interpolator,
                         outputOrigin=origin,
                         outputSpacing=spacing,
                         outputDirection=direction,
                         defaultPixelValue=default_value,
                         outputPixelType=image_type)

def referenced_3D_resample(image, transformation, interpolator, default_value,
                        image_type, reference=None):
    """Do resampling using a given reference.

    Args:
        image: A simpleITK image.
        transformation: A transformation to be applied to the image.
        interpolator: The interpolator used for image interpolation after
            applying transformation.
        default_value: The default values used for voxel values.
        image_type: The data type used for casting the resampled image.
        reference: The image used as the reference for resampling. If None,
            the image itself is used as the reference.

    Returns:
        sitk.Image: The resampled image.

    """
    if reference is None:
        reference = image
    return sitk.Resample(image, reference, transformation, interpolator,
                         default_value, image_type)


def image_equal(image1: sitk.Image, image2: sitk.Image, type_check=True,
                tolerance=1e-6):
    """Check if two images are equal.

    Data type, size, and content are used for comparison. Two image with the
    L2 distance less than a tolerance value are considered equal.

    Args:
        image1: A SimpleITK image.
        image2: A SimpleITK image.
        type_check: True is data type is used for comparison.
        tolerance: The threshold used for the acceptable deviation between the
            euclidean distance of voxel values between the two images.

    Returns:
        bool: True if two images are equal; otherwise, False.

    """
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


def read_image(path: str):
    """ Read an image.

    Args:
        path: The path to the image file or the folder containing a DICOM image.

    Returns:
        sitk.Image: A SimpleITK Image.

    Raises:
        ValueError: if there exist more than one image series or if there is no
            DICOM file in the provided ``path``.

    """
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
        image: A sitk.Image object.

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
    """Check if the size of the image and mask are equal.

    Args:
        image: A sitk.Image.
        mask: A sitk.Image.

    Raises:
        ValueError: If image and mask are not None and their dimension are
            not the same.
    """
    if (mask is not None) and (image is not None):
        if image.GetSize() != mask.GetSize():
            msg = 'image and mask size should be equal, but ({}) != ({}).'
            raise ValueError(msg.format(', '.format([str(x) for x in
                                                     image.GetSize()]),
                                        ', '.format([str(x) for x in
                                                     mask.GetSize()])))


class Label(object):
    """Label a binary image.

    Each distinct connected component (segment) is assigned a unique label. The
        segment labels start from 1 and are consecutive. The order of
        label assignment is based on the the raster position of the
        segments in the binary image.

    Args:
        fully_connected:
        input_foreground_value:
        output_background_value:
        dtype: The data type for the label map. Options include:
            * sitk.sitkUInt8
            * sitk.sitkUInt16
            * sitk.sitkUInt32
            * sitk.sitkUInt64

    """
    def __init__(self, fully_connected: bool = False,
                 input_foreground_value: int = 1,
                 output_background_value: int = 0,
                 dtype=sitk.sitkUInt8):
        self.input_foreground_value = input_foreground_value
        self.output_background_value = output_background_value
        self.fully_connected = fully_connected
        self.dtype = dtype

    def __call__(self, binary_image):
        mask = sitk.BinaryImageToLabelMap(
            binary_image,
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
