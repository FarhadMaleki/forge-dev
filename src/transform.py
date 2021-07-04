"""
"""
import random
import os.path
from numbers import Number
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Iterable, Callable

import numpy as np
import SimpleITK as sitk

from src.utils import get_stats
from src.utils import read_image
from src.utils import check_dimensions
from src.utils import referenced_3D_resample
from src.utils import refrence_free_3D_resample
EPSILON = 1E-8
DIMENSION = 3


class Transformation(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __repr__(self):
        pass


def apply_transform(transformation, image: sitk.Image,
                    mask: sitk.Image = None):
    """Apply a SimpleITK transformation on an image and its mask (if provided).

    Args:
        transformation: A SimpleITK transformation.
        image: A simpleITK Image object that is the image to be transformed.
        mask: A simpleITK Image object that is the contour(s) for the image.
    """
    image = transformation.Execute(image)
    if mask is not None:
        mask = transformation.Execute(mask)
    return image, mask


def expand_parameters(param, dimension, name, convert_fn=None):
    """A helper function for making boundary lists.

    The boundary list must be a list of tuples of size 2. For example,
        in a 3D context this should be:
        [(x_min, x_max), (y_min, y_max), (z_min, z_max)]


    Args:
        param: the value used to create the boundary list.
            In a 3D context:
            * If param is a scale s, the boundary list will be
                [(-|s|, |s|), (-|s|, |s|), (-|s|, |s|)], where |s|
                represents the absolute value of s.
            * If param is a list of 3 scalar [a, b, c], the
                boundary list will be
                [(-|a|, |a|), (-|b|, |b|), (-|c|, |c|)], where |x|
                represents the absolute value of x.
            * param can also have the following form:
                [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        dimension: An integer value representing the dimension of images.
        name: The name used for raising errors, if required.
        convert_fn: if not null a conversion function will be applied to
            the final values.
    Returns:
        A numpy array of the following form
            numpy.array([(x_min, x_max), (y_min, y_max), (z_min, z_max)]).
    Raises:
         ValueError: If the parameter does not follow one of the valid forms.
    """
    error_message = (f'{name} must be a numerical scalar, a list of {dimension} numbers,'
                     f' or list of {dimension} tuples each of  2 numbers.')
    if isinstance(param, np.complex):
        raise ValueError(error_message)
    if isinstance(param, Number):
        if convert_fn is not None:
            param = convert_fn(param)
        lb = -np.abs(param)
        ub = np.abs(param)
        param = [(lb, ub), (lb, ub), (lb, ub)]
    elif isinstance(param, (tuple, list, np.array)):
        if convert_fn is not None:
            param = convert_fn(param)
        param = np.array(param)
        if len(param) != dimension:
            raise ValueError(error_message)
        if len(param.flatten()) == dimension:
            param = np.dstack([-np.abs(param), np.abs(param)]).squeeze()
        elif len(param.flatten()) == 2 * dimension:
            for component in param:
                if len(component) != 2:
                    raise ValueError(error_message)
        else:
            raise ValueError(error_message)
    return param


class Identity(Transformation):
    """Apply identity transformation to an image and its mask (if provided).

    This transformation does not change image or its mask and is just for
        convinience.

    Args:
        copy: If True create and return a copy of the image and mask (if
            provided); otherwise, the input image and mask objects will be
            returned.

    """
    def __init__(self, copy: bool = False):
        self.copy = copy

    def __call__(self, image: sitk.Image, mask:sitk.Image = None):
        """Return an image and its mask (if provided) without any changes.

        Args:
            image: An image.
            mask: A the mask for the image.

        Returns:
            sitk.Image: The foreground crop of the input image.
            sitk.Mask: The mask for the foreground cropped image.
        """
        if self.copy:
            image = sitk.Image(image)
            mask = sitk.Image(mask)
        return image, mask

    def __repr__(self):
        msg = ('{} (copy={})')
        return msg.format(self.__class__.__name__, self.copy)


class Pad(Transformation):
    """Pad an image and a mask (if applicable).

    The available options for padding are constant padding,
        mirror padding, and wrap padding.

    Args:
        padding: The padding size. The acceptable values are an positive
            integer and a sequence of 3 positive integers, used for padding
            width (x), height (y), and depth (z) dimensions, respectively.
            If an integer value is provided, it will be considered as the
            padding value for all dimensions.
            Note that padding must be a positive integer.
        method: The method used for padding. Supported options are as follows:
            * constant: Uses a constant value for padding. Default is constant.
            * mirror: Considers the image edge as mirror and use the
                reflection of values inside the image as the values for
                voxels inf the padded area.
            * wrap: Uses a wrap padding.
        constant: The constant value used for padding. This will be
            ignored if method is not constant. The default is 0.
        background_label: The label used in the mask image for the
            padded region. The default is 0.
        pad_lower_bound (bool): if True padding will be applied to the
            lower-boundary of each dimension. Default is True.
        pad_upper_bound (bool): if True padding will be applied to the
            upper-boundary of each dimension. Default is True.
        p (float): The transformation is applied with a probability of p.
            The default value is ``1.0``.

    Raises:
        ValueError: If padding value or method is not valid.
    """

    def __init__(self, padding: Union[int, Tuple[int, int, int]],
                 method: str = 'constant',
                 constant: Union[int, float] = 0,
                 background_label: int = 0,
                 pad_lower_bound=True, pad_upper_bound=True,
                 p=1.0):
        assert p > 0
        self.constant = constant
        self.method = method.lower()
        self.pad_lower_bound = pad_lower_bound
        self.pad_upper_bound = pad_upper_bound
        self.background_label = background_label
        self.pad_lower_bound = pad_lower_bound
        self.pad_upper_bound = pad_upper_bound
        self.p = p
        if isinstance(padding, int):
            if padding < 0:
                raise ValueError('padding must be non-negative.')
            padding = [padding for _ in range(DIMENSION)]

        if not isinstance(padding, (tuple, list)):
            msg = 'padding must be an integer number, a tuple, or a list.'
            raise ValueError(msg)

        self.padding = list(padding)
        if self.method == 'constant':
            self.filter = sitk.ConstantPadImageFilter()
        elif self.method == 'mirror':
            self.filter = sitk.MirrorPadImageFilter()
        elif self.method == 'wrap':
            self.filter = sitk.WrapPadImageFilter()
        else:
            msg = 'Valid values for method are constant, mirror, and wrap.'
            raise ValueError(msg)
        if self.pad_lower_bound is True:
            self.filter.SetPadLowerBound(self.padding)
        if self.pad_upper_bound is True:
            self.filter.SetPadUpperBound(self.padding)

    def __call__(self, image: sitk.Image, mask: sitk.Image = None):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            if self.method == 'constant':
                constant = self.constant
                if constant is None:
                    INSIDE_VALUE = 1
                    OUTSIDE_VALUE = 0
                    bin_image = sitk.OtsuThreshold(image,
                                                   INSIDE_VALUE,
                                                   OUTSIDE_VALUE)
                    # Get the median background intensity for padding
                    stats_filter = sitk.LabelIntensityStatisticsImageFilter()
                    stats_filter.SetBackgroundValue(OUTSIDE_VALUE)
                    stats_filter.Execute(bin_image, image)
                    constant = stats_filter.GetMedian(INSIDE_VALUE)
                    # Cast constant type to image type
                self.filter.SetConstant(constant)
            image = self.filter.Execute(image)
            if mask is not None:
                filter = sitk.ConstantPadImageFilter()
                if self.pad_lower_bound is True:
                    filter.SetPadLowerBound(self.padding)
                if self.pad_upper_bound is True:
                    filter.SetPadUpperBound(self.padding)
                filter.SetConstant(self.background_label)
                mask = filter.Execute(mask)
        return image, mask

    def __repr__(self):
        msg = ('{} (padding={}, method={}, constant={}, background_label={}, '
               'pad_lower_bound={}, pad_upper_bound={}, p={})')
        return msg.format(self.__class__.__name__,
                          self.padding,
                          self.method,
                          self.constant,
                          self.background_label,
                          self.pad_lower_bound,
                          self.pad_upper_bound,
                          self.p)


class ForegroundMask(Transformation):
    """Create a mask for the foreground part of an image.

    The foreground is detected through Otsu thresholding method.

    Args:
        background: The relationship of background and the Otsu threshold.
            For example, if  background is '<', after applying Otsu method,
            all image voxels less than the Otsu threshold will be considered as
            background. Acceptable values are '<', '<=', '>', and '>='.
        bins: the number of bins used for Otsu thresholding. Default is 128.

    """

    def __init__(self, background: str = '<', bins=128):
        self.bins = bins
        self.background = background
        self.filter = sitk.OtsuThresholdImageFilter()
        self.filter.SetInsideValue(1)
        self.filter.SetOutsideValue(0)
        self.filter.SetNumberOfHistogramBins(bins)
        self.filter.SetMaskOutput(False)

    def __call__(self, image: sitk.Image) -> sitk.Image:
        """Create a foreground mask.

        Args:
            image: An SimpleITK image.

        Returns:
            sitk.Image: The mask created using Otsu thresholding.

        """

        self.filter.Execute(image)
        threshold = self.filter.GetThreshold()
        image_array = sitk.GetArrayFromImage(image)
        mask_array = np.ones_like(image_array, dtype=np.uint8)
        if self.background == '<':
            mask_array[image_array < threshold] = 0
        elif self.background == '<=':
            mask_array[image_array <= threshold] = 0
        elif self.background == '>':
            mask_array[image_array > threshold] = 0
        elif self.background == '>=':
            mask_array[image_array >= threshold] = 0
        else:
            msg = 'Valid background calculation values are:  <, <=, >, and >='
            raise ValueError(msg)
        mask = sitk.GetImageFromArray(mask_array)
        mask.CopyInformation(image)
        return image, mask

    def __repr__(self):
        msg = '{}, (background {} Otsu threshold, bins={})'
        return msg.format(self.__class__.__name__, self.background, self.bins)


class ForegroundCrop(Transformation):
    """A transformation for Cropping foreground of an image.

    Args:
        background: The relationship of background and the Otsu threshold.
            For example, if  background is '<', after applying Otsu method,
            all image voxels less than the Otsu threshold will be considered
            as background. Acceptable values are '<', '<=', '>', and '>='.
        bins: the number of bins used for Otsu thresholding. Default is 128.
    """
    def __init__(self, background: str = '<', bins=128):
        self.background = background
        self.bins = bins

    def __call__(self, image: sitk.Image, mask: sitk.Image = None):
        """Crop foreground of an image and its mask (if provided).

        The foreground is selected using the Otsu method.

        Args:
            image: An image.
            mask: A the mask for the image.

        Returns:
            sitk.Image: The foreground crop of the input image.
            sitk.Mask: The mask for the foreground cropped image.
        """
        foreground = ForegroundMask(self.background, self.bins)
        _, foreground_mask = foreground(image)
        lbl_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        lbl_shape_filter.Execute(foreground_mask)
        FORGROUDN_LABEL = 1
        bbox = lbl_shape_filter.GetBoundingBox(FORGROUDN_LABEL)
        # The first half of entries represents the starting index
        # and the second half of entries represents the size
        mid = len(bbox) // 2
        bbox_index = bbox[0: mid]
        bbox_size = bbox[mid:]
        image = sitk.RegionOfInterest(image, bbox_size, bbox_index)
        if mask is not None:
            mask = sitk.RegionOfInterest(mask, bbox_size, bbox_index)
        return image, mask

    def __repr__(self):
        msg = '{} (background={}, bins={})'
        return msg.format(self.__class__.__name__, self.background, self.bins)


class Rotation(Transformation):
    """Rotation transformation applied to an image and its mask (if provided).

    Args:
        angles: The rotation angles in degrees. This should be a tuple of
            lengtyh 3.
        interpolator(sitk interpolator): The interpolator used by the
            transformation. The defult is ``sitk.sitkBSpline``.
        image_background: The value used as the default for image voxels.
        mask_background: The value used as the default for mask voxels.
        reference: The refrence grid used for resampling during the
            transformation. The default is None, meaning that the imge (mask)
            itself is used for resampling.
    """
    DIMENSION = 3
    def __init__(self, angles: Tuple[int, int, int],
                 interpolator=sitk.sitkBSpline, image_background=0,
                 mask_background=0, reference=None):
        self.angles = angles
        self.interpolator = interpolator
        self.image_background = image_background
        self.mask_background = mask_background
        self.reference = reference
        self.rotation = Affine(self.angles, translation=(0, 0, 0),
                               scales=[1] * DIMENSION,
                               interpolator=self.interpolator,
                               image_background=self.image_background,
                               mask_background=self.mask_background,
                               reference=self.reference)

    def __call__(self, image, mask=None):
        return self.rotation(image, mask)

    def __repr__(self):
        msg = ('{} (angles={}, interpolator={}, image_background={}, '
               'mask_background={}, reference={}')

        return msg.format(self.__class__.__name__,
                          str(self.angles),
                          self.interpolator,
                          self.image_background,
                          self.mask_background,
                          self.reference)


class Affine(Transformation):
    """The affine transformation applied to an image and its mask (if provided).
    
    Args:
        angles: The rotation angles in degrees. This should be a tuple of 
            lengtyh 3.
        translation: The translation components. This should be a tuple of 
            lengtyh 3. The default is None, representing no translation.
        scale: A scale factor. The default is None, representing no scaling.
        interpolator(sitk interpolator): The interpolator used by the 
            transformation. The defult is ``sitk.sitkBSpline``.
        image_background: The value used as the default for image voxels.
        mask_background: The value used as the default for mask voxels.
        reference: The refrence grid used for resampling during the
            transformation. The default is None, meaning that the imge (mask)
            itself is used for resampling.
    """
    DIMENSION = 3

    def __init__(self, angles: Tuple[int, int, int],
                 translation: Tuple = None, scales=None,
                 interpolator=sitk.sitkBSpline, image_background=0,
                 mask_background=0, image_type=sitk.sitkInt16,
                 mask_type=sitk.sitkUInt8, reference=None,
                 spacing=None, direction=None, reshape=True):
        if angles is None:
            angles = (0, ) * DIMENSION
        if  len(angles) != DIMENSION:
            raise ValueError(f'angles must be of length {DIMENSION}')
        if translation is not None and len(translation) != DIMENSION:
            raise ValueError(f'translation must be of length {DIMENSION}')
        self.reference = reference
        self.angles = [-a for a in np.radians(angles)]
        if translation is not None:
            self.translation = [-t for t in translation]
        else:
            self.translation = None
        self.image_type = image_type
        self.mask_type = mask_type
        if scales is not None:
            self.scale = [1./x for x in scales]
        else:
            self.scale = None
        self.interpolator = interpolator
        self.image_background = image_background
        self.mask_background = mask_background
        self.spacing = spacing
        self.direction = direction
        self.reshape = reshape

    def __call__(self, image, mask=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        X, Y, Z = 0, 1, 2
        affine = sitk.AffineTransform(Affine.DIMENSION)
        center = np.array(image.GetSize()) / 2
        center = image.TransformContinuousIndexToPhysicalPoint(center.tolist())
        affine.SetCenter(center)
        affine.Rotate(Y, Z, angle=self.angles[X])
        affine.Rotate(X, Z, angle=self.angles[Y])
        affine.Rotate(X, Y, angle=self.angles[Z])
        if self.scale is not None:
            affine.Scale(self.scale)
        if self.translation is not None:
            affine.Translate(self.translation)
        # Set image_type
        if self.image_type is None:
            image_type = image.PixelIDValueType()
        else:
            image_type = self.image_type
        # Set mask_type
        if self.mask_type is None:
            mask_type = mask.PixelIDValueType()
        else:
            mask_type = self.mask_type
        # Set spacing
        if self.spacing is None:
            spacing = image.GetSpacing()
        else:
            spacing = self.spacing
        # Set direction
        if self.direction is None:
            direction = image.GetDirection()
        else:
            direction = self.direction
        if self.reshape is True:
            image = refrence_free_3D_resample(
                image,
                affine,
                interpolator=self.interpolator,
                default_value=self.image_background,
                image_type=image_type,
                spacing=spacing,
                direction=direction)

            if mask is not None:
                mask = refrence_free_3D_resample(
                    mask,
                    affine,
                    interpolator=sitk.sitkNearestNeighbor,
                    default_value=self.mask_background,
                    image_type=mask_type,
                    spacing=spacing,
                    direction=direction)
        else:
            image = referenced_3D_resample(
                image, affine, self.interpolator,
                default_value=self.image_background,
                image_type=image_type)
            if mask is not None:
                mask = referenced_3D_resample(
                    mask, affine, sitk.sitkNearestNeighbor,
                    default_value=self.mask_background,
                    image_type=mask_type)
        return image, mask

    def __repr__(self):
        msg = ('{} (angles={}, translation={}, scale={}, interpolator={},'
               'image_background={}, mask_background={}, reference={}')

        return msg.format(self.__class__.__name__,
                          str(self.angles),
                          str(self.translation),
                          self.scale,
                          self.interpolator,
                          self.image_background,
                          self.mask_background,
                          self.reference)


class RandomAffine(Transformation):
    """A random affine applied to an image and its mask (if provided).

    Args:
        angles: The interval for rotation angles in degrees. In a 3D context:
            * If angles is a scalar s, the boundary list will be
                [(-|s|, |s|), (-|s|, |s|), (-|s|, |s|)], where |s| represents
                the absolute value of s.
            * If angles is a list of 3 scalars [a, b, c], the boundary list
                will be
                [(-|a|, |a|), (-|b|, |b|), (-|c|, |c|)], where |a| represents
                the absolute value of a.
            * angles can also have the following form:
                [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
            The default value is None representing no rotation.
        translation: The interval for translation components. Similar
            format to that of angels are allowed. The default is None,
            representing no translation.
        scale: A scale factor. The default is None, representing no scaling. If
            it is not None, it should have the following format:
                [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        interpolator(sitk interpolator): The interpolator used by the
            transformation. The defult is ``sitk.sitkBSpline``.
        image_background: The value used as the default for image voxels.
        mask_background: The value used as the default for mask voxels.
        reference: The refrence grid used for resampling during the
            transformation. The default is None, meaning that the imge (mask)
            itself is used for resampling.
        p (float): The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """
    DIMENSION = 3

    def __init__(self, angles,
                 translation=None, scales=None,
                 interpolator=sitk.sitkBSpline, image_background=0,
                 mask_background=0, image_type=sitk.sitkInt16,
                 mask_type=sitk.sitkUInt8, reference=None,
                 spacing=None, direction=None, reshape=True, p: float = 1):
        assert p > 0
        self.angles = angles
        self.translation = translation
        self.scales = scales
        self.interpolator = interpolator
        self.image_background = image_background
        self.mask_background = mask_background
        self.image_type = image_type
        self.mask_type = mask_type
        self.reference = reference
        self.spacing = spacing
        self.direction = direction
        self.reshape = reshape
        self.p = p

    def __call__(self, image, mask=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.
        """
        check_dimensions(image, mask)
        if random.random() <= self.p:
            angles = self.angles
            if angles is not None:
                angle_intervals = expand_parameters(angles, DIMENSION, 'angles',
                                                    convert_fn=None)
                angles = [np.random.randint(theta_min, high=theta_max + 1)
                          for theta_min, theta_max in angle_intervals]
            translation = self.translation
            if translation is not None:
                translation_intervals = expand_parameters(translation,
                                                          DIMENSION,
                                                          'translation',
                                                          convert_fn=None)
                translation = [np.random.uniform(low=t_min, high=t_max)
                               for t_min, t_max in translation_intervals]
            scales = self.scales
            if scales is not None:
                scales = [np.random.uniform(s_min, high=s_max)
                          for s_min, s_max in scales]
            self.tsfm = Affine(angles=angles, translation=translation,
                               scales=scales, interpolator=self.interpolator,
                               image_background=self.image_background,
                               mask_background=self.mask_background,
                               image_type=self.image_type,
                               mask_type=self.mask_type,
                               reference=self.reference, spacing=self.spacing,
                               direction=self.direction, reshape=self.reshape)
            image, mask = self.tsfm(image, mask)
        return image, mask

    def __repr__(self):
        representation = str(self.tsfm)
        msg = ('{} (angles={}, translation={}, scales={}, interpolator={}, '
               'image_background={}, mask_background={}, image_type={},'
               'mask_type={}, reference={}, spacing={}, direction={}, '
               'reshape={}, p={}')
        return '{} (angles={}, p={}'.format(self.__class__.__name__,
                                   self.angles,
                                   self.translation,
                                   self.scales,
                                   self.interpolator,
                                   self.image_background,
                                   self.mask_background,
                                   self.image_type,
                                   self.mask_type,
                                   self.reference,
                                   self.spacing,
                                   self.direction,
                                   self.reshape,
                                   self.p)



class RandomRotation(Transformation):
    """A random rotation applied to an image and its mask (if provided).

    Args:
        angles: The interval for rotation angles in degrees. In a 3D context:
            * If angles is a scalar s, the boundary list will be
                [(-|s|, |s|), (-|s|, |s|), (-|s|, |s|)], where |s| represents
                the absolute value of s.
            * If angles is a list of 3 scalars [a, b, c], the boundary list
                will be
                [(-|a|, |a|), (-|b|, |b|), (-|c|, |c|)], where |a| represents
                the absolute value of a.
            * angles can also have the following form:
                [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
            The default value is None representing no rotation.
        interpolator(sitk interpolator): The interpolator used by the
            transformation. The defult is ``sitk.sitkBSpline``.
        image_background: The value used as the default for image voxels.
        mask_background: The value used as the default for mask voxels.
        reference: The refrence grid used for resampling during the
            transformation. The default is None, meaning that the imge (mask)
            itself is used for resampling.
        p (float): The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """
    DIMENSION = 3

    def __init__(self, angles,
                 translation=None, scales=None,
                 interpolator=sitk.sitkBSpline, image_background=0,
                 mask_background=0, image_type=sitk.sitkInt16,
                 mask_type=sitk.sitkUInt8, reference=None,
                 spacing=None, direction=None, reshape=True, p: float = 1):
        assert p > 0
        self.angles = angles
        self.interpolator = interpolator
        self.image_background = image_background
        self.mask_background = mask_background
        self.image_type = image_type
        self.mask_type = mask_type
        self.reference = reference
        self.spacing = spacing
        self.direction = direction
        self.reshape = reshape
        self.p = p

    def __call__(self, image, mask=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.
        """
        check_dimensions(image, mask)
        if random.random() <= self.p:
            angles = self.angles
            if angles is not None:
                angle_intervals = expand_parameters(angles, DIMENSION, 'angles',
                                                    convert_fn=None)
                angles = [np.random.randint(theta_min, high=theta_max + 1)
                          for theta_min, theta_max in angle_intervals]

            self.tsfm = Affine(angles=angles, translation=None,
                               scales=None, interpolator=self.interpolator,
                               image_background=self.image_background,
                               mask_background=self.mask_background,
                               image_type=self.image_type,
                               mask_type=self.mask_type,
                               reference=self.reference, spacing=self.spacing,
                               direction=self.direction, reshape=self.reshape)
            image, mask = self.tsfm(image, mask)
        return image, mask

    def __repr__(self):
        representation = str(self.tsfm)
        msg = ('{} (angles={}, interpolator={}, '
               'image_background={}, mask_background={}, image_type={},'
               'mask_type={}, reference={}, spacing={}, direction={}, '
               'reshape={}, p={}')
        return '{} (angles={}, p={}'.format(self.__class__.__name__,
                                   self.angles,
                                   self.interpolator,
                                   self.image_background,
                                   self.mask_background,
                                   self.image_type,
                                   self.mask_type,
                                   self.reference,
                                   self.spacing,
                                   self.direction,
                                   self.reshape,
                                   self.p)


class Flip(Transformation):
    """Flips an image and it's mask (if provided) across user specified axes.

    Args:
        axes (List): A list of boolean values for each dimension.
            Default value is [False, True, False] representing horizontal flip.
        p (float): The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """

    def __init__(self, axes: List = [True, False, False], p=1.0):
        assert p > 0
        self.p = p
        self.axes = axes
        self.filter = sitk.FlipImageFilter()
        self.filter.SetFlipAxes(axes)

    def __call__(self, image: sitk.Image, mask: sitk.Image = None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        msg = 'The image dimension should be equal to the length of axes.'
        if len(self.axes) != image.GetDimension():
            raise ValueError(msg)
        if random.random() <= self.p:
            image, mask = apply_transform(self.filter, image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (axes={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.axes,
                          self.p)


class RandomFlipX(Transformation):
    """Flips an image and it's mask (if provided) across x-axis (width).

    Args:
        p (float): The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """

    def __init__(self, p: float = 1.0):
        assert p > 0
        self.p = p
        axes = [True, False, False]
        self.filter = sitk.FlipImageFilter()
        self.filter.SetFlipAxes(axes)

    def __call__(self, image, mask=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        Raises:
            ValueError: If image dimension is not equal 3.
        """
        check_dimensions(image, mask)
        msg = f'The image dimension should be {DIMENSION}.'
        if image.GetDimension() != DIMENSION:
            raise ValueError(msg)
        if random.random() <= self.p:
            image, mask = apply_transform(self.filter, image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (p={})'
        return msg.format(self.__class__.__name__, self.p)


class RandomFlipY(Transformation):
    """Flips an image and it's mask (if provided) across y-axis (height).

    Args:
        p (float): The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """

    def __init__(self, p: float = 1.0):
        assert p > 0
        self.p = p
        axes = [False, True, False]
        self.filter = sitk.FlipImageFilter()
        self.filter.SetFlipAxes(axes)

    def __call__(self, image, mask=None, *args, **kwargs):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        Raises:
            ValueError: If image dimension is not equal 3.
        """
        check_dimensions(image, mask)
        msg = f'The image dimension should be {DIMENSION}.'
        if image.GetDimension() != DIMENSION:
            raise ValueError(msg)
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, mask = apply_transform(self.filter, image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (axes={}, p={})'
        return msg.format(self.__class__.__name__, self.p)


class RandomFlipZ(Transformation):
    """Flips an image and its mask (if provided) across z-axis (depth).

    Args:
        p (float): The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """

    def __init__(self, p: float = 1.0):
        assert p > 0
        self.p = p
        self.axes = [False, False, True]
        self.filter = sitk.FlipImageFilter()
        self.filter.SetFlipAxes(self.axes)

    def __call__(self, image: sitk.Image, mask: sitk.Image = None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        Raises:
            ValueError: If image dimension is not equal 3.
        """
        check_dimensions(image, mask)
        msg = f'The image dimension should be {DIMENSION}.'
        if image.GetDimension() != DIMENSION:
            raise ValueError(msg)
        if random.random() <= self.p:
            image, mask = apply_transform(self.filter, image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (p={})'
        return msg.format(self.__class__.__name__, self.p)


class Crop(Transformation):
    """Crop image based on given coordinates.

    Args:
        size: A tuple representing the size of the region to be extracted.
            The coordinate is (x, y, z) order, i.e. (width, height, depth).
        index: The starting index of the image to be extracted.
            The default value is [0, 0, 0]. The coordinate is (x, y, z) order,
            i.e. (width, height, depth).
        p (float): The transformation is applied with a probability of p.
            The default value is 1.0.
    """

    def __init__(self,
                 size: Tuple[int, int, int],
                 index: Tuple[int, int, int] = (0, 0, 0),
                 p: float = 1.0):
        assert p > 0
        self.size = size
        self.index = index
        self.p = p
        self.filter = sitk.RegionOfInterestImageFilter()
        self.filter.SetIndex(index)
        self.filter.SetSize(size)

    def __call__(self, image: sitk.Image, mask: sitk.Image = None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        Raises:
            ValueError: If size + index is greater than the image size.

        """
        check_dimensions(image, mask)
        end_coordinate = np.array(self.index) + np.array(self.size)
        if all(end_coordinate > np.array(image.GetSize())):
            raise ValueError('size + index cannot be greater than image size')
        if random.random() <= self.p:
            image, mask = apply_transform(self.filter, image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (size={}, index={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.size,
                          self.index,
                          self.p)


class RandomCrop(Transformation):
    """Crop an image and its mask (if provided) randomly but with a fixed size.

        Args:
            size: The size of the region to be extracted.
            p: The transformation is applied with a probability of p.
                The default value is ``1.0``.
        """

    def __init__(self, size: Tuple[int, int, int], p: float = 1.0):
        assert p > 0
        self.size = size
        self.p = p

    def __call__(self, image, mask=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        Raises:
            ValueError: If the crop size is large than the image size for an
                axis.
        """
        check_dimensions(image, mask)
        if random.random() <= self.p:
            diff_width = image.GetWidth() - self.size[0]
            if diff_width < 0:
                msg = 'Copped region width cannot be larger than image width.'
                raise ValueError(msg)
            diff_height = image.GetHeight() - self.size[1]
            if diff_height < 0:
                msg = 'Copped region height cannot be larger than image height.'
                raise ValueError(msg)
            diff_depth = image.GetDepth() - self.size[2]
            if diff_depth < 0:
                msg = 'Copped region depth cannot be larger than image depth.'
                raise ValueError(msg)
            index = tuple([np.random.randint(0, s + 1)
                           for s in [diff_width, diff_height, diff_depth]])
            tsfm = Crop(size=self.size, index=index, p=1.0)
            image, mask = tsfm(image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (size={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.size,
                          self.p)


class CenterCrop(Transformation):
    """Crop an image and its mask (if provided) from the center.

    Args:
        size: An integer or a tuple of 3 integer numbers to be used as
            the size of the crop across each dimension of an image.
            If an integer value is provided, it is considered as a tuple of
            3 elements, all equal to the input image.
            Note: The dimension is in (x, y, z) order, which is
                (width, height, depth).
        p (float): The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """
    def __init__(self, size: Union[int, Tuple[int, int, int]], p: float = 1.0):
        assert p > 0
        assert np.all(np.asarray(size) > 0), 'size should be greater than 0'
        self.p = p
        self.output_size = np.array(size, dtype='uint')
        self.filter = sitk.RegionOfInterestImageFilter()

    def __call__(self, image: sitk.Image, mask: sitk.Image = None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if random.random() <= self.p:
            if len(self.output_size) != image.GetDimension():
                msg = 'length of size should be the same as image dimension'
                raise ValueError(msg)
            image_size = np.array(image.GetSize())
            if not np.all(self.output_size <= image_size):
                raise ValueError('size cannot be larger than image size')
            index = np.array([(s - o)//2
                              for s, o in zip(image_size, self.output_size)],
                             dtype='uint')
            self.filter.SetSize(self.output_size.tolist())
            self.filter.SetIndex(index.tolist())
            image, mask = apply_transform(self.filter, image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (size={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.output_size,
                          self.p)


class RandomSegmentSafeCrop(Transformation):
    """Crop an image and a mask randomly while keeping some reqions of interest.

    Regions of interest can be defined using a mask. Unlike many other
        transformation, this transformation requires a mask.
        This transformation falls back to random crop when there region of
        interest is empty. If crop size is less than segment size (in any
        dimension), a random crop is made from the segment.

    Args:
        crop_size: Minimum size of the cropped region. Like all parameters in
            this package, the dimension order is (x, y, z), i.e. (width, height,
            depth).
        include: Sequence of unique ids for each
            interested segment in the image. Default is `[1]`.
        p: The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """
    def __init__(self, crop_size: Tuple, include: List = [1], p: float = 1.0):
        assert p > 0
        self.crop_size = np.array(crop_size)
        self.include = include
        self.p = p
        assert isinstance(self.include, (tuple, list, np.array))

    def __call__(self, image, mask):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        if mask is None or image is None:
            raise ValueError('SegmentSafeCrop requires an image and a mask.')
        check_dimensions(image, mask)
        image_size = np.array(image.GetSize())
        msg = 'crop_size must be less than or equal to image size.'
        if not np.all(self.crop_size <= image_size):
            raise ValueError(msg)
        if random.random() <= self.p:
            # Create a binary mask
            mask_arr = sitk.GetArrayFromImage(mask)
            mask_array = np.zeros_like(mask_arr)
            mask_array[np.isin(mask_arr, self.include)] = 1
            binary_mask = sitk.GetImageFromArray(mask_array)
            binary_mask.CopyInformation(mask)
            #If there is no segment, this transformation fall back to a
            # random crop
            if mask_array.sum() == 0:
                tsfm = RandomCrop(self.crop_size.tolist(), p=1.0)
                return tsfm(image, mask=mask)
            lsif = sitk.LabelShapeStatisticsImageFilter()
            lsif.Execute(binary_mask)
            bbox = np.array(lsif.GetBoundingBox(1))
            mid = len(bbox) // 2
            seg_index = np.array(bbox[:mid])
            seg_size = np.array(bbox[mid:])
            #
            crop_size = self.crop_size
            rand = np.random.default_rng()
            crop_index = []
            for i, (seg_length, crop_length) in enumerate(zip(seg_size,
                                                              crop_size)):
                if seg_length < crop_length:
                    low = max(0, seg_index[i] + seg_length - crop_length)
                    high = min(seg_index[i], image_size[i] - crop_length)
                    crop_index.append(rand.integers(low, high, endpoint=True))
                else:
                    flexibility = seg_length - crop_length
                    crop_index.append(rand.integers(seg_index[i],
                                                    seg_index[i] + flexibility,
                                                    endpoint=True))
            tsfm = sitk.RegionOfInterestImageFilter()
            tsfm.SetSize(crop_size.astype(int).tolist())
            tsfm.SetIndex(np.array(crop_index).astype(int).tolist())
            image, mask = apply_transform(tsfm, image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (min size={}, interesting segments={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.crop_size,
                          self.include,
                          self.p)


class SegmentCrop(Transformation):
    """Crop a region of interest from an image and its mask.

    Regions of interest is defined using a mask. Unlike many other
        transformation, this transformation requires a mask.

    Args:
        include: Sequence of unique ids for each segment of interest in the
            image. Default is `[1]`.
        if_missing: If 'raise' raises a ValueError when the region of interest
            is empty; if 'ignore' the transformation does not have any effect on
            the image and mask.
        p: The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """
    def __init__(self, include: List = [1], if_missing='raise', p: float = 1.0):
        assert p > 0
        self.include = include
        assert if_missing in {'raise', 'ignore'}
        self.if_missing = if_missing
        self.p = p
        assert isinstance(self.include, (tuple, list, np.array))

    def __call__(self, image, mask):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        if mask is None or image is None:
            raise ValueError('SegmentCrop requires an image and a mask.')
        check_dimensions(image, mask)
        if random.random() <= self.p:
            # Create a binary mask
            mask_arr = sitk.GetArrayFromImage(mask)
            mask_array = np.zeros_like(mask_arr)
            mask_array[np.isin(mask_arr, self.include)] = 1
            binary_mask = sitk.GetImageFromArray(mask_array)
            binary_mask.CopyInformation(mask)
            #
            if mask_array.sum() == 0:
                if self.if_missing == 'raise':
                    included_str = ', '.join([str(x) for x in self.include])
                    msg = f'mask does not include any item from {included_str}'
                    raise ValueError(msg)
                else:
                    return image, mask
            # Determine the region of interest
            lsif = sitk.LabelShapeStatisticsImageFilter()
            lsif.Execute(binary_mask)
            bbox = np.array(lsif.GetBoundingBox(1))
            mid = len(bbox) // 2
            seg_index = np.array(bbox[:mid])
            seg_size = np.array(bbox[mid:])
            # Crop the region of interest from the image and mask
            tsfm = sitk.RegionOfInterestImageFilter()
            tsfm.SetSize(seg_size.astype(int).tolist())
            tsfm.SetIndex(seg_index.astype(int).tolist())
            image, mask = apply_transform(tsfm, image, mask)
        return image, mask

    def __repr__(self):
        included_str = ', '.join(self.include)
        msg = '{} (include={}, if_missing={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.include,
                          self.if_missing,
                          self.p)


class Resize(Transformation):
    """Resize an image and its mask (if provided).

    Args:
        size: A tuple of int values representing image size. The order of
            dimensions should be (x, y, z), i.e. (width, height, and depth).
        interpolator: A SimpleITK interpolator function.
            The default interpolator for image is ``sitk.sitkBSpline``. Other
            options for interpolations are:
                * ``sitk.sitkLinear``
                * ``sitk.sitkGaussian``
                * ``sitk.sitkNearestNeighbor``
                * ``sitk.sitkHammingWindowedSinc``
                * ``sitk.sitkBlackmanWindowedSinc``
                * ``sitk.sitkCosineWindowedSinc``
                * ``sitk.sitkWelchWindowedSinc``
                * ``sitk.sitkLanczosWindowedSinc``

            A mask is always interpolated using ``sitk.sitkNearestNeighbor`` to
            avoid introducing new labels.
        default_image_voxel_value: Set the image voxel value when a transformed
            a voxel is outside of the image volume. The default value is ``0``.
        default_mask_voxel_value: Set the mask voxel value when a transformed
            a voxel is outside of the mask volume. The default value is ``0``.
        p (float): The transformation is applied with a probability of ``p``.
            The default value is ``1.0``.
    """
    def __init__(self, size: Tuple[int, int, int],
                 interpolator=sitk.sitkBSpline,
                 default_image_voxel_value=0,
                 default_mask_voxel_value=0,
                 p: float = 1.0):
        assert p > 0
        self.size = size
        self.interpolator = interpolator
        self.default_image_pixel_value = default_image_voxel_value
        self.default_mask_pixel_value = default_mask_voxel_value
        self.p = p
        self.filter = sitk.ResampleImageFilter()
        self.filter.SetTransform(sitk.Transform())

    def __call__(self, image: sitk.Image, mask: sitk.Image=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if image.GetDimension() != len(self.size):
            msg = f'Image dimension should be equal to {DIMENSION}.'
            raise ValueError(msg)
        if np.any(np.array(self.size) <= 0):
            msg = 'Image size cannot be zero or negative in any dimension'
            raise ValueError(msg)
        if random.random() <= self.p:
            spacing = [img_size * img_spacing / out_size
                       for img_size, img_spacing, out_size in zip(image.GetSize(),
                                                                  image.GetSpacing(),
                                                                  self.size)]
            # Resize image.
            self.filter.SetDefaultPixelValue(self.default_image_pixel_value)
            self.filter.SetSize(self.size)
            self.filter.SetOutputSpacing(spacing)
            self.filter.SetOutputOrigin(image.GetOrigin())
            self.filter.SetOutputDirection(image.GetDirection())
            self.filter.SetOutputPixelType(image.GetPixelIDValue())
            self.filter.SetInterpolator(self.interpolator)
            image = self.filter.Execute(image)
            # Resize mask
            if mask is not None:
                spacing = [img_size * img_spacing / out_size
                           for img_size, img_spacing, out_size in zip(mask.GetSize(),
                                                                      mask.GetSpacing(),
                                                                      self.size)]
                self.filter.SetDefaultPixelValue(self.default_mask_pixel_value)
                self.filter.SetOutputSpacing(spacing)
                self.filter.SetOutputOrigin(mask.GetOrigin())
                self.filter.SetOutputDirection(mask.GetDirection())
                self.filter.SetOutputPixelType(mask.GetPixelIDValue())
                self.filter.SetInterpolator(sitk.sitkNearestNeighbor)
                mask = self.filter.Execute(mask)
            return image, mask

    def __repr__(self):
        msg = '{} (size={}, interpolator={}, default pixel value={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.size,
                          self.interpolator,
                          self.default_pixel_value,
                          self.p)


class Expand(Transformation):
    """Enlarge an image by an integer factor in each dimension.

    Given an image with size of (m, n, k), applying this transformation using an
        expansion factor of (a, b, c) results in an image of size (a * m, b * n,
        c * k). The transformed image (mask) is obtained by interpolating
        the input image (mask). The voxel will change after applying this
        transformation.

    Args:
        expansion: A tuple of positive integer values representing the scale
            factors for each dimension.
        interpolator: A SimpleITK interpolator function.
            The default interpolator for image is ``sitk.sitkLinear``. Other
            options for interpolations are:
                * ``sitk.sitkBSpline``
                * ``sitk.sitkGaussian``
                * ``sitk.sitkNearestNeighbor``
                * ``sitk.sitkHammingWindowedSinc``
                * ``sitk.sitkBlackmanWindowedSinc``
                * ``sitk.sitkCosineWindowedSinc``
                * ``sitk.sitkWelchWindowedSinc``
                * ``sitk.sitkLanczosWindowedSinc``

            A mask is always interpolated using ``sitk.sitkNearestNeighbor`` to
            avoid introducing new labels.
        p (float): The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """
    def __init__(self, expansion: tuple,
                 interpolator=sitk.sitkLinear,
                 p: float = 1.0):
        assert p > 0
        self.expansion = expansion
        self.interpolator = interpolator
        self.p = p
        self.filter = sitk.ExpandImageFilter()
        self.filter.SetExpandFactors(self.expansion)


    def __call__(self, image: sitk.Image, mask: sitk.Image = None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if random.random() <= self.p:
            if len(self.expansion) != image.GetDimension():
                msg = 'Image dimension must equal the length of expansion.'
                raise ValueError(msg)
            self.filter.SetInterpolator(self.interpolator)
            image = self.filter.Execute(image)
            self.filter.SetInterpolator(sitk.sitkNearestNeighbor)
            mask = self.filter.Execute(mask)
        return image, mask

    def __repr__(self):
        msg = '{} (expand_factors={}, interpolator={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.expand_factors,
                          self.interpolator,
                          self.p)


class Shrink(Transformation):
    """Shrink an image by an integer factor in each dimension.

        Given an image with size of (m, n, k), applying this transformation
            using an expansion factor of (a, b, c) results in an image of size
            (a // m, b // n, c // k), where // represents integer division.
            The voxel spacing will change after applying this transformation.

        Args:
            shrinkage: A tuple of positive integer values representing the
                shrinkage factors for each dimension.
            p (float): The transformation is applied with a probability of p.
                The default value is ``1.0``.
        """
    def __init__(self, shrinkage: Tuple[int, int, int], p: float = 1.0):
        assert p > 0
        self.shrinkage = shrinkage
        self.p = p

        self.filter = sitk.BinShrinkImageFilter()
        self.filter.SetShrinkFactors(self.shrinkage)

    def __call__(self, image, mask=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if len(self.shrinkage) != image.GetDimension():
            msg = 'Image dimension must equal the length of shrinkage.'
            raise ValueError(msg)
        if random.random() <= self.p:
            image, mask = apply_transform(self.filter, image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (shrink_factors={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.shrink_factors,
                          self.p)


class Invert(Transformation):
    """Invert the intensity of an image based on a constant value.

    This transformation does not affect the mask.

    Args:
        maximum: The maximum intensity value used for inverting voxel values.
            A voxel value ``v`` is transformed to ``maximum - v``.
        p: The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """

    def __init__(self, maximum: Union[int, float, None] = None, p: float = 1.0):
        assert p > 0
        self.p = p
        self.filter = sitk.InvertIntensityImageFilter()
        self.maximum = maximum

    def __call__(self, image, mask=None, *args, **kwargs):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if random.random() <= self.p:
            if self.maximum is None:
                stat_dict = get_stats(image)
                maximum = stat_dict['max']
            else:
                maximum = self.maximum
            self.filter.SetMaximum(maximum)
            image, _ = apply_transform(self.filter, image)
        return image, mask

    def __repr__(self):
        maximum = self.maximum if self.maximum is not None else 'None'
        msg = '{} (maximum={}, p={})'
        return msg.format(self.__class__.__name__,
                          maximum,
                          self.p)


class BionomialBlur(Transformation):
    """Apply binomial blur filter to an image.

    Args:
        repetition: The number of times to repeat the smoothing filter.
        p: The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """
    def __init__(self, repetition=1, p=1.0):
        assert p > 0
        self.repetition = repetition
        self.p = p
        self.filter = sitk.BinomialBlurImageFilter()
        self.filter.SetRepetitions(self.repetition)

    def __call__(self, image, mask=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, _ = apply_transform(self.filter, image, None)
        return image, mask

    def __repr__(self):
        msg = '{} (repetition={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.repetition,
                          self.p)


class SaltPepperNoise(Transformation):
    """Changes the voxel values with fixed value impulse noise.

    This transformation, which is often called salt and pepper noise does not
        affect the mask.

    Args:
        noise_prob: The noise probability to be applied on a image. The default
        is ``0.01``.
        noise_range: A tuple of size 2 representing the lower and upper
            bounds of noise values.
        random_seed (int): Random integer number to set seed for random noise
            generation. Default is `None`.
        p: The transformation is applied with a probability of p. The default
        is ``1.0``.
    """

    def __init__(self, noise_prob: float = 0.01,
                 noise_range: Tuple = None,
                 random_seed=None, p=1.0):
        assert p > 0
        self.noise_prob = noise_prob
        self.random_seed = random_seed
        self.p = p
        self.min, self.max = None, None
        if noise_range is not None:
            if noise_range[0] >= noise_range[1] or len(noise_range) != 2:
                msg = ('noise_range must be a tuple of size 2 representing'
                       'the lower and upper bounds of noise values')
                raise ValueError(msg)
            self.min, self.max = noise_range

        self.filter = sitk.SaltAndPepperNoiseImageFilter()
        self.filter.SetProbability(self.noise_prob)
        if self.random_seed is not None:
            self.filter.SetSeed(self.random_seed)

    def __call__(self, image, mask=None, *args, **kwargs):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, _ = apply_transform(self.filter, image, None, *args, **kwargs)
            if self.min is not None:
                image_array = sitk.GetArrayFromImage(image)
                image_array[(image_array < self.min)] = self.min
                image_array[(image_array > self.max)] = self.max
                img = sitk.GetImageFromArray(image_array)
                img.CopyInformation(image)
                image = img
        return image, mask

    def __repr__(self):
        msg = '{} (noise_prob={}, random_seed={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.noise_prob,
                          self.random_seed,
                          self.p)


class AdditiveGaussianNoise(Transformation):
    """Apply additive Gaussian white noise to an image.

    This transformation does not affect masks.

    Args:
        mean: The mean value of the gaussian distribution used for noise
            generation. The default value is ``0.0``.
        std: The standard deviation fo the gaussian distribution used for
            noise generation. The default value is ``1.0``.
        p: The transformation is applied with a probability of p. The default
            value is ``1.0``.
    """
    def __init__(self, mean: float = 0.01, std: float = 1.0, p: float = 1.0):
        assert p > 0
        self.mean = mean
        self.std = std
        self.p = p

        self.filter = sitk.AdditiveGaussianNoiseImageFilter()
        self.filter.SetMean(self.mean)
        self.filter.SetStandardDeviation(self.std)

    def __call__(self, image, mask=None, *args, **kwargs):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, _ = apply_transform(self.filter, image, None)
        return image, mask

    def __repr__(self):
        msg = '{} (mean={}, std={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.mean,
                          self.std,
                          self.p)


class MinMaxScaler(Transformation):
    """Linearly transform voxel values to a given range.

     This transformation does not affect masks. This transformation convert an
        constant image to a constant image where all voxels are equal to the
        provided maximum value.

     Args:
         min_value (float): The minimum value in the converted image. The
            default value is ``0``.
         max_value (float): The maximum value in the converted image. The
            default value is ``1``.
         p (float): The transformation is applied with a probability of p.
            The default value is ``1.0``.
     """

    def __init__(self, min_value=0, max_value=1, p: float = 1.0):
        self.p = p
        if min_value >= max_value:
            msg = 'min_value must be smaller than max_value.'
            raise ValueError(msg)
        self.min_value = min_value
        self.max_value = max_value


    def __call__(self, image, mask=None, *args, **kwargs):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if random.random() <= self.p:
            info = get_stats(image)
            minimum = info['min']
            maximum = info['max']
            if (maximum - minimum) < EPSILON:
                image = self.max_value
            else:
                image = image / (maximum - minimum)
                image = image * (self.max_value - self.min_value) + self.min_value
        return image, mask

    def __repr__(self):
        msg = '{} (p={})'
        return msg.format(self.__class__.__name__,
                          self.p)


class UnitNormalize(Transformation):
    """Normalize an mage by transforming mean to ``0`` and variance to ``1``.

    This transformation does not affect masks.

    Args:
        p (float): The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """
    def __init__(self, p: float = 1.0):
        assert p > 0
        self.p = p
        self.filter = sitk.NormalizeImageFilter()

    def __call__(self, image, mask=None, *args, **kwargs):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, _ = apply_transform(self.filter, image, None, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (p={})'
        return msg.format(self.__class__.__name__,
                          self.p)


class WindowLocationClip(Transformation):
    """Clip the voxel values to a specified range.

    Args:
        location: Central point of the window for clipping the image.
        window: Positive integer value as window size, representing the
            range that any voxel values outside this range will be clipped to
            the lower bound and upper bound of this range. The center of the
            window is the ``location``, i.e. the range will be:
            ``(location - window // 2, location + window // 2)``, where // is
            integer division.
        p (float): The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """

    def __init__(self, location, window, p=1.0):
        assert p > 0
        self.location = location
        self.window = window
        self.p = p
        self.filter = sitk.ClampImageFilter()

    def __call__(self, image, mask=None, *args, **kwargs):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if random.random() <= self.p:
            self.filter.SetLowerBound(self.location - self.window)
            self.filter.SetUpperBound(self.location + self.window)
            image, _ = apply_transform(self.filter, image, None, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (location={}, window={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.location,
                          self.window,
                          self.p)


class Clip(Transformation):
    """Clip voxel values to a specified range.

    This transformation does not affect the masks.

    Args:
        lower_bound: Any voxel values less than ``lower_bound`` will be clipped
            to ``lower_bound``
        upper_bound : Any voxel values greater than ``upper_bound`` will be
            clipped to ``upper_bound``
        p: The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """

    def __init__(self, lower_bound, upper_bound, p: float = 1.0):
        assert p > 0
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.p = p
        self.filter = sitk.ClampImageFilter()

    def __call__(self, image, mask=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if random.random() <= self.p:
            self.filter.SetLowerBound(self.lower_bound)
            self.filter.SetUpperBound(self.upper_bound)
            image, _ = apply_transform(self.filter, image)
        return image, mask

    def __repr__(self):
        msg = '{} (location={}, window={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.location,
                          self.window,
                          self.p)


class IsolateRange(Transformation):
    """Set voxel values outside a given range to a given constant.

    This object provide the option to manipulate the outside range in a mask
        image. The outside region for the mask can be set to a specific value.
        Also, the region outside the clipped range can be set to a constant
        value.
        Note: The boundaries defined by the threshold values are considered as
            inside region and will not be affected.
    Args:
        lower_bound: Any voxel values less than this parameter will be set to
            ``image_outside_value``. If  ``recalculate_mask`` is True,
            corresponding voxel values in the mask will be set to
            ``mask_outside_value``.
        upper_bound: Any voxel values greater than this parameter will be set to
            ``image_outside_value``. If  ``recalculate_mask`` is True,
            corresponding voxel values in the mask will be set to
            ``mask_outside_value``.
        image_outside_value: This value is used to set the value of outside
            voxels in the image. Any voxel with a value less than
            ``lower_bound`` or greater than ``upper_bound`` is considered
             to be in the outside region. The default is ``0``.
        mask_outside_value: This value is used to set the value of outside
            voxels in the mask. Any voxel with a value less than
            ``lower_bound`` or greater than ``upper_bound`` is considered
             to be in the outside region. If ``recalculate_mask`` is False,
             this is ignored. The default is ``0``.
        recalculate_mask: If True and the mask is not None, the voxel values
            mask representing the outside region are replaced with
            ``image_outside_value``. If False, the mask is not affected
             by this transformation.
        p: The transformation is applied with a probability of ``p``.
            The default value is ``1.0``.

    Raises:
        ValueError: If recalculate_mask is True and mask is None.
    """
    def __init__(self, lower_bound, upper_bound, image_outside_value=0,
                 mask_outside_value=0, recalculate_mask: bool = False,
                 p: float = 1.0):
        assert p > 0
        if lower_bound > upper_bound:
            raise ValueError('lower_bound must be smaller than upper_bound.')
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.image_outside_value = image_outside_value
        self.mask_outside_value = mask_outside_value
        self.recalculate_mask = recalculate_mask
        self.p = p
        self.filter = sitk.ThresholdImageFilter()
        self.filter.SetLower(self.lower_bound)
        self.filter.SetUpper(self.upper_bound)
        self.filter.SetOutsideValue(self.image_outside_value)

    def __call__(self, image, mask=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if self.recalculate_mask is True and mask is None:
            msg = 'mask cannot be None when recalculate_mask is True.'
            raise ValueError(msg)

        if random.random() <= self.p:
            if self.recalculate_mask is True:
                mask_ar = sitk.GetArrayFromImage(mask)
                img_ar = sitk.GetArrayFromImage(image)
                msk = np.ones_like(mask_ar) * self.mask_outside_value
                target = ((self.lower_bound <= img_ar) &
                          (img_ar <= self.upper_bound))
                msk[target] = mask_ar[target]
                msk_image = sitk.GetImageFromArray(msk)
                msk_image.CopyInformation(mask)
                mask = msk_image
            image, _ = apply_transform(self.filter, image, None)
        return image, mask

    def __repr__(self):
        msg = ('{} (lower_threshold={}, upper_threshold={}, '
              'image_outside_value={}, mask_outside_value={}, '
              'recalculate_mask={}, p={})')
        return msg.format(self.__class__.__name__,
                          self.lower_bound,
                          self.upper_bound,
                          self.image_outside_value,
                          self.mask_outside_value,
                          self.recalculate_mask,
                          self.p)


class IntensityRangeTransfer(Transformation):
    """Apply a linear transformations to the voxel values in a given range.

    Applies a linear transformation to the image voxel values of an image that
        are inside a user-defined range.

    Args:
        interval: A tuple containing the lower bound and upper bound for the
            output image, (lower, upper).
        cast: A data type used for casting the resulting image. If None,
            no casting will be applied. The default is None. Assuming that 
            SimpleITK has been imported as sitk. The following
            options can be used:
                * sitk.sitkUInt8:	Unsigned 8 bit integer
                * sitk.sitkInt8:	Signed 8 bit integer
                * sitk.sitkUInt16:	Unsigned 16 bit integer
                * sitk.sitkInt16:	Signed 16 bit integer
                * sitk.sitkUInt32:	Unsigned 32 bit integer
                * sitk.sitkInt32:	Signed 32 bit integer
                * sitk.sitkUInt64:	Unsigned 64 bit integer
                * sitk.sitkInt64:	Signed 64 bit integer
                * sitk.sitkFloat32:	32 bit float
                * sitk.sitkFloat64:	64 bit float
        p: The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """
    def __init__(self, interval: tuple, cast=None, p=1.0):
        if len(interval) != 2:
            msg = 'window must be a tuple of two numbers.'
            raise ValueError(msg)
        self.window = interval
        self.cast = cast
        self.p = p
        self.filter = sitk.IntensityWindowingImageFilter()

    def __call__(self, image, mask=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if random.random() <= self.p:
            information = get_stats(image)
            self.filter.SetWindowMinimum(information['min'])
            self.filter.SetWindowMaximum(information['max'])

            self.filter.SetOutputMinimum(self.window[0])
            self.filter.SetOutputMaximum(self.window[1])
            if self.cast is not None:
                image = sitk.Cast(image, self.cast)
            image, _ = apply_transform(self.filter, image)
        return image, mask

    def __repr__(self):
        msg = '{} (window={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.window,
                          self.p
        )


class AdaptiveHistogramEqualization(Transformation):
    """Histogram equalization modifies the contrast in an image.

        This transformation uses the AdaptiveHistogramEqualizationImageFilter
            from SimpleITK. "AdaptiveHistogramEqualization can produce an
            adaptively equalized histogram or a version of unsharp mask
            (local mean subtraction). Instead of applying a strict histogram
            equalization in a window about a pixel, this filter prescribes a
            mapping function (power law) controlled by the parameters alpha and
            beta."

    Args:
        alpha: This parameter controls the behaviour of the transformation. A
            value of ``alpha=0`` makes the transformation to act like classical
            histogram equalization and a value of ``alpha=1`` makes the
            transformation to act like a unsharp mask. The values between make a
            trade-off. The default is
            ``0.5``.
        beta: This parameter controls the behaviour of the transformation. A
            value of ``beta=0`` makes the transformations to act like an unsharp
            mask and a value of ``beta=1`` makes the transformation to act like
            a pass through filter (beta=1, with alpha=1). The default is
            ``0.5``.
       radius: This value controls the size of the region over which the local
            statistics are calculated. The default value for Radius is `2` in
            all directions.
        p: The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """
    def __init__(self, alpha=1.0, beta=0.5, radius=2, p=1.0):
        self.alpha = alpha
        self.beta = beta
        self.radius = radius
        self.p = p

        self.filter = sitk.AdaptiveHistogramEqualizationImageFilter()
        self.filter.SetAlpha(self.alpha)
        self.filter.SetBeta(self.beta)
        self.filter.SetRadius(self.radius)

    def __call__(self, image, mask=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        if random.random() <= self.p:
            image, _ = apply_transform(self.filter, image)
        return image, mask

    def __repr__(self):
        msg = '{} (alpha={}, beta={}, radius={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.alpha,
                          self.beta,
                          self.radius,
                          self.p
        )


class MaskImage(Transformation):
    """Erase the region outside a mask.

    Args:
        segment_label: The label of the segment used for determining the 
            region to be kept.
        image_outside_value: All image voxel values that do not correspond to
            the ``segment_label`` are set to this value. The defult is ``0``.
        mask_outside_label: All mask voxel values that are not equal to the
            ``segment_label`` are set to this value. The default is ``0``.
        p: The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """
    def __init__(self, segment_label=1, image_outside_value=0,
                 mask_outside_label=0, p=1.0):
        assert p > 0
        self.segment_label = segment_label
        self.image_outside_value = image_outside_value
        self.mask_outside_label = mask_outside_label
        self.p = p
        self.filter = sitk.MaskImageFilter()
        self.filter.SetMaskingValue(self.segment_label)
        self.filter.SetOutsideValue(self.image_outside_value)

    def __call__(self, image, mask):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        if mask is None:
            msg = 'mask cannot be None for AdaptiveHistogramEqualization.'
            raise ValueError(msg)
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image = self.filter.Execute(image, mask)
            msk = sitk.GetArrayFromImage(mask)
            msk[msk != self.segment_label] = self.mask_outside_label
            msk = sitk.GetImageFromArray(msk)
            msk.CopyInformation(mask)
            mask = msk
        return image, mask

    def __repr__(self):
        msg = '{} (masking_value={}, outside_value={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.segment_label,
                          self.image_outside_value,
                          self.p)


class BinaryFillHole(Transformation):
    """Fill holes that are not connected to the boundary of a binary image.

    Args:
        foreground_value(float): Set the value in the image to consider as
            "foreground". Defaults to `maximum value` of InputPixelType if this
            this parameter is `None`.
        p: The transformation is applied with a probability of p.
            The default value is ``1.0``.
    """
    def __init__(self, foreground_value=1, p: float = 1.0):
        self.p = p
        self.filter = sitk.BinaryFillholeImageFilter()
        self.filter.FullyConnectedOn()
        if foreground_value:
            self.filter.SetForegroundValue(foreground_value)
        self.foreground_value = foreground_value

    def __call__(self, image, mask=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        if mask is None:
            raise ValueError('mask cannot be None.')
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, mask = apply_transform(self.filter, image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (foreground value={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.forground_value,
                          self.p)


class BinaryErode(Transformation):
    def __init__(self, background: int = 0, foreground: int = 1,
                 radius: Tuple[int, int, int] = (1, 1, 1)):
        for r in radius:
            assert r > 0
        self.filter = sitk.BinaryErodeImageFilter()
        self.filter.SetBackgroundValue(background)
        self.filter.SetForegroundValue(foreground)
        self.filter.SetKernelRadius(radius)

    def __call__(self, image: sitk.Image = None, mask: sitk.Image = None):
        assert mask is not None
        return image, self.filter.Execute(mask)

    def __repr__(self):
        msg = '{} (background={}, foreground={}, radius={})'
        return msg.format(self.__class__.__name__,
                          self.background,
                          self.foreground,
                          self.radius)


class BinaryDilate(Transformation):
    def __init__(self, background: int = 0, foreground: int = 1,
                 radius: Tuple[int, int, int] = (1, 1, 1)):
        for r in radius:
            assert r > 0
        self.filter = sitk.BinaryDilateImageFilter()
        self.filter.SetBackgroundValue(background)
        self.filter.SetForegroundValue(foreground)
        self.filter.SetKernelRadius(radius)

    def __call__(self, image: sitk.Image = None, mask: sitk.Image = None):
        assert mask is not None
        return image, self.filter.Execute(mask)

    def __repr__(self):
        msg = '{} (background={}, foreground={}, radius={})'
        return msg.format(self.__class__.__name__,
                          self.background,
                          self.foreground,
                          self.radius)


class MaskLabelRemap(Transformation):
    def __init__(self, mapping: dict):
        self.mapping = mapping

    def __call__(self, image: sitk.Image = None, mask: sitk.Image = None):
        check_dimensions(image, mask)
        data = sitk.GetArrayFromImage(mask)
        for old_value, new_value in self.mapping.items():
            data[data == old_value] = new_value
        msk = sitk.GetImageFromArray(data)
        msk.CopyInformation(mask)
        return image, msk

    def __repr__(self):
        msg = '{} (mapping={})'
        return msg.format(self.__class__.__name__,
                          self.mapping)


class Isotropic(Transformation):
    """Make an image isotropic, i.e. equally spaced in all directions.

    Args:
        interpolator= The interpolator used for resampling. The default is
            ``sitk.sitkLinear``. Other options for interpolations are:
                * ``sitk.sitkBSpline``
                * ``sitk.sitkGaussian``
                * ``sitk.sitkNearestNeighbor``
                * ``sitk.sitkHammingWindowedSinc``
                * ``sitk.sitkBlackmanWindowedSinc``
                * ``sitk.sitkCosineWindowedSinc``
                * ``sitk.sitkWelchWindowedSinc``
                * ``sitk.sitkLanczosWindowedSinc``
            The interpolation for mask is alway
        output_spacing: A single number representing the spacing in all
            directions. The default is ``1``.
        default_image_voxel_value: The default value for new image voxels.
            The default is ``0``.
        default_mask_voxel_value: The default value for new voxel mask voxels.
            The default is ``0``.
        output_image_voxel_type: The voxel type of the output image. The
            default is ``None``, meaning that the voxel type is the same as the
            image.
        output_mask_voxel_type: The voxel type of the output mask. The
            default is ``None``, meaning that the voxel type is the same as
            the mask.
        output_direction: The direction of image and mask (both assumed to be
            the same). The default is ``None``, meaning that the direction is
           is infered from the image and mask.
        output_origin: The origin of image and mask (both assumed to be
            the same). The default is ``None``, meaning that the origin is
           is infered from the image and mask.
        use_nearest_neighbor_extrapolator: Use nearest neighbot for
            extrapolations. The default is ``True``.
        dimension: The dimension of the image and mask. The default is ``3``.

    """
    def __init__(self,
                 interpolator=sitk.sitkLinear,
                 output_spacing: float = 1,
                 default_image_voxel_value=0,
                 default_mask_voxel_value=0,
                 output_image_voxel_type=None,
                 output_mask_voxel_type=None,
                 output_direction=None,
                 output_origin=None,
                 use_nearest_neighbor_extrapolator: bool = True,
                 dimension: int = 3):
        output_spacing = (output_spacing, ) * dimension
        self.resampler = Resample(
            interpolator=interpolator,
            output_spacing=output_spacing,
            default_image_voxel_value=default_image_voxel_value,
            default_mask_voxel_value=default_mask_voxel_value,
            output_image_voxel_type=output_image_voxel_type,
            output_mask_voxel_type=output_mask_voxel_type,
            output_direction=output_direction,
            output_origin=output_origin,
            use_nearest_neighbor_extrapolator=use_nearest_neighbor_extrapolator)
        self.filter = self.resampler.filter

    def __call__(self, image, mask=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        return self.resampler(image, mask)

    def __repr__(self):
        temp = repr(self.resampler)
        msg = '{}{})'
        return msg.format(self.__class__.__name__,
                          temp[len('Resample'):])


class Resample(Transformation):
    """Make an image isotropic, i.e. equally spaced in all directions.

    Args:
        interpolator= The interpolator used for resampling. The default is
            ``sitk.sitkLinear``. Other options for interpolations are:
                * ``sitk.sitkBSpline``
                * ``sitk.sitkGaussian``
                * ``sitk.sitkNearestNeighbor``
                * ``sitk.sitkHammingWindowedSinc``
                * ``sitk.sitkBlackmanWindowedSinc``
                * ``sitk.sitkCosineWindowedSinc``
                * ``sitk.sitkWelchWindowedSinc``
                * ``sitk.sitkLanczosWindowedSinc``
            The interpolation for mask is alway
        output_spacing: A tuple representing the spacing in each directions.
            The order of directions is x (height), y (width), and z (depth).
            The default is ``(1, 1, 1)``.
        default_image_voxel_value: The default value for new image voxels.
            The default is ``0``.
        default_mask_voxel_value: The default value for new voxel mask voxels.
            The default is ``0``.
        output_image_voxel_type: The voxel type of the output image. The
            default is ``None``, meaning that the voxel type is the same as the
            image.
        output_mask_voxel_type: The voxel type of the output mask. The
            default is ``None``, meaning that the voxel type is the same as
            the mask.
        output_direction: The direction of image and mask (both assumed to be
            the same). The default is ``None``, meaning that the direction is
           is infered from the image and mask.
        output_origin: The origin of image and mask (both assumed to be
            the same). The default is ``None``, meaning that the origin is
           is infered from the image and mask.
        use_nearest_neighbor_extrapolator: Use nearest neighbot for
            extrapolations. The default is ``True``.

        """
    def __init__(self,
                 interpolator=sitk.sitkLinear,
                 output_spacing: tuple = (1, 1, 1),
                 default_image_voxel_value=0,
                 default_mask_voxel_value=0,
                 output_image_voxel_type=None,
                 output_mask_voxel_type=None,
                 output_direction=None,
                 output_origin=None,
                 use_nearest_neighbor_extrapolator:bool = True):
        self.interpolator = interpolator
        self.output_spacing = output_spacing
        self.default_image_voxel_value = default_image_voxel_value
        self.default_mask_voxel_value = default_mask_voxel_value
        self.output_image_voxel_type = output_image_voxel_type
        self.output_mask_voxel_type = output_mask_voxel_type
        self.output_direction = output_direction
        self.output_origin = output_origin
        self.nne = use_nearest_neighbor_extrapolator
        # Create the fileter
        self.filter = sitk.ResampleImageFilter()
        self.filter.SetTransform(sitk.Transform())
        self.filter.SetUseNearestNeighborExtrapolator(self.nne)


    def __call__(self, image, mask=None):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        check_dimensions(image, mask)
        if image is not None:
            image = self.__resample(image, self.interpolator,
                                    self.default_image_voxel_value,
                                    self.output_image_voxel_type)
        if mask is not None:
            mask = self.__resample(mask, sitk.sitkNearestNeighbor,
                                   self.default_mask_voxel_value,
                                   self.output_mask_voxel_type)
        return image, mask

    def __resample(self, image, interpolator, default_voxel_value,
                   output_voxel_type):
        """A helper function for resampling one image.
        Args:
            interpolator: The interpolator used for resampling. The default is
                ``sitk.sitkLinear``. Other options for interpolations are:
                    * ``sitk.sitkBSpline``
                    * ``sitk.sitkGaussian``
                    * ``sitk.sitkNearestNeighbor``
                    * ``sitk.sitkHammingWindowedSinc``
                    * ``sitk.sitkBlackmanWindowedSinc``
                    * ``sitk.sitkCosineWindowedSinc``
                    * ``sitk.sitkWelchWindowedSinc``
                    * ``sitk.sitkLanczosWindowedSinc``
                The interpolator for mask must be ``sitk.sitkNearestNeighbor``;
                    otherwise, the interpolation interoduces new labels in
                    the returned mask.
            default_voxel_value: The default value for new image (or mask)
                voxels.
            output_voxel_type: The voxel type of the output image (or mask).

        Returns:
            sitk.Image: The resampled image (or mask).
        """
        # Set spacing
        if self.output_spacing is None:
            out_spacing = image.GetSpacing()
        else:
            out_spacing = self.output_spacing
        self.filter.SetOutputSpacing(out_spacing)
        # Set size
        original_size = np.array(image.GetSize())
        original_spacing = np.array(image.GetSpacing())
        out_spacing = np.array(out_spacing)
        out_size = original_size * original_spacing / out_spacing
        out_size = [int(np.round(x)) for x in out_size]
        self.filter.SetSize(out_size)
        # Set direction
        if self.output_direction is None:
            self.filter.SetOutputDirection(image.GetDirection())
        else:
            self.filter.SetOutputDirection(self.output_direction)
        # Set origin
        if self.output_origin is None:
            self.filter.SetOutputOrigin(image.GetOrigin())
        else:
            self.filter.SetOutputOrigin(self.output_origin)
        # Set default voxel value
        self.filter.SetDefaultPixelValue(default_voxel_value)
        if output_voxel_type is None:
            output_voxel_type = image.GetPixelIDValue()
        self.filter.SetOutputPixelType(output_voxel_type)
        self.filter.SetInterpolator(interpolator)
        return self.filter.Execute(image)

    def __repr__(self):
        msg = ('{} (interpolator={}, output_spacing={}, '
               'default_image_voxel_value={}, default_mask_voxel_value={}, '
               'output_image_voxel_type={}, output_mask_voxel_type={}, '
               'output_direction={}, output_origin={}'
               'use_nearest_neighbor_extrapolator={}')
        return msg.format(self.__class__.__name__,
                          self.interpolator,
                          self.output_spacing,
                          self.default_image_voxel_value,
                          self.default_mask_voxel_value,
                          self.output_image_voxel_type,
                          self.output_mask_voxel_type,
                          self.output_direction,
                          self.output_origin,
                          self.nne)


class Reader(Transformation):
    """Load image and its corresponding mask (if provided) from their addresses.

    Each address could be a directory address or the DICOM file address.

    """
    def __init__(self):
        pass

    def __call__(self, image_path: str, mask_path: str = None):
        """

        Args:
            image_path: The address of an image. This could be a the
                address of a directory containing a dicom file or the address of
                a single file containing an image. All file formats supported
                IO file formats include:
                    * BMPImageIO (*.bmp, *.BMP)
                    * BioRadImageIO (*.PIC, *.pic)
                    * GiplImageIO (*.gipl *.gipl.gz)
                    * JPEGImageIO (*.jpg, *.JPG, *.jpeg, *.JPEG)
                    * LSMImageIO (*.tif, *.TIF, *.tiff, *.TIFF, *.lsm, *.LSM)
                    * MINCImageIO (*.mnc, *.MNC)
                    * MRCImageIO (*.mrc, *.rec)
                    * MetaImageIO (*.mha, *.mhd)
                    * NiftiImageIO (*.nia, *.nii, *.nii.gz, *.hdr, *.img, *.img.gz)
                    * NrrdImageIO (*.nrrd, *.nhdr)
                    * PNGImageIO (*.png, *.PNG)
                    * TIFFImageIO (*.tif, *.TIF, *.tiff, *.TIFF)
                    * VTKImageIO (*.vtk)
            mask_path:The address of a mask. This could be a the address of
                a directory containing a dicom series or the address of
                a single file containing an image. All file formats
                supported Supported IO file formats include:
                    * BMPImageIO (*.bmp, *.BMP)
                    * BioRadImageIO (*.PIC, *.pic)
                    * GiplImageIO (*.gipl *.gipl.gz)
                    * JPEGImageIO (*.jpg, *.JPG, *.jpeg, *.JPEG)
                    * LSMImageIO (*.tif, *.TIF, *.tiff, *.TIFF, *.lsm, *.LSM)
                    * MINCImageIO (*.mnc, *.MNC)
                    * MRCImageIO (*.mrc, *.rec)
                    * MetaImageIO (*.mha, *.mhd)
                    * NiftiImageIO (*.nia, *.nii, *.nii.gz, *.hdr, *.img, *.img.gz)
                    * NrrdImageIO (*.nrrd, *.nhdr)
                    * PNGImageIO (*.png, *.PNG)
                    * TIFFImageIO (*.tif, *.TIF, *.tiff, *.TIFF)
                    * VTKImageIO (*.vtk)

        Returns:
            sitk.Image: An image.
            sitk.Image: A mask. If the ``mask_path`` parameter is None,
                this would also be ``None``.
        """
        if (image_path is not None) and not os.path.exists(image_path):
            raise ValueError(f'Invalid address: {image_path}')
        if (mask_path is not None) and (not os.path.exists(mask_path)):
            raise ValueError(f'Invalid address: {mask_path}')
        image, mask = None, None
        image = read_image(image_path)
        if mask_path is not None:
            mask = read_image(mask_path)
        check_dimensions(image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} ()'
        return msg.format(self.__class__.__name__)


class Writer(Transformation):
    """Wrtie image and its corresponding mask (if provided) to file.

    Each address should be a file address. The supported IO file formats
        include:
        * BMPImageIO (*.bmp, *.BMP)
        * BioRadImageIO (*.PIC, *.pic)
        * GiplImageIO (*.gipl *.gipl.gz)
        * JPEGImageIO (*.jpg, *.JPG, *.jpeg, *.JPEG)
        * LSMImageIO (*.tif, *.TIF, *.tiff, *.TIFF, *.lsm, *.LSM)
        * MINCImageIO (*.mnc, *.MNC)
        * MRCImageIO (*.mrc, *.rec)
        * MetaImageIO (*.mha, *.mhd)
        * NiftiImageIO (*.nia, *.nii, *.nii.gz, *.hdr, *.img, *.img.gz)
        * NrrdImageIO (*.nrrd, *.nhdr)
        * PNGImageIO (*.png, *.PNG)
        * TIFFImageIO (*.tif, *.TIF, *.tiff, *.TIFF)
        * VTKImageIO (*.vtk)

    """

    def __init__(self):
        pass

    def __call__(self, image: sitk.Image = None, mask: sitk.Image = None,
                 image_path: str = None, mask_path: str = None, image_type=None,
                 mask_type=None, compressor='', use_compression=False,
                 compression_level=-1):
        """Write image and mask (if provided) to files.

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.
            image_path: The path in which ``image`` is saved. The file name
                should be at the end of the path and the extension should be the
                extension for a valid file type.
            mask_path: The path in which ``mask`` is saved.  The file name
                should be at the end of the path and the extension should be the
                extension for a valid file type.
            image_type: The image type used for casting the image. The
                default is None, meaning that no casting is applied by default.
            mask_type: The mask type used for casting the image. The
                default is None, meaning that no casting is applied by default.
                If applied, for most cases this shuld be sitk.sitkUInt8. In case
                the labels used in the mask are larger than 255,
                ``sitk.sitkUInt16``, ``sitk.sitkUInt32``, or ``sitk.sitkUInt64``
                should be used. If negative values are included in the labels
                of the mask, use sitk.sitkInt8, sitk.sitkInt16, or
                sitk.sitkUInt32. Note that using larger data types lead to
                increase in memory usage and larger files sizes.
            use_compression: If True, it is a request for compressing the
                image. Note that compression is not available for all file
                types.
            compressor: The compressor used for compression. The default is an
                empty string, representing the default compression. Note that
                not all file formats support compression.
            compression_level: The compression level used for compression. The
                default is -1, representing the default compression level.

        """
        check_dimensions(image, mask)
        if image is not None:
            if image_type is not None:
                image = sitk.Cast(image, image_type)
            sitk.WriteImage(image,
                            fileName=image_path,
                            useCompression=use_compression,
                            compressionLevel=compression_level,
                            compressor=compressor)
        if mask is not None:
            if mask_type is not None:
                mask = sitk.Cast(mask, mask_type)
            sitk.WriteImage(mask,
                            fileName=mask_path,
                            useCompression=use_compression,
                            compressionLevel=compression_level,
                            compressor=compressor)


    def __repr__(self):
        msg = '{} ()'
        return msg.format(self.__class__.__name__)


class SequentialWriter(Transformation):
    """Wrtie image and its corresponding mask (if provided) to files.

    Each address should be a file address. The supported IO file formats
        include:
        * BMPImageIO (*.bmp, *.BMP)
        * BioRadImageIO (*.PIC, *.pic)
        * GiplImageIO (*.gipl *.gipl.gz)
        * JPEGImageIO (*.jpg, *.JPG, *.jpeg, *.JPEG)
        * LSMImageIO (*.tif, *.TIF, *.tiff, *.TIFF, *.lsm, *.LSM)
        * MINCImageIO (*.mnc, *.MNC)
        * MRCImageIO (*.mrc, *.rec)
        * MetaImageIO (*.mha, *.mhd)
        * NiftiImageIO (*.nia, *.nii, *.nii.gz, *.hdr, *.img, *.img.gz)
        * NrrdImageIO (*.nrrd, *.nhdr)
        * PNGImageIO (*.png, *.PNG)
        * TIFFImageIO (*.tif, *.TIF, *.tiff, *.TIFF)
        * VTKImageIO (*.vtk)

    Args:
        dir_path: The default is '.', representing the
        current directory.
        image_prefix: A string used as the prefix for the saved image names.
            THe defaut is 'image'.
        image_postfix: A string used as the postfix for the saved image names.
            THe defaut is empty string.
        mask_prefix: A string used as the prefix for the saved mask names.
            THe defaut is 'mask'.
        mask_postfix=: A string used as the postfix for the saved image names.
            THe defaut is empty string.
        extension: The file extension used for saving files. This determines
            the type of the saved file. The default is 'nrrd'.
        image_type: The image type used for casting the image. The
            default is None, meaning that no casting is applied by default.
        mask_type: The mask type used for casting the image. The
            default is None, meaning that no casting is applied by default.
            If applied, for most cases this shuld be sitk.sitkUInt8. In case
            the labels used in the mask are larger than 255, ``sitk.sitkUInt16``,
            ``sitk.sitkUInt32``, or ``sitk.sitkUInt64`` should be used.
            If negative values are included in the labels of the mask, use
            sitk.sitkInt8, sitk.sitkInt16, or sitk.sitkUInt32. Note that
            using larger data types lead to increase in memory usage and larger
            files sizes.
        compressor: The compressor used for compression. The default is an empty
            string, representing the default compression. Note that not all
            file formats support compression.
        compression_level: The compression level used for compression. The
            default is None, representing the default compression level.
    """

    def __init__(self, dir_path='.', image_prefix='image', image_postfix='',
                 mask_prefix='mask', mask_postfix='', extension='nrrd',
                 image_type=None, mask_type=None, compressor='',
                 compression_level=None):
        self.image_prefix = image_prefix
        self.image_postfix = image_postfix
        self.mask_prefix = mask_prefix
        self.mask_postfix = mask_postfix
        self.extension = extension
        self.image_type = image_type
        self.mask_type = mask_type
        self.compressor = compressor
        self.compression_level = compression_level
        self.writer = sitk.ImageFileWriter()
        if compressor is not None:
            self.writer.UseCompressionOn()
            self.writer.SetCompressor(self.compressor)
            if self.compression_level is not None:
                self.writer.SetCompressionLevel(self.compression_level)
        self.index = 0
        self.dir_path = dir_path

    def generate_path(self):
        self.index += 1
        template= '{}{:0>5}{}.{}'
        image_path = template.format(self.image_prefix, self.index,
                                     self.image_postfix, self.extension)
        mask_path = template.format(self.mask_prefix, self.index,
                                     self.mask_postfix, self.extension)
        return (os.path.join(self.dir_path, image_path),
                os.path.join(self.dir_path, mask_path))

    def __call__(self, image: sitk.Image = None, mask: sitk.Image = None):
        """Write image and mask (if they are None) to files.

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            str: The written image file path.
            str: The wirtten mask file path.
        """
        check_dimensions(image, mask)
        if self.image_type is not None:
            image = sitk.Cast(image, self.image_type)
        if self.mask_type is not None:
            mask = sitk.Cast(mask, self.mask_type)
        image_path, mask_path = self.generate_path()

        if image is not None:
            self.writer.SetFileName(image_path)
            self.writer.Execute(image)
        else:
            image_path = None
        if mask is not None:
            self.writer.SetFileName(mask_path)
            self.writer.Execute(mask)
        else:
            mask_path = None
        return image_path, mask_path

    def __repr__(self):
        msg = '{} ()'
        return msg.format(self.__class__.__name__)


class Compose(Transformation):
    """Compose multiple transformations to a single transformation.

    Args:
        transforms: A list of transformations.
    """
    def __init__(self, transforms: Iterable):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        """Apply the transformations to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (transforms={})'
        tsfms = ', '.join([str(x) for x in self.transforms])
        return msg.format(self.__class__.__name__, tsfms)


class RandomChoices(Transformation):
    """Randomly select k transformations and apply them.

    Args:
        transforms: A list of transformations.
        k: The number of transformations to be selected.
        keep_original_order: If True preserve the order of transformations
            when applying them. Otherwise, transformations will be applied in a
            random order.
    """
    def __init__(self, transforms, k, keep_original_order=True):
        self.transforms = transforms
        self.k = k
        self.keep_original_order = keep_original_order

    def __call__(self, image, mask=None):
        """Apply the transformations to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        Raises:
            ValueError: If ``k`` is less than the number of provided
                transformations.
        """
        if self.k > len(self.transforms):
            raise ValueError('k should be less than the number of transforms.')
        if self.keep_original_order is True:
            temp = list(enumerate(self.transforms))
            temp = random.sample(temp, k=self.k)
            selected_trfms = sorted(temp, key=lambda x: x[0])
            _, selected_trfms = list(zip(*selected_trfms))
        else:
            selected_trfms = random.sample(self.transforms, k=self.k)
        for t in selected_trfms:
            image, mask = t(image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (transforms={}, k={}, keep_original_order={})'
        return msg.format(self.__class__.__name__,
                          self.transforms,
                          self.k,
                          self.keep_original_order)


class OneOf(Transformation):
    """Apply one of the provided transformations.

    Args:
        transforms: A list of transformations. One transformation is selected
            randomly.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        """Choose one transformation and apply it to an image and its mask.

         If mask is None, the chosen transformation is only applied to the
            image.

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        tr = random.choice(self.transforms)
        image, mask = tr(image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (transforms]{})'
        return msg.format(self.__class__.__name__,
                          self.transforms)


class RandomOrder(Transformation):
    """Apply a list of transformations in a random order.

    Args:
        transforms: List of selected transforms to be applied in random order.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        """Apply the transformations in a random order.

        The transformations are applied to an image and its mask (if provided).
            The order in which these transformation are applied is random.

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        random.shuffle(self.transforms)
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (transforms={})'
        return msg.format(self.__class__.__name__,
                          self.transforms)


class Lambda(Transformation):
    """Apply a customized transformation.

    Args:
        image_transformer: A callable object, e.g. a Lambda function or a
            regular function used for transforming the image. Default is
            None, indicting identity transformation, i.e. no change in the
            image.
        mask_transformer: A callable object, e.g. a Lambda function or a
            regular function used for transforming the mask. Default is
            None, indicting identity transformation, i.e. no change on the
            mask.
        p: The transformation is applied with a probability of p.
            The default value is `1.0`.
    """
    def __init__(self, image_transformer: Callable = None,
                 mask_transformer: Callable = None, p: float = 1.0):
        self.image_transformer = image_transformer
        self.mask_transformer = mask_transformer
        self.p = p

    def __call__(self, image, mask=None, *args, **kwargs):
        """Apply the transformation to an image and its mask (if provided).

        Args:
            image: A SimpleITK Image.
            mask: A SimpleITK Image representing the contours for the image.
                The default value is None. If mask is not None, its size should
                be equal to the size of the image.

        Returns:
            sitk.Image: The transformed image.
            sitk.Image: The mask for the transformed image. If the mask
                parameter is None, this would also be None.

        """
        if random.random() <= self.p:
            if image is not None:
                image = self.image_transformer(image, *args, **kwargs)
            if mask is not None:
                mask = self.mask_transformer(mask, *args, **kwargs)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNumpy(Transformation):
    """Convert an image and its mask (if provided) to Numpy arrays.

    Args:
        out_image_dtype: The numpy datatype used to as the type of the numpy
            array representing the image. If `None`, the output data type is
            inferred from the image.
        out_mask_dtype: The numpy datatype used to as the type of the numpy
            array representing the mask. If `None`, the output data type is
            inferred from the mask.
    """
    def __init__(self, out_image_dtype=None, out_mask_dtype=None):
        self.out_image_dtype = out_image_dtype
        self.out_mask_dtype = out_mask_dtype

    def __call__(self, image, mask=None):
        # Convert image and mask.
        assert isinstance(image, sitk.SimpleITK.Image)
        img, msk = None, None
        if image is not None:
            img = sitk.GetArrayFromImage(image)
        if mask is not None:
            msk = sitk.GetArrayFromImage(mask)
        # Change image and mask dtypes.
        if image is not None and self.out_image_dtype is not None:
            img = img.astype(self.out_image_dtype)
        if msk is not None and self.out_mask_dtype is not None:
            msk = msk.astype(self.out_mask_dtype)
        return img, msk

    def __repr__(self):
        msg = '{} (out_image_dtype={}, out_mask_dtype={})'
        return msg.format(self.__class__.__name__,
                          self.out_image_dtype,
                          self.out_mask_dtype)


class FromNumpy(Transformation):
    """Convert arrays to objects of type SimpleITK.Image.

    Args:
        out_image_dtype: The SimpleITK data type used as the type of the
            image. If `None`, the output data type is inferred from the image
            array.
        out_mask_dtype: The SimpleITK data type used as the type of the
            mask. If `None`, the output data type is inferred from the mask
            array.
    """
    def __init__(self, out_image_dtype=None, out_mask_dtype=None):
        self.out_image_dtype = out_image_dtype
        self.out_mask_dtype = out_mask_dtype

    def __call__(self, image, mask=None):
        # Convert image and mask arrays to objects of type SimpleITK.Image.
        img, msk = None, None
        if image is not None:
            assert isinstance(image, np.ndarray)
            img = sitk.GetImageFromArray(image)
        if mask is not None:
            assert isinstance(mask, np.ndarray)
            msk = sitk.GetImageFromArray(mask)
        # Change image and mask dtypes.
        if image is not None and self.out_image_dtype is not None:
            img = sitk.Cast(img, self.out_image_dtype)
        if mask is not None and self.out_mask_dtype is not None:
            msk = sitk.Cast(msk, self.out_mask_dtype)
        return img, msk

    def __repr__(self):
        msg = '{} (out_image_dtype={}, out_mask_dtype={})'
        return msg.format(self.__class__.__name__,
                          self.out_image_dtype,
                          self.out_mask_dtype)


class From2DTo3D(Transformation):
    """Convert a 2D SimpleITK image and mask to a 3D SimpleITK image and mask.

    Convert 2D SimpleITK image (mask) to 3D SimpleITK image (mask) with user
        defined depth. The 2D image will be copied, as a single layer of a 3D
        image, to produce each layer of the 3D image.

    Args:
        repeat (int): Number of times to replicate the 2D image.
            The default value is ``1``.
    """
    def __init__(self, repeat=1):
        self.repeat = repeat
        assert repeat > 0, 'repeat must be greater than to 0.'
        self.filter = sitk.JoinSeriesImageFilter()

    def __call__(self, image, mask=None):
        assert isinstance(image, sitk.SimpleITK.Image)
        msg = 'The {} should be 2D SimpleITK Image objects.'
        if image is not None and image.GetDimension() != 2 :
            raise ValueError(msg.format('image'))
        else:
            slices = [image for _ in range(self.repeat)]
            img = self.filter.Execute(slices)
        if mask is not None and mask.GetDimension() != 2:
            raise ValueError(msg.format('mask'))
        else:
            slices = [mask for _ in range(self.repeat)]
            msk = self.filter.Execute(slices)
        return img, msk

    def __repr__(self):
        msg = '{} (repeat={})'
        return msg.format(self.__class__.__name__,
                          self.repeat)



class From3DTo2D(Transformation):
    """Convert a 3D thin image and its mask to a 2D image and mask.

    A thin image is an image where the components across Z-axis represent only
        one pixel. Common thin images are graysacale, with dimension 1 across
        Z-axis, and RGB images, with dimension 3 across Z-axis.
    """

    def __init__(self, image_pixel_type=sitk.sitkVectorUInt8,
                 mask_pixel_type=sitk.sitkUInt8):
        self.image_pixel_type = image_pixel_type
        self.mask_pixel_type = mask_pixel_type

    @staticmethod
    def make_2D_image(image3D, pixel_type):
        img_array = sitk.GetArrayFromImage(image3D)
        depth, _, _ = img_array.shape
        if depth > 1:
            img_array = img_array.transpose(1, 2, 0)
            img = sitk.GetImageFromArray(img_array, isVector=True)
        else:
            img_array = np.squeeze(img_array, axis=0)
            img = sitk.GetImageFromArray(img_array, isVector=False)
        img.SetSpacing(image3D.GetSpacing()[:2])
        img.SetOrigin(image3D.GetOrigin()[:2])
        direc = np.array(image3D.GetDirection()).reshape(3, 3)[:2, :2].ravel()
        img.SetDirection(direc.tolist())
        return img

    def __call__(self, image, mask=None):
        check_dimensions(image, mask)
        msg = 'The image should be  a 3D SimpleITK Image objects.'
        img, msk = None, None
        if image is not None and image.GetDimension() != 3 :
            raise ValueError(msg)
        else:
            img = From3DTo2D.make_2D_image(image, self.image_pixel_type)
        if mask is not None:
            msk = From3DTo2D.make_2D_image(mask, self.mask_pixel_type)
        return img, msk

    def __repr__(self):
        msg = '{} (pixel_type={})'
        return msg.format(self.__class__.__name__, self.pixel_type)


class Cast(Transformation):
    """Cast SimpleITK objects to new SimpleITK data types.

    Args:
        out_image_dtype: The SimpleITK data type used as the type of the
            image. If `None`, the output data type is the same as the input
            data type.
        out_mask_dtype: The SimpleITK data type used as the type of the
            mask. If `None`, the output data type is the same as the input
            data type.
    """
    def __init__(self, out_image_dtype=None, out_mask_dtype=None):
        self.out_image_dtype = out_image_dtype
        self.out_mask_dtype = out_mask_dtype

    def __call__(self, image, mask=None):
        # Change image and mask data types.
        img, msk = image, mask
        if image is not None:
            assert isinstance(image, sitk.Image)
        if mask is not None:
            assert isinstance(mask, sitk.Image)
        if image is not None and self.out_image_dtype is not None:
            img = sitk.Cast(image, self.out_image_dtype)
        if mask is not None and self.out_mask_dtype is not None:
            msk = sitk.Cast(mask, self.out_mask_dtype)
        return img, msk

    def __repr__(self):
        msg = '{} (out_image_dtype={}, out_mask_dtype={})'
        return msg.format(self.__class__.__name__,
                          self.out_image_dtype,
                          self.out_mask_dtype)


