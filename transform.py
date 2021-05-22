import os
import random
import numpy as np 
import SimpleITK as sitk
import logging
from numbers import Number
from typing import List, Set, Dict, Tuple, Optional, Union

from utils import get_stats
from utils import check_dimensions

# Number = Union[int, float, np.int8, np.int16, np.int32,
#                 np.int64, np.uint8, np.uint16, np.uint32,
#                 np.uint64]

EPSILON = 1E-8
DIMENSION = 3
logging.basicConfig(format='%(levelname)s:%(message)s',
                    level=logging.DEBUG)

def apply_transform(trfm, image, mask=None, *args, **kwargs):
    """Apply a sitk src on the input image and the input mask if exist."""
    image = trfm.Execute(image, *args, **kwargs)
    if mask is not None:
        mask = trfm.Execute(mask, *args, **kwargs)
    return image, mask


def expand_parameters(param, dimension, name, convert_fn=None):
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

class Pad(object):
    """Pad an image and a mask using three methods include constant padding,
        mirror padding, and wrap padding.

    Args:
        padding (int, sequence): An iterable of 3 integer numbers to be used as
            the number of cells for padding across each dimension of an image.
            If an integer value is provided, it will be considered as
            a list tiled with 3 padding values.
        --note: Dimensions need to be width by height by depth in three
            dimensions. In each dimension, padding will be applied before and
            after each dimension.
        constant (float): A value to be used when assigning value to the
            padded cells. This parameter needs to be assigned if constant pad
            is needed and use_constant parameter is True. Default value is `0.0`.
        use_constant(bool): Boolean parameter if constant pad is used.
            Default value is `True`.
        use_mirror(bool): Boolean parameter if Mirror padding is used.
            Default value is `False`.
        use_wrap(bool); Boolean parameter if Wrap padding is used.
            Default is `False`.
        pad_lower_bound (bool): True if padding should be applied to the
            lower-bound of each dimension. Default is `True`.
        pad_upper_bound (bool): True if padding should be applied to the
            upper-bound of each dimension. Default is `True`.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """

    def __init__(self, padding, constant=None, use_constant=True,
                 use_mirror=False, use_wrap=False, pad_lower_bound=True,
                 pad_upper_bound=True, mask_background=0, p=1.0):
        if use_constant == False and use_mirror == False and use_wrap == False:
            raise ValueError('One of the padding types(`constat`, `mirror`, '
                             '`wrap`) must be selected.')
        self.padding = padding
        self.constant = constant
        self.use_constant = use_constant
        self.use_mirror = use_mirror
        self.use_wrap = use_wrap
        self.pad_lower_bound = pad_lower_bound
        self.pad_upper_bound = pad_upper_bound
        self.mask_background = mask_background
        self.p = p
        if not isinstance(self.padding, (tuple, list)):
            raise ValueError('padding should be a tuple of the same dimension '
                             'as image')
        self.padding = (np.array(padding)).tolist()
        if self.use_constant:
            self.tsfm = sitk.ConstantPadImageFilter()
            if constant is not None:
                self.tsfm.SetConstant(constant)
        elif self.use_mirror:
            self.tsfm = sitk.MirrorPadImageFilter()
        else:
            self.tsfm = sitk.WrapPadImageFilter()
        if pad_lower_bound is True:
            self.tsfm.SetPadLowerBound(self.padding)
        if pad_upper_bound is True:
            self.tsfm.SetPadUpperBound(self.padding)

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            constant = self.constant
            if (self.use_constant is True) and (self.constant is None):
                INSIDE_VALUE = 1
                OUTSIDE_VALUE = 0
                bin_image = sitk.OtsuThreshold(image,
                                               INSIDE_VALUE,
                                               OUTSIDE_VALUE)
                # Get the median background intensity for padding
                lbl_intensity_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
                lbl_intensity_stats_filter.SetBackgroundValue(OUTSIDE_VALUE)
                lbl_intensity_stats_filter.Execute(bin_image, image)
                constant = lbl_intensity_stats_filter.GetMedian(INSIDE_VALUE)
            self.tsfm.SetConstant(constant)
            image = self.tsfm.Execute(image)
            self.tsfm.SetConstant(self.mask_background)
            mask = self.tsfm.Execute(mask)
        return image, mask

    def __repr__(self):
        msg = '{} (Padding={}, Constant={}, use_constant={}, ' \
              'use_mirror={}, use_wrap={}, Padding lower-bound={}, Padding ' \
              'upper-bound={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.padding,
                          self.constant,
                          self.use_constant,
                          self.use_mirror,
                          self.use_wrap,
                          self.pad_lower_bound,
                          self.pad_upper_bound,
                          self.p)


class ForegroundMask(object):
    """
    """

    def __init__(self, background: str = '<', bins=128):
        self.background = background
        self.filter = sitk.OtsuThresholdImageFilter()
        self.filter.SetInsideValue(1)
        self.filter.SetOutsideValue(0)
        self.filter.SetNumberOfHistogramBins(bins)
        self.filter.SetMaskOutput(False)

    def __call__(self, image):

        self.filter.Execute(image)
        threshold = self.filter.GetThreshold()
        print(threshold, image.GetSize())
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

class ForegroundCrop(object):
    """
    """
    def __init__(self, background: str = '<', bins=128):
        self.background = background
        self.bins = bins

    def __call__(self, image, mask=None):
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
        msg = '{} (inside_value={}, outside_value={}'
        return msg.format(self.__class__.__name__)

#
# class Rotate(object):
#     """Rotate the given CT image by constant value x, y, z angles.
#
#     Args:
#         angles (sequence): The rotation angles in degrees. default is `(10, 10, 10)`.
#         interpolator(sitk interpolator): interpolator to apply on
#             rotated ct images.
#         p (float): The transformation is applied to the input image with a
#             probability of p. The default value is `0.5`.
#     """
#
#     def __init__(self, angles=(10, 10, 10), interpolator=sitk.sitkLinear, p=0.5):
#         self.p = p
#         if isinstance(angles, int):
#             angles = [0, 0, angles]
#         if isinstance(angles, (tuple, list)):
#             if len(angles) == 2 or len(angles) > 3:
#                 raise ValueError(f'Expected one integer or a sequence of '
#                                  f'length 3 as a angle or each dimension. Got {len(angles)}.')
#         self.x_angle, self.y_angle, self.z_angle = angles
#         self.tz = sitk.VersorTransform((0, 0, 1), np.deg2rad(self.z_angle))
#         self.ty = sitk.VersorTransform((0, 1, 0), np.deg2rad(self.y_angle))
#         self.tx = sitk.VersorTransform((1, 0, 0), np.deg2rad(self.x_angle))
#
#         self.interpolator = interpolator
#
#     def __call__(self, image, mask=None):
#         check_dimensions(image, mask)
#         if random.random() <= self.p:
#             center = image.TransformContinuousIndexToPhysicalPoint(
#                                                         np.array(image.GetSize()) / 2.0)
#
#             self.tz.SetCenter(center)
#             self.ty.SetCenter(center)
#             self.tx.SetCenter(center)
#
#             composite = sitk.CompositeTransform(self.tz)
#             composite.AddTransform(self.ty)
#             composite.AddTransform(self.tx)
#
#             image = sitk.Resample(image, image, composite, self.interpolator)
#             if mask is not None:
#                 mask = sitk.Resample(mask, mask, composite, sitk.sitkNearestNeighbor)
#         return image, mask
#
#     def __repr__(self):
#         msg = '{} (angles={}, interpolator={}, p={})'
#         return msg.format(self.__class__.__name__,
#                           (self.x_angle, self.y_angle, self.z_angle),
#                           self.interpolator,
#                           self.p)

class Affine(object):
    """Rotate the given CT image by constant value x, y, z angles.

    Args:
        angles (sequence): The rotation angles in degrees. default is `(10, 10, 10)`.
        interpolator(sitk interpolator): interpolator to apply on
            rotated ct images.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `0.5`.
    """
    DIMENSION = 3

    def __init__(self, angles=(0, 0, 20), translation=0, scale=1, p=0.5,
                 interpolator=sitk.sitkLinear, image_background=-1024,
                 mask_background=0, reference=None):
        self.p = p
        self.reference = reference
        self.angles = expand_parameters(angles, DIMENSION, 'angles', convert_fn=np.radians)
        self.translation = expand_parameters(translation, DIMENSION, 'translation', convert_fn=None)
        self.scale = scale
        self.interpolator = interpolator
        self.image_background = image_background
        self.mask_background = mask_background

    def __call__(self, image, mask=None):
        if random.random() <= self.p:
            aug_transform = sitk.Similarity3DTransform()
            reference = image if self.reference is None else self.reference
            transform = sitk.AffineTransform(Affine.DIMENSION)
            transform.SetMatrix(image.GetDirection())
            transform.SetTranslation(np.array(image.GetOrigin()) - np.array(reference.GetOrigin()))
            # Modify the transformation to align the centers of the original and
            # reference image instead of their origins.
            centering = sitk.TranslationTransform(Affine.DIMENSION)
            image_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) / 2.0))
            reference_center = reference.TransformContinuousIndexToPhysicalPoint(np.array(reference.GetSize()) / 2.0)
            reference_center = np.array(reference_center)
            centering.SetOffset(np.array(transform.GetInverse().TransformPoint(image_center) - reference_center))
            centered_transform = sitk.CompositeTransform(transform)
            centered_transform.AddTransform(centering)

            # Set the augmenting transform's center so that rotation is around the image center.
            aug_transform.SetCenter(reference_center)
            angles = [np.random.uniform(lb, ub) for (lb, ub) in self.angles]
            translation = [np.random.uniform(*pair) for pair in self.translation]
            parameters = Affine.make_parameters(*angles, *translation, self.scale)
            aug_transform.SetParameters(parameters)
            # Augmentation is done in the reference image space,
            # so we first map the points from the reference image space
            # back onto itself T_aug (e.g. rotate the reference image)
            # and then we map to the original image space T0.
            composite = sitk.CompositeTransform([centered_transform, aug_transform])
            aug_image = sitk.Resample(image, reference, composite,
                                      self.interpolator, self.image_background)
            aug_mask = None
            if mask is not None:
                aug_mask = sitk.Resample(mask, reference, composite,
                                         sitk.sitkNearestNeighbor, self.mask_background)
            return aug_image, aug_mask

    @staticmethod
    def make_parameters(thetaX, thetaY, thetaZ, tx, ty, tz, scale):
        '''
        Create a list representing a regular sampling of the 3D similarity transformation parameter space. As the
        SimpleITK rotation parameterization uses the vector portion of a versor we don't have an
        intuitive way of specifying rotations. We therefor use the ZYX Euler angle parametrization and convert to
        versor.
        Args:
            thetaX, thetaY, thetaZ: numpy ndarrays with the Euler angle values to use.
            tx, ty, tz: numpy ndarrays with the translation values to use.
            scale: numpy array with the scale values to use.
        Return:
            List of lists representing the parameter space sampling (vx,vy,vz,tx,ty,tz,s).
        '''
        return [list(Affine.eul2quat(parameter_values[0], parameter_values[1], parameter_values[2])) +
                [np.asscalar(p) for p in parameter_values[3:]] for parameter_values in
                np.nditer(np.meshgrid(thetaX, thetaY, thetaZ, tx, ty, tz, scale))][0]

    @staticmethod
    def eul2quat(ax, ay, az, atol=1e-8):
        '''
        Translate between Euler angle (ZYX) order and quaternion representation of a rotation.
        Args:
            ax: X rotation angle in radians.
            ay: Y rotation angle in radians.
            az: Z rotation angle in radians.
            atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
        Return:
            Numpy array with three entries representing the vectorial component of the quaternion.
        '''
        # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
        cx = np.cos(ax)
        cy = np.cos(ay)
        cz = np.cos(az)
        sx = np.sin(ax)
        sy = np.sin(ay)
        sz = np.sin(az)
        r = np.zeros((3, 3))
        r[0, 0] = cz * cy
        r[0, 1] = cz * sy * sx - sz * cx
        r[0, 2] = cz * sy * cx + sz * sx

        r[1, 0] = sz * cy
        r[1, 1] = sz * sy * sx + cz * cx
        r[1, 2] = sz * sy * cx - cz * sx

        r[2, 0] = -sy
        r[2, 1] = cy * sx
        r[2, 2] = cy * cx

        # Compute quaternion:
        qs = 0.5 * np.sqrt(r[0, 0] + r[1, 1] + r[2, 2] + 1)
        qv = np.zeros(3)
        # If the scalar component of the quaternion is close to zero, we
        # compute the vector part using a numerically stable approach
        if np.isclose(qs, 0.0, atol):
            i = np.argmax([r[0, 0], r[1, 1], r[2, 2]])
            j = (i + 1) % 3
            k = (j + 1) % 3
            w = np.sqrt(r[i, i] - r[j, j] - r[k, k] + 1)
            qv[i] = 0.5 * w
            qv[j] = (r[i, j] + r[j, i]) / (2 * w)
            qv[k] = (r[i, k] + r[k, i]) / (2 * w)
        else:
            denom = 4 * qs
            qv[0] = (r[2, 1] - r[1, 2]) / denom
            qv[1] = (r[0, 2] - r[2, 0]) / denom
            qv[2] = (r[1, 0] - r[0, 1]) / denom
        return qv


    def __repr__(self):
        msg = '{} (angles={}, interpolator={}, p={})'
        return msg.format(self.__class__.__name__,
                          str(self.angles),
                          self.interpolator,
                          self.p)


class RandomRotation3D(object):
    """Rotate the given CT image by x, y, z angles.

    Args:
        angles (sequence): The upper bound rotation angles in degrees. 
            The real degree with be sampled betweeen [-angle, angle] rotation
            over x, y, z in order. default is `(10, 10, 10)`.
        interpolator(sitk interpolator): interpolator to apply on
            rotated ct images.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `0.5`.
    """
    def __init__(self, angles=(10, 10, 10), interpolator=sitk.sitkLinear, p=0.5):
        self.p = p
        if angles == (0, 0, 0):
            raise ValueError('There is no nonzero angle.')
        if isinstance(angles, int):
            angles = [0, 0, angles]
        if isinstance(angles, (tuple, list)):
            if len(angles) == 2 or len(angles) > 3:
                raise ValueError(f'Expected one integer or a sequence of '
                                 f'length 3 as a angle or each dimension. Got {len(angles)}.')
        self.x_angle, self.y_angle, self.z_angle = angles
        self.interpolator = interpolator
    
    def define_transform(self, axes, angle, center):
        trfm = sitk.VersorTransform(axes, np.deg2rad(angle))
        trfm.SetCenter(center)
        return trfm

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            center = image.TransformContinuousIndexToPhysicalPoint(
                                        np.array(image.GetSize())/2.0)
            transformers_list = []
            angle = random.randint(-self.z_angle, self.z_angle)
            transformers_list.append(self.define_transform((0, 0, 1),
                                                          angle,
                                                          center))
            angle = random.randint(-self.y_angle, self.y_angle)
            transformers_list.append(self.define_transform((0, 1, 0),
                                                           angle,
                                                           center))
            angle = random.randint(-self.x_angle, self.x_angle)
            transformers_list.append(self.define_transform((1, 0, 0),
                                                           angle,
                                                           center))
            composite = sitk.CompositeTransform(transformers_list[0])
            for i in range(1, len(transformers_list)):
                composite.AddTransform(transformers_list[i])

            image = sitk.Resample(image, image, composite, self.interpolator)
            if mask is not None:
                mask = sitk.Resample(mask, mask, composite, self.interpolator)
        return image, mask

    def __repr__(self):
        msg = '{} (angles={}, interpolator={}, p={})'
        return msg.format(self.__class__.__name__,
                          (self.x_angle, self.y_angle, self.z_angle),
                          self.interpolator,
                          self.p)


class Flip(object):
    ''' Flips an image and it's mask across user specified axes.

        The flip axes are set via method SetFlipAxes( array ) where the input
            is a fixed array of length image dimension. The image is flipped
            across axes for which array[i] is true.

    Args:
        axes (sequence): A sequence of boolean values for each dimension.
            Default value is `Vertical flip`, `[True, False, False]`.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `0.5`.
    '''

    def __init__(self, axes=[True, False, False], p=0.5):
        self.p = p
        self.axes = axes
        self.trfm = sitk.FlipImageFilter()
        self.trfm.SetFlipAxes(axes)

    def __call__(self, image, mask=None, *args, **kwargs):
        if mask is not None:
            check_dimensions(image, mask)
        assert len(self.axes) == image.GetDimension(), 'image axes is ' \
                                                       'different from image dimension'
        if random.random() <= self.p:
            image, mask = apply_transform(self.trfm, image, mask, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (axes={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.axes,
                          self.p)


class RandomFlipX(object):
    ''' TODO: 
    '''

    def __init__(self, p=0.5):
        self.p = p
        self.axes = [True, False, False]
        self.trfm = sitk.FlipImageFilter()
        self.trfm.SetFlipAxes(self.axes)

    def __call__(self, image, mask=None):
        if mask is not None:
            check_dimensions(image, mask)
        assert len(self.axes) == image.GetDimension(), 'image axes is ' \
                                                       'different from image dimension'
        if random.random() <= self.p:
            image, mask = apply_transform(self.trfm, image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (axes={}, p={})'
        return msg.format(self.__class__.__name__, self.p)


class RandomFlipY(object):
    ''' TODO: 
    '''

    def __init__(self, p=0.5):
        self.p = p
        self.axes = [False, True, False]
        self.trfm = sitk.FlipImageFilter()
        self.trfm.SetFlipAxes(self.axes)

    def __call__(self, image, mask=None, *args, **kwargs):
        if mask is not None:
            check_dimensions(image, mask)
        assert len(self.axes) == image.GetDimension(), 'image axes is ' \
                                                       'different from image dimension'
        if random.random() <= self.p:
            image, mask = apply_transform(self.trfm, image, mask, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (axes={}, p={})'
        return msg.format(self.__class__.__name__, self.p)


class RandomFlipZ(object):
    ''' TODO: 
    '''

    def __init__(self, p=0.5):
        self.p = p
        self.axes = [False, False, True]
        self.trfm = sitk.FlipImageFilter()
        self.trfm.SetFlipAxes(self.axes)

    def __call__(self, image, mask=None, *args, **kwargs):
        if mask is not None:
            check_dimensions(image, mask)
        assert len(self.axes) == image.GetDimension(), 'image axes is ' \
                                                       'different from image dimension'
        if random.random() <= self.p:
            image, mask = apply_transform(self.trfm, image, mask, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (axes={}, p={})'
        return msg.format(self.__class__.__name__, self.p)


class Crop(object):
    """  Crop image to the selected region bounds.

        Changes the boundary of an image by removing pixels outside the target
            region. The region is specified as a Size and Index. The Size must
            be specified, while the Index defaults to zeros.
    Args:
        size (sequence): The size of the region to extract. Dimensions which
            have a size of 0 are collapsed. The number of non-zero sized
            determines the output dimension.
        index (sequence): The starting index of the input image to extract.
            The default values are [0, 0, 0].
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """

    def __init__(self, size: tuple, index: tuple, p=1.0):
        self.size = size
        self.index = index
        self.p = p
        self.tsfm = sitk.RegionOfInterestImageFilter()
        self.tsfm.SetIndex(index)
        self.tsfm.SetSize(size)

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        end_coordinate = np.array(self.index) + np.array(self.size)
        if all(end_coordinate > np.array(image.GetSize())):
            raise ValueError('size + index cannot be greater than image size')
        if random.random() <= self.p:
            image, mask = apply_transform(self.tsfm, image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (size={}, index={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.size,
                          self.index,
                          self.p)


class RandomCrop(object):
    """Crop image in a random position with a fixed size.

            Randomly crop the input image into a new size which is specified by the user.
        Args:
            size (sequence): The size of the region to extract. Dimensions which
                have a size of 0 are collapsed. The number of non-zero sized
                determines the output dimension.
            p (float): The transformation is applied to the input image with a
                probability of p. The default value is `1.0`.
        """

    def __init__(self, size: tuple, p=1.0):
        self.size = size
        self.p = p

    def __call__(self, image, mask=None, *args, **kwargs):
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


class CenterCrop(object):
    """ Crop an image and a mask with a constant value.
   
    Args:
        output_size (int, list): An iterable of 3 integer numbers to be used as
            the output size of each dimension of an image.
            If an integer value is provided, it will be considered as
            a list tiled with 2 dimensions and a size of voxels or orginal
            image for image voxels.
        --note: Crpping input order is [width, height, depth]. In each
            dimension padding will be applied to before and after dimension.
        crop_lower_bound (bool): True if cropping should be applied to the lower-bound
            of each dimension.
        crop_upper_bound (bool): True if cropping should be applied to the upper-bound
            of each dimension.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """
    def __init__(self, size, p=1.0):
        self.output_size = np.array(size, dtype='uint')
        self.p = p
        self.tsfm = sitk.RegionOfInterestImageFilter()

    def __call__(self, image, mask=None):
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
            self.tsfm.SetSize(self.output_size.tolist())
            self.tsfm.SetIndex(index.tolist())
            image, mask = apply_transform(self.tsfm, image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (output size={},  Cropping lower-bound={}, Cropping ' \
              'upper-bound={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.output_size,
                          self.p)


class SegmentSafeCrop(object):
    """ Crop an image and a mask so that keep informative segments safely.
        first, it gets the regions that contain special infromations and then
        crop the image and mask using those informations.

    Args:
        crop_size (sequence): Minimum interesting size to crop important parts of
            the image based on. Input values must be in order width, height,
            depth.
        include(sequence): Sequence of unique ids for each
            interested segment in the image. Default is `[1]`.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """
    def __init__(self, crop_size, include=[1], p=1.0):
        self.crop_size = np.array(crop_size)
        self.include = include
        self.p = p
        assert isinstance(self.include, (tuple, list))

    def __call__(self, image, mask=None):
        if mask is None:
            raise ValueError('SegmentSafeCrop requires a mask.')
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
            #
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
            for i, (seg_length, crop_length) in enumerate(zip(seg_size, crop_size)):
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


class Resize(object):
    """Resize an image via a coordinate src.

    Args:
        size(sequence): Sequence of int values representing image size. The order
            of dimensions is x, y, and z (i.e., width, height, and depth). The length
            of size should match the dimensions of the image.
        interpolator(itk::simple::InterpolatorEnum Interpolator): The interpolator
            function. The default is `LinearInterpolateImageFunction <InputImageType,
            TInterpolatorPrecisionType>`. Some other options are
            `NearestNeighborInterpolateImageFunction` (useful for binary masks and
            other images with a small number of possible pixel values),
            and `BSplineInterpolateImageFunction` (which provides a higher order of
            interpolation).
        default_pixel_value(int): Set the pixel value when a transformed pixel
            is outside of the image. The default pixel value is `0`.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """
    def __init__(self, size: Tuple[int, int, int], interpolator=sitk.sitkLinear,
                 default_image_pixel_value: int=-1024, default_mask_pixel_value: int=0, p: float=1.0):
        self.size = size
        self.interpolator = interpolator
        self.default_image_pixel_value = default_image_pixel_value
        self.default_mask_pixel_value = default_mask_pixel_value
        self.p = p
        self.tsfm = sitk.ResampleImageFilter()
        self.tsfm.SetTransform(sitk.Transform())



    def __call__(self, image: sitk.Image, mask: sitk.Image=None):
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
            self.tsfm.SetDefaultPixelValue(self.default_image_pixel_value)
            self.tsfm.SetSize(self.size)
            self.tsfm.SetOutputSpacing(spacing)
            self.tsfm.SetOutputOrigin(image.GetOrigin())
            self.tsfm.SetOutputDirection(image.GetDirection())
            self.tsfm.SetOutputPixelType(image.GetPixelIDValue())
            self.tsfm.SetInterpolator(self.interpolator)
            image = self.tsfm.Execute(image)
            # Resize mask
            if mask is not None:
                spacing = [img_size * img_spacing / out_size
                           for img_size, img_spacing, out_size in zip(mask.GetSize(),
                                                                      mask.GetSpacing(),
                                                                      self.size)]
                self.tsfm.SetDefaultPixelValue(self.default_mask_pixel_value)
                self.tsfm.SetOutputSpacing(spacing)
                self.tsfm.SetOutputOrigin(mask.GetOrigin())
                self.tsfm.SetOutputDirection(mask.GetDirection())
                self.tsfm.SetOutputPixelType(mask.GetPixelIDValue())
                self.tsfm.SetInterpolator(sitk.sitkNearestNeighbor)
                mask = self.tsfm.Execute(mask)
            return image, mask

    def __repr__(self):
        msg = '{} (size={}, interpolator={}, default pixel value={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.size,
                          self.interpolator,
                          self.default_pixel_value,
                          self.p)


class Expand(object): 
    """Expand the size of an image by an integer factor in each dimension.
    This filter will produce an output with different pixel spacing that its input image such that:


    Args: 
        expand_factors (int, sequence): Sequence of int values interpreted the 
            increasing factor for each dimension. Values are clamped to a 
            minimum value of 1. Default is `1` for all dimensions.
        interpolator (itk::simple::InterpolatorEnum Interpolator): The interpolator
            function. The default is `LinearInterpolateImageFunction <InputImageType,
            TInterpolatorPrecisionType>`. Some other options are
            `NearestNeighborInterpolateImageFunction` (useful for binary masks and
            other images with a small number of possible pixel values),
            and `BSplineInterpolateImageFunction` (which provides a higher order of
            interpolation).
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """
    def __init__(self, expansion: tuple, interpolator=sitk.sitkLinear,
                 p: float=1.0):
        self.expansion = expansion
        self.interpolator = interpolator
        self.p = p
        self.tsfm = sitk.ExpandImageFilter()
        self.tsfm.SetExpandFactors(self.expansion)
        self.tsfm.SetInterpolator(self.interpolator)

    def __call__(self, image, mask=None):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            if len(self.expansion) != image.GetDimension():
                msg = 'Image dimension must equal the length of expansion.'
                raise ValueError(msg)
            image, mask = apply_transform(self.tsfm, image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (expand_factors={}, interpolator={}, p={})'
        return msg.format(self.__class__.__name__, 
                          self.expand_factors,
                          self.interpolator, 
                          self.p)


class Shrink(object):
    ''' Reduce the size of an image by an integer factor in each dimension
            while performing averaging of an input neighborhood.
            outputSize[j] = max(floor(inputSize[j]/shrinkFactor[j]), 1)
    Args:
        shrink_factors (int, sequence): Rescale parameter to reduce the size of
            image. Default value is `(2, 2, 1)`.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    '''
    def __init__(self, shrinkage: tuple, p: float=1.0):
        self.shrinkage = shrinkage
        self.p = p

        self.tsfm = sitk.BinShrinkImageFilter()
        self.tsfm.SetShrinkFactors(self.shrinkage)

    def __call__(self, image, mask=None):
        check_dimensions(image, mask)
        if len(self.shrinkage) != image.GetDimension():
            msg = 'Image dimension must equal the length of shrinkage.'
            raise ValueError(msg)
        if random.random() <= self.p:
            image, mask = apply_transform(self.tsfm, image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (shrink_factors={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.shrink_factors,
                          self.p)


class Invert(object):
    """Invert the intensity of an image with a constant value.
        This src does not affect mask.
   
    Args:
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `0.5`.
    """

    def __init__(self, maximum: int=None, p=0.5):
        self.p = p
        self.tsfm = sitk.InvertIntensityImageFilter()
        self.maximum = maximum

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            if self.maximum is None:
                stat_dict = get_stats(image)
                maximum = stat_dict['max']
            else:
                maximum = self.maximum
            self.tsfm.SetMaximum(maximum)
            image, _ = apply_transform(self.tsfm, image)
        return image, mask

    def __repr__(self):
        maximum = self.maximum if self.maximum is not None else 'None'
        msg = '{} (maximum={}, p={})'
        return msg.format(self.__class__.__name__,
                          maximum,
                          self.p)


class BionomialBlur(object):
    """ Apply binomial blur to image.

        The binomial blur consists of a nearest neighbor average along each
            image dimension. The net result after n-iterations approaches
            convultion with a gaussian.

    Args:
        repetition (int): Number of times to repeat the smoothing filter.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `0.5`.
    """
    def __init__(self, repetition=1, p=0.5):
        self.repetition = repetition
        self.p = p
        self.trfm = sitk.BinomialBlurImageFilter()
        self.trfm.SetRepetitions(self.repetition)

    def __call__(self, image, mask=None):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, _ = apply_transform(self.trfm, image, None)
        return image, mask

    def __repr__(self):
        msg = '{} (repetition={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.repetition,
                          self.p)


class SaltPepperNoise(object):
    """Alter an image with fixed value impulse noise, often called salt and
        pepper noise. This src doesn't have any effect on mask.

    Args:
        noise_prob (float): Noise probablity to be applied on each image. 
            Default is `0.01`.
        random_seed (int): Random integer number to set seed for random noise
            generation. Default is `None`.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `0.5`.
    """

    def __init__(self, noise_prob=0.01, random_seed=None, p=0.5):
        self.noise_prob = noise_prob
        self.random_seed = random_seed
        self.p = p

        self.trfm = sitk.SaltAndPepperNoiseImageFilter()
        self.trfm.SetProbability(self.noise_prob)
        if self.random_seed is not None: 
            self.trfm.SetSeed(self.random_seed)

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, _ = apply_transform(self.trfm, image, None, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (noise_prob={}, random_seed={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.noise_prob, 
                          self.random_seed, 
                          self.p)


class GaussianWhiteNoise(object):
    """Alter an image with additive Gaussian white noise.
        This src doesn't have any effect on masks.
   
    Args:
        mean (float): Mean value for using in gaussian distribution. 
            Default is `0.0`.
        std (int): Standard deviation value for using in gaussian distribution.
            Default is `1.0`. 
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `0.5`.
    """

    def __init__(self, mean=0.01, std=1.0, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

        self.trfm = sitk.AdditiveGaussianNoiseImageFilter()
        self.trfm.SetMean(self.mean)
        self.trfm.SetStandardDeviation(self.std)

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, _ = apply_transform(self.trfm, image, None, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (mean={}, std={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.mean, 
                          self.std, 
                          self.p)


class LocalNoiseCalculator(object):
    """Calculate the local noise in an image.

        Computes an image where a given pixel is the standard deviation of the
            pixels in a neighborhood about the corresponding input pixel. This
            serves as an estimate of the local noise (or texture) in an image.
            Currently, this noise estimate assumes a piecewise constant image.
            This filter should be extended to fitting a (hyper) plane to the
            neighborhood and calculating the standard deviation of the residuals
            to this (hyper) plane.

    -- note: This src applies only to image.
    Args:
        radius (int): Set the values of the Radius vector all to value.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """
    def __init__(self, radius, p=1.0):
        self.radius = radius
        self.p = p
        self.trfm = sitk.NoiseImageFilter()
        self.trfm.SetRadius(self.radius)

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, _ = apply_transform(self.trfm, image, None, *args,
                                       **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (radius={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.radius,
                          self.p)


class MinMaxScaler(object):
    """ Transform an image by scaling each pixel/voxel value to a given range.

         Pixel/Voxel values will be transformed to a given range
            (min_value, max_value) in a linear manner. This src does not
            affect the mask, if applicable. A constant image will be converted
            to an image where all pixels/voxels are equal to the provided
            max_value.

     Args:
         min_value (float): The minimum value in the converted image. Defaul
            value is 0.
         max_value (float): The maximum value in the converted image. Default
            value is 1.
         p (float): The transformation is applied to the input image with a
             probability of p. The default value is `1.0`.
     """

    def __init__(self, min_value=0.0, max_value=1.0, p=1.0):
        self.p = p
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value


    def __call__(self, image, mask=None, *args, **kwargs):
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


class UnitNormalize(object):
    """ Normalize an image by setting its mean to zero and variance to one.

        This src take no action on masks.

    Args: 
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """
    def __init__(self, p=1.0):
        self.p = p
        self.trfm = sitk.NormalizeImageFilter()

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, _ = apply_transform(self.trfm, image, None, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (p={})'
        return msg.format(self.__class__.__name__,
                          self.p)


class ConstantNormalizer(object):
    """ Scales image pixel intensities to make the sum of all pixels equal a
        user-defined constant.

        This src is especially useful for normalizing a convolution kernel.
        This src hase no effect on masks.

    Args:
        sum_constant (float): Constant float value. Normalizer make the sum of
        all pixels to be equal to this constant. Default value is `1.0`.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `0.5`.
    """
    def __init__(self, sum_constant=1.0, p=0.5):
        self.sum_constant = sum_constant
        self.p = p
        self.trfm = sitk.NormalizeToConstantImageFilter()
        self.trfm.SetConstant(self.sum_constant)

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, _ = apply_transform(self.trfm, image, None, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (sum_constant={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.sum_constant,
                          self.p)


class WindowIntensityClip(object):
    """Casts input pixels to output pixel type and clamps the output pixel
        values to a specified range.

    Args:
        location (int): Central point of the window for cliping image.
        window (int): Positive integer value as window size to identify lower bound
            and upper bound from location point.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """

    def __init__(self, location, window, p=1.0):
        self.location = location
        self.window = window
        self.p = p
        self.trfm = sitk.ClampImageFilter()

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            self.trfm.SetLowerBound(self.location - self.window)
            self.trfm.SetUpperBound(self.location + self.window)
            image, _ = apply_transform(self.trfm, image, None, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (location={}, window={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.location,
                          self.window,
                          self.p)


class ThresholdIntensityClip(object): 
    """ Set image values to a user-specified value if they are below, above, or
        between simple threshold values.

    -- Note: That pixels equal to the threshold value are not set to 
        OutsideValue in any of lower or upper states and the pixels must 
        support the operators >= and <=.
    Args: 
        lower_threshold (float): Lower threshold constant value. The values 
            less than this parameter will be set to outside value. 
            The default is `0.0`.  
        upper_threshold (float): Upper threshold constant value. The values 
            greater than this parameter will be set to outside value. 
            The default is `1.0`.  
        outside_value (float): The pixel type must support comparison operators. 
            The default value is `0.0`.
        recalculate_mask (bool): If True and the mask is not equal None, the pixels
            which are not in the threshold range will be removed from the mask. 
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """
    def __init__(self, lower_threshold=0.0, upper_threshold=1.0, outside_value=0.0,
                 recalculate_mask=False, p=1.0):
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.outside_value = outside_value
        self.recalculate_mask = recalculate_mask
        self.p = p 
        self.trfm = sitk.ThresholdImageFilter()
        self.trfm.SetLower(self.lower_threshold)
        self.trfm.SetUpper(self.upper_threshold)
        self.trfm.SetOutsideValue(self.outside_value)

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, _ = apply_transform(self.trfm, image, None, *args,
                                       **kwargs)
            if self.recalculate_mask is True and mask is not None:
                upper_mask = sitk.GreaterEqual(image, self.lower_threshold)
                lower_mask = sitk.LessEqual(image, self.upper_threshold)
                new_mask = sitk.And(lower_mask, upper_mask)
                mask = sitk.And(mask, new_mask)
        return image, mask

    def __repr__(self):
        msg = '{} (lower_threshold={}, upper_threshold={}, outside_value={}, ' \
              'recalculate_mask={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.lower_threshold,
                          self.upper_threshold,
                          self.outside_value,
                          self.recalculate_mask,
                          self.p)


class IntensityLinearTransfer(object):
    ''' Apply a window intensity transformation.

        Applies a linear transformation to the intensity levels of the input
            image that are inside a user-defined interval. Values below this
            interval are mapped to a constant. Values over the interval are
            mapped to another constant.
    
    Args:
        window (sequence): Sequence  that contains lower bound and upper
        bound for the output image, (lower, upper).  Default sequence is
        `[-1024, 1024]`.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `0.5`.
    '''
    def __init__(self, window=[-1024, 1024], p=1.0):
        self.window = window
        self.p = p
        self.trfm = sitk.IntensityWindowingImageFilter()

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            information = get_stats(image)
            self.trfm.SetWindowMinimum(information['min'])
            self.trfm.SetWindowMaximum(information['max'])
            assert len(self.window) == 2, 'window must contain just lower ' \
                                          'bound and upper bound values.'
            self.trfm.SetOutputMinimum(self.window[0])
            self.trfm.SetOutputMaximum(self.window[1])
            image, _ = apply_transform(self.trfm, image, None, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (window={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.window,
                          self.p
        )


class AdaptiveHistogramEqualization(object):
    """ Histogram equalization modifies the contrast in an image.

        AdaptiveHistogramEqualization can produce an adaptively equalized
            histogram or a version of unsharp mask (local mean subtraction).
            Instead of applying a strict histogram equalization in a window
            about a pixel, this filter prescribes a mapping function (power law)
            controlled by the parameters alpha and beta.

    Args:
       alpha (float):  The parameter alpha controls how much the filter acts like
            the classical histogram equalization method (alpha=0) to how much the
            filter acts like an unsharp mask (alpha=1). Default value is `1.0`.
       beta (float): The parameter beta controls how much the filter acts like an
            unsharp mask (beta=0) to much the filter acts like pass through
            (beta=1, with alpha=1). Default value is `0.5'
       radius (int): The size of the window is controlled by SetRadius.
            The parameter window controls the size of the region over which local
            statistics are calculated. The default Radius is `2` in all directions.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `0.5`.
    """
    def __init__(self, alpha=1.0, beta=0.5, radius=2, p=0.5):
        self.alpha = alpha
        self.beta = beta
        self.radius = radius
        self.p = p

        self.trfm = sitk.AdaptiveHistogramEqualizationImageFilter()
        self.trfm.SetAlpha(self.alpha)
        self.trfm.SetBeta(self.beta)
        self.trfm.SetRadius(self.radius)

    def __call__(self, image, mask=None, *args, **kwargs):
        if random.random() <= self.p:
            image, _ = apply_transform(self.trfm, image, None, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (alpha={}, beta={}, radius={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.alpha,
                          self.beta,
                          self.radius,
                          self.p
        )


class MaskImage(object):
    """ Mask image with it's mask.

        The region of an image that the relevant part in mask is one will be
            returned as a masked image.

    Args:
        masking_value(float): The masking value of the mask. Defaults is `0`.
        -- note: Reversing the masking value can get the whole image but the 
            regions with the mask.  
        outside_value(float): The outside value of the mask. Defaults is `0`.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """
    def __init__(self, masking_value=0, outside_value=0, p=1.0):
        self.masking_value = masking_value
        self.outside_vale = outside_value
        self.p = p
        self.trfm = sitk.MaskImageFilter()
        self.trfm.SetMaskingValue(self.masking_value)
        self.trfm.SetOutsideValue(self.outside_vale)

    def __call__(self, image, mask, *args, **kwargs):
        check_dimensions(image, mask)
        assert mask is not None, 'mask can not be empty.'
        if random.random() <= self.p:
            image = self.trfm.Execute(image, mask)
        return image, mask

    def __repr__(self):
        msg = '{} (masking_value={}, outside_value={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.masking_value,
                          self.outside_vale,
                          self.p)


class LabelToRGB(object):
    """ Apply a colormap to a label image.

        The set of colors is a good selection of distinct colors. 
            The user can choose to use a background value. In that case,
            a gray pixel with the same intensity than the background label
            is produced.
    -- Note that this transformation takes no action on the input mask.
            But to be usable as a part of a pipeline, the src takes mask
            as its second input.
    Args: 
        colormap (int, sequence): The colormap with an integer value 
            or a sequence of 3 integer values. If the input is an integer, 
            It will used for all three color. 
        background (int): The background value for the image.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """
    def __init__(self, colormap, background, p=1.0): 
        self.colormap = colormap
        self.background = background
        self.p = p
        self.trfm = sitk.LabelToRGBImageFilter() 
        self.trfm.SetColormap(self.colormap)
        self.trfm.SetBackgroundValue(self.background)

    def __call__(self, image, mask, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image = self.trfm.Execute(image)
        return image, mask

    def __repr__(self): 
        msg = '{} (size={}, index={}, p={})'
        return msg.format(self.__class__.__name__, 
                          self.size, 
                          self.index, 
                          self.p)


class FillHole(object):
    """ Remove holes not connected to the boundary of the image.

        This transformation fills holes in a binary image.

    Args:
        fully_connected(bool):Set whether the connected components are defined
            strictly by face connectivity or by face+edge+vertex connectivity.
            Default is `False`. For objects that are 1 pixel wide, use `True`.
        foreground_value(float): Set the value in the image to consider as
            "foreground". Defaults to `maximum value` of InputPixelType if this
            this parameter is `None`.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """
    def __init__(self, fully_connected=False, foreground_value=None, p=1.0):
        self.fully_connected = fully_connected
        self.forground_value = foreground_value
        self.p = p
        self.trfm = sitk.BinaryFillholeImageFilter()
        if self.fully_connected:
            self.trfm.FullyConnectedOn()
        if self.forground_value:
            self.trfm.SetForegroundValue(self.forground_value)

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, mask = apply_transform(self.trfm, image, mask, *args,
                                          **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (fully connected={}, foreground value={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.fully_connected,
                          self.forground_value,
                          self.p)


class GrayScaleFillHole(object):
    """ Remove local minima not connected to the boundary of the image.

         Holes are local minima in the grayscale topography that are not 
             connected to boundaries of the image. Gray level values adjacent
             to a hole are extrapolated across the hole.
             This filter is used to smooth over local minima without affecting
             the values of local maxima. If you take the difference between the
             output of this filter and the original image (and perhaps threshold
             the difference above a small value), you'll obtain a map of the
             local minima.
    Args:
        fully_connected(bool):Set whether the connected components are defined
            strictly by face connectivity or by face+edge+vertex connectivity.
            Default is `False`. For objects that are 1 pixel wide, use `True`.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """
    def __init__(self, fully_connected=False, p=1.0):
        self.fully_connected = fully_connected
        self.p = p
        self.trfm = sitk.GrayscaleFillholeImageFilter()
        if self.fully_connected is True:
            self.trfm.FullyConnectedOn()

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, mask = apply_transform(self.trfm, image, mask, *args,
                                          **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (fully connected={}, p={})'
        return msg.format(self.__class__.__name__,
                          self.fully_connected,
                          self.p)


class LoadDicomDir(object):
    """ Load image and its corresponding mask from their addresses.

        Each address could be a directory address or the DICOM file address.
    Args:

    """
    def __init__(self):
        self.reader = sitk.ImageSeriesReader()

    def __call__(self, image_add, mask_add=None, *args, **kwargs):
        image, mask = None, None
        # Read image.
        try:
            if os.path.isdir(image_add):
                series_IDs = self.reader.GetGDCMSeriesIDs(image_add)
                if len(series_IDs) != 1:
                    raise ValueError(f'ValueError: There are {len(series_IDs)} '
                                     f'image series in image directory. It is needed '
                                     f'to be just one image series in each directory.')
                series_id = series_IDs[0]
                dicom_names = self.reader.GetGDCMSeriesFileNames(image_add,
                                                                 series_id)
                self.reader.SetFileNames(dicom_names)
                image = self.reader.Execute()
            elif os.path.isfile(image_add):
                image = sitk.ReadImage(image_add)
        except Exception as e:
            print(e)
        # Read mask.
        if mask_add is not None:
            try:
                if os.path.isdir(mask_add):
                    series_IDs = self.reader.GetGDCMSeriesIDs(mask_add)
                    if len(series_IDs) != 1:
                        raise ValueError(f'ValueError: There are {len(series_IDs)} '
                            f'mask series in mask directory. It is needed '
                            f'to be just one mask series in each directory.')
                    series_id = series_IDs[0]
                    dicom_names = self.reader.GetGDCMSeriesFileNames(mask_add,
                                                                     series_id)
                    self.reader.SetFileNames(dicom_names)
                    mask = self.reader.Execute()
                elif os.path.isfile(mask_add):
                    mask = sitk.ReadImage(mask_add)
            except Exception as e:
                    print(e)
        return image, mask

    def __repr__(self):
        msg = '{} ()'
        return msg.format(self.__class__.__name__)


class LoadDicomList(object):
    """Load image and its corresponding mask from their list of DICOM file addresses.
        DICOM list addresses need to be sorted in the correct order for reading.
    Args:

    """
    def __init__(self):
        self.reader = sitk.ImageSeriesReader()

    def __call__(self, image_dicom_list, mask_dicom_list=None, *args, **kwargs):
        image, mask = None, None
        self.reader.SetFileNames(image_dicom_list)
        image = self.reader.Execute()
        if mask_dicom_list is not None:
            self.reader.SetFileNames(mask_dicom_list)
            mask = self.reader.Execute()
        return image, mask

    def __repr__(self):
        msg = '{} ()'
        return msg.format(self.__class__.__name__)


class Compose(object):
    """ Compose multiple transforms and apply transforms over input data.
    
    Args: 
        transforms(list): List of transforms to be composed.
    """
    def __init__(self, transforms):
        assert isinstance(transforms, list)
        self.transforms = transforms

    def __call__(self, image, mask=None, *args, **kwargs):
        for tr in self.transforms:
            image, mask = tr(image, mask, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (transforms={})'
        return msg.format(self.__class__.__name__,
                          self.transforms)


class RandomChoices(object):
    """ Apply k randomly selected of the provided transformations.
    Args:
        transforms (list): List of transformation in which k transformations will
            be selected and then applied to the input images.
        k (int): Number of transformations to be selected.
        keep_original_order (boolean): After random selection keep the original order
            if True else just random selection will be apply.
    """
    def __init__(self, transforms, k, keep_original_order=True):
        self.transforms = transforms
        self.k = k
        self.keep_original_order = keep_original_order

    def __call__(self, image, mask=None, *args, **kwargs):
        assert self.k < len(self.transforms), 'k value is greater than number of transforms'
        if self.keep_original_order is True:
            temp = list(enumerate(self.transforms))
            temp = random.sample(temp, k=self.k)
            selected_trfms = sorted(temp, key=lambda x: x[0])
            _, selected_trfms = list(zip(*selected_trfms))
        else:
            selected_trfms = random.sample(self.transforms, k=self.k)
        for tr in selected_trfms:
            image, mask = tr(image, mask, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (transforms={}, k={})'
        return msg.format(self.__class__.__name__,
                          self.transforms,
                          self.k)


class OneOf(object):
    """ Apply one of the provided transformations.
    Args:
        transforms(list): List of transforms. One src will be selected
        by random and will be applied.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None, *args, **kwargs):
        tr = random.choice(self.transforms)
        image, mask = tr(image, mask, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (transforms]{})'
        return msg.format(self.__class__.__name__,
                          self.transforms)


class RandomOrder(object):
    """Apply a list of transformations in random order.

    Args:
        transforms(list): List of selected transforms to be applied in random order.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None, *args, **kwargs):
        random.shuffle(self.transforms)
        for tr in self.transforms:
            image, mask = tr(image, mask, *args, **kwargs)
        return image, mask

    def __repr__(self): 
        msg = '{} (transforms={})'
        return msg.format(self.__class__.__name__, 
                          self.transforms)


class Lambda(object):
    """ Apply a cutomized transformation.

    Args:
        lambd (function): Lambda/function to be used for src.
        sitk_lambda(bool): If true lambda function is aSimpleITK function,
            otherwise the lambda function must be numpy function.
        --note: Input images and mask are compatible with lambda function.
        inclue_mask(bool): If true function will be applied to the input
        mask. Otherwise function just apply to input image.
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """
    def __init__(self, lambd, sitk_lambda=True, include_mask=True, p=1.0):
        assert callable(lambd), repr(
            type(lambd).__name__) + ' object is not callable'
        self.lambd = lambd
        self.sitk_lambda = sitk_lambda
        self.include_mask = include_mask
        self.p = p 

    def __call__(self, image, mask=None, *args, **kwargs):
        if self.sitk_lambda:
            error_message = 'Input {} is not of type SimpleITK.SimpleITK.Image'
            assert isinstance(image, sitk.SimpleITK.Image), error_message.format('image')
            if self.include_mask:
                assert isinstance(mask, sitk.SimpleITK.Image), error_message.format('mask')
        else:
            error_message = 'Input {} is not of type numpy.ndarray'
            assert isinstance(image, np.ndarray), error_message.format('image')
            if self.include_mask:
                assert isinstance(mask, np.ndarray), error_message.format('mask')
        if random.random() <= self.p: 
            image = self.lambd(image)
            if mask is not None and self.include_mask is True:
                mask = self.lambd(mask)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + '()'


