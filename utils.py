import os
import random
import numpy as np
import SimpleITK as sitk
import torch


def visualizer(image, mask=None, out_dir = 'out', name='temp.nrrd'):
    os.makedirs(out_dir, exist_ok=True)
    sitk.WriteImage(image, os.path.join(out_dir, f'Image_{name}'))
    if mask is not None:
        sitk.WriteImage(mask, os.path.join(out_dir, f'Mask_{name}'))

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
    """Check the equality of the size of the SimpleITK image and the SimpleITK mask
       Check if the image is 3D image.
    """
    if mask is not None:
        assert image.GetSize() == mask.GetSize(), 'The size of The image is different from the size of the mask.'

def astype(image, image_type):
    return sitk.Cast(image, image_type)


class ABS(object):
    """Computes the absolute value of each pixel.

    Args:
        p (float): The transformation is applied to the input image with a
            probability of p. The default value is `1.0`.
    """
    def __init__(self, p=1.0):
        self.p = p
        self.trfm = sitk.AbsImageFilter()

    def __call__(self, image, mask=None, *args, **kwargs):
        check_dimensions(image, mask)
        if random.random() <= self.p:
            image, mask = self.trfm.Execute(image, *args, **kwargs)
        return image, mask

    def __repr__(self):
        msg = '{} (p={})'
        return msg.format(self.__class__.__name__,
                          self.p)


class Foreground(object):
    """
    """

    def __init__(self, inside_value=0, outside_value=1):
        self.tsfm = sitk.OtsuThresholdImageFilter()
        self.inside_value = inside_value
        self.outside_value = outside_value
        self.tsfm.SetInsideValue(self.inside_value)
        self.tsfm.SetOutsideValue(self.outside_value)

    def __call__(self, image):
        mask = self.tsfm.Execute(image)
        return mask

    def __repr__(self):
        msg = '{} (inside_value={}, outside_value={}'
        return msg.format(self.__class__.__name__,
                          self.inside_value,
                          self.outside_value)


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


class ToNumpy(object):
    """Convert an image and a mask to Numpy ndarray.
    Input image and mask must be Torch Tensor or SimpleITK Image.

    Args:
        out_image_dtype (Numpy dtype): Assign a new Numpy data type to the output image.
            The default value is `None`. This means the output image data type
            remains unchanged.
        out_mask_dtype (Numpy dtype): Assign a new Numpy data type to the output
            mask-image. The default value is `None`. This means the output mask-image
            data type remains unchanged.
    """
    def __init__(self, out_image_dtype=None, out_mask_dtype=None):
        self.img_dtype = out_image_dtype
        self.msk_dtype = out_mask_dtype

    def __call__(self, image, mask=None, *args, **kwargs):
        # Convert image and mask.
        if isinstance(image, torch.Tensor):
            image = image.numpy()
            if mask is not None:
                mask = mask.numpy()
        elif isinstance(image, sitk.SimpleITK.Image):
            image = sitk.GetArrayFromImage(image)
            if mask is not None:
                mask = sitk.GetArrayFromImage(mask)
        else:
            raise ValueError('image and mask must be in type torch Tensor or SimpleITK Image.')
        # Change image and mask dtypes.
        if self.img_dtype is not None:
            image = image.astype(self.img_dtype)
        if mask is not None and self.msk_dtype is not None:
            mask = mask.astype(self.msk_dtype)
        return image, mask

    def __repr__(self):
        msg = '{} (out_image_dtype={}, out_mask_dtype={})'
        return msg.format(self.__class__.__name__,
                          self.img_dtype,
                          self.msk_dtype)


class ToTensor(object):
    """Convert an image and a mask to Torch Tensor.
    Input image and mask must be Numpy ndarray or SimpleITK Image.

    Args:
        out_image_dtype (Torch dtype): Assign a new Torch data type to the output image.
            The default value is `None`. This means the output image data type
            remains unchanged.
        out_mask_dtype (Torch dtype): Assign a new Torch data type to the output
            mask-image. The default value is `None`. This means the output mask-image
            data type remains unchanged.
    """
    def __init__(self, out_image_dtype=None, out_mask_dtype=None):
        self.img_dtype = out_image_dtype
        self.msk_dtype = out_mask_dtype

    def __call__(self, image, mask=None, *args, **kwargs):
        # Convert image and mask.
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
            if mask is not None:
                mask = torch.from_numpy(mask)
        elif isinstance(image, sitk.SimpleITK.Image):
            image = torch.from_numpy(sitk.GetArrayFromImage(image))
            if mask is not None:
                mask = torch.from_numpy(sitk.GetArrayFromImage(mask))
        else:
            raise ValueError(
                'image and mask must be in type torch Tensor or SimpleITK Image.')
        # Change image and mask dtypes.
        if self.img_dtype is not None:
            image = image.type(self.img_dtype)
        if mask is not None and self.msk_dtype is not None:
            mask = mask.type(self.msk_dtype)
        return image, mask

    def __repr__(self):
        msg = '{} (out_image_dtype={}, out_mask_dtype={})'
        return msg.format(self.__class__.__name__,
                          self.img_dtype,
                          self.msk_dtype)


class ToSimpleITK(object):
    """Convert an image and a mask to SimpleITK Image.
    Input image and mask must be Numpy ndarray or Torch Tensor.

    Args:
        out_image_dtype (SimpleITK dtype): Assign a new SimpleITK data type to the output image.
            The default value is `None`. This means the output image data type
            remains unchanged.
        out_mask_dtype (SimpleITK dtype): Assign a new SimpleITK data type to the output
            mask-image. The default value is `None`. This means the output mask-image
            data type remains unchanged.
    """
    def __init__(self, out_image_dtype=None, out_mask_dtype=None):
        self.img_dtype = out_image_dtype
        self.msk_dtype = out_mask_dtype

    def __call__(self, image, mask=None, *args, **kwargs):
        # Convert image and mask.
        if isinstance(image, torch.Tensor):
            image = sitk.GetImageFromArray(image.numpy())
            if mask is not None:
                mask = sitk.GetImageFromArray(mask.numpy())
        elif isinstance(image, np.ndarray):
            image = sitk.GetImageFromArray(image)
            if mask is not None:
                mask = sitk.GetImageFromArray(mask)
        else:
            raise ValueError(
                'image and mask must be in type numpy array or torch Tensor.')
        # Change image and mask dtypes.
        if self.img_dtype is not None:
            image = sitk.Cast(image, self.img_dtype)
        if mask is not None and self.msk_dtype is not None:
            mask = sitk.Cast(mask, self.msk_dtype)
        return image, mask

    def __repr__(self):
        msg = '{} (out_image_dtype={}, out_mask_dtype={})'
        return msg.format(self.__class__.__name__,
                          self.img_dtype,
                          self.msk_dtype)


class To3D(object):
    """Convert 2D image to 3D
    Convert 2D SimpleITK image to 3D SimpleITK image with user defined depth.
    The 2D image will be copied, as a single layer of a 3D image, to produce
    the layers of the 3D image.

    Args:
        depth (int): Number of times to copy the 2D image. Defaults is `1`.
    """
    def __init__(self, depth=1):
        self.depth = depth
        assert depth > 0, 'depth must be greater than to 0.'
        self.trfm = sitk.JoinSeriesImageFilter()

    def __call__(self, image, mask=None, *args, **kwargs):
        assert isinstance(image, sitk.SimpleITK.Image), 'Input image is not SimpleITK image.'
        assert image.GetDimension() == 2, 'Input image must be two dimensional.'
        slices = [image for _ in range(self.depth)]
        image = self.trfm.Execute(slices)
        if mask is not None:
            assert isinstance(mask, sitk.SimpleITK.Image), 'Input mask is not SimpleITK image.'
            assert mask.GetDimension() == 2, 'Input mask must be two dimensional.'
            slices = [mask for _ in range(self.depth)]
            mask = self.trfm.Execute(slices)
        return image, mask

    def __repr__(self):
        msg = '{} (depth={})'
        return msg.format(self.__class__.__name__,
                          self.depth)

