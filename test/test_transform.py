import os
import tempfile
import unittest
import numpy as np

import SimpleITK as sitk

from src import transform as forge
from src.utils import image_equal

EPSILON = 1E-6
class TestTransform(unittest.TestCase):
    @staticmethod
    def load_cube(example: str = 'small') -> sitk.Image:
        """
        This module is used to load sample images used to test methods.
        Args:
            example: The ID of the image/mask pair to be returned.
                * small: This image is a 7x7x7 cube of intensity value of 1
                    centered inside a 11x11x11 canvas. The background intensity
                    is 0. The binary mask is the same as the image.
                * medium: This image is a 41x41x41 cube of intensity value of 1
                    centered inside a 101x101x101 canvas. The background
                    intensity is 0. The binary mask is the same as the image.
                * stripe: stripe image is the same as the medium image.
                    The mask, however, is different.
                    M[i, j, k] = i for 0 <= j, k < 20
                * hole: hole is a 20x20x20 image created as follows:
                   The image "I" and "M", which are 3D arrays of zie 20x20x20,
                   are initialized as zeros. Then I and M are updated as
                   follows:
                    I[3: 17, 3: 17, 3: 17] = 100
                    I[5: 15, 5: 15, 5: 15] = 50
                    I[9: 11, 9: 11, 9: 11] = 0
                    M[3: 17, 3: 17, 3: 17] = 2
                    M[5: 15, 5: 15, 5: 15] = 1
                    M[9: 11, 9: 11, 9: 11] = 0
        Returns:
            sitk.Image: an image and its mask.
            sitk.Image: a mask for the image.
        """
        name = {
            'small': 1,
            'medium': 2,
            'stripe': '_stripe_20x20',
            'hole': '_hole_20x20'}
        image_path = f'test/data/image{name[example]}.nrrd'
        mask_path = f'test/data/mask{name[example]}.nrrd'
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        return image, mask
    
    def test_Identity(self):
        image, mask = TestTransform.load_cube('hole')
        # Identity transformation without copying
        identity = forge.Identity(copy=False)
        img, msk = identity(image, mask)
        self.assertEqual(img, image)
        self.assertEqual(msk, mask)
        self.assertTrue(image_equal(image, img))
        self.assertTrue(image_equal(mask, msk))
        # Identity transformation with copying
        identity = forge.Identity(copy=True)
        img, msk = identity(image, mask)
        self.assertNotEqual(img, image)
        self.assertNotEqual(msk, mask)
        self.assertTrue(image_equal(image, img))
        self.assertTrue(image_equal(mask, msk))


    def test_Pad(self):
        image, mask = TestTransform.load_cube('hole')
        # Constant padding with inferred constant value
        PADING = [1, 1, 1]
        tsfm = forge.Pad(PADING, method='constant', constant=None,
                         background_label=3, pad_lower_bound=True,
                         pad_upper_bound=True, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(list(img.GetSize()), [x + 2 * pad for x, pad in zip(image.GetSize(), PADING)])
        self.assertEqual(list(msk.GetSize()), [x + 2 * pad for x, pad in zip(image.GetSize(), PADING)])
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(msk))), {0, 1, 2, 3})
        # Constant padding with predefined constant value
        PADING = [1, 1, 1]
        CONSTANT = 1024
        tsfm = forge.Pad(PADING, method='constant', constant=CONSTANT,
                         background_label=3, pad_lower_bound=True,
                         pad_upper_bound=True, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(list(img.GetSize()), [x + 2 * pad for x, pad in zip(image.GetSize(), PADING)])
        self.assertEqual(list(msk.GetSize()), [x + 2 * pad for x, pad in zip(image.GetSize(), PADING)])
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(msk))), {0, 1, 2, 3})
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(img))), {0, 50, 100, 1024, -1024})
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())
        # Test representation
        representation = ('{} (padding={}, method={}, constant={}, background_label={}, '
                          'pad_lower_bound={}, pad_upper_bound={}, p={})')
        representation = representation.format('Pad', PADING, 'constant', CONSTANT, 3, True, True, 1.0)
        self.assertEqual(str(tsfm), representation)
        # Mirror padding
        PADING = [1, 2, 3]
        tsfm = forge.Pad(PADING, method='mirror',
                         background_label=3, pad_lower_bound=True,
                         pad_upper_bound=True, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(list(img.GetSize()), [x + 2 * pad for x, pad in zip(image.GetSize(), PADING)])
        self.assertEqual(list(msk.GetSize()), [x + 2 * pad for x, pad in zip(image.GetSize(), PADING)])
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(msk))), {0, 1, 2, 3})
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(img))), {0, 50, 100, -1024})
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())
        # Wrap padding
        PADING = [3, 2, 1]
        tsfm = forge.Pad(PADING, method='mirror',
                         background_label=3, pad_lower_bound=True,
                         pad_upper_bound=True, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(list(img.GetSize()), [x + 2 * pad for x, pad in zip(image.GetSize(), PADING)])
        self.assertEqual(list(msk.GetSize()), [x + 2 * pad for x, pad in zip(image.GetSize(), PADING)])
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(msk))), {0, 1, 2, 3})
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(img))), {0, 50, 100, -1024})
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_ForegroundMask(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.ForegroundMask(background='<',
                                    bins=128)
        img, msk = tsfm(image)
        self.assertEqual(sitk.GetArrayFromImage(image).sum(),
                         sitk.GetArrayFromImage(msk).sum())
        # change the background choice
        tsfm = forge.ForegroundMask(background='>=',
                                    bins=128)
        img, msk = tsfm(image)
        self.assertEqual((1- sitk.GetArrayFromImage(image)).sum(),
                         sitk.GetArrayFromImage(msk).sum())
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_ForegroundMask_raises_Error(self):
        image, mask = TestTransform.load_cube('small')
        try:
            tsfm = forge.ForegroundMask(background='incorrect_value',
                                        bins=128)
            tsfm(image)
        except ValueError as e:
            msg = 'Valid background calculation values are:  <, <=, >, and >='
            self.assertEqual(str(e), msg)

    def test_ForegroundCrop(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.ForegroundCrop(background='<', bins=128)
        img, msk = tsfm(image)
        self.assertTrue(np.all(sitk.GetArrayFromImage(img) == 1))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_Af(self):
        image, mask = TestTransform.load_cube('hole')
        tsfm = forge.Affine(angles=(0, 0, 30), translation=(0, 0, 0),
                            scales=(1, 1, 1),
                            interpolator=sitk.sitkLinear, image_background=-1024,
                            mask_background=0, image_type=sitk.sitkInt16,
                            mask_type=sitk.sitkUInt8, spacing=None, direction=None,
                            reshape=True)


        img, msk = tsfm(image=image, mask=mask)

    def test_only_rotation_Affine(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.Affine(angles=(180, 0, 0), translation=(0, 0, 0),
                            scales=(1, 1, 1), interpolator=sitk.sitkLinear,
                            image_background=-1024, mask_background=0,
                            image_type=sitk.sitkInt16, mask_type=sitk.sitkUInt8,
                            reference=None)
        img, msk = tsfm(image=image, mask=mask)
        self.assertTrue(img.GetSize() == image.GetSize())
        self.assertTrue(msk.GetSize() == mask.GetSize())
        self.assertEqual(img.GetPixelIDValue(), sitk.sitkInt16)
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

        tsfm = forge.Affine(angles=(10, 10, 20), translation=(0, 0, 0), scales=(1, 1, 1),
                            interpolator=sitk.sitkLinear, image_background=-1024,
                            mask_background=0, reference=None, reshape=False)
        img, msk = tsfm(image=image, mask=mask)
        self.assertTrue(img.GetSize() == image.GetSize())
        self.assertTrue(msk.GetSize() == mask.GetSize())
        self.assertEqual(image.GetOrigin(), img.GetOrigin())
        self.assertEqual(img.GetPixelIDValue(), sitk.sitkInt16)
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

        tsfm = forge.Affine(angles=(10, 10, 20),
                            translation=(0, 0, 0), scales=(1, 1, 1),
                            interpolator=sitk.sitkLinear,
                            image_background=-1024, mask_background=0,
                            reference=None,
                            reshape=False)
        img, msk = tsfm(image=image, mask=mask)
        self.assertTrue(img.GetSize() == image.GetSize())
        self.assertTrue(msk.GetSize() == mask.GetSize())
        self.assertEqual(image.GetOrigin(), img.GetOrigin())
        self.assertEqual(img.GetPixelIDValue(), sitk.sitkInt16)
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_only_rotation_and_Scaling_Affine(self):
        image, mask = TestTransform.load_cube('small')
        SCALE = (2, 2, 2)
        tsfm = forge.Affine(angles=(1, 1, 1), translation=(0, 0, 0), scales=SCALE,
                            interpolator=sitk.sitkLinear, image_background=-1024,
                            mask_background=0, reference=None, reshape=True)
        img, msk = tsfm(image=image, mask=mask)
        expected_ime_size = SCALE * np.array(image.GetSize())


    def test_Flip_X_image_and_mask(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.Flip(axes=[True, False, False], p=1)
        img, msk = tsfm(image=image, mask=mask)
        self.assertTrue(TestTransform.checkFilip(image, mask,
                                                 img, msk, xflip=True,
                                                 image_only=False))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_Flip_X_image_only(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.Flip(axes=[True, False, False], p=1)
        img, msk = tsfm(image=image)
        self.assertTrue(TestTransform.checkFilip(image, mask,
                                                 img, msk, xflip=True,
                                                 image_only=True))

    def test_Flip_Y_image_and_mask(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.Flip(axes=[False, True, False], p=1)
        img, msk = tsfm(image=image, mask=mask)
        self.assertTrue(TestTransform.checkFilip(image, mask,
                                                 img, msk, yflip=True,
                                                 image_only=False))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_Flip_Y_image_only(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.Flip(axes=[False, True, False], p=1)
        img, msk = tsfm(image=image)
        self.assertTrue(TestTransform.checkFilip(image, mask,
                                                 img, msk, yflip=True,
                                                 image_only=True))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_Flip_Z_image_and_mask(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.Flip(axes=[False, False, True], p=1)
        img, msk = tsfm(image=image, mask=mask)
        self.assertTrue(TestTransform.checkFilip(image, mask,
                                                 img, msk, zflip=True,
                                                 image_only=False))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_Flip_Z_image_only(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.Flip(axes=[False, False, True], p=1)
        img, msk = tsfm(image=image)
        self.assertTrue(TestTransform.checkFilip(image, mask,
                                                 img, msk, zflip=True,
                                                 image_only=True))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_Flip_XY_image_and_mask(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.Flip(axes=[True, True, False], p=1)
        img, msk = tsfm(image=image, mask=mask)
        self.assertTrue(TestTransform.checkFilip(image, mask,
                                                 img, msk, xflip=True,
                                                 yflip=True, image_only=False))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_Flip_XY_image_only(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.Flip(axes=[True, True, False], p=1)
        img, msk = tsfm(image=image)
        self.assertTrue(TestTransform.checkFilip(image, mask,
                                                 img, msk, xflip=True,
                                                 yflip=True, image_only=True))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_Flip_XYZ_image_and_mask(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.Flip(axes=[True, True, True], p=1)
        img, msk = tsfm(image=image, mask=mask)
        self.assertTrue(TestTransform.checkFilip(image, mask,
                                                 img, msk, xflip=True,
                                                 yflip=True, zflip=True,
                                                 image_only=False))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_Flip_XYZ_image_only(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.Flip(axes=[True, True, True], p=1)
        img, msk = tsfm(image=image)
        self.assertTrue(TestTransform.checkFilip(image, mask,
                                                 img, msk, xflip=True,
                                                 yflip=True, zflip=True,
                                                 image_only=True))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_Flip_nothing_image_and_mask(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.Flip(axes=[False, False, False], p=1)
        img, msk = tsfm(image=image, mask=mask)
        self.assertTrue(TestTransform.checkFilip(image, mask,
                                                 img, msk, image_only=False))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_Flip_nothing_image_only(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.Flip(axes=[False, False, False], p=1)
        img, msk = tsfm(image=image)
        self.assertTrue(TestTransform.checkFilip(image, mask,
                                                 img, msk, image_only=True))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_RandomFlipX_image_and_mask(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.RandomFlipX(p=1)
        img, msk = tsfm(image=image, mask=mask)
        self.assertTrue(TestTransform.checkFilip(image, mask, img, msk,
                                                 xflip=True, image_only=False))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_RandomFlipX_image_only(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.RandomFlipX(p=1)
        img, msk = tsfm(image=image)
        self.assertTrue(TestTransform.checkFilip(image, mask,
                                                 img, msk, xflip=True,
                                                 image_only=True))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_RandomFlipY_image_and_mask(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.RandomFlipY(p=1)
        img, msk = tsfm(image=image, mask=mask)
        self.assertTrue(TestTransform.checkFilip(image, mask, img, msk,
                                                 yflip=True, image_only=False))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_RandomFlipY_image_only(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.RandomFlipY(p=1)
        img, msk = tsfm(image=image)
        self.assertTrue(TestTransform.checkFilip(image, mask, img, msk,
                                                 yflip=True, image_only=True))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_RandomFlipZ_image_and_mask(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.RandomFlipZ(p=1)
        img, msk = tsfm(image=image, mask=mask)
        self.assertTrue(TestTransform.checkFilip(image, mask, img, msk,
                                                 zflip=True, image_only=False))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_RandomFlipZ_image_only(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.RandomFlipZ(p=1)
        img, msk = tsfm(image=image)
        self.assertTrue(TestTransform.checkFilip(image, mask, img, msk,
                                                 zflip=True, image_only=True))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    @staticmethod
    def checkFilip(image, mask, img, msk,xflip: bool=False, yflip: bool=False,
                   zflip:bool=False, image_only: bool=False):
        # Flip should not change the size of image
        if image.GetSize() != img.GetSize():
            return False
        if image_only is False:
            # Flip should not change the size of mask
            if mask.GetSize() != msk.GetSize():
                return False
        x, y, z = image.GetSize()
        x_slice = slice(x, None, -1) if xflip else slice(0, x)
        y_slice = slice(y, None, -1) if yflip else slice(0, x)
        z_slice = slice(z, None, -1) if zflip else slice(0, x)
        # Check if image is flipped correctly
        data = sitk.GetArrayFromImage(image)
        data = data[z_slice, y_slice, x_slice]
        if np.array_equal(data, sitk.GetArrayFromImage(img)) is False:
            return False
        if image_only is True:
            if msk is not None:
                return False
        else:
            # Check if mask is flipped correctly
            data = sitk.GetArrayFromImage(mask)
            data = data[z_slice, y_slice, x_slice]
            if np.array_equal(data, sitk.GetArrayFromImage(msk)) is not True:
                return False
        return True

    def test_Crop(self):
        image, mask = TestTransform.load_cube('small')
        size = (2, 2, 2)
        index = (1, 1, 1)
        tsfm = forge.Crop(size, index)
        img, msk = tsfm(image, mask)
        self.assertEqual(img.GetSize(), size)
        self.assertEqual(msk.GetSize(), size)
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())
        # Crop only an image
        img, msk = tsfm(image)
        self.assertEqual(img.GetSize(), size)
        self.assertIsNone(msk)
        # Crop the whole image
        size = image.GetSize()
        index = (0, 0, 0)
        tsfm = forge.Crop(size, index)
        img, msk = tsfm(image, mask)
        self.assertEqual(img.GetSize(), size)
        self.assertEqual(msk.GetSize(), size)
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(image),
                                       sitk.GetArrayFromImage(img)))
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(mask),
                                       sitk.GetArrayFromImage(msk)))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())
        # Crop only an image
        img, msk = tsfm(image)
        self.assertEqual(img.GetSize(), size)
        self.assertIsNone(msk)
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(image),
                                       sitk.GetArrayFromImage(img)))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_Crop_not_possible(self):
        image, mask = TestTransform.load_cube('small')
        size = image.GetSize()
        index = (1, 1, 1)
        tsfm = forge.Crop(size, index)
        try:
            img, msk = tsfm(image, mask)
        except ValueError as e:
            msg = 'size + index cannot be greater than image size'
            self.assertEqual(str(e), msg)

    def test_RandomCrop(self):
        image, mask = TestTransform.load_cube('small')
        size = (2, 2, 2)
        tsfm = forge.RandomCrop(size, p=1)
        img, msk = tsfm(image, mask)
        self.assertEqual(img.GetSize(), size)
        self.assertEqual(msk.GetSize(), size)
        # Crop only an image
        img, msk = tsfm(image)
        self.assertEqual(img.GetSize(), size)
        self.assertIsNone(msk)
        # Crop the whole image
        size = image.GetSize()
        tsfm = forge.RandomCrop(size, p=1)
        img, msk = tsfm(image, mask)
        self.assertEqual(img.GetSize(), size)
        self.assertEqual(msk.GetSize(), size)
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(image),
                                       sitk.GetArrayFromImage(img)))
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(mask),
                                       sitk.GetArrayFromImage(msk)))
        # Crop only an image
        img, msk = tsfm(image)
        self.assertEqual(img.GetSize(), size)
        self.assertIsNone(msk)
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(image),
                                       sitk.GetArrayFromImage(img)))

    def test_RandomCrop_not_possible(self):
        image, mask = TestTransform.load_cube('small')
        msg = 'Copped region {name} cannot be larger than image {name}.'
        # ValueError for a crop width larger than image width
        size = list(image.GetSize())
        size[0] += 1
        tsfm = forge.RandomCrop(size, p=1)
        try:
            img, msk = tsfm(image, mask=mask)
        except ValueError as e:
            self.assertTrue(str(e), msg.format(name='width'))
        # ValueError for a crop height larger than image height
        size = list(image.GetSize())
        size[1] += 1
        tsfm = forge.RandomCrop(size, p=1)
        try:
            img, msk = tsfm(image, mask=mask)
        except ValueError as e:
            self.assertTrue(str(e), msg.format(name='heigh'))
        # ValueError for a crop depth larger than image depth
        size = list(image.GetSize())
        size[2] += 1
        tsfm = forge.RandomCrop(size, p=1)
        try:
            img, msk = tsfm(image, mask=mask)
        except ValueError as e:
            self.assertTrue(str(e), msg.format(name='depth'))

    def test_CenterCrop(self):
        image, mask = TestTransform.load_cube('small')
        size = (7, 7, 7)
        tsfm = forge.CenterCrop(size, p=1)
        img, msk = tsfm(image, mask)
        self.assertEqual(img.GetSize(), size)
        self.assertEqual(msk.GetSize(), size)
        self.assertTrue(np.all(sitk.GetArrayFromImage(img) == 1))
        self.assertTrue(np.all(sitk.GetArrayFromImage(msk) == 1))
        # Crop only for image
        img, msk = tsfm(image)
        self.assertEqual(img.GetSize(), size)
        self.assertIsNone(msk)
        self.assertTrue(np.all(sitk.GetArrayFromImage(img) == 1))

    def test_CenterCrop_not_possible(self):
        image, mask = TestTransform.load_cube()
        # Length of output_size and image dimension should be equal
        size = (2, 2)
        tsfm = forge.CenterCrop(size, p=1)
        try:
            img, msk = tsfm(image, mask)
        except ValueError as e:
            msg = 'length of size should be the same as image dimension'
            self.assertEqual(str(e), msg)
        # Crop size cannot be larger than image size
        size = list(image.GetSize())
        size[0] += 1
        tsfm = forge.CenterCrop(size, p=1)
        try:
            img, msk = tsfm(image, mask)
        except ValueError as e:
            msg = 'size cannot be larger than image size'
            self.assertEqual(str(e), msg)

    def test_SegmentSafeCrop(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.RandomSegmentSafeCrop(crop_size=(7, 7, 7), include=[1], p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(sitk.GetArrayFromImage(image).sum(),
                         sitk.GetArrayFromImage(img).sum())
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_SegmentSafeCrop_not_possible(self):
        image, mask = TestTransform.load_cube()
        # when crop_size is larger than the image
        crop_size = np.array(image.GetSize()) + 1
        tsfm = forge.RandomSegmentSafeCrop(crop_size=crop_size, include=[1], p=1.0)
        try:
            img, msk = tsfm(image, mask=mask)
        except ValueError as e:
            msg = 'crop_size must be less than or equal to image size.'
            self.assertEqual(str(e), msg)
        # When mask is missing
        crop_size = (7, 7, 7)
        tsfm = forge.RandomSegmentSafeCrop(crop_size=crop_size, include=[1], p=1.0)
        try:
            img, msk = tsfm(image, None)
        except ValueError as e:
            msg = 'SegmentSafeCrop requires an image and a mask.'
            self.assertEqual(str(e), msg)

    def test_SegmentSafeCrop_works_on_empty_masks(self):
        image, mask = TestTransform.load_cube()
        # When mask is empty
        crop_size = (7, 7, 7)
        tsfm = forge.RandomSegmentSafeCrop(crop_size=crop_size, include=[2], p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertTupleEqual(img.GetSize(), crop_size)
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_SegmentCrop_with_mask(self):
        image, mask = TestTransform.load_cube('hole')
        expected_crop_size = (14, 14, 14)
        #
        tsfm = forge.SegmentCrop(include=[1, 2], p=1.0, if_missing='raise')
        img, msk = tsfm(image, mask)
        self.assertEqual(img.GetSize(), expected_crop_size)
        self.assertEqual(msk.GetSize(), expected_crop_size)
        r = slice(3, 17)
        image_array = sitk.GetArrayFromImage(image)
        img_array = sitk.GetArrayFromImage(img)
        self.assertTrue(np.linalg.norm(image_array[r, r, r] - img_array) <
                        EPSILON)
        mask_array = sitk.GetArrayFromImage(mask)
        msk_array = sitk.GetArrayFromImage(msk)
        self.assertTrue(np.linalg.norm(mask_array[r, r, r] - msk_array) <
                        EPSILON)

    def test_SegmentCrop_without_mask_raise_Error(self):
        image, mask = TestTransform.load_cube('hole')
        #
        tsfm = forge.SegmentCrop(include=[1, 2], p=1.0, if_missing='raise')
        try:
            img, msk = tsfm(image, None)
        except ValueError as e:
            msg = 'SegmentCrop requires an image and a mask.'
            self.assertEqual(str(e), msg)

    def test_SegmentCrop_with_empty_region_raises_Error(self):
        image, mask = TestTransform.load_cube('hole')
        #
        INCLUDE = [3]
        tsfm = forge.SegmentCrop(include=INCLUDE, p=1.0, if_missing='raise')
        try:
            img, msk = tsfm(image, mask)
        except ValueError as e:
            included_str = ', '.join([str(x) for x in INCLUDE])
            msg = f'mask does not include any item from {included_str}'
            self.assertEqual(str(e), msg)

    def test_SegmentCrop_with_empty_region_no_action(self):
        image, mask = TestTransform.load_cube('hole')
        #
        INCLUDE = [10]
        tsfm = forge.SegmentCrop(include=INCLUDE, p=1.0, if_missing='ignore')
        img, msk = tsfm(image, mask)
        self.assertEqual(image, img)
        self.assertEqual(mask, msk)
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_Resize(self):
        image, mask = TestTransform.load_cube('small')
        output_size = [2 * x for x in image.GetSize()]
        tsfm = forge.Resize(size=output_size,
                            interpolator=sitk.sitkLinear,
                            default_image_voxel_value=0,
                            default_mask_voxel_value=0,
                            p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertTrue(np.array_equal(output_size, img.GetSize()))
        self.assertTrue(np.array_equal(output_size, msk.GetSize()))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_Resize_image_only(self):
        image, mask = TestTransform.load_cube('small')
        output_size = [2 * x for x in image.GetSize()]
        tsfm = forge.Resize(size=output_size,
                            interpolator=sitk.sitkLinear,
                            default_image_voxel_value=0,
                            default_mask_voxel_value=0,
                            p=1.0)
        img, msk = tsfm(image)
        self.assertTrue(np.array_equal(output_size, img.GetSize()))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_Resize_invalid_size_parameters_raise_error(self):
        image, mask = TestTransform.load_cube('small')
        try:
            output_size = (2, 2)
            tsfm = forge.Resize(size=output_size,
                                interpolator=sitk.sitkLinear,
                                default_image_voxel_value=0,
                                default_mask_voxel_value=0,
                                p=1.0)
            img, msk = tsfm(image, mask=mask)
        except ValueError as e:
            msg = 'Image dimension should be equal to 3.'
            self.assertEqual(str(e), msg)
        # Negative values in size
        try:
            output_size = (2, 2, 2)
            tsfm = forge.Resize(size=output_size,
                                interpolator=sitk.sitkLinear,
                                default_image_voxel_value=0,
                                default_mask_voxel_value=0,
                                p=1.0)
            img, msk = tsfm(image, mask=mask)
        except ValueError as e:
            msg = 'Image size cannot be zero or negative in any dimension'
            self.assertEqual(str(e), msg)

    def test_Expand(self):
        image, mask = TestTransform.load_cube('small')
        expansion_factors = [2, 2, 1]
        tsfm = forge.Expand(expansion=expansion_factors,
                            interpolator=sitk.sitkLinear, p=1)
        img, msk = tsfm(image, mask=mask)
        image_size = np.array(image.GetSize())
        self.assertTrue(np.array_equal(img.GetSize(),
                                       image_size * expansion_factors))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_Affine_RotationOnly(self):
        image, mask = TestTransform.load_cube('hole')
        tsfm = forge.Affine((0, 0, 45), translation=(0, 0, 0), scales=[1, 1, 1],
                            interpolator=sitk.sitkBSpline, image_background=-1024,
                            mask_background=0, reference=None)
        img, msk = tsfm(image, mask)

    def test_Expand_raise_Error(self):
        image, mask = TestTransform.load_cube('small')
        image = sitk.Cast(image, sitk.sitkFloat32)
        try:
            expansion_factors = [2, 2]
            tsfm = forge.Expand(expansion=expansion_factors,
                                interpolator=sitk.sitkLinear,
                                p=1)
            tsfm(image, mask=mask)
        except ValueError as e:
            msg = 'Image dimension must equal the length of expansion.'
            self.assertEqual(str(e), msg)

    def test_Shrink(self):
        image, mask = TestTransform.load_cube('small')
        shrinkage_factors = [2, 2, 1]
        tsfm = forge.Shrink(shrinkage=shrinkage_factors, p=1)
        img, msk = tsfm(image, mask=mask)
        image_size = np.array(image.GetSize())
        self.assertTrue(np.array_equal(img.GetSize(),
                                       image_size // shrinkage_factors))

    def test_Expand_raise_Error(self):
        image, mask = TestTransform.load_cube('small')
        try:
            expansion_factors = [2, 2]
            tsfm = forge.Expand(expansion=expansion_factors,
                                interpolator=sitk.sitkLinear,
                                p=1)
            tsfm(image, mask=mask)
        except ValueError as e:
            msg = 'Image dimension must equal the length of expansion.'
            self.assertEqual(str(e), msg)

    def test_Invert(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.Invert(maximum=1, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(image),
                                       1 - sitk.GetArrayFromImage(img)))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())
        # Inferring maximum when it is not provided
        tsfm = forge.Invert(p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(image),
                                       1 - sitk.GetArrayFromImage(img)))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())
        # Using a maximum value larger than the intensity values
        MAXIMUM = 255
        tsfm = forge.Invert(maximum=MAXIMUM, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(image),
                                       MAXIMUM - sitk.GetArrayFromImage(img)))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_BionomialBlur(self):
        image, mask = TestTransform.load_cube('small')
        image = sitk.Cast(image, sitk.sitkInt32)
        tsfm = forge.BionomialBlur(repetition=3, p=1.0)
        img, msk = tsfm(image, mask=mask)
        # BionomialBlur dose not change the mask
        self.assertTrue(np.all(sitk.GetArrayFromImage(mask) ==
                               sitk.GetArrayFromImage(msk)))
        self.assertEqual(image.GetSize(), img.GetSize())
        # The  input and output image for BionomialBlur are different
        self.assertFalse(np.all(sitk.GetArrayFromImage(image) ==
                                sitk.GetArrayFromImage(img)))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_SaltPepperNoise(self):
        image, mask = TestTransform.load_cube('small')
        image = sitk.Cast(image, sitk.sitkInt32)
        min_value, max_value = -1, 3
        tsfm = forge.SaltPepperNoise(noise_prob=0.2,
                                     noise_range=(min_value, max_value),
                                     random_seed=1, p=1.0)
        img, msk = tsfm(image, mask=mask)
        # SaltPepperNoise dose not change the mask
        self.assertTrue(np.all(sitk.GetArrayFromImage(mask) ==
                               sitk.GetArrayFromImage(msk)))
        self.assertEqual(image.GetSize(), img.GetSize())
        # The  input and output image for SaltPepperNoise are different
        image_array = sitk.GetArrayFromImage(image)
        img_array = sitk.GetArrayFromImage(img)

        self.assertFalse(np.all(image_array == img_array))
        self.assertEqual(img_array.max(), max_value)
        self.assertEqual(img_array.min(), min_value)
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())
        # No noise
        tsfm = forge.SaltPepperNoise(noise_prob=0, random_seed=1, p=1.0)
        img, msk = tsfm(image, mask=mask)
        # SaltPepperNoise dose not change the mask
        self.assertTrue(np.all(sitk.GetArrayFromImage(mask) ==
                               sitk.GetArrayFromImage(msk)))
        self.assertEqual(image.GetSize(), img.GetSize())
        # The  input and output image for SaltPepperNoise are different
        self.assertTrue(np.all(sitk.GetArrayFromImage(image) ==
                               sitk.GetArrayFromImage(img)))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_SaltPepperNoise_raisesError(self):
        image, mask = TestTransform.load_cube('small')
        image = sitk.Cast(image, sitk.sitkInt32)
        # min_value must be less than max_value
        try:
            min_value, max_value = 3, -1
            forge.SaltPepperNoise(noise_prob=0.2,
                                  noise_range=(min_value, max_value),
                                  random_seed=1,
                                  p=1.0)
            assert False
        except ValueError as e:
            msg = ('noise_range must be a tuple of size 2 representing'
                   'the lower and upper bounds of noise values')
            self.assertEqual(str(e), msg)

    def test_AdditiveGaussianNoise(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = forge.AdditiveGaussianNoise(mean=0.0, std=1.0, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(image.GetSize(), img.GetSize())
        self.assertEqual(mask.GetSize(), msk.GetSize())
        self.assertFalse(np.all(sitk.GetArrayFromImage(image) ==
                                sitk.GetArrayFromImage(img)))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_MinMaxScaler(self):
        image, mask = TestTransform.load_cube('small')
        MIN_VALUE = 0
        MAX_VALUE = 1
        tsfm = forge.MinMaxScaler(min_value=MIN_VALUE, max_value=MAX_VALUE, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(sitk.GetArrayFromImage(img).max(), MAX_VALUE)
        self.assertEqual(sitk.GetArrayFromImage(img).min(), MIN_VALUE)
        self.assertTrue(np.all(sitk.GetArrayFromImage(mask) ==
                               sitk.GetArrayFromImage(msk)))
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_MinMaxScaler_image_only(self):
        image, _ = TestTransform.load_cube('small')
        MIN_VALUE = 0
        MAX_VALUE = 255
        tsfm = forge.MinMaxScaler(min_value=MIN_VALUE, max_value=MAX_VALUE, p=1.0)
        img, msk = tsfm(image)
        self.assertEqual(sitk.GetArrayFromImage(img).max(), MAX_VALUE)
        self.assertEqual(sitk.GetArrayFromImage(img).min(), MIN_VALUE)
        self.assertIsNone(msk)

    def test_MinMaxScaler_raise_error(self):
        image, _ = TestTransform.load_cube('small')
        # min_value must be smaller than max_value
        try:
            MIN_VALUE = 1
            MAX_VALUE = 0
            tsfm = forge.MinMaxScaler(min_value=MIN_VALUE,
                                      max_value=MAX_VALUE,
                                      p=1.0)
            img, msk = tsfm(image)
            assert False
        except ValueError as e:
            msg = 'min_value must be smaller than max_value.'
            self.assertEqual(str(e), msg)

    def test_UnitNormalize(self):
        image, mask = TestTransform.load_cube('medium')
        tsfm = forge.UnitNormalize()
        img, msk = tsfm(image, mask)
        self.assertAlmostEqual(sitk.GetArrayFromImage(img).mean(), 0, 3)
        self.assertAlmostEqual(sitk.GetArrayFromImage(img).var(), 1, 2)
        self.assertTrue(np.all(sitk.GetArrayFromImage(mask) ==
                               sitk.GetArrayFromImage(msk)))
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_UnitNormalize_image_only(self):
        image, _ = TestTransform.load_cube('small')
        tsfm = forge.UnitNormalize()
        img, msk = tsfm(image)
        self.assertAlmostEqual(sitk.GetArrayFromImage(img).mean(), 0, 3)
        self.assertAlmostEqual(sitk.GetArrayFromImage(img).var(), 1, 2)

    def test_WindowLocationClip(self):
        image, mask = TestTransform.load_cube('stripe')
        LOCATION = 5
        WINDOW = 2
        tsfm = forge.WindowLocationClip(location=LOCATION, window=WINDOW)
        img, msk = tsfm(image, mask)
        # the stripe image is a 20x20x20 image, where for the
        #   image[i, 5:15, 5:15] == i and the rest of voxels are zero.
        img_array = sitk.GetArrayFromImage(img)
        self.assertEqual(set(np.unique(img_array)), {3, 4, 5, 6, 7})
        self.assertEqual(set(np.unique(img_array[:LOCATION-WINDOW])),
                         {3})
        self.assertEqual(set(np.unique(img_array[LOCATION+WINDOW+1:])),
                         {3, 7})
        self.assertEqual(img.GetSize(), image.GetSize())
        # This transformation does not affect the mask
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(msk),
                                       sitk.GetArrayFromImage(mask)))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_WindowLocationClip_image_only(self):
        image, _ = TestTransform.load_cube('stripe')
        LOCATION = 5
        WINDOW = 2
        tsfm = forge.WindowLocationClip(location=LOCATION, window=WINDOW)
        img, msk = tsfm(image)
        # the stripe image is a 20x20x20 image, where for the
        #   image[i, 5:15, 5:15] == i and the rest of voxels are zero.
        img_array = sitk.GetArrayFromImage(img)
        self.assertEqual(set(np.unique(img_array)), {3, 4, 5, 6, 7})
        self.assertEqual(set(np.unique(img_array[:LOCATION-WINDOW])),
                         {3})
        self.assertEqual(set(np.unique(img_array[LOCATION+WINDOW+1:])),
                         {3, 7})
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertIsNone(msk)
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_Clip(self):
        image, mask = TestTransform.load_cube('stripe')
        LOWER, UPPER = 3, 7
        tsfm = forge.Clip(lower_bound=LOWER, upper_bound=UPPER, p=1.0)
        img, msk = tsfm(image, mask)
        # the stripe image is a 20x20x20 image, where image[i, 5:15, 5:15] == i
        # and the rest of voxels are zero.
        img_array = sitk.GetArrayFromImage(img)
        self.assertEqual(set(np.unique(img_array)), set(range(LOWER, UPPER+1)))
        self.assertEqual(set(np.unique(img_array[:LOWER])),
                         {3})
        self.assertEqual(set(np.unique(img_array[UPPER + 1:])),
                         {3, 7})
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(msk),
                                       sitk.GetArrayFromImage(mask)))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_Clip_image_only(self):
        image, _ = TestTransform.load_cube('stripe')
        LOWER, UPPER = 3, 7
        tsfm = forge.Clip(lower_bound=LOWER, upper_bound=UPPER, p=1.0)
        img, msk = tsfm(image)
        # the stripe image is a 20x20x20 image, where image[i, 5:15, 5:15] == i
        # and the rest of voxels are zero.
        img_array = sitk.GetArrayFromImage(img)
        self.assertEqual(set(np.unique(img_array)), set(range(LOWER, UPPER+1)))
        self.assertEqual(set(np.unique(img_array[:LOWER])),
                         {3})
        self.assertEqual(set(np.unique(img_array[UPPER + 1:])),
                         {3, 7})
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertIsNone(msk)
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_ThresholdClip(self):
        image, mask = TestTransform.load_cube('hole')
        LOWER = 0
        UPPER = 51
        OUTSIDE = 0
        tsfm = forge.IsolateRange(lower_bound=LOWER,
                                  upper_bound=UPPER,
                                  image_outside_value=OUTSIDE,
                                  recalculate_mask=False,
                                  p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(mask),
                                       sitk.GetArrayFromImage(msk)))
        img_array = sitk.GetArrayFromImage(img)
        self.assertEqual(set(np.unique(img_array)), {0, 50})
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_ThresholdClip_update_mask(self):
        image, mask = TestTransform.load_cube('hole')
        LOWER = 0
        UPPER = 51
        OUTSIDE = 0
        tsfm = forge.IsolateRange(lower_bound=LOWER,
                                  upper_bound=UPPER,
                                  image_outside_value=OUTSIDE,
                                  mask_outside_value=OUTSIDE,
                                  recalculate_mask=True,
                                  p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(msk))),
                         {0, 1})
        img_array = sitk.GetArrayFromImage(img)
        self.assertEqual(set(np.unique(img_array)), {0, 50})
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_ThresholdClip_image_only(self):
        image, _ = TestTransform.load_cube('hole')
        LOWER = 0
        UPPER = 51
        OUTSIDE = 0
        tsfm = forge.IsolateRange(lower_bound=LOWER,
                                  upper_bound=UPPER,
                                  image_outside_value=OUTSIDE,
                                  recalculate_mask=False,
                                  p=1.0)
        img, msk = tsfm(image)
        self.assertEqual(img.GetSize(), image.GetSize())
        img_array = sitk.GetArrayFromImage(img)
        self.assertEqual(set(np.unique(img_array)), {0, 50})
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_ThresholdClip_raise_error(self):
        image, _ = TestTransform.load_cube('hole')
        LOWER = 0
        UPPER = -1
        OUTSIDE = 0
        try:
            forge.IsolateRange(lower_bound=LOWER,
                               upper_bound=UPPER,
                               image_outside_value=OUTSIDE,
                               recalculate_mask=False,
                               p=1.0)
            assert False
        except ValueError as e:
            msg = 'lower_bound must be smaller than upper_bound.'
            self.assertEqual(str(e), msg)

    def test_IntensityRangeTransfer(self):
        image, mask = TestTransform.load_cube('hole')
        LOWER = 0
        UPPER = 1
        tsfm = forge.IntensityRangeTransfer(interval=(LOWER, UPPER),
                                            cast=None, p=1.0)
        img, msk = tsfm(image, mask)
        self.assertEqual(image.GetSize(), img.GetSize())
        img_array = sitk.GetArrayFromImage(img)
        self.assertAlmostEqual(img_array.max(), UPPER)
        self.assertAlmostEqual(img_array.min(), LOWER)
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(msk),
                                       sitk.GetArrayFromImage(mask)))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_IntensityRangeTransfer_with_casting(self):
        image, _ = TestTransform.load_cube('hole')
        LOWER = 0
        UPPER = 1
        TYPE = sitk.sitkFloat32
        tsfm = forge.IntensityRangeTransfer(interval=(LOWER, UPPER),
                                            cast=TYPE, p=1.0)
        img, msk = tsfm(image)
        self.assertEqual(image.GetSize(), img.GetSize())
        img_array = sitk.GetArrayFromImage(img)
        self.assertAlmostEqual(img_array.max(), UPPER)
        self.assertAlmostEqual(img_array.min(), LOWER)
        self.assertIsNone(msk)
        self.assertEqual(img.GetPixelIDValue(), TYPE)

    def test_IntensityRangeTransfer_image_only(self):
        image, _ = TestTransform.load_cube('hole')
        LOWER = 0
        UPPER = 1
        tsfm = forge.IntensityRangeTransfer(interval=(LOWER, UPPER),
                                            cast=None, p=1.0)
        img, msk = tsfm(image)
        self.assertEqual(image.GetSize(), img.GetSize())
        img_array = sitk.GetArrayFromImage(img)
        self.assertAlmostEqual(img_array.max(), UPPER)
        self.assertAlmostEqual(img_array.min(), LOWER)
        self.assertIsNone(msk)
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_AdaptiveHistogramEqualization(self):
        image, mask = TestTransform.load_cube('medium')
        tsfm = forge.AdaptiveHistogramEqualization(alpha=1.0, beta=0.5,
                                                   radius=2, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(mask),
                                       sitk.GetArrayFromImage(msk)))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_AdaptiveHistogramEqualization_image_only(self):
        image, _ = TestTransform.load_cube('medium')
        tsfm = forge.AdaptiveHistogramEqualization(alpha=1.0, beta=0.5,
                                                   radius=2, p=1.0)
        img, msk = tsfm(image)
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertIsNone(msk)
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())

    def test_MaskImage(self):
        image, mask = TestTransform.load_cube('hole')
        LABEL = 1
        OUTSIDE_MASK_LABEL = 0
        BACKGROUND = -1024
        tsfm = forge.MaskImage(segment_label=LABEL,
                               image_outside_value=BACKGROUND,
                               mask_outside_label=OUTSIDE_MASK_LABEL,
                               p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(msk))),
                         {LABEL, OUTSIDE_MASK_LABEL})
        self.assertEqual(img.GetSize(), image.GetSize())
        img_array = sitk.GetArrayFromImage(img)
        msk_array = sitk.GetArrayFromImage(msk)
        self.assertEqual(np.asscalar(np.unique(img_array[msk_array == LABEL])),
                         BACKGROUND)
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_MaskImage_raise_error(self):
        image, mask = TestTransform.load_cube('hole')
        LABEL = 1
        BACKGROUND = -1024
        try:
            tsfm = forge.MaskImage(segment_label=LABEL,
                                   image_outside_value=BACKGROUND,
                                   p=1.0)
            tsfm(image, mask=None)
        except ValueError as e:
            msg = 'mask cannot be None for AdaptiveHistogramEqualization.'
            self.assertEqual(str(e), msg)


    def test_BinaryFillHole(self):
        image, mask = TestTransform.load_cube('hole')
        tsfm = forge.BinaryFillHole(foreground_value=2, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(msk))),
                                   {0, 2})
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(img),
                                       sitk.GetArrayFromImage(image)))
        tsfm = forge.BinaryFillHole(foreground_value=1, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(msk))),
                         {0, 1, 2})
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(img),
                                       sitk.GetArrayFromImage(image)))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_FillHole_raise_error(self):
        image, mask = TestTransform.load_cube('hole')
        try:
            tsfm = forge.BinaryFillHole(foreground_value=2, p=1.0)
            tsfm(image, mask=None)
        except ValueError as e:
            msg = 'mask cannot be None.'
            self.assertEqual(str(e), msg)

    def test_Reader(self):
        image, mask = TestTransform.load_cube('small')
        image_path = 'test/data/image1.nrrd'
        mask_path = 'test/data/mask1.nrrd'
        reader = forge.Reader()
        img, msk = reader(image_path=image_path, mask_path=mask_path)
        self.assertTrue(image_equal(image, img))
        self.assertTrue(image_equal(mask, msk))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())

    def test_Writer(self):
        image, mask = TestTransform.load_cube('small')
        writer = forge.SequentialWriter(dir_path='test/data', image_prefix='TempImage',
                                        image_postfix='___', mask_prefix='TempMask',
                                        mask_postfix='___', extension='nrrd')
        temp_image_path, temp_mask_path = writer(image, mask)
        reader = forge.Reader()
        img, msk = reader(image_path=temp_image_path, mask_path=temp_mask_path)
        self.assertTrue(image_equal(image, img))
        self.assertTrue(image_equal(mask, msk))
        self.assertEqual(img.GetPixelIDValue(), image.GetPixelIDValue())
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())
        os.remove(temp_image_path)
        os.remove(temp_mask_path)

    def test_MaskLabelRemap(self):
        image, mask = TestTransform.load_cube('hole')
        remap = forge.MaskLabelRemap({100: 0, 50: 0, 2: 1})
        img, msk = remap(image, mask)
        self.assertEqual(img, image)
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(msk))), {0, 1})
        # Remap Mask with blank image
        remap = forge.MaskLabelRemap({100: 0, 50: 0, 2: 1})
        img, msk = remap(mask=mask)
        self.assertIsNone(img)
        self.assertEqual(msk.GetPixelIDValue(), mask.GetPixelIDValue())
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(msk))), {0, 1})

    def test_BinaryErode(self):
        image, mask = TestTransform.load_cube('hole')
        FOREGROUND = 1
        BACKGROUND = 0
        eroder = forge.BinaryErode(background=BACKGROUND,
                                   foreground=FOREGROUND,
                                   radius=(1, 1, 1))
        img, msk = eroder(image, mask)
        msk_segment = sitk.GetArrayFromImage(msk)
        mask_segment = sitk.GetArrayFromImage(mask)
        msk_segment[msk_segment != FOREGROUND] = 0
        mask_segment[mask_segment != FOREGROUND] = 0
        self.assertGreater(mask_segment.sum(), msk_segment.sum())

    def test_BinaryDilate(self):
        image, mask = TestTransform.load_cube('hole')
        FOREGROUND = 1
        BACKGROUND = 0
        dilate = forge.BinaryDilate(background=BACKGROUND,
                                    foreground=FOREGROUND,
                                    radius=(1, 1, 1))
        img, msk = dilate(image, mask)
        msk_segment = sitk.GetArrayFromImage(msk)
        mask_segment = sitk.GetArrayFromImage(mask)
        msk_segment[msk_segment != FOREGROUND] = 0
        mask_segment[mask_segment != FOREGROUND] = 0
        self.assertGreater(msk_segment.sum(), mask_segment.sum())

    def test_Resample(self):
        image, mask = TestTransform.load_cube('hole')
        output_spacing = (0.5, 2, 1)
        params = {'interpolator': sitk.sitkLinear,
                  'output_spacing': output_spacing,
                  'default_image_voxel_value': 100,
                  'default_mask_voxel_value': 0,
                  'output_image_voxel_type': sitk.sitkInt16,
                  'output_mask_voxel_type': sitk.sitkUInt8,
                  'output_direction': None,
                  'output_origin': None,
                  'use_nearest_neighbor_extrapolator': False}
        resampler = forge.Resample(**params)
        img, msk = resampler(image, mask)
        self.assertEqual(img.GetSize(), msk.GetSize())
        self.assertSequenceEqual(img.GetSize(),
                                 [x / s for x, s in zip(image.GetSize(),
                                                        output_spacing)])

    def test_Isotropic(self):
        image, mask = TestTransform.load_cube('hole')
        output_spacing = 0.5
        params = {'interpolator': sitk.sitkLinear,
                  'output_spacing': output_spacing,
                  'default_image_voxel_value': -1024,
                  'default_mask_voxel_value': 0,
                  'output_image_voxel_type': sitk.sitkInt16,
                  'output_mask_voxel_type': sitk.sitkUInt8,
                  'output_origin': None,
                  'use_nearest_neighbor_extrapolator': True}
        resampler = forge.Isotropic(**params)
        img, msk = resampler(image, mask)
        self.assertEqual(img.GetSize(), msk.GetSize())
        self.assertSequenceEqual(img.GetSize(),
                                 [x / output_spacing for x in image.GetSize()])
    
    def test_ToNumpy(self):
        image, mask = TestTransform.load_cube('hole')
        convertor = forge.ToNumpy()
        img_array, msk_array = convertor(image, mask)
        self.assertTrue(np.array_equal(img_array,
                                       sitk.GetArrayFromImage(image)))
        self.assertTrue(np.array_equal(msk_array,
                                       sitk.GetArrayFromImage(mask)))
    
    def test_FromNumpy(self):
        dimension = (10, 20, 30)
        image_array = np.random.randn(*dimension)
        mask_array = np.random.randint(2, size=dimension)
        image = sitk.GetImageFromArray(image_array)
        mask = sitk.GetImageFromArray(mask_array)
        convertor = forge.ToNumpy()
        img, msk = convertor(image, mask)
        self.assertTrue(np.array_equal(image_array, img))
        self.assertTrue(np.array_equal(mask_array, msk))


    def test_Compose(self):
        ids = [i for i in range(5)]
        tsfms = [MockTransform(i) for i in ids]
        tsfm = forge.Compose(tsfms)
        image, mask = [], []
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img, ids)
        self.assertEqual(msk, ids)

    def test_From2DTo3D(self):
        address = 'test/data/Lung2D.png'
        image = sitk.ReadImage(address)
        tsfm = forge.From2DTo3D(repeat=1)
        image, mask = tsfm(image=image, mask=image)
        self.assertEqual(image.GetDimension(), 3)
        self.assertEqual(image.GetSize(), (596, 592, 1))
        self.assertEqual(mask.GetDimension(), 3)

    def test_From3DTo2D_3_channel(self):
        width, height, depth = 10, 20, 3
        image_array = np.arange(width * height * depth).reshape((depth,
                                                                 height,
                                                                 width))
        image3D = sitk.GetImageFromArray(image_array, isVector=False)
        assert image3D.GetSize() == (width, height, depth)
        tsfm = forge.From3DTo2D(image_pixel_type=sitk.sitkVectorInt32,
                                mask_pixel_type=sitk.sitkUInt8)
        img, msk = tsfm(image3D, image3D)
        self.assertEqual(img.GetWidth(), width)
        self.assertEqual(img.GetHeight(), height)
        self.assertEqual(msk.GetWidth(), width)
        self.assertEqual(msk.GetHeight(), height)

    def test_From3DTo2D_1_channel(self):
        width, height, depth = 10, 20, 1
        image_array = np.arange(width * height * depth).reshape((depth,
                                                                 height,
                                                                 width))
        image3D = sitk.GetImageFromArray(image_array, isVector=False)
        assert image3D.GetSize() == (width, height, depth)
        tsfm = forge.From3DTo2D(image_pixel_type=sitk.sitkVectorInt32,
                                mask_pixel_type=sitk.sitkUInt8)
        img, msk = tsfm(image3D, image3D)
        self.assertEqual(img.GetWidth(), width)
        self.assertEqual(img.GetHeight(), height)
        self.assertEqual(msk.GetWidth(), width)
        self.assertEqual(msk.GetHeight(), height)
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(img),
                                       sitk.GetArrayFromImage(image3D)[0]))

    def test_Writer_Reader_image_and_mask(self):
        image, mask = TestTransform.load_cube('small')
        image_path = 'test/data/image4testingReaderWriter.nii'
        mask_path = 'test/data/mask4testingReaderWriter.nii'
        tsfm = forge.Writer()
        tsfm(image, image_path=image_path,
             mask=mask, mask_path=mask_path)
        tsfm = forge.Reader()
        img, msk = tsfm(image_path=image_path, mask_path=mask_path)
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(image),
                                       sitk.GetArrayFromImage(img)))
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(mask),
                                       sitk.GetArrayFromImage(msk)))
        os.remove(image_path)
        os.remove(mask_path)

    def test_RandomChoices(self):
        ids = [i for i in range(5)]
        tsfms = [MockTransform(i) for i in ids]
        image, mask = [], []
        K = 3
        tsfm = forge.RandomChoices(tsfms, k=K, keep_original_order=True)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(len(img), K)
        self.assertEqual(img, msk)
        self.assertEqual(img, sorted(image))
        # No need to keep original order
        image, mask = [], []
        tsfm = forge.RandomChoices(tsfms, k=K, keep_original_order=False)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(len(img), K)
        self.assertEqual(img, msk)
        self.assertTrue(set(img).issubset(set(ids)))

    def test_OneOf(self):
        ids = [i for i in range(5)]
        tsfms = [MockTransform(i) for i in ids]
        image, mask = [], []
        tsfm = forge.OneOf(tsfms)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img, mask)
        self.assertEqual(len(img), 1)
        self.assertTrue(set(img).issubset(set(ids)))

    def test_RandomOrder(self):
        ids = [i for i in range(5)]
        tsfms = [MockTransform(i) for i in ids]
        image, mask = [], []
        tsfm = forge.RandomOrder(tsfms)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img, msk)
        self.assertEqual(set(img), set(ids))

    def test_Lambda(self):
        def function(x=[]):
            x.append(len(x))
            return x
        tsfm = forge.Lambda(image_transformer=function,
                            mask_transformer=function, p=1)
        image, mask = [], []
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img, [0])
        self.assertEqual(msk, [0])
        
    def test_refrence_free_3D_resample(self):
        pass

    def test_referenced_3D_resample(self):
        pass

class MockTransform(object):
    """ This class is used for testing Transform objects.

    Args:
        identifier: An identifier assigned to the MockTransform object.
    """

    def __init__(self, identifier):
        self.identifier = identifier

    def __call__(self, image: list, mask: list = []):
        """ Append an element to the end of image and mask lists.

        Args:
            image (list): It could be a list of arbitrary values.
            image (list): It could be a list of arbitrary values.

        Returns:
            list: the list resulting from appending the object identifier to the image.
            list: the list resulting from appending the object identifier to the mask.
        """
        assert isinstance(image, list)
        if mask is not None:
            assert isinstance(image, list)
        assert isinstance(mask, list)
        image.append(self.identifier)
        mask.append(self.identifier)
        return image, mask
