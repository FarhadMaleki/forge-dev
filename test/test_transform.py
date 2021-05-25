import unittest
import numpy as np

import SimpleITK as sitk

import transform as tr

EPSILON = 1E-6
class TestTransform(unittest.TestCase):

    @staticmethod
    def load_cube(example='small'):
        """
        This image is a 7x7x7 cube of intensity value of 1
        inside a 11x11x11 canvas. The background intensity
        is 0.
        size: small, medium
                # the stripe image is a 20x20x20 image, where for the
        #   hole is a 20x20x20 image created as as follows:
                I = numpy.zeros((20, 20, 20), dtype=np.uint8) - 1024
                I[3: 17, 3: 17, 3: 17] = 100
                I[5: 15, 5: 15, 5: 15] = 50
                I[9: 11, 9: 11, 9: 11] = 0
                M = numpy.zeros((20, 20, 20), dtype=np.uint8)
                M[3: 17, 3: 17, 3: 17] = 2
                M[5: 15, 5: 15, 5: 15] = 1
                M[9: 11, 9: 11, 9: 11] = 0

            M = np.zeros((20, 20, 20), dtype=np.uint) -1024
            X = np.zeros
            I = 100 *(x-w) - 1024(
            layered is 20x20x20 image I, where I[
        """
        name = {'medium': 2,
                'small': 1,
                'stripe': '_stripe_20x20',
                'hole': '_hole_20x20'}
        image_path = f'test/data/image{name[example]}.nrrd'
        mask_path = f'test/data/mask{name[example]}.nrrd'
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        return image, mask

    def setUp(self):
        image_path = 'test/data/image2.nrrd'
        mask_path = 'test/data/mask2.nrrd'
        self.image = sitk.ReadImage(image_path)
        self.mask = sitk.ReadImage(mask_path)

    def test_ForegroundMask(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = tr.ForegroundMask(background='<',
                                 bins=128)
        img, msk = tsfm(image)
        self.assertEqual(sitk.GetArrayFromImage(image).sum(),
                         sitk.GetArrayFromImage(msk).sum())
        # change the background choice
        tsfm = tr.ForegroundMask(background='>=',
                                 bins=128)
        img, msk = tsfm(image)
        self.assertEqual((1- sitk.GetArrayFromImage(image)).sum(),
                         sitk.GetArrayFromImage(msk).sum())

    def test_ForegroundMask_raises_Error(self):
        image, mask = TestTransform.load_cube('small')
        try:
            tsfm = tr.ForegroundMask(background='incorrect_value',
                                     bins=128)
            tsfm(image)
        except ValueError as e:
            msg = 'Valid background calculation values are:  <, <=, >, and >='
            self.assertEqual(str(e), msg)

    def test_ForegroundCrop(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = tr.ForegroundCrop(background='<', bins=128)
        img, msk = tsfm(image)
        self.assertTrue(np.all(sitk.GetArrayFromImage(img) == 1))

    def test_only_rotation_Affine(self):

        tsfm = tr.Affine(angles=180, translation=0, scale=1,
                         interpolator=sitk.sitkLinear, image_background=-1024,
                         mask_background=0, reference=None, p=1)
        img, msk = tsfm(image=self.image, mask=self.mask)
        self.assertTrue(img.GetSize() == self.image.GetSize())
        self.assertTrue(msk.GetSize() == self.mask.GetSize())
        self.assertEqual(self.image.GetOrigin(), img.GetOrigin())

        tsfm = tr.Affine(angles=(10, 10, 20), translation=0, scale=1,
                         interpolator=sitk.sitkLinear, image_background=-1024,
                         mask_background=0, reference=None, p=1)
        img, msk = tsfm(image=self.image, mask=self.mask)
        self.assertTrue(img.GetSize() == self.image.GetSize())
        self.assertTrue(msk.GetSize() == self.mask.GetSize())
        self.assertEqual(self.image.GetOrigin(), img.GetOrigin())

        tsfm = tr.Affine(angles=((-10, 10), (0, 10), (-50, 20)),
                         translation=0, scale=1, interpolator=sitk.sitkLinear,
                         image_background=-1024, mask_background=0,
                         reference=None, p=1)
        img, msk = tsfm(image=self.image, mask=self.mask)
        self.assertTrue(img.GetSize() == self.image.GetSize())
        self.assertTrue(msk.GetSize() == self.mask.GetSize())
        self.assertEqual(self.image.GetOrigin(), img.GetOrigin())

    def test_only_rotation_and_Scaling_Affine(self):
        SCALE = 2
        tsfm = tr.Affine(angles=1, translation=0, scale=SCALE,
                         interpolator=sitk.sitkLinear, image_background=-1024,
                         mask_background=0, reference=None, p=1)
        img, msk = tsfm(image=self.image, mask=self.mask)
        expected_ime_size = SCALE * np.array(self.image.GetSize())
        "TODO: Solve the issue with scale"
        # self.assertTrue(np.array_equal(img.GetSize(), expected_ime_size))
        # # self.assertItemsEqual(expected_ime_size)
        # self.assertTrue(img.GetSize() == tuple(expected_ime_size))
        # self.assertTrue(msk.GetSize() == self.mask.GetSize())
        # self.assertEqual(self.image.GetOrigin(), img.GetOrigin())
        #
        # tsfm = tr.Affine(angles=(10, 10, 20), translation=0, scale=1,
        #                  interpolator=sitk.sitkLinear, image_background=-1024,
        #                  mask_background=0, reference=None, p=1)
        # img, msk = tsfm(image=self.image, mask=self.mask)
        # self.assertTrue(img.GetSize() == self.image.GetSize())
        # self.assertTrue(msk.GetSize() == self.mask.GetSize())
        # self.assertEqual(self.image.GetOrigin(), img.GetOrigin())
        #
        # tsfm = tr.Affine(angles=((-10, 10), (0, 10), (-50, 20)),
        #                  translation=0, scale=1, interpolator=sitk.sitkLinear,
        #                  image_background=-1024, mask_background=0,
        #                  reference=None, p=1)
        # img, msk = tsfm(image=self.image, mask=self.mask)
        # self.assertTrue(img.GetSize() == self.image.GetSize())
        # self.assertTrue(msk.GetSize() == self.mask.GetSize())
        # self.assertEqual(self.image.GetOrigin(), img.GetOrigin())

    def test_Flip_X_image_and_mask(self):
        tsfm = tr.Flip(axes=[True, False, False], p=1)
        img, msk = tsfm(image=self.image, mask=self.mask)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask,
                                                 img, msk, xflip=True,
                                                 image_only=False))

    def test_Flip_X_image_only(self):
        tsfm = tr.Flip(axes=[True, False, False], p=1)
        img, msk = tsfm(image=self.image)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask,
                                                 img, msk, xflip=True,
                                                 image_only=True))

    def test_Flip_Y_image_and_mask(self):
        tsfm = tr.Flip(axes=[False, True, False], p=1)
        img, msk = tsfm(image=self.image, mask=self.mask)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask,
                                                 img, msk, yflip=True,
                                                 image_only=False))

    def test_Flip_Y_image_only(self):
        tsfm = tr.Flip(axes=[False, True, False], p=1)
        img, msk = tsfm(image=self.image)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask,
                                                 img, msk, yflip=True,
                                                 image_only=True))

    def test_Flip_Z_image_and_mask(self):
        tsfm = tr.Flip(axes=[False, False, True], p=1)
        img, msk = tsfm(image=self.image, mask=self.mask)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask,
                                                 img, msk, zflip=True,
                                                 image_only=False))

    def test_Flip_Z_image_only(self):
        tsfm = tr.Flip(axes=[False, False, True], p=1)
        img, msk = tsfm(image=self.image)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask,
                                                 img, msk, zflip=True,
                                                 image_only=True))

    def test_Flip_XY_image_and_mask(self):
        tsfm = tr.Flip(axes=[True, True, False], p=1)
        img, msk = tsfm(image=self.image, mask=self.mask)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask,
                                                 img, msk, xflip=True,
                                                 yflip=True, image_only=False))

    def test_Flip_XY_image_only(self):
        tsfm = tr.Flip(axes=[True, True, False], p=1)
        img, msk = tsfm(image=self.image)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask,
                                                 img, msk, xflip=True,
                                                 yflip=True, image_only=True))

    def test_Flip_XYZ_image_and_mask(self):
        tsfm = tr.Flip(axes=[True, True, True], p=1)
        img, msk = tsfm(image=self.image, mask=self.mask)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask,
                                                 img, msk, xflip=True,
                                                 yflip=True, zflip=True, image_only=False))

    def test_Flip_XYZ_image_only(self):
        tsfm = tr.Flip(axes=[True, True, True], p=1)
        img, msk = tsfm(image=self.image)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask,
                                                 img, msk, xflip=True,
                                                 yflip=True, zflip=True, image_only=True))

    def test_Flip_nothing_image_and_mask(self):
        tsfm = tr.Flip(axes=[False, False, False], p=1)
        img, msk = tsfm(image=self.image, mask=self.mask)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask,
                                                 img, msk, image_only=False))

    def test_Flip_nothing_image_only(self):
        tsfm = tr.Flip(axes=[False, False, False], p=1)
        img, msk = tsfm(image=self.image)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask,
                                                 img, msk, image_only=True))

    def test_RandomFlipX_image_and_mask(self):
        tsfm = tr.RandomFlipX(p=1)
        img, msk = tsfm(image=self.image, mask=self.mask)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask, img, msk,
                                                 xflip=True, image_only=False))

    def test_RandomFlipX_image_only(self):
        tsfm = tr.RandomFlipX(p=1)
        img, msk = tsfm(image=self.image)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask,
                                                 img, msk, xflip=True,
                                                 image_only=True))

    def test_RandomFlipY_image_and_mask(self):
        tsfm = tr.RandomFlipY(p=1)
        img, msk = tsfm(image=self.image, mask=self.mask)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask, img, msk,
                                                 yflip=True, image_only=False))

    def test_RandomFlipY_image_only(self):
        tsfm = tr.RandomFlipY(p=1)
        img, msk = tsfm(image=self.image)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask, img, msk,
                                                 yflip=True, image_only=True))

    def test_RandomFlipZ_image_and_mask(self):
        tsfm = tr.RandomFlipZ(p=1)
        img, msk = tsfm(image=self.image, mask=self.mask)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask, img, msk,
                                                 zflip=True, image_only=False))

    def test_RandomFlipZ_image_only(self):
        tsfm = tr.RandomFlipZ(p=1)
        img, msk = tsfm(image=self.image)
        self.assertTrue(TestTransform.checkFilip(self.image, self.mask, img, msk,
                                                 zflip=True, image_only=True))

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
        size = (2, 2, 2)
        index = (1, 1, 1)
        tsfm = tr.Crop(size, index)
        img, msk = tsfm(self.image, self.mask)
        self.assertEqual(img.GetSize(), size)
        self.assertEqual(msk.GetSize(), size)
        # Crop only an image
        img, msk = tsfm(self.image)
        self.assertEqual(img.GetSize(), size)
        self.assertIsNone(msk)
        # Crop the whole image
        size = self.image.GetSize()
        index = (0, 0, 0)
        tsfm = tr.Crop(size, index)
        img, msk = tsfm(self.image, self.mask)
        self.assertEqual(img.GetSize(), size)
        self.assertEqual(msk.GetSize(), size)
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(self.image),
                                       sitk.GetArrayFromImage(img)))
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(self.mask),
                                       sitk.GetArrayFromImage(msk)))
        # Crop only an image
        img, msk = tsfm(self.image)
        self.assertEqual(img.GetSize(), size)
        self.assertIsNone(msk)
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(self.image),
                                       sitk.GetArrayFromImage(img)))

    def test_Crop_not_possible(self):
        size = self.image.GetSize()
        index = (1, 1, 1)
        tsfm = tr.Crop(size, index)
        try:
            img, msk = tsfm(self.image, self.mask)
        except ValueError as e:
            msg = 'size + index cannot be greater than image size'
            self.assertEqual(str(e), msg)

    def test_RandomCrop(self):
        size = (2, 2, 2)
        tsfm = tr.RandomCrop(size, p=1)
        img, msk = tsfm(self.image, self.mask)
        self.assertEqual(img.GetSize(), size)
        self.assertEqual(msk.GetSize(), size)
        # Crop only an image
        img, msk = tsfm(self.image)
        self.assertEqual(img.GetSize(), size)
        self.assertIsNone(msk)
        # Crop the whole image
        size = self.image.GetSize()
        tsfm = tr.RandomCrop(size, p=1)
        img, msk = tsfm(self.image, self.mask)
        self.assertEqual(img.GetSize(), size)
        self.assertEqual(msk.GetSize(), size)
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(self.image),
                                       sitk.GetArrayFromImage(img)))
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(self.mask),
                                       sitk.GetArrayFromImage(msk)))
        # Crop only an image
        img, msk = tsfm(self.image)
        self.assertEqual(img.GetSize(), size)
        self.assertIsNone(msk)
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(self.image),
                                       sitk.GetArrayFromImage(img)))

    def test_RandomCrop_not_possible(self):
        msg = 'Copped region {name} cannot be larger than image {name}.'
        # ValueError for a crop width larger than image width
        size = list(self.image.GetSize())
        size[0] += 1
        tsfm = tr.RandomCrop(size, p=1)
        try:
            img, msk = tsfm(self.image, mask=self.mask)
        except ValueError as e:
            self.assertTrue(str(e), msg.format(name='width'))
        # ValueError for a crop height larger than image height
        size = list(self.image.GetSize())
        size[1] += 1
        tsfm = tr.RandomCrop(size, p=1)
        try:
            img, msk = tsfm(self.image, mask=self.mask)
        except ValueError as e:
            self.assertTrue(str(e), msg.format(name='heigh'))
        # ValueError for a crop depth larger than image depth
        size = list(self.image.GetSize())
        size[2] += 1
        tsfm = tr.RandomCrop(size, p=1)
        try:
            img, msk = tsfm(self.image, mask=self.mask)
        except ValueError as e:
            self.assertTrue(str(e), msg.format(name='depth'))

    def test_CenterCrop(self):
        image, mask = TestTransform.load_cube()
        size = (7, 7, 7)
        tsfm = tr.CenterCrop(size, p=1)
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
        tsfm = tr.CenterCrop(size, p=1)
        try:
            img, msk = tsfm(image, mask)
        except ValueError as e:
            msg = 'length of size should be the same as image dimension'
            self.assertEqual(str(e), msg)
        # Crop size cannot be larger than image size
        size = list(image.GetSize())
        size[0] += 1
        tsfm = tr.CenterCrop(size, p=1)
        try:
            img, msk = tsfm(image, mask)
        except ValueError as e:
            msg = 'size cannot be larger than image size'
            self.assertEqual(str(e), msg)

    def test_SegmentSafeCrop(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = tr.SegmentSafeCrop(crop_size=(7, 7, 7), include=[1], p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(sitk.GetArrayFromImage(image).sum(), sitk.GetArrayFromImage(img).sum())

    def test_SegmentSafeCrop_not_possible(self):
        image, mask = TestTransform.load_cube()
        # when crop_size is larger than the image
        crop_size = np.array(image.GetSize()) + 1
        tsfm = tr.SegmentSafeCrop(crop_size=crop_size, include=[1], p=1.0)
        try:
            img, msk = tsfm(image, mask=mask)
        except ValueError as e:
            msg = 'crop_size must be less than or equal to image size.'
            self.assertEqual(str(e), msg)
        # When mask is missing
        crop_size = (7, 7, 7)
        tsfm = tr.SegmentSafeCrop(crop_size=crop_size, include=[1], p=1.0)
        try:
            img, msk = tsfm(image)
        except ValueError as e:
            msg = 'SegmentSafeCrop requires a mask.'
            self.assertEqual(str(e), msg)

    def test_SegmentSafeCrop_works_on_empty_masks(self):
        image, mask = TestTransform.load_cube()
        # When mask is empty
        crop_size = (7, 7, 7)
        tsfm = tr.SegmentSafeCrop(crop_size=crop_size, include=[2], p=1.0)
        img, msk = tsfm(self.image, mask=self.mask)
        self.assertTupleEqual(img.GetSize(), crop_size)

    def test_Resize(self):
        output_size = [2 * x for x in self.image.GetSize()]
        tsfm = tr.Resize(size=output_size,
                         interpolator=sitk.sitkLinear,
                         default_image_pixel_value=0,
                         default_mask_pixel_value=0,
                         p=1.0)
        img, msk = tsfm(self.image, mask=self.mask)
        self.assertTrue(np.array_equal(output_size, img.GetSize()))
        self.assertTrue(np.array_equal(output_size, msk.GetSize()))

    def test_Resize_image_only(self):
        output_size = [2 * x for x in self.image.GetSize()]
        tsfm = tr.Resize(size=output_size,
                         interpolator=sitk.sitkLinear,
                         default_image_pixel_value=0,
                         default_mask_pixel_value=0,
                         p=1.0)
        img, msk = tsfm(self.image)
        self.assertTrue(np.array_equal(output_size, img.GetSize()))

    def test_Resize_invalid_size_parameters_raise_error(self):
        try:
            output_size = (2, 2)
            tsfm = tr.Resize(size=output_size,
                             interpolator=sitk.sitkLinear,
                             default_image_pixel_value=0,
                             default_mask_pixel_value=0,
                             p=1.0)
            img, msk = tsfm(self.image, mask=self.mask)
        except ValueError as e:
            msg = 'Image dimension should be equal to 3.'
            self.assertEqual(str(e), msg)
        # Negative values in size
        try:
            output_size = (2, 2, 2)
            tsfm = tr.Resize(size=output_size,
                             interpolator=sitk.sitkLinear,
                             default_image_pixel_value=0,
                             default_mask_pixel_value=0,
                             p=1.0)
            img, msk = tsfm(self.image, mask=self.mask)
        except ValueError as e:
            msg = 'Image size cannot be zero or negative in any dimension'
            self.assertEqual(str(e), msg)

    def test_Expand(self):
        image, mask = TestTransform.load_cube('small')
        expansion_factors = [2, 2, 1]
        tsfm = tr.Expand(expansion=expansion_factors,
                         interpolator=sitk.sitkLinear, p=1)
        img, msk = tsfm(image, mask=mask)
        image_size = np.array(image.GetSize())
        self.assertTrue(np.array_equal(img.GetSize(),
                                       image_size * expansion_factors))

    def test_Expand_raise_Error(self):
        image, mask = TestTransform.load_cube('small')
        image = sitk.Cast(image, sitk.sitkFloat32)
        try:
            expansion_factors = [2, 2]
            tsfm = tr.Expand(expansion=expansion_factors,
                             interpolator=sitk.sitkLinear,
                             p=1)
            tsfm(image, mask=mask)
        except ValueError as e:
            msg = 'Image dimension must equal the length of expansion.'
            self.assertEqual(str(e), msg)

    def test_Shrink(self):
        image, mask = TestTransform.load_cube('small')
        shrinkage_factors = [2, 2, 1]
        tsfm = tr.Shrink(shrinkage=shrinkage_factors, p=1)
        img, msk = tsfm(image, mask=mask)
        image_size = np.array(image.GetSize())
        self.assertTrue(np.array_equal(img.GetSize(),
                                       image_size // shrinkage_factors))

    def test_Expand_raise_Error(self):
        image, mask = TestTransform.load_cube('small')
        try:
            expansion_factors = [2, 2]
            tsfm = tr.Expand(expansion=expansion_factors,
                             interpolator=sitk.sitkLinear,
                             p=1)
            tsfm(image, mask=mask)
        except ValueError as e:
            msg = 'Image dimension must equal the length of expansion.'
            self.assertEqual(str(e), msg)

    def test_Invert(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = tr.Invert(maximum=1, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(image),
                                       1 - sitk.GetArrayFromImage(img)))
        # Inferring maximum when it is not provided
        tsfm = tr.Invert(p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(image),
                                       1 - sitk.GetArrayFromImage(img)))
        # Using a maximum value larger than the intensity values
        MAXIMUM = 255
        tsfm = tr.Invert(maximum=MAXIMUM, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(image),
                                       MAXIMUM - sitk.GetArrayFromImage(img)))

    def test_BionomialBlur(self):
        image, mask = TestTransform.load_cube('small')
        image = sitk.Cast(image, sitk.sitkInt32)
        tsfm = tr.BionomialBlur(repetition=3, p=1.0)
        img, msk = tsfm(image, mask=mask)
        # BionomialBlur dose not change the mask
        self.assertTrue(np.all(sitk.GetArrayFromImage(mask) ==
                               sitk.GetArrayFromImage(msk)))
        self.assertEqual(image.GetSize(), img.GetSize())
        # The  input and output image for BionomialBlur are different
        self.assertFalse(np.all(sitk.GetArrayFromImage(image) ==
                                sitk.GetArrayFromImage(img)))

    def test_SaltPepperNoise(self):
        image, mask = TestTransform.load_cube('small')
        image = sitk.Cast(image, sitk.sitkInt32)
        min_value, max_value = -1, 3
        tsfm = tr.SaltPepperNoise(noise_prob=0.2,
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
        # No noise
        tsfm = tr.SaltPepperNoise(noise_prob=0, random_seed=1, p=1.0)
        img, msk = tsfm(image, mask=mask)
        # SaltPepperNoise dose not change the mask
        self.assertTrue(np.all(sitk.GetArrayFromImage(mask) ==
                               sitk.GetArrayFromImage(msk)))
        self.assertEqual(image.GetSize(), img.GetSize())
        # The  input and output image for SaltPepperNoise are different
        self.assertTrue(np.all(sitk.GetArrayFromImage(image) ==
                               sitk.GetArrayFromImage(img)))

    def test_SaltPepperNoise_raisesError(self):
        image, mask = TestTransform.load_cube('small')
        image = sitk.Cast(image, sitk.sitkInt32)
        # min_value must be less than max_value
        try:
            min_value, max_value = 3, -1
            tsfm = tr.SaltPepperNoise(noise_prob=0.2,
                                      noise_range=(min_value, max_value),
                                      random_seed=1, p=1.0)
            raise ValueError
        except ValueError as e:
            msg = ('noise_range must be a tuple of size 2 representing'
                   'the lower and upper bounds of noise values')
            self.assertEqual(str(e), msg)

    def test_AdditiveGaussianNoise(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = tr.AdditiveGaussianNoise(mean=0.0, std=1.0, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(image.GetSize(), img.GetSize())
        self.assertEqual(mask.GetSize(), msk.GetSize())
        self.assertFalse(np.all(sitk.GetArrayFromImage(image) ==
                                sitk.GetArrayFromImage(img)))

    def test_LocalNoiseCalculator(self):
        image, mask = TestTransform.load_cube('small')
        tsfm = tr.LocalNoiseCalculator(radius=1)
        img = tsfm(image)
        self.assertTrue(np.all(sitk.GetArrayFromImage(img) == 0))

    def test_MinMaxScaler(self):
        image, mask = TestTransform.load_cube('small')
        MIN_VALUE = 0
        MAX_VALUE = 1
        tsfm = tr.MinMaxScaler(min_value=MIN_VALUE, max_value=MAX_VALUE, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(sitk.GetArrayFromImage(img).max(), MAX_VALUE)
        self.assertEqual(sitk.GetArrayFromImage(img).min(), MIN_VALUE)
        self.assertTrue(np.all(sitk.GetArrayFromImage(mask) ==
                               sitk.GetArrayFromImage(msk)))

    def test_MinMaxScaler_image_only(self):
        image, _ = TestTransform.load_cube('small')
        MIN_VALUE = 0
        MAX_VALUE = 255
        tsfm = tr.MinMaxScaler(min_value=MIN_VALUE, max_value=MAX_VALUE, p=1.0)
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
            tsfm = tr.MinMaxScaler(min_value=MIN_VALUE,
                                   max_value=MAX_VALUE,
                                   p=1.0)
            img, msk = tsfm(image)
            assert False
        except ValueError as e:
            msg = 'min_value must be smaller than max_value.'
            self.assertEqual(str(e), msg)

    def test_UnitNormalize(self):
        image, mask = TestTransform.load_cube('medium')
        tsfm = tr.UnitNormalize()
        img, msk = tsfm(image, mask)
        self.assertAlmostEqual(sitk.GetArrayFromImage(img).mean(), 0, 3)
        self.assertAlmostEqual(sitk.GetArrayFromImage(img).var(), 1, 2)
        self.assertTrue(np.all(sitk.GetArrayFromImage(mask) ==
                               sitk.GetArrayFromImage(msk)))

    def test_UnitNormalize_image_only(self):
        image, _ = TestTransform.load_cube('small')
        tsfm = tr.UnitNormalize()
        img, msk = tsfm(image)
        self.assertAlmostEqual(sitk.GetArrayFromImage(img).mean(), 0, 3)
        self.assertAlmostEqual(sitk.GetArrayFromImage(img).var(), 1, 2)

    def test_WindowLocationClip(self):
        image, mask = TestTransform.load_cube('stripe')
        LOCATION = 5
        WINDOW = 2
        tsfm = tr.WindowLocationClip(location=LOCATION, window=WINDOW)
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

    def test_WindowLocationClip_image_only(self):
        image, _ = TestTransform.load_cube('stripe')
        LOCATION = 5
        WINDOW = 2
        tsfm = tr.WindowLocationClip(location=LOCATION, window=WINDOW)
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

    def test_Clip(self):
        image, mask = TestTransform.load_cube('stripe')
        LOWER, UPPER = 3, 7
        tsfm = tr.Clip(lower_bound=LOWER, upper_bound=UPPER, p=1.0)
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

    def test_Clip_image_only(self):
        image, _ = TestTransform.load_cube('stripe')
        LOWER, UPPER = 3, 7
        tsfm = tr.Clip(lower_bound=LOWER, upper_bound=UPPER, p=1.0)
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

    def test_ThresholdClip(self):
        image, mask = TestTransform.load_cube('hole')
        LOWER = 0
        UPPER = 51
        OUTSIDE = 0
        tsfm = tr.ThresholdClip(lower_bound=LOWER,
                                upper_bound=UPPER,
                                outside_value=OUTSIDE,
                                recalculate_mask=False,
                                p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(mask),
                                       sitk.GetArrayFromImage(msk)))
        img_array = sitk.GetArrayFromImage(img)
        self.assertEqual(set(np.unique(img_array)), {0, 50})

    def test_ThresholdClip_update_mask(self):
        image, mask = TestTransform.load_cube('hole')
        LOWER = 0
        UPPER = 51
        OUTSIDE = 0
        tsfm = tr.ThresholdClip(lower_bound=LOWER,
                                upper_bound=UPPER,
                                outside_value=OUTSIDE,
                                recalculate_mask=True,
                                p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(msk))),
                         {0, 1})
        img_array = sitk.GetArrayFromImage(img)
        self.assertEqual(set(np.unique(img_array)), {0, 50})

    def test_ThresholdClip_image_only(self):
        image, _ = TestTransform.load_cube('hole')
        LOWER = 0
        UPPER = 51
        OUTSIDE = 0
        tsfm = tr.ThresholdClip(lower_bound=LOWER,
                                upper_bound=UPPER,
                                outside_value=OUTSIDE,
                                recalculate_mask=False,
                                p=1.0)
        img, msk = tsfm(image)
        self.assertEqual(img.GetSize(), image.GetSize())
        img_array = sitk.GetArrayFromImage(img)
        self.assertEqual(set(np.unique(img_array)), {0, 50})

    def test_ThresholdClip_raise_error(self):
        image, _ = TestTransform.load_cube('hole')
        LOWER = 0
        UPPER = -1
        OUTSIDE = 0
        try:
            tr.ThresholdClip(lower_bound=LOWER,
                             upper_bound=UPPER,
                             outside_value=OUTSIDE,
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
        tsfm = tr.IntensityRangeTransfer(window=(LOWER, UPPER), cast=None, p=1.0)
        img, msk = tsfm(image, mask)
        self.assertEqual(image.GetSize(), img.GetSize())
        img_array = sitk.GetArrayFromImage(img)
        self.assertAlmostEqual(img_array.max(), UPPER)
        self.assertAlmostEqual(img_array.min(), LOWER)
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(msk),
                                       sitk.GetArrayFromImage(mask)))

    def test_IntensityRangeTransfer_with_casting(self):
        image, _ = TestTransform.load_cube('hole')
        LOWER = 0
        UPPER = 1
        TYPE = sitk.sitkFloat32
        tsfm = tr.IntensityRangeTransfer(window=(LOWER, UPPER), cast=TYPE, p=1.0)
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
        tsfm = tr.IntensityRangeTransfer(window=(LOWER, UPPER), cast=None, p=1.0)
        img, msk = tsfm(image)
        self.assertEqual(image.GetSize(), img.GetSize())
        img_array = sitk.GetArrayFromImage(img)
        self.assertAlmostEqual(img_array.max(), UPPER)
        self.assertAlmostEqual(img_array.min(), LOWER)
        self.assertIsNone(msk)

    def test_AdaptiveHistogramEqualization(self):
        image, mask = TestTransform.load_cube('medium')
        tsfm = tr.AdaptiveHistogramEqualization(alpha=1.0, beta=0.5, radius=2, p=1.0)
        img, msk = tsfm(image, mask=mask)
        sitk.WriteImage(img, 'test/data/out.nrrd')
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(mask),
                                       sitk.GetArrayFromImage(msk)))

    def test_AdaptiveHistogramEqualization_image_only(self):
        image, _ = TestTransform.load_cube('medium')
        tsfm = tr.AdaptiveHistogramEqualization(alpha=1.0, beta=0.5, radius=2, p=1.0)
        img, msk = tsfm(image)
        self.assertEqual(img.GetSize(), image.GetSize())
        self.assertIsNone(msk)

    def test_MaskImage(self):
        image, mask = TestTransform.load_cube('hole')
        LABEL = 1
        BACKGROUND = -1024
        tsfm = tr.MaskImage(segment_label=LABEL,
                            outside_value=BACKGROUND,
                            p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(mask),
                                       sitk.GetArrayFromImage(msk)))
        self.assertEqual(img.GetSize(), image.GetSize())
        img_array = sitk.GetArrayFromImage(img)
        msk_array = sitk.GetArrayFromImage(msk)
        self.assertEqual(np.asscalar(np.unique(img_array[msk_array == LABEL])),
                         BACKGROUND)

    def test_MaskImage_raise_error(self):
        image, mask = TestTransform.load_cube('hole')
        LABEL = 1
        BACKGROUND = -1024
        try:
            tsfm = tr.MaskImage(segment_label=LABEL,
                                outside_value=BACKGROUND,
                                p=1.0)
            tsfm(image, mask=None)
        except ValueError as e:
            msg = 'mask cannot be None for AdaptiveHistogramEqualization.'
            self.assertEqual(str(e), msg)

    def test_LabelToRGB(self):
        pass

    def test_FillHole(self):
        image, mask = TestTransform.load_cube('hole')
        tsfm = tr.FillHole(fully_connected=False, foreground_value=2, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(msk))),
                                   {0, 2})
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(img),
                                       sitk.GetArrayFromImage(image)))
        tsfm = tr.FillHole(fully_connected=False, foreground_value=1, p=1.0)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(msk))),
                         {0, 1, 2})
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(img),
                                       sitk.GetArrayFromImage(image)))

    def test_FillHole_raise_error(self):
        image, mask = TestTransform.load_cube('hole')
        try:
            tsfm = tr.FillHole(fully_connected=False, foreground_value=2, p=1.0)
            tsfm(image, mask=None)
        except ValueError as e:
            msg = 'mask cannot be None.'
            self.assertEqual(str(e), msg)

    def test_ReadFromPath(self):
        #TODO:
        pass

    def test_Compose(self):
        ids = [i for i in range(5)]
        tsfms = [MockTransform(i) for i in ids]
        tsfm = tr.Compose(tsfms)
        image, mask = [], []
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img, ids)
        self.assertEqual(msk, ids)

    def test_RandomChoices(self):
        ids = [i for i in range(5)]
        tsfms = [MockTransform(i) for i in ids]
        image, mask = [], []
        K = 3
        tsfm = tr.RandomChoices(tsfms, k=K, keep_original_order=True)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(len(img), K)
        self.assertEqual(img, msk)
        self.assertEqual(img, sorted(image))
        # No need to keep original order
        image, mask = [], []
        tsfm = tr.RandomChoices(tsfms, k=K, keep_original_order=False)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(len(img), K)
        self.assertEqual(img, msk)
        self.assertTrue(set(img).issubset(set(ids)))

    def test_OneOf(self):
        ids = [i for i in range(5)]
        tsfms = [MockTransform(i) for i in ids]
        image, mask = [], []
        tsfm = tr.OneOf(tsfms)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img, mask)
        self.assertEqual(len(img), 1)
        self.assertTrue(set(img).issubset(set(ids)))

        # sitk.WriteImage(img, 'test/data/out.nrrd')
        # sitk.WriteImage(msk, 'test/data/out_msk.nrrd')
        # image = sitk.ReadImage('/home/fam918/Downloads/TCs/Batch2NRRD/199/anon_DICOM_anon/PA000001/ST000001/SE000004/Image.nrrd')

    def test_RandomOrder(self):
        ids = [i for i in range(5)]
        tsfms = [MockTransform(i) for i in ids]
        image, mask = [], []
        tsfm = tr.RandomOrder(tsfms)
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img, msk)
        self.assertEqual(set(img), set(ids))

    def test_Lambda(self):
        def function(x=[]):
            x.append(len(x))
            return x
        tsfm = tr.Lambda(image_transformer=function,
                         mask_transformer=function, p=1)
        image, mask = [], []
        img, msk = tsfm(image, mask=mask)
        self.assertEqual(img, [0])
        self.assertEqual(msk, [0])

class MockTransform(object):
    def __init__(self, identifier):
        self.identifier = identifier

    def __call__(self, image: list, mask: list = []):
        assert isinstance(image, list)
        if mask is not None:
            assert isinstance(image, list)
        assert isinstance(mask, list)
        image.append(self.identifier)
        mask.append(self.identifier)
        return image, mask
