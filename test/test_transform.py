import unittest
import numpy as np

import SimpleITK as sitk

import transform as tr

class TestTransform(unittest.TestCase):
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
        print(expected_ime_size, img.GetSize())
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

    @staticmethod
    def load_cube(size='small'):
        """
        This image is a 7x7x7 cube of intensity value of 1
        inside a 11x11x11 canvas. The background intensity
        is 0.
        size: small, medium
        """
        size_map = {'medium': 2, 'small': 1}
        image_path = f'test/data/image{size_map[size]}.nrrd'
        mask_path = f'test/data/mask{size_map[size]}.nrrd'
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        return image, mask

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
        #sitk.WriteImage(msk, '/home/fam918/Downloads/trashit/msk.nrrd')

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

    def test_Shrink(self):
        image, mask = TestTransform.load_cube('small')
        shrinkage_factors = [2, 2, 1]
        tsfm = tr.Shrink(shrinkage=shrinkage_factors, p=1)
        img, msk = tsfm(image, mask=mask)
        image_size = np.array(image.GetSize())
        print(img.GetSize(), image_size / shrinkage_factors)
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

#sitk.WriteImage(msk, '/home/fam918/Downloads/trashit/msk.nrrd')