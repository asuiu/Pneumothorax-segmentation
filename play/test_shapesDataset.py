#!/usr/bin/env python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 7/12/2019
import os
from os.path import join, dirname
from unittest import TestCase, main

import mrcnn.model as modellib
import numpy as np
import pydicom

from train_shapes_from_tutorial import MaskRCNNAuxFeatures, ShapesDataset, ShapesConfig

__author__ = 'ASU'


class TestShapesConfig(TestCase):
    def test_load_image(self):
        dataset_train = ShapesDataset()
        config = ShapesConfig()
        # 500
        dataset_train.load_shapes(5, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
        dataset_train.prepare()
        ROOT_DIR = os.path.abspath("pretrainings/")
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        # Create model in training mode
        model = MaskRCNNAuxFeatures(mode="training", config=config,
                                  model_dir=MODEL_DIR)

        train_generator = modellib.data_generator(dataset_train, model.config, shuffle=True,
                                                  augmentation=None,
                                                  batch_size=model.config.BATCH_SIZE,
                                                  no_augmentation_sources=None)
        first_image = next(train_generator)
        self.assertTrue(first_image)

    def test_dicom_metadata(self):
        def get_metadata_from_dicom(ds: pydicom.dataset.FileDataset):
            age = int(ds[(0x0010, 0x1010)].value)
            view_position = {'PA': 0, 'AP': 255}[ds[(0x0018, 0x5101)].value]
            sex = {'M': 0, 'F': 255}[ds[((0x0010, 0x0040))].value]
            return (age, view_position, sex)

        fname = "1.2.276.0.7230010.3.1.4.8323329.1000.1517875165.878027.dcm"
        ds = pydicom.read_file(join(dirname(__file__), '..', 'sample-data/', fname))
        self.assertEqual(int(ds[(0x0010, 0x1010)].value), 38)
        self.assertEqual(ds[(0x0018, 0x5101)].value, 'PA')
        self.assertEqual(ds[((0x0010, 0x0040))].value, 'M')
        image = ds.pixel_array

        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        md = get_metadata_from_dicom(ds)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i, j, 1] = md[j % 3]
        self.assertListEqual(list(image[:, :, 1][0, :4]), [38, 0, 0,38])


if __name__ == '__main__':
    main()
